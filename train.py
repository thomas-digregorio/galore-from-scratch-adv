import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import LlamaConfig, LlamaForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
from datasets import load_dataset
import wandb
import argparse
from tqdm import tqdm
import os
import math

from galore.optimizer import GaLoreAdamW

# --- Configuration ---
MODEL_CONFIGS = {
    '60M':  dict(hidden_size=512, intermediate_size=1024, num_hidden_layers=6, num_attention_heads=8),
    '125M': dict(hidden_size=768, intermediate_size=3072, num_hidden_layers=12, num_attention_heads=12),
    '350M': dict(hidden_size=1024, intermediate_size=4096, num_hidden_layers=24, num_attention_heads=16),
    '1B':   dict(hidden_size=2048, intermediate_size=5632, num_hidden_layers=22, num_attention_heads=32),
}

def parse_args():
    parser = argparse.ArgumentParser(description="Train Llama with GaLore from Scratch")
    parser.add_argument("--size", type=str, default="125M", choices=MODEL_CONFIGS.keys(), help="Model size")
    parser.add_argument("--optimizer", type=str, default="galore", choices=["adamw", "galore"], help="Optimizer choice")
    parser.add_argument("--batch_size", type=int, default=8, help="Training batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--steps", type=int, default=1000, help="Total training steps")
    parser.add_argument("--quantized", action="store_true", help="Use Int8 optimizer states for GaLore")
    parser.add_argument("--update_proj_gap", type=int, default=500, help="Steps between SVD updates")
    parser.add_argument("--proj_start_steps", type=int, default=500, help="Steps before starting GaLore (Full Rank Warmup)")
    return parser.parse_args()

def get_wikitext_loader(batch_size, tokenizer, block_size=1024):
    print("Loading WikiText-103...")
    dataset = load_dataset("wikitext", "wikitext-103-v1", split="train[:10%]") # Streamlined for demo
    
    # Simple streaming generator
    def data_generator():
        text_buffer = ""
        for item in dataset:
            text_buffer += item['text'] + tokenizer.eos_token
            while len(text_buffer) >= block_size * 4: # Approximation
                yield text_buffer[:block_size*4] # Gross over-estimation to ensure tokens
                text_buffer = text_buffer[block_size*4:]
    
    # Tokenize and block
    data = []
    print("Tokenizing data subset...")
    # Just take 5000 samples for the demo
    iter_data = iter(dataset)
    tokens_list = []
    
    # Quick & Dirty packing for demo
    # In production, use standard collator
    count = 0 
    MAX_SAMPLES = 20000 
    
    for _ in tqdm(range(MAX_SAMPLES)):
        try:
            row = next(iter_data)
            ids = tokenizer.encode(row['text'] + tokenizer.eos_token)
            tokens_list.extend(ids)
        except StopIteration:
            break
            
    # Chunk into block_size
    # Truncate
    total_len = (len(tokens_list) // block_size) * block_size
    tokens_list = tokens_list[:total_len]
    
    if total_len == 0:
        raise ValueError("Dataset is too small or tokenization failed.")
        
    input_ids = torch.tensor(tokens_list, dtype=torch.long).view(-1, block_size)
    print(f"Created dataset with {input_ids.shape[0]} batches of size {block_size}")
    
    loader = DataLoader(input_ids, batch_size=batch_size, shuffle=True, drop_last=True)
    return loader

def generate_text(model, tokenizer, prompt="The king said", max_new_tokens=30):
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True, top_k=50, top_p=0.95)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def train():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Training must run on GPU.")
    device = torch.device("cuda")
    print(f"Using device: {device}")

    # 1. Setup WandB
    wandb.init(project="galore-from-scratch", config=vars(args))
    
    # 2. Model Setup
    print(f"Initializing Llama-{args.size} (Random Weights)...")
    config_args = MODEL_CONFIGS[args.size]
    # Llama 2 style config
    config = LlamaConfig(
        vocab_size=32000,
        **config_args,
        max_position_embeddings=1024,
    )
    model = LlamaForCausalLM(config).to(device).to(torch.bfloat16) # Use BF16 for training
    
    # Tokenizer (Use standard llama tokenizer structure, but we rely on a pretrained one for vocab)
    # Using 'gpt2' as proxy or 'TinyPixel/Llama-2-7B-bf16-sharded' tokenizer if available. 
    # For simplicity, let's use GPT2 tokenizer mapped to size 32000 or retrain?
    # Simplest: Use GPT2 tokenizer and resize embeddings to match
    print("Loading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    # Resize model if tokenizer vocab != 32000
    if len(tokenizer) != 32000:
        model.resize_token_embeddings(len(tokenizer))
    
    print(f"Model Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    # 3. Optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(params, lr=args.lr)
    else:
        print(f"Using GaLoreAdamW (Rank={128}, Gap={args.update_proj_gap}, Quantized={args.quantized})")
        optimizer = GaLoreAdamW(
            params, 
            lr=args.lr, 
            rank=128, 
            update_proj_gap=args.update_proj_gap, 
            quantized=args.quantized,
            proj_start_steps=args.proj_start_steps
        )

    # 4. Data
    train_loader = get_wikitext_loader(args.batch_size, tokenizer)
    iter_loader = iter(train_loader)

    # 4.5 Scheduler
    # Warmup for 10% of steps
    num_warmup_steps = int(0.1 * args.steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=args.steps
    )

    # 5. Training Loop
    model.train()
    print(f"Starting Training (Warmup Steps: {num_warmup_steps})...")
    
    progress_bar = tqdm(range(args.steps))
    
    for step in progress_bar:
        try:
            batch = next(iter_loader)
        except StopIteration:
            iter_loader = iter(train_loader)
            batch = next(iter_loader)
            
        inputs = batch.to(device)
        
        # Forward
        outputs = model(inputs, labels=inputs)
        loss = outputs.loss
        
        # Backward
        loss.backward()
        
        # Grad Norm Monitor
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # Global clipping
        
        # Step
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        
        # Metrics
        current_lr = scheduler.get_last_lr()[0]
        mem_usage = torch.cuda.max_memory_allocated() / 1e9 # GB
        perplexity = math.exp(loss.item()) if loss.item() < 20 else -1.0
        
        wandb.log({
            "train/loss": loss.item(),
            "train/perplexity": perplexity,
            "train/grad_norm": grad_norm.item(),
            "train/lr": current_lr,
            "system/peak_memory": mem_usage,
        })
        
        progress_bar.set_description(f"Loss: {loss.item():.4f} | Mem: {mem_usage:.2f}GB")
        
        # Generation Check (every 100 steps)
        if step % 100 == 0 and step > 0:
            sample_text = generate_text(model, tokenizer)
            tqdm.write(f"\n[Step {step}] Generated: {sample_text}")
            
            # Log table to wandb
            gen_table = wandb.Table(columns=["Step", "Prompt", "Generated"])
            gen_table.add_data(step, "The king said", sample_text)
            wandb.log({"generated_text": gen_table})
            
            model.train()

    print("Training Complete.")
    wandb.finish()

if __name__ == "__main__":
    train()
