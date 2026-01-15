import torch
import torch.optim as optim
from torch import Tensor
import math
import random

class GaLoreAdamW(optim.Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
        rank: int = 128,
        update_proj_gap: int = 200,
        scale: float = 1.0,
        quantized: bool = False,
        proj_start_steps: int = 0,
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            rank=rank,
            update_proj_gap=update_proj_gap,
            scale=scale,
            quantized=quantized,
            proj_start_steps=proj_start_steps
        )
        super().__init__(params, defaults)

    def _get_orthogonal_matrix(self, tensor: Tensor, rank: int) -> Tensor:
        """
        Computes the Low-Rank Projector P using SVD.
        Projector P shape: (In_Features, Rank) if determining column space
        """
        # We want to project to the lower dimension.
        # If weights are (Out, In), grad is (Out, In).
        rows, cols = tensor.shape
        try:
            # We use float32 for SVD stability even if training in bf16
            U, S, Vh = torch.linalg.svd(tensor.float(), full_matrices=False)
        except torch._C._LinAlgError:
             # Fallback for stability issues
             torch.nn.init.orthogonal_(tensor)
             return tensor.new_zeros((rows, rank)) # Should not happen ideally
             
        # Low rank approximation
        if rows < cols:
            # Fat matrix (Out < In) -> Project columns (Right side)
            # Vh is (min(M,N), N). We want Vh.T[:, :rank] -> (N, r)
            projector = Vh.mT[:, :rank] 
        else:
            # Tall matrix (Out > In) -> Project rows (Left side)
            # U is (M, min(M,N)). We want U[:, :rank] -> (M, r)
            projector = U[:, :rank]
            
        return projector.to(tensor.dtype)

    def _quantize(self, tensor: Tensor) -> tuple[Tensor, Tensor]:
        """
        Naive Block-wise Int8 quantization.
        Returns: (quantized_tensor, scale)
        """
        scale = tensor.abs().max() / 127.0
        scale = torch.maximum(scale, torch.tensor(1e-12, device=tensor.device))
        quantized = (tensor / scale).round().clamp(-127, 127).to(torch.int8)
        return quantized, scale

    def _dequantize(self, quantized: Tensor, scale: Tensor) -> Tensor:
        return quantized.float() * scale

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']
            weight_decay = group['weight_decay']
            rank = group['rank']
            update_proj_gap = group['update_proj_gap']
            scale_factor = group['scale']
            use_quant = group['quantized']
            proj_start_steps = group['proj_start_steps']

            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("GaLoreAdamW does not support sparse gradients")

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    
                    # Store projector
                    state['projector'] = None
                    state['proj_type'] = None # 'left' or 'right'
                    
                    # Synchronous Schedule (Robust Fix)
                    # We update ALL layers at the same time to avoid "Creeping Loss"
                    state['step_offset'] = 0

                    # Optimizer States (Momentum)
                    if not use_quant:
                        state['exp_avg'] = torch.zeros(0) # Placeholder
                        state['exp_avg_sq'] = torch.zeros(0)
                    else:
                        state['exp_avg_int8'] = torch.zeros(0) 
                        state['exp_avg_scale'] = torch.tensor(0.0)
                        state['exp_avg_sq_int8'] = torch.zeros(0)
                        state['exp_avg_sq_scale'] = torch.tensor(0.0)

                state['step'] += 1
                
                # Perform Weight Decay
                if weight_decay > 0:
                    p.mul_(1 - lr * weight_decay)

                # --- GALORE LOGIC ---
                
                # We work with low-rank grad if > 1D AND we passed the warmup phase
                is_low_rank = (p.dim() >= 2) and (state['step'] >= proj_start_steps)
                
                if is_low_rank:
                    # 1. Update Projector logic (Synchronous)
                    should_update = ((state['step'] + state['step_offset']) % update_proj_gap == 0) or (state['projector'] is None)
                    
                    if should_update:
                        # Recompute SVD on *current gradient*
                        if p.shape[0] < p.shape[1]: 
                            state['proj_type'] = 'right'
                            state['projector'] = self._get_orthogonal_matrix(grad, rank)
                        else:
                            state['proj_type'] = 'left'
                            state['projector'] = self._get_orthogonal_matrix(grad, rank)
                            
                        # Reset States (Option 3: Synchronous Reset)
                        # We wipe the slate clean every 200 steps.
                        # Since it's synchronous, the model adapts globally.
                        if not use_quant:
                            state['exp_avg'].zero_()
                            state['exp_avg_sq'].zero_()
                        else:
                             state['exp_avg_int8'].zero_()
                             state['exp_avg_sq_int8'].zero_()
                             state['exp_avg_scale'].fill_(0.0)
                             state['exp_avg_sq_scale'].fill_(0.0)
                
                    projector = state['projector']
                
                    projector = state['projector']
                    
                    # 2. Project Gradient to Low Rank
                    if state['proj_type'] == 'right':
                        # G (M,N), P (N,R) -> G @ P -> (M, R)
                        grad_low = torch.matmul(grad, projector)
                    else:
                        # P (M,R), G (M,N) -> P.t() @ G -> (R, N)
                        grad_low = torch.matmul(projector.t(), grad)
                        
                else:
                    # Standard Adam path for 1D params
                    grad_low = grad
                
                # --- ADAM LOGIC (On grad_low) ---
                
                low_rank_shape = grad_low.shape
                
                if not use_quant:
                    # Standard float32/bf16 moments
                    if state['exp_avg'].numel() == 0 or state['exp_avg'].shape != low_rank_shape:
                             # Re-allocate if empty OR shape mismatch (Transition from Full -> Low Rank)
                             state['exp_avg'] = torch.zeros_like(grad_low)
                             state['exp_avg_sq'] = torch.zeros_like(grad_low)
                    
                    exp_avg = state['exp_avg']
                    exp_avg_sq = state['exp_avg_sq']
                else:
                    # Quantized moments
                    if state['exp_avg_int8'].numel() == 0:
                         state['exp_avg_int8'] = torch.zeros(low_rank_shape, dtype=torch.int8, device=p.device)
                         state['exp_avg_scale'] = torch.tensor(1.0, device=p.device)
                         state['exp_avg_sq_int8'] = torch.zeros(low_rank_shape, dtype=torch.int8, device=p.device)
                         state['exp_avg_sq_scale'] = torch.tensor(1.0, device=p.device)
                        
                    exp_avg = self._dequantize(state['exp_avg_int8'], state['exp_avg_scale'])
                    exp_avg_sq = self._dequantize(state['exp_avg_sq_int8'], state['exp_avg_sq_scale'])

                # Actual Adam Math
                exp_avg.mul_(beta1).add_(grad_low, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad_low, grad_low, value=1 - beta2)
                
                denom = exp_avg_sq.sqrt().add_(eps)
                norm_step_low = exp_avg / denom
                
                # Apply Scale
                norm_step_low.mul_(scale_factor)
                
                # Save State Back
                if not use_quant:
                     state['exp_avg'] = exp_avg
                     state['exp_avg_sq'] = exp_avg_sq
                else:
                     state['exp_avg_int8'], state['exp_avg_scale'] = self._quantize(exp_avg)
                     state['exp_avg_sq_int8'], state['exp_avg_sq_scale'] = self._quantize(exp_avg_sq)
                
                # --- PROJECT BACK & UPDATE ---
                
                if is_low_rank:
                     projector = state['projector']
                     
                     # Ensure dtype match for matmul (projector is bf16, norm_step_low might be float32)
                     if norm_step_low.dtype != projector.dtype:
                         norm_step_low = norm_step_low.to(projector.dtype)
                         
                     if state['proj_type'] == 'right':
                         # Low (M, R), P (N, R) -> Update (M, N) = Low @ P.t()
                         norm_step_full = torch.matmul(norm_step_low, projector.t())
                     else:
                         # Low (R, N), P (M, R) -> Update (M, N) = P @ Low
                         norm_step_full = torch.matmul(projector, norm_step_low)
                else:
                    norm_step_full = norm_step_low

                p.add_(norm_step_full, alpha=-lr)

        return loss
