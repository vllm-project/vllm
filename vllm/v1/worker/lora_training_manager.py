# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
LoRATrainingManager: Handles LoRA adapter training with the detach-reattach pattern.

This module implements true colocation of inference and training by:
1. Using vLLM's fast kernels for base model forward pass (all requests)
2. Extracting hidden states from vLLM output (no gradients)
3. Applying LoRA adapters with gradients enabled (training requests only)
4. Performing backward pass through LoRA weights only

Key insight: Base model is frozen, so we only need gradients through LoRA adapters.
The detach-reattach pattern allows us to leverage vLLM's speed while still training.
"""

from typing import TYPE_CHECKING, Optional, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

logger = init_logger(__name__)


class LoRALayer(nn.Module):
    """
    Simple LoRA adapter layer: output = input + B @ A @ input
    
    This is a minimal implementation. In practice, you'd use PEFT library.
    """
    def __init__(self, input_dim: int, output_dim: int, rank: int = 8, alpha: float = 16.0, dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # LoRA matrices (much smaller than full weight matrix)
        # Use the same dtype as the model (bfloat16)
        self.lora_A = nn.Parameter(torch.randn(rank, input_dim, dtype=dtype) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(output_dim, rank, dtype=dtype))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply LoRA adapter.
        
        Args:
            x: Input tensor [..., input_dim]
            
        Returns:
            LoRA delta to add to base output
        """
        # LoRA: delta = B @ A @ x
        # x: [..., input_dim]
        # A: [rank, input_dim]
        # B: [output_dim, rank]
        
        # Step 1: x @ A.T -> [..., rank]
        h = F.linear(x, self.lora_A)
        
        # Step 2: h @ B.T -> [..., output_dim]
        delta = F.linear(h, self.lora_B)
        
        return delta * self.scaling


class LoRATrainingManager:
    """
    Manages LoRA adapter training using the detach-reattach pattern.
    
    This enables true colocation: inference and training requests share
    the same vLLM forward pass, then training requests apply LoRA adapters
    with gradients enabled.
    """
    
    def __init__(
        self,
        model_runner: "GPUModelRunner",
        hidden_size: int = 2048,
        vocab_size: int = 128256,
        lora_rank: int = 8,
        lora_alpha: float = 16.0,
        learning_rate: float = 1e-4,
    ):
        """
        Initialize LoRA training manager.
        
        Args:
            model_runner: The GPUModelRunner instance
            hidden_size: Model hidden dimension
            vocab_size: Vocabulary size
            lora_rank: Rank of LoRA matrices
            lora_alpha: LoRA scaling factor
            learning_rate: Learning rate for optimizer
        """
        self.model_runner = model_runner
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        
        # Create LoRA adapter for LM head
        # In practice, you'd have adapters for multiple layers
        # Use bfloat16 to match model dtype
        self.lm_head_lora = LoRALayer(
            input_dim=hidden_size,
            output_dim=vocab_size,
            rank=lora_rank,
            alpha=lora_alpha,
            dtype=torch.bfloat16,
        ).cuda()
        
        # Optimizer for LoRA parameters
        self.optimizer = AdamW(
            self.lm_head_lora.parameters(),
            lr=learning_rate,
        )
        
        # Storage for training results (for API retrieval)
        self._last_training_stats = []
        
        logger.info(
            f"LoRATrainingManager initialized: "
            f"rank={lora_rank}, alpha={lora_alpha}, lr={learning_rate}"
        )
    
    def apply_lora_and_train(
        self,
        hidden_states_dict: Dict[str, torch.Tensor],
        scheduler_output: "SchedulerOutput",
    ) -> Optional[Dict[str, float]]:
        """
        Apply LoRA adapters to hidden states and perform training.
        
        This is where the magic happens:
        1. Take detached hidden states from vLLM (no gradients)
        2. Apply LoRA adapters (with gradients enabled)
        3. Compute loss and backward through LoRA only
        
        Args:
            hidden_states_dict: Detached hidden states per training request
            scheduler_output: Scheduler output with request info
            
        Returns:
            Dictionary with training statistics
        """
        if not hidden_states_dict:
            return None
        
        self.optimizer.zero_grad()
        
        training_losses = {}
        total_loss_tensor = None  # Will accumulate loss tensors (not floats!)
        total_loss_value = 0.0     # For reporting only
        
        # Check LoRA parameter gradients
        print(f"[DEBUG LoRA] Checking LoRA parameters:")
        print(f"[DEBUG LoRA]   lora_A.requires_grad: {self.lm_head_lora.lora_A.requires_grad}")
        print(f"[DEBUG LoRA]   lora_B.requires_grad: {self.lm_head_lora.lora_B.requires_grad}")
        print(f"[DEBUG LoRA]   torch.is_grad_enabled(): {torch.is_grad_enabled()}")
        print(f"[DEBUG LoRA]   torch.is_inference_mode_enabled(): {torch.is_inference_mode_enabled()}")
        
        with torch.enable_grad():
            print(f"[DEBUG LoRA] Inside enable_grad context: torch.is_grad_enabled()={torch.is_grad_enabled()}")
            
            for req_id, hidden_states in hidden_states_dict.items():
                # Get request state for labels
                req_state = self.model_runner.requests.get(req_id)
                if not req_state or not req_state.training_config:
                    continue
                
                print(f"[DEBUG LoRA] Processing request {req_id}")
                print(f"[DEBUG LoRA]   hidden_states.requires_grad: {hidden_states.requires_grad}")
                print(f"[DEBUG LoRA]   hidden_states.grad_fn: {hidden_states.grad_fn}")
                print(f"[DEBUG LoRA]   hidden_states.shape: {hidden_states.shape}")
                
                # CRITICAL: Re-create hidden states with gradients enabled in current context
                # The issue is that hidden_states are leaf tensors created outside enable_grad()
                # We need to create NEW tensors that are part of the current computation graph
                hidden_states = hidden_states.detach()  # Fully detach from any existing graph
                hidden_states.requires_grad = True      # Enable gradients
                # Now perform an operation that creates a grad_fn
                hidden_states = hidden_states * 1.0     # This should create AddBackward or MulBackward
                print(f"[DEBUG LoRA]   After re-creation: hidden_states.grad_fn: {hidden_states.grad_fn}")
                
                # hidden_states: [num_tokens, hidden_size], requires_grad=True
                # Apply LoRA adapter to get logits
                lora_delta = self.lm_head_lora(hidden_states)
                print(f"[DEBUG LoRA]   lora_delta.requires_grad: {lora_delta.requires_grad}")
                print(f"[DEBUG LoRA]   lora_delta.shape: {lora_delta.shape}")
                
                # Get base logits (without gradients)
                with torch.no_grad():
                    base_logits = self.model_runner.model.compute_logits(
                        hidden_states.detach(), None
                    )
                
                # Combine: final_logits = base_logits + lora_delta
                # Only lora_delta has gradients
                final_logits = base_logits + lora_delta
                print(f"[DEBUG LoRA]   final_logits.requires_grad: {final_logits.requires_grad}")
                
                # Compute loss (same as before)
                labels = req_state.training_config.labels.to(final_logits.device)
                num_tokens = hidden_states.size(0)
                
                if len(labels) >= num_tokens and num_tokens > 1:
                    # Standard causal LM loss
                    shift_logits = final_logits[:-1, :]
                    shift_labels = labels[1:num_tokens]
                    
                    loss = F.cross_entropy(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1),
                        ignore_index=-100,
                    )
                    
                    print(f"[DEBUG LoRA]   loss.requires_grad: {loss.requires_grad}")
                    print(f"[DEBUG LoRA]   loss.item(): {loss.item():.4f}")
                    
                    training_losses[req_id] = loss.item()
                    total_loss_value += loss.item()
                    
                    # Accumulate tensor (not float!)
                    if total_loss_tensor is None:
                        total_loss_tensor = loss
                    else:
                        total_loss_tensor = total_loss_tensor + loss
        
        # Backward pass (gradients flow only to LoRA weights)
        if total_loss_tensor is not None:
            print(f"[DEBUG LoRA] Calling backward on total_loss_tensor")
            print(f"[DEBUG LoRA]   total_loss_tensor.requires_grad: {total_loss_tensor.requires_grad}")
            total_loss_tensor.backward()
            print(f"[DEBUG LoRA] Backward successful! Calling optimizer.step()")
            self.optimizer.step()
            
            avg_loss = total_loss_value / len(training_losses)
            
            logger.debug(
                f"LoRA training step: {len(training_losses)} requests, "
                f"avg_loss={avg_loss:.4f}"
            )
            
            batch_stats = {
                "num_requests": len(training_losses),
                "avg_loss": avg_loss,
                "total_loss": total_loss_value,
                "individual_losses": training_losses,
            }
            
            # Store for API retrieval
            self._last_training_stats.append(batch_stats)
            
            return batch_stats
        
        return None
    
    def get_lora_state_dict(self) -> Dict[str, torch.Tensor]:
        """Get LoRA adapter weights for checkpointing."""
        return self.lm_head_lora.state_dict()
    
    def load_lora_state_dict(self, state_dict: Dict[str, torch.Tensor]):
        """Load LoRA adapter weights from checkpoint."""
        self.lm_head_lora.load_state_dict(state_dict)

