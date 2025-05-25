"""Minimal layer skip support for Qwen models."""
import torch
from typing import Dict, Optional, TYPE_CHECKING
from contextlib import contextmanager

if TYPE_CHECKING:
    from vllm.sequence import IntermediateTensors

class LayerSkipModelMixin:
    """Add early exit with LSQ heads to Qwen models."""
    
    def __init__(self):
        super().__init__()
        self.lsq_heads: Dict[int, 'ParallelLMHead'] = {}  # Now stores ParallelLMHead instances
        
    def load_lsq_head(self, path: str, layer: int):
        """Load pre-trained LSQ head and convert to TP-sharded ParallelLMHead."""
        import os
        from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
        from vllm.distributed import get_pp_group
        
        head_file = os.path.join(path, f"h{layer}.pt")
        
        # Validate file exists
        if not os.path.isfile(head_file):
            raise FileNotFoundError(f"LSQ head file missing: {head_file}")
        
        try:
            full_head_weight = torch.load(head_file, map_location="cpu")
        except Exception as e:
            raise ValueError(f"Failed to load LSQ head from {head_file}: {e}") from e
        
        # Validate shape matches model config
        expected_shape = (self.config.vocab_size, self.config.hidden_size)
        if full_head_weight.shape != expected_shape:
            raise ValueError(f"LSQ head shape {full_head_weight.shape} "
                           f"does not match expected {expected_shape} "
                           f"for vocab_size={self.config.vocab_size}, "
                           f"hidden_size={self.config.hidden_size}")
        
        # Only create LSQ head on last PP rank (like lm_head)
        if get_pp_group().is_last_rank:
            # Create TP-sharded ParallelLMHead
            lsq_head = ParallelLMHead(
                num_embeddings=self.config.vocab_size,
                embedding_dim=self.config.hidden_size,
                quant_config=self.quant_config,
                prefix=f"lsq_head_{layer}"
            )
            
            # Load the full weight and extract this TP rank's shard
            # Use the same sharding as the regular lm_head
            vocab_start_index = lsq_head.shard_indices.org_vocab_start_index  
            vocab_end_index = lsq_head.shard_indices.org_vocab_end_index
            lsq_head.weight.data = full_head_weight[vocab_start_index:vocab_end_index].clone()
            
            # Move LSQ head to same device as main model (critical for device consistency)
            try:
                target_device = next(self.parameters()).device
                lsq_head = lsq_head.to(target_device)
            except Exception as e:
                raise RuntimeError(f"Failed to move LSQ head to device: {e}") from e
            
            self.lsq_heads[layer] = lsq_head
        else:
            # Non-last ranks don't have LSQ heads (like PPMissingLayer)
            from vllm.model_executor.models.utils import PPMissingLayer
            self.lsq_heads[layer] = PPMissingLayer()
            
        print(f"Loaded TP-sharded LSQ head for layer {layer} from {head_file}")
    
    @contextmanager
    def draft_mode_ctx(self, stop_layer: int):
        """Context manager for safe draft mode toggling."""
        # Save current state
        old_draft_mode = getattr(self, 'draft_mode', False)
        old_draft_layer = getattr(self, 'draft_layer', None)
        
        # Set draft mode
        self.draft_mode = True
        self.draft_layer = stop_layer
        
        try:
            yield
        finally:
            # Restore original state
            self.draft_mode = old_draft_mode
            self.draft_layer = old_draft_layer
    
    def forward_with_early_exit(
            self,
            input_ids:      torch.Tensor,
            positions:      torch.Tensor,
            stop_layer:     int,
            intermediate_tensors: Optional['IntermediateTensors'] = None,
    ) -> torch.Tensor:
        """Forward pass with early exit at specified layer for Qwen models."""
        from vllm.distributed import get_pp_group
        
        # 1. Get initial hidden states (PP-aware)
        if get_pp_group().is_first_rank:
            hidden_states = self.get_input_embeddings(input_ids)
            residual = None
        else:
            # Later ranks get hidden states from previous rank
            assert intermediate_tensors is not None, "PP ranks need intermediate tensors"
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]
        
        # 2. Process layers in this PP rank's range
        start_layer = self.model.start_layer
        end_layer = self.model.end_layer
        
        # Only process layers if this rank contains layers <= stop_layer
        actual_stop = min(stop_layer, end_layer)
        
        # If this rank starts after early exit point, do nothing (bypass all computation)
        if actual_stop <= start_layer:
            # This rank has no work to do for early exit - return unchanged inputs
            pass
        else:
            # Process the layers this rank is responsible for
            for i in range(start_layer, actual_stop):
                layer_idx = i - start_layer  # Index into this rank's layers
                hidden_states, residual = self.model.layers[layer_idx](
                    positions,
                    hidden_states,
                    residual,
                )
        
        # 3. Handle PP return types correctly
        if not get_pp_group().is_last_rank:
            # Non-last ranks must return IntermediateTensors for PP coordination
            from vllm.sequence import IntermediateTensors
            return IntermediateTensors({
                "hidden_states": hidden_states,
                "residual": residual
            })
        
        # 4. Last rank applies norm and returns final tensor
        if hasattr(self.model, 'norm'):
            hidden_states, _ = self.model.norm(hidden_states, residual)
        
        return hidden_states