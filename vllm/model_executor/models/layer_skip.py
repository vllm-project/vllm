"""Minimal layer skip support for Qwen models."""
import torch
from typing import Dict, Optional

class LayerSkipModelMixin:
    """Add early exit with LSQ heads to Qwen models."""
    
    def __init__(self):
        super().__init__()
        self.lsq_heads: Dict[int, torch.Tensor] = {}
        self._lsq_heads_gpu: Dict[int, torch.Tensor] = {}  # Cache GPU copies
        
    def load_lsq_head(self, path: str, layer: int):
        """Load pre-trained LSQ head."""
        import os
        head_file = os.path.join(path, f"h{layer}.pt")
        self.lsq_heads[layer] = torch.load(head_file, map_location="cpu")
        print(f"Loaded LSQ head for layer {layer} from {head_file}")
    
    def forward_with_early_exit(
            self,
            input_ids:      torch.Tensor,
            positions:      torch.Tensor,
            stop_layer:     int,
    ) -> torch.Tensor:
        """Forward pass with early exit at specified layer for Qwen models."""
        
        # 1. Get embeddings
        hidden_states = self.get_input_embeddings(input_ids)
        residual = None
        
        # 2. Process layers up to early exit point 
        for i in range(stop_layer):
            hidden_states, residual = self.model.layers[i](
                positions,
                hidden_states,
                residual,
            )
        # 3. Apply final norm (only on last PP rank, like main Qwen2Model)
        from vllm.distributed import get_pp_group
        if get_pp_group().is_last_rank:
            hidden_states, _ = self.model.norm(hidden_states, residual)
        
        return hidden_states