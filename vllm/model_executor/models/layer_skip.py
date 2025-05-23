"""Minimal layer skip support."""
import torch
import torch.nn.functional as F
from typing import Optional, Dict, Tuple

class LayerSkipModelMixin:
    """Add early exit with LSQ heads to any model."""
    
    def __init__(self):
        super().__init__()
        self.lsq_heads: Dict[int, torch.Tensor] = {}
        self._lsq_heads_gpu: Dict[int, torch.Tensor] = {}  # Cache GPU copies
        
    def load_lsq_head(self, path: str, layer: int):
        """Load pre-trained LSQ head."""
        import os
        head_file = os.path.join(path, f"h{layer}.pt")
        self.lsq_heads[layer] = torch.load(head_file, map_location="cpu")
        
    def forward_with_early_exit(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor, 
        kv_caches: Optional[list],
        attn_metadata,
        stop_layer: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass stopping at layer N."""
        # Use model.embed_tokens directly (works for Llama/OPT/Qwen)
        hidden_states = self.model.embed_tokens(input_ids)
        if hasattr(self.config, 'hidden_size'):
            hidden_states = hidden_states * (self.config.hidden_size ** 0.5)
        
        # Process layers with generic calling convention
        for i in range(stop_layer):
            # FIXED: Use vLLM's generic layer calling pattern
            hidden_states = self.model.layers[i](
                hidden_states,
                position_ids=positions,
                kv_cache=kv_caches[i] if kv_caches else None,
                attn_metadata=attn_metadata,
            )
        
        # Apply final norm
        hidden_states = self.model.norm(hidden_states)
        
        # Use LSQ head
        if stop_layer not in self.lsq_heads:
            raise ValueError(f"No LSQ head for layer {stop_layer}")
        
        # FIXED: Cache GPU copy for performance
        if stop_layer not in self._lsq_heads_gpu:
            self._lsq_heads_gpu[stop_layer] = self.lsq_heads[stop_layer].to(
                device=hidden_states.device, 
                dtype=hidden_states.dtype
            )
        head = self._lsq_heads_gpu[stop_layer]
        
        # Proper reshape for matmul
        B, S, D = hidden_states.shape
        logits = (hidden_states.reshape(-1, D) @ head.T).reshape(B, S, -1)
        
        # Numerically stable entropy
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        entropy = -(probs * log_probs).sum(-1)  # Shape: (B, S)
        
        return logits, entropy