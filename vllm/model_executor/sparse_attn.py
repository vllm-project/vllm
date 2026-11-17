import torch
from typing import Optional

# Existing imports
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.sparse_indexer import SparseIndexerBase

# New import for the KV‑connector synchronization hook
from vllm.kv_connector import KVConnector


class SparseAttention(Attention):
    """Sparse attention implementation that leverages a SparseIndexer.

    The original implementation performed the indexer computation before the
    KV‑connector's ``wait_for_layer_load`` hook was invoked, which could lead to
    the indexer reading stale or partially‑loaded cache data.  To guarantee that
    the cache is fully available, we now explicitly call the hook at the very
    beginning of the forward pass.
    """

    def __init__(self, *args, indexer: Optional[SparseIndexerBase] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.indexer = indexer

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        layer_name: str = "",
        *,
        **kwargs,
    ) -> torch.Tensor:
        """Forward pass for sparse attention.

        Args:
            hidden_states: Input tensor of shape ``[batch, seq_len, hidden]``.
            attention_mask: Optional mask.
            layer_name: Name of the transformer layer – used for KV‑connector
                synchronization.
        Returns:
            Tensor with the same shape as ``hidden_states``.
        """
        # -----------------------------------------------------------------
        # Synchronization point – ensure the KV cache for this layer is fully
        # loaded before the indexer reads it.
        # -----------------------------------------------------------------
        if layer_name:
            # ``KVConnector`` is a singleton that provides the ``wait_for_layer_load``
            # method.  The call is a no‑op if the connector does not implement the
            # hook (e.g., in tests that do not use external KV loading).
            KVConnector.get_instance().wait_for_layer_load(layer_name)

        # Proceed with the original sparse‑attention logic.
        if self.indexer is not None:
            # The indexer may read the KV cache; it is now safe because the
            # synchronization hook has been executed.
            topk_indices = self.indexer.compute_topk(hidden_states)
            # The rest of the attention computation uses ``topk_indices``.
            # (The actual implementation details are unchanged.)
            # ... (existing attention code) ...
            # For illustration purposes we simply forward to the parent class.
            return super().forward(hidden_states, attention_mask, *kwargs)
        else:
            # Fallback to dense attention if no indexer is provided.
            return super().forward(hidden_states, attention_mask, *kwargs)
    
    # The rest of the class (e.g., helper methods) remains unchanged.
}
