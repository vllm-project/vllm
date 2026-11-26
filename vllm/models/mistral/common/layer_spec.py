# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass

from transformers import PreTrainedConfig


@dataclass(frozen=True)
class MistralLayerSpec:
    """Architecture choices of one decoder layer, resolved from the config."""

    layer_idx: int
    use_rope: bool
    # Layer whose latent KV cache this layer attends to (itself if it owns one).
    kv_source_layer: int
    # Sparse (DSA) attention: whether this layer runs its own indexer. Layers
    # without one reuse the top-k indices of the closest preceding indexer.
    use_sparse: bool
    has_indexer: bool
    indexer_use_rope: bool
    is_moe: bool

    @property
    def shares_kv(self) -> bool:
        return self.kv_source_layer != self.layer_idx


def get_layer_specs(config: PreTrainedConfig) -> list[MistralLayerSpec]:
    num_layers = config.num_hidden_layers
    nope = getattr(config, "nope_layer_flags", None) or [False] * num_layers
    kv_sources = getattr(config, "kv_source_layers", None) or list(range(num_layers))
    index_sharing = getattr(config, "index_sharing", None)
    use_sparse = getattr(config, "index_topk", None) is not None
    indexer_nope = (
        getattr(config, "indexer_nope_layer_flags", None) or [False] * num_layers
    )
    n_routed_experts = getattr(config, "n_routed_experts", None)
    first_k_dense = getattr(config, "first_k_dense_replace", 0)
    moe_layer_freq = getattr(config, "moe_layer_freq", 1)

    return [
        MistralLayerSpec(
            layer_idx=i,
            use_rope=not nope[i],
            kv_source_layer=kv_sources[i],
            use_sparse=use_sparse,
            has_indexer=use_sparse
            and (index_sharing is None or index_sharing[i] == "full"),
            indexer_use_rope=not indexer_nope[i],
            is_moe=bool(n_routed_experts)
            and i >= first_k_dense
            and i % moe_layer_freq == 0,
        )
        for i in range(num_layers)
    ]
