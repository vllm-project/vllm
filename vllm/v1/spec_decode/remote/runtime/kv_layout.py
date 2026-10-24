# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Single-group own-KV layouts for the standalone draft planner.

Sliding-window attention keeps a full-context block table here. The attention
backend applies its window; the planner does not evict or recycle old positions.
"""

from collections.abc import Mapping

import torch

from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheSpec,
    KVCacheTensor,
    KVQuantMode,
    MLAAttentionSpec,
    SlidingWindowSpec,
    SparseCacheRole,
    compute_layout_strides,
)
from vllm.v1.kv_cache_layout import KVCacheLayout


def select_standalone_kv_spec(specs: Mapping[str, KVCacheSpec]) -> AttentionSpec:
    """Validate the homogeneous dense cache consumed by the own-KV planner."""
    if not specs:
        raise ValueError("standalone drafter has no attention KV layers")
    first = next(iter(specs.values()))
    # Subclasses include sparse/compressed caches and target-owned state with
    # different position or allocation contracts; accepting their base is unsafe.
    if type(first) not in (FullAttentionSpec, MLAAttentionSpec, SlidingWindowSpec):
        raise ValueError(
            "standalone runtime requires full-attention, dense MLA, or "
            "sliding-window KV; " + type(first).__name__ + " is unsupported"
        )
    if any(type(spec) is not type(first) or spec != first for spec in specs.values()):
        raise ValueError("standalone runtime requires one uniform KV group")
    assert isinstance(first, AttentionSpec)
    if (
        first.tokens_per_state != 1
        or first.num_head_slots is not None
        or first.state_content_bytes is not None
        or first.kv_quant_mode != KVQuantMode.NONE
        or first.dtype not in (torch.float16, torch.bfloat16, torch.float32)
        or first.page_size_padded is not None
    ):
        raise ValueError("standalone runtime requires dense unquantized KV pages")
    if isinstance(first, FullAttentionSpec) and (
        first.non_causal or first.attention_chunk_size is not None
    ):
        raise ValueError("standalone runtime requires causal token-addressed KV")
    if isinstance(first, MLAAttentionSpec) and (
        first.num_kv_heads != 1
        or first.head_size_v != 0
        or first.model_version is not None
        or first.is_index_group_leader
        or first.cache_role != SparseCacheRole.SPARSE
        or first.non_causal_multi_token_decode
        or first.sliding_window is not None
    ):
        raise ValueError("standalone runtime supports dense MLA without an indexer")
    return first


def make_standalone_kv_config(
    spec: AttentionSpec,
    layer_names: list[str],
    kv_cache_bytes: int,
    layout: KVCacheLayout,
) -> KVCacheConfig:
    """Describe disjoint layer ranges for the shared KV allocator."""
    if not layer_names or len(set(layer_names)) != len(layer_names):
        raise ValueError("standalone KV layer names must be nonempty and unique")
    select_standalone_kv_spec(dict.fromkeys(layer_names, spec))
    if not layout.is_layer_compact:
        raise ValueError("standalone runtime requires layer-compact KV layout")
    page_bytes = spec.page_size_bytes
    num_blocks = kv_cache_bytes // (len(layer_names) * page_bytes)
    if num_blocks < 2:
        raise ValueError("draft KV budget cannot hold null block plus one block")
    layer_stride, block_stride, *_ = compute_layout_strides(
        spec, num_blocks, len(layer_names), layout
    )
    if block_stride % spec.get_block_stride_alignment():
        raise ValueError("standalone KV layout does not satisfy block alignment")
    # All layers share one allocation, but must own distinct byte ranges.
    return KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[
            KVCacheTensor(
                size=page_bytes * num_blocks * len(layer_names),
                layers=list(layer_names),
                layer_stride=layer_stride,
                block_stride=block_stride,
            )
        ],
        kv_cache_groups=[
            KVCacheGroupSpec(list(layer_names), spec, is_eagle_group=True)
        ],
        kv_cache_layout=layout.name,
    )
