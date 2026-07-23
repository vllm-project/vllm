# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inkling short-conv state managed as a sliding-window KV cache.

Each decoder layer owns one ``InklingConvState`` (an ``AttentionLayerBase``) that
emits a single ``SlidingWindowSpec`` for the layer's 4 sconv streams (K, V,
attn-output, mlp-output), packed head-major into one block:

    H = num_kv_heads (per-rank), N = block_size = sconv_kernel_size,
    D = head_dim(K) + head_dim(V) + hidden/H(attn) + hidden/H(mlp)

``D`` is TP-invariant; per rank we store ``H/TP`` heads of width ``D``. The conv
reads/writes this cache out-of-band via a custom backend; the (smaller) conv page
is padded up to the uniform attention page by ``unify_kv_cache_spec_page_size``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import torch
from torch import nn

from vllm.config import VllmConfig, get_current_vllm_config
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionMetadata,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
)
from vllm.v1.kv_cache_interface import AttentionSpec, KVCacheSpec, SlidingWindowSpec

from .ops.sconv import sconv_seq_metadata

# Stream order within the per-head packed D (== contiguous sub-ranges).
_K, _V, _ATTN, _MLP = 0, 1, 2, 3


@dataclass
class InklingSconvMetadata(AttentionMetadata):
    block_table: torch.Tensor  # [num_reqs, max_blocks] physical blocks per req
    slot_mapping: torch.Tensor  # [T] int64 flat slot of each token (-1 => skip)
    seq_idx: torch.Tensor  # [T] int32 token -> batch request
    query_start: torch.Tensor  # [T] int32 first x-row of each token's request


class InklingSconvMetadataBuilder(AttentionMetadataBuilder[InklingSconvMetadata]):
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ) -> None:
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        assert isinstance(kv_cache_spec, SlidingWindowSpec)
        # Persistent per-token buffers for CUDA graph capture.
        max_num_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        self.seq_idx_buffer = torch.empty(
            max_num_tokens, dtype=torch.int32, device=device
        )
        self.query_start_buffer = torch.empty(
            max_num_tokens, dtype=torch.int32, device=device
        )

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> InklingSconvMetadata:
        num_reqs = common_attn_metadata.num_reqs
        num_actual_tokens = int(common_attn_metadata.query_start_loc_cpu[-1])
        num_padded_tokens = common_attn_metadata.slot_mapping.shape[0]
        assert num_padded_tokens >= num_actual_tokens

        # Per-token seq_idx (owning request) and query_start (first x-row of
        # that request; the fused kernel uses it to tell same-forward taps,
        # read from x, from pre-forward taps, read from cache) in one launch.
        sconv_seq_metadata(
            common_attn_metadata.query_start_loc,
            num_reqs,
            num_actual_tokens,
            self.seq_idx_buffer,
            self.query_start_buffer,
            num_padded_tokens,
        )

        return InklingSconvMetadata(
            block_table=common_attn_metadata.block_table_tensor,
            slot_mapping=common_attn_metadata.slot_mapping[:num_padded_tokens],
            seq_idx=self.seq_idx_buffer[:num_padded_tokens],
            query_start=self.query_start_buffer[:num_padded_tokens],
        )


class InklingSconvBackend(AttentionBackend):
    """Custom dummy backend for the sconv sliding-window cache management."""

    @staticmethod
    def get_name() -> str:
        return "INKLING_SCONV_SWA"

    @classmethod
    def indexes_kv_by_block_stride(cls) -> bool:
        # num_blocks is the outermost dim (HND, see get_kv_cache_shape), so the
        # padded conv page is read through a strided view.
        return True

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        # HND, num-blocks-first, head-major: [num_blocks, H, N, D].
        return (num_blocks, num_kv_heads, block_size, head_size)

    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        # Identity: physical layout == logical [num_blocks, H, N, D].
        if include_num_layers_dimension:
            return (0, 1, 2, 3, 4)
        return (0, 1, 2, 3)

    @staticmethod
    def get_impl_cls():
        raise NotImplementedError(
            "InklingSconvBackend has no attention impl; the conv runs out-of-band."
        )

    @staticmethod
    def get_builder_cls() -> type[InklingSconvMetadataBuilder]:
        return InklingSconvMetadataBuilder


class InklingConvState(nn.Module, AttentionLayerBase):
    """Per-decoder-layer owner emitting one sliding-window conv-state spec."""

    def __init__(
        self,
        *,
        num_kv_heads: int,
        head_dim: int,
        hidden_size: int,
        kernel_size: int,
        prefix: str,
    ) -> None:
        super().__init__()
        self.prefix = prefix
        # Bound to the manager-allocated paged cache by bind_kv_cache; a
        # placeholder until then. Read out-of-band by InklingShortConv.
        self.kv_cache = torch.tensor([])
        tp_size = get_tensor_model_parallel_world_size()
        # Guardrails for the conv-state layout below; only these are exercised.
        # tp_size <= num_kv_heads keeps >=1 whole KV head per rank (no
        # replication/clamping), so the per-head width stays TP-invariant.
        assert tp_size <= num_kv_heads, (
            f"sconv SWA cache supports tp_size <= num_kv_heads ({num_kv_heads}), "
            f"got {tp_size}"
        )
        # Per-rank head count; D is TP-invariant (K/V heads and the hidden
        # chunk both scale 1/TP together). The attn-/mlp-output sconv streams
        # are hidden-sharded: each rank owns its H/tp chunk (the sublayer
        # outputs are reduce-scattered / all-gathered around the conv).
        self.num_kv_heads = num_kv_heads // tp_size
        hidden_per_head = hidden_size // num_kv_heads
        # Packed per-head width: K + V + attn-output chunk + mlp-output chunk,
        # padded to a power of two so every layer's conv page is the same size
        # and an exact multiple of the attention page (the page unifier then
        # scales attention block sizes instead of padding).
        raw_head_size = 2 * head_dim + 2 * hidden_per_head
        self.head_size = 1 << (raw_head_size - 1).bit_length()
        self.sliding_window = kernel_size
        self.block_size = kernel_size
        # Per-head D-sub-range (offset, width) for each stream. Streams share
        # the cache; each writes/reads its own width across all H heads.
        self.stream_ranges: tuple[tuple[int, int], ...] = (
            (0, head_dim),  # _K
            (head_dim, head_dim),  # _V
            (2 * head_dim, hidden_per_head),  # _ATTN
            (2 * head_dim + hidden_per_head, hidden_per_head),  # _MLP
        )
        vllm_config = get_current_vllm_config()
        self._dtype = vllm_config.model_config.dtype
        assert self._dtype == torch.bfloat16, (
            f"sconv SWA cache supports bfloat16 only, got {self._dtype}"
        )
        # Register in the forward context so the runner enumerates this owner as
        # an attention-like layer (get_kv_cache_spec / get_attn_backend).
        compilation_config = vllm_config.compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self

    def forward(self): ...

    def get_attn_backend(self) -> type[AttentionBackend]:
        return InklingSconvBackend

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        return SlidingWindowSpec(
            block_size=self.block_size,
            num_kv_heads=self.num_kv_heads,
            head_size=self.head_size,
            head_size_v=0,  # all 4 streams packed into head_size
            dtype=self._dtype,
            sliding_window=self.sliding_window,
        )
