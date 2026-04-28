# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sibling KV-cache layer for the mixed-precision path.

This module owns the higher-precision (FP8 or BF16) sibling cache that holds N tokens
per sequence alongside the primary cache (NVFP4 today). It exposes:

- ``MixedPrecisionKVCache``: the sibling cache layer.
- ``MixedKVCacheBackend`` and ``MixedKVMetadataBuilder``: spec / metadata plumbing for
  the standard ``KVCacheManager``.
- ``MixedKVMetadata``: per-step metadata shared across layers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Literal

import torch
import torch.nn as nn

from vllm.config import CacheConfig, VllmConfig, get_current_vllm_config
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
)
from vllm.v1.attention.backends.flashinfer import FlashInferBackend
from vllm.v1.kv_cache_interface import FirstNSpec, KVCacheSpec, SlidingWindowSpec

# Sentinel suffix appended to the parent attention layer prefix when
# constructing the sibling cache. Used by FlashInferImpl to look the
# sibling up via the static_forward_context.
MIXED_KV_SUFFIX = ".mixed_kv"


def mixed_kv_layer_prefix(parent_prefix: str) -> str:
    return f"{parent_prefix}{MIXED_KV_SUFFIX}"


@dataclass
class MixedKVMetadata:
    """Per-step metadata for the sibling cache.

    block_table is read by the dual-kernel decode in FlashInferImpl;
    slot_mapping carries the first-N -1 mask through to do_kv_cache_update
    (the runner's per-layer slot_mapping is built before our mask is
    applied, so we route the masked version via attn_metadata instead).
    block_size is a constant the impl uses to compute skip_blocks for the
    primary cache's read."""

    block_table: torch.Tensor
    slot_mapping: torch.Tensor
    block_size: int


class MixedKVCacheBackend(FlashInferBackend):
    """Backend that piggybacks on the FlashInfer page layout. We do not
    plan/execute attention through this backend directly; FlashInferImpl
    handles the kernel call. This class exists so the KVCacheManager has
    a backend to query for shape / stride info. Inherits supported kernel
    block sizes, stride order, and supported head sizes from
    ``FlashInferBackend``; overrides ``get_kv_cache_shape`` (natural shape
    without NVFP4 head padding) and ``get_builder_cls``.
    """

    @staticmethod
    def get_name() -> str:
        return "MIXED_KV_SIBLING"

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        # Return the NATURAL shape (no head-padding). Inter-block padding
        # is handled by ``page_size_padded`` on the spec via
        # ``model_runner.py``'s ``torch.as_strided`` path, which adds extra
        # bytes BETWEEN blocks while keeping each block internally
        # contiguous. Padding the head dim into the shape (as we used to)
        # made the kernel's slice [..., :natural_h, :] non-contiguous and
        # corrupted strides for downstream dims.
        return (num_blocks, 2, block_size, num_kv_heads, head_size)

    @staticmethod
    def get_builder_cls() -> type[MixedKVMetadataBuilder]:  # type: ignore[override]
        return MixedKVMetadataBuilder


class MixedPrecisionKVCache(nn.Module, AttentionLayerBase):
    """Sibling KV-cache layer paired with a primary (NVFP4) Attention layer.

    Holds ``n_tokens`` per sequence at higher precision. Allocation,
    slot-mapping, and block-table management are performed by the
    standard vLLM KVCacheManager via the spec returned from
    ``get_kv_cache_spec``. Reads happen inside FlashInferImpl, not here;
    this module just owns the tensor handle.
    """

    def __init__(
        self,
        head_size: int,
        num_kv_heads: int,
        n_tokens: int,
        dtype: Literal["fp8", "bf16"],
        location: Literal["first", "last"],
        prefix: str,
        cache_config: CacheConfig,
    ) -> None:
        super().__init__()
        # Sibling chooses its own block_size (smallest kernel-supported
        # size); unify_kv_cache_spec_page_size grows it as needed so the
        # sibling's --block-size is decoupled from the user's
        # --block-size (which only constrains primary attention).
        supported = MixedKVCacheBackend.get_supported_kernel_block_sizes()
        self.block_size = min(s for s in supported if isinstance(s, int))
        self.head_size = head_size
        self.num_kv_heads = num_kv_heads
        self.n_tokens = n_tokens
        self.mixed_kv_dtype = dtype
        self.location = location
        self.prefix = prefix
        self.cache_config = cache_config
        # Storage dtype: uint8 for FP8 (the kernel writes raw E4M3 bytes via
        # `reshape_and_cache_flash` with kv_cache_dtype="fp8" so the tensor
        # type doesn't need to be FP8 itself); bfloat16 for BF16 (the kernel
        # uses kv_cache_dtype="auto" and writes through unchanged).
        self.dtype = torch.uint8 if dtype == "fp8" else torch.bfloat16
        # Will be filled by bind_kv_cache.
        self.kv_cache = torch.tensor([])

        compilation_config = get_current_vllm_config().compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        # For ``location='last'``, sliding-window auto-eviction matches our
        # semantics (only the most recent N tokens stay resident).
        # For ``location='first'``, FirstNSpec caps allocation at the first
        # ``ceil(N/block_size)`` blocks and disables eviction (anchor mode).
        # The sibling page is rounded up to match the primary NVFP4 page
        # by ``unify_kv_cache_spec_page_size`` (which has visibility into
        # all specs and can pick a single ``target`` that satisfies the
        # shared block pool's same-page requirement).
        spec_cls = FirstNSpec if self.location == "first" else SlidingWindowSpec
        return spec_cls(
            block_size=self.block_size,
            num_kv_heads=self.num_kv_heads,
            head_size=self.head_size,
            head_size_v=self.head_size,
            dtype=self.dtype,
            sliding_window=self.n_tokens,
            supported_kernel_block_sizes=tuple(
                MixedKVCacheBackend.get_supported_kernel_block_sizes()
            ),
        )

    def get_attn_backend(self) -> type[AttentionBackend]:
        return MixedKVCacheBackend


class MixedKVMetadataBuilder(AttentionMetadataBuilder):
    """Builds MixedKVMetadata once per step (shared across layers).

    The standard KVCacheManager produces the sibling's block_table and
    slot_mapping; for first-N we additionally mask slot_mapping to -1 at
    positions >= N (the kernel that computes slot_mapping reads past the
    capped block_table rows and produces garbage slots otherwise).
    """

    reorder_batch_threshold: int = 1
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert isinstance(self.kv_cache_spec, SlidingWindowSpec), (
            "MixedKVMetadataBuilder expects a SlidingWindowSpec or "
            f"FirstNSpec; got {type(self.kv_cache_spec).__name__}."
        )
        self.n_tokens = int(self.kv_cache_spec.sliding_window)
        self.block_size = self.kv_cache_spec.block_size
        # FirstNSpec is a SlidingWindowSpec subclass; only it needs the
        # past-N slot_mapping mask.
        self.is_first_n = isinstance(self.kv_cache_spec, FirstNSpec)
        # Persistent buffer for the masked slot_mapping. The runner's
        # ``common_attn_metadata.slot_mapping`` is itself a stable buffer,
        # but the masked version needs separate storage. A fresh
        # ``slot_mapping.clone()`` per ``build()`` would be captured at a
        # stale address under cuda-graph capture/replay and produce
        # garbled writes to the sibling cache (mismatched slots →
        # corrupted K/V for the first-N positions → token output
        # degenerates after a few decode steps). Allocate once here so the
        # cuda-graph captures a stable pointer on its very first build()
        # call, including the warm-up before capture starts.
        if self.is_first_n:
            max_num_tokens = self.vllm_config.scheduler_config.max_num_batched_tokens
            self._masked_slot_mapping = torch.empty(
                max_num_tokens, dtype=torch.int64, device=self.device
            )
        else:
            self._masked_slot_mapping = None

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> MixedKVMetadata:
        block_table = common_attn_metadata.block_table_tensor
        slot_mapping = common_attn_metadata.slot_mapping

        # First-N: FirstNManager allocates only ceil(N/block_size) blocks per request,
        # so positions >= N index past the end of the block_table row and the
        # slot-mapping kernel produces garbage slots there. Mask them to -1 so
        # reshape_and_cache_flash skips those writes. positions is sized
        # `num_actual_tokens` while slot_mapping may be padded for cuda-graph FULL mode;
        # the padding region is already -1 from the runner so we only touch the
        # actual-tokens prefix.
        if self.is_first_n:
            positions = common_attn_metadata.positions
            assert positions is not None, (
                "first_n mixed-precision KV requires CommonAttentionMetadata."
                "positions to be populated; the model runner sets this on "
                "all standard call sites."
            )
            n_actual = positions.shape[0]
            # Reuse the persistent buffer allocated in ``__init__`` so
            # cuda-graph capture sees a stable input pointer for the
            # sibling write kernel. See ``__init__`` for why this matters.
            assert self._masked_slot_mapping is not None
            assert self._masked_slot_mapping.dtype == slot_mapping.dtype, (
                f"masked slot_mapping dtype mismatch: buf="
                f"{self._masked_slot_mapping.dtype} vs "
                f"runner={slot_mapping.dtype}"
            )
            buf = self._masked_slot_mapping[: slot_mapping.shape[0]]
            buf.copy_(slot_mapping)
            buf[:n_actual] = torch.where(
                positions < self.n_tokens,
                slot_mapping[:n_actual],
                torch.full_like(positions, -1, dtype=slot_mapping.dtype),
            )
            slot_mapping = buf

        return MixedKVMetadata(
            block_table=block_table,
            slot_mapping=slot_mapping,
            block_size=self.block_size,
        )
