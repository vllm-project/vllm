# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Cross-stage cache relay on the ROCm caches.

The compressed main KV keeps the record the shared relay copies. aiter writes
the MXFP4 indexer K pages preshuffled for its MQA-logits kernel, so those rows
are gathered and scattered in that order instead.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import torch

from vllm.forward_context import get_forward_context
from vllm.models.deepseek_v41.common.ops.fused_compress_quant_cache import (
    copy_pipeline_cache_rows,
)
from vllm.triton_utils import tl, triton
from vllm.v1.attention.ops.rocm_paged_mxfp4_indexer import (
    MXFP4_BLOCK_SIZE,
    rocm_paged_mxfp4_cache_layout,
)

from ..nvidia.pipeline import PipelineBoundary, PipelineRelay

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.sequence import IntermediateTensors


class RocmPipelineRelay(PipelineRelay):
    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str,
        incoming: PipelineBoundary,
        outgoing: PipelineBoundary,
    ):
        for boundary in (incoming, outgoing):
            # The candidate pool is not relayed, and ROCm consumers look their
            # top-k memos up under the index source's name.
            if boundary.candidate_source is not None or (
                boundary.index_source != boundary.kv_source
            ):
                raise NotImplementedError(
                    "DeepSeek-V4.1 on ROCm relays caches only across cuts whose "
                    "index source is their KV source, at or before the last KV "
                    "source layer."
                )
        super().__init__(vllm_config, prefix, incoming, outgoing)
        self.index_heads = self.config.index_n_heads
        self.index_head_dim = self.config.index_head_dim
        if self.main_cache is not None:
            # The relayed source stands in for the index source whose ragged
            # top-k memos DeepseekV41ROCMAiterMLAAttention's consumers share.
            self.main_cache._topk_ragged_cache = {}
            self.main_cache._prefill_topk_ragged_cache = {}

    def configure_consumers(self, layers) -> None:
        """Have the stage's first consumer of the relayed index source refresh
        its memos each step: that consumer's attention runs eagerly under the
        breakable CUDA graph, unlike this relay."""
        if self.main_cache is None:
            return
        source = f"{self.prefix}.layers.{self.incoming.index_source}.attn"
        for layer in layers:
            if getattr(layer.attn, "_index_source_prefix", None) == source:
                layer.attn._refreshes_relayed_index_source = True
                return

    def transfer(
        self,
        positions: torch.Tensor,
        topk: torch.Tensor,
        candidates: torch.Tensor | None,
        intermediate_tensors: IntermediateTensors | None = None,
    ) -> dict[str, torch.Tensor]:
        importing = intermediate_tensors is not None
        boundary = self.incoming if importing else self.outgoing
        if intermediate_tensors is not None:
            tensors = intermediate_tensors.tensors
        else:
            tensors = self.empty_tensors(boundary, positions.numel(), positions.device)
        metadata = get_forward_context().attn_metadata
        if boundary.kv_source is None or not isinstance(metadata, dict) or not metadata:
            return tensors
        ratio = self.config.compress_ratios[boundary.kv_source]
        for key, prefix, _, value_bytes in self._cache_entries(boundary):
            cache = self.context[prefix].kv_cache
            slots = cast(Any, metadata[prefix]).slot_mapping
            if key.startswith("pp_index_k_"):
                copy_rocm_mxfp4_index_rows(
                    cache,
                    tensors[key],
                    positions,
                    slots,
                    ratio,
                    self.index_heads,
                    self.index_head_dim,
                    scatter=importing,
                )
            else:
                copy_pipeline_cache_rows(
                    cache,
                    tensors[key],
                    positions,
                    slots,
                    ratio,
                    value_bytes,
                    scatter=importing,
                )
        if "pp_topk" in tensors:
            if importing:
                topk[: positions.numel()].copy_(tensors["pp_topk"])
            else:
                tensors["pp_topk"].copy_(topk[: positions.numel()])
        return tensors


def copy_rocm_mxfp4_index_rows(
    cache: torch.Tensor,
    rows: torch.Tensor,
    positions: torch.Tensor,
    slot_mapping: torch.Tensor,
    compress_ratio: int,
    num_heads: int,
    head_dim: int,
    *,
    scatter: bool,
) -> None:
    """Gather (``scatter=False``) or scatter each token's MXFP4 key between
    aiter's preshuffled indexer K pages and plain ``[values | scales]`` rows."""
    token_bytes = head_dim // 2
    num_scales = head_dim // MXFP4_BLOCK_SIZE
    assert cache.dtype == rows.dtype == torch.uint8
    assert cache.ndim == 3 and cache.shape[-1] == token_bytes + num_scales
    assert rows.shape == (positions.numel(), token_bytes + num_scales)
    assert rows.is_contiguous() and positions.is_contiguous()
    assert slot_mapping.is_contiguous() and slot_mapping.numel() <= rows.shape[0]
    if rows.shape[0] == 0:
        return
    layout = rocm_paged_mxfp4_cache_layout(num_heads, head_dim, cache.shape[1])
    _copy_rocm_mxfp4_index_rows[(rows.shape[0],)](
        cache,
        rows,
        positions,
        slot_mapping,
        slot_mapping.numel(),
        cache.stride(0),
        PAGE_ENTRIES=cache.shape[1],
        TOKEN_BYTES=token_bytes,
        NUM_SCALES=num_scales,
        N_PER_TILE=layout.n_per_tile,
        D_PER_TILE=layout.d_per_tile,
        SCALE_LANES=layout.scale_lanes,
        COMPRESS_RATIO=compress_ratio,
        SCATTER=scatter,
        num_warps=1,
    )


@triton.jit(do_not_specialize=["num_slots"])
def _copy_rocm_mxfp4_index_rows(
    cache,
    rows,
    positions,
    slots,
    num_slots,
    page_stride: tl.int64,
    PAGE_ENTRIES: tl.constexpr,
    TOKEN_BYTES: tl.constexpr,
    NUM_SCALES: tl.constexpr,
    N_PER_TILE: tl.constexpr,
    D_PER_TILE: tl.constexpr,
    SCALE_LANES: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    SCATTER: tl.constexpr,
):
    # The offsets aiter's k_norm_rope_mxfp4_cache writes with its shuffle.
    token = tl.program_id(0)
    slot = tl.load(slots + token, mask=token < num_slots, other=-1)
    position = tl.load(positions + token)
    valid = (slot >= 0) & ((position + 1) % COMPRESS_RATIO == 0)
    slot = tl.maximum(slot, 0)
    page = cache + (slot // PAGE_ENTRIES).to(tl.int64) * page_stride
    entry = slot % PAGE_ENTRIES
    group = entry // N_PER_TILE
    lane = entry % N_PER_TILE
    byte = tl.arange(0, TOKEN_BYTES)
    value_off = (
        group * (N_PER_TILE * TOKEN_BYTES)
        + byte // D_PER_TILE * (N_PER_TILE * D_PER_TILE)
        + lane * D_PER_TILE
        + byte % D_PER_TILE
    )
    S_HI: tl.constexpr = NUM_SCALES // SCALE_LANES
    scale = tl.arange(0, NUM_SCALES)
    scale_off = (
        PAGE_ENTRIES * TOKEN_BYTES
        + group * (N_PER_TILE * NUM_SCALES)
        + scale % SCALE_LANES * (N_PER_TILE * S_HI)
        + lane * S_HI
        + scale // SCALE_LANES
    )
    row = rows + token.to(tl.int64) * (TOKEN_BYTES + NUM_SCALES)
    if SCATTER:
        tl.store(page + value_off, tl.load(row + byte), mask=valid)
        tl.store(page + scale_off, tl.load(row + TOKEN_BYTES + scale), mask=valid)
    else:
        tl.store(row + byte, tl.load(page + value_off, mask=valid, other=0))
        tl.store(
            row + TOKEN_BYTES + scale,
            tl.load(page + scale_off, mask=valid, other=0),
        )
