# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SDSA selection: each query row's exact Top-K keys, by one of two selectors.

Dense: score every key of the row's request, then Top-K.
Fused: sample a threshold, select candidates in one sweep, exact Top-K among
them. SAMPLES keys of every PAGES 64-key pages are scored, each at its own
random offset into them, the offsets drawn once per step. A row's RANK-th
largest sampled score is its threshold; every key is equally likely to be
sampled, wherever it sits in the row. One sweep keeps the keys above it
(candidates) and up to K keys equal to it (ties), counting both exactly; a
row left short of K, or with more than CAPACITY candidates, is swept again
inside the same launch at a radix-selected threshold at most its K-th
largest score. No host synchronization.

K is the config's index_topk, 512 or 2048.
"""

from __future__ import annotations

import dataclasses
import math
import os
from typing import Any, Literal

import numpy as np
import torch

from vllm import _custom_ops as ops
from vllm import envs
from vllm.config.utils import config
from vllm.distributed import get_tp_group
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm.model_executor.warmup.jit_warmup import VllmJitKernel
from vllm.triton_utils import tl, triton
from vllm.utils.math_utils import cdiv, round_up
from vllm.utils.torch_utils import async_tensor_h2d
from vllm.v1.worker.workspace import current_workspace_manager

logger = init_logger(__name__)

PAGE = 64  # keys per cache page (FlashInfer's SDSA page)
# Threshold samples: SAMPLES keys at independent random offsets into each run
# of PAGES pages (1.5 per page).
SAMPLES, PAGES = 3, 2
# The indexer's key formats: a key's value type, value bytes and scale bytes.
# An indexer page holds its PAGE keys, then their scales. NVFP4 packs a key's
# 128 E2M1 values two to a byte, with an E4M3 scale per 16; BF16 has no scales.
INDEXER_FORMATS = {
    "nvfp4": (torch.uint8, 64, 8),
    "bf16": (torch.bfloat16, 256, 0),
}
# Rows per stock top_k_per_row_prefill call. The stock kernel sorts its first
# 12288 rows with insertion sort and the rest with radix sort (sampler.cu,
# kSortingAlgorithmThreshold), so every panel row takes the insertion path.
PANEL_ROWS = 12288
# The most keys a step's context can have for the union core: group_union_masks
# keeps a bit per key in shared memory, and 2**19 keys fit at every group and K.
UNION_KEY_LIMIT = 2**19


DecodeCore = Literal["token", "fa4"]
PrefillCore = DecodeCore | Literal["union"]


@config(frozen=True, kw_only=True)
class SdsaConfig:
    """Each phase's core attention, and which of its requests take the fused
    selector. The cores: token: trtllm-gen per token; fa4: FlashAttention 4 per
    token; union: FA4 per group of tokens over their union of keys.
    get_sdsa_config builds the config from the model's config and resolves
    "auto" values; the config is frozen."""

    top_k: Literal[512, 2048] = 2048
    """K, the number of keys the indexer selects for each query."""
    indexer_dtype: Literal["auto", "nvfp4", "bf16"] = "auto"
    """The dtype of the k cache and of the indexer dot product."""
    decode_core: DecodeCore | Literal["auto"] = "auto"
    """The decode core; "auto" picks it per step."""
    decode_fused: bool | tuple[int, int] | Literal["auto"] = "auto"
    """Which decode requests take the fused selector: all, none, or those with
    context in [lo, hi). "auto" fuses none."""
    prefill_core: PrefillCore | Literal["auto"] = "auto"
    """The prefill core; "auto" picks it per step."""
    prefill_fused: bool | tuple[int, int] | Literal["auto"] = "auto"
    """Which prefill requests take the fused selector: all, none, or those with
    context in [lo, hi). "auto" fuses all."""

    # FA4's tile holds 128 query rows; trtllm-gen's token-sparse tile 64 heads.
    # Packed NVFP4 K/V is converted to FP8 per selected token, so only the token
    # core reads it.
    def decode(
        self, *, heads: int, drafts: int, kv_cache_dtype: str = "auto"
    ) -> DecodeCore:
        if kv_cache_dtype == "nvfp4":
            if self.decode_core not in ("auto", "token"):
                raise ValueError("NVFP4 SDSA requires the token decode core")
            return "token"
        if self.decode_core != "auto":
            return self.decode_core
        return "fa4" if heads > 64 and drafts else "token"

    def prefill(
        self, *, heads: int, context: int, rows: int, kv_cache_dtype: str = "auto"
    ) -> PrefillCore:
        """The prefill core for a step.

        >>> config = SdsaConfig()
        >>> config.prefill(heads=128, context=4096, rows=4096)
        'fa4'
        >>> config.prefill(heads=64, context=4096, rows=4096)
        'token'
        >>> config.prefill(heads=32, context=4096, rows=4096)
        'union'
        >>> config.prefill(heads=32, context=4096, rows=32)
        'token'
        >>> config.prefill(heads=32, context=UNION_KEY_LIMIT, rows=UNION_KEY_LIMIT)
        'union'
        >>> config.prefill(heads=32, context=UNION_KEY_LIMIT + 1, rows=UNION_KEY_LIMIT)
        'token'
        """
        if kv_cache_dtype == "nvfp4":
            if self.prefill_core not in ("auto", "token"):
                raise ValueError("NVFP4 SDSA requires the token prefill core")
            return "token"
        if self.prefill_core != "auto":
            return self.prefill_core
        if heads > 64:
            return "fa4"
        if context > UNION_KEY_LIMIT:
            return "token"
        return "union" if heads <= 32 and context <= 64 * rows else "token"

    def fused(self, phase: Literal["decode", "prefill"]) -> bool | tuple[int, int]:
        rule = self.decode_fused if phase == "decode" else self.prefill_fused
        return phase == "prefill" if rule == "auto" else rule


def get_sdsa_config(text_config: Any, k_proj: LinearBase) -> SdsaConfig:
    """The one place that builds an SDSA layer's config, from the model's text
    config: K is `index_topk`, and the `sdsa_*` fields set the others, each
    "auto" when the checkpoint's config does not have it.
    `--hf-overrides` sets any of them, for example
    `--hf-overrides '{"text_config": {"sdsa_prefill_core": "token"}}'`.
    `sdsa_indexer_dtype` "auto" gives NVFP4 if the checkpoint quantizes the
    indexer's key projection `k_proj`, and BF16 if not. The config is frozen."""
    if text_config.index_topk not in RANK_AND_CAPACITY:
        raise ValueError(
            f"SDSA serves index_topk {list(RANK_AND_CAPACITY)}, "
            f"got {text_config.index_topk}"
        )
    indexer_dtype: Literal["auto", "nvfp4", "bf16"] = getattr(
        text_config, "sdsa_indexer_dtype", "auto"
    )
    source = "config"
    if indexer_dtype == "auto" and isinstance(
        k_proj.quant_method, UnquantizedLinearMethod
    ):
        if k_proj.params_dtype != torch.bfloat16:
            raise ValueError(
                f"SDSA's BF16 indexer needs a BF16 model, got {k_proj.params_dtype}"
            )
        indexer_dtype, source = "bf16", "model dtype"
    elif indexer_dtype == "auto":
        indexer_dtype, source = "nvfp4", "quantized k_proj"
    sdsa_config = SdsaConfig(
        top_k=text_config.index_topk,
        indexer_dtype=indexer_dtype,
        decode_core=getattr(text_config, "sdsa_decode_core", "auto"),
        decode_fused=getattr(text_config, "sdsa_decode_fused", "auto"),
        prefill_core=getattr(text_config, "sdsa_prefill_core", "auto"),
        prefill_fused=getattr(text_config, "sdsa_prefill_fused", "auto"),
    )
    logger.info_once(
        "SDSA: %s; k cache + indexer dot product dtype from %s",
        sdsa_config,
        source,
        scope="global",
    )
    return sdsa_config


# The sweep's threshold is a row's RANK-th largest sampled score, and a row has
# CAPACITY candidate slots after its K tie slots. A row comes up short of
# K when its top K - 1 keys hold RANK samples, and overflows when its top
# CAPACITY keys (but up to 127 past its last whole run of PAGES pages) hold
# fewer. Per row, over the random offsets, by the Poisson approximation (Chernoff
# bound):
#   K = 2048: short 1 in 1.4 billion (120 million), over 1 in 12 quintillion
#   (870 quadrillion);
#   K = 512: short 1 in 2 billion (180 million), over 1 in 21 trillion (2.1
#   trillion).
# RANK is the least that keeps short rows as rare at 512 as at 2048; CAPACITY
# keeps overflows 10,000 times rarer still. A row that misses is swept again in
# the same launch, so the result stays exact.
RANK_AND_CAPACITY = {2048: (96, 9216), 512: (39, 4608)}


def _fused(rule: bool | tuple[int, int], contexts):
    """Which requests of `contexts` keys take the fused selector under `rule`."""
    lo, hi = (0, math.inf) if rule is True else (0, 0) if rule is False else rule
    return (lo <= contexts) & (contexts < hi)


def union_group(heads: int) -> int:
    """The union core's tokens per group: the most whose heads fit FA4's tile."""
    return 1 << ((128 // heads).bit_length() - 1)


def union_columns(group: int, top_k: int, max_model_len: int) -> int:
    """The columns of a union row, and of a mask row at 32 per int32 word: a
    group's K keys each, at most max_model_len, in whole 256-column blocks."""
    return round_up(min(group * top_k, max_model_len), 256)


_Layout = list[tuple[tuple[int, ...], torch.dtype]]  # workspace buffers


def _top_k(
    scores: torch.Tensor,
    lengths: torch.Tensor,
    top: torch.Tensor,
    workspace: torch.Tensor,
    rows_per_length: int = 1,
) -> None:
    """Write each row's exact Top-K positions (K = top.shape[1]) among its
    valid scores; a shorter row gets -1 in its last slots. Row r of a length's
    `rows_per_length` rows has lengths[length] - rows_per_length + r + 1 (the
    MTP rows). Uses gvr_2's host entry, since top_k_varlen takes only K from
    512 to 2048 and this also takes K = RANK. One launch, with lengths read on
    the device, so it can be captured in a CUDA graph. gvr_2 takes its
    threshold from its own sample of the row, so it is given no hints."""
    from flashinfer.topk_varlen.kernels.gvr2_topk_host import run_varlen

    run_varlen(
        scores,
        None,
        lengths,
        top,
        next_n=rows_per_length,
        max_seq_len=scores.shape[1],
        workspace=workspace,
        top_k=top.shape[1],
    )


def indexer_pages(
    cache: torch.Tensor, indexer_dtype: str
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """The keys and scales of an indexer cache, as the SDSA kernels take them:
    NVFP4 [pages, PAGE, 64] uint8 and [pages, PAGE, 8] E4M3 views, BF16
    [pages, PAGE, 128] and no scales."""
    value_type, key_bytes, scale_bytes = INDEXER_FORMATS[indexer_dtype]
    pages = cache.view(torch.uint8).view(-1, PAGE * (key_bytes + scale_bytes))
    keys = pages[:, : PAGE * key_bytes].view(value_type).view(len(pages), PAGE, -1)
    if not scale_bytes:
        return keys, None
    scales = pages[:, PAGE * key_bytes :].view(torch.float8_e4m3fn)
    return keys, scales.view(-1, PAGE, scale_bytes)


def _rows(scales: torch.Tensor | None, rows) -> torch.Tensor | None:
    """Rows of the NVFP4 query scales; BF16 has none."""
    return None if scales is None else scales[rows]


def _sample_capacity(context: int) -> int:
    """Return the number of sampled keys per request of up to `context` keys,
    in whole 128-key scorer tiles."""
    return 128 * cdiv(context * SAMPLES, 128 * PAGE * PAGES)


def _sample_offsets(capacity: int, device: torch.device) -> torch.Tensor:
    """Each sample's random offset into its run of PAGES pages, drawn on the
    device from an OS entropy seed, not the seeded `random`, so no input can
    anticipate them."""
    seed = int.from_bytes(os.urandom(8), "little")
    generator = torch.Generator(device).manual_seed(seed)
    return torch.randint(
        PAGES * PAGE, (capacity,), generator=generator, dtype=torch.uint8, device=device
    )


def _arrivals() -> torch.Tensor:
    """Return the threshold select's per-tile arrival counts for a panel:
    zeroed once, left zeroed by every call, shared by all layers."""
    return current_workspace_manager().get_persistent(
        "sdsa_arrivals", (PANEL_ROWS // 128,), torch.int32, zero_init=True
    )


@triton.jit
def _token_slots_kernel(
    positions,
    query_rows,
    requests,
    block_table,
    table_stride,
    slots,
    counts,
    top_k: tl.constexpr,
    page: tl.constexpr,
    keep_positions: tl.constexpr,
):
    row = tl.program_id(0)
    query = tl.load(query_rows + row).to(tl.int64)
    if query >= 0:
        columns = tl.arange(0, top_k)
        keys = tl.load(positions + row.to(tl.int64) * top_k + columns)
        valid = keys >= 0
        if keep_positions:
            tl.store(slots + query * top_k + columns, tl.where(valid, keys, -1))
        else:
            request = tl.load(requests + row).to(tl.int64)
            pages = tl.load(
                block_table + request * table_stride + keys // page,
                mask=valid,
                other=0,
            )
            tl.store(
                slots + query * top_k + columns,
                tl.where(valid, pages * page + keys % page, -1),
            )
        tl.store(counts + query, tl.sum(valid.to(tl.int32), 0))


def _token_slots(
    positions: torch.Tensor,
    query_rows: torch.Tensor,
    requests: torch.Tensor,
    block_table: torch.Tensor,
    slots: torch.Tensor,
    counts: torch.Tensor,
    keep_positions: bool = False,
) -> None:
    """Write each tiled row's Top-K keys (key positions, -1 last) as cache
    slots, -1 last, to its query's row of `slots`, and their count to
    `counts`. Row r holds query query_rows[r] (-1: none) of request
    requests[r], whose pages are its row of `block_table`. With
    keep_positions, the rows get the key positions themselves."""
    rows, top_k = positions.shape
    if rows:
        _token_slots_kernel[(rows,)](
            positions,
            query_rows,
            requests,
            block_table,
            block_table.stride(0),
            slots,
            counts,
            top_k,
            PAGE,
            keep_positions,
        )


@triton.jit
def _write_index_qk_kernel(
    key,
    key_stride,
    slots,
    keys,
    keys_stride,
    query,
    query_stride,
    query_slots,
    packed_query,
    num_keys,
    DIM: tl.constexpr,
):
    item = tl.program_id(0)
    columns = tl.arange(0, DIM)
    if item < num_keys:
        slot = tl.load(slots + item).to(tl.int64)
        if slot >= 0:
            row = tl.load(key + item.to(tl.int64) * key_stride + columns)
            tl.store(keys + slot * keys_stride + columns, row)
    else:
        item -= num_keys
        slot = tl.load(query_slots + item).to(tl.int64)
        row = tl.load(query + item.to(tl.int64) * query_stride + columns)
        tl.store(packed_query + slot * DIM + columns, row)


def write_index_qk(
    key: torch.Tensor,
    slots: torch.Tensor,
    keys: torch.Tensor,
    query: torch.Tensor,
    query_slots: torch.Tensor,
    packed_query: torch.Tensor,
) -> None:
    """BF16 indexer: copy key i to cache row slots[i] of `keys` (none for -1)
    and query token j to row query_slots[j] of `packed_query`, in one launch."""
    items = len(key) + len(query)
    if items:
        _write_index_qk_kernel[(items,)](
            key,
            key.stride(0),
            slots,
            keys,
            keys.stride(0),
            query,
            query.stride(0),
            query_slots,
            packed_query,
            len(key),
            DIM=key.shape[1],
        )


def _dense_tiles(width: int) -> int:
    """Tiles of a dense prefill panel of `width`-key rows: within the indexer's
    logits budget, at least one."""
    budget = envs.VLLM_SPARSE_INDEXER_MAX_LOGITS_MB << 20
    return min(PANEL_ROWS // 128, max(1, budget // (4 * width * 128)))


def _arena_layout(
    num_requests: int,
    sample_capacity: int,
    panel_rows: int,
    config: SdsaConfig,
    dense: bool,
) -> _Layout:
    """Return the layout of a step's arena: every request's sampled keys and
    their scales (as 8-byte words), then a panel's scratch, every buffer
    `panel_rows` rows, and, if `dense`, dense prefill's scores."""
    samples = num_requests * sample_capacity
    rank, capacity = RANK_AND_CAPACITY[config.top_k]
    _, key_bytes, scale_bytes = INDEXER_FORMATS[config.indexer_dtype]
    budget = envs.VLLM_SPARSE_INDEXER_MAX_LOGITS_MB << 18
    return [
        ((samples, key_bytes // 8), torch.int64),
        ((samples, scale_bytes // 8), torch.int64),
        ((panel_rows, sample_capacity), torch.float32),  # sample scores
        ((panel_rows, rank), torch.int32),  # their Top-RANK positions
        ((panel_rows,), torch.float32),  # thresholds
        ((panel_rows, 2), torch.int32),  # counters: candidates, ties
        ((panel_rows, 2), torch.int32),  # Top-K slot ranges, read as [2, rows]
        ((panel_rows, config.top_k + capacity, 2), torch.int32),  # candidates
        ((panel_rows, config.top_k), torch.int32),  # Top-K positions
        ((budget if dense else 0,), torch.float32),  # dense scores
    ]


class SdsaTiles:
    """Query rows in 128-row tiles of one request each, for the fused selector:
    tile t is request tile_requests[t], its row r seeing keys
    < row_ends[128 t + r] (0: no query). `context_lens` are the requests'
    key counts; `block_table` holds their pages."""

    def __init__(
        self,
        tile_requests: torch.Tensor,
        row_ends: torch.Tensor,
        context_lens: torch.Tensor,
        block_table: torch.Tensor,
        *,
        max_context: int,
        sample_offsets: torch.Tensor,
    ) -> None:
        self.tile_requests, self.row_ends = tile_requests, row_ends
        # The block table bounds the keys a row can see.
        self.block_table = block_table
        self.max_context = min(max_context, PAGE * block_table.shape[1])
        self.key_tiles = cdiv(self.max_context, 128)
        # Row r samples the runs of PAGES pages wholly before its end. The dense
        # scorer scores each 8-row group of a tile (its MMA's N = 8 query
        # columns) against its request's sample pages, up to the group's
        # longest sample end.
        self.sample_capacity = _sample_capacity(max_context)
        self.sample_offsets = sample_offsets[: self.sample_capacity]
        self.sample_ends = context_lens // (PAGES * PAGE) * SAMPLES
        self.sample_row_ends = row_ends // (PAGES * PAGE) * SAMPLES
        self.sample_table = torch.arange(
            len(context_lens) * self.sample_capacity // PAGE,
            dtype=torch.int32,
            device=row_ends.device,
        ).view(len(context_lens), -1)
        self.group_requests = tile_requests.repeat_interleave(128 // 8)
        self.sample_group_ends = self.sample_row_ends.view(-1, 8).amax(1)

    def gather_sample(self, key_cache, key_scales, sample_keys, sample_scales):
        """Each request's sampled keys and their scales, as 64-key sample pages
        (`select`'s keys); NVFP4's scales share the keys' pages."""
        from flashinfer.sdsa import gather_key_sample

        num_requests, width = len(self.sample_ends), key_cache.shape[2]
        sample_keys = sample_keys.view(key_cache.dtype).view(num_requests, -1, width)
        if key_scales is not None:
            sample_scales = sample_scales.view(key_scales.dtype)
            sample_scales = sample_scales.view(num_requests, -1, key_scales.shape[2])
        else:
            sample_scales = None
        gather_key_sample(
            key_cache,
            self.block_table,
            self.sample_offsets,
            self.sample_ends,
            sample_keys=sample_keys,
            samples_per_run=SAMPLES,
            pages_per_run=PAGES,
            key_scales=key_scales,
            sample_scales=sample_scales,
        )
        sample_keys = sample_keys.view(-1, PAGE, width)
        if key_scales is not None:
            sample_scales = sample_scales.view(-1, PAGE, key_scales.shape[2])
        return sample_keys, sample_scales

    def select(
        self,
        begin: int,
        end: int,
        query: torch.Tensor,
        query_scales: torch.Tensor | None,
        key_cache: torch.Tensor,
        key_scales: torch.Tensor | None,
        samples: tuple,
        scratch: list[torch.Tensor],
        topk_workspace: torch.Tensor,
        config: SdsaConfig,
    ) -> torch.Tensor:
        """Select tiles begin..end - 1 (query: their rows): their rows' Top-K
        key positions, -1 last."""
        from flashinfer.sdsa import (
            indexer_scores,
            indexer_threshold_select,
            thresholds_from_sample,
            top_k_from_candidates,
        )

        rows = slice(128 * begin, 128 * end)
        (
            sample_scores,
            sample_top,
            thresholds,
            counters,
            ranges,
            candidates,
            positions,
        ) = (buffer[: 128 * (end - begin)] for buffer in scratch[:7])
        ranges = ranges.view(2, -1)
        sample_keys, sample_scales = samples
        # The threshold: the least of each row's Top-RANK sampled scores.
        indexer_scores(
            query,
            sample_keys,
            self.sample_table,
            self.sample_group_ends[128 // 8 * begin : 128 // 8 * end],
            requests=self.group_requests[128 // 8 * begin : 128 // 8 * end],
            scores=sample_scores,
            query_scales=query_scales,
            key_scales=sample_scales,
        )
        _top_k(sample_scores, self.sample_row_ends[rows], sample_top, topk_workspace)
        thresholds_from_sample(
            sample_scores, sample_top, thresholds=thresholds, counters=counters
        )
        # Candidates and ties of the rows; a row that misses is swept again
        # inside the same launch, at most at its K-th largest score.
        indexer_threshold_select(
            query,
            key_cache,
            self.row_ends[rows],
            self.tile_requests[begin:end],
            self.block_table,
            thresholds=thresholds,
            counters=counters,
            candidates=candidates,
            max_context=self.max_context,
            top_k=config.top_k,
            query_scales=query_scales,
            key_scales=key_scales,
            arrivals=_arrivals()[: end - begin],
            ranges=ranges,
        )
        # The keys of the exact Top-K among each row's candidates and ties.
        top_k_from_candidates(candidates, ranges, top_keys=positions)
        return positions


def reserve(
    max_requests: int,
    max_tokens: int,
    max_context: int,
    topk_workspace: torch.Tensor,
    config: SdsaConfig,
) -> None:
    """Request, in the profiling run, the arena of the largest step: at most
    `max_requests` requests, `max_tokens` query tokens and `max_context` keys
    per request, every buffer at its bound.

    Then run gvr_2's sample Top-RANK over empty rows for every panel size at
    every sample width a step can take: gvr_2 compiles a kernel per launch
    shape on first use, which would stall a step (and every TP rank, in its
    next collective). Its first call per K also builds a hint table, which
    allocates and syncs with the host and is refused under CUDA graph
    capture; doing it here puts that memory in the profiled budget. Also
    allocate the arrival counts."""
    manager = current_workspace_manager()
    panel_rows = min(PANEL_ROWS, 128 * (max_tokens // 128 + max_requests))
    layout = _arena_layout(
        max_requests,
        _sample_capacity(max_context),
        panel_rows,
        config,
        config.fused("prefill") is not True,
    )
    sample_scores, sample_top = manager.get_simultaneous(*layout)[2:4]
    _arrivals()
    for width in range(128, sample_scores.shape[1] + 1, 128):
        for rows in range(128, panel_rows + 1, 128):
            scores = sample_scores.flatten()[: rows * width].view(rows, width)
            lengths = sample_top.new_zeros(rows)
            _top_k(scores, lengths, sample_top[:rows], topk_workspace)


class GroupUnionMasksKernel(VllmJitKernel["GroupUnionMasksKernel.CompileKey"]):
    """Builds every FlashInfer group_union_masks kernel a prefill step can run.

    FlashInfer compiles a kernel per group size and per key bitmap size, on
    first use: a power of two of 16K-key rounds up to 2**20 keys, and a
    multiple of 8 rounds (128K keys) past it. The JIT warm-up builds them at
    startup, so that no step stalls to compile one. With the warm-up off
    (`--enforce-eager` turns it off), each one compiles in the first step that
    needs it.
    """

    @dataclasses.dataclass(frozen=True)
    class CompileKey:
        group: int
        top_k: int
        num_keys: int
        # The launch's block table and union widths; not part of the identity.
        table_pages: int = dataclasses.field(compare=False)
        columns: int = dataclasses.field(compare=False)

    def get_warmup_keys(
        self,
        *,
        group: int,
        top_k: int,
        max_model_len: int,
        table_pages: int,
        columns: int,
    ) -> list[CompileKey]:
        """One key per bitmap size that a step can take, for groups of `group`
        tokens. A launch takes a block table of `table_pages` pages, and unions
        and mask rows of `columns` columns."""
        # A step has at most max_model_len keys, and above UNION_KEY_LIMIT it
        # takes the token core.
        limit = min(max_model_len, UNION_KEY_LIMIT)
        # The bitmap sizes: powers of two from 16K keys; the last holds `limit`.
        sizes = [2**n for n in range(14, 19) if 2**n < limit] + [limit]
        return [self.CompileKey(group, top_k, n, table_pages, columns) for n in sizes]

    def compile(self, compile_key: CompileKey) -> None:
        from flashinfer.token_sparse import group_union_masks

        group = compile_key.group
        columns = compile_key.columns
        # One empty group: the launch builds the kernel and writes nothing.
        positions = torch.full(
            (group, compile_key.top_k), -1, dtype=torch.int32, device="cuda"
        )
        rows = positions.new_full((group,), -1)
        group_union_masks(
            positions,
            rows,
            rows[:1],
            positions.new_zeros(1),
            positions.new_zeros(1, compile_key.table_pages),
            union=positions.new_zeros(1, columns),
            union_sizes=positions.new_zeros(1),
            mask=positions.new_zeros(group, columns // 32),
            num_keys=compile_key.num_keys,
        )


GROUP_UNION_MASKS_KERNEL = GroupUnionMasksKernel()


class SdsaPrefillPlan:
    """Tile layout for one scheduler step; `run` is called once per layer.
    `block_table` holds the planned requests' pages of PAGE tokens in order,
    which locate both the indexer's keys and the core's K/V. `core` is the
    step's core attention for layers of `heads` heads. The step's query rows
    are its `first_row` decode rows, then the tiles' rows; `query_slots` maps
    the step's query tokens to them."""

    def __init__(
        self,
        *,
        first_row: int,
        query_lens: tuple[int, ...],
        context_lens: tuple[int, ...],
        seq_lens: torch.Tensor,
        block_table: torch.Tensor,
        sample_offsets: torch.Tensor,
        heads: int,
        config: SdsaConfig,
        kv_cache_dtype: str = "auto",
    ) -> None:
        num_requests = len(query_lens)
        self.block_table = block_table
        self.config = config
        self.query_lens, self.context_lens = query_lens, context_lens
        device = block_table.device

        # Every request occupies whole 128-query tiles. Row r of a tile holds
        # query query_rows[r] and sees keys < ends[r] (causal); padding rows
        # hold no query (-1) and see no keys.
        query_len_array = np.asarray(query_lens)
        context_len_array = np.asarray(context_lens)
        tiles = (query_len_array + 127) // 128
        tile_request = np.repeat(np.arange(num_requests), tiles)
        tile_start = 128 * (
            np.arange(tile_request.size) - np.repeat(np.cumsum(tiles) - tiles, tiles)
        )
        self.tile_counts = np.minimum(128, query_len_array[tile_request] - tile_start)
        row_in_tile = np.arange(128)
        is_query_row = row_in_tile < self.tile_counts[:, None]
        first_end = (context_len_array - query_len_array)[tile_request] + tile_start + 1
        ends = np.where(is_query_row, first_end[:, None] + row_in_tile, 0)
        request_query_start = np.cumsum(query_len_array) - query_len_array
        first_query = request_query_start[tile_request] + tile_start
        rows = np.where(is_query_row, first_query[:, None] + row_in_tile, -1).ravel()
        self.query_rows = async_tensor_h2d(rows, device, torch.int32)
        tiled = np.flatnonzero(rows >= 0)
        token_rows = np.empty(len(tiled), np.int64)
        token_rows[rows[tiled]] = tiled
        self.query_slots = async_tensor_h2d(
            np.r_[np.arange(first_row), first_row + token_rows], device, torch.int64
        )
        # The host's context lengths bound the tiles' work; the rows see keys up
        # to the exact lengths on the device, which async speculative decoding
        # gives the host only upper bounds of.
        tile_requests = async_tensor_h2d(tile_request.astype(np.int32), device)
        offsets = np.where(
            is_query_row, ends - context_len_array[tile_request, None], -(2**30)
        )
        offsets = async_tensor_h2d(offsets.astype(np.int32), device)
        self.tiles = SdsaTiles(
            tile_requests,
            (seq_lens[tile_requests, None] + offsets).clamp(min=0).flatten(),
            seq_lens,
            block_table,
            max_context=max(context_lens),
            sample_offsets=sample_offsets,
        )
        # The indexer has one head, so the TP ranks split its query tiles
        # instead (sequence parallel): rank r selects a contiguous run of them,
        # the runs balanced by their key tiles, and the ranks all-gather the
        # selections (`run`).
        tp = get_tp_group()
        work = np.cumsum((ends.max(1) + 127) // 128)
        shares = work[-1] * np.arange(1, tp.world_size) / tp.world_size
        self.tile_runs = np.r_[0, np.searchsorted(work, shares), len(work)]
        first, last = self.tile_runs[tp.rank_in_group : tp.rank_in_group + 2]

        # Panels: runs of whole tiles of one selector, a dense one within
        # _dense_tiles of its rows' longest end (its score width).
        fused = _fused(config.fused("prefill"), context_len_array[tile_request])
        self.panels: list[tuple[int, int, int]] = []  # begin, end, dense width
        for tile in range(first, last):
            width = 0 if fused[tile] else round_up(int(ends[tile].max()), 4)
            if self.panels:
                begin, end, panel_width = self.panels[-1]
                merged = max(width, panel_width)
                limit = _dense_tiles(merged) if merged else PANEL_ROWS // 128
                if (width > 0) == (panel_width > 0) and end + 1 - begin <= limit:
                    self.panels[-1] = (begin, end + 1, merged)
                    continue
            self.panels.append((tile, tile + 1, width))
        self.fused = any(not width for *_, width in self.panels)
        self.dense = any(width for *_, width in self.panels)
        # The step's workspace views, taken by the first run.
        self.arena: list[torch.Tensor] | None = None
        # The per-token cores take groups of one token: each union is then its
        # token's selected slots.
        self.core = config.prefill(
            heads=heads,
            context=max(context_lens),
            rows=sum(query_lens),
            kv_cache_dtype=kv_cache_dtype,
        )
        self.group = union_group(heads) if self.core == "union" else 1
        self.union_requests = tile_requests.repeat_interleave(128 // self.group)
        # NVFP4 cores convert by key position: each token's row of `unions`
        # then holds its Top-K key positions, not slots.
        self.keep_positions = kv_cache_dtype == "nvfp4"
        # The core's NVFP4 conversion launches for this step, set by its first
        # layer (SdsaAttentionImpl).
        self.nvfp4_launches: list | None = None

    def run(
        self,
        query: torch.Tensor,
        query_scales: torch.Tensor | None,
        key_cache: torch.Tensor,
        key_scales: torch.Tensor | None,
        *,
        out: torch.Tensor,
        valid_counts: torch.Tensor,
        mask: torch.Tensor,
        topk_workspace: torch.Tensor,
    ) -> None:
        """Select each row's Top-K keys, given the query in tiled rows (row r
        is query query_rows[r]) and the keys, both in the config's indexer
        dtype. The core attends to prefill tokens in groups of `group` tiled
        rows. Each group
        that holds queries is, in order, a batch b of FA4: its queries start at
        batch_starts[b], its union of the rows' keys is unions[b] (core cache
        slots, carved from `out`; size in valid_counts[b]), and each query's
        bitmask over that union is its row of `mask` (group_union_masks).
        Group g is batch group_batches[g], -1 if it holds no queries. The
        per-token cores take no masks: a token's union is its Top-K keys'
        slots (key positions if keep_positions), in Top-K order, -1 last.
        `topk_workspace` is gvr_2's zeroed scratch, shared with the decode rows.

        Initial and cached prefill are one pipeline: cached context is just
        more keys before the row's causal end. The step's threshold sample is
        gathered once; panels of whole 128-row query tiles of one selector
        then run in turn, in one panel scratch."""
        from flashinfer.sdsa import indexer_scores
        from flashinfer.token_sparse import group_union_masks

        tiles = self.tiles
        num_tiles = len(tiles.tile_requests)
        if self.arena is None:  # every layer of the step reuses the views
            layout = _arena_layout(
                len(tiles.sample_ends),
                tiles.sample_capacity if self.fused else 0,
                128 * max((end - begin for begin, end, _ in self.panels), default=0),
                self.config,
                self.dense,
            )
            self.arena = current_workspace_manager().get_simultaneous(*layout)
            group = self.group
            queries = np.arange(0, 128, group)
            sizes = np.clip(self.tile_counts[:, None] - queries, 0, group).ravel()
            used, device = sizes > 0, self.block_table.device
            batches = np.where(used, np.cumsum(used) - 1, -1)
            self.group_batches = async_tensor_h2d(batches, device, torch.int32)
            first_queries = np.r_[0, np.cumsum(sizes[used])]
            self.batch_starts = async_tensor_h2d(first_queries, device, torch.int32)
            columns = min(32 * mask.shape[1], group * self.config.top_k)
            if self.core != "union":  # each token's slots fill its row of `out`
                columns = self.config.top_k
            self.unions = out.flatten()[: used.sum() * columns].view(-1, columns)
            self.mask = mask
            # Each rank's selections: its tiles' groups' unions and their
            # queries' mask rows.
            unions = np.r_[0, np.cumsum(used.reshape(num_tiles, -1).sum(1))]
            queries = np.r_[0, np.cumsum(self.tile_counts)]
            self.rank_unions = np.diff(unions[self.tile_runs]).tolist()
            self.rank_queries = np.diff(queries[self.tile_runs]).tolist()
        sample_keys, sample_scales, *scratch, dense_scores = self.arena
        if self.fused:
            samples = tiles.gather_sample(
                key_cache, key_scales, sample_keys, sample_scales
            )
        for begin, end, width in self.panels:
            rows = slice(128 * begin, 128 * end)
            tile_query, tile_scales = query[rows], _rows(query_scales, rows)
            if width:  # dense: score the keys, then Top-K
                positions = scratch[6][: rows.stop - rows.start]
                scores = dense_scores[: len(positions) * width].view(-1, width)
                ends = tiles.row_ends[rows]
                indexer_scores(
                    tile_query,
                    key_cache,
                    self.block_table,
                    ends.view(-1, 8).amax(1),
                    requests=tiles.group_requests[16 * begin : 16 * end],
                    scores=scores,
                    query_scales=tile_scales,
                    key_scales=key_scales,
                )
                ops.top_k_per_row_prefill(
                    scores,
                    torch.zeros_like(ends),
                    ends,
                    positions,
                    len(ends),
                    width,
                    1,
                    self.config.top_k,
                )
            else:
                positions = tiles.select(
                    begin,
                    end,
                    tile_query,
                    tile_scales,
                    key_cache,
                    key_scales,
                    samples,
                    scratch,
                    topk_workspace,
                    self.config,
                )
            if self.core != "union":  # each token's Top-K keys as its slots
                _token_slots(
                    positions,
                    self.query_rows[rows],
                    self.union_requests[rows],
                    self.block_table,
                    self.unions,
                    valid_counts,
                    keep_positions=self.keep_positions,
                )
                continue
            # Each group's union of its rows' Top-K keys, and their bitmasks.
            groups = slice(128 * begin // self.group, 128 * end // self.group)
            group_union_masks(
                positions,
                self.query_rows[rows],
                self.group_batches[groups],
                self.union_requests[groups],
                self.block_table,
                union=self.unions,
                union_sizes=valid_counts,
                mask=mask,
                num_keys=tiles.max_context,
            )
        tp = get_tp_group()
        if tp.world_size > 1:  # every rank's unions, their sizes and mask rows
            width = min(self.unions.shape[1], 128 * tiles.key_tiles)
            words = min(mask.shape[1], 4 * tiles.key_tiles)
            rank = tp.rank_in_group
            unions, queries = (
                slice(sum(sizes[:rank]), sum(sizes[: rank + 1]))
                for sizes in (self.rank_unions, self.rank_queries)
            )
            mine = [self.unions[unions, :width].contiguous(), valid_counts[unions]]
            gathered = tp.all_gatherv(mine, sizes=self.rank_unions)
            self.unions[: len(gathered[1]), :width] = gathered[0]
            valid_counts[: len(gathered[1])] = gathered[1]
            if self.core == "union":
                mine = mask[queries, :words].contiguous()
                gathered = tp.all_gatherv(mine, sizes=self.rank_queries)
                mask[: len(gathered), :words] = gathered
