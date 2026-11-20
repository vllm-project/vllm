# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sparse attention indexer on aiter's paged MXFP4 MQA-logits kernel.

gfx950 only. The kernel reads the preshuffled paged indexer K cache in place,
so no layer gathers K into a contiguous buffer, prefill included. Any model
whose indexer supplies the inputs `rocm_mxfp4_sparse_attn_indexer` documents
can run the dense path. DeepSeek-V4.1's two-level indexer adds to it: the
candidate source takes its block maxima from the same walk that writes its
logits, and the candidate consumers can walk the candidate pool instead of
the whole context: the pool is resolved once per step and shared by all of
them.

With `VLLM_ROCM_USE_AITER_MXFP4_INDEXER_TOPK`, decode layers where aiter's
fused FlyDSL op (`aiter.ops.flydsl.mxfp4_indexer_topk`) is faster score and
select their top-512 in it instead of writing the logits; see
`_fused_decode_topk`.
"""

import functools
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from types import ModuleType
from typing import TYPE_CHECKING, Any, NamedTuple

import torch

import vllm.envs as envs
from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.config import CUDAGraphMode, get_current_vllm_config
from vllm.distributed import get_tp_group
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.kernels.attention.dsa.candidate_blocks import (
    apply_candidate_mask,
)
from vllm.model_executor.warmup.jit_warmup import VllmJitKernel, kernel_launcher
from vllm.model_executor.warmup.jit_warmup_triton_helper import (
    LaunchSpec,
    TritonWarmupTensor,
    VllmTritonJitKernel,
)
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.v1.attention.ops.metadata import _token_request
from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (
    _apply_candidate_mask_strided,
    _max_decode_logits_rows,
)
from vllm.v1.worker.workspace import current_workspace_manager

if TYPE_CHECKING:
    from vllm.models.deepseek_v41.attention import DeepseekV4IndexerCache
    from vllm.v1.attention.backends.mla.indexer import (
        DeepseekV32IndexerPrefillChunkMetadata,
    )
    from vllm.v1.attention.backends.mla.rocm_paged_mxfp4_indexer import (
        RocmMxfp4GatherLaunch,
        RocmMxfp4IndexerMetadata,
        RocmMxfp4NativeDecode,
        RocmMxfp4PrefillPlan,
    )

logger = init_logger(__name__)

MXFP4_BLOCK_SIZE = 32


@functools.cache
def _aiter():
    """The aiter module with the paged MXFP4 MQA-logits launcher, its schedule
    and cache_format."""
    from aiter.ops.triton.attention import pa_mqa_logits_mxfp4

    return pa_mqa_logits_mxfp4


@functools.cache
def _aiter_topk() -> Callable[..., None]:
    """The aiter per-row top-k."""
    from aiter.ops.topk import top_k_per_row_decode

    return top_k_per_row_decode


@functools.cache
def _aiter_fused_topk() -> ModuleType | None:
    """The aiter module with the fused MXFP4 indexer top-512, or None when
    `VLLM_ROCM_USE_AITER_MXFP4_INDEXER_TOPK` is off or aiter has no such op."""
    if not (envs.VLLM_ROCM_USE_AITER and envs.VLLM_ROCM_USE_AITER_MXFP4_INDEXER_TOPK):
        return None
    try:
        from aiter.ops.flydsl import mxfp4_indexer_topk
    except Exception as e:
        logger.warning_once(
            "VLLM_ROCM_USE_AITER_MXFP4_INDEXER_TOPK is set, but "
            "aiter.ops.flydsl.mxfp4_indexer_topk could not be imported (%s); "
            "the indexer keeps the MQA logits + top-k path.",
            e,
        )
        return None
    return mxfp4_indexer_topk


class RocmPagedMxfp4CacheLayout(NamedTuple):
    """The byte order of an indexer K page, as the kernel reads it, and the
    shuffle pattern aiter's K cache op writes it with.

    A page is cut into runs of `n_per_tile` tokens. Inside a run, each token's
    packed values are split into `d_per_tile`-byte chunks along the head
    dimension, and the run stores chunk 0 of every token, then chunk 1, and so
    on: `[chunk, token, byte]`. Its e8m0 scales are stored as
    `[scale % scale_lanes, token, scale // scale_lanes]`.
    """

    n_per_tile: int
    d_per_tile: int
    scale_lanes: int


# The scale order the kernel reads: aiter's mode 1, a lane's scales innermost.
# cache_format does not report the order or the lane split yet.
_SCALE_ORDER = 1
_WAVE_SIZE = 64


@functools.cache
def rocm_paged_mxfp4_cache_layout(
    num_heads: int, head_dim: int, page_entries: int
) -> RocmPagedMxfp4CacheLayout:
    """The K page layout from aiter's `cache_format`, the only place vLLM
    reads it, so a change there cannot reach the writer unchecked.

    Raises:
        ValueError: aiter rejects the page size, or describes an order this
            layout cannot express.

    """
    fmt = _aiter().cache_format(num_heads, head_dim, page_entries)
    try:
        n_per_tile = int(fmt["n_per_tile"])
        d_per_tile = int(fmt["d_per_tile"])
        scale_order = fmt.get("scale_mode", _SCALE_ORDER)
        scale_lanes = int(fmt.get("scale_lanes", _WAVE_SIZE // n_per_tile))
    except (KeyError, TypeError, ValueError, ZeroDivisionError) as e:
        raise ValueError(f"aiter's cache_format() returned {fmt!r}") from e
    if (
        scale_order != _SCALE_ORDER
        or min(n_per_tile, d_per_tile, scale_lanes) <= 0
        or page_entries % n_per_tile
        or (head_dim // 2) % d_per_tile
        or (head_dim // MXFP4_BLOCK_SIZE) % scale_lanes
    ):
        raise ValueError(
            f"aiter's cache_format() returned {fmt!r} for {num_heads} heads of "
            f"{head_dim} in {page_entries}-entry pages, a K page order "
            "RocmPagedMxfp4CacheLayout cannot express"
        )
    return RocmPagedMxfp4CacheLayout(n_per_tile, d_per_tile, scale_lanes)


@functools.cache
def _aiter_indexer_cache_ops() -> tuple[Callable[..., None], Callable[..., tuple]]:
    """The aiter indexer key writer and query quantizer."""
    from aiter.ops.triton.fusions.k_norm_rope_mxfp4_cache import (
        k_norm_rope_mxfp4_cache,
    )
    from aiter.ops.triton.rope.q_rope_mxfp4_quant import q_rope_mxfp4_quant

    return k_norm_rope_mxfp4_cache, q_rope_mxfp4_quant


def rocm_mxfp4_indexer_k_store(
    k_pre: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    rms_norm_weight: torch.Tensor,
    rms_norm_eps: float,
    k_cache: torch.Tensor,
    kv_slot_mapping: torch.Tensor,
    compress_ratio: int,
    use_fp4_cache: bool,
    *,
    num_heads: int,
) -> None:
    """`indexer_k_norm_rope_store` for the ROCm MXFP4 cache: aiter's cache op
    writes the key in the order its MQA-logits kernel reads with `num_heads`
    query heads."""
    assert use_fp4_cache, "the ROCm indexer cache op writes MXFP4 only"
    layout = rocm_paged_mxfp4_cache_layout(num_heads, k_pre.shape[1], k_cache.shape[1])
    k_norm_rope_mxfp4_cache, _ = _aiter_indexer_cache_ops()
    k_norm_rope_mxfp4_cache(
        k_pre,
        positions,
        cos_sin_cache,
        rms_norm_weight,
        rms_norm_eps,
        k_cache,
        kv_slot_mapping,
        compress_ratio,
        shuffle=layout,
    )


def rocm_mxfp4_indexer_q_quant(
    positions: torch.Tensor,
    index_q: torch.Tensor,
    index_q_cos_sin_cache: torch.Tensor,
    index_weights: torch.Tensor,
    index_weights_softmax_scale: float,
    index_weights_head_scale: float,
    use_fp4: bool = True,
    weights_out_dtype: torch.dtype = torch.float32,
) -> tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
    """`fused_indexer_q_rope_quant` for the ROCm MXFP4 indexer, on aiter's op:
    ((packed [T, H, D // 2], e8m0 as one int32 per head [T, H]), fp32
    weights)."""
    assert use_fp4 and weights_out_dtype == torch.float32, (
        "the ROCm indexer quantizes Q to MXFP4 and scores with fp32 weights"
    )
    _, q_rope_mxfp4_quant = _aiter_indexer_cache_ops()
    q_packed, q_scale, weights_out = q_rope_mxfp4_quant(
        index_q,
        positions,
        index_q_cos_sin_cache,
        index_weights,
        index_weights_softmax_scale * index_weights_head_scale,
    )
    return (q_packed, q_scale.view(torch.int32).squeeze(-1)), weights_out


def rocm_mxfp4_decode_schedule_words(
    num_heads: int, head_dim: int, page_entries: int, next_n: int = 1
) -> int:
    """int32 words of the largest schedule a decode step can build, flattened or
    not. The slice cap can take it past target_wgs, up to SCHED_SLOT_CAP."""
    words = 4 * _aiter().SCHED_SLOT_CAP
    for rows in {1, next_n}:
        config = _aiter().select_config(num_heads, head_dim, rows, page_entries)
        words = max(words, 4 * config["target_wgs"])
    return words


def build_rocm_mxfp4_decode_schedule(
    row_lens: torch.Tensor,
    num_heads: int,
    head_dim: int,
    page_entries: int,
    out: torch.Tensor,
    logits_width: int,
    native: "RocmMxfp4NativeDecode | None" = None,
) -> torch.Tensor | None:
    """Work descriptors that even out a decode step, or None where the static
    grid already fills the machine. Depends only on the rows' lengths, the
    cache geometry and the logits width, so one serves every layer of a group.
    The width sizes the slices, capped the way the static grid caps them."""
    if native is None:
        return _aiter().build_schedule(
            row_lens,
            1,
            num_heads,
            head_dim,
            page_entries,
            out=out,
            max_model_len=logits_width,
        )
    return _aiter().build_schedule(
        native.context_lens,
        native.next_n,
        num_heads,
        head_dim,
        page_entries,
        out=out,
        row_ends=row_lens,
        max_model_len=logits_width,
    )


class RocmMxfp4DecodeRowsKernel(
    VllmTritonJitKernel["RocmMxfp4DecodeRowsKernel.CompileKey"]
):
    TAIL_BLOCK = 256

    @dataclass(frozen=True, slots=True)
    class CompileKey:
        compress_ratio: int
        block_cols: int

    @staticmethod
    @triton.jit(
        do_not_specialize=[
            "num_reqs",
            "num_rows",
            "capacity",
            "num_ctx_reqs",
            "candidate_block",
            "block_table_stride",
            "row_block_table_stride",
        ]
    )
    def kernel(
        query_start_loc_ptr,
        seq_lens_ptr,
        block_table_ptr,
        row_lens_ptr,
        row_block_table_ptr,
        decode_lens_ptr,
        per_req_lens_ptr,
        context_lens_ptr,
        block_ends_ptr,
        num_reqs,
        num_rows,
        capacity,
        num_ctx_reqs,
        candidate_block,
        block_table_stride,
        row_block_table_stride,
        COMPRESS_RATIO: tl.constexpr,
        BLOCK_COLS: tl.constexpr,
        TAIL_BLOCK: tl.constexpr,
    ):
        pid = tl.program_id(0)
        if pid < num_rows:
            req = _token_request(
                query_start_loc_ptr, tl.full((1,), pid, tl.int32), num_reqs
            )
            req = tl.sum(req, 0)
            valid = pid < tl.load(query_start_loc_ptr + num_reqs)
            start = tl.load(query_start_loc_ptr + req)
            end = tl.load(query_start_loc_ptr + req + 1)
            seq_len = tl.load(seq_lens_ptr + req)
            length = tl.maximum(seq_len - (end - start) + pid - start + 1, 0)
            length = tl.where(valid, length, 0) // COMPRESS_RATIO
            tl.store(row_lens_ptr + pid, length)
            tl.store(decode_lens_ptr + pid, 1)
            if candidate_block > 0:
                ends = (length + candidate_block - 1) // candidate_block
                tl.store(block_ends_ptr + pid, ends)
            cols = tl.arange(0, triton.next_power_of_2(BLOCK_COLS))
            blocks = tl.load(
                block_table_ptr + req.to(tl.int64) * block_table_stride + cols,
                valid & (cols < BLOCK_COLS),
                other=0,
            )
            tl.store(
                row_block_table_ptr + pid.to(tl.int64) * row_block_table_stride + cols,
                blocks,
                cols < BLOCK_COLS,
            )
        else:
            tail = num_rows + (pid - num_rows) * TAIL_BLOCK + tl.arange(0, TAIL_BLOCK)
            tl.store(row_lens_ptr + tail, 0, tail < capacity)
        if pid < num_reqs:
            query_len = tl.load(query_start_loc_ptr + pid + 1) - tl.load(
                query_start_loc_ptr + pid
            )
            tl.store(per_req_lens_ptr + pid, query_len)
        if pid < num_ctx_reqs:
            context_len = tl.load(seq_lens_ptr + pid) // COMPRESS_RATIO
            tl.store(context_lens_ptr + pid, context_len)

    def dispatch(self, *, compress_ratio: int, block_cols: int) -> CompileKey:
        return self.CompileKey(compress_ratio=compress_ratio, block_cols=block_cols)

    def get_warmup_keys(
        self, *, compress_ratio: int, block_cols: int
    ) -> list[CompileKey]:
        return [self.dispatch(compress_ratio=compress_ratio, block_cols=block_cols)]

    def warmup_inputs(self, compile_key: CompileKey) -> dict[str, Any]:
        int32_ptr = TritonWarmupTensor(torch.int32)
        block_table = TritonWarmupTensor(torch.int32, shape=(1, compile_key.block_cols))
        return dict(
            query_start_loc=int32_ptr,
            seq_lens=int32_ptr,
            block_table=block_table,
            row_lens=int32_ptr,
            row_block_table=block_table,
            decode_lens=int32_ptr,
            per_req_lens=int32_ptr,
            context_lens=int32_ptr,
            block_ends=int32_ptr,
            num_reqs=1,
            num_rows=1,
            num_ctx_reqs=1,
            candidate_block=0,
            compress_ratio=compile_key.compress_ratio,
        )

    @kernel_launcher
    def __call__(
        self,
        query_start_loc: torch.Tensor,
        seq_lens: torch.Tensor,
        block_table: torch.Tensor,
        row_lens: torch.Tensor,
        row_block_table: torch.Tensor,
        decode_lens: torch.Tensor,
        per_req_lens: torch.Tensor,
        context_lens: torch.Tensor,
        block_ends: torch.Tensor,
        num_reqs: int,
        num_rows: int,
        num_ctx_reqs: int,
        candidate_block: int,
        compress_ratio: int,
    ) -> LaunchSpec:
        capacity = row_lens.shape[0]
        tail = num_rows + triton.cdiv(capacity - num_rows, self.TAIL_BLOCK)
        return (max(tail, num_ctx_reqs),), dict(
            capacity=capacity,
            block_table_stride=block_table.stride(0),
            row_block_table_stride=row_block_table.stride(0),
            COMPRESS_RATIO=compress_ratio,
            BLOCK_COLS=block_table.shape[1],
            TAIL_BLOCK=self.TAIL_BLOCK,
            num_warps=4,
        )


_ROCM_MXFP4_DECODE_ROWS_KERNEL = RocmMxfp4DecodeRowsKernel()


def _kv_view(kv_cache: torch.Tensor, head_dim: int) -> torch.Tensor:
    """The indexer cache as [pages, entries, 1, bytes]. The page stride is the
    block-major pool's, not the page's own size."""
    num_pages, entries, width = kv_cache.shape
    assert kv_cache.dtype == torch.uint8
    assert width == head_dim // 2 + head_dim // MXFP4_BLOCK_SIZE, width
    return torch.as_strided(
        kv_cache, (num_pages, entries, 1, width), (kv_cache.stride(0), width, width, 1)
    )


def rocm_mxfp4_consumer_rows(num_candidate_cols: int) -> int:
    """Query rows per candidate-consumer launch. Its logits are [rows, pool]
    fp32 whatever the context, so the logits budget alone sizes it."""
    budget = envs.VLLM_SPARSE_INDEXER_MAX_LOGITS_MB * 1024 * 1024
    return max(1, budget // (4 * num_candidate_cols))


def reserve_rocm_mxfp4_indexer_workspace(
    hidden_states: torch.Tensor,
    logits_width: int,
    candidate_block_size: int = 0,
    gather_block_size: int = 0,
    num_candidate_cols: int = 0,
    fused_kind: str | None = None,
) -> None:
    """Profiling run: claim the decode logits workspace and the peak prefill
    logits, block scores included when the layer writes them, candidate lists
    when it gathers, and the fused top-k's scratch for a `fused_kind` layer."""
    rows = _max_decode_logits_rows(hidden_states.shape[0])
    fused_bytes = _fused_topk_workspace_bytes(fused_kind, rows, logits_width)
    if fused_bytes:
        current_workspace_manager().get_simultaneous(((fused_bytes,), torch.uint8))
    specs = [((rows, logits_width), torch.float32)]
    budget = envs.VLLM_SPARSE_INDEXER_MAX_LOGITS_MB * 1024 * 1024
    if candidate_block_size:
        nblocks = triton.cdiv(logits_width, candidate_block_size)
        specs.append(((rows, nblocks), torch.float32))
        budget += budget // candidate_block_size
    if gather_block_size:
        # aiter allocates the candidate lists: each gathered row, 8 B per
        # candidate block (int32 slot and position). The first consumer builds
        # them and they live until the step ends, across the later layers, so
        # one reservation is held for the rest of the forward.
        held = get_forward_context().additional_kwargs
        if "rocm_mxfp4_candidate_lists" not in held:
            held["rocm_mxfp4_candidate_lists"] = torch.empty(
                hidden_states.shape[0] * (num_candidate_cols // gather_block_size) * 8,
                dtype=torch.uint8,
                device=hidden_states.device,
            )
    current_workspace_manager().get_simultaneous(*specs)
    torch.empty(budget, dtype=torch.uint8, device=hidden_states.device)


def _vllm_topk(rows: int, k: int) -> bool:
    return rows < 128 and k < 2048


def _topk(
    logits: torch.Tensor,
    lengths: torch.Tensor,
    out: torch.Tensor,
    k: int,
    step_rows: int | None = None,
    stable: bool = False,
) -> None:
    """Top-k over each row's [0, lengths[row]); -1 pads rows shorter than k.

    aiter's kernel is faster on the candidate top-k and from 128 rows on; below
    that, a wide decode step is faster on vLLM's. Shape alone decides, so a FULL
    graph replays the same choice at every context length. `step_rows` is the
    decode step's row count when `logits` holds a TP rank's slice of it, so the
    slice runs the kernel the whole step would. `stable` returns the indices
    in ascending order: aiter's kernel when it can run, else every row is at
    most k wide, which vLLM's kernel writes in order.
    """
    rows, width = logits.shape
    if rows == 0:
        return
    kernel_rows = rows if step_rows is None else step_rows
    if width >= k and (stable or not _vllm_topk(kernel_rows, k)):
        _aiter_topk()(
            logits,
            1,
            lengths.reshape(-1),
            out,
            rows,
            logits.stride(0),
            logits.stride(1),
            k=k,
            stable=stable,
        )
    else:
        torch.ops._C.top_k_per_row_decode(
            logits, 1, lengths, out, rows, logits.stride(0), logits.stride(1), k
        )


@triton.jit
def _remap_compact_topk_kernel(
    idx_ptr,
    idx_stride,
    pos_ptr,
    pos_stride,
    K: tl.constexpr,
    BLOCK: tl.constexpr,
    PADDED_K: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, PADDED_K)
    live = cols < K
    slot = tl.load(idx_ptr + row * idx_stride + cols, mask=live, other=-1)
    hit = slot >= 0
    slot = tl.where(hit, slot, 0)
    start = tl.load(
        pos_ptr + row * pos_stride + slot // BLOCK, mask=live & hit, other=0
    )
    tl.store(
        idx_ptr + row * idx_stride + cols,
        tl.where(hit, start + slot % BLOCK, -1).to(tl.int32),
        mask=live,
    )


def _remap_compact_topk(
    indices: torch.Tensor, positions: torch.Tensor, block: int
) -> None:
    """Candidate slots to context positions, in place: slot j sits in pool
    block j // block, which starts at positions[j // block]."""
    rows, k = indices.shape
    if rows == 0:
        return
    _remap_compact_topk_kernel[(rows,)](
        indices,
        indices.stride(0),
        positions,
        positions.stride(0),
        K=k,
        BLOCK=block,
        PADDED_K=triton.next_power_of_2(k),
        num_warps=4,
    )


@dataclass(slots=True)
class _Layer:
    """One indexer layer's inputs and outputs for this step."""

    metadata: "RocmMxfp4IndexerMetadata"
    kv: torch.Tensor
    q: torch.Tensor
    q_scale: torch.Tensor
    weights: torch.Tensor
    topk_buffer: torch.Tensor
    topk_tokens: int
    compress_ratio: int
    candidates: torch.Tensor | None
    block: int
    full_graph: bool

    @property
    def num_heads(self) -> int:
        return self.q.shape[1]

    @property
    def head_dim(self) -> int:
        return self.q.shape[2] * 2

    def rows(self, lo: int, hi: int, seqs: int = 1) -> tuple[torch.Tensor, ...]:
        """Q, its scales and the weights of token rows [lo, hi), as `seqs`
        sequences of next_n rows each."""
        return (
            self.q[lo:hi].view(seqs, -1, *self.q.shape[1:]),
            self.q_scale[lo:hi].view(seqs, -1, *self.q_scale.shape[1:]),
            self.weights[lo:hi],
        )

    def decode_rows(self, n: int) -> tuple[torch.Tensor, ...]:
        """The first n token rows, one sequence each."""
        return (
            self.q[:n].unsqueeze(1),
            self.q_scale[:n].unsqueeze(1),
            self.weights[:n],
        )


def _layer(
    k_cache_prefix: str,
    kv_cache: torch.Tensor,
    q_values: torch.Tensor,
    q_scale: torch.Tensor,
    weights: torch.Tensor,
    topk_indices_buffer: torch.Tensor,
    topk_tokens: int,
    head_dim: int,
    compress_ratio: int,
    candidate_blocks: torch.Tensor | None,
    candidate_block_size: int,
) -> _Layer:
    from vllm.v1.attention.backends.mla.rocm_paged_mxfp4_indexer import (
        RocmMxfp4IndexerMetadata,
    )

    forward_context = get_forward_context()
    attn_metadata = forward_context.attn_metadata
    assert isinstance(attn_metadata, dict)
    metadata = attn_metadata[k_cache_prefix]
    assert isinstance(metadata, RocmMxfp4IndexerMetadata), (
        "the ROCm MXFP4 indexer needs RocmMxfp4IndexerMetadataBuilder metadata"
    )
    kv = _kv_view(kv_cache, head_dim)
    num_heads = q_values.shape[1]
    return _Layer(
        metadata=metadata,
        kv=kv,
        q=q_values,
        # The fused Q kernel returns one int32 per head; the kernel reads the
        # four ue8m0 bytes behind it.
        q_scale=q_scale.view(torch.uint8).view(q_scale.shape[0], num_heads, -1),
        weights=weights,
        topk_buffer=topk_indices_buffer,
        topk_tokens=topk_tokens,
        compress_ratio=compress_ratio,
        candidates=candidate_blocks,
        block=candidate_block_size,
        full_graph=forward_context.cudagraph_runtime_mode == CUDAGraphMode.FULL,
    )


def _fill_topk_rows(layer: _Layer, num_tokens: int) -> None:
    lengths = layer.metadata.decode_row_lens
    start = 0
    if layer.metadata.decode is not None and lengths is not None:
        rows = _decode_step_rows(layer.metadata, lengths.shape[0])
        if _vllm_topk(rows, layer.topk_tokens):
            start = rows
    if start < num_tokens:
        layer.topk_buffer[start:num_tokens] = -1


def _block_scores(layer: _Layer, scores: torch.Tensor | None) -> dict:
    if scores is None:
        return {}
    return dict(
        calc_block_scores=True, block_scores=scores, candidate_block_size=layer.block
    )


def _dense_prefill(
    layer: _Layer,
    chunk: "DeepseekV32IndexerPrefillChunkMetadata",
    plan: "RocmMxfp4PrefillPlan",
    candidate_write: bool,
) -> None:
    pa = _aiter()
    t0, t1 = chunk.token_start, chunk.token_end
    logits = layer.q.new_empty((t1 - t0, plan.width), dtype=torch.float32)
    scores = None
    if candidate_write:
        nblocks = triton.cdiv(plan.width, layer.block)
        scores = layer.q.new_empty((t1 - t0, nblocks), dtype=torch.float32)
    # One launch for the chunk: the kernel finds each row's request from
    # query_start_loc on its own grid.
    pa.paged_mxfp4_mqa_logits(
        layer.q[t0:t1],
        layer.q_scale[t0:t1],
        layer.kv,
        layer.weights[t0:t1],
        plan.context_lens,
        chunk.block_table,
        plan.width,
        out_logits=logits,
        clean_logits=False,
        row_ends=plan.row_ends,
        query_start_loc=plan.query_start_loc,
        **_block_scores(layer, scores),
    )
    candidates = None if layer.candidates is None else layer.candidates[t0:t1]
    if scores is not None:
        assert candidates is not None and plan.block_ends is not None
        _topk(scores, plan.block_ends, candidates, candidates.shape[1])
    elif candidates is not None:
        apply_candidate_mask(logits, None, plan.row_ends, candidates, layer.block)
    _topk(
        logits,
        plan.row_ends,
        layer.topk_buffer[t0:t1, : layer.topk_tokens],
        layer.topk_tokens,
    )


# The candidate consumer's fused op beat the gather chain from this many
# entries a row on; at 4096 it lost with few rows.
_FUSED_CONSUMER_MIN_CTX = 16384
# A captured consumer launch is bounded by the static bound, whatever the
# rows' lengths. It beat the gather chain on 16K-128K-entry rows from this
# bound up; at 128K and 256K bounds it mostly lost (to 0.54x) with over 160
# rows of 8K-32K entries.
_FUSED_CAPTURED_MIN_BOUND = 1 << 19
# aiter's smallest launch tier, in entries a row; tiers double from it.
_FUSED_MIN_TIER = 4096


def _fused_topk_max_ctx(
    mod: ModuleType,
    kind: str,
    rows: int,
    step_bound: int,
    static_bound: int,
    captured: bool,
) -> int | None:
    """The fused op's `max_ctx` for a decode layer, or None to keep the chain.

    `kind` is aiter's layer name: "dense", "source" (also writes the candidate
    blocks) or "consumer" (selects among them). The bound sizes the launch, so
    an eager step takes its own longest row. A captured launch replays at every
    step: only the static bound fits them all, and there only the consumer was
    measured ahead, from `_FUSED_CAPTURED_MIN_BOUND` up. Dense and source
    layers follow aiter's measured crossover, `mxfp4_indexer_dense_preferred`.
    """
    if captured:
        if kind != "consumer" or static_bound < _FUSED_CAPTURED_MIN_BOUND:
            return None
        max_ctx = static_bound
    else:
        max_ctx = max(1, min(step_bound, static_bound))
    if kind == "consumer":
        return max_ctx if max_ctx >= _FUSED_CONSUMER_MIN_CTX else None
    source = kind == "source"
    if mod.mxfp4_indexer_dense_preferred(max_ctx, rows, source=source):
        return max_ctx
    return None


def _fused_topk_tiers(static_bound: int) -> list[int]:
    """One `max_ctx` per launch tier a call bounded by `static_bound` can
    take: aiter sizes its launches in power-of-two tiers of entries."""
    tiers = {static_bound}
    tier = _FUSED_MIN_TIER
    while tier < static_bound:
        tiers.add(tier)
        tier *= 2
    return sorted(tiers)


@functools.cache
def _fused_topk_workspace_bytes(
    kind: str | None, max_rows: int, static_bound: int
) -> int:
    """The most scratch a `kind` layer's fused calls of up to `max_rows` rows
    take; 0 when the op is off."""
    mod = _aiter_fused_topk()
    if mod is None or kind is None:
        return 0
    tiers = [
        t for t in _fused_topk_tiers(static_bound) if t <= mod.MXFP4_INDEXER_MAX_ENTRIES
    ]
    return max(
        (
            mod.mxfp4_indexer_workspace_bytes(rows, t, kind)
            for t in tiers
            for rows in range(1, max_rows + 1)
        ),
        default=0,
    )


class RocmMxfp4FusedTopkKernel(VllmJitKernel["RocmMxfp4FusedTopkKernel.CompileKey"]):
    """aiter's fused MXFP4 indexer top-512, which builds and loads a launch's
    kernels on its first call. A graph captures only launches that already ran
    (the rest keep the chain); the JIT warm-up runs every launch a decode step
    can pick, so graphs get them and eager steps do not compile while serving.
    """

    @dataclass(frozen=True, slots=True)
    class CompileKey:
        kind: str
        next_n: int
        page_entries: int
        max_ctx: int
        rows: int

    def get_warmup_keys(
        self,
        *,
        kind: str,
        k_cache: "DeepseekV4IndexerCache",
        static_bound: int,
        next_n: int,
        max_rows: int,
    ) -> list[CompileKey]:
        mod = _aiter_fused_topk()
        # The cache is bound after the layer registers.
        kv = k_cache.kv_cache
        if mod is None or kv.dim() != 3:
            return []
        page = kv.shape[1]
        keys: dict[RocmMxfp4FusedTopkKernel.CompileKey, None] = {}
        for rows in range(next_n, max_rows + 1, next_n):
            for step_bound, captured in (
                *((t, False) for t in _fused_topk_tiers(static_bound)),
                (static_bound, True),
            ):
                max_ctx = _fused_topk_max_ctx(
                    mod, kind, rows, step_bound, static_bound, captured
                )
                if max_ctx is not None and not mod.mxfp4_indexer_unsupported_reason(
                    page, max_ctx, next_n
                ):
                    keys[self.CompileKey(kind, next_n, page, max_ctx, rows)] = None
        return list(keys)

    def compile(self, compile_key: CompileKey) -> None:
        self.compile_many([compile_key])

    def compile_many(self, compile_keys: Iterable[CompileKey]) -> None:
        """Run each launch once on zero-length rows."""
        mod = _aiter_fused_topk()
        missing = [k for k in compile_keys if k not in self._compiled_cache]
        if mod is None or not missing:
            return
        start = time.perf_counter()
        rows = max(k.rows for k in missing)
        heads, dim = mod.MXFP4_INDEXER_HEADS, mod.MXFP4_INDEXER_HEAD_DIM
        width = dim // 2 + dim // MXFP4_BLOCK_SIZE

        def zeros(*shape, dtype=torch.uint8):
            return torch.zeros(shape, dtype=dtype, device=current_platform.device_type)

        q = zeros(rows, heads, dim // 2)
        q_scale = zeros(rows, heads, dim // MXFP4_BLOCK_SIZE)
        weights = zeros(rows, heads, dtype=torch.float32)
        block_table = zeros(rows, 1, dtype=torch.int32)
        row_ends = zeros(rows, dtype=torch.int32)
        out = zeros(rows, mod.MXFP4_INDEXER_TOPK, dtype=torch.int32)
        candidates = zeros(rows, mod.MXFP4_INDEXER_CANDIDATES, dtype=torch.int32) - 1
        for key in missing:
            n = key.rows
            self.launch(
                mod,
                key,
                q[:n],
                q_scale[:n],
                weights[:n],
                zeros(1, key.page_entries, 1, width),
                block_table[: n // key.next_n],
                row_ends[:n],
                None if key.kind == "dense" else candidates[:n],
                out[:n],
            )
        torch.accelerator.synchronize()
        logger.info(
            "Ran %d fused MXFP4 indexer top-k launches in %.1fs.",
            len(missing),
            time.perf_counter() - start,
        )

    def launch(
        self,
        mod: ModuleType,
        key: CompileKey,
        q: torch.Tensor,
        q_scale: torch.Tensor,
        weights: torch.Tensor,
        kv: torch.Tensor,
        block_table: torch.Tensor,
        row_ends: torch.Tensor,
        candidates: torch.Tensor | None,
        out: torch.Tensor,
    ) -> bool:
        """Launch the fused op; False (nothing launched) when the launch never
        ran before and the stream is capturing."""
        if key not in self._compiled_cache and torch.cuda.is_current_stream_capturing():
            return False
        nbytes = mod.mxfp4_indexer_workspace_bytes(key.rows, key.max_ctx, key.kind)
        workspace = None
        if nbytes:
            (workspace,) = current_workspace_manager().get_simultaneous(
                ((nbytes,), torch.uint8)
            )
        args = (q, q_scale, weights, kv, block_table, row_ends)
        if key.kind == "consumer":
            assert candidates is not None
            mod.flydsl_mxfp4_indexer_candidate_topk(
                *args,
                candidates,
                out,
                next_n=key.next_n,
                max_ctx=key.max_ctx,
                workspace=workspace,
            )
        else:
            mod.flydsl_mxfp4_indexer_topk(
                *args,
                out,
                next_n=key.next_n,
                max_ctx=key.max_ctx,
                candidates_out=candidates,
                workspace=workspace,
            )
        self._compiled_cache[key] = True
        return True


_ROCM_MXFP4_FUSED_TOPK_KERNEL = RocmMxfp4FusedTopkKernel()


def register_rocm_mxfp4_fused_topk_warmup(
    kind: str | None, k_cache: "DeepseekV4IndexerCache", static_bound: int
) -> None:
    """Register a decode layer's fused top-k launches for the JIT warm-up. Call
    from the layer's constructor; `static_bound` is its row bound in entries.

    The warm-up runs outside any config context, so the decode row bound is
    taken here, the one the profile run reserves the op's scratch for.
    """
    if kind is None or not envs.VLLM_ROCM_USE_AITER_MXFP4_INDEXER_TOPK:
        return
    vllm_config = get_current_vllm_config()
    spec = vllm_config.speculative_config
    # Steps without next_n-row requests (no speculation, adaptive
    # verification) never take the fused op.
    if (
        not vllm_config.kernel_config.enable_jit_warmup
        or spec is None
        or spec.enable_adaptive_verification
    ):
        return
    next_n = spec.num_speculative_tokens + 1
    max_rows = _max_decode_logits_rows(
        vllm_config.scheduler_config.max_num_batched_tokens, vllm_config
    )
    _ROCM_MXFP4_FUSED_TOPK_KERNEL.register_warmup(
        kind=kind,
        k_cache=k_cache,
        static_bound=static_bound,
        next_n=next_n,
        max_rows=max_rows - max_rows % next_n,
    )


def _fused_decode_topk(layer: _Layer, kind: str, static_bound: int) -> bool:
    """Score and select the decode rows' top-k with aiter's fused op if it
    serves this step, writing `layer.topk_buffer` (and, for the source, the
    candidate blocks). False leaves the step to the logits + top-k chain.

    Host values decide, so a captured graph replays its choice. Only steps of
    next_n-row requests (`decode_native`) are served, the case aiter measured.
    """
    mod = _aiter_fused_topk()
    metadata = layer.metadata
    native, lengths = metadata.decode_native, metadata.decode_row_lens
    if mod is None or native is None or lengths is None or lengths.shape[0] == 0:
        return False
    rows = lengths.shape[0]
    max_ctx = _fused_topk_max_ctx(
        mod,
        kind,
        rows,
        metadata.max_seq_len // layer.compress_ratio,
        static_bound,
        layer.full_graph or torch.cuda.is_current_stream_capturing(),
    )
    kv = layer.kv
    if max_ctx is None or mod.mxfp4_indexer_unsupported_reason(
        kv.shape[1],
        max_ctx,
        native.next_n,
        layer.num_heads,
        layer.head_dim,
        layer.topk_tokens,
    ):
        return False
    out = layer.topk_buffer[:rows]
    candidates = None
    if kind != "dense":
        assert layer.candidates is not None
        candidates = layer.candidates[:rows]
        if (
            layer.block != mod.MXFP4_INDEXER_CANDIDATE_BLOCK
            or candidates.shape[1] != mod.MXFP4_INDEXER_CANDIDATES
            or candidates.dtype != torch.int32
            or not candidates.is_contiguous()
        ):
            return False
    q, q_scale, weights = layer.q[:rows], layer.q_scale[:rows], layer.weights[:rows]
    # aiter raises on any other layout; those steps keep the chain.
    block_table = native.block_table
    if (
        out.shape[1] != layer.topk_tokens
        or kv.stride(0) % 16
        or q.dtype != torch.uint8
        or q_scale.dtype not in (torch.uint8, torch.int32)
        or weights.dtype != torch.float32
        or any(t.dtype != torch.int32 for t in (lengths, out, block_table))
        or block_table.stride(1) != 1
        or not all(t.is_contiguous() for t in (out, q, q_scale, weights, lengths))
    ):
        return False
    key = RocmMxfp4FusedTopkKernel.CompileKey(
        kind, native.next_n, kv.shape[1], max_ctx, rows
    )
    return _ROCM_MXFP4_FUSED_TOPK_KERNEL.launch(
        mod, key, q, q_scale, weights, kv, block_table, lengths, candidates, out
    )


def rocm_mxfp4_fused_topk_kind(
    candidate_blocks: torch.Tensor | None, candidate_write: bool
) -> str | None:
    """A dense-indexer layer's kind for the fused op; None for a consumer
    masked to the pool, which keeps the chain."""
    if candidate_write:
        return "source"
    return "dense" if candidate_blocks is None else None


def _dense_decode(layer: _Layer, logits_width: int, candidate_write: bool) -> None:
    metadata = layer.metadata
    lengths = metadata.decode_row_lens
    assert metadata.decode is not None and lengths is not None
    rows = lengths.shape[0]
    if rows == 0:
        return
    kind = rocm_mxfp4_fused_topk_kind(layer.candidates, candidate_write)
    if kind is not None and _fused_decode_topk(layer, kind, logits_width):
        return
    specs = [((rows, logits_width), torch.float32)]
    if candidate_write:
        specs.append(((rows, triton.cdiv(logits_width, layer.block)), torch.float32))
    logits, *scores = current_workspace_manager().get_simultaneous(*specs)
    native = metadata.decode_native
    if native is not None:
        # A request's next_n rows go in as one sequence, so a workgroup walks
        # each KV tile once for all of them.
        q, q_scale, weights = layer.rows(0, rows, native.context_lens.shape[0])
        context_lens, block_table, row_ends = (
            native.context_lens,
            native.block_table,
            lengths,
        )
    else:
        q, q_scale, weights = layer.decode_rows(rows)
        context_lens, block_table, row_ends = lengths, metadata.decode.block_table, None
        assert block_table.shape[0] == rows, "one block-table row per query row"
    _aiter().paged_mxfp4_mqa_logits(
        q,
        q_scale,
        layer.kv,
        weights,
        context_lens,
        block_table,
        logits_width,
        out_logits=logits,
        clean_logits=False,
        row_ends=row_ends,
        schedule=metadata.decode_schedule,
        **_block_scores(layer, scores[0] if scores else None),
    )
    candidates = None if layer.candidates is None else layer.candidates[:rows]
    if scores:
        assert candidates is not None and metadata.decode_block_ends is not None
        # The fused consumers take the candidate blocks in ascending order.
        _topk(
            scores[0],
            metadata.decode_block_ends,
            candidates,
            candidates.shape[1],
            stable=_aiter_fused_topk() is not None,
        )
    elif candidates is not None:
        _apply_candidate_mask_strided(logits, None, lengths, candidates, layer.block)
    _topk(
        logits,
        lengths,
        layer.topk_buffer[:rows, : layer.topk_tokens],
        layer.topk_tokens,
        _decode_step_rows(metadata, rows),
    )


def _gather_prefill(
    layer: _Layer, launch: "RocmMxfp4GatherLaunch", num_cols: int
) -> None:
    pa = _aiter()
    assert layer.candidates is not None
    t0, t1 = launch.token_start, launch.token_end
    if launch.pool is None:
        launch.pool = pa.build_candidate_gather(
            layer.candidates[t0:t1],
            launch.row_ends,
            launch.block_table.expand(t1 - t0, -1),
            layer.kv,
            layer.num_heads,
            layer.head_dim,
            layer.block,
        )
    gather, slot_ends = launch.pool
    compact = layer.q.new_empty((t1 - t0, num_cols), dtype=torch.float32)
    q, q_scale, weights = layer.rows(t0, t1)
    pa.paged_mxfp4_mqa_logits(
        q,
        q_scale,
        layer.kv,
        weights,
        launch.context_len,
        launch.block_table,
        num_cols,
        out_logits=compact,
        clean_logits=False,
        row_ends=slot_ends,
        use_gather=True,
        candidates=gather,
    )
    out = layer.topk_buffer[t0:t1, : layer.topk_tokens]
    _topk(compact, slot_ends, out, layer.topk_tokens)
    _remap_compact_topk(out, gather["positions"], layer.block)


def _gather_decode(layer: _Layer, num_cols: int) -> None:
    pa = _aiter()
    assert layer.candidates is not None
    metadata = layer.metadata
    lengths = metadata.decode_row_lens
    assert metadata.decode is not None and lengths is not None
    block_table = metadata.decode.block_table
    rows = lengths.shape[0]
    if rows == 0:
        return
    if metadata.decode_gather is None:
        metadata.decode_gather = pa.build_candidate_gather(
            layer.candidates[:rows],
            lengths,
            block_table,
            layer.kv,
            layer.num_heads,
            layer.head_dim,
            layer.block,
        )
    gather, slot_ends = metadata.decode_gather
    (compact,) = current_workspace_manager().get_simultaneous(
        ((rows, num_cols), torch.float32)
    )
    q, q_scale, weights = layer.decode_rows(rows)
    pa.paged_mxfp4_mqa_logits(
        q,
        q_scale,
        layer.kv,
        weights,
        lengths,
        block_table,
        num_cols,
        out_logits=compact,
        clean_logits=False,
        row_ends=slot_ends,
        use_gather=True,
        candidates=gather,
    )
    out = layer.topk_buffer[:rows, : layer.topk_tokens]
    _topk(compact, slot_ends, out, layer.topk_tokens, _decode_step_rows(metadata, rows))
    _remap_compact_topk(out, gather["positions"], layer.block)


def _decode_step_rows(metadata: "RocmMxfp4IndexerMetadata", rows: int) -> int:
    shard = metadata.decode_shard
    return rows if shard is None else shard.num_rows


def _decode_rank_rows(layer: _Layer) -> _Layer:
    shard = layer.metadata.decode_shard
    if shard is None:
        return layer
    lo = shard.start
    return replace(
        layer,
        q=layer.q[lo:],
        q_scale=layer.q_scale[lo:],
        weights=layer.weights[lo:],
        topk_buffer=layer.topk_buffer[lo:],
        candidates=None if layer.candidates is None else layer.candidates[lo:],
    )


def _exchange_decode_rows(layer: _Layer) -> None:
    shard = layer.metadata.decode_shard
    if shard is None:
        return
    width = layer.topk_tokens
    local = layer.topk_buffer[shard.start : shard.start + shard.span, :width]
    gathered = get_tp_group().all_gather(local.contiguous().view(torch.float32), 0)
    layer.topk_buffer[: shard.num_rows, :width] = gathered.view(torch.int32)[
        : shard.num_rows
    ]


@eager_break_during_capture
def rocm_mxfp4_sparse_attn_indexer(
    hidden_states: torch.Tensor,
    k_cache_prefix: str,
    kv_cache: torch.Tensor,
    q_values: torch.Tensor,
    q_scale: torch.Tensor,
    weights: torch.Tensor,
    topk_tokens: int,
    head_dim: int,
    max_model_len: int,
    topk_indices_buffer: torch.Tensor,
    compress_ratio: int = 1,
    candidate_blocks: torch.Tensor | None = None,
    candidate_block_size: int = 0,
    candidate_write: bool = False,
) -> torch.Tensor:
    """Dense indexer: every layer that scores the whole context, the
    candidate source included. With `candidate_blocks` and not
    `candidate_write` the scores are masked to the pool first, as the
    shared path does.

    The model supplies `q_values` [T, H, D // 2] packed e2m1, `q_scale`
    holding each head's D // 32 ue8m0 bytes (one int32 per head at D = 128),
    fp32 `weights` with the softmax and head scales folded in, and
    `kv_cache` pages in `rocm_paged_mxfp4_cache_layout` order for H heads.
    """
    if not isinstance(get_forward_context().attn_metadata, dict):
        reserve_rocm_mxfp4_indexer_workspace(
            hidden_states,
            max_model_len,
            candidate_block_size if candidate_write else 0,
            fused_kind=rocm_mxfp4_fused_topk_kind(candidate_blocks, candidate_write),
        )
        return topk_indices_buffer
    layer = _layer(
        k_cache_prefix,
        kv_cache,
        q_values,
        q_scale,
        weights,
        topk_indices_buffer,
        topk_tokens,
        head_dim,
        compress_ratio,
        candidate_blocks,
        candidate_block_size,
    )
    metadata = layer.metadata
    _fill_topk_rows(layer, hidden_states.shape[0])
    if metadata.prefill is not None:
        for chunk, plan in zip(metadata.prefill.chunks, metadata.prefill_plans):
            _dense_prefill(layer, chunk, plan, candidate_write)
    if metadata.decode is not None:
        _dense_decode(_decode_rank_rows(layer), max_model_len, candidate_write)
        _exchange_decode_rows(layer)
    return topk_indices_buffer


@eager_break_during_capture
def rocm_mxfp4_sparse_mqa_indexer(
    hidden_states: torch.Tensor,
    k_cache_prefix: str,
    kv_cache: torch.Tensor,
    q_values: torch.Tensor,
    q_scale: torch.Tensor,
    weights: torch.Tensor,
    topk_tokens: int,
    head_dim: int,
    max_model_len: int,
    topk_indices_buffer: torch.Tensor,
    compress_ratio: int,
    candidate_blocks: torch.Tensor,
    candidate_block_size: int,
    num_candidate_cols: int,
) -> torch.Tensor:
    """Candidate consumer: score only the source's pool where the builder's
    length gate says it pays, else the dense walk masked to the pool."""
    if not isinstance(get_forward_context().attn_metadata, dict):
        reserve_rocm_mxfp4_indexer_workspace(
            hidden_states,
            max(max_model_len, num_candidate_cols),
            gather_block_size=candidate_block_size,
            num_candidate_cols=num_candidate_cols,
            fused_kind="consumer",
        )
        return topk_indices_buffer
    layer = _layer(
        k_cache_prefix,
        kv_cache,
        q_values,
        q_scale,
        weights,
        topk_indices_buffer,
        topk_tokens,
        head_dim,
        compress_ratio,
        candidate_blocks,
        candidate_block_size,
    )
    metadata = layer.metadata
    _fill_topk_rows(layer, hidden_states.shape[0])
    if metadata.prefill is not None:
        for chunk, plan in zip(metadata.prefill.chunks, metadata.prefill_plans):
            if not plan.use_gather:
                _dense_prefill(layer, chunk, plan, candidate_write=False)
        for launch in metadata.gather_launches:
            _gather_prefill(layer, launch, num_candidate_cols)
    if metadata.decode is not None:
        decode = _decode_rank_rows(layer)
        # A FULL graph cannot follow the per-step gate; the gather is the side
        # that stays flat in the context length, as is the fused op.
        if metadata.decode_use_gather or layer.full_graph:
            if not _fused_decode_topk(decode, "consumer", max_model_len):
                _gather_decode(decode, num_candidate_cols)
        else:
            _dense_decode(decode, max_model_len, candidate_write=False)
        _exchange_decode_rows(layer)
    return topk_indices_buffer
