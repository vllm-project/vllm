# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The low-batch sparse attention core of MiniMax-M3.1-Flash decode steps.

After the stock qkv projection and KV insert, for 1-32 query rows (up to 32
requests of up to 8 queries) of one TP rank (16 query heads on one KV head,
one index head, NVFP4 KV and index caches, E4M3 queries), three PDL-chained
launches a layer replace the stock indexer, top-k, sparse FMHA, o_proj input
quantization and GEMM:

1. ``indexer_cute.IndexScores``: every history page's index score, from the
   step's list of pages (``indexer_cute.IndexWork``, once a step in
   ``prepare_step``);
2. ``sparse_cute.SparseAttention``: each row's top-15 pages and local page,
   the attention over them, o_proj's MXFP8 input;
3. ``o_proj``: the MXFP8 GEMM, whose weight loads overlap the attention.

Every launch is sized from the step's request and query counts; the context
lengths and pages come from the device seq_lens and block table only, so a
CUDA graph captured once serves every later context length up to 8192 pages.
"""

import math
from functools import cache
from typing import Any, NamedTuple

import torch

from vllm.models.common.ops.moe_tail_fusion.primitives import current_cu_stream
from vllm.platforms import current_platform

from .o_proj import compile_all, mxfp8_weights, o_proj_gemm

MAX_ROWS = 32
MAX_REQS = 32
MAX_QUERY_LEN = 8
HEADS = 16
HEAD_DIM = 128
HIDDEN = 6144
PAGE = 128
MAX_PAGES = 8192
INDEX_CTAS_PER_SM = 4
ATTENTION_CTAS = 8  # CTAs a row, two pages each
# Partial maxima and sums, then partial outputs, of up to 8 CTAs a row.
PARTIAL_BYTES = MAX_ROWS * 8 * (2 * HEADS + HEADS * HEAD_DIM) * 4


class AttentionStep(NamedTuple):
    """One decode step's attention metadata, shared by every layer."""

    block_table: torch.Tensor
    seq_lens: torch.Tensor
    query_len: int
    # The pages the indexer scores (``indexer_cute.IndexWork``).
    work: torch.Tensor


def prepare_step(
    block_table: torch.Tensor, seq_lens: torch.Tensor, query_len: int
) -> AttentionStep:
    """The step of ``seq_lens.shape[0]`` requests of ``query_len`` queries:
    ``block_table`` int32 [requests, pages] and ``seq_lens`` int32
    [requests] (0 for padding requests), both read in place. Lists the pages
    to score on the current stream."""
    from cutlass import Int32, Int64

    assert block_table.dtype == torch.int32 and seq_lens.dtype == torch.int32
    assert block_table.shape[0] == seq_lens.shape[0] and block_table.stride(1) == 1
    assert block_table.shape[1] <= MAX_PAGES and seq_lens.is_contiguous()
    kernels = _kernels(seq_lens.device.index)
    work = _scratch(seq_lens.device)[3]
    kernels.work(
        Int64(block_table.data_ptr()),
        Int32(block_table.stride(0)),
        Int64(seq_lens.data_ptr()),
        Int64(work.data_ptr()),
        Int32(seq_lens.shape[0]),
        current_cu_stream(),
    )
    return AttentionStep(block_table, seq_lens, query_len, work)


class _Kernels(NamedTuple):
    sms: int
    work: Any
    indexer: Any
    attention: Any


@cache
def _kernels(device_index: int) -> _Kernels:
    import cutlass.cute as cute
    from cutlass import Float32, Int32, Int64
    from cutlass.cute.runtime import make_fake_stream

    from .indexer_cute import IndexScores, IndexWork
    from .sparse_cute import SparseAttention

    sms = torch.cuda.get_device_properties(device_index).multi_processor_count
    with torch.accelerator.device_index(device_index):
        work = cute.compile(
            IndexWork(),
            Int64(0),
            Int32(0),
            Int64(0),
            Int64(0),
            Int32(1),
            make_fake_stream(),
        )
        indexer = cute.compile(
            IndexScores(INDEX_CTAS_PER_SM * sms),
            *(Int64(0),) * 5,
            Int32(1),
            make_fake_stream(),
        )
        attention = cute.compile(
            SparseAttention(ATTENTION_CTAS),
            *(Int64(0),) * 4,
            Int32(0),
            *(Int64(0),) * 8,
            Float32(0.0),
            Int32(1),
            Int32(1),
            make_fake_stream(),
        )
    compile_all(device_index)
    return _Kernels(sms, work, indexer, attention)


@cache
def _scratch(device: torch.device):
    """Buffers the layers reuse in turn (index scores with their physical
    pages, o_proj's MXFP8 input and its scales) and the step's work list."""
    from .indexer_cute import work_entries

    ctas = (
        INDEX_CTAS_PER_SM
        * torch.cuda.get_device_properties(device).multi_processor_count
    )
    return (
        torch.empty(MAX_ROWS, MAX_PAGES, 2, dtype=torch.int32, device=device),
        torch.zeros(MAX_ROWS, HEADS * HEAD_DIM, dtype=torch.uint8, device=device),
        torch.zeros(128 * HEADS * HEAD_DIM // 32, dtype=torch.uint8, device=device),
        torch.zeros(work_entries(ctas) + 1, 2, dtype=torch.int32, device=device),
        # The attention CTAs' partials of a row, and per row the CTAs done.
        torch.empty(PARTIAL_BYTES, dtype=torch.uint8, device=device),
        torch.zeros(MAX_ROWS, dtype=torch.int32, device=device),
    )


class LowBatchAttention:
    """One sparse attention layer's low-batch path (module docstring)."""

    def __init__(self, layer, weight: torch.Tensor, weight_scale: torch.Tensor):
        self.logit_scale = (
            float(layer.impl.scale) * layer._q_scale_float * math.log2(math.e)
        )
        self.k_scale = layer._k_scale
        self.v_scale = layer._v_scale
        self.weight = weight
        self.weight_scale = weight_scale
        self.kernels = _kernels(weight.device.index)

    def accepts(self, num_reqs: int, query_len: int) -> bool:
        return (
            0 < query_len <= MAX_QUERY_LEN
            and 0 < num_reqs <= MAX_REQS
            and num_reqs * query_len <= MAX_ROWS
        )

    def run(
        self,
        step: AttentionStep,
        q: torch.Tensor,
        index_q: torch.Tensor,
        kv_cache: torch.Tensor,
        index_cache: torch.Tensor,
    ) -> torch.Tensor:
        """o_proj's BF16 output [rows, 6144] of the E4M3 queries ``q`` [rows,
        2048] and index queries ``index_q`` [rows, 128] (request-major) on the
        NVFP4 ``kv_cache`` [pages, 2, 128, 72] and ``index_cache`` [pages,
        128, 72], on the current stream."""
        from cutlass import Float32, Int32, Int64

        rows = step.seq_lens.shape[0] * step.query_len
        scores, act, act_scale, _, partial, counters = _scratch(q.device)
        stream = current_cu_stream()
        self.kernels.indexer(
            Int64(index_q.data_ptr()),
            Int64(index_cache.data_ptr()),
            Int64(index_cache.stride(0)),
            Int64(step.work.data_ptr()),
            Int64(scores.data_ptr()),
            Int32(step.query_len),
            stream,
        )
        self.kernels.attention(
            Int64(q.data_ptr()),
            Int64(kv_cache.data_ptr()),
            Int64(kv_cache.stride(0)),
            Int64(step.block_table.data_ptr()),
            Int32(step.block_table.stride(0)),
            Int64(step.seq_lens.data_ptr()),
            Int64(scores.data_ptr()),
            Int64(self.k_scale.data_ptr()),
            Int64(self.v_scale.data_ptr()),
            Int64(act.data_ptr()),
            Int64(act_scale.data_ptr()),
            Int64(partial.data_ptr()),
            Int64(counters.data_ptr()),
            Float32(self.logit_scale),
            Int32(rows),
            Int32(step.query_len),
            stream,
        )
        out = torch.empty(rows, HIDDEN, dtype=torch.bfloat16, device=q.device)
        o_proj_gemm(self.weight, self.weight_scale, act, act_scale, out, rows, stream)
        return out


def build_low_batch_attention(layer) -> LowBatchAttention | None:
    """``layer``'s (a ``MiniMaxM3SparseAttention``) low-batch attention, or
    None when it does not have the shapes and caches the kernels serve.
    Compiles the kernels once per process; call outside CUDA graph capture."""
    if not (
        current_platform.is_cuda() and current_platform.is_device_capability_family(100)
    ):
        return None
    if (
        layer.num_heads != HEADS
        or layer.num_kv_heads != 1
        or layer.num_idx_heads != 1
        or layer.head_dim != HEAD_DIM
        or layer.q_size != HEADS * HEAD_DIM
        or layer.kv_cache_dtype != "nvfp4"
        or layer.impl.topk_blocks != 16
        or layer._k_scale_float != 1.0
        or layer._v_scale_float != 1.0
    ):
        return None
    weights = mxfp8_weights(layer.o_proj, HIDDEN, HEADS * HEAD_DIM)
    if weights is None:
        return None
    return LowBatchAttention(layer, *weights)
