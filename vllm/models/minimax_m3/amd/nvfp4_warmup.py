# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Startup compile of the FlyDSL kernels behind the ROCm NVFP4 KV path.

Each launcher compiles a kernel variant on its first launch, picked from the
batch (token, request and row counts) and the context length. FULL decode
graphs fix their variants at capture, but prefill and mixed steps run these
launchers eagerly, so without this every new variant compiles inside a request
(about 3 s for the large prefill index scorer). The sweeps below drive each
launcher over the values its variant choice reads, bounded by the scheduler
limits, with the layer's own caches and throwaway outputs: inserts are given
slot -1 and every other kernel only reads the caches.
"""

from collections.abc import Callable, Iterable

import torch

from vllm.utils.math_utils import cdiv

from .indexer_msa import _score_width
from .ops.flydsl_utils import cu_count
from .ops.index_topk_decode import prefill_kernel as _topk_kernel
from .ops.index_topk_decode.host import (
    minimax_m3_index_topk_decode_indices,
    minimax_m3_index_topk_prefill_indices,
)
from .ops.nvfp4_index_decode import kernel as _index_decode_kernel
from .ops.nvfp4_index_decode.host import minimax_m3_nvfp4_index_score_decode
from .ops.nvfp4_index_prefill.host import (
    SMALL_WORK,
    minimax_m3_nvfp4_index_score_prefill,
)
from .ops.nvfp4_prefill.host import minimax_m3_nvfp4_sparse_prefill
from .ops.nvfp4_sparse_decode.host import minimax_m3_nvfp4_sparse_decode_fly

BLK = 128
TOPK = 16
# Above this many tokens the fused insert uses its 4-pairs-per-wave variant.
INSERT_SMALL_TOKENS = 512


def _pow2_upto(limit: int, start: int = 1) -> list[int]:
    out, v = [], start
    while v < limit:
        out.append(v)
        v *= 2
    return [*out, limit]


def _i32(*shape: int, fill: int = 0, device: torch.device) -> torch.Tensor:
    return torch.full(shape, fill, dtype=torch.int32, device=device)


def warmup_index(
    index_cache: torch.Tensor,
    num_index_heads: int,
    init_blocks: int,
    local_blocks: int,
    query_lens: Iterable[int],
    max_num_seqs: int,
    max_num_batched_tokens: int,
    max_model_len: int,
) -> None:
    """Index scorers and top-k, prefill and decode."""
    dev = index_cache.device
    h = num_index_heads

    def q_idx(rows: int) -> torch.Tensor:
        return torch.zeros(rows, h, 128, device=dev).to(torch.float8_e4m3fn)

    for ql in query_lens:
        if h * ql > _index_decode_kernel.MAX_ROWS:
            continue
        for reqs in _pow2_upto(max_num_seqs):
            for seq in _pow2_upto(max_model_len, BLK):
                minimax_m3_nvfp4_index_score_decode(
                    q_idx(reqs * ql),
                    index_cache,
                    torch.empty(h, reqs * ql, _score_width(seq, BLK), device=dev),
                    _i32(reqs, cdiv(seq, BLK), device=dev),
                    _i32(reqs, fill=seq, device=dev),
                    init_blocks=init_blocks,
                    local_blocks=local_blocks,
                    query_len=ql,
                    max_seq_len=seq,
                )
    minimax_m3_index_topk_decode_indices(
        torch.empty(h, 1, _score_width(BLK, BLK), device=dev),
        _i32(h, 1, TOPK, device=dev),
        _i32(1, fill=BLK, device=dev),
        max_seq_len=BLK,
        query_len=1,
    )

    # The scorer takes its shared kernel below SMALL_WORK block-tiles and the
    # large one above; the top-k picks one kernel per strip count.
    def prefill(q: int, seq: int) -> None:
        score = torch.empty(h, q, _score_width(seq, BLK), device=dev)
        minimax_m3_nvfp4_index_score_prefill(
            q_idx(q),
            index_cache,
            score,
            _i32(1, cdiv(seq, BLK), device=dev),
            torch.tensor([0, q], dtype=torch.int32, device=dev),
            _i32(1, fill=seq, device=dev),
            init_blocks=init_blocks,
            local_blocks=local_blocks,
            max_query_len=q,
            max_seq_len=seq,
        )
        minimax_m3_index_topk_prefill_indices(
            score,
            _i32(h, q, TOPK, device=dev),
            max_seq_len=seq,
            num_valid_pages=_i32(q, fill=cdiv(seq, BLK), device=dev),
        )

    prefill(BLK, BLK)
    q = min(max_num_batched_tokens, 4096)
    seq = max(q, cdiv(SMALL_WORK, cdiv(q, 256) * h) * BLK)
    if seq <= max_model_len:
        prefill(q, seq)
    lower = 0
    for strips in _topk_kernel.PREFILL_STRIPS:
        upper = strips * BLK * 64
        if lower < max_model_len:
            prefill(1, min(upper, max_model_len))
        lower = upper


def warmup_attention(
    kv_cache: torch.Tensor,
    num_heads: int,
    num_kv_heads: int,
    scale: float,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    query_lens: Iterable[int],
    max_num_seqs: int,
    max_num_batched_tokens: int,
) -> None:
    """Sparse decode (one variant per split count) and sparse prefill."""
    dev = kv_cache.device

    def q(rows: int) -> torch.Tensor:
        return torch.zeros(rows, num_heads, 128, dtype=torch.bfloat16, device=dev)

    def topk(rows: int) -> torch.Tensor:
        return _i32(num_kv_heads, rows, TOPK, fill=-1, device=dev)

    for ql in query_lens:
        for reqs in range(1, max_num_seqs + 1):
            rows = reqs * ql
            minimax_m3_nvfp4_sparse_decode_fly(
                q(rows),
                kv_cache,
                topk(rows),
                _i32(reqs, 1, device=dev),
                _i32(reqs, fill=1, device=dev),
                scale,
                q(rows),
                k_scale=k_scale,
                v_scale=v_scale,
                decode_query_len=ql,
            )
    # One combine round up to a workgroup per CU, two above.
    for tokens in (4, 4 * (cu_count(dev) // num_kv_heads + 1)):
        if tokens > max_num_batched_tokens:
            continue
        minimax_m3_nvfp4_sparse_prefill(
            q(tokens),
            kv_cache,
            topk(tokens),
            _i32(1, 1, device=dev),
            _i32(tokens, device=dev),
            _i32(tokens, device=dev),
            num_kv_heads,
            scale,
            q(tokens),
            k_scale=k_scale,
            v_scale=v_scale,
        )


def warmup_insert(
    norm_rope_insert: Callable[..., object],
    qkv_width: int,
    max_num_batched_tokens: int,
    device: torch.device,
) -> None:
    """The fused norm / RoPE / NVFP4 insert at both token-count variants."""
    for n in (1, INSERT_SMALL_TOKENS + 1):
        if n > max_num_batched_tokens:
            continue
        skip = torch.full((n,), -1, dtype=torch.int64, device=device)
        norm_rope_insert(
            torch.zeros(n, qkv_width, dtype=torch.bfloat16, device=device),
            torch.zeros(n, dtype=torch.int64, device=device),
            skip,
            skip,
        )
