# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""After the ROCm NVFP4 warmup, launches anywhere inside the scheduler limits
compile no new FlyDSL kernel variant."""

import random

import pytest
import torch

from vllm.platforms import current_platform

if not current_platform.is_rocm():
    pytest.skip("ROCm (gfx950) FlyDSL kernels", allow_module_level=True)

from vllm.platforms.rocm import on_gfx950  # noqa: E402

if not on_gfx950():
    pytest.skip("gfx950 instructions", allow_module_level=True)

from vllm.models.minimax_m3.amd import nvfp4_warmup as W  # noqa: E402
from vllm.models.minimax_m3.amd.ops.index_topk_decode import (  # noqa: E402
    host as topk_host,
)
from vllm.models.minimax_m3.amd.ops.norm_rope_nvfp4_insert import (  # noqa: E402
    host as insert_host,
)
from vllm.models.minimax_m3.amd.ops.nvfp4_index_decode import (  # noqa: E402
    host as index_decode_host,
)
from vllm.models.minimax_m3.amd.ops.nvfp4_index_prefill import (  # noqa: E402
    host as index_prefill_host,
)
from vllm.models.minimax_m3.amd.ops.nvfp4_prefill import (  # noqa: E402
    host as prefill_host,
)
from vllm.models.minimax_m3.amd.ops.nvfp4_sparse_decode import (  # noqa: E402
    host as decode_host,
)
from vllm.utils.math_utils import cdiv  # noqa: E402

BLK, D, ROT, EPS = 128, 128, 64, 1e-6
NQ, NKV = 16, 1  # one TP4 rank
MAX_SEQS, MAX_TOKENS, MAX_LEN = 64, 8192, 65536
QUERY_LENS = range(1, 5)  # DSpark with 3 speculative tokens
PAGES = 8

KERNEL_CACHES = (
    insert_host._get_kernel,
    prefill_host._get_kernel,
    decode_host._get_kernel,
    index_prefill_host._get_kernel,
    index_prefill_host._get_kernel_shared,
    index_decode_host._get_kernel,
    topk_host._get_prefill_kernel,
    topk_host._get_decode_kernel,
)


def _variants() -> list[int]:
    return [c.cache_info().currsize for c in KERNEL_CACHES]


def _i32(*shape, fill=0):
    return torch.full(shape, fill, dtype=torch.int32, device="cuda")


@pytest.mark.parametrize("niq", [1, 4])
def test_warmup_covers_every_variant(niq):
    torch.manual_seed(0)
    rng = random.Random(0)
    kv = torch.randint(0, 256, (PAGES, 2 * NKV, BLK, 72), device="cuda")
    kv = kv.to(torch.uint8)
    idx = torch.randint(0, 256, (PAGES, BLK, 72), device="cuda").to(torch.uint8)
    ks = torch.ones((), device="cuda")
    w = torch.ones(D, dtype=torch.bfloat16, device="cuda")
    cos_sin = torch.zeros(4096, ROT, dtype=torch.bfloat16, device="cuda")

    def insert(qkv, pos, slots, islots):
        n = qkv.shape[0]
        insert_host.minimax_m3_norm_rope_nvfp4_insert(
            qkv, w, w, w, w, cos_sin, pos, slots, islots, kv, idx, ks, ks,
            torch.empty(n, NQ * D, dtype=torch.bfloat16, device="cuda"),
            torch.empty(n, niq * D, dtype=torch.float8_e4m3fn, device="cuda"),
            num_heads=NQ, num_kv_heads=NKV, num_index_heads=niq, eps=EPS,
        )  # fmt: skip

    W.warmup_insert(insert, (NQ + 2 * NKV + niq + 1) * D, MAX_TOKENS, kv.device)
    W.warmup_attention(kv, NQ, NKV, D**-0.5, ks, ks, QUERY_LENS, MAX_SEQS, MAX_TOKENS)
    W.warmup_index(idx, niq, 0, 1, QUERY_LENS, MAX_SEQS, MAX_TOKENS, MAX_LEN)
    torch.accelerator.synchronize()
    warmed = _variants()

    for _ in range(40):
        ql = rng.choice(QUERY_LENS)
        reqs = rng.randint(1, MAX_SEQS)
        seq = rng.randint(1, MAX_LEN)
        tokens = rng.randint(1, MAX_TOKENS)
        rows = reqs * ql
        n = rng.randint(1, MAX_TOKENS)
        skip = torch.full((n,), -1, dtype=torch.int64, device="cuda")
        insert(
            torch.zeros(n, (NQ + 2 * NKV + niq + 1) * D, dtype=torch.bfloat16,
                        device="cuda"),
            torch.zeros(n, dtype=torch.int64, device="cuda"), skip, skip,
        )  # fmt: skip
        q = torch.zeros(rows, NQ, D, dtype=torch.bfloat16, device="cuda")
        decode_host.minimax_m3_nvfp4_sparse_decode_fly(
            q, kv, _i32(NKV, rows, 16, fill=-1), _i32(reqs, 1),
            _i32(reqs, fill=1), D**-0.5, torch.empty_like(q),
            decode_query_len=ql,
        )  # fmt: skip
        qp = torch.zeros(tokens, NQ, D, dtype=torch.bfloat16, device="cuda")
        prefill_host.minimax_m3_nvfp4_sparse_prefill(
            qp, kv, _i32(NKV, tokens, 16, fill=-1), _i32(1, 1), _i32(tokens),
            _i32(tokens), NKV, D**-0.5, torch.empty_like(qp),
        )  # fmt: skip
        if niq * ql <= 32:
            score = torch.empty(niq, rows, W._score_width(seq, BLK), device="cuda")
            index_decode_host.minimax_m3_nvfp4_index_score_decode(
                torch.zeros(rows, niq, D, device="cuda").to(torch.float8_e4m3fn),
                idx, score, _i32(reqs, cdiv(seq, BLK)), _i32(reqs, fill=seq),
                init_blocks=0, local_blocks=1, query_len=ql, max_seq_len=seq,
            )  # fmt: skip
        # A chunked prefill: `tokens` new rows at the end of a `ctx` context.
        ctx = max(tokens, seq)
        score = torch.empty(niq, tokens, W._score_width(ctx, BLK), device="cuda")
        index_prefill_host.minimax_m3_nvfp4_index_score_prefill(
            torch.zeros(tokens, niq, D, device="cuda").to(torch.float8_e4m3fn),
            idx, score, _i32(1, cdiv(ctx, BLK)),
            torch.tensor([0, tokens], dtype=torch.int32, device="cuda"),
            _i32(1, fill=ctx), init_blocks=0, local_blocks=1,
            max_query_len=tokens, max_seq_len=ctx,
        )  # fmt: skip
        topk_host.minimax_m3_index_topk_prefill_indices(
            score, _i32(niq, tokens, 16), max_seq_len=ctx,
            num_valid_pages=_i32(tokens, fill=cdiv(ctx, BLK)),
        )  # fmt: skip
        assert _variants() == warmed, (ql, reqs, seq, tokens, n)
