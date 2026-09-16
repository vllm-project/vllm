# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Accuracy and perf: fuse_score_remap vs DeepGEMM decode indexer.

Requires SM90 CUDA. First fused launch JIT-compiles the kernel.

  pytest -s -v tests/v1/attention/test_fuse_score_remap_vs_deepgemm.py
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

from vllm.model_executor.kernels.attention.dsa.fuse_score_remap.indexer import (
    fuse_score_remap_available,
    fuse_score_remap_topk_indexer,
    pack_parts,
)
from vllm.utils.deep_gemm import (
    fp8_fp4_paged_mqa_logits,
    get_paged_mqa_logits_metadata,
)

NUM_HEADS = 32
HEAD_DIM = 128
PAGE_SIZE = 64
HEAD_DIM_WITH_SCALE = 132
FP8_E4M3_MAX = 448.0
RADIX_TOPK_WORKSPACE_SIZE = 1024 * 1024
MIN_SET_RECALL = 0.99

requires_sm90 = pytest.mark.skipif(
    not (
        torch.cuda.is_available()
        and torch.cuda.get_device_capability()[0] == 9
    ),
    reason="fuse_score_remap requires SM90",
)


@dataclass
class DecodeCase:
    q_fp8: torch.Tensor
    kv_cache: torch.Tensor
    weights: torch.Tensor
    seq_lens: torch.Tensor
    block_table: torch.Tensor
    topk: int
    max_model_len: int
    name: str


def _pack_fp8_kv_cache(k_fp32: torch.Tensor) -> torch.Tensor:
    num_pages, page_size, head_dim = k_fp32.shape
    amax = k_fp32.abs().amax(dim=-1).clamp(min=1e-6)
    scale = (amax / FP8_E4M3_MAX).to(torch.float32)
    k_q = (k_fp32 / scale.unsqueeze(-1)).to(torch.float8_e4m3fn)
    fp8_bytes = k_q.view(torch.uint8).reshape(num_pages, page_size * head_dim)
    scale_bytes = scale.contiguous().view(torch.uint8).reshape(num_pages, page_size * 4)
    packed = torch.cat([fp8_bytes, scale_bytes], dim=1)
    return packed.view(num_pages, page_size, 1, HEAD_DIM_WITH_SCALE)


def make_case(
    batch: int,
    seq: int,
    topk: int,
    *,
    ragged: bool = False,
    seed: int = 0,
) -> DecodeCase:
    torch.manual_seed(seed)
    device = torch.device("cuda")
    if ragged:
        seq_lens = torch.randint(
            max(1, seq // 4), seq + 1, (batch,), dtype=torch.int32, device=device
        )
        seq_lens[0] = seq
    else:
        seq_lens = torch.full((batch,), seq, dtype=torch.int32, device=device)
    max_seq = int(seq_lens.max().item())
    max_pages = max((max_seq + PAGE_SIZE - 1) // PAGE_SIZE, 1)
    max_model_len = max(max_pages * PAGE_SIZE, 256)
    q_fp8 = torch.randn(batch, NUM_HEADS, HEAD_DIM, device=device).to(torch.float8_e4m3fn)
    weights = torch.randn(batch, NUM_HEADS, device=device, dtype=torch.float32) * (
        NUM_HEADS**-0.5
    )
    pages_needed = 0
    for length in seq_lens.tolist():
        pages_needed += (int(length) + PAGE_SIZE - 1) // PAGE_SIZE if length > 0 else 0
    num_pages = max(pages_needed + batch, 1)
    kv_cache = _pack_fp8_kv_cache(
        torch.randn(num_pages, PAGE_SIZE, HEAD_DIM, device=device)
    )
    block_table = torch.full((batch, max_pages), -1, dtype=torch.int32, device=device)
    phys = torch.randperm(num_pages, device=device, dtype=torch.int32)
    cursor = 0
    for b, length in enumerate(seq_lens.tolist()):
        n = (int(length) + PAGE_SIZE - 1) // PAGE_SIZE if length > 0 else 0
        block_table[b, :n] = phys[cursor : cursor + n]
        cursor += n
    tag = f"B{batch}_N{seq}_K{topk}"
    if ragged:
        tag += "_ragged"
    return DecodeCase(
        q_fp8, kv_cache, weights, seq_lens, block_table, topk, max_model_len, tag
    )


def logical_to_physical(logical: torch.Tensor, block_table: torch.Tensor) -> torch.Tensor:
    valid = logical >= 0
    safe = logical.clamp(min=0)
    max_pages = max(int(block_table.shape[1]) - 1, 0)
    page = (safe // PAGE_SIZE).clamp(max=max_pages).to(torch.int64)
    offs = safe % PAGE_SIZE
    phys = torch.gather(block_table, 1, page) * PAGE_SIZE + offs
    return torch.where(valid, phys, logical)


def _schedule(case: DecodeCase) -> torch.Tensor:
    batch = int(case.q_fp8.size(0))
    seq2d = case.seq_lens.reshape(batch, 1).contiguous()
    num_sms = torch.cuda.get_device_properties(case.q_fp8.device).multi_processor_count
    return get_paged_mqa_logits_metadata(seq2d, PAGE_SIZE, num_sms)


def deepgemm_physical(case: DecodeCase) -> torch.Tensor:
    batch = int(case.q_fp8.size(0))
    q = case.q_fp8.reshape(batch, 1, NUM_HEADS, HEAD_DIM).contiguous()
    w = case.weights.reshape(batch, NUM_HEADS).contiguous()
    seq2d = case.seq_lens.reshape(batch, 1).contiguous()
    kv = case.kv_cache.view(torch.uint8)
    if kv.ndim == 3:
        kv = kv.unsqueeze(-2)
    logits = fp8_fp4_paged_mqa_logits(
        (q, None),
        kv.contiguous(),
        w,
        seq2d,
        case.block_table.contiguous(),
        _schedule(case),
        max_model_len=case.max_model_len,
        clean_logits=False,
    )
    logical = torch.full((batch, case.topk), -1, dtype=torch.int32, device=q.device)
    workspace = torch.empty(RADIX_TOPK_WORKSPACE_SIZE, dtype=torch.uint8, device=q.device)
    torch.ops._C.cooperative_topk(
        logits, seq2d, logical, workspace, case.topk, case.max_model_len
    )
    return logical_to_physical(logical, case.block_table)


def fused_physical(case: DecodeCase) -> torch.Tensor:
    batch = int(case.q_fp8.size(0))
    q = case.q_fp8.reshape(batch, 1, NUM_HEADS, HEAD_DIM).contiguous()
    kv = case.kv_cache.view(torch.uint8)
    if kv.ndim == 3:
        kv = kv.unsqueeze(-2)
    table = case.block_table.contiguous()
    out = torch.empty((batch, case.topk), dtype=torch.int32, device=q.device)
    max_parts = pack_parts(table)
    pack_scores = torch.empty(
        (batch, max_parts, 256), dtype=torch.float32, device=q.device
    )
    pack_indices = torch.empty(
        (batch, max_parts, 256), dtype=torch.int32, device=q.device
    )
    logical = torch.empty((batch, case.topk), dtype=torch.int32, device=q.device)
    workspace = torch.empty(RADIX_TOPK_WORKSPACE_SIZE, dtype=torch.uint8, device=q.device)
    fuse_score_remap_topk_indexer(
        q,
        kv.contiguous(),
        case.weights.contiguous(),
        case.seq_lens.contiguous(),
        table,
        _schedule(case),
        out,
        case.topk,
        pack_scores=pack_scores,
        pack_indices=pack_indices,
        logical=logical,
        topk_workspace=workspace,
        max_seq_len=case.max_model_len,
    )
    return out


def valid_sets(indices: torch.Tensor) -> list[set[int]]:
    return [{int(v) for v in row if v >= 0} for row in indices.detach().cpu().tolist()]


def set_recall(pred: list[set[int]], ref: list[set[int]]) -> float:
    hit = sum(len(p & r) for p, r in zip(pred, ref, strict=True))
    total = sum(len(r) for r in ref)
    return hit / max(total, 1)


def _median_ms(fn, warmup: int, iters: int) -> float:
    torch.cuda.synchronize()
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times: list[float] = []
    for _ in range(iters):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    times.sort()
    return times[len(times) // 2]


ACCURACY_CASES = [
    dict(batch=4, seq=512, topk=512, ragged=False),
    dict(batch=4, seq=1024, topk=512, ragged=True),
    dict(batch=2, seq=4096, topk=2048, ragged=False),
    dict(batch=4, seq=4096, topk=2048, ragged=True),
    dict(batch=1, seq=8192, topk=2048, ragged=False),
    dict(batch=1, seq=65536, topk=2048, ragged=False),
]

BENCH_CASES = [
    dict(batch=1, seq=4096, topk=2048),
    dict(batch=8, seq=4096, topk=2048),
    dict(batch=8, seq=8192, topk=2048),
    dict(batch=32, seq=8192, topk=2048),
    dict(batch=1, seq=65536, topk=2048),
]


@requires_sm90
def test_fuse_score_remap_available():
    ok, msg = fuse_score_remap_available()
    assert ok, msg


@requires_sm90
@pytest.mark.parametrize(
    "spec",
    ACCURACY_CASES,
    ids=lambda s: f"B{s['batch']}_N{s['seq']}_K{s['topk']}"
    + ("_ragged" if s.get("ragged") else ""),
)
def test_fuse_score_remap_matches_deepgemm_topk_set(spec):
    ok, msg = fuse_score_remap_available()
    if not ok:
        pytest.skip(msg)
    case = make_case(**spec)
    orig = deepgemm_physical(case)
    fused = fused_physical(case)
    recall = set_recall(valid_sets(fused), valid_sets(orig))
    assert recall >= MIN_SET_RECALL, f"{case.name} opt/orig recall={recall:.4f}"


@requires_sm90
def test_fuse_score_remap_bench_vs_deepgemm():
    ok, msg = fuse_score_remap_available()
    if not ok:
        pytest.skip(msg)
    print(
        f"\ndevice={torch.cuda.get_device_name(0)}\n"
        f"{'case':<24} {'deepgemm_ms':>12} {'fused_ms':>10} {'speedup':>8}"
    )
    for spec in BENCH_CASES:
        case = make_case(**spec, ragged=False)
        fused_physical(case)
        deepgemm_physical(case)

        def orig(c=case):
            deepgemm_physical(c)

        def fused(c=case):
            fused_physical(c)

        orig_ms = _median_ms(orig, warmup=3, iters=10)
        fused_ms = _median_ms(fused, warmup=3, iters=10)
        speedup = orig_ms / fused_ms if fused_ms > 0 else 0.0
        print(f"{case.name:<24} {orig_ms:12.3f} {fused_ms:10.3f} {speedup:7.2f}x")
