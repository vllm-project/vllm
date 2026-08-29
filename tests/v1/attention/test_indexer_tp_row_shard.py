# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

import vllm.model_executor.layers.sparse_attn_indexer as sparse_indexer
import vllm.v1.attention.backends.mla.indexer as indexer
from vllm.config import CUDAGraphMode
from vllm.v1.attention.backends.mla.indexer import (
    DeepseekV32IndexerMetadata,
    DeepseekV32IndexerPrefillChunkMetadata,
)

LAYER = "model.layers.0.self_attn.indexer.k_cache"
TOPK, NUM_KV, DECODE_ROWS = 8, 64, 5


def _ref_topk(logits, ks, ke, out, rows, _s0, _s1, topk):
    cols = torch.arange(logits.shape[1]).unsqueeze(0)
    lo, hi = ks[:rows, None].long(), ke[:rows, None].long()
    scores = (
        logits[:rows].float().masked_fill((cols < lo) | (cols >= hi), -float("inf"))
    )
    k = min(topk, logits.shape[1])
    picked = scores.topk(k, dim=1).indices.int()
    keep = torch.arange(k).unsqueeze(0) < (hi - lo).clamp(0, k)
    out[:rows, :k] = torch.where(keep, picked, -1)
    out[:rows, k:] = -1


def _chunks(row_counts):
    chunks, start = [], DECODE_ROWS
    for i, rows in enumerate(row_counts):
        empty = i == len(row_counts) - 1
        ke = ((torch.arange(rows) * 7 + i * 3) % (NUM_KV + 1)).int()
        chunks.append(
            DeepseekV32IndexerPrefillChunkMetadata(
                block_table=torch.zeros(1, 1, dtype=torch.int32),
                cu_seqlen_ks=torch.zeros(rows, dtype=torch.int32),
                cu_seqlen_ke=torch.zeros_like(ke) if empty else ke,
                cu_seq_lens=torch.zeros(2, dtype=torch.int32),
                token_to_seq=torch.zeros(1, dtype=torch.int32),
                total_seq_lens=0 if empty else NUM_KV,
                token_start=start,
                token_end=start + rows,
                num_reqs=1,
                skip_kv_gather=i % 2 == 1,
                local_cu_seq_lens=torch.zeros(2, dtype=torch.int32),
                local_total_seq_lens=0 if empty else NUM_KV,
                max_local_total_seq_lens=NUM_KV,
            )
        )
        start += rows
    return chunks, start


def _bounds(chunks, tokens):
    ks = torch.zeros(tokens, dtype=torch.int32)
    ke = torch.zeros_like(ks)
    rows = slice(DECODE_ROWS, chunks[-1].token_end)
    ks[rows] = torch.cat([c.cu_seqlen_ks for c in chunks])
    ke[rows] = torch.cat([c.cu_seqlen_ke for c in chunks])
    return ks, ke


def _run_rank(monkeypatch, rank, chunks, tokens, logits, exchange, split=None):
    ks, ke = _bounds(chunks, tokens)
    gather_calls, pending_rows = [], []
    metadata = DeepseekV32IndexerMetadata(
        seq_lens=torch.empty(0, dtype=torch.int32),
        max_seq_len=2048,
        slot_mapping=torch.zeros(tokens, dtype=torch.long),
        num_decodes=0,
        num_decode_tokens=DECODE_ROWS,
        num_prefills=len(chunks),
        num_prefill_tokens=tokens - DECODE_ROWS,
        prefill=SimpleNamespace(chunks=chunks, row_shard_sizes=split),
    )
    set_ = monkeypatch.setattr
    set_(
        sparse_indexer,
        "get_forward_context",
        lambda: SimpleNamespace(
            attn_metadata={LAYER: metadata},
            cudagraph_runtime_mode=CUDAGraphMode.PIECEWISE,
        ),
    )
    set_(sparse_indexer.current_platform, "fp8_dtype", lambda: torch.float16)
    set_(sparse_indexer.current_platform, "is_xpu", lambda: False)
    set_(sparse_indexer, "get_tensor_model_parallel_rank", lambda: rank)
    set_(
        sparse_indexer,
        "get_tp_group",
        lambda: SimpleNamespace(all_gatherv=lambda t, dim, sizes: exchange(t, sizes)),
    )
    set_(
        sparse_indexer,
        "current_workspace_manager",
        lambda: SimpleNamespace(
            get_simultaneous=lambda *specs: tuple(
                torch.zeros(shape, dtype=dtype) for shape, dtype in specs
            )
        ),
    )
    set_(sparse_indexer.ops, "top_k_per_row_prefill", _ref_topk)
    set_(
        sparse_indexer.ops,
        "cp_gather_indexer_k_quant_cache",
        lambda *args: gather_calls.append(1),
    )

    def fake_logits(q, _k, weights, row_ks, row_ke, clean_logits=True):
        rows = q[0][:, 0, 0].long()
        torch.testing.assert_close(weights[:, 0].long(), rows)
        torch.testing.assert_close(row_ks, ks[rows])
        torch.testing.assert_close(row_ke, ke[rows])
        pending_rows.append(rows)
        return logits[rows]

    def fake_candidate_mask(_logits, _ks, _ke, candidates, _block_size):
        torch.testing.assert_close(candidates[:, 0].long(), pending_rows.pop(0))

    set_(sparse_indexer, "fp8_fp4_mqa_logits", fake_logits)
    set_(sparse_indexer, "_apply_candidate_mask", fake_candidate_mask)
    row_ids = torch.arange(tokens, dtype=torch.float32)
    out = torch.full((tokens, TOPK), 17, dtype=torch.int32)
    sparse_indexer.sparse_attn_indexer(
        torch.zeros(tokens, 1),
        LAYER,
        torch.empty(1),
        row_ids.reshape(tokens, 1, 1),
        None,
        None,
        row_ids.reshape(tokens, 1),
        128,
        "ue8m0",
        TOPK,
        4,
        4096,
        NUM_KV,
        out,
        True,
        False,
        "",
        candidate_blocks=torch.arange(tokens, dtype=torch.int32)[:, None],
        candidate_block_size=1,
    )
    assert not pending_rows
    return out, len(gather_calls)


def _run_group(monkeypatch, world, chunks, tokens, logits, split):
    saved: dict[int, torch.Tensor] = {}
    results: list[tuple[torch.Tensor, int]] = []
    for replay in (False, True):
        results = []
        for rank in range(world):

            def exchange(local, sizes, rank=rank, replay=replay):
                assert local.is_contiguous() and sizes == split
                assert local.shape[0] == sizes[rank]
                saved[rank] = local.clone()
                return (
                    torch.cat([saved[r] for r in range(world)])
                    if replay
                    else torch.zeros(sum(sizes), TOPK, dtype=torch.int32)
                )

            with monkeypatch.context() as m:
                results.append(
                    _run_rank(m, rank, chunks, tokens, logits, exchange, split)
                )
    return results


@pytest.mark.parametrize("world", [2, 3, 4, 8])
@pytest.mark.parametrize("ties", [False, True])
@pytest.mark.parametrize("uneven", [False, True])
def test_tp_row_shard_matches_reference(monkeypatch, world, ties, uneven):
    chunks, real_end = _chunks([3300, 2500, 3300, 7, 1100])
    tokens = real_end + 13
    logits = torch.randn(tokens, NUM_KV, generator=torch.Generator().manual_seed(53691))
    if ties:
        logits = (logits * 2).round() / 2
    baseline, baseline_gathers = _run_rank(
        monkeypatch,
        0,
        chunks,
        tokens,
        logits,
        lambda *_: (_ for _ in ()).throw(AssertionError("unexpected exchange")),
    )
    rows = real_end - DECODE_ROWS
    if uneven:
        split = [rows - (world - 1) * TOPK] + [TOPK] * (world - 1)
    else:
        base, rem = divmod(rows, world)
        split = [base + (rank < rem) for rank in range(world)]
    for out, gathers in _run_group(monkeypatch, world, chunks, tokens, logits, split):
        torch.testing.assert_close(out[:real_end], baseline[:real_end])
        assert torch.all(out[:DECODE_ROWS] == -1)
        assert torch.all(out[real_end:] == -1)
        assert gathers == baseline_gathers


def _costs(seq_lens, query_lens, ratio):
    return [
        (seq_len - query_len + 1 + row) // ratio
        for seq_len, query_len in zip(seq_lens, query_lens)
        for row in range(query_len)
    ]


@pytest.mark.parametrize("tp_size", [2, 3, 4, 8])
@pytest.mark.parametrize(
    "shape", ["fresh", "prefix", "ragged", "tail", "tight", "deep_first"]
)
def test_balanced_row_shard(tp_size, shape):
    floor = indexer.MIN_TP_SHARD_ROWS_PER_RANK * tp_size
    cases = {
        "fresh": ([4 * floor], [4 * floor]),
        "prefix": ([4 * floor], [4 * floor + 200_000]),
        "ragged": ([2 * floor, 7, floor], [2 * floor, 9007, floor + 60_000]),
        "tail": ([4 * floor], [64 * floor]),
        "tight": ([floor + 2], [floor + 2]),
        "deep_first": (
            [floor // 2, floor * 3 // 2],
            [500_000 + floor // 2, floor * 3 // 2],
        ),
    }
    query_lens, seq_lens = cases[shape]
    sizes = indexer.balanced_prefill_row_shard(
        torch.tensor(seq_lens), torch.tensor(query_lens), 4, tp_size
    )
    assert sizes is not None and len(sizes) == tp_size and min(sizes) > 0
    assert sum(sizes) == sum(query_lens)
    costs = _costs(seq_lens, query_lens, 4)

    def imbalance(parts):
        rank_costs, offset = [], 0
        for size in parts:
            rank_costs.append(sum(costs[offset : offset + size]))
            offset += size
        return max(rank_costs) / (sum(rank_costs) / len(rank_costs))

    base, rem = divmod(sum(query_lens), tp_size)
    equal = [base + (rank < rem) for rank in range(tp_size)]
    assert imbalance(sizes) <= imbalance(equal) + 1e-9
    assert imbalance(sizes) < 1.02


def test_balanced_row_shard_declines_below_floor():
    rows = indexer.MIN_TP_SHARD_ROWS_PER_RANK * 4 - 1
    args = (torch.tensor([rows]), torch.tensor([rows]), 4)
    assert indexer.balanced_prefill_row_shard(*args, 4) is None
    assert indexer.balanced_prefill_row_shard(*args, 1) is None


@pytest.mark.parametrize(
    "max_len,force,expected",
    [
        (2047, False, False),
        (2048, False, False),
        (2049, False, True),
        (2048, True, True),
    ],
)
def test_prefill_uses_mqa(max_len, force, expected):
    assert indexer._prefill_uses_mqa(max_len, 2048, force) is expected


@pytest.mark.parametrize(
    "tp,dcp,pcp,mode,env,expected",
    [
        (4, 1, False, CUDAGraphMode.PIECEWISE, None, True),
        (1, 1, False, CUDAGraphMode.PIECEWISE, None, False),
        (4, 2, False, CUDAGraphMode.PIECEWISE, None, False),
        (4, 1, True, CUDAGraphMode.PIECEWISE, None, False),
        (4, 1, False, CUDAGraphMode.NONE, None, True),
        (4, 1, False, CUDAGraphMode.FULL_DECODE_ONLY, None, True),
        (4, 1, False, CUDAGraphMode.FULL, None, False),
        (4, 1, False, CUDAGraphMode.FULL_AND_PIECEWISE, None, True),
        (4, 1, False, CUDAGraphMode.PIECEWISE, "VLLM_DISABLE_PYNCCL", False),
        (4, 1, False, CUDAGraphMode.PIECEWISE, "VLLM_USE_NCCL_SYMM_MEM", False),
        (4, 1, False, CUDAGraphMode.PIECEWISE, "VLLM_BATCH_INVARIANT", False),
    ],
)
def test_row_sharding_gate(monkeypatch, tp, dcp, pcp, mode, env, expected):
    monkeypatch.setattr(indexer.current_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(
        indexer.current_platform, "is_device_capability_family", lambda _: True
    )
    for name in (
        "VLLM_DISABLE_PYNCCL",
        "VLLM_USE_NCCL_SYMM_MEM",
        "VLLM_BATCH_INVARIANT",
    ):
        monkeypatch.setenv(name, str(int(name == env)))
    config = SimpleNamespace(compilation_config=SimpleNamespace(cudagraph_mode=mode))
    assert indexer.tp_prefill_row_sharding_supported(config, dcp, pcp, tp) is expected


def test_row_sharding_gate_rejects_unmeasured_arch(monkeypatch):
    monkeypatch.setattr(indexer.current_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(
        indexer.current_platform, "is_device_capability_family", lambda _: False
    )
    config = SimpleNamespace(
        compilation_config=SimpleNamespace(cudagraph_mode=CUDAGraphMode.PIECEWISE)
    )
    assert not indexer.tp_prefill_row_sharding_supported(config, 1, False, 4)
