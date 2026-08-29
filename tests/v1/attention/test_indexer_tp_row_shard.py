# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TP query-row sharding of the DSA indexer prefill.

The indexer projections are replicated across TP, so every rank recomputes the
same prefill logits and the same top-k. Rows are independent (one block per row
over that row's ``[ks, ke)``), so each rank can own a disjoint slice and the
group exchanges ``index_topk`` int32s per row instead of the logits.
"""

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

INDEXER_LAYER = "model.layers.0.self_attn.indexer.k_cache"

_TOPK = 8
_NUM_KV = 64
# Leading decode rows of the shared buffer. The prefill shard must address the
# window [num_decode_tokens, num_decode_tokens + num_prefill_tokens) and leave
# these alone; mixed decode+prefill batches are the concurrency > 1 case.
_DECODE_ROWS = 5


def _ref_top_k_per_row_prefill(logits, cu_ks, cu_ke, out, num_rows, _s0, _s1, topk):
    """Row-independent stand-in for ``ops.top_k_per_row_prefill``.

    Mirrors the real kernel's contract (one block per row, reading only that
    row's ``[ks, ke)``), which is the property the row shard rests on.
    """
    positions = torch.arange(logits.shape[1]).unsqueeze(0)
    lo = cu_ks[:num_rows].long().unsqueeze(1)
    hi = cu_ke[:num_rows].long().unsqueeze(1)
    scores = (
        logits[:num_rows]
        .float()
        .masked_fill((positions < lo) | (positions >= hi), -float("inf"))
    )
    k = min(topk, logits.shape[1])
    picked = scores.topk(k, dim=1).indices.int()
    keep = torch.arange(k).unsqueeze(0) < (hi - lo).clamp(min=0, max=k)
    out[:num_rows, :k] = torch.where(keep, picked, torch.full_like(picked, -1))
    out[:num_rows, k:] = -1


def _build_chunks(row_counts):
    """Ragged chunks: uneven row counts, per-row causal bounds that leave some
    rows short of topk, a continuation chunk that must reuse the gathered K,
    and a trailing empty-KV chunk."""
    chunks, token_start = [], _DECODE_ROWS
    for idx, num_rows in enumerate(row_counts):
        empty_kv = idx == len(row_counts) - 1
        ke = ((torch.arange(num_rows) * 7 + idx * 3) % (_NUM_KV + 1)).int()
        chunks.append(
            DeepseekV32IndexerPrefillChunkMetadata(
                block_table=torch.zeros(1, 1, dtype=torch.int32),
                cu_seqlen_ks=torch.zeros(num_rows, dtype=torch.int32),
                cu_seqlen_ke=torch.zeros_like(ke) if empty_kv else ke,
                cu_seq_lens=torch.zeros(2, dtype=torch.int32),
                token_to_seq=torch.zeros(1, dtype=torch.int32),
                total_seq_lens=0 if empty_kv else _NUM_KV,
                token_start=token_start,
                token_end=token_start + num_rows,
                num_reqs=1,
                skip_kv_gather=idx % 2 == 1,
                local_cu_seq_lens=torch.zeros(2, dtype=torch.int32),
                local_total_seq_lens=0 if empty_kv else _NUM_KV,
                max_local_total_seq_lens=_NUM_KV,
            )
        )
        token_start += num_rows
    return chunks, token_start


def _bound_tables(chunks, num_tokens):
    """Per-global-row causal bounds, zero for the leading decode rows."""
    ks = torch.zeros(num_tokens, dtype=torch.int32)
    ke = torch.zeros(num_tokens, dtype=torch.int32)
    ks[_DECODE_ROWS:] = torch.cat([c.cu_seqlen_ks for c in chunks])
    ke[_DECODE_ROWS:] = torch.cat([c.cu_seqlen_ke for c in chunks])
    return ks, ke


def _run_rank(
    monkeypatch, *, world, rank, chunks, num_tokens, logits, exchange, split=None
):
    """Drive the real ``sparse_attn_indexer`` prefill path for one TP rank.

    ``exchange(local_rows, sizes)`` stands in for the group's all_gatherv.
    Returns ``(topk_buffer, gather_call_count)``.
    """
    ks_table, ke_table = _bound_tables(chunks, num_tokens)
    gathers = []

    metadata = DeepseekV32IndexerMetadata(
        seq_lens=torch.empty(0, dtype=torch.int32),
        max_seq_len=2048,
        slot_mapping=torch.zeros(num_tokens, dtype=torch.long),
        num_decodes=0,
        # num_decodes=0 keeps the decode path out of this test while still
        # placing the prefill rows behind a decode offset in the buffer.
        num_decode_tokens=_DECODE_ROWS,
        num_prefills=len(chunks),
        num_prefill_tokens=num_tokens - _DECODE_ROWS,
        prefill=SimpleNamespace(chunks=chunks, row_shard_sizes=split),
    )

    set_ = monkeypatch.setattr
    set_(
        sparse_indexer,
        "get_forward_context",
        lambda: SimpleNamespace(
            attn_metadata={INDEXER_LAYER: metadata},
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
    set_(sparse_indexer.ops, "top_k_per_row_prefill", _ref_top_k_per_row_prefill)
    set_(
        sparse_indexer.ops,
        "cp_gather_indexer_k_quant_cache",
        lambda *args: gathers.append(1),
    )

    def fake_mqa_logits(q, _k, weights, cu_ks, cu_ke, clean_logits=True):
        rows = q[0][:, 0, 0].long()
        # The q slice, the weight slice and the causal bounds must all name the
        # same global rows; an off-by-one anywhere in the shard breaks this.
        torch.testing.assert_close(weights[:, 0].long(), rows)
        torch.testing.assert_close(cu_ks, ks_table[rows])
        torch.testing.assert_close(cu_ke, ke_table[rows])
        return logits[rows]

    set_(sparse_indexer, "fp8_fp4_mqa_logits", fake_mqa_logits)

    row_ids = torch.arange(num_tokens, dtype=torch.float32)
    buffer = torch.full((num_tokens, _TOPK), 17, dtype=torch.int32)
    sparse_indexer.sparse_attn_indexer(
        torch.zeros(num_tokens, 1),  # hidden_states
        INDEXER_LAYER,
        torch.empty(1),  # kv_cache
        row_ids.reshape(num_tokens, 1, 1),  # q_quant carries its global row id
        None,  # q_scale
        None,  # k
        row_ids.reshape(num_tokens, 1),  # weights carry it too
        128,
        "ue8m0",
        _TOPK,
        4,
        4096,
        _NUM_KV,
        buffer,
        True,  # skip_k_cache_insert
        False,  # use_pcp
        "",  # dense_mha_metadata_layer_name
    )
    return buffer, len(gathers)


def _run_group(monkeypatch, world, chunks, num_tokens, logits, split=None):
    """Collect every rank's slice, then replay the concatenation each rank
    would receive. There is exactly one exchange per forward."""
    rows = num_tokens - _DECODE_ROWS
    split = split or [rows // world + int(rank < rows % world) for rank in range(world)]
    slices: dict[int, torch.Tensor] = {}
    results: list[tuple[torch.Tensor, int]] = []
    for replay in (False, True):
        results = []
        for rank in range(world):

            def exchange(local, sizes, rank=rank, replay=replay):
                assert local.is_contiguous()
                assert sizes == split
                assert local.shape[0] == sizes[rank]
                slices[rank] = local.clone()
                if not replay:
                    return torch.zeros(sum(sizes), _TOPK, dtype=torch.int32)
                return torch.cat([slices[r] for r in range(world)])

            with monkeypatch.context() as m:
                results.append(
                    _run_rank(
                        m,
                        world=world,
                        rank=rank,
                        chunks=chunks,
                        num_tokens=num_tokens,
                        logits=logits,
                        exchange=exchange,
                        split=split,
                    )
                )
    return results


@pytest.mark.parametrize("world", [2, 3, 8])
@pytest.mark.parametrize("ties", [False, True])
@pytest.mark.parametrize("uneven", [False, True])
def test_tp_row_shard_prefill_matches_row_independent_reference(
    monkeypatch: pytest.MonkeyPatch, world: int, ties: bool, uneven: bool
) -> None:
    """Every TP rank receives the reference result for every completed row.

    Covers non-power-of-two tp_size, ragged chunks that straddle the shard
    boundary, rows with fewer than topk candidates, an empty-KV chunk, and a
    dense-tie logits table (ties are where a row-order-dependent merge would
    diverge), and both the equal-row and a lopsided cost-balanced split.
    """
    # Two chunks deliberately have the same local query range length. The
    # gather decision is positional metadata, not a set keyed by row bounds.
    chunks, num_tokens = _build_chunks([3300, 2500, 3300, 7, 1100])
    logits = torch.randn(
        num_tokens, _NUM_KV, generator=torch.Generator().manual_seed(1003)
    )
    if ties:
        logits = (logits * 2).round() / 2

    def no_exchange(local, sizes):
        raise AssertionError("tp_size == 1 must not exchange")

    baseline, baseline_gathers = _run_rank(
        monkeypatch,
        world=1,
        rank=0,
        chunks=chunks,
        num_tokens=num_tokens,
        logits=logits,
        exchange=no_exchange,
    )
    expected_gathers = sum(
        chunk.total_seq_lens > 0 and not chunk.skip_kv_gather for chunk in chunks
    )
    assert baseline_gathers == expected_gathers
    # No row is silently dropped: each holds exactly min(ke - ks, topk) valid
    # slots, padded with -1.
    ks_table, ke_table = _bound_tables(chunks, num_tokens)
    torch.testing.assert_close(
        (baseline >= 0).sum(dim=1).int(),
        (ke_table - ks_table).clamp(min=0, max=_TOPK).int(),
    )
    assert torch.all(baseline[:_DECODE_ROWS] == -1)

    rows = num_tokens - _DECODE_ROWS
    split = None
    if uneven:
        # a lopsided but valid partition, as cost balancing produces
        split = [rows - (world - 1) * _TOPK] + [_TOPK] * (world - 1)
    for rank, (buffer, gathers) in enumerate(
        _run_group(monkeypatch, world, chunks, num_tokens, logits, split)
    ):
        torch.testing.assert_close(buffer, baseline, msg=f"rank {rank} diverged")
        assert torch.all(buffer[:_DECODE_ROWS] == -1), "exchange clobbered decode rows"
        # The K gather is a workspace side effect that later chunks reuse via
        # `skip_kv_gather`; narrowing the query rows must not change it.
        assert gathers == baseline_gathers


def _scored_keys(seq_lens, query_lens, compress_ratio):
    """Per-row ke - ks, spelled out the way the Triton metadata kernel does."""
    out = []
    for seq_len, query_len in zip(seq_lens, query_lens):
        context = seq_len - query_len
        out += [(context + 1 + j) // compress_ratio for j in range(query_len)]
    return out


def _rank_costs(sizes, per_row):
    costs, off = [], 0
    for size in sizes:
        costs.append(sum(per_row[off : off + size]))
        off += size
    return costs


@pytest.mark.parametrize("tp_size", [2, 3, 4, 8])
@pytest.mark.parametrize(
    "shape",
    [
        pytest.param("fresh", id="fresh_prompt"),
        pytest.param("prefix", id="prefix_context"),
        pytest.param("ragged", id="mixed_ragged"),
        pytest.param("chunk_tail", id="chunked_prefill_tail"),
        pytest.param("tight", id="just_above_floor"),
        pytest.param("deep_first", id="deep_context_first"),
    ],
)
def test_balanced_row_shard_equalises_scored_keys(tp_size: int, shape: str) -> None:
    """The split beats equal rows on cost balance, or declines to split."""
    rows_needed = indexer.MIN_TP_SHARD_ROWS_PER_RANK * tp_size
    per_req = 4 * rows_needed
    if shape == "fresh":  # no context: the full causal ramp
        query_lens, seq_lens = [per_req], [per_req]
    elif shape == "prefix":  # long shared prefix already in cache
        query_lens, seq_lens = [per_req], [per_req + 200_000]
    elif shape == "ragged":  # uneven requests at different depths
        query_lens = [per_req // 2, 7, per_req, per_req // 3 + 5]
        seq_lens = [q + c for q, c in zip(query_lens, [0, 9_000, 500, 60_000])]
    elif shape == "deep_first":
        # An expensive deep-context request ahead of a cheap fresh one, so the
        # cost profile falls and the early ranks own fewer rows.
        query_lens = [rows_needed // 2, 2 * rows_needed - rows_needed // 2]
        seq_lens = [query_lens[0] + 500_000, query_lens[1]]
    elif shape == "tight":
        # Just above the total-size gate, where sub-1024 inexpensive shards are
        # still beneficial.
        query_lens, seq_lens = [rows_needed + 2], [rows_needed + 2]
    else:  # a mid-prompt chunk: high, nearly flat cost
        query_lens, seq_lens = [per_req], [per_req + 15 * per_req]

    compress_ratio = 4
    sizes = indexer.balanced_prefill_row_shard(
        torch.tensor(seq_lens, dtype=torch.int32),
        torch.tensor(query_lens, dtype=torch.int32),
        compress_ratio,
        tp_size,
    )
    assert sizes is not None
    num_rows = sum(query_lens)
    assert len(sizes) == tp_size
    assert sum(sizes) == num_rows, "the split must cover every row exactly once"
    assert min(sizes) >= 1

    per_row = _scored_keys(seq_lens, query_lens, compress_ratio)
    assert per_row == [
        int(x) for x in _reference_ke_minus_ks(seq_lens, query_lens, compress_ratio)
    ]
    base, rem = divmod(num_rows, tp_size)
    equal_sizes = [base + int(r < rem) for r in range(tp_size)]

    def imbalance(split):
        costs = _rank_costs(split, per_row)
        return max(costs) / (sum(costs) / len(costs))

    assert imbalance(sizes) <= imbalance(equal_sizes) + 1e-9
    assert imbalance(sizes) < 1.02


def _reference_ke_minus_ks(seq_lens, query_lens, compress_ratio):
    """Independent restatement of the kernel formula, vectorised."""
    out = []
    for seq_len, query_len in zip(seq_lens, query_lens):
        pos = torch.arange(query_len) + (seq_len - query_len) + 1
        out += (pos // compress_ratio).tolist()
    return out


def test_balanced_row_shard_declines_below_the_floor() -> None:
    """Too few rows to give every rank the floor -> no split at all."""
    tp_size = 4
    rows = indexer.MIN_TP_SHARD_ROWS_PER_RANK * tp_size - 1
    assert (
        indexer.balanced_prefill_row_shard(
            torch.tensor([rows], dtype=torch.int32),
            torch.tensor([rows], dtype=torch.int32),
            4,
            tp_size,
        )
        is None
    )
    assert (
        indexer.balanced_prefill_row_shard(
            torch.tensor([rows + 1], dtype=torch.int32),
            torch.tensor([rows + 1], dtype=torch.int32),
            4,
            1,
        )
        is None
    )


def _sharding_config(cudagraph_mode=CUDAGraphMode.PIECEWISE):
    return SimpleNamespace(
        compilation_config=SimpleNamespace(cudagraph_mode=cudagraph_mode)
    )


@pytest.mark.parametrize(
    "kwargs,env,expected",
    [
        ({}, {}, True),
        ({"tp_size": 1}, {}, False),
        ({"dcp_world_size": 2}, {}, False),
        ({"use_pcp": True}, {}, False),
        ({}, {"VLLM_DISABLE_PYNCCL": True}, False),
        ({}, {"VLLM_USE_NCCL_SYMM_MEM": True}, False),
        ({}, {"VLLM_BATCH_INVARIANT": True}, False),
    ],
)
def test_row_sharding_gate_rejects_unsupported_configurations(
    monkeypatch: pytest.MonkeyPatch, kwargs: dict, env: dict, expected: bool
) -> None:
    """The gate is the whole safety envelope; nothing else guards the exchange."""
    monkeypatch.setattr(indexer.current_platform, "is_cuda", lambda: True)
    for name, value in env.items():
        monkeypatch.setattr(indexer.envs, name, value)
    args = {"dcp_world_size": 1, "use_pcp": False, "tp_size": 4, **kwargs}
    supported = indexer.tp_prefill_row_sharding_supported(_sharding_config(), **args)
    assert supported is expected


@pytest.mark.parametrize(
    "cudagraph_mode,expected",
    [
        (None, True),
        (CUDAGraphMode.NONE, True),
        (CUDAGraphMode.PIECEWISE, True),
        # Mixed batches run under PIECEWISE, so prefill is never captured whole.
        (CUDAGraphMode.FULL_AND_PIECEWISE, True),
        (CUDAGraphMode.FULL_DECODE_ONLY, True),
        # Mixed batches would be captured whole; the exchange must stay out.
        (CUDAGraphMode.FULL, False),
    ],
)
def test_row_sharding_gate_follows_the_mixed_batch_cudagraph_mode(
    monkeypatch: pytest.MonkeyPatch, cudagraph_mode, expected: bool
) -> None:
    monkeypatch.setattr(indexer.current_platform, "is_cuda", lambda: True)
    supported = indexer.tp_prefill_row_sharding_supported(
        _sharding_config(cudagraph_mode),
        dcp_world_size=1,
        use_pcp=False,
        tp_size=4,
    )
    assert supported is expected
