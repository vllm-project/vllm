# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TP query-row sharding of the DSA indexer prefill.

The indexer projections are replicated across TP, so every rank recomputes the
same prefill logits and the same top-k. Rows are independent (one block per row
over that row's ``[ks, ke)``), so each rank can own a disjoint slice and the
group exchanges ``index_topk`` int32s per row instead of the logits.
"""

import importlib
import os
from types import SimpleNamespace

import pytest
import torch

import vllm.model_executor.layers.sparse_attn_indexer as sparse_indexer
import vllm.utils.deep_gemm as deep_gemm
import vllm.v1.attention.backends.mla.indexer as indexer
from vllm.config import CUDAGraphMode
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.attention.backends.mla.indexer import (
    DeepseekV32IndexerMetadata,
    DeepseekV32IndexerPrefillChunkMetadata,
    balanced_prefill_row_shard,
)
from vllm.v1.attention.backends.utils import split_decodes_and_prefills

INDEXER_LAYER = "model.layers.0.self_attn.indexer.k_cache"


def _direct_topk_worker(rank, port):
    import torch.distributed as dist

    from vllm.model_executor.layers import tp_topk_publication as publication

    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    torch.accelerator.set_device_index(rank)
    dist.init_process_group("gloo", rank=rank, world_size=4)
    # Two disjoint TP groups model TP2 x PCP2. Group creation is collective.
    groups = [
        dist.new_group(ranks=list(range(start, start + 2)), backend="nccl")
        for start in (0, 2)
    ]
    group = SimpleNamespace(
        device_group=groups[rank // 2], rank_in_group=rank % 2, world_size=2
    )
    publication.get_tp_group = lambda: group
    os.environ["VLLM_TP_TOPK_DIRECT"] = "1"
    buffer = publication.allocate_topk_buffer(
        43, 2176, dtype=torch.int32, device=torch.device("cuda", rank)
    )
    publisher = publication.get_topk_publication(buffer)
    assert publisher is not None
    sizes = [11, 19]
    start = 5 + sum(sizes[: rank % 2])
    stop = start + sizes[rank % 2]
    for epoch in range(8):
        buffer.fill_(-1)
        publisher.begin()
        values = torch.arange(30 * 2051, device=buffer.device, dtype=torch.int32)
        values = values.reshape(30, 2051) + (rank // 2) * 1_000_000 + epoch * 100_000
        buffer[start:stop, :2051].copy_(values[start - 5 : stop - 5])
        publisher.publish(buffer, start, stop - start, 2051)
        torch.testing.assert_close(buffer[5:35, :2051], values)
        assert torch.all(buffer[:5] == -1)
        assert torch.all(buffer[35:] == -1)
        assert torch.all(buffer[:, 2051:] == -1)
    from vllm.models.glm5next.nvidia.ops.kpool_compress import (
        expand_pools_and_append_tail,
    )

    for epoch in range(3):
        buffer.fill_(-1)
        publisher.begin()
        pools = torch.arange(30 * 512, device=buffer.device, dtype=torch.int32)
        pools = (pools.reshape(30, 512) + epoch + (rank // 2) * 100) % 1000
        pools[:, -3:] = -1
        seq_lens = torch.arange(30, device=buffer.device, dtype=torch.int32) + 4096
        expanded = expand_pools_and_append_tail(
            pools[start - 5 : stop - 5],
            seq_lens[start - 5 : stop - 5],
            4,
            out=buffer[start:stop, :2051],
            peer_ptrs=publisher.peers,
            peer_rank=publisher.rank,
            peer_row_start=start,
        )
        assert expanded.data_ptr() == buffer[start:stop].data_ptr()
        publisher.finish()
        offsets = torch.arange(4, device=buffer.device)
        history = torch.where(pools[..., None] >= 0, pools[..., None] * 4 + offsets, -1)
        tail_offsets = torch.arange(3, device=buffer.device)
        tail = torch.where(
            tail_offsets < (seq_lens % 4)[:, None],
            (seq_lens // 4)[:, None] * 4 + tail_offsets,
            -1,
        )
        expected = torch.cat((history.flatten(1), tail), dim=1).int()
        torch.testing.assert_close(buffer[5:35, :2051], expected)
        assert torch.all(buffer[:, 2051:] == -1)
    # Capture fixed final addresses, then change inputs across repeated replays.
    # This validates component graph safety, not the currently disabled sparse
    # MLA PCP engine graph route.
    static_rows = torch.empty_like(buffer[start:stop, :2051])
    static_pools = pools[start - 5 : stop - 5].clone()
    static_seq = seq_lens[start - 5 : stop - 5].clone()
    for fused in (False, True):
        torch.accelerator.synchronize()
        dist.barrier()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            buffer.fill_(-1)
            publisher.begin()
            if fused:
                expand_pools_and_append_tail(
                    static_pools,
                    static_seq,
                    4,
                    out=buffer[start:stop, :2051],
                    peer_ptrs=publisher.peers,
                    peer_rank=publisher.rank,
                    peer_row_start=start,
                )
                publisher.finish()
            else:
                buffer[start:stop, :2051].copy_(static_rows)
                publisher.publish(buffer, start, stop - start, 2051)
            consumed = buffer.clone()
        for epoch in range(8):
            if fused:
                static_pools.copy_(pools[start - 5 : stop - 5] + epoch)
                shifted = pools + epoch
                history = torch.where(
                    shifted[..., None] >= 0, shifted[..., None] * 4 + offsets, -1
                )
                expected = torch.cat((history.flatten(1), tail), dim=1).int()
            else:
                expected = (
                    torch.arange(
                        30 * 2051, device=buffer.device, dtype=torch.int32
                    ).reshape(30, 2051)
                    + (rank // 2) * 1_000_000
                    + epoch * 100_000
                )
                static_rows.copy_(expected[start - 5 : stop - 5])
            graph.replay()
            torch.testing.assert_close(consumed[5:35, :2051], expected)
            assert torch.all(consumed[:5] == -1)
            assert torch.all(consumed[35:] == -1)
            assert torch.all(consumed[:, 2051:] == -1)
        del graph, consumed
    torch.accelerator.synchronize()
    dist.barrier()
    del publisher, buffer
    dist.destroy_process_group()


@pytest.mark.skipif(torch.accelerator.device_count() < 4, reason="requires four GPUs")
def test_direct_topk_publication_isolates_pcp_lanes():
    """Uneven TP rows preserve tails/padding and never mix two PCP lanes."""
    import torch.multiprocessing as mp

    from vllm.utils.network_utils import get_open_port

    mp.spawn(_direct_topk_worker, args=(get_open_port(),), nprocs=4, join=True)


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
    monkeypatch,
    *,
    world,
    rank,
    chunks,
    num_tokens,
    logits,
    exchange,
    split=None,
    padded_rows=0,
    index_kpool=1,
    metadata_includes_padding=False,
    pcp_size=1,
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
        num_prefill_tokens=(
            num_tokens
            - _DECODE_ROWS
            + (padded_rows if metadata_includes_padding else 0)
        ),
        prefill=SimpleNamespace(
            chunks=chunks, row_shard_sizes=split, max_prefill_seq_len=2048
        ),
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
    set_(sparse_indexer, "get_pcp_group", lambda: SimpleNamespace(world_size=pcp_size))
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

    row_ids = torch.arange(num_tokens + padded_rows, dtype=torch.float32)
    width = _TOPK + index_kpool - 1
    buffer = torch.full((num_tokens + padded_rows, width), 17, dtype=torch.int32)
    if index_kpool > 1:
        # Initialize the model package first, as in model loading, to avoid
        # its existing circular import through the k-pool layer.
        from vllm.models.glm5next.nvidia.ops import kpool_compress

        kpool_indexer = importlib.import_module(
            "vllm.model_executor.layers.sparse_attn_indexer_kpool"
        )

        for name in (
            "get_forward_context",
            "get_tensor_model_parallel_rank",
            "get_tp_group",
            "current_workspace_manager",
        ):
            set_(kpool_indexer, name, getattr(sparse_indexer, name))
        set_(kpool_indexer.current_platform, "is_rocm", lambda: False)
        set_(deep_gemm, "fp8_fp4_mqa_logits", fake_mqa_logits)
        set_(torch.ops._C, "top_k_per_row_prefill", _ref_top_k_per_row_prefill)

        def expand(pool_ids, seq_lens, pool_size, out=None, **kwargs):
            offsets = torch.arange(pool_size)
            tokens = pool_ids[..., None] * pool_size + offsets
            tokens = torch.where(pool_ids[..., None] >= 0, tokens, -1)
            tail_offsets = torch.arange(pool_size - 1)
            tail = (seq_lens // pool_size)[:, None] * pool_size + tail_offsets
            tail = torch.where(tail_offsets < (seq_lens % pool_size)[:, None], tail, -1)
            result = torch.cat((tokens.flatten(1), tail), dim=1).int()
            if out is not None:
                out.copy_(result)
                return out
            return result

        set_(kpool_compress, "expand_pools_and_append_tail", expand)
        positions = torch.zeros(num_tokens + padded_rows, dtype=torch.int64)
        positions[:num_tokens] = (
            ke_table * index_kpool + (torch.arange(num_tokens) % index_kpool) - 1
        )
        kpool_indexer.sparse_attn_indexer_kpool(
            torch.zeros(num_tokens + padded_rows, 1),
            INDEXER_LAYER,
            torch.empty(1),
            row_ids.reshape(-1, 1, 1),
            None,
            torch.zeros(num_tokens + padded_rows, 4),
            row_ids.reshape(-1, 1),
            128,
            "ue8m0",
            _TOPK,
            4,
            4096,
            _NUM_KV,
            buffer,
            True,
            index_kpool=index_kpool,
            positions=positions,
        )
        return buffer, len(gathers)
    sparse_indexer.sparse_attn_indexer(
        torch.zeros(num_tokens + padded_rows, 1),  # hidden_states
        INDEXER_LAYER,
        torch.empty(1),  # kv_cache
        row_ids.reshape(-1, 1, 1),  # q_quant carries its global row id
        None,  # q_scale
        None,  # k
        row_ids.reshape(-1, 1),  # weights carry it too
        128,
        "ue8m0",
        _TOPK,
        4,
        4096,
        _NUM_KV,
        buffer,
        True,  # skip_k_cache_insert
        pcp_size > 1,
        "",  # dense_mha_metadata_layer_name
    )
    return buffer, len(gathers)


def _run_group(
    monkeypatch,
    world,
    chunks,
    num_tokens,
    logits,
    split=None,
    exchange_observations=None,
    **kwargs,
):
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
                if exchange_observations is not None:
                    exchange_observations.append(
                        (rank, replay, tuple(local.shape), tuple(sizes))
                    )
                slices[rank] = local.clone()
                if not replay:
                    return torch.zeros(sum(sizes), local.shape[1], dtype=torch.int32)
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
                        **kwargs,
                    )
                )
    return results


def test_kpool_sharded_prefill_exchanges_expanded_tail_and_excludes_padding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The k-pool path gathers expanded token indices, including the tail.

    This directly exercises ``sparse_attn_indexer_kpool`` with the same
    row-shard metadata used by V2.  ``num_prefill_tokens`` intentionally counts
    graph padding, while ``row_shard_sizes`` counts only real rows; the gather
    must use the latter and must exchange ``topk + kpool - 1`` columns.
    """
    chunks, num_tokens = _build_chunks([19, 11, 13, 3])
    logits = torch.randn(
        num_tokens, _NUM_KV, generator=torch.Generator().manual_seed(54952)
    )
    observations: list[tuple[int, bool, tuple[int, ...], tuple[int, ...]]] = []
    outputs = _run_group(
        monkeypatch,
        4,
        chunks,
        num_tokens,
        logits,
        split=[7, 17, 9, 13],
        exchange_observations=observations,
        index_kpool=4,
        padded_rows=13,
        metadata_includes_padding=True,
    )

    # Two passes are used by the CPU collective replay: one to collect each
    # rank's shard and one to replay the all-gather for every rank.
    assert len(observations) == 8
    assert {entry[2] for entry in observations} == {
        (7, _TOPK + 4 - 1),
        (17, _TOPK + 4 - 1),
        (9, _TOPK + 4 - 1),
        (13, _TOPK + 4 - 1),
    }
    assert all(entry[3] == (7, 17, 9, 13) for entry in observations)
    for output, _ in outputs:
        # The metadata advertises 13 extra rows, but prefill_end must stop at
        # the real shard total. Padding remains untouched by the collective.
        assert torch.all(output[num_tokens:] == -1)


@pytest.mark.parametrize("index_kpool", [1, 4])
@pytest.mark.parametrize("metadata_includes_padding", [False, True])
@pytest.mark.parametrize("pcp_size", [1, 2])
def test_sharded_prefill_preserves_padding_and_kpool_tail(
    monkeypatch, index_kpool, metadata_includes_padding, pcp_size
):
    """Gather only scored rows, including incomplete pools and excluding padding.

    GPU kernels use row-independent CPU references here; the real forward and
    collective slicing run unchanged, including the GLM k-pool implementation.
    """
    chunks, num_tokens = _build_chunks([19, 11, 13, 3])
    logits = torch.randn(
        num_tokens, _NUM_KV, generator=torch.Generator().manual_seed(54951)
    )
    options = dict(
        padded_rows=13,
        index_kpool=index_kpool,
        metadata_includes_padding=metadata_includes_padding,
        pcp_size=pcp_size,
    )

    def no_exchange(*args, **kwargs):
        pytest.fail("Unsharded reference must not exchange rows")

    baseline, gathers = _run_rank(
        monkeypatch,
        world=1,
        rank=0,
        chunks=chunks,
        num_tokens=num_tokens,
        logits=logits,
        exchange=no_exchange,
        **options,
    )
    for output, shard_gathers in _run_group(
        monkeypatch, 4, chunks, num_tokens, logits, split=[7, 17, 9, 13], **options
    ):
        torch.testing.assert_close(output, baseline)
        assert shard_gathers == gathers
        assert torch.all(output[num_tokens:] == -1)
        assert torch.all(output[:_DECODE_ROWS] == -1)
        if index_kpool > 1:
            # All four phases occur, so a gather of only topk columns fails.
            expected_tail_count = torch.arange(_DECODE_ROWS, num_tokens) % index_kpool
            torch.testing.assert_close(
                (output[_DECODE_ROWS:num_tokens, _TOPK:] >= 0).sum(1),
                expected_tail_count,
            )


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
        # Just above the total-size gate, where the minimum per-rank floor is
        # large enough to amortize the all-gather.
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


@pytest.mark.parametrize("context_len", [128_000, 256_000, 512_000, 1_000_000])
def test_long_context_tp4_shard_is_exact_and_cost_balanced(
    context_len: int,
) -> None:
    """Exercise the production TP4 planner at every reported A/B length.

    The treatment must be an exact partition/permutation of the baseline row
    order, while balancing the causal compressed-key work that dominates the
    long-prefill indexer.
    """
    compress_ratio = 4
    sizes = indexer.balanced_prefill_row_shard(
        torch.tensor([context_len], dtype=torch.int32),
        torch.tensor([context_len], dtype=torch.int32),
        compress_ratio,
        4,
    )
    assert sizes is not None
    assert len(sizes) == 4
    assert sum(sizes) == context_len
    assert min(sizes) > 0

    # Reassembling the per-rank slices must preserve every baseline row in the
    # same order. This is the semantic contract of the layout-only gather.
    baseline_rows = torch.arange(context_len, dtype=torch.int32)
    treatment_rows = torch.cat(baseline_rows.split(sizes))
    torch.testing.assert_close(treatment_rows, baseline_rows)

    per_row_cost = torch.arange(1, context_len + 1, dtype=torch.int64) // 4
    cumulative = torch.cat(
        [torch.zeros(1, dtype=torch.int64), torch.cumsum(per_row_cost, dim=0)]
    )
    boundaries = torch.tensor([0, *torch.cumsum(torch.tensor(sizes), 0).tolist()])
    rank_costs = cumulative[boundaries[1:]] - cumulative[boundaries[:-1]]
    imbalance = float(rank_costs.max()) / float(rank_costs.double().mean())
    assert imbalance < 1.001


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
        ({"use_pcp": True}, {}, True),
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


@pytest.mark.parametrize("rows", [8192, 65535, 65536, 65539])
@pytest.mark.parametrize("compress_ratio", [1, 4])
@pytest.mark.parametrize("mode", [CUDAGraphMode.NONE, CUDAGraphMode.PIECEWISE])
@pytest.mark.parametrize("decode_rows", [0, 1])
def test_runner_v2_builds_shards_from_scheduled_rows(
    monkeypatch, rows, compress_ratio, mode, decode_rows
):
    """V2 eager/piecewise metadata excludes padding and gates on query rows.

    Exercise the real V2 prepare_attn -> shared builder -> planner path on CPU;
    replace only the device metadata kernels. A 1M history alone must not
    activate sharding when the scheduler gives the batch fewer than 64K rows.
    """
    from vllm.v1.worker.gpu.model_states.default import DefaultModelState

    builder = object.__new__(indexer.DeepseekV32IndexerMetadataBuilder)
    builder.vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=SimpleNamespace(index_topk=2048)),
        attention_config=SimpleNamespace(sparse_mla_force_mqa=False),
        speculative_config=None,
    )
    for name, value in dict(
        decode_threshold=1,
        use_flattening=False,
        supports_varlen=False,
        num_speculative_tokens=0,
        decode_lens_buffer=torch.empty(4, dtype=torch.int32),
        per_req_decode_lens_buffer=torch.empty(4, dtype=torch.int32),
        expanded_seq_lens_buffer=torch.empty(4, dtype=torch.int32),
        scheduler_metadata_buffer=torch.empty(1, dtype=torch.int32),
        use_pcp=False,
        compress_ratio=compress_ratio,
        kernel_block_size=None,
        pcp_world_size=1,
        dcp_world_size=1,
        dcp_rank=0,
        cp_kv_cache_interleave_size=1,
        max_prefill_buffer_size=2_000_000,
        enable_tp_prefill_row_sharding=True,
        kv_cache_spec=SimpleNamespace(num_states=64),
        compressed_slot_mapping_buffer=torch.empty(rows + 13, dtype=torch.int64),
    ).items():
        setattr(builder, name, value)
    monkeypatch.setattr(indexer, "get_tensor_model_parallel_world_size", lambda: 4)
    monkeypatch.setattr(indexer, "has_deep_gemm", lambda: False)
    monkeypatch.setattr(
        indexer,
        "get_compressed_slot_mapping",
        lambda num_tokens, *args, out: out[:num_tokens],
    )

    def chunk(start, end, query_start, query_start_cpu, *args, query_slice, **kwargs):
        token_start = int(query_start_cpu[start]) + query_slice.start
        return SimpleNamespace(
            token_start=token_start,
            token_end=token_start + query_slice.stop - query_slice.start,
        )

    monkeypatch.setattr(indexer, "build_prefill_chunk_metadata", chunk)
    query_lens = [1] * decode_rows + [rows - 7, 7] + [0] * (2 - decode_rows)
    query_start = torch.tensor([0, *query_lens], dtype=torch.int32).cumsum(0).int()
    seq_lens = [4096] * decode_rows + [800_000, 1_000_000]
    lengths = torch.tensor(seq_lens + [0] * (2 - decode_rows), dtype=torch.int32)
    num_tokens = rows + decode_rows
    batch = SimpleNamespace(
        num_reqs=2 + decode_rows,
        num_reqs_after_padding=4,
        num_tokens=num_tokens,
        num_tokens_after_padding=num_tokens + 13,
        query_start_loc_np=query_start.numpy(),
        query_start_loc=query_start,
        max_query_len=rows - 7,
        seq_lens=lengths,
        seq_lens_cpu_upper_bound=lengths,
        dcp_local_seq_lens=None,
        positions=torch.zeros(num_tokens + 13, dtype=torch.int64),
        is_prefilling_np=torch.tensor(
            [False] * decode_rows + [True, True] + [False] * (2 - decode_rows)
        ).numpy(),
        prompt_lens=None,
    )
    group = SimpleNamespace(
        layer_names=[INDEXER_LAYER], get_metadata_builder=lambda _: builder
    )
    result = DefaultModelState.prepare_attn(
        SimpleNamespace(supports_mm_inputs=False),
        batch,
        mode,
        (torch.zeros(4, 8, dtype=torch.int32),),
        torch.zeros(1, num_tokens + 13, dtype=torch.int64),
        [[group]],
        SimpleNamespace(kv_cache_groups=[SimpleNamespace(layer_names=[INDEXER_LAYER])]),
    )[INDEXER_LAYER]
    assert result.num_prefill_tokens == rows
    assert result.num_prefills == 2
    assert result.num_decode_tokens == decode_rows
    assert result.seq_lens.tolist() == seq_lens
    if decode_rows:
        assert result.decode.seq_lens.tolist() == [[4096 // compress_ratio]]
    assert result.prefill.chunks[0].token_start == decode_rows
    assert sum(c.token_end - c.token_start for c in result.prefill.chunks) == rows
    sizes = result.prefill.row_shard_sizes
    if rows < 65536:
        assert sizes is None
    else:
        assert len(sizes) == 4 and min(sizes) > 0
        assert sum(sizes) == rows


def test_spec_decode_rows_are_excluded_from_prefill_row_sharding():
    """MTP-5 rows stay in decode metadata when a long prefill is present.

    DeepSeek's indexer uses ``1 + num_speculative_tokens`` as the decode
    threshold.  With MTP-5, the two six-token requests below are speculative
    decode rows and only the final long request may be assigned to TP row
    shards.  This also checks the token offset consumed by the runtime
    indexer, which is the boundary that matters for mixed SpecDecode batches.
    """
    num_speculative_tokens = 5
    decode_threshold = 1 + num_speculative_tokens
    query_lens = torch.tensor(
        [decode_threshold, decode_threshold, 65536], dtype=torch.int32
    )
    query_start_loc = torch.cat(
        (torch.zeros(1, dtype=torch.int32), query_lens.cumsum(0))
    )
    common = CommonAttentionMetadata(
        query_start_loc=query_start_loc,
        query_start_loc_cpu=query_start_loc,
        seq_lens=torch.tensor([128, 256, 1_000_000], dtype=torch.int32),
        seq_lens_cpu_upper_bound=torch.tensor([128, 256, 1_000_000], dtype=torch.int32),
        num_reqs=3,
        num_actual_tokens=int(query_start_loc[-1]),
        max_query_len=65536,
        max_seq_len=1_000_000,
        block_table_tensor=torch.zeros(3, 1, dtype=torch.int32),
        slot_mapping=torch.zeros(int(query_start_loc[-1]), dtype=torch.int64),
        is_prefilling=torch.tensor([False, False, True]),
    )

    num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (
        split_decodes_and_prefills(common, decode_threshold=decode_threshold)
    )
    assert (num_decodes, num_prefills) == (2, 1)
    assert (num_decode_tokens, num_prefill_tokens) == (12, 65536)

    shard_sizes = balanced_prefill_row_shard(
        common.seq_lens_cpu_upper_bound[num_decodes:],
        torch.diff(common.query_start_loc_cpu[num_decodes:]),
        compress_ratio=1,
        tp_size=4,
    )
    assert shard_sizes is not None
    assert sum(shard_sizes) == num_prefill_tokens
    assert min(shard_sizes) > 0
