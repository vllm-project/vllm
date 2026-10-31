# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Correctness of the sparse-MLA CP indexer.

``rocm_aiter_sparse_attn_indexer`` normally has every TP rank score every
prefill query row and reach the same Top-K. When ``use_cp_indexer`` says so,
the rows are dealt across the ranks as mirrored stripes -- rank ``r`` takes
stripe ``r`` of the first half and stripe ``tp - 1 - r`` of the second -- and
the batch is rebuilt with an ``all_reduce(MAX)`` over a ``-1``-filled Top-K
buffer.

Run `pytest tests/distributed/test_rocm_aiter_indexer_m_split.py`.
"""

import dataclasses
import queue
from itertools import accumulate

import pytest
import torch
import torch.multiprocessing as mp

from vllm._aiter_ops import is_aiter_found_and_supported, rocm_aiter_ops
from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
from vllm.distributed import cleanup_dist_env_and_memory
from vllm.distributed.parallel_state import (
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm.forward_context import set_forward_context
from vllm.platforms import current_platform
from vllm.utils.network_utils import get_open_port
from vllm.utils.system_utils import update_environment_variables
from vllm.v1.attention.backends.mla.indexer import (
    DeepseekV32IndexerMetadata,
    DeepseekV32IndexerMetadataBuilder,
    DeepseekV32IndexerPrefillChunkMetadata,
    DeepseekV32IndexerPrefillMetadata,
)
from vllm.v1.attention.ops import rocm_aiter_mla_sparse as mla_sparse
from vllm.v1.worker.workspace import init_workspace_manager

LAYER = "model.layers.0.self_attn.indexer"
# Indexer geometry from the models that have one. index_head_dim is 128 and
# index_topk is 2048 for both families; index_n_heads is 32 on GLM-5.2/5.3 and
# 64 on DeepSeek V4, and it does not enter the row split, so take the smaller.
HEAD_DIM = 128
NUM_HEADS = 32
TOPK_TOKENS = 2048
# (num_tokens, context, num_chunks). The context must exceed TOPK_TOKENS by a
# good margin, or every row keeps every key and the Top-K stops discriminating.
SHAPES = (
    # A split chunk and a tail with fewer rows than ranks: the tail is split
    # too, and every row must still have one writer.
    (17, 8192, (16, 1)),
    # A ragged split plus rows whose window is narrower than TOPK_TOKENS, so
    # -1 padding survives the merge.
    (9001, 8192, 1),
)


def _build_inputs(num_tokens, context, num_chunks, device):
    """Deterministic inputs, identical on every rank.

    TP ranks see the same batch, so seeding identically is the real situation
    rather than a simplification.
    """
    torch.manual_seed(1234)
    fp8_dtype = current_platform.fp8_dtype()

    # k_cache is [num_blocks, block_size, head_dim + 4]: block_size * head_dim
    # fp8 values followed by block_size fp32 scales. block_size 1 keeps the
    # gather on its NORMAL layout. Both regions must hold well-formed values --
    # random bytes reinterpreted as fp32 are ~20% NaN/Inf, which makes the
    # logits garbage and the Top-K unstable between runs.
    kv_cache = torch.empty((context, 1, HEAD_DIM + 4), dtype=torch.uint8, device=device)
    flat = kv_cache.view(context, -1)
    flat[:, :HEAD_DIM] = (
        (torch.randn(context, HEAD_DIM, device=device) * 0.3)
        .to(fp8_dtype)
        .view(torch.uint8)
    )
    flat[:, HEAD_DIM:].view(torch.float32).fill_(1.0)

    # Offset the windows so the last row reaches the end of the context.
    base = max(0, context - num_tokens)
    if isinstance(num_chunks, tuple):
        assert sum(num_chunks) == num_tokens
        bounds = [0, *accumulate(num_chunks)]
    else:
        bounds = [round(i * num_tokens / num_chunks) for i in range(num_chunks + 1)]
    chunks = [
        DeepseekV32IndexerPrefillChunkMetadata(
            block_table=torch.arange(
                context, dtype=torch.int32, device=device
            ).unsqueeze(0),
            cu_seqlen_ks=torch.zeros(hi - lo, dtype=torch.int32, device=device),
            # Causal windows ending at the context, as a chunk of a long
            # prefill would: row i scores keys [0, base + i]. Widths still
            # vary, so the narrow rows keep fewer than TOPK_TOKENS entries and
            # stay -1-padded through the MAX merge.
            cu_seqlen_ke=(
                base + torch.arange(lo + 1, hi + 1, dtype=torch.int32, device=device)
            ).clamp(1, context),
            cu_seq_lens=torch.tensor([0, context], dtype=torch.int32, device=device),
            token_to_seq=torch.zeros(context, dtype=torch.int32, device=device),
            total_seq_lens=context,
            max_seqlen_k=context,
            token_start=lo,
            token_end=hi,
            num_reqs=1,
        )
        for lo, hi in zip(bounds[:-1], bounds[1:])
    ]

    metadata = DeepseekV32IndexerMetadata(
        seq_lens=torch.tensor([context], dtype=torch.int32, device=device),
        max_seq_len=context,
        slot_mapping=torch.arange(num_tokens, dtype=torch.int32, device=device),
        num_decodes=0,
        num_decode_tokens=0,
        num_prefills=1,
        num_prefill_tokens=num_tokens,
        prefill=DeepseekV32IndexerPrefillMetadata(
            chunks=chunks, max_prefill_seq_len=context
        ),
    )

    q_fp8 = (torch.randn(num_tokens, NUM_HEADS, HEAD_DIM, device=device) * 0.3).to(
        fp8_dtype
    )
    weights = torch.randn(num_tokens, NUM_HEADS, dtype=torch.float32, device=device)
    return kv_cache, metadata, q_fp8, weights


def _reference_logits(inputs, device):
    """Recompute the op's internal logits, plus each row's valid window.

    The op discards its logits, but tie-breaking can only be judged against
    them. This mirrors the replicated branch: gather K per chunk, then score
    the chunk's whole row range in one launch.
    """
    kv_cache, metadata, q_fp8, weights = inputs
    fp8_dtype = current_platform.fp8_dtype()
    num_tokens = q_fp8.shape[0]
    context = metadata.max_seq_len

    logits = torch.empty(num_tokens, context, dtype=torch.float32, device=device)
    row_starts = torch.empty(num_tokens, dtype=torch.int32, device=device)
    row_ends = torch.empty(num_tokens, dtype=torch.int32, device=device)

    for chunk in metadata.prefill.chunks:
        lo, hi = chunk.token_start, chunk.token_end
        k_fp8 = torch.empty(context, HEAD_DIM, dtype=fp8_dtype, device=device)
        k_scale = torch.empty(context, 4, dtype=torch.uint8, device=device)
        mla_sparse.cp_gather_indexer_k_quant_cache_triton(
            kv_cache,
            k_fp8,
            k_scale,
            chunk.block_table,
            chunk.cu_seq_lens,
            token_to_seq=chunk.token_to_seq,
        )
        logits[lo:hi] = mla_sparse.rocm_fp8_mqa_logits(
            q_fp8[lo:hi],
            (k_fp8, k_scale.view(torch.float32)),
            weights[lo:hi],
            chunk.cu_seqlen_ks,
            chunk.cu_seqlen_ke,
        )
        row_starts[lo:hi] = chunk.cu_seqlen_ks
        row_ends[lo:hi] = chunk.cu_seqlen_ke
    return logits, row_starts, row_ends


def _assert_topk_equivalent(cp_topk, non_cp_topk, logits, row_starts, row_ends):
    """Assert the two runs selected equally good keys for every row.

    The operator's contract is the selected *set*, not its position in the
    buffer: equal scores may be broken either way, and because indices are
    stored sorted, one differing pick shifts the rest of the row. This mirrors
    ``compare_top_k_results`` in tests/kernels/test_top_k_per_row.py -- sets
    first, then the values behind any disagreement.
    """
    expected_count = (row_ends - row_starts).clamp(max=cp_topk.shape[1])
    for selected in (cp_topk, non_cp_topk):
        valid = selected >= 0
        assert torch.equal(valid.sum(dim=1), expected_count), "missing or extra keys"
        assert torch.all(
            ~valid
            | ((selected >= row_starts[:, None]) & (selected < row_ends[:, None]))
        ), "key outside the causal window"
        ordered = selected.sort(dim=1).values
        assert not (
            (ordered[:, 1:] == ordered[:, :-1]) & (ordered[:, 1:] >= 0)
        ).any(), "duplicate selected keys"
        padding = torch.arange(selected.shape[1], device=selected.device)[None, :]
        assert torch.all(selected[padding >= expected_count[:, None]] == -1)

    # Sorting first makes the scan order-insensitive, so only rows whose
    # selected sets really differ reach the per-row inspection below.
    differing = (
        (cp_topk.sort(dim=1).values != non_cp_topk.sort(dim=1).values)
        .any(dim=1)
        .nonzero()
        .flatten()
        .tolist()
    )
    failures = []

    for row in differing:
        start, end = int(row_starts[row]), int(row_ends[row])
        num_valid = min(cp_topk.shape[1], end - start)
        picked_cp = cp_topk[row, :num_valid].tolist()
        picked_non_cp = non_cp_topk[row, :num_valid].tolist()
        set_cp, set_non_cp = set(picked_cp), set(picked_non_cp)
        if set_cp == set_non_cp:
            continue

        row_logits = logits[row]
        only_cp = sorted(set_cp - set_non_cp)
        only_non_cp = sorted(set_non_cp - set_cp)
        vals_cp = row_logits[
            torch.tensor(only_cp, dtype=torch.long, device=logits.device)
        ]
        vals_non_cp = row_logits[
            torch.tensor(only_non_cp, dtype=torch.long, device=logits.device)
        ]
        if len(only_cp) != len(only_non_cp) or not torch.allclose(
            vals_cp.sort().values, vals_non_cp.sort().values, rtol=1e-4, atol=1e-4
        ):
            failures.append(
                f"row {row}: CP-only {only_cp[:5]} score "
                f"{vals_cp[:5].tolist()}, non-CP-only {only_non_cp[:5]} score "
                f"{vals_non_cp[:5].tolist()}"
            )
            continue

        # Nothing left out may beat something selected.
        unpicked = sorted(set(range(start, end)) - set_cp)
        if unpicked:
            worst_kept = row_logits[torch.tensor(picked_cp, device=logits.device)].min()
            best_dropped = row_logits[
                torch.tensor(unpicked, device=logits.device)
            ].max()
            if worst_kept < best_dropped - 1e-4:
                failures.append(
                    f"row {row}: dropped a better key "
                    f"({best_dropped.item()} > {worst_kept.item()})"
                )

    assert not failures, (
        f"{len(failures)} of {len(differing)} differing rows are real "
        f"mismatches:\n" + "\n".join(failures[:5])
    )


def _cp_indexer_threshold() -> int | None:
    """This GPU's CP-indexer threshold, or None where the CP indexer is off."""
    return mla_sparse._CP_INDEXER_MIN_SEQ_LEN.get(mla_sparse._GCN_ARCH.split(":")[0])


def _run_indexer(m, enabled, inputs, config, device):
    """One forward of the real op, with the threshold just met or just missed."""
    kv_cache, metadata, q_fp8, weights = inputs
    num_tokens = q_fp8.shape[0]
    context = metadata.max_seq_len

    # use_cp_indexer() also requires AITER MLA, so it has to be on for both
    # runs. The op reads max_prefill_seq_len only to decide whether to split,
    # so setting it just below or at this GPU's threshold switches the CP
    # indexer off or on for the same inputs.
    m.setenv("VLLM_ROCM_USE_AITER", "1")
    m.setenv("VLLM_ROCM_USE_AITER_MLA", "1")
    rocm_aiter_ops.refresh_env_variables()
    threshold = _cp_indexer_threshold()
    assert threshold is not None
    metadata = dataclasses.replace(
        metadata,
        prefill=dataclasses.replace(
            metadata.prefill,
            max_prefill_seq_len=threshold if enabled else threshold - 1,
        ),
    )
    # Without this the feature silently stays off and every comparison passes
    # by doing the same thing twice.
    assert (
        mla_sparse.use_cp_indexer(
            metadata.prefill.max_prefill_seq_len,
            config.parallel_config.tensor_parallel_size,
        )
        is enabled
    )

    topk_indices_buffer = torch.zeros(
        (num_tokens, TOPK_TOKENS), dtype=torch.int32, device=device
    )
    with set_forward_context({LAYER: metadata}, config):
        torch.ops.vllm.rocm_aiter_sparse_attn_indexer(
            torch.zeros(num_tokens, 1, device=device),
            LAYER,
            kv_cache,
            q_fp8,
            None,
            weights,
            128,
            "ue8m0",
            TOPK_TOKENS,
            HEAD_DIM,
            context,
            context,
            topk_indices_buffer,
            skip_k_cache_insert=True,
        )
    torch.accelerator.synchronize()
    return topk_indices_buffer


def _worker(local_rank, world_size, q, shape, port):
    """Join the TP group on one GPU and compare the two paths on one shape.

    A failed assertion in a spawned process would surface only as a non-zero
    exit code, so it is reported through ``q``.
    """
    monkeypatch = pytest.MonkeyPatch()
    config = VllmConfig(parallel_config=ParallelConfig(tensor_parallel_size=world_size))
    with monkeypatch.context() as m, set_current_vllm_config(config):
        device = torch.device(f"cuda:{local_rank}")
        torch.accelerator.set_device_index(device)
        update_environment_variables(
            {
                "RANK": str(local_rank),
                "LOCAL_RANK": str(local_rank),
                "WORLD_SIZE": str(world_size),
                "MASTER_ADDR": "localhost",
                "MASTER_PORT": str(port),
            }
        )
        init_distributed_environment()
        initialize_model_parallel(tensor_model_parallel_size=world_size)
        init_workspace_manager(device)

        try:
            num_tokens, context, num_chunks = shape
            inputs = _build_inputs(num_tokens, context, num_chunks, device)
            non_cp_topk = _run_indexer(m, False, inputs, config, device)
            cp_topk = _run_indexer(m, True, inputs, config, device)
            assert (non_cp_topk >= 0).any(), "op wrote nothing"
            _assert_topk_equivalent(
                cp_topk, non_cp_topk, *_reference_logits(inputs, device)
            )
        except Exception as exc:  # noqa: BLE001 - reported to the parent
            import traceback

            q.put(f"rank {local_rank}: {exc}\n{traceback.format_exc()}")


@pytest.mark.skipif(
    not current_platform.is_rocm(), reason="ROCm AITER sparse indexer only"
)
# CI has 2 GPUs; TP4 runs where 4 are available.
@pytest.mark.parametrize("tp_size", [2, 4])
@pytest.mark.parametrize("shape", SHAPES, ids=[f"m{m}-n{n}-c{c}" for m, n, c in SHAPES])
def test_m_split(tp_size, shape):
    """Splitting rows changes who computes which row, never the Top-K."""
    if torch.accelerator.device_count() < tp_size:
        pytest.skip(f"needs {tp_size} GPUs")
    if _cp_indexer_threshold() is None:
        pytest.skip("the CP indexer is not enabled on this GPU arch")

    q = mp.get_context("spawn").Queue()
    try:
        mp.spawn(_worker, args=(tp_size, q, shape, get_open_port()), nprocs=tp_size)
        failures = []
        while True:
            try:
                failures.append(q.get(timeout=1))
            except queue.Empty:
                break
        assert not failures, "\n\n".join(failures)
    finally:
        cleanup_dist_env_and_memory()


@pytest.fixture
def aiter_mla(monkeypatch: pytest.MonkeyPatch):
    """Turn AITER MLA on through its env vars; restore the cached flags after."""
    monkeypatch.setenv("VLLM_ROCM_USE_AITER", "1")
    monkeypatch.setenv("VLLM_ROCM_USE_AITER_MLA", "1")
    rocm_aiter_ops.refresh_env_variables()
    yield
    monkeypatch.undo()
    rocm_aiter_ops.refresh_env_variables()


@pytest.mark.skipif(not is_aiter_found_and_supported(), reason="needs AITER")
@pytest.mark.parametrize(
    ("arch", "seq_len", "tp_size", "chunk_rows", "candidates", "expected"),
    [
        ("gfx950", 16384, 2, None, False, True),
        ("gfx950", 16383, 2, None, False, False),
        ("gfx950:sramecc+:xnack-", 16384, 2, None, False, True),
        ("gfx942", 65536, 2, None, False, False),
        ("gfx950", 16384, 1, None, False, False),
        # no chunk has a row per rank, so nothing would be split
        ("gfx950", 16384, 2, (1, 1), False, False),
        ("gfx950", 16384, 2, (1, 2), False, True),
        ("gfx950", 16384, 2, (2,), True, False),
    ],
)
def test_use_cp_indexer(
    aiter_mla, monkeypatch, arch, seq_len, tp_size, chunk_rows, candidates, expected
):
    """Only long prefills on listed archs, with TP and enough rows, are split."""
    monkeypatch.setattr(mla_sparse, "_GCN_ARCH", arch)
    chunks = None
    if chunk_rows is not None:
        metadata = _build_inputs(sum(chunk_rows), 64, chunk_rows, "cpu")[1]
        chunks = metadata.prefill.chunks
    candidate_blocks = torch.zeros(1, dtype=torch.int32) if candidates else None
    assert (
        mla_sparse.use_cp_indexer(seq_len, tp_size, chunks, candidate_blocks)
        is expected
    )


@pytest.mark.parametrize("tp_size", [1, 2, 8])
@pytest.mark.parametrize("rows", [0, 1, 2, 7, 8, 9, 15, 16, 17, 63, 2000, 9000])
def test_m_split_stripes_cover_rows_once(tp_size, rows):
    """Ragged halves and sub-TP tails cannot lose or duplicate query rows."""
    owners = [0] * rows
    for rank in range(tp_size):
        for offset, count in mla_sparse._m_split_stripes(rows, rank, tp_size):
            assert 0 <= offset < offset + count <= rows
            for row in range(offset, offset + count):
                owners[row] += 1
    assert owners == [1] * rows


@pytest.mark.parametrize("tp_size", [1, 2, 8])
@pytest.mark.parametrize("queries", [[1], [7], [17], [1, 16, 2], [63, 129]])
@pytest.mark.parametrize("budget_rows", [1, 3, 32])
def test_m_split_chunk_budget(tp_size, queries, budget_rows):
    """Chunk sizing must bound logits even when every chunk stays replicated."""
    contexts = torch.tensor([1024] * len(queries), dtype=torch.int32)
    budget = budget_rows * 1024 * 4
    chunks = DeepseekV32IndexerMetadataBuilder._split_indexer_prefill_chunks(
        contexts, torch.tensor(queries), 4096, budget, m_split_tp_size=tp_size
    )
    # As in use_cp_indexer(): split only if a chunk has a row for every rank.
    split = tp_size > 1 and any(q.stop - q.start >= tp_size for _, q in chunks)
    assert sum(q.stop - q.start for _, q in chunks) == sum(queries)
    for requests, query in chunks:
        width = int(contexts[requests].sum())
        rows = query.stop - query.start
        launches = [rows]
        if split:
            launches = [
                count
                for rank in range(tp_size)
                for _, count in mla_sparse._m_split_stripes(rows, rank, tp_size)
            ]
        assert max(launches) * width * 4 <= budget
