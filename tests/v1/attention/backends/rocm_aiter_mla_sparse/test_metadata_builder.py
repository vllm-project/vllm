# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ROCMAiterMLASparseMetadataBuilder: every field against an independent value.

Covers the decode/prefill split (spec tokens as decodes), the ragged
per-token KV extents the kernels walk, the dense-prefill carve-out, the
persistent AITER work metadata and its fingerprint cache, persistent buffers,
padded (CUDA-graph) batches and the GPU-sync contract of decode builds.
"""

from contextlib import contextmanager

import pytest
import torch

from tests.v1.attention.backends.rocm_aiter_mla_sparse.sparse_harness import (
    MLA_METADATA,
    make_case,
)
from tests.v1.attention.model_shapes import DEEPSEEK_V3_2, GLM_5_3_FLASH
from vllm.utils import gpu_sync_debug

DSV32 = DEEPSEEK_V3_2.shard(8)

# (query_lens, seq_lens) in reorder order: decodes first.
BATCHES = {
    "decode": ([1, 1, 1], [17, 300, 4096]),
    "spec": ([4, 2, 3], [17, 300, 4096]),
    "prefill_short": ([64, 7], [64, 7]),
    "prefill_long": ([300, 200], [300, 900]),
    "mixed": ([1, 4, 40, 300], [100, 300, 200, 300]),
}


@contextmanager
def runner_sync_check(monkeypatch):
    """Police syncs the way the runner does with VLLM_GPU_SYNC_CHECK=error:
    a sync outside a ``gpu_sync_allowed()`` region raises."""
    monkeypatch.setattr(gpu_sync_debug, "_SYNC_CHECK_MODE", "error")
    monkeypatch.setattr(gpu_sync_debug, "_sync_check_enabled", True)
    torch.cuda.synchronize()
    with gpu_sync_debug._checked_region("error"):
        yield


def _expected_extents(query_lens, seq_lens, topk, num_reqs):
    """Per-token count of top-k slots the kernel may read: min(pos + 1, topk)."""
    extents = []
    for q, s in list(zip(query_lens, seq_lens))[:num_reqs]:
        extents.extend(min(pos + 1, topk) for pos in range(s - q, s))
    return extents


@pytest.mark.parametrize("name", list(BATCHES))
def test_phase_split_and_ragged_extents(env, name):
    query_lens, seq_lens = BATCHES[name]
    case = make_case(DSV32, seq_lens, spec_tokens=3)
    batch = case.batch(query_lens, seq_lens)
    md = case.build(batch)

    is_decode = [q <= 4 for q in query_lens]
    n_dec = sum(is_decode)
    assert md.num_decodes == n_dec
    assert md.num_prefills == len(query_lens) - n_dec
    assert md.num_decode_tokens == sum(query_lens[:n_dec])
    assert md.num_actual_tokens == sum(query_lens)
    assert md.topk_tokens == case.topk
    assert md.block_size == case.block_size
    prefill_lens = seq_lens[n_dec:]
    dense = bool(prefill_lens) and max(prefill_lens) <= case.topk
    assert (md.prefill is not None) == (n_dec < len(query_lens))
    if md.prefill is not None:
        assert md.prefill.use_dense_mha == dense
        assert md.prefill_max_seq_len == max(prefill_lens)

    # Dense prefills run outside the sparse kernel; only decodes are ragged.
    num_reqs = n_dec if dense else len(query_lens)
    extents = _expected_extents(query_lens, seq_lens, case.topk, num_reqs)
    n = len(extents)
    indptr = torch.tensor([0, *extents]).cumsum(0)
    assert md.paged_kv_indptr.tolist() == indptr.tolist()
    # The kernels' persistent buffers hold the full tail at the last extent.
    tail = case.builder.paged_kv_indptr[n + 1 :]
    assert (tail == indptr[-1].item()).all()
    assert md.qo_indptr.tolist() == list(range(n + 1))
    assert md.paged_kv_last_page_len.tolist() == [1] * n
    assert md.paged_kv_indices.numel() == n * case.topk
    req_ids = [r for r, q in enumerate(query_lens[:num_reqs]) for _ in range(q)]
    assert md.req_id_per_token.tolist() == req_ids
    assert md.attn_out_dtype == case.dtype


def test_spec_rows_are_prefills_without_spec_config(env):
    query_lens, seq_lens = BATCHES["spec"]
    case = make_case(DSV32, seq_lens)
    md = case.build(case.batch(query_lens, seq_lens))
    assert md.num_decodes == 0 and md.num_prefills == 3


@pytest.mark.parametrize("topk", [128, 2048])
def test_extents_saturate_at_topk(env, topk):
    query_lens, seq_lens = [1, 1, 1], [topk - 1, topk, topk + 1]
    case = make_case(DSV32, seq_lens, topk=topk)
    md = case.build(case.batch(query_lens, seq_lens))
    assert torch.diff(md.paged_kv_indptr).tolist() == [topk - 1, topk, topk]


def test_persistent_buffers_are_reused(env):
    case = make_case(DSV32, [4096] * 4)
    ptrs = None
    for query_lens, seq_lens in BATCHES.values():
        md = case.build(case.batch(query_lens, seq_lens))
        views = (
            md.paged_kv_indptr,
            md.paged_kv_indices,
            md.qo_indptr,
            md.req_id_per_token,
        )
        now = [t.data_ptr() if t.numel() else None for t in views]
        ptrs = ptrs or now
        assert all(a is None or a == b for a, b in zip(now, ptrs))


def test_shrinking_batch_zeroes_stale_index_tail(env):
    case = make_case(DSV32, [4096] * 4)
    big = case.batch([300, 300], [300, 900])
    case.build(big)
    case.builder.paged_kv_indices.fill_(7)
    small = case.batch([1, 1], [17, 300])
    md = case.build(small)
    n = md.paged_kv_indptr.numel() - 1
    stale = case.builder.paged_kv_indices[n * case.topk : 600 * case.topk]
    assert (stale == 0).all()


@pytest.mark.parametrize("name", ["decode", "spec", "prefill_long", "mixed"])
def test_persistent_work_metadata(env, spy, name):
    query_lens, seq_lens = BATCHES[name]
    case = make_case(DSV32, seq_lens, spec_tokens=3)
    md = case.build(case.batch(query_lens, seq_lens))
    # A long pure-bf16 prefill runs the non-persistent kernel.
    persistent = name != "prefill_long"
    assert (md.work_meta_data is not None) == persistent
    assert spy.names().count(MLA_METADATA) == int(persistent)
    if persistent:
        (kw,) = spy.kwargs_of(MLA_METADATA)
        assert kw["max_seqlen_qo"] == 1 and kw["page_size"] == 1
        # Splits cover at most the top-k selected rows.
        effective = min(max(seq_lens), case.topk)
        assert kw["max_split_per_batch"] <= max(1, effective // 128)


def test_work_metadata_is_cached_while_extents_are_unchanged(env, spy):
    """Contexts past topk clamp to the same extents, so decode steps of long
    requests reuse the metadata; a changed extent recomputes it."""
    case = make_case(DSV32, [5000, 5000])
    for seq_lens in ([3000, 4000], [3001, 4001], [3002, 4002]):
        case.build(case.batch([1, 1], seq_lens))
    assert spy.names().count(MLA_METADATA) == 1
    case.build(case.batch([1, 1], [100, 4003]))
    assert spy.names().count(MLA_METADATA) == 2


def test_cached_work_metadata_gives_correct_output(env):
    """Reusing cached metadata with new block tables and lengths is exact."""
    case = make_case(DSV32, [3000, 4000, 3001, 4001])
    for i, seq_lens in enumerate(([3000, 4000], [3001, 4001])):
        case.seed = i
        batch = case.batch([1, 1], seq_lens)
        query = case.query(batch.num_tokens)
        indices = case.indices(batch)
        out = case.run(batch, query, indices)
        case.check(batch, query, indices, out[case.names[0]])


@pytest.mark.parametrize(
    "case_kw",
    [
        {"use_sinks": True},
        {"shape": GLM_5_3_FLASH.shard(8)},  # bf16 NoPE runs vLLM Triton
    ],
    ids=["sinks", "nope-bf16"],
)
def test_no_work_metadata_for_paths_that_never_read_it(env, spy, case_kw):
    shape = case_kw.pop("shape", DSV32)
    query_lens, seq_lens = BATCHES["decode"]
    case = make_case(shape, seq_lens, **case_kw)
    md = case.build(case.batch(query_lens, seq_lens))
    assert md.work_meta_data is None
    assert MLA_METADATA not in spy.names()


@pytest.mark.parametrize("name", ["decode", "spec"])
def test_decode_build_syncs_only_where_allowed(env, monkeypatch, name):
    query_lens, seq_lens = BATCHES[name]
    case = make_case(DSV32, seq_lens, spec_tokens=3)
    batch = case.batch(query_lens, seq_lens)
    case.build(batch)  # warm the fingerprint
    batch = case.batch(query_lens, [s + 1 for s in seq_lens])
    with runner_sync_check(monkeypatch):
        case.build(batch)


def test_padded_batch_reads_nothing_for_padding_tokens(env):
    """CUDA-graph padding tokens (slot -1) get empty KV extents."""
    case = make_case(DSV32, [300, 900])
    batch = case.batch([1, 1], [300, 900], num_padded_tokens=8)
    md = case.build(batch)
    indptr = case.builder.paged_kv_indptr[: 8 + 1]
    assert torch.diff(indptr)[2:].eq(0).all()
    assert md.num_actual_tokens == 2


def test_capture_build_is_accepted_by_forward(env):
    case = make_case(DSV32, [64] * 8)
    batch = case.batch([1] * 8, [64] * 8)
    md = case.builder.build_for_cudagraph_capture(batch.common)
    query = case.query(batch.num_tokens)
    indices = case.indices(batch)
    out = case.run(batch, query, indices, md)
    case.check(batch, query, indices, out[case.names[0]])
