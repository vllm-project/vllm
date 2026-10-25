# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepseekV32IndexerMetadataBuilder on ROCm: fields against independent values.

Covers the decode/prefill split (spec rows as decodes, native next_n 2 vs
per-token flattening), per-token decode context lengths, prefill chunk
bounds (``cu_seqlen_ks/ke`` per query token, request and query sub-chunks),
persistent buffers and the GPU-sync contract of decode builds.
"""

from contextlib import contextmanager

import pytest
import torch

from tests.v1.attention.backends.rocm_dsv32_indexer.idx_harness import (
    make_idx_case,
)
from tests.v1.attention.model_shapes import DEEPSEEK_V3_2
from vllm import envs
from vllm.utils import gpu_sync_debug


@contextmanager
def runner_sync_check(monkeypatch):
    monkeypatch.setattr(gpu_sync_debug, "_SYNC_CHECK_MODE", "error")
    monkeypatch.setattr(gpu_sync_debug, "_sync_check_enabled", True)
    torch.cuda.synchronize()
    with gpu_sync_debug._checked_region("error"):
        yield


@pytest.mark.parametrize(
    "spec_tokens, query_lens, seq_lens",
    [
        (0, [1, 1, 1], [5, 300, 4000]),
        (1, [2, 2], [9, 300]),
        (3, [4, 2, 3], [9, 300, 2000]),
    ],
    ids=["plain", "native-next2", "flattened"],
)
def test_decode_context_lengths(env, spec_tokens, query_lens, seq_lens):
    case = make_idx_case(DEEPSEEK_V3_2, seq_lens, spec_tokens=spec_tokens)
    md = case.build(case.batch(query_lens, seq_lens))
    assert md.num_decodes == len(query_lens) and md.num_prefills == 0
    assert md.num_decode_tokens == sum(query_lens)
    # Token j of a request with L keys and q queries sees L - q + j + 1 keys.
    per_token = [s - q + j + 1 for q, s in zip(query_lens, seq_lens) for j in range(q)]
    got = md.decode.seq_lens.flatten().tolist()
    if case.builder.use_flattening:
        assert got[: len(per_token)] == per_token
        assert md.decode.block_table.shape[0] == sum(query_lens)
    else:
        assert got == per_token
    assert not md.decode.requires_padding


@pytest.mark.parametrize(
    "query_lens, seq_lens",
    [([100, 7], [100, 7]), ([64, 200], [500, 1200]), ([1, 1, 120], [900, 255, 300])],
    ids=["prefill", "extend", "mixed"],
)
def test_prefill_chunk_bounds(env, query_lens, seq_lens):
    case = make_idx_case(DEEPSEEK_V3_2, seq_lens)
    md = case.build(case.batch(query_lens, seq_lens))
    n_dec = sum(q == 1 for q in query_lens)
    assert md.num_decodes == n_dec
    ks, ke = [], []
    offset = 0
    for q, s in list(zip(query_lens, seq_lens))[n_dec:]:
        for j in range(q):
            ks.append(offset)
            ke.append(offset + s - q + j + 1)
        offset += s
    got_ks = torch.cat([c.cu_seqlen_ks for c in md.prefill.chunks]).tolist()
    got_ke = torch.cat([c.cu_seqlen_ke for c in md.prefill.chunks]).tolist()
    assert got_ks == ks and got_ke == ke
    (chunk,) = md.prefill.chunks
    assert chunk.total_seq_lens == sum(seq_lens[n_dec:])
    assert chunk.token_start == sum(query_lens[:n_dec])
    assert chunk.token_end == sum(query_lens)


def test_query_sub_chunks_cover_every_token_once(env, monkeypatch):
    monkeypatch.setattr(envs, "VLLM_SPARSE_INDEXER_MAX_LOGITS_MB", 1)
    query_lens, seq_lens = [700, 300, 1000], [700, 300, 1000]
    case = make_idx_case(DEEPSEEK_V3_2, seq_lens)
    md = case.build(case.batch(query_lens, seq_lens))
    spans = [(c.token_start, c.token_end) for c in md.prefill.chunks]
    covered = [t for a, b in spans for t in range(a, b)]
    assert covered == list(range(sum(query_lens)))
    for c in md.prefill.chunks:
        logits_bytes = (c.token_end - c.token_start) * c.total_seq_lens * 4
        assert logits_bytes <= 1 << 20 or c.token_end - c.token_start == 1


def test_persistent_decode_buffers_are_reused(env):
    case = make_idx_case(DEEPSEEK_V3_2, [4000] * 4, spec_tokens=3)
    ptrs = set()
    for seq_lens in ([9, 300, 2000], [10, 301, 2001]):
        md = case.build(case.batch([4, 2, 3], seq_lens))
        ptrs.add(md.decode.seq_lens.data_ptr())
    assert len(ptrs) == 1


@pytest.mark.parametrize("spec_tokens, q", [(0, 1), (1, 2), (3, 4)])
def test_decode_build_does_not_sync(env, monkeypatch, spec_tokens, q):
    seq_lens = [9, 300, 2000]
    case = make_idx_case(DEEPSEEK_V3_2, seq_lens, spec_tokens=spec_tokens)
    batch = case.batch([q] * 3, seq_lens)
    case.build(batch)
    with runner_sync_check(monkeypatch):
        case.build(batch)
