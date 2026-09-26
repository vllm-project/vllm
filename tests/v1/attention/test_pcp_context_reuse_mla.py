# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import numpy as np
import torch

from vllm.model_executor.layers.attention.mla_attention import (
    _build_pcp_context_reuse,
)


def _builder():
    return SimpleNamespace(
        dcp_world_size=1,
        dcp_local_block_size=1,
        dcp_virtual_block_size=1,
        use_sparse=False,
        page_size=16,
        device=torch.device("cpu"),
        chunked_prefill_workspace_size=128,
        chunked_prefill_workspace=torch.empty((128, 1)),
        model_config=SimpleNamespace(get_sliding_window=lambda: None),
    )


def _metadata(req_idx=(7, 7), causal=True):
    return SimpleNamespace(
        causal=causal,
        req_idx=np.array(req_idx, dtype=np.intp),
    )


def _build_plan(
    *,
    builder=None,
    metadata=None,
    num_decodes=0,
    context_lens=(64, 112),
    query_starts=(0, 16, 32),
):
    return _build_pcp_context_reuse(
        builder or _builder(),
        metadata or _metadata(),
        num_decodes=num_decodes,
        num_prefills=len(context_lens),
        context_lens_cpu=torch.tensor(context_lens, dtype=torch.int32),
        prefill_query_start_loc_cpu=torch.tensor(query_starts, dtype=torch.int32),
    )


def test_pcp_context_reuse_builds_shared_and_tail_chunks() -> None:
    plan = _build_plan()
    assert plan is not None
    query_start_loc, query_lens_cpu, max_query_len, chunked = plan
    assert query_start_loc.tolist() == [0, 32]
    assert query_lens_cpu.tolist() == [32]
    assert max_query_len == 32

    assert chunked is not None
    assert len(chunked.chunks) == 2
    shared, tail = chunked.chunks
    assert shared.request_slice == slice(0, 1)
    assert shared.token_slice == slice(0, 32)
    assert shared.starts.tolist() == [0]
    assert not shared.is_continuation
    assert tail.token_slice == slice(16, 32)
    assert tail.starts.tolist() == [80]
    assert tail.is_continuation


def test_pcp_context_reuse_supports_unaligned_cached_gap() -> None:
    plan = _build_plan(
        context_lens=(70, 100),
        query_starts=(0, 8, 24),
    )
    assert plan is not None
    query_start_loc, _, _, chunked = plan
    assert query_start_loc.tolist() == [0, 24]
    assert chunked is not None
    assert len(chunked.chunks) == 2
    shared, tail = chunked.chunks
    assert shared.starts.tolist() == [0]
    assert tail.starts.tolist() == [78]
    assert tail.token_slice == slice(8, 24)
    assert tail.is_continuation


def test_pcp_context_reuse_supports_mixed_decode_batch() -> None:
    plan = _build_plan(
        metadata=_metadata(req_idx=(99, 7, 7)),
        num_decodes=1,
    )
    assert plan is not None
    assert plan[0].tolist() == [0, 32]
    assert plan[1].tolist() == [32]


def test_pcp_context_reuse_keeps_singletons_around_pairs() -> None:
    plan = _build_plan(
        metadata=_metadata(req_idx=(10, 20, 20)),
        context_lens=(32, 64, 112),
        query_starts=(0, 8, 24, 40),
    )
    assert plan is not None
    query_start_loc, query_lens_cpu, _, chunked = plan
    assert query_start_loc.tolist() == [0, 8, 40]
    assert query_lens_cpu.tolist() == [8, 32]
    assert chunked is not None
    assert [chunk.request_slice for chunk in chunked.chunks] == [
        slice(0, 1),
        slice(1, 2),
        slice(1, 2),
    ]


def test_pcp_context_reuse_falls_back_for_dcp() -> None:
    builder = _builder()
    builder.dcp_world_size = 2
    assert _build_plan(builder=builder) is None
