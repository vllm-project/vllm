# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Candidate indexer configuration without loading model weights."""

import os

import pytest

from vllm.engine.arg_utils import EngineArgs
from vllm.platforms import current_platform

sm90_only = pytest.mark.skipif(
    not current_platform.is_cuda()
    or not current_platform.is_device_capability_family(90),
    reason="Requires an SM90 GPU",
)


@sm90_only
@pytest.mark.parametrize(
    "options,expected",
    [
        pytest.param({}, True, id="default"),
        pytest.param(
            {"attention_config": {"indexer_sparse_logits": False}},
            False,
            id="disabled",
        ),
        pytest.param(
            {"attention_config": {"indexer_sparse_logits": True}},
            True,
            id="enabled",
        ),
        pytest.param(
            {"hf_overrides": {"candidate_topk_blocks": 3}},
            True,
            id="candidate-geometry",
        ),
        pytest.param(
            {"hf_overrides": {"index_n_heads": 16}},
            True,
            id="head-geometry",
        ),
        pytest.param(
            {"attention_config": {"indexer_kv_dtype": "mxfp4"}},
            True,
            id="cache-validation",
        ),
        pytest.param(
            {
                "tensor_parallel_size": 2,
                "decode_context_parallel_size": 2,
                "nnodes": 2,
            },
            False,
            id="decode-cp",
        ),
        pytest.param(
            {"prefill_context_parallel_size": 2, "nnodes": 2},
            False,
            id="prefill-cp",
        ),
    ],
)
def test_candidate_indexer_default_respects_support_and_override(options, expected):
    """Enable by platform and overrides; leave kernel constraints to the backend."""
    config = EngineArgs(
        model=os.getenv("MODEL_NAME", "deepseek-ai/DeepSeek-V4.1-Flash"),
        tokenizer_mode="deepseek_v41",
        language_model_only=True,
        max_model_len=4096,
        **options,
    ).create_engine_config()
    assert config.attention_config.indexer_sparse_logits is expected


@sm90_only
def test_candidate_indexer_microbatching_respects_model_runner_validation():
    """Text-only mode retains the multimodal architecture's microbatching limit."""
    args = EngineArgs(
        model=os.getenv("MODEL_NAME", "deepseek-ai/DeepSeek-V4.1-Flash"),
        tokenizer_mode="deepseek_v41",
        language_model_only=True,
        max_model_len=4096,
        ubatch_size=2,
        hf_overrides={"engram_layer_ids": []},
        all2all_backend="deepep_high_throughput",
    )
    with pytest.raises(ValueError, match="dual batch overlap with multimodal models"):
        args.create_engine_config()


def _pcp_common_metadata(rank, ratio, replay):
    import numpy as np
    import torch

    from vllm.v1.attention.backend import CommonAttentionMetadata
    from vllm.v1.worker.gpu.pcp_manager import PCPManager

    manager = PCPManager(
        pcp_world_size=2,
        pcp_rank=rank,
        device=torch.device("cpu"),
        shard_decode_requests=False,
    )
    query_lengths = np.array([13, 7], dtype=np.int32)
    contexts = [32768, 65536]
    tables = torch.arange(64, dtype=torch.int32).view(2, 32)
    positions, row_requests, expected_tables = [], [], []
    starts, lengths, local_tables = [0], [], []
    for request, offset, length in manager._iter_rank_chunks(
        rank, query_lengths, np.array([True, True])
    ):
        segment = list(
            range(contexts[request] + offset, contexts[request] + offset + length)
        )
        if replay:
            segment = segment[-min(2, length) :]
        local_request = len(lengths)
        positions.extend(segment)
        row_requests.extend([local_request] * len(segment))
        expected_tables.extend([tables[request]] * len(segment))
        starts.append(len(positions))
        lengths.append(contexts[request] + offset + length)
        local_tables.append(tables[request])
    valid_rows = len(positions)
    # An empty local request and padded tokens must not read table row -1.
    starts.append(valid_rows)
    lengths.append(0)
    local_tables.append(torch.full((32,), -123, dtype=torch.int32))
    positions += [0, 0]
    positions_tensor = torch.tensor(positions, dtype=torch.int64)
    query_starts = torch.tensor(starts, dtype=torch.int32)
    common = CommonAttentionMetadata(
        query_start_loc=query_starts,
        query_start_loc_cpu=query_starts.clone(),
        seq_lens=torch.tensor(lengths, dtype=torch.int32),
        num_reqs=len(lengths),
        num_actual_tokens=len(positions),
        max_query_len=max(np.diff(starts)),
        max_seq_len=max(lengths),
        block_table_tensor=torch.stack(local_tables),
        slot_mapping=torch.full((len(positions) * 2,), -1, dtype=torch.int64),
        positions=positions_tensor,
    )
    expected_lens = (positions_tensor[:valid_rows] + 1) // ratio
    return common, valid_rows, row_requests, expected_lens, expected_tables


@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("ratio", [1, 2])
@pytest.mark.parametrize("replay", [False, True])
def test_candidate_pcp_rows_preserve_segment_tables_and_absolute_bounds(
    rank, ratio, replay
):
    """PCP front/back and replay suffixes retain their local table identity."""
    import torch

    from vllm.models.deepseek_v41.nvidia.ops.candidate_pcp import (
        prepare_pcp_candidate_rows,
    )

    common, valid, requests, lengths, tables = _pcp_common_metadata(rank, ratio, replay)
    row_lens = torch.full((common.num_actual_tokens,), -123, dtype=torch.int32)
    row_indices = torch.full_like(row_lens, -123)
    assert prepare_pcp_candidate_rows(common, ratio, row_lens, row_indices)
    torch.testing.assert_close(row_lens[:valid], lengths.to(torch.int32))
    assert row_indices[:valid].tolist() == requests
    torch.testing.assert_close(
        common.block_table_tensor[row_indices[:valid].long()],
        torch.stack(tables),
    )
    assert row_lens[valid:].tolist() == [0, 0]
    assert row_indices[valid:].tolist() == [0, 0]
    # Cached/replayed K slots are -1 but their query rows must remain valid.
    assert (row_lens[:valid] > 0).all()


@pytest.mark.parametrize("missing", [True, False])
def test_candidate_pcp_missing_or_short_positions_leave_dense_fallback(missing):
    """Unsupported position metadata must not partially overwrite a plan."""
    import torch

    from vllm.models.deepseek_v41.nvidia.ops.candidate_pcp import (
        prepare_pcp_candidate_rows,
    )

    common, *_ = _pcp_common_metadata(0, 1, False)
    common = common.replace(positions=None if missing else common.positions[:1])
    row_lens = torch.full((common.num_actual_tokens,), -123, dtype=torch.int32)
    row_indices = torch.full_like(row_lens, -123)
    assert not prepare_pcp_candidate_rows(common, 1, row_lens, row_indices)
    assert (row_lens == -123).all() and (row_indices == -123).all()


def test_candidate_pcp_replay_drops_front_segment_before_valid_suffix():
    """An empty PCP front segment must not steal the kept suffix's table."""
    import torch

    from vllm.models.deepseek_v41.nvidia.ops.candidate_pcp import (
        prepare_pcp_candidate_rows,
    )

    common, *_ = _pcp_common_metadata(0, 1, False)
    bounds = torch.tensor([0, 0, 2, 2], dtype=torch.int32)
    common = common.replace(
        query_start_loc=bounds,
        query_start_loc_cpu=bounds.clone(),
        seq_lens=torch.tensor([32768, 98304, 0], dtype=torch.int32),
        num_reqs=3,
        num_actual_tokens=3,
        max_query_len=2,
        max_seq_len=98304,
        slot_mapping=torch.full((6,), -1, dtype=torch.int64),
        block_table_tensor=torch.tensor([[17], [29], [-123]], dtype=torch.int32),
        positions=torch.tensor([98302, 98303, 0], dtype=torch.int64),
    )
    lens = torch.full((3,), -123, dtype=torch.int32)
    indices = torch.full_like(lens, -123)
    assert prepare_pcp_candidate_rows(common, 1, lens, indices)
    assert indices.tolist() == [1, 1, 0]
    assert lens.tolist() == [98303, 98304, 0]
    assert common.block_table_tensor[indices[:2].long(), 0].tolist() == [29, 29]
