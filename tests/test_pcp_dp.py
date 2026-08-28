# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.config import ParallelConfig
from vllm.distributed.device_communicators.all2all import AgRsAll2AllManager
from vllm.distributed.parallel_state import _get_moe_dp_pcp_group_ranks
from vllm.forward_context import DPMetadata, _compute_sp_num_tokens
from vllm.v1.worker.gpu_worker import _dp_local_rank_offset


def test_parallel_config_accepts_pcp_with_dp():
    config = ParallelConfig(
        tensor_parallel_size=2,
        prefill_context_parallel_size=2,
        data_parallel_size=2,
    )

    assert config.world_size_across_dp == 8


def test_dp_local_rank_offset_includes_pcp():
    config = SimpleNamespace(
        data_parallel_rank_local=1,
        data_parallel_index=3,
        pipeline_parallel_size=1,
        prefill_context_parallel_size=2,
        tensor_parallel_size=2,
    )

    assert _dp_local_rank_offset(config) == 4


def test_ep_dispatch_sizes_include_pcp_and_sequence_parallel_ranks():
    sizes = _compute_sp_num_tokens(
        torch.tensor([17, 8]),
        sequence_parallel_size=2,
        num_replicas_per_token_count=4,
    )

    assert sizes == [9, 9, 9, 9, 4, 4, 4, 4]


def test_ep_dispatch_sizes_preserve_dp_pcp_order_without_sp():
    metadata = DPMetadata(
        num_tokens_across_dp_cpu=torch.tensor([17, 8]),
        num_tokens_across_dp_pcp_cpu=torch.tensor([10, 7, 5, 3]),
    )

    with metadata.sp_local_sizes(
        sequence_parallel_size=1,
        num_dispatchers_per_dp_rank=2,
    ) as sizes:
        assert sizes == [10, 7, 5, 3]


@pytest.mark.parametrize(
    ("dp_size", "pcp_size", "expected"),
    [
        (2, 2, [[0, 2, 4, 6], [1, 3, 5, 7]]),
        (2, 1, [[0, 2], [1, 3]]),
        (1, 2, [[0, 2], [1, 3]]),
        (1, 1, [[0], [1]]),
    ],
)
def test_moe_dp_pcp_group_ranks_fix_tp_coordinate(dp_size, pcp_size, expected):
    all_ranks = torch.arange(dp_size * pcp_size * 2).reshape(1, dp_size, 1, pcp_size, 2)

    group_ranks = _get_moe_dp_pcp_group_ranks(
        all_ranks,
        data_parallel_size=dp_size,
        prefill_context_parallel_size=pcp_size,
    )

    assert group_ranks == expected


def test_ag_rs_selects_moe_dp_pcp_group_for_non_sp(monkeypatch):
    moe_dp_pcp_group = object()
    manager = AgRsAll2AllManager.__new__(AgRsAll2AllManager)
    monkeypatch.setattr(
        "vllm.distributed.device_communicators.all2all.get_moe_dp_pcp_group",
        lambda: moe_dp_pcp_group,
    )
    monkeypatch.setattr(
        "vllm.distributed.device_communicators.all2all.has_moe_dp_pcp_group",
        lambda: True,
    )
    assert manager._uses_moe_dp_pcp_group(is_sequence_parallel=False)
    assert manager._get_comm_group(is_sequence_parallel=False) is moe_dp_pcp_group


def test_ag_rs_moe_dp_pcp_uses_one_collective_for_each_direction(monkeypatch):
    calls = []

    class FakeMoeDpPcpGroup:
        world_size = 4
        rank_in_group = 2

        def all_gatherv(self, tensors, dim, sizes):
            calls.append(("moe_dp_pcp_ag", sizes))
            assert tensors[0].tolist() == [20, 21]
            return [torch.tensor([10, 11, 12, 20, 21, 22]) for _ in tensors]

        def reduce_scatterv(self, tensor, dim, sizes):
            calls.append(("moe_dp_pcp_rs", sizes))
            assert tensor.tolist() == [10, 11, 12, 20, 21, 22]
            return tensor[3:5]

    monkeypatch.setattr(
        "vllm.distributed.device_communicators.all2all.get_moe_dp_pcp_group",
        lambda: FakeMoeDpPcpGroup(),
    )
    manager = AgRsAll2AllManager.__new__(AgRsAll2AllManager)
    sizes = [1, 2, 2, 1]
    group = FakeMoeDpPcpGroup()
    manager._get_comm_group = lambda is_sequence_parallel: group
    manager._get_sizes = lambda num_local_tokens, comm_group: sizes
    manager._uses_staged_dp_pcp = lambda is_sequence_parallel: False

    hidden_states, _, _ = manager.dispatch(
        torch.tensor([20, 21]),
        torch.tensor([1, 1]),
        torch.tensor([0, 0]),
    )
    combined = manager.combine(hidden_states)

    assert hidden_states.tolist() == [10, 11, 12, 20, 21, 22]
    assert combined.tolist() == [20, 21]
    assert calls == [
        ("moe_dp_pcp_ag", [1, 2, 2, 1]),
        ("moe_dp_pcp_rs", [1, 2, 2, 1]),
    ]
