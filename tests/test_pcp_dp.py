# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import torch

from vllm.config import ParallelConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.distributed.device_communicators.all2all import AgRsAll2AllManager
from vllm.forward_context import DPMetadata, _compute_sp_num_tokens
from vllm.v1.worker.dp_utils import coordinate_batch_across_dp
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor
from vllm.v1.worker.gpu.dp_utils import sync_cudagraph_and_dp_padding
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
        moe_non_sp_token_counts_cpu=torch.tensor([10, 7, 5, 3]),
    )

    with metadata.sp_local_sizes(
        sequence_parallel_size=1,
        num_dispatchers_per_dp_rank=2,
    ) as sizes:
        assert sizes == [10, 7, 5, 3]


def test_batch_coordination_returns_dp_and_moe_non_sp_token_counts(monkeypatch):
    config = ParallelConfig(
        prefill_context_parallel_size=2,
        data_parallel_size=2,
        enable_expert_parallel=True,
    )
    collective_calls = 0

    monkeypatch.setattr(
        "vllm.v1.worker.dp_utils._get_device_and_group",
        lambda parallel_config: ("cpu", object(), 4, 0, 2, 0),
    )

    def fake_all_reduce(tensor, group):
        nonlocal collective_calls
        collective_calls += 1
        tensor.copy_(
            torch.tensor(
                [
                    [10, 7, 5, 3],
                    [10, 7, 5, 3],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                ],
                dtype=torch.int32,
            )
        )

    monkeypatch.setattr("vllm.v1.worker.dp_utils.dist.all_reduce", fake_all_reduce)

    _, num_tokens_across_dp, moe_non_sp_token_counts, _ = coordinate_batch_across_dp(
        num_tokens_unpadded=10,
        allow_microbatching=False,
        parallel_config=config,
    )

    assert collective_calls == 1
    assert num_tokens_across_dp.tolist() == [10, 5]
    assert moe_non_sp_token_counts.tolist() == [10, 7, 5, 3]


def test_v2_batch_coordination_returns_dp_and_moe_non_sp_token_counts(monkeypatch):
    collective_calls = 0
    group = SimpleNamespace(world_size=4, rank_in_group=1, cpu_group=object())
    monkeypatch.setattr("vllm.v1.worker.gpu.dp_utils.get_dp_group", lambda: group)
    monkeypatch.setattr(
        "vllm.v1.worker.gpu.dp_utils.get_moe_non_sp_group", lambda: group
    )

    def fake_all_reduce(tensor, group):
        nonlocal collective_calls
        collective_calls += 1
        tensor.copy_(
            torch.tensor(
                [
                    [10, 7, 5, 3],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [-1, -1, -1, -1],
                ],
                dtype=torch.int32,
            )
        )

    monkeypatch.setattr("vllm.v1.worker.gpu.dp_utils.dist.all_reduce", fake_all_reduce)

    _, sync = sync_cudagraph_and_dp_padding(
        cudagraph_manager=None,
        desired_batch_desc=BatchExecutionDescriptor(
            cg_mode=CUDAGraphMode.NONE,
            num_tokens=7,
            num_reqs=1,
        ),
        num_tokens=7,
        num_reqs=1,
        uniform_token_count=None,
        dp_size=2,
        dp_rank=0,
        pcp_size=2,
        enable_expert_parallel=True,
    )

    assert collective_calls == 1
    assert sync is not None
    assert sync.num_tokens_across_dp.tolist() == [7, 3]
    assert sync.moe_non_sp_token_counts is not None
    assert sync.moe_non_sp_token_counts.tolist() == [10, 7, 5, 3]


def test_ag_rs_selects_moe_non_sp_group(monkeypatch):
    moe_non_sp_group = object()
    manager = AgRsAll2AllManager.__new__(AgRsAll2AllManager)
    monkeypatch.setattr(
        "vllm.distributed.device_communicators.all2all.get_moe_non_sp_group",
        lambda: moe_non_sp_group,
    )
    assert manager._get_comm_group(is_sequence_parallel=False) is moe_non_sp_group


def test_ag_rs_selects_ep_group_for_sp(monkeypatch):
    ep_group = object()
    manager = AgRsAll2AllManager.__new__(AgRsAll2AllManager)
    monkeypatch.setattr(
        "vllm.distributed.device_communicators.all2all.get_ep_group",
        lambda: ep_group,
    )
    assert manager._get_comm_group(is_sequence_parallel=True) is ep_group


def test_ag_rs_moe_non_sp_uses_one_collective_for_each_direction(monkeypatch):
    calls = []

    class FakeMoeNonSpGroup:
        world_size = 4
        rank_in_group = 2

        def all_gatherv(self, tensors, dim, sizes):
            calls.append(("moe_non_sp_ag", sizes))
            assert tensors[0].tolist() == [20, 21]
            return [torch.tensor([10, 11, 12, 20, 21, 22]) for _ in tensors]

        def reduce_scatterv(self, tensor, dim, sizes):
            calls.append(("moe_non_sp_rs", sizes))
            assert tensor.tolist() == [10, 11, 12, 20, 21, 22]
            return tensor[3:5]

    manager = AgRsAll2AllManager.__new__(AgRsAll2AllManager)
    sizes = [1, 2, 2, 1]
    group = FakeMoeNonSpGroup()
    monkeypatch.setattr(
        "vllm.distributed.device_communicators.all2all.get_moe_non_sp_group",
        lambda: group,
    )
    manager._get_sizes = lambda num_local_tokens, comm_group: sizes
    hidden_states, _, _ = manager.dispatch(
        torch.tensor([20, 21]),
        torch.tensor([1, 1]),
        torch.tensor([0, 0]),
    )
    combined = manager.combine(hidden_states)

    assert hidden_states.tolist() == [10, 11, 12, 20, 21, 22]
    assert combined.tolist() == [20, 21]
    assert calls == [
        ("moe_non_sp_ag", [1, 2, 2, 1]),
        ("moe_non_sp_rs", [1, 2, 2, 1]),
    ]
