# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Test the communication operators.

Run `pytest tests/distributed/test_comm_ops.py`.
"""

from collections.abc import Callable
from typing import Any
from unittest.mock import Mock

import pytest
import ray
import torch

from vllm.distributed import (
    broadcast_tensor_dict,
    get_pp_group,
    tensor_model_parallel_all_gather,
    tensor_model_parallel_all_reduce,
    tensor_model_parallel_reduce_scatter,
)
from vllm.distributed.device_communicators import flashinfer_all_reduce
from vllm.distributed.device_communicators.cuda_communicator import CudaCommunicator
from vllm.distributed.parallel_state import GroupCoordinator, TensorMetadata
from vllm.v1.worker.gpu_worker import AsyncIntermediateTensors

from ..utils import (
    init_test_distributed_environment,
    multi_gpu_test,
    multi_process_parallel,
)


@ray.remote(num_gpus=1, max_calls=1)
def all_reduce_test_worker(
    monkeypatch: pytest.MonkeyPatch,
    tp_size: int,
    pp_size: int,
    rank: int,
    distributed_init_port: str,
):
    # it is important to delete the CUDA_VISIBLE_DEVICES environment variable
    # so that each worker can see all the GPUs
    # they will be able to set the device to the correct GPU
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

    device = torch.device(f"cuda:{rank}")
    torch.accelerator.set_device_index(device)
    init_test_distributed_environment(tp_size, pp_size, rank, distributed_init_port)
    num_elements = 8
    all_tensors = [
        torch.arange(num_elements, dtype=torch.float32, device="cuda") * (r + 1)
        for r in range(tp_size)
    ]
    expected = torch.sum(torch.stack(all_tensors, dim=0), dim=0)
    t = all_tensors[rank % tp_size]
    t = tensor_model_parallel_all_reduce(t)
    torch.testing.assert_close(t, expected)


@ray.remote(num_gpus=1, max_calls=1)
def reduce_scatter_test_worker(
    monkeypatch: pytest.MonkeyPatch,
    tp_size: int,
    pp_size: int,
    rank: int,
    distributed_init_port: str,
):
    # it is important to delete the CUDA_VISIBLE_DEVICES environment variable
    # so that each worker can see all the GPUs
    # they will be able to set the device to the correct GPU
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    device = torch.device(f"cuda:{rank}")
    torch.accelerator.set_device_index(device)
    init_test_distributed_environment(tp_size, pp_size, rank, distributed_init_port)

    num_elements = 8
    all_tensors = [
        torch.arange(num_elements, dtype=torch.float32, device="cuda") * (r + 1)
        for r in range(tp_size)
    ]

    index = rank % tp_size
    partition_size = num_elements // tp_size
    all_reduce = torch.sum(torch.stack(all_tensors, dim=0), dim=0)
    expected = all_reduce[index * partition_size : (index + 1) * partition_size]
    t = all_tensors[index]
    t = tensor_model_parallel_reduce_scatter(t, 0)
    torch.testing.assert_close(t, expected)


@ray.remote(num_gpus=1, max_calls=1)
def all_gather_test_worker(
    monkeypatch: pytest.MonkeyPatch,
    tp_size: int,
    pp_size: int,
    rank: int,
    distributed_init_port: str,
):
    # it is important to delete the CUDA_VISIBLE_DEVICES environment variable
    # so that each worker can see all the GPUs
    # they will be able to set the device to the correct GPU
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    device = torch.device(f"cuda:{rank}")
    torch.accelerator.set_device_index(device)
    init_test_distributed_environment(tp_size, pp_size, rank, distributed_init_port)
    num_dimensions = 3
    tensor_size = list(range(2, num_dimensions + 2))
    total_size = 1
    for s in tensor_size:
        total_size *= s
    for all_gather_dimension in range(num_dimensions):
        all_tensors = [
            torch.arange(total_size, dtype=torch.float32, device="cuda").reshape(
                tensor_size
            )
            * (r + 1)
            for r in range(tp_size)
        ]
        expected = torch.cat(all_tensors, dim=all_gather_dimension)
        t = all_tensors[rank % tp_size]
        t = tensor_model_parallel_all_gather(t, all_gather_dimension)
        torch.testing.assert_close(t, expected)


@ray.remote(num_gpus=1, max_calls=1)
def broadcast_tensor_dict_test_worker(
    monkeypatch: pytest.MonkeyPatch,
    tp_size: int,
    pp_size: int,
    rank: int,
    distributed_init_port: str,
):
    # it is important to delete the CUDA_VISIBLE_DEVICES environment variable
    # so that each worker can see all the GPUs
    # they will be able to set the device to the correct GPU
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    device = torch.device(f"cuda:{rank}")
    torch.accelerator.set_device_index(device)
    init_test_distributed_environment(tp_size, pp_size, rank, distributed_init_port)
    test_dict = {
        # device tensor
        "a": torch.arange(8, dtype=torch.float32, device="cuda"),
        # CPU tensor
        "b": torch.arange(16, dtype=torch.int8, device="cpu"),
        "c": "test",
        "d": [1, 2, 3],
        "e": {"a": 1, "b": 2},
        # empty tensor
        "f": torch.tensor([], dtype=torch.float32, device="cuda"),
    }

    if (rank % tp_size) == 0:
        broadcast_tensor_dict(test_dict, src=0)
    else:
        recv_dict = broadcast_tensor_dict(src=0)
        assert len(recv_dict) == len(test_dict)
        torch.testing.assert_close(recv_dict["a"], test_dict["a"])
        torch.testing.assert_close(recv_dict["b"], test_dict["b"])
        assert recv_dict["c"] == test_dict["c"]
        assert recv_dict["d"] == test_dict["d"]
        assert recv_dict["e"] == test_dict["e"]
        torch.testing.assert_close(recv_dict["f"], test_dict["f"])


@ray.remote(num_gpus=1, max_calls=1)
def send_recv_tensor_dict_test_worker(
    monkeypatch: pytest.MonkeyPatch,
    tp_size: int,
    pp_size: int,
    rank: int,
    distributed_init_port: str,
):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    device = torch.device(f"cuda:{rank}")
    torch.accelerator.set_device_index(device)
    init_test_distributed_environment(tp_size, pp_size, rank, distributed_init_port)

    test_dict = {
        # device tensor
        "a": torch.arange(8, dtype=torch.float32, device="cuda"),
        # CPU tensor
        "b": torch.arange(16, dtype=torch.int8, device="cpu"),
        "c": "test",
        "d": [1, 2, 3],
        "e": {"a": 1, "b": 2},
        # empty tensor
        "f": torch.tensor([], dtype=torch.float32, device="cuda"),
    }

    if not get_pp_group().is_first_rank:
        recv_dict = get_pp_group().recv_tensor_dict()

    if not get_pp_group().is_last_rank:
        get_pp_group().send_tensor_dict(test_dict)

    if not get_pp_group().is_first_rank:
        assert len(recv_dict) == len(test_dict)
        torch.testing.assert_close(recv_dict["a"], test_dict["a"])
        torch.testing.assert_close(recv_dict["b"], test_dict["b"])
        assert recv_dict["c"] == test_dict["c"]
        assert recv_dict["d"] == test_dict["d"]
        assert recv_dict["e"] == test_dict["e"]
        torch.testing.assert_close(recv_dict["f"], test_dict["f"])


class _DummyWork:
    def __init__(self) -> None:
        self.wait_calls = 0

    def wait(self) -> None:
        self.wait_calls += 1


class _DummyAllGatherGroup:
    def __init__(self, world_size: int, rank_in_group: int) -> None:
        self.world_size = world_size
        self.rank_in_group = rank_in_group

    def all_gather(self, t: torch.Tensor, dim: int = 0) -> torch.Tensor:
        # duplicate local slice across ranks.
        assert dim == 0
        return torch.cat([t for _ in range(self.world_size)], dim=0)


def _make_group_for_unit_test(
    rank_in_group: int = 0, world_size: int = 2
) -> GroupCoordinator:
    # avoid running GroupCoordinator.__init__ (it wires up real process groups).
    g = GroupCoordinator.__new__(GroupCoordinator)
    g.world_size = world_size
    g.rank_in_group = rank_in_group
    g.ranks = list(range(world_size))
    g.use_cpu_custom_send_recv = False
    g.device_group = None
    g.cpu_group = None
    return g


def test_irecv_tensor_dict_send_allgather_postprocess_binds_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_irecv(t: torch.Tensor, *args: Any, **kwargs: Any) -> _DummyWork:
        t.fill_(1)
        return _DummyWork()

    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "irecv", fake_irecv)

    g = _make_group_for_unit_test(rank_in_group=0, world_size=2)
    # 2 tensors so we can catch late-binding bugs in postprocess closures.
    metadata_list = [
        ("a", TensorMetadata("cpu", torch.int32, torch.Size([4]))),
        ("b", TensorMetadata("cpu", torch.int32, torch.Size([4]))),
    ]
    g.recv_object = lambda src=None: metadata_list  # type: ignore[method-assign]

    ag = _DummyAllGatherGroup(world_size=2, rank_in_group=0)
    td, handles, postprocess = g.irecv_tensor_dict(all_gather_group=ag)

    assert td is not None
    assert len(handles) == 2
    assert len(postprocess) == 2

    # before postprocess, dict holds the TP slice (shape 2).
    assert td["a"].shape == torch.Size([2])
    assert td["b"].shape == torch.Size([2])

    # simulate worker-side "defer wait": wait + postprocess later.
    for handle in handles:
        handle.wait()
    for fn in postprocess:
        fn()

    # after postprocess, dict values are reconstructed to full shape (shape 4),
    # and each key should be updated independently
    assert td["a"].shape == torch.Size([4])
    assert td["b"].shape == torch.Size([4])
    torch.testing.assert_close(td["a"], torch.ones(4, dtype=torch.int32))
    torch.testing.assert_close(td["b"], torch.ones(4, dtype=torch.int32))


@pytest.mark.parametrize("aliased", [False, True])
def test_cuda_communicator_checkpoints_flashinfer_workspaces(
    monkeypatch: pytest.MonkeyPatch,
    aliased: bool,
) -> None:
    group = object()
    normal_workspace = Mock()
    quant_workspace = normal_workspace if aliased else Mock()
    unique_workspaces = (
        [normal_workspace] if aliased else [normal_workspace, quant_workspace]
    )

    monkeypatch.setattr(flashinfer_all_reduce, "_fi_ar_workspace", normal_workspace)
    monkeypatch.setattr(
        flashinfer_all_reduce, "_fi_ar_quant_workspace", quant_workspace
    )
    monkeypatch.setattr(
        flashinfer_all_reduce,
        "_fi_ar_workspace_groups",
        {id(workspace): group for workspace in unique_workspaces},
    )
    monkeypatch.setattr(
        flashinfer_all_reduce, "TorchDistBackend", lambda group: group, raising=False
    )

    communicator = CudaCommunicator.__new__(CudaCommunicator)
    communicator.cpu_group = group
    communicator.fi_ar_comm = None
    communicator.all2all_manager = None
    communicator.checkpoint_prepare()
    communicator.checkpoint_restore()

    for workspace in unique_workspaces:
        workspace.checkpoint_prepare.assert_called_once_with()
        workspace.checkpoint_restore.assert_called_once_with(group)


def _fabric_probe_must_not_run(device: int) -> bool:
    raise AssertionError("fabric probe must not be consulted on this path")


def _patch_fi_ar_module(
    monkeypatch: pytest.MonkeyPatch,
    node_count: int,
    fabric_supported: Callable[[int], bool],
    same_node: list[bool] | None = None,
) -> tuple[Mock, Mock]:
    # patch flashinfer_all_reduce for CPU-only _create_workspace tests; returns
    # the fake flashinfer_comm and the all_ranks_support_mnnvl vote mock.
    fake_comm = Mock()
    fake_comm.create_allreduce_fusion_workspace.return_value.mc_ptr = 1
    vote = Mock(
        side_effect=lambda local_supported, world_size, comm_backend: local_supported
    )
    monkeypatch.setattr(
        flashinfer_all_reduce, "flashinfer_comm", fake_comm, raising=False
    )
    monkeypatch.setattr(
        flashinfer_all_reduce, "TorchDistBackend", lambda group: group, raising=False
    )
    monkeypatch.setattr(flashinfer_all_reduce, "_fi_ar_workspace_groups", {})
    monkeypatch.setattr(
        flashinfer_all_reduce, "_mnnvl_supported_groups", {}, raising=False
    )
    monkeypatch.setattr(flashinfer_all_reduce, "get_node_count", lambda: node_count)
    monkeypatch.setattr(
        flashinfer_all_reduce,
        "in_the_same_node_as",
        lambda group: same_node if same_node is not None else [True, False],
        raising=False,
    )
    monkeypatch.setattr(
        flashinfer_all_reduce,
        "is_mnnvl_fabric_supported",
        fabric_supported,
        raising=False,
    )
    monkeypatch.setattr(
        flashinfer_all_reduce, "all_ranks_support_mnnvl", vote, raising=False
    )
    monkeypatch.setattr(torch.accelerator, "current_device_index", lambda: 0)
    return fake_comm, vote


def _create_mnnvl_workspace(group: object | None = None):
    return flashinfer_all_reduce._create_workspace(
        backend="mnnvl",
        world_size=4,
        rank=0,
        max_token_num=128,
        hidden_dim=64,
        dtype=torch.bfloat16,
        group=group if group is not None else object(),
    )


def test_create_workspace_skips_mnnvl_when_multinode_fabric_unsupported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # on IB-only multi-node the creation attempt itself hangs and leaks
    # (#51986), so the fabric must be probed before it is ever made.
    group = object()
    fake_comm, vote = _patch_fi_ar_module(
        monkeypatch, node_count=2, fabric_supported=lambda device: False
    )

    assert _create_mnnvl_workspace(group) is None
    fake_comm.create_allreduce_fusion_workspace.assert_not_called()
    # TorchDistBackend is identity-patched: the vote must get this group's
    # comm backend.
    assert vote.call_args.args == (False, 4, group)


def test_create_workspace_attempts_mnnvl_when_multinode_fabric_supported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_comm, _ = _patch_fi_ar_module(
        monkeypatch, node_count=2, fabric_supported=lambda device: True
    )

    workspace = _create_mnnvl_workspace()

    assert workspace is fake_comm.create_allreduce_fusion_workspace.return_value
    fake_comm.create_allreduce_fusion_workspace.assert_called_once()


def test_create_workspace_skips_fabric_probe_on_single_node(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # single-node mnnvl uses NVSwitch multicast, not the NVLink fabric.
    fake_comm, _ = _patch_fi_ar_module(
        monkeypatch, node_count=1, fabric_supported=_fabric_probe_must_not_run
    )

    workspace = _create_mnnvl_workspace()

    assert workspace is fake_comm.create_allreduce_fusion_workspace.return_value
    fake_comm.create_allreduce_fusion_workspace.assert_called_once()


def test_create_workspace_skips_fabric_probe_for_intra_node_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # a group confined to one node (e.g. TP=8 in a 2-node DP job) works via
    # node-local handle exchange without an NVLink fabric.
    fake_comm, _ = _patch_fi_ar_module(
        monkeypatch,
        node_count=2,
        fabric_supported=_fabric_probe_must_not_run,
        same_node=[True, True, True, True],
    )

    workspace = _create_mnnvl_workspace()

    assert workspace is fake_comm.create_allreduce_fusion_workspace.return_value
    fake_comm.create_allreduce_fusion_workspace.assert_called_once()


def test_create_workspace_joins_collective_vote_when_local_probe_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # a rank whose probe fails must still vote (as unsupported), otherwise
    # the other ranks deadlock in the allgather.
    def broken_probe(device: int) -> bool:
        raise RuntimeError("NVML unavailable")

    fake_comm, vote = _patch_fi_ar_module(
        monkeypatch, node_count=2, fabric_supported=broken_probe
    )

    assert _create_mnnvl_workspace() is None
    vote.assert_called_once()
    assert vote.call_args.args[0] is False
    fake_comm.create_allreduce_fusion_workspace.assert_not_called()


def test_create_workspace_memoizes_negative_mnnvl_vote(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # creation is retried on every call after a failure, so the negative vote
    # must be latched to keep probes/allgathers out of the per-layer hot path.
    group = object()
    _, vote = _patch_fi_ar_module(
        monkeypatch, node_count=2, fabric_supported=lambda device: False
    )

    assert _create_mnnvl_workspace(group) is None
    assert _create_mnnvl_workspace(group) is None
    vote.assert_called_once()


def test_async_intermediate_tensors_lazy_wait() -> None:
    work = _DummyWork()
    post_calls = {"n": 0}

    def post() -> None:
        post_calls["n"] += 1

    it = AsyncIntermediateTensors(
        {"x": torch.tensor([1])},
        comm_handles=[work],
        comm_postprocess=[post],
    )

    # accessing non-tensor attributes should not trigger wait.
    assert it._comm_handles is not None
    assert work.wait_calls == 0
    assert post_calls["n"] == 0

    # first access of `.tensors` triggers wait + postprocess.
    _ = it.tensors
    assert work.wait_calls == 1
    assert post_calls["n"] == 1

    # subsequent access should not re-wait.
    _ = it.tensors
    assert work.wait_calls == 1
    assert post_calls["n"] == 1


@ray.remote(num_gpus=1, max_calls=1)
def send_recv_test_worker(
    monkeypatch: pytest.MonkeyPatch,
    tp_size: int,
    pp_size: int,
    rank: int,
    distributed_init_port: str,
):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    device = torch.device(f"cuda:{rank}")
    torch.accelerator.set_device_index(device)
    init_test_distributed_environment(tp_size, pp_size, rank, distributed_init_port)

    size = 64
    test_tensor = torch.arange(64, dtype=torch.float32, device="cuda")

    if not get_pp_group().is_first_rank:
        recv_tensor = get_pp_group().recv(size, dtype=torch.float32)

    if not get_pp_group().is_last_rank:
        get_pp_group().send(test_tensor)

    if not get_pp_group().is_first_rank:
        torch.testing.assert_close(test_tensor, recv_tensor)


@multi_gpu_test(num_gpus=2)
@pytest.mark.parametrize("tp_size", [2])
@pytest.mark.parametrize(
    "test_target",
    [all_reduce_test_worker, all_gather_test_worker, broadcast_tensor_dict_test_worker],
)
def test_multi_process_tensor_parallel(
    monkeypatch: pytest.MonkeyPatch,
    tp_size: int,
    test_target: Callable[..., Any],
):
    multi_process_parallel(monkeypatch, tp_size, 1, test_target)


@multi_gpu_test(num_gpus=2)
@pytest.mark.parametrize("pp_size", [2])
@pytest.mark.parametrize(
    "test_target", [send_recv_test_worker, send_recv_tensor_dict_test_worker]
)
def test_multi_process_pipeline_parallel(
    monkeypatch: pytest.MonkeyPatch,
    pp_size: int,
    test_target: Callable[..., Any],
):
    multi_process_parallel(monkeypatch, 1, pp_size, test_target)


@multi_gpu_test(num_gpus=4)
@pytest.mark.parametrize("tp_size", [2])
@pytest.mark.parametrize("pp_size", [2])
@pytest.mark.parametrize(
    "test_target",
    [
        send_recv_test_worker,
        send_recv_tensor_dict_test_worker,
        all_reduce_test_worker,
        all_gather_test_worker,
        broadcast_tensor_dict_test_worker,
    ],
)
def test_multi_process_tensor_parallel_pipeline_parallel(
    tp_size: int,
    pp_size: int,
    test_target: Callable[..., Any],
    monkeypatch: pytest.MonkeyPatch,
):
    multi_process_parallel(monkeypatch, tp_size, pp_size, test_target)
