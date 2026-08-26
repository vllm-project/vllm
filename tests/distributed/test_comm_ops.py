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


def _mock_flashinfer_allreduce(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(flashinfer_all_reduce, "fi_ar_available", True)
    monkeypatch.setattr(
        flashinfer_all_reduce.current_platform, "is_cuda", lambda: True
    )
    monkeypatch.setattr(
        flashinfer_all_reduce.dist, "get_world_size", lambda group: 8
    )
    monkeypatch.setattr(flashinfer_all_reduce.dist, "get_rank", lambda group: 0)


def test_flashinfer_allreduce_workspace_uses_integer_token_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _mock_flashinfer_allreduce(monkeypatch)
    monkeypatch.setattr(
        flashinfer_all_reduce, "get_current_vllm_config_or_none", lambda: None
    )
    monkeypatch.setattr(
        flashinfer_all_reduce.PassConfig,
        "default_fi_allreduce_fusion_max_size_mb",
        staticmethod(lambda: {8: 1.5}),
    )
    get_workspace = Mock(return_value=object())
    monkeypatch.setattr(
        flashinfer_all_reduce, "get_fi_ar_workspace", get_workspace
    )

    comm = flashinfer_all_reduce.FlashInferAllReduce(
        group=Mock(), device="cuda:0"
    )
    assert comm._ensure_workspace(hidden_dim=7168, dtype=torch.bfloat16)

    assert comm.max_workspace_size == int(1.5 * 1024 * 1024)
    assert comm.max_num_tokens == 109
    assert isinstance(comm.max_workspace_size, int)
    assert isinstance(comm.max_num_tokens, int)
    assert get_workspace.call_args.kwargs["max_token_num"] == comm.max_num_tokens


def test_flashinfer_allreduce_honors_pass_config_size_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _mock_flashinfer_allreduce(monkeypatch)
    pass_config = Mock()
    pass_config.flashinfer_max_size.return_value = 2 * 1024 * 1024
    config = Mock()
    config.compilation_config.pass_config = pass_config
    monkeypatch.setattr(
        flashinfer_all_reduce, "get_current_vllm_config_or_none", lambda: config
    )

    comm = flashinfer_all_reduce.FlashInferAllReduce(
        group=Mock(), device="cuda:0"
    )

    assert comm.max_workspace_size == 2 * 1024 * 1024
    assert isinstance(comm.max_workspace_size, int)
    assert pass_config.flashinfer_max_size.call_args.args == (8,)


def test_flashinfer_allreduce_accepts_two_mib_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    comm = object.__new__(flashinfer_all_reduce.FlashInferAllReduce)
    comm.disabled = False
    comm.world_size = 8
    comm.rank = 0
    comm.group = Mock()
    comm.max_workspace_size = 2 * 1024 * 1024
    comm.max_num_tokens = 0
    get_workspace = Mock(return_value=object())
    monkeypatch.setattr(
        flashinfer_all_reduce, "get_fi_ar_workspace", get_workspace
    )
    tensor = Mock(
        is_cuda=True,
        dtype=torch.bfloat16,
        nbytes=146 * 7168 * 2,
        shape=(146, 7168),
    )
    tensor.is_contiguous.return_value = True

    assert comm.should_use_fi_ar(tensor)
    assert comm.max_num_tokens == 146
    assert isinstance(comm.max_num_tokens, int)
    assert get_workspace.call_args.kwargs["max_token_num"] == 146


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({"is_cuda": False}, False),
        ({"is_contiguous": False}, False),
        ({"shape": (1, 2, 3)}, False),
        ({"dtype": torch.int8}, False),
        ({"nbytes": 2 * 1024 * 1024 + 1}, False),
    ],
)
def test_flashinfer_allreduce_rejects_ineligible_inputs(
    overrides: dict[str, object], expected: bool
) -> None:
    comm = object.__new__(flashinfer_all_reduce.FlashInferAllReduce)
    comm.disabled = False
    comm.max_workspace_size = 2 * 1024 * 1024
    comm.max_num_tokens = 0
    tensor = Mock(
        is_cuda=True,
        dtype=torch.bfloat16,
        nbytes=1024,
        shape=(1, 7168),
    )
    tensor.is_contiguous.return_value = True
    for name, value in overrides.items():
        if name == "is_contiguous":
            tensor.is_contiguous.return_value = value
        else:
            setattr(tensor, name, value)

    assert comm.should_use_fi_ar(tensor) is expected


def test_explicit_flashinfer_precedes_symmetric_memory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    communicator = object.__new__(CudaCommunicator)
    communicator.qr_comm = None
    communicator.fi_ar_comm = Mock(disabled=False)
    communicator.fi_ar_comm.should_use_fi_ar.return_value = True
    expected = torch.empty(1)
    communicator.fi_ar_comm.all_reduce.return_value = expected
    communicator.pynccl_comm = Mock(world_size=8, disabled=False)
    symmetric_selector = Mock(return_value=True)
    monkeypatch.setattr(
        "vllm.distributed.device_communicators.cuda_communicator."
        "should_nccl_symm_mem_allreduce",
        symmetric_selector,
    )

    result = communicator.all_reduce(torch.empty(128, 7168))

    assert result is expected
    communicator.fi_ar_comm.all_reduce.assert_called_once()
    symmetric_selector.assert_not_called()


@pytest.mark.parametrize("flashinfer_enabled", [False, True])
def test_flashinfer_ineligible_or_disabled_falls_back_to_pynccl(
    monkeypatch: pytest.MonkeyPatch, flashinfer_enabled: bool
) -> None:
    communicator = object.__new__(CudaCommunicator)
    communicator.qr_comm = None
    communicator.fi_ar_comm = (
        Mock(disabled=False) if flashinfer_enabled else None
    )
    if communicator.fi_ar_comm is not None:
        communicator.fi_ar_comm.should_use_fi_ar.return_value = False
    communicator.aiter_ar_comm = None
    communicator.ca_comm = None
    communicator.symm_mem_comm = None
    expected = torch.empty(1)
    communicator.pynccl_comm = Mock(world_size=8, disabled=False)
    communicator.pynccl_comm.all_reduce.return_value = expected
    monkeypatch.setattr(
        "vllm.distributed.device_communicators.cuda_communicator."
        "should_nccl_symm_mem_allreduce",
        lambda world_size, input_tensor: False,
    )

    result = communicator.all_reduce(torch.empty(128, 7168))

    assert result is expected
    communicator.pynccl_comm.all_reduce.assert_called_once()
