# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from tests.utils import multi_gpu_test
from vllm.distributed.device_communicators import cuda_communicator
from vllm.distributed.device_communicators import (
    flashinfer_pcie_ipc_all_reduce as pcie_ipc,
)
from vllm.distributed.parallel_state import GroupCoordinator, get_tp_group
from vllm.platforms import current_platform
from vllm.v1.worker.gpu_worker import Worker


def _uninitialized_comm() -> pcie_ipc.FlashInferPcieIpcAllReduce:
    comm = pcie_ipc.FlashInferPcieIpcAllReduce.__new__(
        pcie_ipc.FlashInferPcieIpcAllReduce
    )
    comm.disabled = False
    comm.group = object()
    comm.tune_group = object()
    comm.device = torch.device("cuda:0")
    comm.world_size = 4
    comm.rank = 0
    comm.workspace = None
    comm.hidden_dim = 0
    comm.dtype = None
    return comm


def test_setup_uses_exact_graph_capacity(monkeypatch, tmp_path):
    workspace = Mock()
    factory = Mock(return_value=workspace)
    monkeypatch.setattr(
        pcie_ipc,
        "flashinfer_comm",
        SimpleNamespace(PcieIpcAllReduceWorkspace=factory),
    )
    monkeypatch.setattr(torch.accelerator, "synchronize", Mock())

    comm = _uninitialized_comm()
    cache = tmp_path / "tp4.json"
    comm.setup(
        hidden_dim=4096,
        dtype=torch.bfloat16,
        capture_sizes=[32, 8, 16, 32],
        tune_cache=cache,
    )

    factory.assert_called_once_with(
        group=comm.group,
        max_numel=32 * 4096,
        dtype=torch.bfloat16,
        tune_batches=(8, 16, 32),
        tune_cache=str(cache),
    )
    workspace.tune.assert_called_once_with(
        [4096], dtype=torch.bfloat16, tune_group=comm.tune_group
    )
    workspace.prepare.assert_called_once_with(
        [(8, 4096), (16, 4096), (32, 4096)], dtype=torch.bfloat16
    )
    assert workspace.rebind_stream.call_count == 2


@pytest.mark.parametrize(
    ("shape", "dtype", "contiguous", "expected"),
    [
        ((16, 4096), torch.bfloat16, True, True),
        ((16, 2048), torch.bfloat16, True, False),
        ((16, 4096), torch.float16, True, False),
        ((16, 4096), torch.bfloat16, False, False),
    ],
)
def test_should_use_only_prepared_hidden_shape(shape, dtype, contiguous, expected):
    comm = _uninitialized_comm()
    comm.hidden_dim = 4096
    comm.dtype = torch.bfloat16
    comm.workspace = Mock()
    comm.workspace.supports.return_value = True
    inp = SimpleNamespace(
        is_cuda=True,
        is_contiguous=lambda: contiguous,
        dim=lambda: len(shape),
        shape=shape,
        dtype=dtype,
    )

    assert comm.should_use(inp) is expected


def test_capture_rebinds_workspace_even_on_error(monkeypatch):
    comm = _uninitialized_comm()
    comm.workspace = Mock()
    synchronize = Mock()
    monkeypatch.setattr(torch.accelerator, "synchronize", synchronize)

    with pytest.raises(RuntimeError, match="capture failed"), comm.capture():
        raise RuntimeError("capture failed")

    assert synchronize.call_count == 2
    assert comm.workspace.rebind_stream.call_count == 2


def test_cuda_dispatch_prefers_pcie_ipc_before_existing_flashinfer(monkeypatch):
    communicator = cuda_communicator.CudaCommunicator.__new__(
        cuda_communicator.CudaCommunicator
    )
    communicator.pynccl_comm = Mock(world_size=4)
    communicator.qr_comm = None
    communicator.fi_pcie_ipc_ar_comm = Mock()
    communicator.fi_pcie_ipc_ar_comm.should_use.return_value = True
    expected = object()
    communicator.fi_pcie_ipc_ar_comm.all_reduce.return_value = expected
    communicator.fi_ar_comm = Mock()
    communicator.aiter_ar_comm = None
    communicator.ca_comm = None
    communicator.symm_mem_comm = None
    monkeypatch.setattr(
        cuda_communicator, "should_nccl_symm_mem_allreduce", lambda *_: False
    )

    assert communicator.all_reduce(object()) is expected
    communicator.fi_ar_comm.should_use_fi_ar.assert_not_called()


def test_group_destroy_releases_communicator_before_process_groups(monkeypatch):
    events = []
    coordinator = GroupCoordinator.__new__(GroupCoordinator)
    coordinator.device_communicator = SimpleNamespace(
        destroy=lambda: events.append("communicator")
    )
    coordinator.device_group = object()
    coordinator.cpu_group = object()
    coordinator.mq_broadcaster = None
    monkeypatch.setattr(
        torch.distributed,
        "destroy_process_group",
        lambda _group: events.append("process_group"),
    )

    coordinator.destroy()

    assert events == ["communicator", "process_group", "process_group"]


class _PcieIpcProfilingWorker(Worker):
    def determine_available_memory(self) -> int:
        available = super().determine_available_memory()
        comm = get_tp_group().device_communicator.fi_pcie_ipc_ar_comm
        assert comm is not None and comm.initialized, (
            "PCIe IPC workspace must exist before KV cache sizing"
        )
        self.profiled_pcie_workspace = comm.workspace
        from flashinfer.comm.pcie_ipc_ar import get_pcie_ipc_comm_module

        workspace = comm.workspace
        slab_bytes = get_pcie_ipc_comm_module().workspace_size(
            comm.world_size,
            workspace.max_numel,
            workspace.elem_size,
            workspace.max_blocks,
        )
        assert self.total_consumed >= (
            int(self.model_runner.model_memory_usage) + slab_bytes
        ), "The IPC slab must be included in the profiled non-KV memory"
        return available

    def pcie_workspace_was_reused(self) -> bool:
        comm = get_tp_group().device_communicator.fi_pcie_ipc_ar_comm
        return comm.initialized and comm.workspace is self.profiled_pcie_workspace


@pytest.mark.skipif(
    not current_platform.is_cuda(), reason="FlashInfer PCIe IPC requires CUDA"
)
@pytest.mark.parametrize("extensible", [False, True])
@multi_gpu_test(num_gpus=2)
def test_workspace_is_profiled_before_kv_cache_sizing(
    monkeypatch, vllm_runner, extensible
):
    """KV sizing includes the IPC slab and capture reuses that same workspace."""
    from vllm import SamplingParams

    monkeypatch.setenv("VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC", "1")
    monkeypatch.setenv("VLLM_ENABLE_STARTUP_PLAN", "0")
    monkeypatch.setenv("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    with vllm_runner(
        "Qwen/Qwen3-0.6B",
        dtype="bfloat16",
        tensor_parallel_size=2,
        distributed_executor_backend="mp",
        worker_cls=f"{__name__}._PcieIpcProfilingWorker",
        max_model_len=128,
        max_num_seqs=4,
        max_num_batched_tokens=128,
        gpu_memory_utilization=0.3,
        enable_extensible_kv_cache=extensible,
        compilation_config={"cudagraph_capture_sizes": [1, 2, 4, 8, 16, 32, 64, 128]},
    ) as runner:
        assert runner.llm.collective_rpc("pcie_workspace_was_reused") == [True, True]
        outputs = runner.llm.generate(
            ["The capital of France is"],
            SamplingParams(temperature=0, max_tokens=4, ignore_eos=True),
        )
        assert len(outputs) == 1
        assert outputs[0].finished
        assert len(outputs[0].outputs[0].token_ids) == 4
