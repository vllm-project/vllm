# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm.distributed.device_communicators import cuda_communicator
from vllm.distributed.device_communicators import custom_all_reduce as custom_ar
from vllm.distributed.device_communicators import (
    flashinfer_pcie_ipc_all_reduce as pcie_ipc,
)
from vllm.distributed.parallel_state import GroupCoordinator


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


@pytest.mark.parametrize(
    ("missing_local_peer", "other_rank_ok", "expected_disabled"),
    [
        (None, True, False),  # full peer access: keep PCIe IPC
        (2, True, True),  # this rank cannot reach rank 2 (P2P only per switch pair)
        (None, False, True),  # another rank lacks peer access: all ranks fall back
    ],
)
def test_init_falls_back_without_peer_access(
    monkeypatch, missing_local_peer, other_rank_ok, expected_disabled
):
    """Hosts that expose P2P only within a PCIe-switch pair must fall back to
    another all-reduce backend instead of failing in the IPC workspace."""
    monkeypatch.setattr(pcie_ipc, "_pcie_ipc_available", True)
    monkeypatch.setattr(pcie_ipc.current_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(pcie_ipc.dist, "get_world_size", lambda group: 4)
    monkeypatch.setattr(pcie_ipc.dist, "get_rank", lambda group: 0)
    monkeypatch.setattr(
        pcie_ipc, "in_the_same_node_as", lambda *args, **kwargs: [True] * 4
    )
    monkeypatch.setattr(custom_ar.envs, "VLLM_SKIP_P2P_CHECK", False, raising=False)
    monkeypatch.setattr(
        custom_ar, "gpu_p2p_access_check", lambda src, tgt: tgt != missing_local_peer
    )

    def fake_all_reduce(tensor, op, group):
        # MIN reduction over this rank's vote and one remote rank's vote.
        tensor.fill_(min(int(tensor.item()), int(other_rank_ok)))

    monkeypatch.setattr(pcie_ipc.dist, "all_reduce", fake_all_reduce)

    comm = pcie_ipc.FlashInferPcieIpcAllReduce(
        group=object(), tune_group=object(), device="cuda:0"
    )

    assert comm.disabled is expected_disabled
