# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import ray
import torch

from vllm.distributed.device_communicators import cuda_communicator
from vllm.distributed.device_communicators import (
    flashinfer_pcie_ipc_ag_rs as pcie_ag_rs,
)
from vllm.distributed.device_communicators import (
    flashinfer_pcie_ipc_all_reduce as pcie_ipc,
)
from vllm.distributed.parallel_state import GroupCoordinator, get_tp_group

from ..utils import (
    init_test_distributed_environment,
    multi_gpu_test,
    multi_process_parallel,
)


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


def _uninitialized_ag_rs() -> pcie_ag_rs.FlashInferPcieIpcAgRs:
    comm = pcie_ag_rs.FlashInferPcieIpcAgRs.__new__(pcie_ag_rs.FlashInferPcieIpcAgRs)
    comm.disabled = False
    comm.group = object()
    comm.tune_group = object()
    comm.device = torch.device("cuda:0")
    comm.world_size = 8
    comm.rank = 0
    comm.ag_workspace = None
    comm.rs_workspace = None
    comm.hidden_dim = 4096
    comm.dtype = torch.bfloat16
    return comm


def test_ag_rs_setup_uses_exact_graph_capacity(monkeypatch, tmp_path):
    ag_workspace = Mock()
    rs_workspace = Mock()
    ag_factory = Mock(return_value=ag_workspace)
    rs_factory = Mock(return_value=rs_workspace)
    monkeypatch.setattr(
        pcie_ag_rs,
        "flashinfer_comm",
        SimpleNamespace(
            PcieIpcAllGatherWorkspace=ag_factory,
            PcieIpcReduceScatterWorkspace=rs_factory,
        ),
    )
    monkeypatch.setattr(torch.accelerator, "synchronize", Mock())
    monkeypatch.setattr(pcie_ag_rs, "_all_ranks_true", lambda _group, value: value)

    comm = _uninitialized_ag_rs()
    cache = tmp_path / "ep.json"
    comm.setup(
        hidden_dim=4096,
        dtype=torch.bfloat16,
        capture_sizes=[32, 8, 16, 32],
        tune_cache=cache,
    )

    common = dict(
        group=comm.group,
        max_numel=32 * 4096,
        dtype=torch.bfloat16,
        tune_batches=(8, 16, 32),
    )
    ag_factory.assert_called_once_with(
        **common, tune_cache=str(tmp_path / "ag_ep.json")
    )
    rs_factory.assert_called_once_with(
        **common, tune_cache=str(tmp_path / "rs_ep.json")
    )
    ag_workspace.tune.assert_called_once_with([4096, 2048])
    rs_workspace.tune.assert_called_once_with([4096])
    assert ag_workspace.rebind_stream.call_count == 2
    assert rs_workspace.rebind_stream.call_count == 2


def _unit_cuda_communicator() -> cuda_communicator.CudaCommunicator:
    communicator = cuda_communicator.CudaCommunicator.__new__(
        cuda_communicator.CudaCommunicator
    )
    communicator.world_size = 2
    communicator.rank_in_group = 0
    communicator.use_aiter_ag_rs = False
    communicator.pynccl_comm = Mock(disabled=False, nccl_version=22902)
    communicator.fi_pcie_ipc_ag_rs_comm = Mock()
    return communicator


def test_uniform_mixed_all_gather_selects_pcie_per_dtype(monkeypatch):
    communicator = _unit_cuda_communicator()
    monkeypatch.setattr(cuda_communicator, "should_nccl_symm_mem_ag_rs", lambda: False)
    hidden = torch.ones(2, 8, dtype=torch.bfloat16)
    expert_ids = torch.ones(2, 2, dtype=torch.int64)
    pcie_comm = communicator.fi_pcie_ipc_ag_rs_comm
    pcie_comm.should_use_ag.side_effect = lambda inp: inp.dtype == torch.bfloat16
    pcie_comm.all_gather.side_effect = lambda inp: ("pcie", inp)

    gathered = communicator.all_gatherv([hidden, expert_ids], sizes=[2, 2])

    assert gathered[0][0] == "pcie"
    assert gathered[0][1] is hidden
    assert gathered[1].shape == (4, 2)
    pcie_comm.all_gather.assert_called_once_with(hidden)
    assert communicator.pynccl_comm.all_gather.call_count == 1


def test_ragged_all_gather_stays_on_nccl(monkeypatch):
    communicator = _unit_cuda_communicator()
    monkeypatch.setattr(cuda_communicator, "should_nccl_symm_mem_ag_rs", lambda: False)

    gathered = communicator.all_gatherv(
        torch.ones(1, 8, dtype=torch.bfloat16), sizes=[1, 2]
    )

    assert gathered.shape == (3, 8)
    communicator.fi_pcie_ipc_ag_rs_comm.all_gather.assert_not_called()
    assert communicator.pynccl_comm.all_gatherv.call_count == 1


@pytest.mark.parametrize("preallocate", [False, True])
def test_uniform_reduce_scatter_uses_pcie(monkeypatch, preallocate):
    communicator = _unit_cuda_communicator()
    monkeypatch.setattr(cuda_communicator, "should_nccl_symm_mem_ag_rs", lambda: False)
    expected = torch.empty(2, 8, dtype=torch.bfloat16)
    pcie_comm = communicator.fi_pcie_ipc_ag_rs_comm
    pcie_comm.should_use_rs.return_value = True
    pcie_comm.reduce_scatter.return_value = expected
    inp = torch.ones(4, 8, dtype=torch.bfloat16)

    if preallocate:
        reduced = communicator.reduce_scatterv_into_output(
            inp, expected, dim=0, sizes=[2, 2]
        )
    else:
        reduced = communicator.reduce_scatterv(inp, dim=0, sizes=[2, 2])

    assert reduced.data_ptr() == expected.data_ptr()
    assert pcie_comm.reduce_scatter.call_args.args[1] is (
        expected if preallocate else None
    )


def test_ragged_reduce_scatter_stays_on_nccl(monkeypatch):
    communicator = _unit_cuda_communicator()
    monkeypatch.setattr(cuda_communicator, "should_nccl_symm_mem_ag_rs", lambda: False)

    reduced = communicator.reduce_scatterv(
        torch.ones(4, 8, dtype=torch.bfloat16), dim=0, sizes=[1, 3]
    )

    assert reduced.shape == (1, 8)
    communicator.fi_pcie_ipc_ag_rs_comm.reduce_scatter.assert_not_called()
    assert communicator.pynccl_comm.reduce_scatterv.call_count == 1


def test_ag_rs_capture_rebinds_both_on_error(monkeypatch):
    comm = _uninitialized_ag_rs()
    comm.ag_workspace = Mock()
    comm.rs_workspace = Mock()
    synchronize = Mock()
    monkeypatch.setattr(torch.accelerator, "synchronize", synchronize)

    with pytest.raises(RuntimeError, match="capture failed"), comm.capture():
        raise RuntimeError("capture failed")

    assert synchronize.call_count == 2
    assert comm.ag_workspace.rebind_stream.call_count == 2
    assert comm.rs_workspace.rebind_stream.call_count == 2


def test_ag_rs_destroy_releases_both():
    comm = _uninitialized_ag_rs()
    comm.ag_workspace = Mock()
    comm.rs_workspace = Mock()
    ag_workspace = comm.ag_workspace
    rs_workspace = comm.rs_workspace

    comm.destroy()

    ag_workspace.destroy.assert_called_once_with()
    rs_workspace.destroy.assert_called_once_with()
    assert comm.ag_workspace is None
    assert comm.rs_workspace is None


@ray.remote(num_gpus=1, max_calls=1)
def _pcie_ipc_ag_rs_graph_worker(
    monkeypatch, tp_size, pp_size, rank, distributed_init_port
):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    torch.accelerator.set_device_index(torch.device(f"cuda:{rank}"))
    init_test_distributed_environment(tp_size, pp_size, rank, distributed_init_port)
    group = get_tp_group()
    hidden = 64
    tune_cache = Path("/tmp") / f"pcie_ag_rs_{distributed_init_port}.json"
    comm = pcie_ag_rs.FlashInferPcieIpcAgRs(
        group=group.device_group, tune_group=group.cpu_group, device=rank
    )
    try:
        comm.setup(
            hidden_dim=hidden,
            dtype=torch.bfloat16,
            capture_sizes=(2,),
            tune_cache=tune_cache,
        )
        ag_input = (
            torch.arange(2 * hidden, device="cuda", dtype=torch.int32)
            .reshape(2, hidden)
            .to(torch.bfloat16)
        )
        rs_input = (
            torch.arange(4 * hidden, device="cuda", dtype=torch.int32)
            .reshape(4, hidden)
            .to(torch.bfloat16)
        )
        fp8_input = (
            torch.arange(2 * hidden, device="cuda", dtype=torch.uint8)
            .reshape(2, hidden)
            .view(torch.float8_e4m3fn)
        )
        expected_ag = torch.empty(4, hidden, dtype=torch.bfloat16, device="cuda")
        expected_rs = torch.empty(2, hidden, dtype=torch.bfloat16, device="cuda")
        expected_fp8 = torch.empty(4, hidden, dtype=torch.float8_e4m3fn, device="cuda")
        torch.distributed.all_gather_into_tensor(expected_ag, ag_input)
        torch.distributed.all_gather_into_tensor(
            expected_fp8.view(torch.bfloat16), fp8_input.view(torch.bfloat16)
        )
        torch.distributed.reduce_scatter_tensor(expected_rs, rs_input)

        graph = torch.cuda.CUDAGraph()
        with comm.capture(), torch.cuda.graph(graph):
            actual_ag = comm.all_gather(ag_input)
            actual_rs = comm.reduce_scatter(rs_input)
            actual_fp8 = comm.all_gather(fp8_input)
        graph.replay()
        torch.accelerator.synchronize()

        assert torch.equal(actual_ag, expected_ag)
        assert torch.equal(actual_rs, expected_rs)
        assert torch.equal(actual_fp8.view(torch.uint8), expected_fp8.view(torch.uint8))
    finally:
        comm.destroy()
        torch.distributed.barrier()
        if rank == 0:
            for prefix in ("ag_", "rs_"):
                tune_cache.with_name(f"{prefix}{tune_cache.name}").unlink(
                    missing_ok=True
                )


@multi_gpu_test(num_gpus=2)
def test_pcie_ipc_ag_rs_cuda_graph(monkeypatch):
    multi_process_parallel(monkeypatch, 2, 1, _pcie_ipc_ag_rs_graph_worker)


def test_ag_rs_warmup_selects_dp_group_for_dp_ep_pcp1(monkeypatch, tmp_path):
    from vllm.distributed import parallel_state
    from vllm.model_executor.warmup import flashinfer_autotune_cache

    communicator = cuda_communicator.CudaCommunicator.__new__(
        cuda_communicator.CudaCommunicator
    )
    communicator.fi_pcie_ipc_ag_rs_comm = Mock(disabled=False)
    group = SimpleNamespace(
        device_communicator=communicator, ranks=list(range(8)), unique_name="dp:0"
    )
    worker = SimpleNamespace(
        vllm_config=SimpleNamespace(
            parallel_config=SimpleNamespace(
                enable_expert_parallel=True, use_ubatching=False
            ),
            compilation_config=SimpleNamespace(cudagraph_capture_sizes=[1, 3]),
        ),
        model_runner=object(),
        model_config=SimpleNamespace(
            get_hidden_size=lambda: 5120, dtype=torch.bfloat16
        ),
    )
    monkeypatch.setattr(parallel_state, "get_dp_group", lambda: group)
    monkeypatch.setattr(
        parallel_state,
        "get_ep_group",
        Mock(side_effect=AssertionError("must not select EP here")),
    )
    monkeypatch.setattr(
        parallel_state, "get_pcp_group", lambda: SimpleNamespace(world_size=1)
    )
    monkeypatch.setattr(
        flashinfer_autotune_cache,
        "resolve_flashinfer_autotune_file",
        lambda _runner: tmp_path / "cache.json",
    )

    pcie_ag_rs.warmup_flashinfer_pcie_ipc_ag_rs(worker)

    communicator.fi_pcie_ipc_ag_rs_comm.setup.assert_called_once_with(
        hidden_dim=5120,
        dtype=torch.bfloat16,
        capture_sizes=[1, 3],
        tune_cache=tmp_path / "pcie_ipc_ag_rs_dp_0_0-1-2-3-4-5-6-7.json",
    )


def test_fp8_all_gather_uses_bf16_view_without_copying_payload():
    comm = _uninitialized_ag_rs()
    comm.hidden_dim = 8
    comm.ag_workspace = Mock()
    comm.ag_workspace.supports.return_value = True
    inp = torch.empty((2, 8), dtype=torch.float8_e4m3fn, device="cuda")
    expected = torch.empty((16, 4), dtype=torch.bfloat16, device="cuda")
    comm.ag_workspace.all_gather.return_value = expected

    assert comm.should_use_ag(inp)
    actual = comm.all_gather(inp)

    candidate = comm.ag_workspace.all_gather.call_args.args[0]
    assert candidate.dtype == torch.bfloat16
    assert candidate.shape == (2, 4)
    assert candidate.data_ptr() == inp.data_ptr()
    assert actual.data_ptr() == expected.data_ptr()
    assert actual.dtype == inp.dtype
    assert actual.shape == (16, 8)


@pytest.mark.parametrize("p2p_ok", [False, True])
def test_ag_rs_constructor_requires_group_p2p(monkeypatch, p2p_ok):
    monkeypatch.setattr(pcie_ag_rs, "_pcie_ipc_available", True)
    monkeypatch.setattr(pcie_ag_rs.dist, "get_world_size", lambda _group: 2)
    monkeypatch.setattr(pcie_ag_rs.dist, "get_rank", lambda _group: 0)
    monkeypatch.setattr(pcie_ag_rs.current_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(
        pcie_ag_rs, "in_the_same_node_as", lambda _group, source_rank: [True, True]
    )
    monkeypatch.setattr(pcie_ag_rs, "_can_p2p", lambda _rank, _world: p2p_ok)
    monkeypatch.setattr(pcie_ag_rs, "_all_ranks_true", lambda _group, local: local)

    comm = pcie_ag_rs.FlashInferPcieIpcAgRs(object(), object(), "cuda:0")

    assert comm.disabled is not p2p_ok


def test_ag_rs_setup_reuses_only_complete_matching_cache(monkeypatch, tmp_path):
    ag_workspace = Mock()
    rs_workspace = Mock()
    monkeypatch.setattr(
        pcie_ag_rs,
        "flashinfer_comm",
        SimpleNamespace(
            PcieIpcAllGatherWorkspace=Mock(return_value=ag_workspace),
            PcieIpcReduceScatterWorkspace=Mock(return_value=rs_workspace),
        ),
    )
    monkeypatch.setattr(torch.accelerator, "synchronize", Mock())
    monkeypatch.setattr(pcie_ag_rs, "_all_ranks_true", lambda _group, value: value)
    cache = tmp_path / "ep.json"
    kwargs = dict(
        hidden_dim=4096,
        dtype=torch.bfloat16,
        capture_sizes=[8, 16, 32],
        tune_cache=cache,
    )

    _uninitialized_ag_rs().setup(**kwargs)
    cache.with_name("ag_ep.json").write_text("{}")
    cache.with_name("rs_ep.json").write_text("{}")
    _uninitialized_ag_rs().setup(**kwargs)
    assert ag_workspace.tune.call_count == 1
    assert rs_workspace.tune.call_count == 1

    _uninitialized_ag_rs().setup(**{**kwargs, "capture_sizes": [8, 16, 32, 64]})
    assert ag_workspace.tune.call_count == 2
    assert rs_workspace.tune.call_count == 2
