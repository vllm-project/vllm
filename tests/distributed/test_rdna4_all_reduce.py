# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import ast
import importlib.util
import multiprocessing as mp
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
import torch.distributed as dist

from vllm.distributed.device_communicators import rdna4_all_reduce as rdna4_module
from vllm.distributed.device_communicators.rdna4_all_reduce import (
    DEFAULT_MAX_SIZE,
    MAPPED_ALL_REDUCE_MAX_SIZE,
    P2P_GRAPH_MIN_SIZE,
    TP2_MAPPED_BF16_MAX_SIZE,
    TP4_EAGER_PYNCCL_SIZES,
    TP8_PYNCCL_GRAPH_RANGE,
    RDNA4AllReduce,
)
from vllm.platforms import current_platform
from vllm.utils.network_utils import get_open_port

from ..utils import multi_gpu_test


class _Delegate:
    def __init__(self, accepts, result):
        self.accepts = accepts
        self.result = result
        self.should_use_calls = 0
        self.all_reduce_calls = 0

    def should_use(self, _tensor):
        self.should_use_calls += 1
        return self.accepts

    def all_reduce(self, *_args, **_kwargs):
        self.all_reduce_calls += 1
        return self.result


def _bare_communicator(world_size, *, max_size=2 * 1024 * 1024):
    communicator = RDNA4AllReduce.__new__(RDNA4AllReduce)
    communicator.disabled = False
    communicator.world_size = world_size
    communicator.rank = 0
    communicator.device = torch.device("cpu")
    communicator.max_size = max_size
    communicator._tp2_mapped = None
    communicator._mapped = None
    communicator._p2p_ready = True
    communicator._IS_CAPTURING = False
    communicator.fully_connected = True
    return communicator


def _bf16_tensor(nbytes):
    assert nbytes % torch.bfloat16.itemsize == 0
    return torch.empty(nbytes // torch.bfloat16.itemsize, dtype=torch.bfloat16)


def test_transport_priority_is_tp2_mapped_then_mapped_then_p2p():
    tensor = _bf16_tensor(64)
    communicator = _bare_communicator(2)
    communicator._IS_CAPTURING = True
    tp2_mapped = _Delegate(True, "tp2_mapped")
    mapped = _Delegate(True, "mapped")
    communicator._tp2_mapped = tp2_mapped
    communicator._mapped = mapped

    assert communicator.custom_all_reduce(tensor) == "tp2_mapped"
    assert tp2_mapped.all_reduce_calls == 1
    assert mapped.should_use_calls == 0

    communicator._tp2_mapped = _Delegate(False, None)
    assert communicator.custom_all_reduce(tensor) == "mapped"
    assert mapped.all_reduce_calls == 1


@pytest.mark.parametrize(
    ("world_size", "nbytes", "expected"),
    [
        (4, P2P_GRAPH_MIN_SIZE[4] - 64, False),
        (4, P2P_GRAPH_MIN_SIZE[4], True),
        (2, P2P_GRAPH_MIN_SIZE[2] - 16, False),
        (2, P2P_GRAPH_MIN_SIZE[2], True),
        (8, MAPPED_ALL_REDUCE_MAX_SIZE[8], False),
        (8, MAPPED_ALL_REDUCE_MAX_SIZE[8] + 64, True),
        (8, TP8_PYNCCL_GRAPH_RANGE[0], False),
        (8, TP8_PYNCCL_GRAPH_RANGE[1], False),
        (8, TP8_PYNCCL_GRAPH_RANGE[1] + 64, True),
    ],
)
def test_p2p_policy_boundaries(world_size, nbytes, expected):
    communicator = _bare_communicator(world_size, max_size=2 * 1024 * 1024)
    tensor = _bf16_tensor(nbytes)
    assert communicator._p2p_tensor_supported(tensor) is expected
    assert communicator.should_use_graph(tensor) is expected
    assert not communicator.should_use(tensor)


@pytest.mark.parametrize("world_size", [4, 8])
def test_graph_dispatches_to_tp_specific_p2p_kernel(monkeypatch, world_size):
    communicator = _bare_communicator(world_size)
    communicator._IS_CAPTURING = True
    communicator._captured_outputs = []
    communicator._captured_inputs = []
    communicator._capture_base_index = 0
    communicator._gpu_graph_output_ptrs_array = torch.zeros((1, 8), dtype=torch.int64)
    communicator._gpu_graph_input_ptrs_array = torch.zeros((1, 8), dtype=torch.int64)
    communicator._select_route = MagicMock(return_value=rdna4_module._Route.P2P)
    communicator._run_p2p_tp4_pull = MagicMock(side_effect=lambda _, out, **__: out)
    communicator._run_p2p_tp4_push_rsag = MagicMock(
        side_effect=lambda _, out, **__: out
    )
    communicator._run_p2p_hierarchical_tp8 = MagicMock(
        side_effect=lambda _, out, **__: out
    )
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    tensor = _bf16_tensor(1024 * 1024)
    output = torch.empty_like(tensor)

    assert communicator.custom_all_reduce(tensor, out=output) is output
    if world_size == 4:
        communicator._run_p2p_tp4_pull.assert_called_once()
        communicator._run_p2p_tp4_push_rsag.assert_not_called()
        communicator._run_p2p_hierarchical_tp8.assert_not_called()
    else:
        communicator._run_p2p_hierarchical_tp8.assert_called_once()
        communicator._run_p2p_tp4_push_rsag.assert_not_called()


def test_measured_pynccl_routing_exclusions():
    assert DEFAULT_MAX_SIZE == 128 * 1024 * 1024

    tp2 = _bare_communicator(2, max_size=DEFAULT_MAX_SIZE)
    tp2._tp2_mapped = _Delegate(True, "tp2_mapped")
    small_tp2 = _bf16_tensor(TP2_MAPPED_BF16_MAX_SIZE)
    assert not tp2.should_use(small_tp2)
    assert tp2.custom_all_reduce(small_tp2) is None
    assert tp2.should_use_graph(small_tp2)
    tp2._tp2_mapped = _Delegate(False, "tp2_mapped")
    graph_only_tp2 = _bf16_tensor(P2P_GRAPH_MIN_SIZE[2])
    assert not tp2.should_use(graph_only_tp2)
    assert tp2.should_use_graph(graph_only_tp2)
    large_tp2 = _bf16_tensor(DEFAULT_MAX_SIZE)
    assert not tp2.should_use(large_tp2)
    assert tp2.should_use_graph(large_tp2)

    tp4 = _bare_communicator(4, max_size=DEFAULT_MAX_SIZE)
    tp4._mapped = _Delegate(True, "mapped")
    eager_exclusion = _bf16_tensor(next(iter(TP4_EAGER_PYNCCL_SIZES)))
    assert not tp4.should_use(eager_exclusion)
    assert tp4.custom_all_reduce(eager_exclusion) is None
    assert tp4.should_use_graph(eager_exclusion)

    tp4._mapped = _Delegate(False, "mapped")
    graph_boundary = _bf16_tensor(32 * 1024 * 1024 - 64)
    assert not tp4.should_use(graph_boundary)
    assert tp4.should_use_graph(graph_boundary)


@pytest.mark.parametrize(
    ("numel", "kind", "blocks"),
    [
        (65_536, "full", 1),
        (65_544, "full", 2),
        (196_608, "full", 2),
        (196_616, "full", 4),
        (786_432, "full", 4),
        (786_440, "pipeline", 8),
        (2_097_160, "pipeline", 10),
        (3_801_096, "pipeline", 12),
        (5_242_888, "pipeline", 16),
    ],
)
def test_tp2_policy_boundaries(numel, kind, blocks):
    pytest.importorskip("flydsl")
    from vllm.distributed.device_communicators.rdna4_all_reduce.mapped_tp2 import (
        RDNA4TP2MappedAllReduce,
    )

    selected_kind, selected_blocks, chunk_packs = RDNA4TP2MappedAllReduce._policy(
        numel, 1024
    )
    assert (selected_kind, selected_blocks) == (kind, blocks)
    assert chunk_packs == 0 if kind == "full" else chunk_packs > 0


def test_mapped_transport_selects_direct_then_rsag(monkeypatch):
    pytest.importorskip("flydsl")
    from vllm.distributed.device_communicators.rdna4_all_reduce import (
        mapped as mapped_module,
    )
    from vllm.distributed.device_communicators.rdna4_all_reduce.mapped import (
        RDNA4MappedAllReduce,
    )

    communicator = RDNA4MappedAllReduce.__new__(RDNA4MappedAllReduce)
    communicator.disabled = False
    communicator.device = torch.device("cpu")
    communicator.max_size = MAPPED_ALL_REDUCE_MAX_SIZE[8]
    communicator.world_size = 8
    communicator.rank = 0
    communicator.threads = 512
    communicator.pipeline_blocks = 16
    communicator.pipeline_min_numel = 65_536
    communicator._device_address = 0x1000
    communicator._slot_bytes = 384 * 1024
    communicator._full_launcher = MagicMock()
    rsag_launcher = MagicMock()
    monkeypatch.setattr(mapped_module, "Int32", lambda value: value)
    monkeypatch.setattr(mapped_module, "Int64", lambda value: value)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda _device: object())
    monkeypatch.setattr(
        mapped_module, "make_mapped_rsag_launcher", lambda **_kwargs: rsag_launcher
    )

    communicator.all_reduce(_bf16_tensor((65_536 - 8) * 2))
    communicator._full_launcher.assert_called_once()
    rsag_launcher.assert_not_called()

    communicator.all_reduce(_bf16_tensor(65_536 * 2))
    rsag_launcher.assert_called_once()


def test_missing_flydsl_disables_before_gpu_allocation(monkeypatch):
    monkeypatch.setattr(rdna4_module.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(rdna4_module.dist, "get_backend", lambda _group: "gloo")
    monkeypatch.setattr(rdna4_module.dist, "get_world_size", lambda _group: 4)
    monkeypatch.setattr(rdna4_module.dist, "get_rank", lambda _group: 0)
    monkeypatch.setattr(rdna4_module.current_platform, "is_rocm", lambda: True)
    monkeypatch.setattr("vllm.platforms.rocm.on_rdna4", lambda: True)
    monkeypatch.setattr(rdna4_module, "_is_rdna4_flydsl_available", lambda: False)
    get_properties = MagicMock()
    monkeypatch.setattr(torch.cuda, "get_device_properties", get_properties)

    communicator = RDNA4AllReduce(object(), torch.device("cuda:0"))

    assert communicator.disabled
    get_properties.assert_not_called()


@pytest.mark.parametrize("error", [ImportError, OSError, RuntimeError])
def test_flydsl_probe_fails_closed(monkeypatch, error):
    def unavailable(_name):
        raise error("FlyDSL is unavailable")

    monkeypatch.setattr(rdna4_module.importlib, "import_module", unavailable)
    assert not rdna4_module._is_rdna4_flydsl_available()


@pytest.mark.parametrize(
    "missing", [None, "generic_load", "generic_store", "AtomicOrdering"]
)
def test_flydsl_probe_accepts_generic_memory_api(monkeypatch, missing):
    fx = SimpleNamespace(
        AtomicOrdering=object(),
        PointerType=object(),
        inttoptr=object(),
        generic_load=object(),
        generic_store=object(),
        rocdl=SimpleNamespace(SyncScope=object()),
    )
    if missing is not None:
        delattr(fx, missing)
    monkeypatch.setattr(rdna4_module.importlib, "import_module", lambda _: fx)
    assert rdna4_module._is_rdna4_flydsl_available() is (missing is None)


def test_public_router_has_no_eager_flydsl_imports():
    assert Path(rdna4_module.__file__).name == "rdna4_all_reduce.py"
    assert RDNA4AllReduce.__module__ == rdna4_module.__name__
    tree = ast.parse(Path(rdna4_module.__file__).read_text())
    eager_flydsl_imports: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            eager_flydsl_imports.extend(
                alias.name for alias in node.names if alias.name.startswith("flydsl")
            )
        elif (
            isinstance(node, ast.ImportFrom)
            and node.module is not None
            and "flydsl" in node.module
        ):
            eager_flydsl_imports.append(node.module)
    assert eager_flydsl_imports == []


@pytest.mark.parametrize(
    ("rdna4", "aiter", "rdna4_disabled", "rdna4_raises"),
    [
        (True, True, False, False),
        (True, True, True, False),
        (True, True, False, True),
        (False, True, False, False),
        (False, False, False, False),
    ],
)
def test_communicator_initializes_only_architecture_compatible_backends(
    monkeypatch, rdna4, aiter, rdna4_disabled, rdna4_raises
):
    """Gate RDNA4/custom backends by architecture without excluding AITER."""
    from vllm.distributed.device_communicators import cuda_communicator as module

    def initialize(self, cpu_group, device, *args, **kwargs):
        self.cpu_group = cpu_group
        self.device = device
        self.device_group = None
        self.world_size = 2
        self.use_all2all = False

    monkeypatch.setattr(module.DeviceCommunicatorBase, "__init__", initialize)
    monkeypatch.setattr(module.current_platform, "is_rocm", lambda: True)
    monkeypatch.setattr(module.current_platform, "is_cuda", lambda: False)
    monkeypatch.setattr("vllm.platforms.rocm.on_rdna4", lambda: rdna4)
    monkeypatch.setattr(
        "vllm.distributed.parallel_state._ENABLE_CUSTOM_ALL_REDUCE", True
    )
    monkeypatch.setattr(
        module.rocm_aiter_ops, "is_custom_all_reduce_enabled", lambda: aiter
    )
    for name in (
        "VLLM_ALLREDUCE_USE_SYMM_MEM",
        "VLLM_ALLREDUCE_USE_FLASHINFER",
        "VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC",
    ):
        monkeypatch.setattr(module.envs, name, False)
    monkeypatch.setattr(module, "is_symmetric_memory_enabled", lambda: False)
    monkeypatch.setattr(
        module.CudaCommunicator, "_log_all_reduce_backend_selection", lambda _: None
    )
    prefix = "vllm.distributed.device_communicators."
    constructors = {}
    for name in (
        "pynccl.PyNcclCommunicator",
        "custom_all_reduce.CustomAllreduce",
        "quick_all_reduce.QuickAllReduce",
        "rdna4_all_reduce.RDNA4AllReduce",
    ):
        constructors[name] = MagicMock()
        monkeypatch.setattr(prefix + name, constructors[name])
    aiter_constructor = MagicMock()
    monkeypatch.setattr(module, "AiterCustomAllreduce", aiter_constructor)
    rdna4_constructor = constructors["rdna4_all_reduce.RDNA4AllReduce"]
    rdna4_constructor.return_value.disabled = rdna4_disabled
    if rdna4_raises:
        rdna4_constructor.side_effect = RuntimeError("initialization failed")

    communicator = module.CudaCommunicator(
        object(), torch.device("cuda:0"), unique_name="tp:0"
    )

    assert rdna4_constructor.call_count == int(rdna4)
    assert aiter_constructor.call_count == int(aiter)
    assert constructors["custom_all_reduce.CustomAllreduce"].call_count == int(
        not rdna4 and not aiter
    )
    assert constructors["quick_all_reduce.QuickAllReduce"].call_count == int(not rdna4)
    assert communicator.pynccl_comm is not None
    if rdna4_disabled or rdna4_raises or not rdna4:
        assert communicator.rdna4_ar_comm is None


@pytest.fixture
def rdna4_gpu():
    if not current_platform.is_rocm():
        pytest.skip("RDNA4 all-reduce requires ROCm")
    from vllm.platforms.rocm import on_rdna4

    if not on_rdna4() or importlib.util.find_spec("flydsl") is None:
        pytest.skip("RDNA4 all-reduce requires RDNA4 GPUs and FlyDSL")


def _worker(rank, world_size, port, element_counts, automatic=False):
    device = torch.device(f"cuda:{rank}")
    torch.accelerator.set_device_index(rank)
    dist.init_process_group(
        backend="gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
    )

    from vllm.distributed.device_communicators.rdna4_all_reduce import (
        RDNA4AllReduce,
    )

    cuda_communicator = None
    if automatic:
        from vllm.distributed.device_communicators.cuda_communicator import (
            CudaCommunicator,
        )

        os.environ.pop("VLLM_ROCM_USE_RDNA4_ALL_REDUCE", None)
        cuda_communicator = CudaCommunicator(
            cpu_group=dist.group.WORLD, device=device, unique_name="tp:0"
        )
        communicator = cuda_communicator.rdna4_ar_comm
        assert communicator is not None
    else:
        communicator = RDNA4AllReduce(
            group=dist.group.WORLD,
            device=device,
            max_size=max(element_counts) * torch.bfloat16.itemsize,
        )
    try:
        assert not communicator.disabled
        expected = world_size * (world_size + 1) / 2
        for numel in element_counts:
            inp = torch.full((numel,), rank + 1, dtype=torch.bfloat16, device=device)
            out = torch.empty_like(inp)
            graph = torch.cuda.CUDAGraph()
            torch.accelerator.synchronize()
            dist.barrier()
            with communicator.capture(), torch.cuda.graph(graph):
                if cuda_communicator is not None:
                    assert communicator.should_use(inp)
                    out = cuda_communicator.all_reduce(inp)
                else:
                    result = communicator.custom_all_reduce(inp, out=out)
                    assert result is out
            for _ in range(16):
                graph.replay()
            torch.accelerator.synchronize()
            torch.testing.assert_close(
                out,
                torch.full_like(out, expected),
                rtol=0,
                atol=0,
            )
    finally:
        torch.accelerator.synchronize()
        dist.barrier()
        if cuda_communicator is not None:
            cuda_communicator.destroy()
        else:
            communicator.close()
        dist.destroy_process_group()


def _run(world_size, element_counts, automatic=False):
    context = mp.get_context("spawn")
    port = get_open_port()
    processes = [
        context.Process(
            target=_worker,
            args=(rank, world_size, port, element_counts, automatic),
        )
        for rank in range(world_size)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=180)
        if process.is_alive():
            process.terminate()
            process.join(timeout=10)
            pytest.fail("RDNA4 all-reduce worker timed out")
        assert process.exitcode == 0, (
            f"RDNA4 all-reduce worker exited with code {process.exitcode}"
        )


@pytest.mark.usefixtures("rdna4_gpu")
@multi_gpu_test(num_gpus=2)
def test_rdna4_all_reduce_tp2_without_env():
    _run(2, (8, 128, 512, 2048, 4096, 8192, 16_384, 24_576, 32_768), automatic=True)


@pytest.mark.usefixtures("rdna4_gpu")
@multi_gpu_test(num_gpus=2)
def test_rdna4_all_reduce_tp2():
    _run(2, (8, 128, 512, 2048, 4096, 8192, 16_384, 24_576, 32_768))


@pytest.mark.usefixtures("rdna4_gpu")
@multi_gpu_test(num_gpus=2)
def test_rdna4_all_reduce_tp2_p2p():
    if not all(
        torch.cuda.can_device_access_peer(src, dst) for src, dst in ((0, 1), (1, 0))
    ):
        pytest.skip("TP2 direct-P2P requires bidirectional peer access")
    _run(2, (32_776, 524_288, 4_194_304))


@pytest.mark.usefixtures("rdna4_gpu")
@multi_gpu_test(num_gpus=4)
def test_rdna4_all_reduce_tp4():
    _run(4, (8, 65_536, 524_288))


@pytest.mark.usefixtures("rdna4_gpu")
@multi_gpu_test(num_gpus=8)
def test_rdna4_all_reduce_tp8():
    _run(8, (8, 65_536, 196_640))
