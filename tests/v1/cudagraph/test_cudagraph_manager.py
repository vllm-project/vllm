# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm.config import (
    CompilationConfig,
    CUDAGraphMode,
    ParallelConfig,
    SchedulerConfig,
    VllmConfig,
)
from vllm.distributed.device_communicators import pynccl_allocator
from vllm.v1.worker.gpu import cudagraph_utils as gpu_cudagraph_utils
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor

pytestmark = pytest.mark.cpu_test


@pytest.fixture(autouse=True)
def _reset_graph_pool_id():
    pynccl_allocator._graph_pool_id = None
    yield
    pynccl_allocator._graph_pool_id = None


def _create_vllm_config() -> MagicMock:
    compilation_config = CompilationConfig(
        cudagraph_mode="FULL",
        cudagraph_capture_sizes=[4],
    )
    compilation_config.max_cudagraph_capture_size = 4
    compilation_config.post_init_cudagraph_sizes()

    vllm_config = MagicMock(spec=VllmConfig)
    vllm_config.compilation_config = compilation_config
    vllm_config.scheduler_config = SchedulerConfig.default_factory(max_num_seqs=4)
    vllm_config.parallel_config = ParallelConfig()
    vllm_config.speculative_config = None
    vllm_config.num_speculative_tokens = 0
    return vllm_config


def test_full_capture_sets_graph_pool_id_before_cuda_graph(monkeypatch):
    """FULL capture must set graph_pool_id before entering torch.cuda.graph().

    NCCL symmetric memory checks this global during graph capture; without
    it, capture fails with:
    AssertionError: graph_pool_id is not set under graph capture
    """
    graph_pool = object()
    monkeypatch.setattr(
        gpu_cudagraph_utils,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=True, is_last_rank=True),
    )
    monkeypatch.setattr(
        gpu_cudagraph_utils.current_platform,
        "get_global_graph_pool",
        lambda: graph_pool,
    )

    manager = gpu_cudagraph_utils.CudaGraphManager(
        vllm_config=_create_vllm_config(),
        device=torch.device("cpu"),
        cudagraph_mode=CUDAGraphMode.FULL,
        decode_query_len=1,
    )

    desc = BatchExecutionDescriptor(
        cg_mode=CUDAGraphMode.FULL,
        num_tokens=4,
        num_reqs=4,
        uniform_token_count=1,
    )
    manager._capture_descs[CUDAGraphMode.FULL] = [desc]

    def create_forward_fn(desc, warmup):
        return lambda _mode: None

    @contextmanager
    def fake_graph_capture(*args, **kwargs):
        yield SimpleNamespace(stream=MagicMock())

    fake_offloader = MagicMock()

    def cuda_graph_enter(*args, **kwargs):
        assert pynccl_allocator._graph_pool_id is graph_pool

    mock_cuda_graph_ctx = MagicMock()
    mock_cuda_graph_ctx.__enter__ = cuda_graph_enter
    mock_cuda_graph_ctx.__exit__ = MagicMock(return_value=False)

    with (
        patch.object(gpu_cudagraph_utils, "graph_capture", fake_graph_capture),
        patch.object(gpu_cudagraph_utils, "get_offloader", lambda: fake_offloader),
        patch.object(gpu_cudagraph_utils.torch.cuda, "CUDAGraph"),
        patch.object(
            gpu_cudagraph_utils.torch.cuda,
            "graph",
            return_value=mock_cuda_graph_ctx,
        ) as mock_cuda_graph,
    ):
        manager.capture(create_forward_fn)

    mock_cuda_graph.assert_called_once()


def test_full_capture_prewarms_capture_compatible_forward_fn(monkeypatch):
    """FULL capture eagerly exercises the exact capture-compatible path."""
    manager = object.__new__(gpu_cudagraph_utils.CudaGraphManager)
    manager.device = torch.device("cpu")
    manager._capture_descs = {
        CUDAGraphMode.FULL: [
            BatchExecutionDescriptor(
                cg_mode=CUDAGraphMode.FULL,
                num_tokens=8,
                num_reqs=8,
                uniform_token_count=1,
            )
        ]
    }
    manager.graphs = {}
    manager.pool = object()
    manager._graphs_captured = False
    manager.use_breakable_cg = False

    calls: list[tuple[str, CUDAGraphMode, bool]] = []
    in_graph = False
    capture_factory_calls = 0

    def create_forward_fn(desc, warmup):
        nonlocal capture_factory_calls
        if warmup:
            label = "warmup"
        else:
            capture_factory_calls += 1
            label = f"capture-{capture_factory_calls}"

        def forward_fn(cg_mode):
            calls.append((label, cg_mode, in_graph))

        return forward_fn

    @contextmanager
    def fake_cuda_graph(graph, pool):
        nonlocal in_graph
        assert not in_graph
        in_graph = True
        try:
            yield
        finally:
            in_graph = False

    fake_graph_pool_setter = MagicMock()
    fake_offloader = MagicMock()
    monkeypatch.setattr(
        gpu_cudagraph_utils, "graph_capture", lambda device: nullcontext()
    )
    monkeypatch.setattr(gpu_cudagraph_utils, "is_global_first_rank", lambda: False)
    monkeypatch.setattr(torch.cuda, "CUDAGraph", MagicMock)
    monkeypatch.setattr(torch.cuda, "graph", fake_cuda_graph)
    monkeypatch.setattr(gpu_cudagraph_utils, "get_offloader", lambda: fake_offloader)
    monkeypatch.setattr(
        gpu_cudagraph_utils, "set_graph_pool_id", fake_graph_pool_setter
    )

    manager.capture(create_forward_fn)

    assert calls == [
        ("warmup", CUDAGraphMode.NONE, False),
        ("capture-1", CUDAGraphMode.NONE, False),
        ("capture-2", CUDAGraphMode.NONE, True),
    ]
    assert fake_offloader.sync_prev_onload.call_count == 2
    assert fake_offloader.join_after_forward.call_count == 2
    fake_graph_pool_setter.assert_called_once_with(manager.pool)
