# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import contextmanager
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


def test_fullgraph_aux_outputs_use_per_descriptor_leading_dims(monkeypatch):
    large_desc = BatchExecutionDescriptor(
        cg_mode=CUDAGraphMode.FULL,
        num_tokens=8,
        num_reqs=8,
        uniform_token_count=1,
    )
    small_desc = BatchExecutionDescriptor(
        cg_mode=CUDAGraphMode.FULL,
        num_tokens=4,
        num_reqs=4,
        uniform_token_count=1,
    )

    manager = object.__new__(gpu_cudagraph_utils.ModelCudaGraphManager)
    manager.aux_hidden_states = []
    manager.aux_hidden_state_leading_dims = {}

    large_aux = [
        torch.full((4, 2), 1.0),
        torch.full((3, 3), 2.0),
    ]
    manager._copy_aux_hidden_state_outputs(large_desc, large_aux)
    small_aux = [
        torch.full((2, 2), 3.0),
        torch.full((1, 3), 4.0),
    ]
    manager._copy_aux_hidden_state_outputs(small_desc, small_aux)

    assert manager.aux_hidden_state_leading_dims == {
        large_desc: (4, 3),
        small_desc: (2, 1),
    }
    assert [x.shape[0] for x in manager.aux_hidden_states] == [4, 3]
    torch.testing.assert_close(manager.aux_hidden_states[0][:2], small_aux[0])
    torch.testing.assert_close(manager.aux_hidden_states[1][:1], small_aux[1])

    manager.is_last_pp_rank = True
    manager.use_aux_hidden_state_outputs = True
    manager.hidden_states = torch.arange(16, dtype=torch.float32).view(8, 2)
    graph = MagicMock()
    manager.graphs = {small_desc: graph}
    offloader = MagicMock()
    monkeypatch.setattr(gpu_cudagraph_utils, "get_offloader", lambda: offloader)

    hidden_states, aux_hidden_states = manager.run_fullgraph(small_desc)

    torch.testing.assert_close(hidden_states, manager.hidden_states[:4])
    assert [x.shape[0] for x in aux_hidden_states] == [2, 1]
    torch.testing.assert_close(aux_hidden_states[0], small_aux[0])
    torch.testing.assert_close(aux_hidden_states[1], small_aux[1])
    graph.replay.assert_called_once_with()
