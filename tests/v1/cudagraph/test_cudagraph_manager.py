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


_DECODE_QUERY_LEN = 3


def _create_decode_vllm_config(capture_sizes: list[int]) -> MagicMock:
    compilation_config = CompilationConfig(
        cudagraph_mode="FULL_AND_PIECEWISE",
        cudagraph_capture_sizes=capture_sizes,
    )
    compilation_config.max_cudagraph_capture_size = capture_sizes[-1]
    compilation_config.post_init_cudagraph_sizes()

    vllm_config = MagicMock(spec=VllmConfig)
    vllm_config.compilation_config = compilation_config
    vllm_config.scheduler_config = SchedulerConfig.default_factory(max_num_seqs=8)
    vllm_config.parallel_config = ParallelConfig()
    vllm_config.speculative_config = None
    vllm_config.num_speculative_tokens = 0
    return vllm_config


def _make_spec_decode_manager(
    monkeypatch,
    decode_query_len: int = _DECODE_QUERY_LEN,
    capture_sizes: list[int] | None = None,
) -> gpu_cudagraph_utils.CudaGraphManager:
    monkeypatch.setattr(
        gpu_cudagraph_utils,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=True, is_last_rank=True),
    )
    monkeypatch.setattr(
        gpu_cudagraph_utils.current_platform,
        "get_global_graph_pool",
        lambda: object(),
    )
    manager = gpu_cudagraph_utils.CudaGraphManager(
        vllm_config=_create_decode_vllm_config(
            capture_sizes or [1, 2, 4, 8, 16, 24],
        ),
        device=torch.device("cpu"),
        cudagraph_mode=CUDAGraphMode.FULL_AND_PIECEWISE,
        decode_query_len=decode_query_len,
    )
    manager._graphs_captured = True
    return manager


def test_uniform_decode_pads_up_to_full_graph(monkeypatch):
    manager = _make_spec_decode_manager(monkeypatch)
    assert [
        (desc.cg_mode, desc.num_tokens) for desc in manager._candidates[(12, 0)]
    ] == [
        (CUDAGraphMode.FULL, 18),
        (CUDAGraphMode.PIECEWISE, 16),
    ]

    desc = manager.dispatch(
        num_reqs=4,
        num_tokens=12,
        uniform_token_count=_DECODE_QUERY_LEN,
        num_active_loras=0,
    )

    assert desc.cg_mode == CUDAGraphMode.FULL
    assert desc.uniform_token_count == _DECODE_QUERY_LEN
    assert desc.num_tokens == 18
    assert desc.num_reqs == 6


def test_uniform_decode_exact_match_is_not_over_padded(monkeypatch):
    manager = _make_spec_decode_manager(monkeypatch)

    desc = manager.dispatch(
        num_reqs=3,
        num_tokens=9,
        uniform_token_count=_DECODE_QUERY_LEN,
        num_active_loras=0,
    )

    assert desc.cg_mode == CUDAGraphMode.FULL
    assert desc.num_tokens == 9
    assert desc.num_reqs == 3


def test_mixed_batch_never_selects_a_uniform_decode_graph(monkeypatch):
    manager = _make_spec_decode_manager(monkeypatch)

    desc = manager.dispatch(
        num_reqs=2,
        num_tokens=12,
        uniform_token_count=None,
        num_active_loras=0,
    )

    assert desc.cg_mode == CUDAGraphMode.PIECEWISE
    assert desc.uniform_token_count is None
    assert desc.num_tokens == 16


def test_uniform_decode_beyond_capture_ladder_falls_back(monkeypatch):
    manager = _make_spec_decode_manager(monkeypatch)

    desc = manager.dispatch(
        num_reqs=9,
        num_tokens=27,
        uniform_token_count=_DECODE_QUERY_LEN,
        num_active_loras=0,
    )

    assert desc.cg_mode == CUDAGraphMode.NONE


@pytest.mark.parametrize(
    "decode_query_len,capture_sizes",
    [(1, [1, 2, 4, 8]), (2, [2, 4, 8, 16]), (8, [8, 16, 32, 64])],
)
def test_divisor_query_len_dispatch_is_unchanged(
    monkeypatch, decode_query_len, capture_sizes
):
    manager = _make_spec_decode_manager(
        monkeypatch,
        decode_query_len=decode_query_len,
        capture_sizes=capture_sizes,
    )

    max_reqs = capture_sizes[-1] // decode_query_len
    for num_reqs in range(1, max_reqs + 1):
        num_tokens = num_reqs * decode_query_len
        desc = manager.dispatch(
            num_reqs=num_reqs,
            num_tokens=num_tokens,
            uniform_token_count=decode_query_len,
            num_active_loras=0,
        )
        expected = min(s for s in capture_sizes if s >= num_tokens)
        assert desc.cg_mode == CUDAGraphMode.FULL, num_tokens
        assert desc.num_tokens == expected, num_tokens
        assert desc.num_reqs == expected // decode_query_len, num_tokens
