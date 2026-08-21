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


# Spec-decode query length used by the padding tests below. With qlen=3 the
# uniform-decode graphs land on multiples of 3 (round_up(size, 3)), which leaves
# gaps against the power-of-two capture ladder -- exactly the case the padding
# dispatch has to cover.
_DECODE_QUERY_LEN = 3


def _create_spec_decode_vllm_config() -> MagicMock:
    """Config whose capture ladder leaves a uniform-decode gap at 12 tokens.

    capture_sizes [1, 2, 4, 8, 16, 24] with qlen=3 produce FULL decode graphs at
    round_up(size, 3) = 3, 6, 9, 18, 24 tokens and PIECEWISE graphs at the raw
    sizes. 12 tokens (4 requests x qlen 3) therefore has no exact FULL decode
    graph; the only descriptor staged for it is the size-16 PIECEWISE one.
    """
    compilation_config = CompilationConfig(
        cudagraph_mode="FULL_AND_PIECEWISE",
        cudagraph_capture_sizes=[1, 2, 4, 8, 16, 24],
    )
    compilation_config.max_cudagraph_capture_size = 24
    compilation_config.post_init_cudagraph_sizes()

    vllm_config = MagicMock(spec=VllmConfig)
    vllm_config.compilation_config = compilation_config
    vllm_config.scheduler_config = SchedulerConfig.default_factory(max_num_seqs=8)
    vllm_config.parallel_config = ParallelConfig()
    vllm_config.speculative_config = None
    vllm_config.num_speculative_tokens = 0
    return vllm_config


def _make_spec_decode_manager(monkeypatch) -> gpu_cudagraph_utils.CudaGraphManager:
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
        vllm_config=_create_spec_decode_vllm_config(),
        device=torch.device("cpu"),
        cudagraph_mode=CUDAGraphMode.FULL_AND_PIECEWISE,
        decode_query_len=_DECODE_QUERY_LEN,
    )
    # dispatch() only consults the candidate lists once capture has run; these
    # tests exercise selection, not capture, so mark it done.
    manager._graphs_captured = True
    return manager


def test_uniform_decode_pads_up_to_full_graph(monkeypatch):
    """A uniform-decode batch with no exact FULL graph must pad, not go PIECEWISE.

    Without pad-up dispatch, 12 tokens finds only the size-16 PIECEWISE
    descriptor -- which _is_compatible accepts, because a PIECEWISE descriptor
    has uniform_token_count=None and matches anything -- so attention runs eager
    (metadata build plus kernels on the host critical path) every decode step.
    The size-18 FULL decode graph can serve the batch by padding 4 requests up
    to 6, and must win.
    """
    manager = _make_spec_decode_manager(monkeypatch)

    desc = manager.dispatch(
        num_reqs=4,
        num_tokens=12,
        uniform_token_count=_DECODE_QUERY_LEN,
        num_active_loras=0,
    )

    assert desc.cg_mode == CUDAGraphMode.FULL
    assert desc.uniform_token_count == _DECODE_QUERY_LEN
    # Smallest FULL decode graph that fits, not merely any of them.
    assert desc.num_tokens == 18
    assert desc.num_reqs == 6


def test_uniform_decode_exact_match_is_not_over_padded(monkeypatch):
    """An exact FULL decode graph still wins over larger pad-up candidates."""
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
    """The safety property: pad-up candidates must be inert for mixed batches.

    Uniform-decode descriptors are offered ahead of the mixed fallback for every
    token count, so this asserts _is_compatible really does reject them when the
    batch is not uniform -- otherwise a prefill batch would replay a decode-only
    graph and silently produce wrong results.
    """
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
    """Past the largest captured graph there is nothing to pad up to."""
    manager = _make_spec_decode_manager(monkeypatch)

    desc = manager.dispatch(
        num_reqs=9,
        num_tokens=27,
        uniform_token_count=_DECODE_QUERY_LEN,
        num_active_loras=0,
    )

    assert desc.cg_mode == CUDAGraphMode.NONE
