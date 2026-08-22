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


def _create_decode_vllm_config(capture_sizes: list[int]) -> MagicMock:
    """Config for the padding tests below.

    With capture_sizes [1, 2, 4, 8, 16, 24] and qlen=3, the FULL decode graphs
    land on round_up(size, 3) = 3, 6, 9, 18, 24 tokens while the PIECEWISE
    graphs stay on the raw sizes. 12 tokens (4 requests x qlen 3) therefore has
    no exact FULL decode graph; the only descriptor staged for it is the size-16
    PIECEWISE one.
    """
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
    is_rocm: bool = True,
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
    # Pad-up dispatch is ROCm-gated. Patching the platform keeps these tests
    # running (and meaningful) on the CPU runner that executes this file in CI.
    monkeypatch.setattr(
        gpu_cudagraph_utils.current_platform,
        "is_rocm",
        lambda: is_rocm,
    )
    manager = gpu_cudagraph_utils.CudaGraphManager(
        vllm_config=_create_decode_vllm_config(
            capture_sizes or [1, 2, 4, 8, 16, 24],
        ),
        device=torch.device("cpu"),
        cudagraph_mode=CUDAGraphMode.FULL_AND_PIECEWISE,
        decode_query_len=decode_query_len,
    )
    # dispatch() only consults the candidate lists once capture has run; these
    # tests exercise selection, not capture, so mark it done.
    manager._graphs_captured = True
    return manager


def test_uniform_decode_pads_up_to_full_graph(monkeypatch):
    """A uniform-decode batch with no exact FULL graph must pad, not go PIECEWISE.

    ROCm only -- see test_pad_up_is_rocm_gated for the other platforms.

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


def test_pad_up_is_rocm_gated(monkeypatch):
    """Off ROCm, the same batch keeps the pre-change behaviour.

    Identical to test_uniform_decode_pads_up_to_full_graph, except the platform
    reports non-ROCm. The size-18 FULL decode graph is not offered, so dispatch
    falls to the size-16 PIECEWISE descriptor exactly as it did before this
    change. This is what pins the gate: delete it and this test fails.
    """
    manager = _make_spec_decode_manager(monkeypatch, is_rocm=False)

    desc = manager.dispatch(
        num_reqs=4,
        num_tokens=12,
        uniform_token_count=_DECODE_QUERY_LEN,
        num_active_loras=0,
    )

    assert desc.cg_mode == CUDAGraphMode.PIECEWISE
    assert desc.num_tokens == 16


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


@pytest.mark.parametrize(
    "decode_query_len,capture_sizes",
    [(1, [1, 2, 4, 8]), (2, [2, 4, 8, 16]), (8, [8, 16, 32, 64])],
)
def test_divisor_query_len_dispatch_is_unchanged(
    monkeypatch, decode_query_len, capture_sizes
):
    """Pad-up dispatch is inert when the query length divides the ladder.

    round_up(size, qlen) == size for every captured size, so a FULL decode
    graph already exists at each one and there is no gap to pad across. The
    smallest pad-up candidate that fits is then exactly the descriptor the
    pre-change code took from the staged list. This covers decode_query_len=1
    -- every deployment not running speculative decoding -- as well as
    speculative query lengths that happen to divide the ladder.
    """
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
