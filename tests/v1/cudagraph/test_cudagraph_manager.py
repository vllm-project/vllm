# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections import defaultdict
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
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu import cudagraph_utils as gpu_cudagraph_utils
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor
from vllm.v1.worker.gpu.input_batch import InputBuffers
from vllm.v1.worker.gpu.pcp_manager import PCPManager

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


def test_piecewise_capture_uses_pcp_dummy_slot_mappings():
    num_reqs = 32
    num_tokens = 56
    pcp_world_size = 2
    input_buffers = InputBuffers(num_reqs, num_tokens, torch.device("cpu"))

    pcp_block_tables = SimpleNamespace(
        num_kv_cache_groups=1,
        input_block_tables=(torch.zeros(num_reqs * 2, 1, dtype=torch.int32),),
    )
    pcp_manager = PCPManager(
        pcp_world_size=pcp_world_size,
        pcp_rank=0,
        device=torch.device("cpu"),
        max_num_reqs=num_reqs,
        max_num_tokens=num_tokens,
        block_tables=pcp_block_tables,
    )

    block_tables = MagicMock()
    block_tables.cp_size = 1
    block_tables.get_dummy_block_tables.return_value = ()
    block_tables.get_dummy_slot_mappings.return_value = torch.zeros(
        1, num_tokens, dtype=torch.int64
    )
    model_state = MagicMock()
    model_state.prepare_attn.return_value = {}
    kv_cache_config = KVCacheConfig(
        num_blocks=0,
        kv_cache_tensors=[],
        kv_cache_groups=[],
    )

    gpu_cudagraph_utils.prepare_inputs_to_capture(
        num_reqs,
        num_tokens,
        model_state,
        input_buffers,
        block_tables,
        [],
        kv_cache_config,
        full_cudagraph=False,
        pcp_manager=pcp_manager,
    )

    slot_mappings = model_state.prepare_attn.call_args.args[3]
    assert slot_mappings.shape == (1, num_tokens * pcp_world_size)
    block_tables.get_dummy_slot_mappings.assert_not_called()


_DECODE_QUERY_LEN = 3


def _create_decode_vllm_config(
    capture_sizes: list[int],
    num_speculative_tokens: int = 0,
    dynamic_spec_num_tokens: list[int] | None = None,
) -> MagicMock:
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
    vllm_config.num_speculative_tokens = num_speculative_tokens
    if dynamic_spec_num_tokens is None:
        vllm_config.speculative_config = None
    else:
        speculative_config = MagicMock()
        speculative_config.uses_dynamic_speculative_decoding.return_value = True
        # Each entry is (range_start, range_end, num_speculative_tokens); only
        # the third element is read by the manager.
        speculative_config.num_speculative_tokens_per_batch_size = [
            (0, 0, n) for n in dynamic_spec_num_tokens
        ]
        vllm_config.speculative_config = speculative_config
    return vllm_config


def _make_spec_decode_manager(
    monkeypatch,
    decode_query_len: int = _DECODE_QUERY_LEN,
    capture_sizes: list[int] | None = None,
    num_speculative_tokens: int = 0,
    dynamic_spec_num_tokens: list[int] | None = None,
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
            num_speculative_tokens=num_speculative_tokens,
            dynamic_spec_num_tokens=dynamic_spec_num_tokens,
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


def test_mixed_batch_at_decode_only_token_count_still_gets_a_graph(monkeypatch):
    """A mixed batch must not fall to eager where only decode graphs are staged.

    ``round_up(size, decode_query_len)`` lands FULL decode graphs on token counts
    the PIECEWISE ladder never uses -- with capture sizes [1, 2, 4, 8, 16, 24]
    and query length 3, FULL covers 3, 6, 9 and 18 while PIECEWISE has only
    1, 2, 4, 8, 16, 24. Building each mode's candidate ranges independently keeps
    a PIECEWISE graph reachable there. Deriving the ranges from the staged sizes
    instead offers a mixed batch nothing but decode descriptors, whose
    ``uniform_token_count`` it can never match, so it runs fully eager.
    """
    manager = _make_spec_decode_manager(monkeypatch)

    decode_only_token_counts = sorted(
        {desc.num_tokens for desc in manager._capture_descs[CUDAGraphMode.FULL]}
        - {desc.num_tokens for desc in manager._capture_descs[CUDAGraphMode.PIECEWISE]}
    )
    # Guard the premise: with no such token counts this test would cover nothing.
    assert decode_only_token_counts

    for num_tokens in decode_only_token_counts:
        desc = manager.dispatch(
            num_reqs=1,
            num_tokens=num_tokens,
            uniform_token_count=None,
            num_active_loras=0,
        )
        assert desc.cg_mode == CUDAGraphMode.PIECEWISE, num_tokens
        assert desc.uniform_token_count is None, num_tokens
        assert desc.num_tokens >= num_tokens, num_tokens


def test_uniform_decode_beyond_capture_ladder_falls_back(monkeypatch):
    manager = _make_spec_decode_manager(monkeypatch)

    desc = manager.dispatch(
        num_reqs=9,
        num_tokens=27,
        uniform_token_count=_DECODE_QUERY_LEN,
        num_active_loras=0,
    )

    assert desc.cg_mode == CUDAGraphMode.NONE


def test_dynamic_spec_decode_shared_token_count_stays_reachable(monkeypatch):
    """Every captured decode graph must remain reachable from dispatch().

    Under dynamic speculative decoding ``decode_query_lens`` is a list, and the
    staging loop rounds each capture size up to *every* query length. Several
    decode graphs therefore land on the same ``num_tokens`` -- e.g. capture size
    2 stages both ``round_up(2, 1) == 2`` and ``round_up(2, 2) == 2``. Building
    the candidate ranges one descriptor at a time would give all but the first
    of those an empty range, so they would never enter a candidate list and
    their batches would silently fall through to PIECEWISE.
    """
    manager = _make_spec_decode_manager(
        monkeypatch,
        decode_query_len=4,
        capture_sizes=[2, 4, 6, 8],
        num_speculative_tokens=3,
        dynamic_spec_num_tokens=[0, 1, 2, 3],  # -> decode_query_lens [1, 2, 3, 4]
    )

    full_descs = manager._capture_descs[CUDAGraphMode.FULL]
    by_num_tokens: dict[int, list] = defaultdict(list)
    for desc in full_descs:
        by_num_tokens[desc.num_tokens].append(desc)
    # Guard the premise: without collisions this test would not cover the bug.
    assert any(len(descs) > 1 for descs in by_num_tokens.values())

    for desc in full_descs:
        assert desc in manager._candidates[(desc.num_tokens, 0)], desc
        assert (
            manager.dispatch(
                num_reqs=desc.num_reqs,
                num_tokens=desc.num_tokens,
                uniform_token_count=desc.uniform_token_count,
                num_active_loras=0,
            )
            == desc
        ), desc


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
