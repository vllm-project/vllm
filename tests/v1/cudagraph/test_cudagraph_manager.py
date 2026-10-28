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
from vllm.v1.worker.gpu.dp_utils import dispatch_cg_and_sync_dp
from vllm.v1.worker.gpu.input_batch import InputBuffers
from vllm.v1.worker.gpu.pcp_manager import PCPManager
from vllm.v1.worker.gpu.spec_decode.target_dependent_ar import (
    cudagraph_utils as spec_cudagraph_utils,
)
from vllm.v1.worker.gpu.spec_decode.target_dependent_ar.cudagraph_utils import (
    SpeculatorCudaGraphManager,
)

pytestmark = pytest.mark.cpu_test


@pytest.fixture(autouse=True)
def _reset_graph_pool_id():
    pynccl_allocator._graph_pool_id = None
    yield
    pynccl_allocator._graph_pool_id = None


def _create_vllm_config() -> MagicMock:
    compilation_config = CompilationConfig(
        cudagraph_mode=CUDAGraphMode.FULL,
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
    vllm_config.use_cumem_cudagraph_pool = False
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
        decode_query_lens={1: 4},
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
        cp_interleave=1,
    )
    pcp_manager = PCPManager(
        pcp_world_size=pcp_world_size,
        pcp_rank=0,
        device=torch.device("cpu"),
        shard_decode_requests=False,
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


@pytest.mark.parametrize("full_cudagraph,expected_context_len", [(True, 2), (False, 0)])
def test_dcp_full_capture_has_kv_context(
    monkeypatch, full_cudagraph, expected_context_len
):
    """DCP skips context attention on the host when no request has context, so a
    FULL graph captured without context would ignore the KV cache on replay."""
    monkeypatch.setattr(
        gpu_cudagraph_utils, "prepare_dcp_local_seq_lens", lambda *a, **kw: None
    )
    num_tokens = num_reqs = 8
    buffers = InputBuffers(num_reqs, num_tokens, torch.device("cpu"))
    block_tables = MagicMock()
    block_tables.cp_size = 2
    block_tables.cp_interleave = 1
    block_tables.block_sizes = [16]
    block_tables.get_dummy_block_tables.return_value = (
        torch.zeros(num_reqs, 4, dtype=torch.int32),
    )
    block_tables.get_dummy_slot_mappings.return_value = torch.empty(
        1, num_tokens, dtype=torch.int64
    )
    model_state = MagicMock()
    model_state.max_model_len = 1024
    kv_cache_config = KVCacheConfig(
        num_blocks=8, kv_cache_tensors=[], kv_cache_groups=[]
    )

    gpu_cudagraph_utils.prepare_inputs_to_capture(
        num_reqs,
        num_tokens,
        model_state,
        buffers,
        block_tables,
        [],
        kv_cache_config,
        full_cudagraph=full_cudagraph,
        max_query_len=1,
    )

    batch = model_state.prepare_attn.call_args.args[0]
    context_lens = batch.seq_lens_cpu_upper_bound[:num_reqs] - torch.from_numpy(
        batch.num_scheduled_tokens[:num_reqs]
    )
    assert (context_lens == expected_context_len).all()


@pytest.mark.parametrize(
    "full_cudagraph,max_query_len,expected_max_query_len",
    [(False, None, None), (True, None, 8), (True, 1, 1), (True, 3, 3)],
)
def test_capture_query_length_covers_replay_batches(
    full_cudagraph, max_query_len, expected_max_query_len
):
    """Mixed FULL graphs must capture prefill kernels as well as decode kernels."""
    num_tokens = num_reqs = 8
    buffers = InputBuffers(num_reqs, num_tokens, torch.device("cpu"))
    block_tables = MagicMock()
    block_tables.cp_size = 1
    block_tables.get_dummy_block_tables.return_value = ()
    block_tables.get_dummy_slot_mappings.return_value = torch.empty(
        0, num_tokens, dtype=torch.int64
    )
    model_state = MagicMock()
    kv_cache_config = KVCacheConfig(
        num_blocks=0, kv_cache_tensors=[], kv_cache_groups=[]
    )

    gpu_cudagraph_utils.prepare_inputs_to_capture(
        num_reqs,
        num_tokens,
        model_state,
        buffers,
        block_tables,
        [],
        kv_cache_config,
        full_cudagraph=full_cudagraph,
        max_query_len=max_query_len,
    )

    batch = model_state.prepare_attn.call_args.args[0]
    assert batch.num_scheduled_tokens.max() == 1
    assert batch.max_query_len == expected_max_query_len


@pytest.mark.parametrize(
    "uniform_token_count,max_query_len,expected_max_query_len",
    [(1, None, 1), (None, 3, 3), (None, None, 8)],
)
def test_speculator_capture_preserves_decode_query_bounds(
    monkeypatch, uniform_token_count, max_query_len, expected_max_query_len
):
    """Draft decode graphs keep narrow queries while mixed graphs cover prefill."""
    num_tokens = num_reqs = 8
    desc = BatchExecutionDescriptor(
        CUDAGraphMode.FULL,
        num_tokens,
        num_reqs,
        uniform_token_count=uniform_token_count,
        max_query_len=max_query_len,
    )
    manager = SpeculatorCudaGraphManager.__new__(SpeculatorCudaGraphManager)
    manager.max_num_reqs = num_reqs
    manager.decode_query_len = uniform_token_count or max_query_len or num_tokens
    manager.vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(data_parallel_size=1),
        num_speculative_tokens=3,
    )
    buffers = InputBuffers(num_reqs, num_tokens, torch.device("cpu"))
    block_tables = MagicMock()
    block_tables.cp_size = 1
    block_tables.get_dummy_block_tables.return_value = ()
    block_tables.get_dummy_slot_mappings.return_value = torch.empty(
        0, num_tokens, dtype=torch.int64
    )
    model_state = MagicMock()
    kv_cache_config = KVCacheConfig(
        num_blocks=0, kv_cache_tensors=[], kv_cache_groups=[]
    )

    def capture(self, create_forward_fn, progress_bar_desc):
        for warmup in (True, False):
            create_forward_fn(desc, warmup)(desc.cg_mode)

    monkeypatch.setattr(gpu_cudagraph_utils.CudaGraphManager, "capture", capture)
    manager.capture(
        MagicMock(), model_state, buffers, block_tables, [], kv_cache_config
    )

    assert model_state.prepare_attn.call_count == 2
    for call in model_state.prepare_attn.call_args_list:
        assert call.args[0].max_query_len == expected_max_query_len


_DECODE_QUERY_LEN = 3


def _create_decode_vllm_config(
    capture_sizes: list[int],
    num_speculative_tokens: int = 0,
    dynamic_spec_schedule: list[tuple[int, int, int]] | None = None,
    max_num_seqs: int = 8,
    use_kda_recoverssm: bool = False,
    decode_sizes: list[int] | None = None,
    max_decode_tokens: int | None = None,
) -> MagicMock:
    if decode_sizes is None:
        # The deprecated shared sizes set both kinds of graphs.
        compilation_config = CompilationConfig(
            cudagraph_mode=CUDAGraphMode.FULL_AND_PIECEWISE,
            cudagraph_capture_sizes=capture_sizes,
        )
        compilation_config.max_cudagraph_capture_size = capture_sizes[-1]
    else:
        compilation_config = CompilationConfig(
            cudagraph_mode=CUDAGraphMode.FULL_AND_PIECEWISE,
            prefill_cudagraph_capture_token_sizes=capture_sizes,
            decode_cudagraph_capture_request_sizes=decode_sizes,
            max_decode_cudagraph_capture_tokens=max_decode_tokens,
        )

    vllm_config = MagicMock(spec=VllmConfig)
    vllm_config.compilation_config = compilation_config
    vllm_config.scheduler_config = SchedulerConfig.default_factory(
        max_num_seqs=max_num_seqs
    )
    vllm_config.parallel_config = ParallelConfig()
    vllm_config.cache_config = SimpleNamespace(use_kda_recoverssm=use_kda_recoverssm)
    vllm_config.num_speculative_tokens = num_speculative_tokens
    if dynamic_spec_schedule is None:
        vllm_config.speculative_config = None
    else:
        speculative_config = MagicMock()
        speculative_config.uses_dynamic_speculative_decoding.return_value = True
        speculative_config.num_speculative_tokens_per_batch_size = dynamic_spec_schedule
        speculative_config.enable_adaptive_verification = False
        vllm_config.speculative_config = speculative_config
    _split_capture_sizes(vllm_config, legacy_sizes=decode_sizes is None)
    return vllm_config


def _split_capture_sizes(vllm_config: MagicMock, legacy_sizes: bool) -> None:
    """Resolve the prefill and decode sizes, as config resolution does."""
    vllm_config.uniform_decode_query_len = vllm_config.num_speculative_tokens + 1
    vllm_config.get_varlen_decode_cudagraph_token_sizes = (
        lambda decode_query_len: VllmConfig.get_varlen_decode_cudagraph_token_sizes(
            vllm_config, decode_query_len
        )
    )
    VllmConfig._split_cudagraph_capture_sizes(
        vllm_config,
        legacy_sizes,
        VllmConfig.get_decode_cudagraph_query_lens(
            vllm_config, vllm_config.uniform_decode_query_len
        ),
    )
    vllm_config.compilation_config.post_init_cudagraph_sizes()


def _make_spec_decode_manager(
    monkeypatch,
    decode_query_len: int = _DECODE_QUERY_LEN,
    capture_sizes: list[int] | None = None,
    num_speculative_tokens: int | None = None,
    dynamic_spec_schedule: list[tuple[int, int, int]] | None = None,
    max_num_seqs: int = 8,
    use_kda_recoverssm: bool = False,
    varlen_decode: bool = False,
) -> gpu_cudagraph_utils.CudaGraphManager:
    if num_speculative_tokens is None:
        num_speculative_tokens = decode_query_len - 1
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
    vllm_config = _create_decode_vllm_config(
        capture_sizes or [1, 2, 4, 8, 16, 24],
        num_speculative_tokens=num_speculative_tokens,
        dynamic_spec_schedule=dynamic_spec_schedule,
        max_num_seqs=max_num_seqs,
        use_kda_recoverssm=use_kda_recoverssm,
    )
    manager = gpu_cudagraph_utils.CudaGraphManager(
        vllm_config=vllm_config,
        device=torch.device("cpu"),
        cudagraph_mode=CUDAGraphMode.FULL_AND_PIECEWISE,
        decode_query_lens=VllmConfig.get_decode_cudagraph_query_lens(
            vllm_config, decode_query_len
        ),
        varlen_decode=varlen_decode,
    )
    manager._graphs_captured = True
    return manager


@pytest.mark.parametrize("decode_query_len", [1, 3])
def test_decode_capture_sizes_are_request_counts(monkeypatch, decode_query_len):
    """Each runner captures the uniform decode sizes at its own query length,
    and the mixed sizes alone back other batches; `[]` captures none of a kind."""
    monkeypatch.setattr(
        gpu_cudagraph_utils,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=True, is_last_rank=True),
    )
    monkeypatch.setattr(
        gpu_cudagraph_utils.current_platform, "get_global_graph_pool", lambda: None
    )

    def capture_plan(decode_sizes, mixed_sizes):
        vllm_config = _create_decode_vllm_config(
            [1, 2, 4, 8, 16, 24], num_speculative_tokens=2
        )
        compilation_config = vllm_config.compilation_config
        compilation_config.decode_cudagraph_capture_request_sizes = decode_sizes
        compilation_config.prefill_cudagraph_capture_token_sizes = mixed_sizes
        manager = gpu_cudagraph_utils.CudaGraphManager(
            vllm_config=vllm_config,
            device=torch.device("cpu"),
            cudagraph_mode=CUDAGraphMode.FULL_AND_PIECEWISE,
            decode_query_lens={decode_query_len: 8},
        )
        descs = manager._capture_descs
        return (
            sorted(
                (d.num_reqs, d.num_tokens) for d in descs.get(CUDAGraphMode.FULL, [])
            ),
            sorted(d.num_tokens for d in descs.get(CUDAGraphMode.PIECEWISE, [])),
        )

    q = decode_query_len
    assert capture_plan([1, 2, 5], [8, 16]) == (
        [(1, q), (2, 2 * q), (5, 5 * q)],
        [8, 16],
    )
    assert capture_plan([], [8]) == ([], [8])
    assert capture_plan([2], []) == ([(2, 2 * q)], [])


def test_varlen_decode_graphs_take_the_decode_cap_as_a_token_budget(monkeypatch):
    """Adaptive verification's decode graphs are token-keyed up to the decode
    request cap times the query length, whatever the prefill sizes are."""
    monkeypatch.setattr(
        gpu_cudagraph_utils,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=True, is_last_rank=True),
    )
    monkeypatch.setattr(
        gpu_cudagraph_utils.current_platform, "get_global_graph_pool", lambda: None
    )

    def varlen_tokens(prefill_sizes):
        vllm_config = _create_decode_vllm_config(
            prefill_sizes, num_speculative_tokens=3, decode_sizes=[1, 2, 4]
        )
        manager = gpu_cudagraph_utils.CudaGraphManager(
            vllm_config=vllm_config,
            device=torch.device("cpu"),
            cudagraph_mode=CUDAGraphMode.FULL_AND_PIECEWISE,
            decode_query_lens={4: 8},
            varlen_decode=True,
        )
        return sorted(
            d.num_tokens
            for d in manager._capture_descs[CUDAGraphMode.FULL]
            if d.max_query_len is not None
        )

    assert varlen_tokens([8, 64]) == varlen_tokens([24]) == [1, 2, 4, 8, 10, 12, 14, 16]


def test_decode_graphs_stay_within_the_recorded_max(monkeypatch):
    """A drafter that runs a query length at more requests than the target does
    (DFlash's K + 1 under dynamic speculative decoding) captures no graph wider
    than the recorded max_cudagraph_capture_size, which backends size by."""
    monkeypatch.setattr(
        gpu_cudagraph_utils,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=True, is_last_rank=True),
    )
    monkeypatch.setattr(
        gpu_cudagraph_utils.current_platform, "get_global_graph_pool", lambda: None
    )
    vllm_config = _create_decode_vllm_config(
        [1, 2, 4, 8, 16, 24],
        num_speculative_tokens=3,
        dynamic_spec_schedule=[(1, 2, 3), (3, 8, 1)],
        decode_sizes=[1, 2, 4, 8],
    )
    manager = gpu_cudagraph_utils.CudaGraphManager(
        vllm_config=vllm_config,
        device=torch.device("cpu"),
        cudagraph_mode=CUDAGraphMode.FULL_AND_PIECEWISE,
        decode_query_lens={4: 8},
    )
    widest = max(d.num_tokens for d in manager._capture_descs[CUDAGraphMode.FULL])
    assert widest <= vllm_config.compilation_config.max_cudagraph_capture_size


@pytest.mark.parametrize(
    "max_decode_tokens,dynamic_spec_schedule",
    [(None, None), (32, None), (40, None), (None, [(1, 2, 3), (3, 16, 1)])],
)
def test_deepseek_v41_metadata_graphs_match_the_forward_graphs(
    monkeypatch, max_decode_tokens, dynamic_spec_schedule
):
    """DeepSeek-V4.1's recorded metadata builds serve exactly the FULL decode
    graphs: a budget-truncated boundary graph has its key, and no key is
    recorded past the budget or the widest query length's reach."""
    from vllm.models.deepseek_v41.nvidia.specialized_decode.runtime import (
        metadata_graphs,
    )

    monkeypatch.setattr(
        gpu_cudagraph_utils,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=True, is_last_rank=True),
    )
    monkeypatch.setattr(
        gpu_cudagraph_utils.current_platform, "get_global_graph_pool", lambda: None
    )
    vllm_config = _create_decode_vllm_config(
        [8, 16, 32, 64],
        num_speculative_tokens=3,
        dynamic_spec_schedule=dynamic_spec_schedule,
        max_num_seqs=16,
        decode_sizes=[1, 2, 4, 8, 12, 16],
        max_decode_tokens=max_decode_tokens,
    )
    vllm_config.get_decode_cudagraph_query_lens = (
        lambda q: VllmConfig.get_decode_cudagraph_query_lens(vllm_config, q)
    )
    manager = gpu_cudagraph_utils.CudaGraphManager(
        vllm_config=vllm_config,
        device=torch.device("cpu"),
        cudagraph_mode=CUDAGraphMode.FULL_AND_PIECEWISE,
        decode_query_lens=vllm_config.get_decode_cudagraph_query_lens(4),
    )
    forward = {
        d.num_reqs
        for d in manager._capture_descs[CUDAGraphMode.FULL]
        if d.uniform_token_count == 4 and d.num_reqs is not None
    }

    keys = metadata_graphs.AttnMetadataCudaGraphs(vllm_config, 4).capture_keys()

    assert {padded for _, padded, _, _ in keys} == forward
    assert sorted(n for n, _, _, _ in keys) == list(range(1, max(forward) + 1))


def test_decode_graph_must_hold_every_request(monkeypatch):
    """A decode batch with more requests than its tokens imply (prefill context
    parallel shards the tokens) takes no decode graph too small for it, and
    falls back to a token-keyed graph."""
    manager = _make_spec_decode_manager(
        monkeypatch, decode_query_len=1, capture_sizes=[1, 2, 4, 8, 16], max_num_seqs=16
    )
    rank_a = manager.dispatch(
        num_reqs=10, num_tokens=5, uniform_token_count=1, num_active_loras=0
    )
    rank_b = manager.dispatch(
        num_reqs=4, num_tokens=5, uniform_token_count=1, num_active_loras=0
    )
    assert (rank_a.cg_mode, rank_a.num_tokens) == (CUDAGraphMode.PIECEWISE, 8)
    assert (rank_b.cg_mode, rank_b.num_tokens) == (CUDAGraphMode.FULL, 8)


def test_uniform_decode_pads_up_to_full_graph(monkeypatch):
    manager = _make_spec_decode_manager(monkeypatch)
    # A non-uniform batch of the same size takes the PIECEWISE graph.
    desc = manager.dispatch(
        num_reqs=5, num_tokens=15, uniform_token_count=None, num_active_loras=0
    )
    assert (desc.cg_mode, desc.num_tokens) == (CUDAGraphMode.PIECEWISE, 16)

    desc = manager.dispatch(
        num_reqs=5,
        num_tokens=15,
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


def test_varlen_decode_graph_takes_bounded_decode_batches_only(monkeypatch):
    """A varlen decode graph replays any decode batch within its query-length
    bound. The runner passes no bound for a batch with a prefill, which must not
    replay one however short the prefill is."""
    manager = _make_spec_decode_manager(monkeypatch, varlen_decode=True)

    def dispatch(max_query_len):
        return manager.dispatch(
            num_reqs=3,
            num_tokens=6,
            uniform_token_count=None,
            num_active_loras=0,
            max_query_len=max_query_len,
        )

    desc = dispatch(_DECODE_QUERY_LEN)
    assert desc.cg_mode == CUDAGraphMode.FULL
    assert desc.max_query_len == _DECODE_QUERY_LEN
    assert dispatch(None).cg_mode == CUDAGraphMode.PIECEWISE
    assert dispatch(_DECODE_QUERY_LEN + 1).cg_mode == CUDAGraphMode.PIECEWISE


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
        # max_num_seqs is 8, and K narrows as the batch grows, so the schedule
        # covers K 3, 2, 1 and 0, giving decode_query_lens [1, 2, 3, 4].
        dynamic_spec_schedule=[(1, 2, 3), (3, 4, 2), (5, 6, 1), (7, 8, 0)],
    )

    full_descs = manager._capture_descs[CUDAGraphMode.FULL]
    by_num_tokens: dict[int, list] = defaultdict(list)
    for desc in full_descs:
        by_num_tokens[desc.num_tokens].append(desc)
    # Guard the premise: without collisions this test would not cover the bug.
    assert any(len(descs) > 1 for descs in by_num_tokens.values())

    for desc in full_descs:
        assert desc.num_reqs is not None
        assert (
            manager.dispatch(
                num_reqs=desc.num_reqs,
                num_tokens=desc.num_tokens,
                uniform_token_count=desc.uniform_token_count,
                num_active_loras=0,
            )
            == desc
        ), desc


@pytest.mark.parametrize("specialize", [False, True])
def test_speculator_replays_graph_for_runtime_width(monkeypatch, specialize):
    """Equal batch shapes must replay the graph captured for the selected K."""
    monkeypatch.setattr(
        gpu_cudagraph_utils,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=True, is_last_rank=True),
    )
    monkeypatch.setattr(
        gpu_cudagraph_utils.current_platform, "get_global_graph_pool", lambda: object()
    )
    monkeypatch.setattr(gpu_cudagraph_utils, "get_offloader", MagicMock)
    monkeypatch.setattr(
        spec_cudagraph_utils, "prepare_inputs_to_capture", lambda *a, **kw: ({}, {})
    )
    forward = MagicMock()
    replayed_widths: list[int] = []

    def capture(manager, create_forward_fn, progress_bar_desc):
        for desc in manager._capture_descs[CUDAGraphMode.FULL]:
            create_forward_fn(desc, warmup=False)(CUDAGraphMode.NONE)
            k = forward.call_args.kwargs["num_speculative_steps"]
            graph = MagicMock()
            graph.replay.side_effect = lambda k=k: replayed_widths.append(k)
            manager.graphs[desc] = graph
        manager._graphs_captured = True

    monkeypatch.setattr(gpu_cudagraph_utils.CudaGraphManager, "capture", capture)
    manager = spec_cudagraph_utils.SpeculatorCudaGraphManager(
        _create_decode_vllm_config(
            capture_sizes=[4],
            num_speculative_tokens=4,
            dynamic_spec_schedule=[(1, 2, 4), (3, 4, 2), (5, 6, 1), (7, 8, 0)],
            decode_sizes=[4],
        ),
        device=torch.device("cpu"),
        cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY,
        decode_query_lens={1: 8},
    )
    manager.capture(
        forward,
        MagicMock(),
        MagicMock(),
        MagicMock(),
        [],
        MagicMock(),
        specialize_spec_tokens=specialize,
    )
    captured_widths = [4, 2] if specialize else [4]
    assert [c.kwargs["num_speculative_steps"] for c in forward.call_args_list] == (
        captured_widths
    )

    for k in (2, 4, 2, 3):
        desc, _ = dispatch_cg_and_sync_dp(
            manager,
            num_reqs=3,
            num_tokens=3,
            uniform_token_count=1,
            dp_size=1,
            dp_rank=0,
        )
        desc = manager.specialize_spec_tokens(desc, k)
        captured_k = k if specialize else 4
        if captured_k in captured_widths:
            assert desc.cg_mode == CUDAGraphMode.FULL
            assert desc.num_tokens == 4
            manager.run_fullgraph(desc)
            assert replayed_widths[-1] == captured_k
        else:
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
