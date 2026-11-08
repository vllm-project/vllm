# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import contextlib
import logging
from collections.abc import Iterator
from dataclasses import dataclass
from functools import partial
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

import vllm.distributed.aux_output_connector.worker as aux_output_worker
import vllm.v1.worker.gpu.model_runner as model_runner_module
from vllm.config import (
    AttentionConfig,
    CacheConfig,
    KernelConfig,
    LoadConfig,
    LoRAConfig,
    ParallelConfig,
    SpeculativeConfig,
)
from vllm.model_executor.warmup.jit_warmup import JitWarmupRegistry
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import (
    CircularBufferSpec,
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.outputs import AsyncModelRunnerOutput
from vllm.v1.worker.gpu.async_utils import async_copy_to_np
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.model_runner import ExecuteModelState, GPUModelRunner
from vllm.v1.worker.gpu.spec_decode.standalone_ar import speculator as ar_module
from vllm.v1.worker.gpu.spec_decode.utils import get_drafter_hidden_states


@pytest.mark.parametrize("eligible,expected_bound", [(True, 8), (False, None)])
def test_prompt_tail_keeps_decode_graph_bound(monkeypatch, eligible, expected_bound):
    """A decode-eligible prompt tail must not lose its varlen graph bound."""
    runner = Mock(
        lora_config=None,
        aux_output_connector=None,
        encoder_cache=None,
        pcp_manager=None,
        is_encoder_decoder=False,
    )
    runner.gather_batch_req_state.return_value = (
        SimpleNamespace(num_tokens=8, has_prefill=True, decode_graph_eligible=eligible),
        None,
    )
    scheduler_output = SchedulerOutput.make_empty()
    scheduler_output.num_scheduled_tokens = {"request": 8}
    scheduler_output.total_num_scheduled_tokens = 8
    dispatch = Mock(side_effect=RuntimeError("stop before forward"))
    monkeypatch.setattr(model_runner_module, "dispatch_cg_and_sync_dp", dispatch)

    with pytest.raises(RuntimeError, match="stop before forward"):
        GPUModelRunner.execute_model(runner, scheduler_output)

    assert dispatch.call_args.kwargs["max_query_len"] == expected_bound


def test_non_last_pp_rank_uses_global_batch_for_sample_feedback(monkeypatch):
    runner = GPUModelRunner.__new__(GPUModelRunner)
    runner.state_slots = None
    runner.is_last_pp_rank = False
    local_batch = object()
    global_batch = SimpleNamespace(idx_mapping=object())
    # Keep handles on the mocks: the runner's attributes are typed as the real
    # PCPManager / PPHandler / ModelState, which have no assert_* methods.
    restore_for_sampling = Mock()
    runner.pcp_manager = SimpleNamespace(
        global_batch=global_batch,
        restore_for_sampling=restore_for_sampling,
    )
    pp_receive = Mock(return_value=False)
    runner.pp_handler = SimpleNamespace(receive=pp_receive)
    postprocess_num_computed_tokens = Mock()
    monkeypatch.setattr(
        runner, "postprocess_num_computed_tokens", postprocess_num_computed_tokens
    )
    postprocess_state = Mock()
    runner.model_state = SimpleNamespace(postprocess_state=postprocess_state)
    runner.kv_connector = SimpleNamespace(post_forward=Mock(return_value=None))
    runner.eplb = SimpleNamespace(step=Mock())
    runner.execute_model_state = ExecuteModelState(
        input_batch=local_batch,
        attn_metadata=None,
        slot_mappings_by_layer=None,
        hidden_states=None,
        aux_hidden_states=None,
        dp_sync_state=None,
        finished_req_ids=set(),
        ec_connector_output=None,
        cudagraph_stats=None,
        num_spec_tokens_to_schedule=0,
    )

    runner.sample_tokens(None)

    pp_receive.assert_called_once_with(global_batch, recv_drafts=False)
    postprocess_num_computed_tokens.assert_called_once_with(global_batch)
    postprocess_state.assert_called_once_with(global_batch.idx_mapping, 0)
    restore_for_sampling.assert_not_called()


@pytest.mark.parametrize(
    "enabled,expected_events",
    [
        (True, ["allocate", "bind", "dummy_run"]),
        (False, ["dummy_run"]),
    ],
    ids=["aux-output-enabled", "aux-output-disabled"],
)
def test_profile_run_creates_aux_output_connector_before_forward(
    monkeypatch, enabled, expected_events
):
    events = []
    capturer = object()
    model = torch.nn.Module()
    vllm_config = SimpleNamespace(
        aux_output_config=SimpleNamespace(enabled=enabled),
        max_concurrent_batches=1,
        scheduler_config=SimpleNamespace(max_num_batched_tokens=8),
    )

    def allocate(**kwargs):
        assert kwargs == {"max_num_batched_tokens": 8, "vllm_config": vllm_config}
        events.append("allocate")
        return capturer

    def bind(bind_model, bind_capturer):
        assert bind_model is model and bind_capturer is capturer
        events.append("bind")

    def dummy_run(*args, **kwargs):
        events.append("dummy_run")
        return None, None

    monkeypatch.setattr(aux_output_worker, "RoutedExpertsCapturer", allocate)
    monkeypatch.setattr(aux_output_worker, "bind_routed_experts_capturer", bind)
    monkeypatch.setattr(torch.accelerator, "synchronize", lambda: None)
    runner = GPUModelRunner.__new__(GPUModelRunner)
    runner.vllm_config = vllm_config
    runner.model = model
    runner.aux_output_connector = None
    runner.supports_mm_inputs = False
    runner.is_last_pp_rank = False
    runner.max_num_tokens = 8
    runner.encoder_cache = None
    runner.pooling_runner = None
    monkeypatch.setattr(runner, "_dummy_run", dummy_run)

    runner.profile_run()

    assert events == expected_events
    connector = runner.aux_output_connector
    if enabled:
        assert connector is not None and connector._capturer is capturer
    else:
        assert connector is None


def test_qsa_circular_group_uses_custom_slot_mapping(monkeypatch):
    runner = GPUModelRunner.__new__(GPUModelRunner)
    runner.state_slots = None
    runner.max_model_len = 262144
    runner.is_encoder_decoder = False
    runner.dcp_size = 1
    runner.dcp_rank = 0
    runner.cp_interleave = 1
    runner.cache_config = SimpleNamespace(enable_prefix_caching=True)
    parallel_config = SimpleNamespace(
        decode_context_parallel_size=1,
        cp_kv_cache_interleave_size=1,
    )
    runner.parallel_config = parallel_config
    runner.vllm_config = SimpleNamespace(
        parallel_config=parallel_config,
        cache_config=SimpleNamespace(mamba_cache_mode="none"),
    )
    runner.jit_warmup_registry = JitWarmupRegistry(runner.vllm_config)
    runner.model_state = SimpleNamespace(
        get_additional_cg_support=lambda: (),
        num_new_sampled_tokens_per_step=1,
    )
    runner.speculator = None
    runner.batch_sharder = None
    runner.req_states = []
    runner.input_buffers = SimpleNamespace(query_start_loc=None)
    runner.vocab_size = 1
    runner.max_num_reqs = 1
    runner.max_num_tokens = 2
    runner.device = torch.device("cuda")

    raw_spec = CircularBufferSpec(
        block_size=8,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.bfloat16,
    )
    compressed_spec = FullAttentionSpec(
        block_size=262144,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.bfloat16,
    )
    kv_cache_config = KVCacheConfig(
        num_blocks=1,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                layer_names=["raw"],
                kv_cache_spec=UniformTypeKVCacheSpecs(
                    block_size=8,
                    kv_cache_specs={"raw": raw_spec},
                ),
            ),
            KVCacheGroupSpec(layer_names=["compressed"], kv_cache_spec=compressed_spec),
        ],
    )

    class FakeAttnCGSupport:
        def narrow(self, *args):
            return self

    attn_cg_support = FakeAttnCGSupport()
    monkeypatch.setattr(
        model_runner_module,
        "init_attn_backend",
        lambda *args, **kwargs: ([], attn_cg_support, [8, 262144]),
    )
    monkeypatch.setattr(
        model_runner_module,
        "maybe_create_adaptive_verification_manager",
        lambda **kwargs: None,
    )

    captured = {}

    class BlockTablesCaptured(Exception):
        pass

    def capture_block_tables(**kwargs):
        captured.update(kwargs)
        raise BlockTablesCaptured

    monkeypatch.setattr(model_runner_module, "BlockTables", capture_block_tables)

    with pytest.raises(BlockTablesCaptured):
        runner.initialize_kv_cache(kv_cache_config)

    assert captured["max_num_blocks_per_group"] == [1, 1]
    assert captured["slot_mapping_enabled"] == [False, True]


@pytest.mark.parametrize(
    ("mamba_cache_mode", "num_speculative_blocks", "expected"),
    [
        pytest.param("align", 0, 65_536, id="align-prefix-cache"),
        pytest.param("none", 7, 8, id="no-prefix-cache-with-speculation"),
    ],
)
def test_initialize_kv_cache_does_not_dcp_shard_mamba_block_table(
    monkeypatch,
    mamba_cache_mode: str,
    num_speculative_blocks: int,
    expected: int,
):
    """Mamba/GDN block-table rows index global positions, unlike DCP KV."""
    max_model_len = 1_048_576
    attention_block_size = 1_536
    mamba_block_size = 16
    dcp_size = 8
    full_attention_spec = FullAttentionSpec(
        block_size=attention_block_size,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.bfloat16,
    )
    mamba_spec = MambaSpec(
        shapes=((1,),),
        dtypes=(torch.bfloat16,),
        block_size=mamba_block_size,
        mamba_cache_mode=mamba_cache_mode,
        num_speculative_blocks=num_speculative_blocks,
    )
    kv_cache_config = KVCacheConfig(
        num_blocks=1,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(["attention"], full_attention_spec),
            KVCacheGroupSpec(["kda"], mamba_spec),
        ],
    )
    parallel_config = SimpleNamespace(
        decode_context_parallel_size=dcp_size,
        cp_kv_cache_interleave_size=1,
    )
    vllm_config = SimpleNamespace(
        parallel_config=parallel_config,
        cache_config=SimpleNamespace(mamba_cache_mode=mamba_cache_mode),
    )
    runner = SimpleNamespace(
        max_model_len=max_model_len,
        is_encoder_decoder=False,
        vllm_config=vllm_config,
        parallel_config=parallel_config,
    )

    class _CapturedWidths(Exception):
        pass

    captured: list[int] = []

    def capture_width(max_num_blocks: int, *_args, **_kwargs) -> int:
        captured.append(max_num_blocks)
        if len(captured) == 2:
            raise _CapturedWidths
        return max_num_blocks

    monkeypatch.setattr(model_runner_module, "get_block_table_width", capture_width)

    with pytest.raises(_CapturedWidths):
        GPUModelRunner.initialize_kv_cache(runner, kv_cache_config)

    # Attention KV is local to one of eight DCP ranks; KDA state is replicated
    # and therefore needs one table entry for every global 16-token page.
    assert captured == [86, expected]


def test_append_block_ids_rejects_write_past_row_capacity():
    """Reject an oversized staged write before it can corrupt the next row."""

    class _BlockTable:
        gpu = torch.empty((2, 4), dtype=torch.int32)

        def stage_write(self, *_args):
            pytest.fail("an oversized write must not be staged")

    block_tables = BlockTables.__new__(BlockTables)
    block_tables.num_kv_cache_groups = 1
    # append_block_ids only touches .gpu and .stage_write on each table.
    block_tables.block_tables = [_BlockTable()]  # type: ignore[list-item]  # duck-typed stub
    block_tables.num_blocks = SimpleNamespace(
        np=torch.tensor([[0, 3]], dtype=torch.int32)
    )

    with pytest.raises(
        RuntimeError,
        match=r"request 1, group 0 exceeds row capacity \(5 > 4\)",
    ):
        block_tables.append_block_ids(
            req_index=1,
            new_block_ids=([4, 5],),
            overwrite=False,
        )

    assert block_tables.num_blocks.np[0, 1] == 3


@contextlib.contextmanager
def _no_dummy_loras(
    lora_config: LoRAConfig | None, remove_lora: bool = True
) -> Iterator[None]:
    """No-op stand-in for LoRAModelRunnerMixin.maybe_setup_dummy_loras."""
    yield


def _make_capture_runner(
    monkeypatch: pytest.MonkeyPatch, captured: bool
) -> GPUModelRunner:
    """Minimal V2 runner for capture_model: fakes everything except the
    cudagraph_manager's needs_capture decision."""
    runner = GPUModelRunner.__new__(GPUModelRunner)
    runner.state_slots = None
    runner.model_state = SimpleNamespace(
        supports_mm_inputs=False, capture_inner_cudagraphs=Mock()
    )
    runner.cudagraph_manager = SimpleNamespace(
        needs_capture=lambda: captured,
        capture=lambda *args, **kwargs: None,
        capture_stream=object(),
    )
    runner.lora_config = None
    monkeypatch.setattr(runner, "maybe_setup_dummy_loras", _no_dummy_loras)
    runner.speculator = None
    runner.adaptive_verification = None
    runner.model = None
    runner.input_buffers = object()
    runner.pcp_manager = None
    runner.intermediate_tensors = None
    runner.block_tables = object()
    runner.attn_groups = object()
    runner.kv_cache_config = object()
    runner.use_aux_hidden_state_outputs = False
    runner.kv_connector = model_runner_module.NO_OP_KV_CONNECTOR
    return runner


def test_capture_model_locks_workspace_after_capture(monkeypatch):
    """A workspace resize after capture frees the buffer the captured graphs
    baked in, so capture_model must lock the workspace before returning
    (https://github.com/vllm-project/vllm/issues/55336)."""
    runner = _make_capture_runner(monkeypatch, captured=True)
    monkeypatch.setattr(
        model_runner_module, "freeze_gc_for_cudagraph_capture", contextlib.nullcontext
    )
    monkeypatch.setattr(torch.accelerator, "empty_cache", lambda: None)
    monkeypatch.setattr(
        torch.accelerator, "get_memory_info", lambda: (1 << 30, 1 << 30)
    )
    lock_calls = []
    monkeypatch.setattr(
        model_runner_module, "lock_workspace", lambda: lock_calls.append("lock")
    )

    runner.capture_model()

    assert lock_calls == ["lock"]
    assert runner.cudagraph_manager is not None
    capture_inner_cudagraphs = runner.model_state.capture_inner_cudagraphs
    assert isinstance(capture_inner_cudagraphs, Mock)
    capture_inner_cudagraphs.assert_called_once_with(
        runner.input_buffers,
        runner.block_tables,
        runner.attn_groups,
        runner.kv_cache_config,
        runner.cudagraph_manager.capture_stream,
    )


def test_capture_model_skips_lock_when_nothing_captured(monkeypatch):
    """With no graphs to capture (e.g. enforce_eager) there is nothing baked
    into the workspace, so the early return must not lock it."""
    runner = _make_capture_runner(monkeypatch, captured=False)
    lock_calls = []
    monkeypatch.setattr(
        model_runner_module, "lock_workspace", lambda: lock_calls.append("lock")
    )

    assert runner.capture_model() == 0
    assert lock_calls == []


def test_capture_model_profile_only_skips_lock(monkeypatch):
    """The memory-profiling capture pass runs before kernel warmup and the
    real capture; locking there would stop the warmup from growing the
    workspace to its scheduler-realistic size."""
    runner = _make_capture_runner(monkeypatch, captured=True)
    monkeypatch.setattr(
        model_runner_module, "freeze_gc_for_cudagraph_capture", contextlib.nullcontext
    )
    monkeypatch.setattr(torch.accelerator, "empty_cache", lambda: None)
    monkeypatch.setattr(
        torch.accelerator, "get_memory_info", lambda: (1 << 30, 1 << 30)
    )
    lock_calls = []
    monkeypatch.setattr(
        model_runner_module, "lock_workspace", lambda: lock_calls.append("lock")
    )

    runner.capture_model(profile_only=True)

    assert lock_calls == []


@pytest.mark.parametrize("profile_only", [False, True])
def test_capture_model_excludes_adaptive_verification_profiling_cache(
    monkeypatch, profile_only
):
    """Adaptive verification's cost-profiling dummy runs execute inside
    capture_model's free-memory window. The activations they leave in the
    allocator cache must not be counted as CUDA graph memory, which is taken
    from the KV cache. The memory-profiling pass skips them entirely: it runs
    on a throwaway KV cache and the real capture recalibrates."""
    runner = _make_capture_runner(monkeypatch, captured=True)
    monkeypatch.setattr(
        runner.cudagraph_manager, "captured_token_counts", lambda: [1, 2], raising=False
    )
    runner.step_timing = SimpleNamespace(collect=lambda: contextlib.nullcontext([]))
    curves: list[list] = []
    runner.adaptive_verification = SimpleNamespace(
        batches_to_profile=lambda sizes: [{"num_tokens": n} for n in sizes],
        set_initial_cost_curves=curves.append,
    )
    total, cached = 16 << 30, [0]

    def _dummy_run(num_tokens):
        cached[0] = 2 << 30

    monkeypatch.setattr(runner, "_dummy_run", _dummy_run)
    monkeypatch.setattr(
        model_runner_module, "freeze_gc_for_cudagraph_capture", contextlib.nullcontext
    )
    monkeypatch.setattr(
        torch.accelerator, "empty_cache", lambda: cached.__setitem__(0, 0)
    )
    monkeypatch.setattr(
        torch.accelerator, "get_memory_info", lambda: (total - cached[0], total)
    )
    monkeypatch.setattr(model_runner_module, "lock_workspace", lambda: None)

    assert runner.capture_model(profile_only=profile_only) == 0
    assert curves == ([] if profile_only else [[]])


@pytest.mark.parametrize("target_buffer", ["absent", "none", "tensor"])
def test_get_drafter_hidden_states_tolerates_missing_target_buffer(target_buffer):
    """Targets allocate the MTP hidden buffer only for hidden-state drafters."""
    hidden_states = torch.zeros(4, 8)
    buffer = torch.arange(16 * 8, dtype=torch.float32).view(16, 8)
    if target_buffer == "absent":
        model = SimpleNamespace()
    else:
        returned = buffer if target_buffer == "tensor" else None
        model = SimpleNamespace(get_mtp_target_hidden_states=lambda: returned)

    out = get_drafter_hidden_states(model, hidden_states)

    if target_buffer == "tensor":
        assert torch.equal(out, buffer[:4])
    else:
        assert out is hidden_states


def test_async_copy_to_np_does_not_alias_reused_buffer():
    buffer = torch.zeros(4, dtype=torch.int64)

    snapshot = async_copy_to_np(buffer)
    buffer.fill_(1)

    assert snapshot.tolist() == [0, 0, 0, 0]


@pytest.mark.parametrize(
    ("draft_dtype", "expected_dtype"),
    [("auto", "auto"), ("bfloat16", "bfloat16"), (None, "fp8_e4m3")],
)
@pytest.mark.parametrize("data_parallel_index", [0, 3])
@pytest.mark.parametrize(
    ("draft_moe", "draft_attn"),
    [(None, None), ("triton", AttentionBackendEnum.TRITON_ATTN)],
)
def test_standalone_draft_kv_dtype_override(
    monkeypatch, draft_dtype, expected_dtype, data_parallel_index, draft_moe, draft_attn
):
    @dataclass
    class Config:
        cache_config: CacheConfig
        parallel_config: ParallelConfig
        model_config: object
        kernel_config: KernelConfig
        attention_config: AttentionConfig
        quant_config: object | None = None

    target = Config(
        CacheConfig(cache_dtype="fp8_e4m3"),
        ParallelConfig(),
        object(),
        KernelConfig(moe_backend="flashinfer_cutlass"),
        AttentionConfig(backend=AttentionBackendEnum.FLASHINFER),
    )
    target.parallel_config.data_parallel_index = data_parallel_index
    speculator = object.__new__(ar_module.StandaloneARSpeculator)
    speculator.vllm_config = target
    speculator.draft_model_config = object()
    spec = SimpleNamespace(
        kv_cache_dtype=draft_dtype,
        moe_backend=draft_moe,
        attention_backend=draft_attn,
        draft_parallel_config=ParallelConfig(),
    )
    spec.apply_draft_overrides = partial(SpeculativeConfig.apply_draft_overrides, spec)
    speculator.speculative_config = spec
    get_model = Mock(return_value=object())
    monkeypatch.setattr(ar_module, "get_model", get_model)
    monkeypatch.setattr(ar_module, "set_model_tag", lambda _: contextlib.nullcontext())
    monkeypatch.setattr(ar_module, "get_draft_load_config", lambda _: LoadConfig())
    monkeypatch.setattr(
        ar_module, "get_pp_safe_draft_load_config", lambda load_config: load_config
    )

    speculator.load_draft_model(object(), set())

    draft_config = get_model.call_args.kwargs["vllm_config"]
    assert draft_config.cache_config.cache_dtype == expected_dtype
    assert speculator.attn_vllm_config.cache_config.cache_dtype == expected_dtype
    assert target.cache_config.cache_dtype == "fp8_e4m3"
    assert draft_config.parallel_config.data_parallel_index == data_parallel_index
    assert target.parallel_config.data_parallel_index == data_parallel_index
    assert draft_config.kernel_config.moe_backend == (draft_moe or "flashinfer_cutlass")
    assert draft_config.attention_config.backend == (
        draft_attn or AttentionBackendEnum.FLASHINFER
    )
    assert target.kernel_config.moe_backend == "flashinfer_cutlass"
    assert target.attention_config.backend == AttentionBackendEnum.FLASHINFER


class _StateOnlyModelState:
    """Generic model state for a state-only cache; its hooks are the defaults."""

    num_new_sampled_tokens_per_step = 1
    supports_mm_inputs = False
    state_slots: object | None = None

    def initialize_state_slots(self, bindings):
        from vllm.v1.worker.gpu.model_states.interface import ModelState

        ModelState.initialize_state_slots(self, bindings)

    def post_warmup(self):
        from vllm.v1.worker.gpu.model_states.interface import ModelState

        ModelState.post_warmup(self)

    def shutdown(self):
        from vllm.v1.worker.gpu.model_states.interface import ModelState

        ModelState.shutdown(self)

    def get_additional_cg_support(self):
        from vllm.v1.attention.backend import AttentionCGSupport

        return AttentionCGSupport.ALWAYS, None

    def initialize_kv_cache(self, cache_config, block_tables):
        return None


def _state_only_config(monkeypatch):
    from tests.v1.core.utils import create_scheduler
    from vllm.platforms import current_platform
    from vllm.v1.worker.gpu import buffer_utils

    monkeypatch.setattr(buffer_utils, "is_uva_available", lambda: False)
    monkeypatch.setattr(current_platform, "device_type", "cpu")
    config = create_scheduler(max_num_seqs=2, max_model_len=32).vllm_config
    config.cache_config.kv_cache_layout = "LBNHC"
    # The executor turns the extensible KV cache off for persistent state; the
    # config defaults it on for a CUDA host, whose device memory calls reject
    # this CPU runner.
    config.cache_config.enable_extensible_kv_cache = False
    return config


def _state_only_runner(monkeypatch, config, model_state):
    """A CPU runner whose ``initialize_kv_cache`` runs unmodified."""
    from vllm.config.compilation import CUDAGraphMode

    runner = GPUModelRunner.__new__(GPUModelRunner)
    runner.batch_sharder = None
    runner.draft_tokens_handler = Mock()
    runner.encoder_cache = None
    runner.state_slots = None
    runner.vllm_config = config
    runner.parallel_config = config.parallel_config
    runner.model_config = config.model_config
    runner.scheduler_config = config.scheduler_config
    runner.cache_config = config.cache_config
    runner.compilation_config = config.compilation_config
    runner.jit_warmup_registry = JitWarmupRegistry(config)
    runner.max_model_len = 32
    runner.is_encoder_decoder = False
    runner.dcp_size = 1
    runner.dcp_rank = 0
    runner.pcp_manager = None
    runner.speculator = None
    runner.req_states = []
    runner.input_buffers = SimpleNamespace(query_start_loc=None)
    runner.vocab_size = 100
    runner.max_num_reqs = 2
    runner.max_num_tokens = 32
    runner.device = torch.device("cpu")
    runner.model_state = model_state
    runner.model = torch.nn.Module()
    runner.decode_query_len = 1
    runner.lora_capture_cases = []
    runner.lora_config = None
    runner.aux_output_connector = None
    monkeypatch.setattr(
        config.compilation_config,
        "resolve_cudagraph_mode_and_sizes",
        lambda *a, **kw: CUDAGraphMode.NONE,
    )
    monkeypatch.setattr(model_runner_module, "ModelCudaGraphManager", Mock())
    return runner


def _state_plan(config):
    """Three pages of the generic test state spec."""
    from tests.v1.core.utils import create_state_spec, get_kv_cache_configs

    return get_kv_cache_configs(config, [{"state": create_state_spec()}], [96])[0]


def _initialized_state_runner(monkeypatch, model_state, **kwargs):
    config = _state_only_config(monkeypatch)
    runner = _state_only_runner(monkeypatch, config, model_state)
    runner.initialize_kv_cache(_state_plan(config), **kwargs)
    return runner


_CONTRIBUTORS = ("first_example_option", "second_example_option")


@pytest.mark.parametrize("is_profiling", [True, False], ids=["profiling", "serving"])
@pytest.mark.parametrize("contributors", [_CONTRIBUTORS, ()], ids=["named", "none"])
@pytest.mark.parametrize(
    "failure", [torch.OutOfMemoryError, RuntimeError], ids=["oom", "raise"]
)
def test_state_kv_allocation_failure_names_memory_contributors(
    monkeypatch, contributors, failure, is_profiling
):
    """A failed state allocation keeps its exception and notes the contributors.

    The allocation precedes every model-state hook, so core attributes it, and
    neither hook runs for the failing call.
    """
    import traceback

    from tests.v1.core.utils import create_state_spec, get_kv_cache_configs

    config = _state_only_config(monkeypatch)
    spec = create_state_spec(memory_contributors=contributors)
    plan = get_kv_cache_configs(config, [{"state": spec}], [96])[0]
    calls = []

    class _ModelState(_StateOnlyModelState):
        def initialize_state_slots(self, bindings):
            calls.append("initialize_state_slots")

        def initialize_kv_cache(self, cache_config, block_tables):
            calls.append("initialize_kv_cache")

    error = failure("injected allocation failure")

    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(model_runner_module, "init_kv_cache", fail)
    runner = _state_only_runner(monkeypatch, config, _ModelState())
    with pytest.raises(failure) as raised:
        runner.initialize_kv_cache(plan, is_profiling=is_profiling)
    assert raised.value is error
    assert str(error) == "injected allocation failure"
    assert calls == []
    notes = "".join(getattr(error, "__notes__", []))
    formatted = "".join(traceback.format_exception(error))
    if not contributors:
        assert not hasattr(error, "__notes__")
    for name in _CONTRIBUTORS:
        assert (name in notes) == bool(contributors)
        assert (name in formatted) == bool(contributors)


def test_state_only_cache_reaches_mrv2_initialization(monkeypatch):
    """Real MRv2 initialization exposes typed per-slot views from core allocation."""
    from tests.v1.core.utils import create_state_spec
    from vllm.model_executor.layers.state import PersistentStateLayerBase
    from vllm.v1.core.kv_cache_manager import KVCacheManager
    from vllm.v1.worker.gpu.attn_utils import get_kv_cache_spec

    config = _state_only_config(monkeypatch)
    state = create_state_spec()
    plan = _state_plan(config)

    PersistentStateLayerBase(state, prefix="state", vllm_config=config)
    assert get_kv_cache_spec(config)["state"] == state

    class _ModelState(_StateOnlyModelState):
        def initialize_kv_cache(self, cache_config, block_tables):
            assert len(cache_config.kv_cache_groups) == 1
            assert block_tables.num_kv_cache_groups == 1

    runner = _state_only_runner(monkeypatch, config, _ModelState())
    runner.initialize_kv_cache(plan, is_profiling=True)
    assert runner.block_tables.block_sizes == [1]
    assert runner.state_slots is not None
    assert runner.state_slots is runner.model_state.state_slots
    allocator = KVCacheManager(
        kv_cache_config=plan,
        max_model_len=32,
        scheduler_block_size=1,
        hash_block_size=1,
        enable_caching=False,
    ).coordinator.state_managers["state"]
    a, b = allocator.allocate("a"), allocator.allocate("b")
    views = runner.state_slots.views["state"]
    views.bind("a", a)
    views.bind("b", b)
    av, bv = views.view("a", a), views.view("b", b)
    assert av["hidden"].dtype == torch.float32
    assert av["count"].dtype == torch.int64
    assert av["hidden"].shape == (4,)
    assert torch.all(av["hidden"] == 1)
    assert torch.all(av["count"] == 0)
    assert av["hidden"].untyped_storage() == bv["count"].untyped_storage()
    av["hidden"].fill_(11)
    bv["hidden"].fill_(13)
    assert torch.all(av["hidden"] == 11) and torch.all(bv["hidden"] == 13)
    allocator.retire("a", 0)
    assert allocator.release(a, 0)
    c = allocator.allocate("c")
    views.bind("c", c)
    with pytest.raises(ValueError, match="Stale"):
        views.view("a", a)
    with pytest.raises(ValueError, match="Stale"):
        views.bind("a", a)
    assert torch.all(bv["hidden"] == 13)
    assert torch.all(views.view("c", c)["hidden"] == 1)
    with pytest.raises(ValueError, match="Stale"):
        allocator.complete(a, 10)


def _state_bindings(num_pages=4):
    from tests.v1.core.utils import create_state_spec
    from vllm.v1.kv_cache_interface import KVCacheGroupSpec
    from vllm.v1.worker.gpu.state_slots import StateSlotBindings

    spec = create_state_spec()
    raw = torch.zeros(num_pages * spec.page_size_bytes, dtype=torch.uint8)
    return StateSlotBindings([KVCacheGroupSpec(["state"], spec)], {"state": raw}), raw


@contextlib.contextmanager
def _runner_log_records() -> Iterator[list[logging.LogRecord]]:
    """Collect the runner module's log records; its logger does not propagate."""
    records: list[logging.LogRecord] = []

    class _Handler(logging.Handler):
        def emit(self, record):
            records.append(record)

    handler = _Handler()
    model_runner_module.logger.addHandler(handler)
    try:
        yield records
    finally:
        model_runner_module.logger.removeHandler(handler)


def _stub_device(monkeypatch, events=None):
    record = events.append if events is not None else lambda _: None
    monkeypatch.setattr(torch.accelerator, "synchronize", lambda: record("sync"))
    monkeypatch.setattr(torch.accelerator, "empty_cache", lambda: None)


def test_profiling_then_serving_initialization_gets_fresh_bindings(monkeypatch):
    """Profiling and serving initialization each get their own bindings, a
    phase flag, and a startup window that admits only their own access."""
    from vllm.v1.worker.gpu.cudagraph_utils import _teardown_profiling_state
    from vllm.v1.worker.gpu.state_slots import StartupPoolAccess, StateSlotBindings

    seen: list[
        tuple[StateSlotBindings, StartupPoolAccess, tuple[bool, list[bool]]]
    ] = []
    addresses: list[int] = []

    class _ModelState(_StateOnlyModelState):
        def initialize_state_slots(self, bindings):
            super().initialize_state_slots(bindings)
            stale = []
            for _, old_access, _ in seen:
                try:
                    old_access.component_pools  # noqa: B018
                except ValueError:
                    stale.append(True)
                else:
                    stale.append(False)
            access = bindings.startup_pool_access("state")
            hidden = access.component_pools["hidden"]
            assert hidden.shape == (3, 4)
            addresses.append(hidden.data_ptr())
            seen.append((bindings, access, (bindings.is_profiling, stale)))

    config = _state_only_config(monkeypatch)
    runner = _state_only_runner(monkeypatch, config, _ModelState())
    _stub_device(monkeypatch)
    runner.initialize_kv_cache(_state_plan(config), is_profiling=True)
    _teardown_profiling_state(runner)
    assert runner.state_slots is None
    runner.initialize_kv_cache(_state_plan(config))
    (profiling, _, first), (serving, _, second) = seen
    assert first == (True, []) and second == (False, [True])
    assert profiling is not serving and runner.state_slots is serving
    # The profiling bindings are still alive, so distinct addresses mean
    # distinct allocations rather than allocator reuse.
    profiling_address, serving_address = addresses
    assert profiling_address != serving_address


def test_shutdown_calls_model_state_hook_before_releasing_state(monkeypatch):
    """After the device synchronizes, the model state releases its resources
    while its bindings and KV cache still exist; a repeated shutdown does not
    call it again."""
    events = []

    class _ModelState(_StateOnlyModelState):
        def shutdown(self):
            events.append(
                (
                    "hook",
                    getattr(runner, "model_state", None) is self,
                    runner.state_slots is bindings,
                    self.state_slots is bindings,
                    len(runner.kv_caches),
                    "state" in runner.kv_caches_by_layer,
                )
            )

    runner = _initialized_state_runner(monkeypatch, _ModelState())
    bindings = runner.state_slots
    assert bindings is not None
    runner.kv_caches.append(torch.zeros(1))
    _stub_device(monkeypatch, events)
    runner.shutdown()
    assert not hasattr(runner, "model_state")
    assert runner.state_slots is None
    assert runner.kv_caches == [] and runner.kv_caches_by_layer == {}
    runner.shutdown()
    assert events == ["sync", ("hook", True, True, True, 1, True), "sync"]


def test_shutdown_hook_tolerates_incomplete_initialization(monkeypatch):
    """Shutdown before the KV cache exists still reaches the model state."""
    calls = []

    class _ModelState(_StateOnlyModelState):
        def shutdown(self):
            assert self.state_slots is None
            calls.append(runner.state_slots)

    config = _state_only_config(monkeypatch)
    runner = _state_only_runner(monkeypatch, config, _ModelState())
    _stub_device(monkeypatch)
    runner.shutdown()
    assert calls == [None]
    assert not hasattr(runner, "model_state")


def test_shutdown_hook_failure_is_logged_and_cleanup_continues(monkeypatch):
    error = RuntimeError("hook failed")

    class _ModelState(_StateOnlyModelState):
        def shutdown(self):
            raise error

    runner = _initialized_state_runner(monkeypatch, _ModelState())
    runner.kv_caches.append(torch.zeros(1))
    assert "state" in runner.kv_caches_by_layer
    _stub_device(monkeypatch)
    with _runner_log_records() as records:
        runner.shutdown()
    failures = [r for r in records if r.exc_info is not None]
    assert len(failures) == 1
    assert failures[0].levelno >= logging.ERROR
    exc_info = failures[0].exc_info
    assert exc_info is not None and exc_info[1] is error
    assert exc_info[2] is not None
    assert runner.kv_caches == [] and runner.kv_caches_by_layer == {}
    assert runner.state_slots is None
    assert not hasattr(runner, "model_state")
    assert not hasattr(runner, "model")


def test_shutdown_cleanup_failure_after_hook_still_propagates(monkeypatch):
    """Only the model state's hook is contained; later cleanup failures are not."""
    calls = []

    class _ModelState(_StateOnlyModelState):
        def shutdown(self):
            calls.append("hook")

    def fail(vllm_config):
        raise RuntimeError("cleanup failed")

    runner = _initialized_state_runner(monkeypatch, _ModelState())
    _stub_device(monkeypatch)
    monkeypatch.setattr(model_runner_module, "free_before_shutdown", fail)
    with pytest.raises(RuntimeError, match="cleanup failed"):
        runner.shutdown()
    assert calls == ["hook"]


def test_startup_pool_access_expires_after_initialize_state_slots(monkeypatch):
    """Unowned resident pools are reachable only inside slot initialization.

    The startup access covers every page, including unbound ones, and is
    refused once ``initialize_state_slots`` has returned.
    """
    accesses = []

    class _ModelState(_StateOnlyModelState):
        def initialize_state_slots(self, bindings):
            super().initialize_state_slots(bindings)
            access = bindings.startup_pool_access("state")
            hidden = access.component_pools["hidden"]
            assert hidden.shape == (3, 4) and hidden.dtype == torch.float32
            assert access.spec is bindings.views["state"].spec
            accesses.append(access)

    runner = _initialized_state_runner(monkeypatch, _ModelState(), is_profiling=True)
    (access,) = accesses
    with pytest.raises(ValueError):
        access.component_pools  # noqa: B018
    with pytest.raises(ValueError):
        access.spec  # noqa: B018
    with pytest.raises(ValueError):
        runner.state_slots.startup_pool_access("state")


def test_startup_pool_access_expires_when_initialize_state_slots_raises(
    monkeypatch,
):
    accesses = []

    class _ModelState(_StateOnlyModelState):
        def initialize_state_slots(self, bindings):
            super().initialize_state_slots(bindings)
            accesses.append(bindings.startup_pool_access("state"))
            raise RuntimeError("capture failed")

    config = _state_only_config(monkeypatch)
    runner = _state_only_runner(monkeypatch, config, _ModelState())
    with pytest.raises(RuntimeError, match="capture failed"):
        runner.initialize_kv_cache(_state_plan(config), is_profiling=True)
    (access,) = accesses
    with pytest.raises(ValueError):
        access.component_pools  # noqa: B018
    with pytest.raises(ValueError):
        runner.state_slots.startup_pool_access("state")


def test_startup_pool_access_windows_are_per_call_and_per_bindings():
    """Each window admits only its own accesses on its own bindings."""
    slots, _ = _state_bindings()
    other, _ = _state_bindings()
    with pytest.raises(ValueError):
        slots.startup_pool_access("state")
    with slots.startup_access_window():
        first = slots.startup_pool_access("state")
        assert first.component_pools["hidden"].shape == (4, 4)
        with pytest.raises(RuntimeError), slots.startup_access_window():
            pass
        assert first.component_pools["hidden"].shape == (4, 4)
        with pytest.raises(ValueError):
            other.startup_pool_access("state")
        with pytest.raises(ValueError):
            slots.startup_pool_access("missing")
    with (
        pytest.raises(RuntimeError, match="raised"),
        slots.startup_access_window(),
    ):
        second = slots.startup_pool_access("state")
        raise RuntimeError("raised")
    with slots.startup_access_window():
        with pytest.raises(ValueError):
            first.component_pools  # noqa: B018
        with pytest.raises(ValueError):
            second.spec  # noqa: B018
        assert slots.startup_pool_access("state").component_pools.keys() == {
            "hidden",
            "count",
        }
    with pytest.raises(ValueError):
        slots.startup_pool_access("state")


def test_startup_access_preserves_ownership_and_bind_initialization():
    """Startup access writes nothing and changes no ownership; the first bind
    of a slot and every newer generation write the initial values, and
    rebinding the same handle keeps the page."""
    from vllm.v1.kv_cache_interface import StateSlotHandle

    slots, raw = _state_bindings()
    views = slots.views["state"]
    page_bytes = views.spec.page_size_bytes
    a = StateSlotHandle("epoch", 1, 1)
    views.bind("a", a)
    page = views.view("a", a)
    page["hidden"].fill_(19)
    page["count"].fill_(5)
    before = raw.clone()
    with slots.startup_access_window():
        pools = slots.startup_pool_access("state").component_pools
        assert torch.all(pools["hidden"][1] == 19)
        assert torch.all(pools["hidden"][[0, 2, 3]] == 0)
    assert torch.equal(raw, before)

    views.bind("a", a)
    page = views.view("a", a)
    assert torch.all(page["hidden"] == 19) and torch.all(page["count"] == 5)

    b = StateSlotHandle("epoch", 2, 1)
    raw[2 * page_bytes : 3 * page_bytes].fill_(7)  # Whatever an unbound page held.
    views.bind("b", b)
    page = views.view("b", b)
    assert torch.all(page["hidden"] == 1) and torch.all(page["count"] == 0)

    newer = StateSlotHandle("epoch", 1, 2)
    views.bind("c", newer)
    page = views.view("c", newer)
    assert torch.all(page["hidden"] == 1) and torch.all(page["count"] == 0)
    with pytest.raises(ValueError, match="Stale"):
        views.view("a", a)


def _pool_layout_with_weak_refs():
    """Take a layout, check it against both exposures, and return it with weak
    references to everything it must not keep alive."""
    import weakref

    from vllm.v1.kv_cache_interface import StateSlotHandle

    slots, raw = _state_bindings()
    with slots.startup_access_window():
        startup = dict(slots.startup_pool_access("state").component_pools)
    step = SchedulerOutput.make_empty()
    step.state_step_seq = 1
    a = StateSlotHandle("epoch", 1, 1)
    step.state_slot_handles = {"a": {"state": a}}
    slots.begin_step(step, SimpleNamespace(wait_stream=lambda _: None))
    slots.activate_requests()
    rows = dict(slots.bind_rows("state", ("a",), (a,)).component_pools)

    layout = slots.pool_layout("state")
    assert layout.keys() == startup.keys() == rows.keys() == {"hidden", "count"}
    for name, entry in layout.items():
        for tensor in (startup[name], rows[name]):
            assert (entry.data_ptr, entry.shape, entry.stride, entry.dtype) == (
                tensor.data_ptr(),
                tuple(tensor.shape),
                tensor.stride(),
                tensor.dtype,
            )
    again = slots.pool_layout("state")
    assert again == layout and again is not layout
    with pytest.raises(ValueError):
        slots.pool_layout("missing")
    refs = [weakref.ref(obj) for obj in (slots, raw, *startup.values(), *rows.values())]
    return layout, refs


def test_pool_layout_is_a_detached_snapshot_of_the_exposed_pools():
    import gc

    layout, refs = _pool_layout_with_weak_refs()
    gc.collect()
    assert all(ref() is None for ref in refs)
    hidden = layout["hidden"]
    assert hidden.shape == (4, 4) and hidden.dtype == torch.float32
    assert isinstance(hidden.data_ptr, int) and isinstance(hidden.stride, tuple)
    with pytest.raises(TypeError):
        layout["hidden"] = layout["count"]  # type: ignore[index]
    with pytest.raises(AttributeError):
        hidden.data_ptr = 0  # type: ignore[misc]


@pytest.fixture
def warmup_worker(monkeypatch):
    """A V2 worker over a CPU runner whose warmup stages record events."""
    import vllm.compilation.compiler_interface as compiler_interface
    import vllm.v1.worker.gpu_worker as gpu_worker
    from vllm.config.compilation import CompilationMode, CUDAGraphMode

    def make(
        model_state,
        events,
        *,
        with_state=True,
        is_last_rank=True,
        is_pooling_model=False,
        enforce_eager=False,
        compile=True,
        jit_warmup=True,
    ):
        config = _state_only_config(monkeypatch)
        runner = _state_only_runner(monkeypatch, config, model_state)
        if with_state:
            runner.initialize_kv_cache(_state_plan(config))
        compilation = config.compilation_config
        compilation.mode = (
            CompilationMode.VLLM_COMPILE if compile else CompilationMode.NONE
        )
        compilation.backend = "inductor"
        compilation.compile_sizes = [8]
        compilation.cudagraph_mode = CUDAGraphMode.NONE
        monkeypatch.setattr(compilation, "get_compile_ranges", lambda: [])
        config.kernel_config.enable_jit_warmup = jit_warmup
        config.model_config.enforce_eager = enforce_eager
        runner.is_pooling_model = is_pooling_model

        def dummy_run(num_tokens, **_):
            events.append(f"dummy_run:{num_tokens}")
            return None, None

        def record(name):
            return lambda *_, **__: events.append(name)

        monkeypatch.setattr(runner, "_dummy_run", dummy_run)
        monkeypatch.setattr(runner, "_dummy_sampler_run", record("sampler_run"))
        monkeypatch.setattr(
            runner, "capture_model", lambda: events.append("capture_model") or 0
        )
        monkeypatch.setattr(runner, "get_model", lambda: runner.model)
        monkeypatch.setattr(gpu_worker, "has_compiled_submodule", lambda _: True)
        monkeypatch.setattr(
            gpu_worker,
            "get_pp_group",
            lambda: SimpleNamespace(is_last_rank=is_last_rank),
        )
        monkeypatch.setattr(gpu_worker, "kernel_warmup", record("kernel_warmup"))
        monkeypatch.setattr(gpu_worker, "warmup_kernels", record("warmup_kernels"))
        monkeypatch.setattr(gpu_worker, "set_random_seed", record("set_random_seed"))
        monkeypatch.setattr(
            compiler_interface,
            "trigger_inductor_lazy_init",
            record("inductor_lazy_init"),
        )
        monkeypatch.setattr(gpu_worker, "freeze_gc_heap", record("freeze_gc"))
        monkeypatch.setattr(gpu_worker, "maybe_attach_gc_debug_callback", lambda: None)
        monkeypatch.setattr(gpu_worker, "enable_gpu_sync_check", record("sync_check"))
        worker = gpu_worker.Worker.__new__(gpu_worker.Worker)
        worker.vllm_config = config
        worker.model_config = config.model_config
        worker.cache_config = config.cache_config
        worker.compilation_config = compilation
        worker.parallel_config = config.parallel_config
        worker.scheduler_config = config.scheduler_config
        worker.model_runner = runner
        worker.device = torch.device("cpu")
        monkeypatch.setattr(
            worker, "_get_cudagraph_capture_context", contextlib.nullcontext
        )
        monkeypatch.setattr(worker, "_maybe_activate_jit_monitor", record("jit_mon"))
        return worker

    return make


_WARMUP_CASES = {
    "last-rank": {},
    "earlier-rank": {"is_last_rank": False},
    "pooling": {"is_pooling_model": True},
    "eager": {"enforce_eager": True},
    "no-compile": {"compile": False},
    "no-jit-warmup": {"jit_warmup": False},
    "no-state": {"with_state": False},
}


@pytest.mark.parametrize("case", _WARMUP_CASES.values(), ids=_WARMUP_CASES.keys())
def test_post_warmup_hook_runs_once_after_every_warmup(warmup_worker, case):
    """The model state verifies once, after every enabled warmup stage and
    before the worker starts JIT monitoring, freezes the GC heap and checks
    syncs, whatever the rank, pooling, eager, compile or state setting."""
    events = []

    class _ModelState(_StateOnlyModelState):
        def post_warmup(self):
            events.append("post_warmup")

    worker = warmup_worker(_ModelState(), events, **case)
    worker._compile_or_warm_up_model()
    compile = case.get("compile", True)
    stages = ["kernel_warmup", "set_random_seed"]
    if compile:
        stages += ["dummy_run:8", "inductor_lazy_init"]
    if case.get("jit_warmup", True):
        stages.append("warmup_kernels")
    if not case.get("enforce_eager", False):
        stages.append("capture_model")
    if case.get("is_last_rank", True) and not case.get("is_pooling_model", False):
        stages += ["dummy_run:2", "sampler_run"]
    assert events.count("post_warmup") == 1
    hook = events.index("post_warmup")
    assert sorted(events[:hook]) == sorted(stages)
    assert events[hook + 1 :] == ["jit_mon", "freeze_gc", "sync_check"]


def test_default_lifecycle_hooks_let_startup_and_shutdown_finish(
    monkeypatch, warmup_worker
):
    """A model state that keeps the default hooks starts and shuts down."""
    events: list[str] = []
    worker = warmup_worker(_StateOnlyModelState(), events)
    worker._compile_or_warm_up_model()
    assert events[-3:] == ["jit_mon", "freeze_gc", "sync_check"]
    runner = worker.model_runner
    assert "state" in runner.kv_caches_by_layer
    _stub_device(monkeypatch)
    with _runner_log_records() as records:
        runner.shutdown()
    assert not [r for r in records if r.levelno >= logging.ERROR or r.exc_info]
    assert not hasattr(runner, "model_state") and runner.state_slots is None
    assert runner.kv_caches_by_layer == {}


def test_post_warmup_hook_failure_fails_startup(warmup_worker):
    events = []

    class _ModelState(_StateOnlyModelState):
        def post_warmup(self):
            events.append("post_warmup")
            raise RuntimeError("signature changed after warmup")

    worker = warmup_worker(_ModelState(), events)
    with pytest.raises(RuntimeError, match="signature changed"):
        worker._compile_or_warm_up_model()
    assert events[-1] == "post_warmup"


def test_state_slot_rows_validate_step_handles_and_alias_resident_pools(monkeypatch):
    from dataclasses import replace

    from tests.v1.core.utils import create_state_spec
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.kv_cache_interface import KVCacheGroupSpec, StateSlotHandle
    from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT
    from vllm.v1.worker.gpu.state_slots import StateSlotBindings

    spec = create_state_spec()
    raw = torch.zeros(4 * spec.page_size_bytes, dtype=torch.uint8)
    slots = StateSlotBindings([KVCacheGroupSpec(["state"], spec)], {"state": raw})
    main = SimpleNamespace(wait_stream=lambda _: None)
    step = SchedulerOutput.make_empty()
    step.state_step_seq = 1
    a, b = (StateSlotHandle("epoch", i, 1) for i in (1, 3))
    step.state_slot_handles = {"a": {"state": a}, "b": {"state": b}}
    with pytest.raises(ValueError, match="live"):
        slots.bind_rows("state", ("b", "a"), (b, a))
    slots.begin_step(step, main)
    with pytest.raises(ValueError, match="live"):
        slots.bind_rows("state", ("b", "a"), (b, a))
    slots.activate_requests()
    access = slots.bind_rows("state", ("b", "a"), (b, a))
    assert access.step_seq == 1 and access.slot_indices_cpu == (3, 1)
    pools = access.component_pools
    assert pools["hidden"].untyped_storage() == raw.untyped_storage()
    assert pools["hidden"].shape == (4, 4)
    pools["hidden"][3].fill_(19)
    assert torch.all(slots.views["state"].view("b", b)["hidden"] == 19)
    assert torch.all(pools["hidden"][1] == 1)
    assert torch.all(pools["hidden"][[0, 2]] == 0)
    for ids, handles in [
        (("a", "a"), (a, a)),
        (("a", "b"), (b, a)),
        (("a",), (replace(a, engine_epoch="old"),)),
        (("a",), (replace(a, generation=2),)),
        (("a",), (replace(a, slot=0),)),
        (("a",), (replace(a, slot=4),)),
        (("a", "b"), (a,)),
    ]:
        with pytest.raises(ValueError):
            slots.bind_rows("state", ids, handles)
    with pytest.raises(ValueError, match="group"):
        slots.bind_rows("other", ("a",), (a,))
    slots.fail_requests({"a"})
    access.validate_current()  # Failure reporting does not invalidate the step.
    monkeypatch.setattr(
        torch.cuda, "Event", lambda **_: SimpleNamespace(record=lambda _: None)
    )
    slots.finish_step(EMPTY_MODEL_RUNNER_OUTPUT, main)
    with pytest.raises(ValueError, match="Stale"):
        access.validate_current()
    slots.begin_step(step, main)  # Even a repeated sequence/handle is a new scope.
    slots.activate_requests()
    with pytest.raises(ValueError, match="Stale"):
        access.validate_current()
    fresh = slots.bind_rows("state", ("a",), (a,))
    slots.views["state"].bind("replacement", replace(a, generation=2))
    before = raw.clone()
    with pytest.raises(ValueError, match="Stale"):
        fresh.validate_current()
    assert torch.equal(raw, before)


def test_state_completion_waits_for_postprocess_and_registered_stream(monkeypatch):
    """Completion follows postprocessing and all state-writing streams."""
    from tests.v1.core.utils import create_state_spec
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.kv_cache_interface import StateSlotHandle
    from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT
    from vllm.v1.worker.gpu import async_utils
    from vllm.v1.worker.gpu.state_slots import StateSlotBindings

    events = []
    main = SimpleNamespace(wait_stream=lambda stream: events.append("join"))

    class Event:
        def __init__(self, **kwargs):
            pass

        def record(self, stream):
            self.kind = "state" if stream is main else "copy"
            events.append(self.kind + "-record")

        def synchronize(self):
            events.append(self.kind + "-wait")

    monkeypatch.setattr(torch.cuda, "Event", Event)
    slots = StateSlotBindings(
        [KVCacheGroupSpec(["state"], create_state_spec())],
        {"state": torch.zeros(96, dtype=torch.int8)},
    )
    slots.register_writing_stream(
        SimpleNamespace(wait_stream=lambda stream: events.append("ready"))
    )
    scheduled = SchedulerOutput.make_empty()
    scheduled.state_step_seq = 7
    scheduled.state_slot_handles = {"a": {"state": StateSlotHandle("epoch", 1, 1)}}
    slots.begin_step(scheduled, main)
    runner = GPUModelRunner.__new__(GPUModelRunner)
    runner.batch_sharder = None
    runner.draft_tokens_handler = Mock()
    runner.encoder_cache = None
    runner.state_slots = slots
    runner.main_stream = main
    runner.output_copy_stream = SimpleNamespace(
        wait_stream=lambda stream: events.append("copy-join")
    )
    runner.is_last_pp_rank = True
    runner.pcp_manager = runner.pp_handler = runner.speculator = None
    runner.aux_output_connector = None
    runner.num_speculative_steps = 0
    runner.check_ep_fault = False
    runner.eplb = SimpleNamespace(step=Mock())
    runner.kv_connector = SimpleNamespace(post_forward=Mock(return_value=None))
    runner.model = SimpleNamespace(compute_logits=Mock())
    runner.req_states = SimpleNamespace(
        all_token_ids=SimpleNamespace(gpu=None),
        num_computed_tokens=SimpleNamespace(gpu=None),
        prompt_len=SimpleNamespace(np=None),
    )
    runner.prompt_logprobs_worker = SimpleNamespace(
        compute_prompt_logprobs=Mock(return_value={}),
        compute_prompt_token_id_logprobs=Mock(return_value={}),
    )
    monkeypatch.setattr(
        runner,
        "sample",
        lambda *a: (
            SimpleNamespace(
                sampled_token_ids=torch.tensor([[1]]),
                logprobs_tensors=None,
                num_nans=None,
                sampling_mask_tensors=None,
            ),
            torch.tensor([1]),
            torch.tensor([0]),
        ),
    )
    monkeypatch.setattr(
        runner, "postprocess_sampled", lambda *a: events.append("postprocess")
    )
    runner.execute_model_state = ExecuteModelState(
        input_batch=SimpleNamespace(
            req_ids=["a"], idx_mapping=None, query_start_loc=None
        ),
        attn_metadata=None,
        slot_mappings_by_layer=None,
        hidden_states=torch.empty(1, 1),
        aux_hidden_states=None,
        dp_sync_state=None,
        finished_req_ids=set(),
        ec_connector_output=None,
        cudagraph_stats=None,
        num_spec_tokens_to_schedule=0,
    )

    monkeypatch.setattr(async_utils, "stream", lambda *args: contextlib.nullcontext())
    monkeypatch.setattr(async_utils, "async_copy_to_np", lambda tensor: tensor.numpy())
    pending = runner.sample_tokens(None)
    assert EMPTY_MODEL_RUNNER_OUTPUT.state_completion is None
    expected = [
        "join",
        "ready",
        "copy-join",
        "copy-record",
        "postprocess",
        "join",
        "state-record",
    ]
    assert events == expected
    result = pending.get_output()
    assert events == expected + ["state-wait", "copy-wait"]
    assert result.sampled_token_ids == [[1]]
    assert result.state_completion.step_seq == 7
    assert result.state_completion.handles == scheduled.state_slot_handles
    assert EMPTY_MODEL_RUNNER_OUTPUT.state_completion is None


@pytest.mark.parametrize("override", [2, 4])
def test_persistent_state_profiling_ignores_block_override(override):
    from tests.v1.core.utils import (
        create_scheduler,
        create_state_spec,
        get_kv_cache_configs,
    )
    from vllm.v1.worker.gpu.cudagraph_utils import _init_minimal_kv_cache_for_profiling

    config = create_scheduler(max_num_seqs=2, max_model_len=32).vllm_config
    config.cache_config.kv_cache_layout = "LBNHC"
    config.cache_config.num_gpu_blocks_override = override
    spec = create_state_spec()
    assert (
        get_kv_cache_configs(config, [{"state": spec}], [128])[0].num_blocks == override
    )
    runner = SimpleNamespace(
        vllm_config=config,
        cache_config=config.cache_config,
        max_num_reqs=2,
        get_kv_cache_spec=lambda: {"state": spec},
        initialize_kv_cache=Mock(),
    )
    _init_minimal_kv_cache_for_profiling(runner)
    assert runner.initialize_kv_cache.call_args.args[0].num_blocks == 3
    assert config.cache_config.num_gpu_blocks_override == override


@pytest.mark.parametrize("lose_ack", [False, True])
def test_state_cleanup_is_fenced_through_real_runner(monkeypatch, lose_ack):
    from tests.v1.core.test_scheduler import _state_output, _state_scheduler
    from tests.v1.core.utils import create_requests, create_state_spec
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT
    from vllm.v1.request import RequestStatus
    from vllm.v1.worker.gpu.state_slots import StateSlotBindings

    scheduler = _state_scheduler()
    (req,) = create_requests(num_requests=1, num_tokens=4)
    scheduler.add_request(req)
    step = scheduler.schedule()
    slots = StateSlotBindings(
        [KVCacheGroupSpec(["state"], create_state_spec())],
        {"state": torch.zeros(96, dtype=torch.int8)},
    )
    events = []
    main = SimpleNamespace(wait_stream=lambda stream: events.append("join"))
    writer = SimpleNamespace(wait_stream=lambda stream: events.append("ready"))
    slots.register_writing_stream(writer)

    class Event:
        def __init__(self, **kwargs):
            pass

        def record(self, stream):
            events.append("record")

        def synchronize(self):
            events.append("wait")

    monkeypatch.setattr(torch.cuda, "Event", Event)
    slots.begin_step(step, main)
    slots.activate_requests()
    slots.finish_step(_state_output(step), main).get_output()
    scheduler.update_from_output(step, _state_output(step))
    scheduler.finish_requests(req.request_id, RequestStatus.FINISHED_ABORTED)
    manager = scheduler.kv_cache_manager.coordinator.state_managers["state"]
    assert manager.num_free_slots == 1
    cleanup = scheduler.schedule()
    old = cleanup.state_cleanup_handles[req.request_id]["state"]
    events.clear()

    def remove(request_id):
        assert slots.get_handle(request_id, "state") == old
        slots.views["state"].view(request_id, old)["hidden"].fill_(17)
        events.append("cleanup")

    runner = GPUModelRunner.__new__(GPUModelRunner)
    runner.batch_sharder = None
    runner.draft_tokens_handler = Mock()
    runner.state_slots = slots
    runner.main_stream = main
    runner.model_state = SimpleNamespace(
        remove_request=remove, prepare_step=lambda *args, **kwargs: None
    )
    runner.req_states = SimpleNamespace(remove_request=lambda _: 0)
    runner.lora_state = SimpleNamespace(remove_request=lambda _: None)
    runner.pooling_runner = runner.pp_handler = runner.encoder_cache = None
    runner.speculator = None
    runner.prompt_logprobs_worker = runner.aux_output_connector = None
    for name in ("update_pp_decode_requests", "add_requests", "update_requests"):
        setattr(runner, name, lambda *args: None)
    runner.block_tables = SimpleNamespace(apply_staged_writes=lambda: None)
    runner.kv_connector = SimpleNamespace(
        no_forward=lambda _: EMPTY_MODEL_RUNNER_OUTPUT
    )
    monkeypatch.setattr(
        runner, "_merge_ec_connector_no_forward", lambda _, output: output
    )
    pending = runner.execute_model(cleanup)
    assert isinstance(pending, AsyncModelRunnerOutput)
    assert events.index("cleanup") < events.index("record")
    assert events[-2:] == ["join", "record"]
    assert manager.num_free_slots == 1
    output = pending.get_output()
    assert events[-1] == "wait"
    assert output.state_completion is not None
    assert output.state_completion.cleanup_handles == cleanup.state_cleanup_handles
    if lose_ack:
        output.state_completion = None
        scheduler.update_from_output(cleanup, output)
        assert manager.num_free_slots == 1
        assert scheduler.has_requests()
        cleanup = scheduler.schedule()
        assert not cleanup.finished_req_ids
        pending = runner.execute_model(cleanup)
        assert isinstance(pending, AsyncModelRunnerOutput)
        output = pending.get_output()
        assert events.count("cleanup") == 1
    scheduler.update_from_output(cleanup, output)
    assert manager.num_free_slots == 2
    assert not scheduler.has_requests()
    assert EMPTY_MODEL_RUNNER_OUTPUT.state_completion is None

    # Reuse the cleaned slot with the same request ID.
    replacement = manager.allocate(req.request_id)
    assert replacement.slot == old.slot
    assert replacement.generation > old.generation
    rebind = SchedulerOutput.make_empty()
    rebind.state_step_seq = cleanup.state_step_seq + 1
    rebind.state_slot_handles = {req.request_id: {"state": replacement}}

    def add_requests(step):
        assert slots.get_handle(req.request_id, "state") == replacement
        view = slots.views["state"].view(req.request_id, replacement)
        assert torch.all(view["hidden"] == 1)
        with pytest.raises(ValueError, match="Stale"):
            slots.views["state"].view(req.request_id, old)

    monkeypatch.setattr(runner, "add_requests", add_requests)
    pending = runner.execute_model(rebind)
    assert isinstance(pending, AsyncModelRunnerOutput)
    result = pending.get_output()
    assert result.state_completion is not None
    assert result.state_completion.handles == rebind.state_slot_handles
    assert events.count("cleanup") == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_state_completion_cuda_delayed_auxiliary_write(monkeypatch):
    """An output copy cannot acknowledge or recycle an unfinished state writer."""
    from concurrent.futures import ThreadPoolExecutor, TimeoutError
    from threading import Event

    from tests.v1.core.test_scheduler import _state_output, _state_scheduler
    from tests.v1.core.utils import create_requests, create_state_spec
    from vllm.v1.request import RequestStatus
    from vllm.v1.worker.gpu.state_slots import StateSlotBindings

    scheduler = _state_scheduler()
    (request,) = create_requests(num_requests=1, num_tokens=4)
    scheduler.add_request(request)
    step = scheduler.schedule()
    manager = scheduler.kv_cache_manager.coordinator.state_managers["state"]
    peer = manager.allocate("peer")
    assert peer is not None and manager.num_free_slots == 0
    handle = step.state_slot_handles[request.request_id]["state"]
    main = torch.cuda.current_stream()
    writer, copier = torch.cuda.Stream(), torch.cuda.Stream()
    spec = create_state_spec()
    slots = StateSlotBindings(
        [KVCacheGroupSpec(["state"], spec)],
        {
            "state": torch.zeros(
                3 * spec.page_size_bytes, dtype=torch.uint8, device="cuda"
            )
        },
    )
    slots.register_writing_stream(writer)
    slots.begin_step(step, main)
    slots.activate_requests()
    resident = slots.views["state"].view(request.request_id, handle)["hidden"]
    sampled = torch.tensor([[42]], device="cuda")
    counts = torch.ones(1, dtype=torch.int32, device="cuda")
    rejected = torch.zeros_like(counts)
    # Warm CUDA initialization before the deliberately bounded race window.
    with torch.cuda.stream(writer):
        torch.cuda._sleep(1)
    torch.accelerator.synchronize()

    runner = GPUModelRunner.__new__(GPUModelRunner)
    runner.state_slots, runner.main_stream, runner.output_copy_stream = (
        slots,
        main,
        copier,
    )
    runner.is_last_pp_rank = True
    runner.pcp_manager = runner.pp_handler = runner.speculator = None
    runner.aux_output_connector = None
    runner.num_speculative_steps = 0
    runner.check_ep_fault = False
    runner.eplb = SimpleNamespace(step=lambda **kwargs: None)
    runner.kv_connector = SimpleNamespace(post_forward=lambda *a, **kw: None)
    runner.model = SimpleNamespace(compute_logits=lambda _: None)
    runner.req_states = SimpleNamespace(
        all_token_ids=SimpleNamespace(gpu=None),
        num_computed_tokens=SimpleNamespace(gpu=None),
        prompt_len=SimpleNamespace(np=None),
    )
    runner.prompt_logprobs_worker = SimpleNamespace(
        compute_prompt_logprobs=lambda *a: {},
        compute_prompt_token_id_logprobs=lambda *a: {},
    )
    monkeypatch.setattr(
        runner,
        "sample",
        lambda *a: (
            SimpleNamespace(
                sampled_token_ids=sampled,
                logprobs_tensors=None,
                num_nans=None,
                sampling_mask_tensors=None,
            ),
            counts,
            rejected,
        ),
    )

    def delayed_postprocess(*args):
        with torch.cuda.stream(writer):
            torch.cuda._sleep(2_000_000_000)
            resident.fill_(17)

    monkeypatch.setattr(runner, "postprocess_sampled", delayed_postprocess)
    runner.execute_model_state = ExecuteModelState(
        input_batch=SimpleNamespace(
            req_ids=[request.request_id], idx_mapping=None, query_start_loc=None
        ),
        attn_metadata=None,
        slot_mappings_by_layer=None,
        hidden_states=torch.empty(1, 1, device="cuda"),
        aux_hidden_states=None,
        dp_sync_state=None,
        finished_req_ids=set(),
        ec_connector_output=None,
        cudagraph_stats=None,
        num_spec_tokens_to_schedule=0,
    )
    pending = runner.sample_tokens(None)
    scheduler.finish_requests(request.request_id, RequestStatus.FINISHED_ABORTED)
    pending.output.copy_event.synchronize()
    assert pending.output.copy_event.query()
    assert not pending.event.query(), "race window elapsed before the copy check"
    assert pending.output.model_runner_output.state_completion is None
    assert manager.allocate("replacement") is None

    entered = Event()

    def retrieve():
        torch.accelerator.set_device_index(main.device)
        entered.set()
        return pending.get_output()

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(retrieve)
        assert entered.wait(timeout=5)
        assert not pending.event.query(), "race window elapsed before retrieval check"
        assert not future.done()
        assert manager.num_free_slots == 0
        with pytest.raises(TimeoutError):
            future.result(timeout=0.05)
        result = future.result(timeout=15)
    assert pending.event.query()
    assert torch.all(resident == 17).item()
    assert result.sampled_token_ids == [[42]]
    assert result.state_completion.handles == step.state_slot_handles
    scheduler.update_from_output(step, result)
    # The compute acknowledgment alone does not cover terminal cleanup.
    assert manager.num_free_slots == 0
    cleanup = scheduler.schedule()
    assert cleanup.state_cleanup_handles[request.request_id]["state"] == handle
    slots.begin_step(cleanup, main)
    slots.activate_requests()
    cleaned = runner._finish_state_step(_state_output(cleanup, acknowledge=False))
    assert manager.allocate("replacement") is None
    scheduler.update_from_output(cleanup, cleaned.get_output())
    replacement = manager.allocate("replacement")
    assert replacement.slot == handle.slot
    assert replacement.generation > handle.generation
