# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import contextlib
from collections.abc import Generator
from dataclasses import dataclass
from functools import partial
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

import vllm.distributed.aux_output_connector.worker as aux_output_worker
import vllm.v1.worker.gpu.model_runner as model_runner_module
from tests.v1.core.utils import new_kv_cache_group
from vllm.config import (
    AttentionConfig,
    CacheConfig,
    KernelConfig,
    LoadConfig,
    LoRAConfig,
    ModelConfig,
    ParallelConfig,
    SchedulerConfig,
    SpeculativeConfig,
    VllmConfig,
)
from vllm.config.compilation import CUDAGraphMode
from vllm.model_executor.warmup.jit_warmup import JitWarmupRegistry
from vllm.platforms import current_platform
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.single_type_kv_cache_manager import (
    CircularBufferManager,
    MambaManager,
)
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    KVCacheConfig,
    MambaSpec,
)
from vllm.v1.worker.gpu.async_utils import async_copy_to_np
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor
from vllm.v1.worker.gpu.input_batch import InputBuffers
from vllm.v1.worker.gpu.mm.encoder_cache import EncoderCache
from vllm.v1.worker.gpu.model_runner import ExecuteModelState, GPUModelRunner
from vllm.v1.worker.gpu.model_states.default import DefaultModelState
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


class _StopAfterDummyBatch(Exception):
    pass


@pytest.mark.parametrize(
    "padded_tokens,max_model_len,max_num_reqs,expected_num_reqs",
    [
        (1024, 1024, 8, 1),
        (1025, 1024, 8, 2),
        (8190, 1536, 32, 6),
        # max_num_batched_tokens > max_num_seqs * max_model_len, which
        # SchedulerConfig only warns about: stay within the request buffers.
        (16384, 2048, 4, 4),
    ],
)
def test_idle_dp_dummy_batch_respects_max_model_len(
    monkeypatch, padded_tokens, max_model_len, max_num_reqs, expected_num_reqs
):
    """An idle DP rank schedules one dummy request, which the DP sync pads to
    the token count its peers agreed on. Kept as one request past
    max_model_len, its attention reads past its block-table row."""
    runner = Mock(
        lora_config=None,
        is_encoder_decoder=False,
        pcp_manager=None,
        max_model_len=max_model_len,
        max_num_reqs=max_num_reqs,
        input_buffers=InputBuffers(
            max_num_reqs=max_num_reqs,
            max_num_tokens=padded_tokens,
            device=torch.device("cpu"),
        ),
    )
    runner.gather_batch_req_state.return_value = (None, 1)
    dummy_batches = []

    def stop(input_batch, *args):
        dummy_batches.append(input_batch)
        raise _StopAfterDummyBatch

    runner.prepare_dummy_attn.side_effect = stop
    # A PIECEWISE descriptor leaves num_reqs to the caller.
    padded_desc = BatchExecutionDescriptor(
        cg_mode=CUDAGraphMode.PIECEWISE, num_tokens=padded_tokens, num_reqs=None
    )
    monkeypatch.setattr(
        model_runner_module,
        "dispatch_cg_and_sync_dp",
        Mock(return_value=(padded_desc, None)),
    )
    # What GPUWorker.execute_dummy_batch schedules: one request, one token.
    scheduler_output = SchedulerOutput.make_empty()
    scheduler_output.num_scheduled_tokens = {"_dummy_req_0": 1}
    scheduler_output.total_num_scheduled_tokens = 1

    with pytest.raises(_StopAfterDummyBatch):
        GPUModelRunner.execute_model(runner, scheduler_output, dummy_run=True)

    (batch,) = dummy_batches
    seq_lens = batch.seq_lens.tolist()
    assert sum(seq_lens) == padded_tokens
    assert max(seq_lens) <= max(max_model_len, -(-padded_tokens // max_num_reqs))
    assert batch.num_reqs == len(seq_lens) == expected_num_reqs


def test_non_last_pp_rank_uses_global_batch_for_sample_feedback(monkeypatch):
    runner = GPUModelRunner.__new__(GPUModelRunner)
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
    runner.model_state = SimpleNamespace(prepare_execution_memory=lambda: None)
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

    raw_layer = CircularBufferManager.layer_config(
        AttentionSpec(block_size=8, num_kv_heads=1, head_size=128, dtype=torch.bfloat16)
    )
    compressed_spec = AttentionSpec(
        block_size=262144,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.bfloat16,
    )
    kv_cache_config = KVCacheConfig(
        num_blocks=1,
        kv_cache_tensors=[],
        kv_cache_groups=[
            new_kv_cache_group(["raw"], raw_layer),
            new_kv_cache_group(["compressed"], compressed_spec),
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
    full_attention_spec = AttentionSpec(
        block_size=attention_block_size,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.bfloat16,
    )
    mamba_spec = MambaSpec(
        shapes=((1,),),
        dtypes=(torch.bfloat16,),
        block_size=mamba_block_size,
    )
    mamba_config = MambaManager.config(
        mamba_cache_mode=mamba_cache_mode,
        num_speculative_blocks=num_speculative_blocks,
    )
    kv_cache_config = KVCacheConfig(
        num_blocks=1,
        kv_cache_tensors=[],
        kv_cache_groups=[
            new_kv_cache_group(["attention"], full_attention_spec),
            new_kv_cache_group(["kda"], mamba_spec, mamba_config),
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

    monkeypatch.setattr(
        "vllm.v1.kv_cache_interface.get_block_table_width", capture_width
    )

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
) -> Generator[None]:
    """No-op stand-in for LoRAModelRunnerMixin.maybe_setup_dummy_loras."""
    yield


def _make_capture_runner(
    monkeypatch: pytest.MonkeyPatch, captured: bool
) -> GPUModelRunner:
    """Minimal V2 runner for capture_model: fakes everything except the
    cudagraph_manager's needs_capture decision."""
    runner = GPUModelRunner.__new__(GPUModelRunner)
    runner.model_state = SimpleNamespace(
        supports_mm_inputs=False,
        capture_inner_cudagraphs=Mock(),
        needs_cudagraph_capture=lambda: False,
    )
    runner.model_config = SimpleNamespace(enforce_eager=False)
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

    def _dummy_run(num_tokens, **kwargs):
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
    @dataclass(slots=True)
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
        draft_enable_eplb=True,
        resolved_draft_num_redundant_experts=None,
        draft_parallel_config=ParallelConfig(),
    )
    spec.apply_draft_overrides = partial(SpeculativeConfig.apply_draft_overrides, spec)
    spec.apply_draft_eplb_overrides = partial(
        SpeculativeConfig.apply_draft_eplb_overrides, spec
    )
    speculator.speculative_config = spec
    get_model = Mock(return_value=object())
    monkeypatch.setattr(ar_module, "get_model", get_model)
    monkeypatch.setattr(ar_module, "set_model_tag", lambda _: contextlib.nullcontext())
    monkeypatch.setattr(ar_module, "get_draft_load_config", lambda _: LoadConfig())

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


class _EmbeddingModel(torch.nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int):
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(vocab_size, hidden_size)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)


@pytest.mark.parametrize("multimodal", [True, False])
def test_dummy_inputs_embeds_holds_random_input_id_embeddings(multimodal):
    """Models fed inputs_embeds must see the randomized dummy ids, not zeros."""
    model_config = ModelConfig(
        model="facebook/opt-125m", dtype="float32", enable_prompt_embeds=True
    )
    vllm_config = VllmConfig(
        model_config=model_config,
        scheduler_config=SchedulerConfig(
            max_num_seqs=4,
            max_num_batched_tokens=16,
            max_model_len=64,
            is_encoder_decoder=False,
        ),
    )
    model = _EmbeddingModel(32, model_config.get_inputs_embeds_size())
    encoder_cache = EncoderCache() if multimodal else None
    state = DefaultModelState(vllm_config, model, encoder_cache, torch.device("cpu"))
    input_ids = torch.tensor([3, 7, 7, 1, 30])

    with torch.inference_mode():
        inputs_embeds = state.dummy_inputs_embeds(6, input_ids)
        expected = model.embed_input_ids(input_ids)

    assert inputs_embeds.shape == (6, model_config.get_inputs_embeds_size())
    torch.testing.assert_close(inputs_embeds[:5], expected)
    torch.testing.assert_close(state.dummy_inputs_embeds(5), expected)


# Persistent-state bindings on the worker. See docs/design/persistent_state.md
# for the design. Where a test stops before `finish_step`, the pool is a CPU
# tensor and the main stream a CPU stream: no writing stream is registered, so
# no device synchronization is involved. The tests that finish a step record
# its completion event on a device.


def _state_bindings():
    from tests.v1.core.utils import create_state_spec, state_layer
    from vllm.v1.kv_cache_interface import KVCacheGroup
    from vllm.v1.worker.gpu.state_slots import StateSlotBindings

    spec = create_state_spec()
    pool = torch.zeros(3 * spec.page_size_bytes, dtype=torch.uint8)
    return StateSlotBindings(
        [KVCacheGroup.from_layers({"state": state_layer(spec)})], {"state": pool}
    )


def _begin_state_step(bindings, step_seq, handles):
    """Begin a scheduled step that binds `handles` ({request_id: handle})."""
    step = SchedulerOutput.make_empty()
    step.state_step_seq = step_seq
    step.state_slot_handles = {rid: {"state": h} for rid, h in handles.items()}
    step.state_cleanup_handles = {}
    bindings.begin_step(step, torch.Stream(device="cpu"))
    bindings.activate_requests()


def _handle(epoch="epoch", slot=1, generation=1):
    from vllm.v1.kv_cache_interface import StateSlotHandle

    return StateSlotHandle(epoch, slot, generation)


def test_state_first_bind_initializes_and_rebind_keeps_the_page():
    """The first bind of a generation writes the descriptors' initial values;
    binding the same handle again leaves what the model wrote."""
    bindings = _state_bindings()
    handle = _handle()
    _begin_state_step(bindings, 1, {"r": handle})
    rows = bindings.bind_rows("state", ("r",), (handle,))
    hidden = rows.component_pools["hidden"]
    assert hidden[1].tolist() == [1.0] * 4
    hidden[1].fill_(5.0)
    _begin_state_step(bindings, 2, {"r": handle})
    assert hidden[1].tolist() == [5.0] * 4
    _begin_state_step(bindings, 3, {"s": _handle(generation=2)})
    assert hidden[1].tolist() == [1.0] * 4


@pytest.mark.parametrize(
    ("first", "second", "match"),
    [
        (_handle(epoch="a"), _handle(epoch="b"), "epoch"),
        (_handle(generation=2), _handle(generation=1), "generation"),
    ],
    ids=["other-epoch", "older-generation"],
)
def test_state_bind_rejects_a_stale_handle(first, second, match):
    """A handle from another engine epoch, or an older generation of a bound
    slot, is rejected."""
    bindings = _state_bindings()
    _begin_state_step(bindings, 1, {"r": first})
    with pytest.raises(ValueError, match=match):
        _begin_state_step(bindings, 2, {"s": second})


def test_state_rows_only_for_the_current_steps_handles():
    """Rows are exposed only for handles bound in the current step: not for a
    request outside it, not under a wrong handle, and not through rows taken
    in an earlier step."""
    bindings = _state_bindings()
    first, second = _handle(slot=1), _handle(slot=2)
    _begin_state_step(bindings, 1, {"a": first, "b": second})
    earlier = bindings.bind_rows("state", ("a",), (first,))
    _begin_state_step(bindings, 2, {"b": second})
    with pytest.raises(ValueError):
        bindings.bind_rows("state", ("a",), (first,))
    with pytest.raises(ValueError):
        bindings.bind_rows("state", ("b",), (_handle(slot=2, generation=7),))
    with pytest.raises(ValueError, match="Stale state step access"):
        earlier.component_pools  # noqa: B018
    assert bindings.bind_rows("state", ("b",), (second,)).slot_indices_cpu == (2,)


# Startup pool access and the pool layout: what a model that captures CUDA
# graphs over its state pools reads before any request exists. The tests open
# the startup-access window as the runner does around `capture_cudagraphs`.


def test_startup_pool_access_is_refused_outside_a_window():
    """`startup_pool_access` raises while no startup-access window is open on
    the bindings: before the first window and after it has closed."""
    bindings = _state_bindings()
    with pytest.raises(ValueError):
        bindings.startup_pool_access("state")
    with bindings._startup_access_window():
        bindings.startup_pool_access("state")
    with pytest.raises(ValueError):
        bindings.startup_pool_access("state")


def test_startup_pool_access_exposes_every_page_of_each_component():
    """Inside a window, the access gives the layer's spec and one tensor per
    descriptor that covers every page, the null page and unbound slots
    included."""
    bindings = _state_bindings()
    with bindings._startup_access_window():
        access = bindings.startup_pool_access("state")
        assert access.spec == bindings.views["state"].spec
        pools = access.component_pools
        assert {name: tuple(pool.shape) for name, pool in pools.items()} == {
            "hidden": (3, 4),
            "count": (3, 2),
        }
        assert pools["hidden"].dtype == torch.float32
        assert pools["count"].dtype == torch.int64


@pytest.mark.parametrize("accessor", ["startup_pool_access", "pool_layout"])
def test_state_pool_accessors_reject_an_unknown_layer(accessor):
    """Both accessors raise for a layer that is not a state layer of the
    bindings, also inside a window."""
    bindings = _state_bindings()
    with bindings._startup_access_window():
        getattr(bindings, accessor)("state")
        with pytest.raises(ValueError):
            getattr(bindings, accessor)("other")


@pytest.mark.parametrize("use", ["spec", "component_pools"])
def test_startup_access_is_stale_once_its_window_has_closed(use):
    """An access object is refused on every use after its window has closed,
    also while a later window, which issues its own access, is open."""
    bindings = _state_bindings()
    with bindings._startup_access_window():
        stale = bindings.startup_pool_access("state")
        getattr(stale, use)
    with pytest.raises(ValueError):
        getattr(stale, use)
    with bindings._startup_access_window():
        getattr(bindings.startup_pool_access("state"), use)
        with pytest.raises(ValueError):
            getattr(stale, use)


def test_startup_access_binds_no_slot():
    """Startup access owns nothing: after a window closes, the first bind of
    every slot is still a first bind, which accepts a first-generation handle
    and writes the descriptors' initial values over the zeroed pool."""
    bindings = _state_bindings()
    with bindings._startup_access_window():
        hidden = bindings.startup_pool_access("state").component_pools["hidden"]
        assert hidden.tolist() == [[0.0] * 4] * 3
    first, second = _handle(slot=1), _handle(slot=2)
    _begin_state_step(bindings, 1, {"a": first, "b": second})
    rows = bindings.bind_rows("state", ("a", "b"), (first, second))
    assert rows.component_pools["hidden"].tolist() == [[0.0] * 4, [1.0] * 4, [1.0] * 4]


def test_pool_layout_describes_the_tensors_the_bindings_expose():
    """`pool_layout` reports, by descriptor name, the data address, shape,
    strides and dtype of the tensors that `startup_pool_access` and
    `bind_rows` expose for the same bindings."""
    bindings = _state_bindings()
    layout = bindings.pool_layout("state")
    with bindings._startup_access_window():
        startup = dict(bindings.startup_pool_access("state").component_pools)
    handle = _handle()
    _begin_state_step(bindings, 1, {"r": handle})
    rows = bindings.bind_rows("state", ("r",), (handle,)).component_pools
    assert set(layout) == {"hidden", "count"}
    for name, entry in layout.items():
        for tensor in (startup[name], rows[name]):
            assert (entry.data_ptr, entry.shape, entry.stride, entry.dtype) == (
                tensor.data_ptr(),
                tuple(tensor.shape),
                tuple(tensor.stride()),
                tensor.dtype,
            )


def test_pool_layout_is_a_new_snapshot_at_any_time():
    """Each call builds a new mapping with the same values, whether no window
    is open, a window is open, or a step is live."""
    bindings = _state_bindings()
    outside = bindings.pool_layout("state")
    with bindings._startup_access_window():
        inside = bindings.pool_layout("state")
    _begin_state_step(bindings, 1, {"r": _handle()})
    in_step = bindings.pool_layout("state")
    assert dict(outside) == dict(inside) == dict(in_step)
    assert outside is not inside and inside is not in_step


def test_pool_layout_snapshot_is_immutable_and_keeps_nothing_alive():
    """The snapshot is read-only, holds plain values only, and references
    neither the pool nor the bindings: both are freed while it is kept."""
    import dataclasses
    import gc
    import weakref

    from tests.v1.core.utils import create_state_spec, state_layer
    from vllm.v1.kv_cache_interface import KVCacheGroup
    from vllm.v1.worker.gpu.state_slots import StateSlotBindings

    spec = create_state_spec()
    pool = torch.zeros(3 * spec.page_size_bytes, dtype=torch.uint8)
    bindings = StateSlotBindings(
        [KVCacheGroup.from_layers({"state": state_layer(spec)})], {"state": pool}
    )
    layout = bindings.pool_layout("state")
    with pytest.raises(TypeError):
        layout["hidden"] = layout["count"]  # type: ignore[index]
    for entry in layout.values():
        with pytest.raises(dataclasses.FrozenInstanceError):
            entry.data_ptr = 0  # type: ignore[misc]
        assert type(entry.data_ptr) is int
        assert all(type(n) is int for n in (*entry.shape, *entry.stride))
        assert isinstance(entry.dtype, torch.dtype)
    alive = [weakref.ref(pool), weakref.ref(bindings)]
    del pool, bindings
    gc.collect()
    assert [ref() for ref in alive] == [None, None]
    assert set(layout) == {"hidden", "count"}


# The model-state hooks every `ModelState` has. A state that defines none of
# them captures nothing and needs no call.


@pytest.mark.parametrize(
    ("hook", "kwargs", "expected"),
    [
        ("needs_cudagraph_capture", {}, False),
        ("prepare_execution_memory", {}, None),
        (
            "capture_cudagraphs",
            {"graph_pool": None, "stream": None, "profile_only": True},
            None,
        ),
        (
            "capture_cudagraphs",
            {"graph_pool": None, "stream": None, "profile_only": False},
            None,
        ),
        ("clear_cudagraphs", {}, None),
        ("post_warmup", {}, None),
        ("shutdown", {}, None),
    ],
)
def test_model_state_hook_defaults(hook, kwargs, expected):
    """On a model state that defines none of them, `needs_cudagraph_capture`
    returns `False` and the other hooks return `None` and leave the state as
    it was."""
    model_config = ModelConfig(model="facebook/opt-125m", dtype="float32")
    vllm_config = VllmConfig(
        model_config=model_config,
        scheduler_config=SchedulerConfig(
            max_num_seqs=4,
            max_num_batched_tokens=16,
            max_model_len=64,
            is_encoder_decoder=False,
        ),
    )
    model = _EmbeddingModel(32, model_config.get_inputs_embeds_size())
    state = DefaultModelState(vllm_config, model, None, torch.device("cpu"))
    before = dict(vars(state))

    assert getattr(state, hook)(**kwargs) is expected
    after = vars(state)
    assert after.keys() == before.keys()
    assert all(after[name] is value for name, value in before.items())


_NEEDS_CUDA = pytest.mark.skipif(
    not current_platform.is_cuda(), reason="records a completion event on a device"
)


def _device_state_step(handle):
    """Bindings over a device pool, with one live step that binds `handle`
    to request "r" on the current stream."""
    from tests.v1.core.utils import create_state_spec, state_layer
    from vllm.v1.kv_cache_interface import KVCacheGroup
    from vllm.v1.worker.gpu.state_slots import StateSlotBindings

    spec = create_state_spec()
    pool = torch.zeros(3 * spec.page_size_bytes, dtype=torch.uint8, device="cuda")
    bindings = StateSlotBindings(
        [KVCacheGroup.from_layers({"state": state_layer(spec)})], {"state": pool}
    )
    step = SchedulerOutput.make_empty()
    step.state_step_seq = 1
    step.state_slot_handles = {"r": {"state": handle}}
    step.state_cleanup_handles = {}
    return bindings, step


@_NEEDS_CUDA
def test_state_model_failure_reaches_the_steps_completion():
    """A failure the model reports during a live step is carried by that
    step's completion; outside a live step (before it, during its cleanup
    phase or after it) or for a request outside it, the report is refused."""
    from vllm.v1.outputs import ModelRunnerOutput

    handle = _handle()
    bindings, step = _device_state_step(handle)
    stream = torch.cuda.current_stream()
    with pytest.raises(ValueError, match="No live state step"):
        bindings.fail_requests({"r"})
    bindings.begin_step(step, stream)
    with pytest.raises(ValueError, match="No live state step"):
        bindings.fail_requests({"r"})
    bindings.activate_requests()
    with pytest.raises(ValueError, match="outside the active state step"):
        bindings.fail_requests({"other"})
    bindings.fail_requests({"r"})
    completion = bindings.finish_step(ModelRunnerOutput(req_ids=["r"]), stream)
    output = completion.get_output()
    assert output.state_completion is not None
    assert output.state_completion.step_seq == 1
    assert output.state_completion.handles == {"r": {"state": handle}}
    assert output.state_completion.failed_req_ids == {"r"}
    with pytest.raises(ValueError, match="No live state step"):
        bindings.fail_requests({"r"})


@_NEEDS_CUDA
@pytest.mark.parametrize(
    "held", [0, 1], ids=["first_stream_held", "second_stream_held"]
)
def test_state_completion_waits_for_every_registered_writing_stream(held):
    """A write pending on either of two registered auxiliary streams holds
    back the step's completion: one stream is held by a host function until
    the test lets it go, so its write is pending when the step is fenced, and
    the completion event is reached only after that write completes. The
    worker joins every registered stream into the main stream before
    recording the event. Only one stream is held at a time, because the driver
    may serialize host functions of different streams.

    Two detectors observe the device. The completion event must not have
    fired while the auxiliary stream is held. And right after `finish_step`,
    the main stream copies the written elements into a witness tensor; stream
    order fixes the values the copy reads when it runs, so releasing the
    auxiliary stream later cannot change them."""
    import ctypes
    import threading
    import time

    from vllm.v1.outputs import ModelRunnerOutput

    timeout = 60

    def done(work: torch.cuda.Stream | torch.cuda.Event, seconds: float) -> bool:
        """Wait up to `seconds` for a stream's work, or an event, to complete."""
        deadline = time.monotonic() + seconds
        while not work.query():
            if time.monotonic() > deadline:
                return False
            time.sleep(0.001)
        return True

    handle = _handle()
    bindings, step = _device_state_step(handle)
    main = torch.cuda.current_stream()
    writers = [torch.cuda.Stream(), torch.cuda.Stream()]
    for writer in writers:
        bindings.register_writing_stream(writer)
    bindings.begin_step(step, main)
    bindings.activate_requests()
    rows = bindings.bind_rows("state", ("r",), (handle,))
    count = rows.component_pools["count"]
    before = count[1].tolist()
    written_value, sentinel = [7, 8], -1
    assert all(b not in written_value + [sentinel] for b in before)
    # The witness starts at a sentinel, so a copy that never ran is visible.
    witness = torch.full_like(count[1], sentinel)

    # A host function holds one auxiliary stream until the test releases it
    # (bounded, so a failure cannot hang the device).
    release = threading.Event()

    def hold(_):
        release.wait(timeout)

    host_fn = ctypes.CFUNCTYPE(None, ctypes.c_void_p)(hold)
    libcuda = ctypes.CDLL("libcuda.so.1")
    written = [torch.cuda.Event(), torch.cuda.Event()]
    idle = False
    try:
        # The main stream does not yet depend on the auxiliary streams, so
        # this cannot wait on the held stream.
        assert done(main, timeout), "the main stream did not start idle"
        for index, writer in enumerate(writers):
            with torch.cuda.stream(writer):
                if index == held:
                    assert (
                        libcuda.cuLaunchHostFunc(
                            ctypes.c_void_p(writer.cuda_stream), host_fn, None
                        )
                        == 0
                    )
                # Each stream writes its own element of the slot's row.
                count[1, index].fill_(written_value[index])
                written[index].record(writer)
        completion = bindings.finish_step(ModelRunnerOutput(req_ids=["r"]), main)
        # Ordered on the main stream after whatever `finish_step` enqueued.
        witness.copy_(count[1])
        assert not written[held].query()
        # The main stream waits behind the held auxiliary stream. This can
        # only pass by waiting the whole second: a main stream that did not
        # wait would finish one copy long before it ends.
        assert not done(main, 1), "the main stream did not wait"
        # The event is recorded behind the joins, so it has not fired.
        assert not completion.event.query(), "completion preceded the write"
        release.set()
        assert done(main, timeout), "the main stream did not drain"
        assert done(completion.event, timeout), "the completion did not fire"
    finally:
        release.set()
        # Bounded: the host function returns within the timeout of its start.
        idle = all([done(writer, timeout) for writer in writers]) and done(
            main, timeout
        )
    assert idle, "the streams did not drain"
    # The witness copy read both elements after their writes, because the
    # main stream waited for both auxiliary streams.
    assert witness.tolist() == written_value
    output = completion.get_output()
    assert written[0].query() and written[1].query()
    assert output.state_completion is not None
    assert output.state_completion.handles == {"r": {"state": handle}}
    assert count[1].tolist() == written_value
