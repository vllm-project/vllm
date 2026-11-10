# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import replace

import pytest
import torch
from transformers import NemotronHConfig

import vllm.platforms
from vllm.config import (
    AuxOutputConfig,
    CacheConfig,
    CompilationConfig,
    KernelConfig,
    ModelConfig,
    SchedulerConfig,
    VllmConfig,
)
from vllm.config.elastic import ElasticConfig, SliceConfig, elastic_phase
from vllm.config.speculative import SpeculativeConfig
from vllm.distributed.parallel_state import GroupCoordinator
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.completion.protocol import CompletionRequest
from vllm.exceptions import VLLMValidationError
from vllm.model_executor.models.nemotron_h_elastic import _local_prefix_size
from vllm.platforms.cpu import CpuPlatform
from vllm.sampling_params import SamplingParams, StructuredOutputsParams
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.engine.input_processor import InputProcessor
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
)
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus
from vllm.v1.structured_output import StructuredOutputManager
from vllm.v1.worker.gpu import cudagraph_utils

pytestmark = pytest.mark.cpu_test


@pytest.fixture
def elastic_config(tmp_path, monkeypatch):
    hf = NemotronHConfig(
        architectures=["NemotronHForCausalLM"],
        hidden_size=128,
        moe_intermediate_size=128,
        vocab_size=256,
        num_hidden_layers=2,
        hybrid_override_pattern="M*",
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=64,
        elastic_use_lazy_slicing=True,
        elastic_thinking={
            "elastic_hidden_size": 64,
            "elastic_moe_intermediate_size": 64,
        },
        elastic_think_end_token_id=13,
    )
    hf.save_pretrained(tmp_path)
    return VllmConfig(
        model_config=ModelConfig(
            model=str(tmp_path),
            dtype="bfloat16",
            skip_tokenizer_init=True,
            max_model_len=256,
            enforce_eager=True,
        ),
        scheduler_config=SchedulerConfig(
            max_num_seqs=2,
            max_num_batched_tokens=256,
            max_model_len=256,
            async_scheduling=False,
            is_encoder_decoder=False,
        ),
        cache_config=CacheConfig(block_size=16, enable_prefix_caching=False),
        kernel_config=KernelConfig(moe_backend="triton"),
        compilation_config=CompilationConfig(mode=0, cudagraph_mode="NONE"),
    )


def request(name, phase):
    return Request(
        name,
        [13, 1, 2],
        SamplingParams(
            max_tokens=8, extra_args={"elastic_phase": phase} if phase else None
        ),
        pooling_params=None,
    )


def make_scheduler(config, num_blocks=128, log_stats=False):
    config.cache_config.num_gpu_blocks = num_blocks
    spec = FullAttentionSpec(
        block_size=16, num_kv_heads=1, head_size=64, dtype=torch.bfloat16
    )
    return Scheduler(
        config,
        KVCacheConfig(num_blocks, [], [KVCacheGroupSpec(["layer"], spec)]),
        StructuredOutputManager(config),
        block_size=16,
        log_stats=log_stats,
    )


@pytest.mark.parametrize("hidden,intermediate", [(64, 64), (128, 64)])
def test_profiles_read_checkpoint_dimensions(elastic_config, hidden, intermediate):
    hf = elastic_config.model_config.hf_config
    hf.elastic_thinking = {
        "elastic_hidden_size": hidden,
        "elastic_moe_intermediate_size": intermediate,
    }
    config = ElasticConfig.from_vllm_config(elastic_config)
    assert config is not None
    assert config.profiles == {
        "full": SliceConfig(hf.hidden_size, hf.moe_intermediate_size),
        "sliced": SliceConfig(hidden, intermediate),
    }


def test_hf_overrides_enable_runtime_slicing(elastic_config):
    """A public checkpoint can opt in without modifying its stored config."""
    path = elastic_config.model_config.model
    hf = NemotronHConfig.from_pretrained(path)
    for field in (
        "elastic_use_lazy_slicing",
        "elastic_thinking",
        "elastic_think_end_token_id",
    ):
        delattr(hf, field)
    hf.save_pretrained(path)
    elastic_config.model_config = ModelConfig(
        model=path,
        dtype="bfloat16",
        skip_tokenizer_init=True,
        max_model_len=256,
        enforce_eager=True,
        hf_overrides={
            "elastic_use_lazy_slicing": True,
            "elastic_thinking": {
                "elastic_hidden_size": 64,
                "elastic_moe_intermediate_size": 64,
            },
            "elastic_think_end_token_id": 13,
        },
    )
    config = ElasticConfig.from_vllm_config(elastic_config)
    assert config is not None
    assert config.profiles["sliced"] == SliceConfig(64, 64)
    assert config.think_end_token_id == 13
    assert not hasattr(NemotronHConfig.from_pretrained(path), "elastic_thinking")


def test_decode_graph_plan_keeps_both_widths(elastic_config, tmp_path, monkeypatch):
    """Uniform decode tables must not let one width shadow the other."""
    platform = CpuPlatform()
    monkeypatch.setattr(vllm.platforms, "current_platform", platform)
    monkeypatch.setattr(cudagraph_utils, "current_platform", platform)
    # Only plan the graphs; no GPU pool or CUDA capture is needed.
    monkeypatch.setattr(platform, "get_global_graph_pool", lambda: None)
    torch.distributed.init_process_group(
        "gloo",
        init_method=f"file://{tmp_path / 'gloo'}",
        rank=0,
        world_size=1,
    )
    group = GroupCoordinator([[0]], 0, "gloo", use_device_communicator=False)
    try:
        monkeypatch.setattr(cudagraph_utils, "get_pp_group", lambda: group)
        elastic_config.compilation_config = CompilationConfig(
            mode=0,
            cudagraph_mode="FULL_DECODE_ONLY",
            decode_cudagraph_capture_request_sizes=[1, 2],
            prefill_cudagraph_capture_token_sizes=[],
            max_cudagraph_capture_size=2,
        )
        manager = cudagraph_utils.CudaGraphManager(
            elastic_config,
            torch.device("cpu"),
            elastic_config.compilation_config.cudagraph_mode,
            decode_query_lens={1: 2},
        )
        assert manager.needs_capture()
        for phase in ("full", "sliced"):
            table = manager._decode_descs[(1, 0, 1, phase)]
            assert [(d.num_reqs, d.elastic_phase) for d in table[1:]] == [
                (1, phase),
                (2, phase),
            ]
    finally:
        group.destroy()
        torch.distributed.destroy_process_group()


@pytest.mark.parametrize("checkpoint_default", [None, "sliced", "auto"])
def test_request_selection_defaults_to_full(elastic_config, checkpoint_default):
    if checkpoint_default is not None:
        elastic_config.model_config.hf_config.elastic_default_slice = checkpoint_default
    processor = InputProcessor(elastic_config)
    config = ElasticConfig.from_vllm_config(elastic_config)
    assert config is not None
    for selected in (None, "full", "sliced", "auto"):
        api_request = CompletionRequest(
            model="elastic", prompt="test", elastic_phase=selected
        )
        params = api_request.to_sampling_params(8, {})
        incoming = processor.process_inputs(
            "default",
            {"type": "token", "prompt_token_ids": [1, 2]},
            params,
            ("generate",),
        )
        actual = config.request_phase(Request.from_engine_core_request(incoming, None))
        expected = "full" if selected is None else selected
        assert actual == ("sliced" if expected == "auto" else expected)


def step(scheduler, tokens):
    scheduled = scheduler.schedule()
    ids = list(scheduled.num_scheduled_tokens)
    scheduler.update_from_output(
        scheduled,
        ModelRunnerOutput(
            req_ids=ids,
            sampled_token_ids=[tokens for _ in ids],
        ),
    )
    return scheduled


@pytest.mark.parametrize(
    "unsupported", ["async", "speculative", "pcp", "dcp", "pp", "dp", "ep"]
)
def test_unsupported_elastic_settings_are_rejected(elastic_config, unsupported):
    if unsupported == "async":
        elastic_config.scheduler_config.async_scheduling = True
    elif unsupported == "speculative":
        elastic_config.speculative_config = SpeculativeConfig(
            model="ngram", num_speculative_tokens=1
        )
    elif unsupported == "pcp":
        elastic_config.parallel_config.prefill_context_parallel_size = 2
    elif unsupported == "dcp":
        elastic_config.parallel_config.decode_context_parallel_size = 2
    elif unsupported == "pp":
        elastic_config.parallel_config.pipeline_parallel_size = 2
    elif unsupported == "dp":
        elastic_config.parallel_config.data_parallel_size = 2
    else:
        elastic_config.parallel_config.enable_expert_parallel = True
    with pytest.raises(ValueError, match="[Ee]lastic"):
        ElasticConfig.from_vllm_config(elastic_config)


@pytest.mark.parametrize("return_routed_experts", [False, True])
def test_elastic_scheduler_rejects_routed_expert_outputs(
    elastic_config, return_routed_experts
):
    """Reject auxiliary capture before sliced MoE can return stale expert IDs."""
    config = replace(
        elastic_config,
        cache_config=CacheConfig(block_size=16, enable_prefix_caching=True),
        aux_output_config=AuxOutputConfig(
            enable_return_routed_experts=return_routed_experts
        ),
    )
    if return_routed_experts:
        with pytest.raises(ValueError, match="Elastic.*auxiliary outputs"):
            make_scheduler(config)
    else:
        scheduler = make_scheduler(config)
        assert scheduler.elastic is not None
        assert scheduler.elastic.profiles == {
            "full": SliceConfig(128, 128),
            "sliced": SliceConfig(64, 64),
        }


@pytest.mark.parametrize("tp_size", [1, 2, 4])
def test_tensor_parallelism_keeps_global_profile_dimensions(elastic_config, tp_size):
    elastic_config.parallel_config.tensor_parallel_size = tp_size
    elastic_config.model_config.hf_config.use_bias = True
    elastic_config.model_config.hf_config.n_groups = tp_size
    config = ElasticConfig.from_vllm_config(elastic_config)
    assert config is not None
    assert config.profiles == {
        "full": SliceConfig(128, 128),
        "sliced": SliceConfig(64, 64),
    }


def test_tensor_parallelism_rejects_indivisible_full_expert_width(elastic_config):
    elastic_config.parallel_config.tensor_parallel_size = 3
    with pytest.raises(ValueError, match="full MoE intermediate size.*divisible"):
        ElasticConfig.from_vllm_config(elastic_config)


@pytest.mark.parametrize("tp_size,groups", [(2, 1), (2, 3), (4, 1)])
def test_tensor_parallelism_rejects_biased_replicated_mamba_groups(
    elastic_config, tp_size, groups
):
    elastic_config.parallel_config.tensor_parallel_size = tp_size
    hf = elastic_config.model_config.hf_config
    hf.use_bias = True
    hf.n_groups = groups
    with pytest.raises(ValueError, match="biased Mamba.*groups.*tensor parallel"):
        ElasticConfig.from_vllm_config(elastic_config)


@pytest.mark.parametrize("tp_size", [1, 2, 4])
@pytest.mark.parametrize(
    "full_width,sliced_width", [(256, 64), (256, 192), (256, 256), (1856, 960)]
)
def test_rank_shards_reconstruct_global_expert_prefix(
    tp_size, full_width, sliced_width
):
    """Partial and empty shards must preserve the same checkpoint neurons."""
    generator = torch.Generator().manual_seed(0)
    x = torch.randint(-2, 3, (3, 64), generator=generator).float()
    up = torch.randint(-2, 3, (full_width, 128), generator=generator).float()
    down = torch.randint(-2, 3, (128, full_width), generator=generator).float()
    reference = (x @ up[:sliced_width, :64].T).relu() @ down[:64, :sliced_width].T
    partition_size = full_width // tp_size
    output = torch.zeros_like(reference)
    for rank in range(tp_size):
        start = rank * partition_size
        local_up = up[start : start + partition_size]
        local_down = down[:, start : start + partition_size]
        active = _local_prefix_size(sliced_width, partition_size, rank)
        output += (x @ local_up[:active, :64].T).relu() @ local_down[:64, :active].T
    torch.testing.assert_close(output, reference, rtol=0, atol=0)


def test_auto_switch_uses_generated_tokens_only(elastic_config):
    scheduler = make_scheduler(elastic_config)
    r = request("auto", "auto")
    scheduler.add_request(r)
    assert step(scheduler, [4]).elastic_phase == "sliced"
    assert step(scheduler, [13]).elastic_phase == "sliced"
    assert step(scheduler, [13]).elastic_phase == "full"
    assert r.elastic_transition_position == r.num_prompt_tokens + 1
    assert step(scheduler, [4]).elastic_phase == "full"
    assert scheduler.elastic.request_phase(request("fixed", "sliced")) == "sliced"


def test_auto_without_generated_marker_stays_sliced(elastic_config):
    scheduler = make_scheduler(elastic_config)
    r = request("auto", "auto")
    scheduler.add_request(r)
    for token in (4, 5, 6):
        assert step(scheduler, [token]).elastic_phase == "sliced"
    assert r.elastic_transition_position is None


@pytest.mark.parametrize("phase", [None, "full", "sliced"])
def test_fixed_phase_does_not_switch_on_generated_marker(elastic_config, phase):
    scheduler = make_scheduler(elastic_config)
    scheduler.add_request(request("fixed", phase))
    assert step(scheduler, [13]).elastic_phase == (phase or "full")
    assert step(scheduler, [4]).elastic_phase == (phase or "full")


@pytest.mark.parametrize("sliced_count", [1, 2])
def test_interleaved_cohorts_and_full_capacity(elastic_config, sliced_count):
    scheduler = make_scheduler(elastic_config)
    requests = [("sliced-1", "sliced"), ("full", "full")]
    if sliced_count == 2:
        requests.append(("sliced-2", "sliced"))
    for name, phase in requests:
        scheduler.add_request(request(name, phase))
    first = scheduler.schedule()
    assert first.elastic_phase == "sliced"
    assert set(first.num_scheduled_tokens) == {
        f"sliced-{i + 1}" for i in range(sliced_count)
    }
    ids = list(first.num_scheduled_tokens)
    scheduler.update_from_output(
        first,
        ModelRunnerOutput(
            req_ids=ids,
            sampled_token_ids=[[4]] * len(ids),
        ),
    )
    # The waiting profile cannot run while both request slots are occupied.
    second = scheduler.schedule()
    assert second.elastic_phase == ("sliced" if sliced_count == 2 else "full")
    assert set(second.num_scheduled_tokens) == (
        set(ids) if sliced_count == 2 else {"full"}
    )


@pytest.mark.parametrize("running_phase", ["full", "sliced"])
def test_kv_starved_width_does_not_stall_running_cohort(elastic_config, running_phase):
    """Keep decoding when the other width cannot acquire its first KV block."""
    scheduler = make_scheduler(elastic_config, num_blocks=2)
    running = request("running", running_phase)
    waiting_phase = "sliced" if running_phase == "full" else "full"
    waiting = request("waiting", waiting_phase)
    scheduler.add_request(running)
    step(scheduler, [4])
    scheduler.add_request(waiting)
    for _ in range(3):
        previous_step = scheduler.current_step
        output = step(scheduler, [4])
        assert scheduler.current_step == previous_step + 1
        assert output.elastic_phase == running_phase
        assert output.num_scheduled_tokens == {"running": 1}
        assert waiting.status == RequestStatus.WAITING
    scheduler.finish_requests([running.request_id], RequestStatus.FINISHED_ABORTED)
    output = step(scheduler, [4])
    assert output.elastic_phase == waiting_phase
    assert output.num_scheduled_tokens == {"waiting": waiting.num_prompt_tokens}


def test_width_delayed_request_is_counted_as_waiting(elastic_config):
    scheduler = make_scheduler(elastic_config, log_stats=True)
    scheduler.add_request(request("full", "full"))
    delayed = request("sliced", "sliced")
    scheduler.add_request(delayed)
    assert step(scheduler, [4]).elastic_phase == "full"
    stats = scheduler.make_stats()
    assert stats.num_waiting_reqs == 1
    assert stats.num_skipped_waiting_reqs == 0
    assert delayed not in scheduler.deferred_waiting
    scheduler.finish_requests([delayed.request_id], RequestStatus.FINISHED_ABORTED)
    stats = scheduler.make_stats()
    assert stats.num_waiting_reqs == stats.num_skipped_waiting_reqs == 0
    assert step(scheduler, [4]).num_scheduled_tokens == {"full": 1}


def test_other_width_with_pending_grammar_is_still_skipped(elastic_config):
    scheduler = make_scheduler(elastic_config, log_stats=True)
    scheduler.add_request(request("full", "full"))
    blocked = Request(
        "grammar",
        [1, 2],
        SamplingParams(
            max_tokens=1,
            extra_args={"elastic_phase": "sliced"},
            structured_outputs=StructuredOutputsParams(regex="[0-9]+"),
        ),
        pooling_params=None,
    )
    scheduler.add_request(blocked)
    assert step(scheduler, [4]).elastic_phase == "full"
    stats = scheduler.make_stats()
    assert stats.num_waiting_reqs == 0
    assert stats.num_skipped_waiting_reqs == 1
    assert blocked in scheduler.deferred_waiting
    scheduler.finish_requests([blocked.request_id], RequestStatus.FINISHED_ABORTED)
    assert not scheduler.deferred_waiting


@pytest.mark.parametrize(
    "request_cls,fields",
    [
        (CompletionRequest, {"prompt": "hello"}),
        (ChatCompletionRequest, {"messages": [{"role": "user", "content": "hello"}]}),
    ],
)
@pytest.mark.parametrize("phase", ["full", "sliced", "auto"])
def test_api_phase_overrides_xargs_without_mutating_request(request_cls, fields, phase):
    xargs = {"other": 1, "elastic_phase": "full" if phase == "sliced" else "sliced"}
    r = request_cls(model="elastic", elastic_phase=phase, vllm_xargs=xargs, **fields)
    params = r.to_sampling_params(10, {})
    assert params.extra_args == {"other": 1, "elastic_phase": phase}
    assert r.vllm_xargs == xargs


@pytest.mark.parametrize("phase", ["base", "thinking", "big", "small", "invalid"])
def test_invalid_phase_is_rejected_by_api_and_sampling(phase):
    with pytest.raises(ValueError, match="elastic_phase"):
        CompletionRequest(model="elastic", prompt="hello", elastic_phase=phase)
    with pytest.raises(ValueError, match="elastic_phase"):
        ChatCompletionRequest(
            model="elastic",
            messages=[{"role": "user", "content": "hello"}],
            elastic_phase=phase,
        )
    with pytest.raises(ValueError, match="elastic_phase"):
        elastic_phase({"elastic_phase": phase})


@pytest.mark.parametrize(
    "phase,enabled",
    [("bad", True), ("full", False), ("sliced", False), ("auto", False)],
)
def test_invalid_phase_is_a_client_error(elastic_config, phase, enabled):
    elastic_config.model_config.hf_config.elastic_use_lazy_slicing = enabled
    processor = InputProcessor(elastic_config)
    with pytest.raises(VLLMValidationError, match="elastic_phase"):
        processor.process_inputs(
            "invalid",
            {"type": "token", "prompt_token_ids": [1, 2]},
            SamplingParams(max_tokens=1, extra_args={"elastic_phase": phase}),
            ("generate",),
        )


@pytest.mark.parametrize("prompt_len", [16, 17])
def test_preemption_recomputes_with_current_slice(elastic_config, prompt_len):
    """Once auto switches, even a full cache rebuild must use the full model."""
    blocks_per_prompt = (prompt_len + 15) // 16
    num_blocks = 2 * blocks_per_prompt + 1
    scheduler = make_scheduler(elastic_config, num_blocks)

    older = Request(
        "older",
        [1] * (blocks_per_prompt * 16),
        SamplingParams(max_tokens=8, extra_args={"elastic_phase": "full"}),
        None,
    )
    auto = Request(
        "auto",
        [1] * prompt_len,
        SamplingParams(max_tokens=8, extra_args={"elastic_phase": "auto"}),
        None,
    )
    scheduler.add_request(older)
    step(scheduler, [4])
    scheduler.add_request(auto)
    assert step(scheduler, [13]).elastic_phase == "sliced"
    step(scheduler, [4])
    assert auto.status == RequestStatus.PREEMPTED
    scheduler.finish_requests([older.request_id], RequestStatus.FINISHED_ABORTED)
    resumed = step(scheduler, [4])
    assert resumed.elastic_phase == "full"
    assert resumed.num_scheduled_tokens == {"auto": prompt_len + 1}
    assert list(auto.output_token_ids) == [13, 4]


@pytest.mark.parametrize(
    "request_cls,fields",
    [
        (CompletionRequest, {"prompt": "hello"}),
        (ChatCompletionRequest, {"messages": [{"role": "user", "content": "hello"}]}),
    ],
)
def test_beam_search_rejects_explicit_elastic_phase(request_cls, fields):
    request = request_cls(
        model="elastic", use_beam_search=True, elastic_phase="full", **fields
    )
    with pytest.raises(ValueError, match="[Bb]eam search.*Elastic"):
        request.to_beam_search_params(8, {})


def test_resumable_input_is_rejected_for_elastic(elastic_config):
    with pytest.raises(VLLMValidationError, match="resumable input"):
        InputProcessor(elastic_config).process_inputs(
            "session",
            {"type": "token", "prompt_token_ids": [1, 2]},
            SamplingParams(max_tokens=1),
            ("generate",),
            resumable=True,
        )


@pytest.mark.parametrize("prompt_len,evict", [(16, False), (17, False), (16, True)])
def test_resumed_cache_hit_keeps_current_slice(elastic_config, prompt_len, evict):
    elastic_config.cache_config.enable_prefix_caching = True
    scheduler = make_scheduler(elastic_config)
    init_none_hash(sha256)
    auto = Request(
        "auto",
        [1] * prompt_len,
        SamplingParams(max_tokens=8, extra_args={"elastic_phase": "auto"}),
        None,
        block_hasher=get_request_block_hasher(16, sha256),
    )
    scheduler.add_request(auto)
    step(scheduler, [13])
    scheduler.running.remove(auto)
    scheduler._preempt_request(auto, auto.arrival_time)
    if evict:
        assert scheduler.reset_prefix_cache()
    scheduled = scheduler.schedule()
    assert scheduled.elastic_phase == "full"
    cached_tokens = 0 if evict else 16
    assert scheduled.num_scheduled_tokens == {"auto": prompt_len + 1 - cached_tokens}
