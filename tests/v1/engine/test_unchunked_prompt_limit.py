# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prompts that cannot be scheduled without chunked prefill are rejected.

The engine core disables chunked prefill after startup validation for models
without KV cache (e.g. a decoder with ``is_causal=False``) and for non-causal
attention layers. A prompt longer than the per-step token budget could then
never be scheduled and used to wait forever, blocking the queue behind it.
"""

import threading
from types import SimpleNamespace

import msgspec
import pytest

from vllm.config.scheduler import SchedulerConfig
from vllm.exceptions import VLLMValidationError
from vllm.v1.engine import EngineCoreReadyResponse
from vllm.v1.engine.core_client import MPClient
from vllm.v1.engine.input_processor import InputProcessor


def _input_processor(
    enable_chunked_prefill: bool,
    max_num_batched_tokens: int = 256,
    max_num_scheduled_tokens: int | None = None,
    num_draft_slots: int = 0,
    is_mm_encoder_only: bool = False,
) -> InputProcessor:
    processor = object.__new__(InputProcessor)
    processor.skip_prompt_length_check = False
    processor.supports_mm_inputs = False
    processor.model_config = SimpleNamespace(max_model_len=2048, runner_type="generate")
    speculative_config = (
        SimpleNamespace(max_num_new_slots_for_drafting=num_draft_slots)
        if num_draft_slots
        else None
    )
    scheduler_config = SchedulerConfig(
        max_model_len=2048,
        is_encoder_decoder=False,
        max_num_batched_tokens=max_num_batched_tokens,
        max_num_scheduled_tokens=max_num_scheduled_tokens,
        max_num_seqs=8,
    )
    # Mirrors the engine core disabling chunked prefill after validation.
    scheduler_config.enable_chunked_prefill = enable_chunked_prefill
    processor.vllm_config = SimpleNamespace(
        scheduler_config=scheduler_config,
        speculative_config=speculative_config,
        is_mm_encoder_only=is_mm_encoder_only,
    )
    return processor


@pytest.mark.cpu_test
def test_max_unchunked_prompt_len():
    config = SchedulerConfig(
        max_model_len=2048, is_encoder_decoder=False, max_num_batched_tokens=256
    )
    assert config.max_unchunked_prompt_len() == 256
    assert config.max_unchunked_prompt_len(num_draft_slots=6) == 250
    config.max_num_scheduled_tokens = 200
    assert config.max_unchunked_prompt_len(num_draft_slots=6) == 200


@pytest.mark.cpu_test
def test_rejects_prompt_over_budget_without_chunked_prefill():
    processor = _input_processor(enable_chunked_prefill=False)
    processor._validate_prompt_len(256, "decoder")
    with pytest.raises(VLLMValidationError, match="can schedule in one step"):
        processor._validate_prompt_len(257, "decoder")
    # The max_model_len check still takes precedence.
    with pytest.raises(VLLMValidationError, match="maximum model length"):
        processor._validate_prompt_len(4096, "decoder")


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    ("kwargs", "limit"),
    [
        ({"max_num_scheduled_tokens": 200}, 200),
        ({"num_draft_slots": 4}, 252),
    ],
)
def test_unchunked_limit_matches_scheduler_budget(kwargs, limit):
    processor = _input_processor(enable_chunked_prefill=False, **kwargs)
    processor._validate_prompt_len(limit, "decoder")
    with pytest.raises(VLLMValidationError):
        processor._validate_prompt_len(limit + 1, "decoder")


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    "kwargs",
    [
        {"enable_chunked_prefill": True},
        {"enable_chunked_prefill": False, "is_mm_encoder_only": True},
    ],
)
def test_long_prompt_allowed_when_it_can_be_scheduled(kwargs):
    processor = _input_processor(**kwargs)
    processor._validate_prompt_len(1024, "decoder")


@pytest.mark.cpu_test
def test_ready_response_syncs_disabled_chunked_prefill():
    client = object.__new__(MPClient)
    client._effective_attention_block_sizes = set()
    client.stats_update_address = None
    client.vllm_config = SimpleNamespace(
        cache_config=SimpleNamespace(block_size=16, num_gpu_blocks=0),
        model_config=SimpleNamespace(max_model_len=2048),
        scheduler_config=SimpleNamespace(enable_chunked_prefill=True),
    )

    def ready(enable_chunked_prefill: bool) -> bytes:
        return msgspec.msgpack.encode(
            EngineCoreReadyResponse(
                max_model_len=2048,
                num_gpu_blocks=0,
                block_size=16,
                dp_stats_address=None,
                dtype="bfloat16",
                vllm_version="test",
                world_size=1,
                data_parallel_size=1,
                tensor_parallel_size=1,
                pipeline_parallel_size=1,
                decode_context_parallel_size=1,
                data_parallel_rank=0,
                max_num_seqs=8,
                max_num_batched_tokens=256,
                instance_id="test-instance",
                supports_lora=False,
                max_loras=0,
                enable_chunked_prefill=enable_chunked_prefill,
            )
        )

    client._apply_ready_response(ready(True))
    assert client.vllm_config.scheduler_config.enable_chunked_prefill
    client._apply_ready_response(ready(False))
    assert not client.vllm_config.scheduler_config.enable_chunked_prefill
    # Another data-parallel engine reporting True does not re-enable it.
    client._apply_ready_response(ready(True))
    assert not client.vllm_config.scheduler_config.enable_chunked_prefill


def test_bidirectional_decoder_rejects_unschedulable_prompt():
    """End to end: a decoder with ``is_causal=False`` has no KV cache, so the
    engine disables chunked prefill and prefix caching after startup (default
    settings used to fail with no cacheable KV cache group). A prompt longer
    than the token budget must fail fast instead of hanging the engine."""
    from vllm import LLM, SamplingParams

    llm = LLM(
        model="hmellor/tiny-random-LlamaForCausalLM",
        hf_overrides={"is_causal": False},
        max_model_len=1024,
        max_num_batched_tokens=128,
        max_num_seqs=4,
        enforce_eager=True,
        gpu_memory_utilization=0.2,
    )
    scheduler_config = llm.llm_engine.vllm_config.scheduler_config
    assert not scheduler_config.enable_chunked_prefill
    params = SamplingParams(max_tokens=1, temperature=0.0)

    result: dict = {}

    def generate_long():
        try:
            llm.generate({"prompt_token_ids": [1] * 300}, params, use_tqdm=False)
        except Exception as exc:  # recorded for the assertion below
            result["error"] = exc

    thread = threading.Thread(target=generate_long, daemon=True)
    thread.start()
    thread.join(timeout=120)
    assert not thread.is_alive(), "an unschedulable prompt hung the engine"
    assert isinstance(result.get("error"), VLLMValidationError)
    assert "can schedule in one step" in str(result["error"])

    # The engine stays usable for prompts that fit.
    outputs = llm.generate({"prompt_token_ids": [1] * 100}, params, use_tqdm=False)
    assert len(outputs[0].outputs[0].token_ids) == 1
