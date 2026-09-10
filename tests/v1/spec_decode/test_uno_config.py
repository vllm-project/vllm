# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import MagicMock

import pytest
import torch

from vllm.config import (
    DeviceConfig,
    LoRAConfig,
    ModelConfig,
    ParallelConfig,
    SchedulerConfig,
    SpeculativeConfig,
    VllmConfig,
    set_current_vllm_config,
)

pytestmark = pytest.mark.skip_global_cleanup


@pytest.fixture
def cpu_vllm_config():
    config = VllmConfig(device_config=DeviceConfig(device="cpu"))
    with set_current_vllm_config(config):
        yield config


@pytest.fixture
def uno_config_factory():
    """Test config validation without fetching a model or loading weights."""
    target = MagicMock(spec=ModelConfig)
    target.runner_type = "generate"
    target.is_diffusion = False
    target.is_multimodal_model = False
    target.is_encoder_decoder = False
    target.is_hybrid = False
    target.is_attention_free = False
    target.get_sliding_window.return_value = None
    target.get_vocab_size.return_value = 128
    parallel = ParallelConfig(distributed_executor_backend="uni")

    def make(**overrides):
        kwargs = dict(
            method="uno",
            uno_lora_path="test-adapter",
            num_speculative_tokens=8,
            target_model_config=target,
            target_parallel_config=parallel,
        )
        kwargs.update(overrides)
        return SpeculativeConfig(**kwargs)

    return make


def test_uno_shares_target_and_retains_dense_draft_probabilities(uno_config_factory):
    config = uno_config_factory()
    assert config.draft_model_config is config.target_model_config
    assert config.draft_parallel_config is config.target_parallel_config
    assert config.use_uno()
    assert config.parallel_drafting
    assert config.enforce_eager
    assert config.draft_sample_method == "probabilistic"
    assert config.rejection_sample_method == "standard"
    assert config.uno_mask_token_id == 128


def test_uno_draft_width_matches_rejection_sampler_limit(uno_config_factory):
    from vllm.v1.sample.rejection_sampler import MAX_SPEC_LEN

    config = uno_config_factory(num_speculative_tokens=MAX_SPEC_LEN)
    assert config.num_speculative_tokens == MAX_SPEC_LEN
    with pytest.raises(ValueError, match=f"num_speculative_tokens <= {MAX_SPEC_LEN}"):
        uno_config_factory(num_speculative_tokens=MAX_SPEC_LEN + 1)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"uno_lora_path": None}, "uno_lora_path"),
        ({"uno_lora_path": " "}, "uno_lora_path"),
        ({"num_speculative_tokens": None}, "num_speculative_tokens"),
        ({"num_speculative_tokens": 0}, "greater than 0"),
        ({"uno_mask_token_id": 1}, "greater than 1"),
        ({"uno_mask_token_id": 129}, "target vocabulary size"),
        ({"model": "other-model"}, "shares the target"),
        ({"target_model_config": None}, "target model"),
        ({"draft_tensor_parallel_size": 2}, "single-GPU"),
        ({"enforce_eager": False}, "eager draft"),
        ({"rejection_sample_method": "block"}, "standard"),
        ({"num_speculative_tokens_per_batch_size": [(1, 8, 4)]}, "fixed"),
    ],
)
def test_uno_rejects_incompatible_options(uno_config_factory, overrides, message):
    with pytest.raises(ValueError, match=message):
        uno_config_factory(**overrides)


@pytest.mark.parametrize(
    ("attribute", "value", "message"),
    [
        ("runner_type", "pooling", "text-only"),
        ("is_diffusion", True, "text-only"),
        ("is_multimodal_model", True, "text-only"),
        ("is_encoder_decoder", True, "text-only"),
        ("is_hybrid", True, "full attention"),
        ("is_attention_free", True, "full attention"),
    ],
)
def test_uno_rejects_unsupported_model_types(
    uno_config_factory, attribute, value, message
):
    target = uno_config_factory().target_model_config
    setattr(target, attribute, value)
    with pytest.raises(ValueError, match=message):
        uno_config_factory(target_model_config=target)


@pytest.mark.parametrize(
    "parallelism",
    [
        "tensor_parallel_size",
        "pipeline_parallel_size",
        "data_parallel_size",
        "prefill_context_parallel_size",
        "decode_context_parallel_size",
    ],
)
def test_uno_rejects_parallel_execution(uno_config_factory, parallelism):
    parallel = uno_config_factory().target_parallel_config
    setattr(parallel, parallelism, 2)
    with pytest.raises(ValueError, match="single-GPU"):
        uno_config_factory(target_parallel_config=parallel)


def test_uno_rejects_sliding_window_attention(uno_config_factory):
    target = uno_config_factory().target_model_config
    target.get_sliding_window.return_value = 64
    with pytest.raises(ValueError, match="full attention"):
        uno_config_factory(target_model_config=target)


@pytest.mark.parametrize("width", [1, 8])
def test_uno_distinguishes_query_budget_from_kv_lookahead(
    cpu_vllm_config, uno_config_factory, width
):
    config = VllmConfig(
        device_config=cpu_vllm_config.device_config,
        speculative_config=uno_config_factory(num_speculative_tokens=width),
        lora_config=LoRAConfig(lora_dtype=torch.bfloat16),
        scheduler_config=SchedulerConfig(
            max_model_len=2048,
            is_encoder_decoder=False,
            async_scheduling=None,
        ),
    )
    # K=1 drafts only the base-weight seed; it still needs one fresh KV slot.
    assert config.speculative_config.max_num_new_slots_for_drafting == width - 1
    assert config.num_lookahead_tokens == width
    assert config.uniform_decode_query_len == width + 1
    assert config.scheduler_config.async_scheduling is False
    assert config.use_v2_model_runner is False


@pytest.mark.parametrize("unsupported", ["async", "v2", "lora", "capacity"])
def test_uno_rejects_unsupported_engine_config(
    cpu_vllm_config, uno_config_factory, monkeypatch, unsupported
):
    scheduler = SchedulerConfig(
        max_model_len=2048,
        is_encoder_decoder=False,
        async_scheduling=None,
    )
    lora = LoRAConfig(lora_dtype=torch.bfloat16)
    message = ""
    if unsupported == "async":
        scheduler.async_scheduling = True
        message = "synchronous scheduling"
    elif unsupported == "v2":
        monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
        message = "Model Runner V1"
    elif unsupported == "lora":
        lora = None
        message = "enable-lora"
    else:
        scheduler.max_num_batched_tokens = scheduler.max_num_seqs
        message = "max_num_batched_tokens"
    with pytest.raises(ValueError, match=message):
        VllmConfig(
            device_config=cpu_vllm_config.device_config,
            speculative_config=uno_config_factory(),
            lora_config=lora,
            scheduler_config=scheduler,
        )
