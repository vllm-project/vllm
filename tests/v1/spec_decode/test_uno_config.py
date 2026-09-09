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
)
from vllm.exceptions import VLLMValidationError
from vllm.v1.engine.input_processor import InputProcessor

pytestmark = pytest.mark.skip_global_cleanup


@pytest.fixture
def uno_config_factory():
    """Build Uno configs without fetching a model or loading weights."""
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


def test_uno_shares_target_and_uses_probabilistic_draft_rows(uno_config_factory):
    config = uno_config_factory()

    assert config.model == "uno"
    assert config.draft_model_config is config.target_model_config
    assert config.draft_parallel_config is config.target_parallel_config
    assert config.use_uno()
    assert config.parallel_drafting
    assert config.draft_sample_method == "probabilistic"
    assert config.rejection_sample_method == "standard"
    assert config.uno_mask_token_id == 128
    assert config.uno_noise_seed == 0


def test_uno_forces_probabilistic_draft_sampling(uno_config_factory):
    config = uno_config_factory(draft_sample_method="greedy")

    assert config.draft_sample_method == "probabilistic"


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
        ({"rejection_sample_method": "synthetic"}, "standard"),
        ({"num_speculative_tokens_per_batch_size": [(1, 8, 4)]}, "fixed"),
        ({"use_heterogeneous_vocab": True}, "target vocabulary"),
        ({"use_local_argmax_reduction": True}, "local argmax"),
        ({"enable_adaptive_verification": True}, "Adaptive verification"),
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


def _make_vllm_uno_config(uno_config_factory, **scheduler_overrides):
    scheduler_kwargs = dict(
        max_model_len=2048,
        max_num_seqs=4,
        max_num_batched_tokens=256,
        is_encoder_decoder=False,
        async_scheduling=None,
    )
    scheduler_kwargs.update(scheduler_overrides)
    return VllmConfig(
        device_config=DeviceConfig(device="cpu"),
        speculative_config=uno_config_factory(),
        lora_config=LoRAConfig(max_loras=2, lora_dtype=torch.bfloat16),
        scheduler_config=SchedulerConfig(**scheduler_kwargs),
    )


def test_uno_uses_native_async_v2_and_reserves_k_lookahead(
    uno_config_factory, monkeypatch
):
    monkeypatch.setattr("vllm.config.vllm.HAS_TRITON", True)
    monkeypatch.setattr("vllm.config.vllm.envs.VLLM_USE_V2_MODEL_RUNNER", True)
    # This is a CUDA-only contract.  Keep the config test runnable in the CPU
    # Docker suite without importing or initializing a CUDA device; CPU's
    # platform hook intentionally disables async scheduling.
    from vllm.platforms import current_platform

    monkeypatch.setattr(
        current_platform, "apply_config_platform_defaults", lambda _config: None
    )
    monkeypatch.setattr(
        current_platform, "check_and_update_config", lambda _config: None
    )
    config = _make_vllm_uno_config(uno_config_factory)

    assert config.scheduler_config.async_scheduling is True
    assert config.use_v2_model_runner
    assert config.speculative_config.max_num_new_slots_for_drafting == 7
    assert config.num_lookahead_tokens == 8
    assert config.uniform_decode_query_len == 9


def test_uno_rejects_platform_async_override(uno_config_factory, monkeypatch):
    monkeypatch.setattr("vllm.config.vllm.HAS_TRITON", True)
    monkeypatch.setattr("vllm.config.vllm.envs.VLLM_USE_V2_MODEL_RUNNER", True)
    from vllm.platforms import current_platform

    monkeypatch.setattr(
        current_platform,
        "check_and_update_config",
        lambda config: setattr(config.scheduler_config, "async_scheduling", False),
    )
    with pytest.raises(ValueError, match="asynchronous scheduling"):
        _make_vllm_uno_config(uno_config_factory)


@pytest.mark.parametrize(
    ("scheduler_overrides", "message"),
    [
        ({"async_scheduling": False}, "asynchronous scheduling"),
        ({"max_num_batched_tokens": 31}, "max_num_batched_tokens"),
    ],
)
def test_uno_rejects_unsupported_v2_scheduler_options(
    uno_config_factory, monkeypatch, scheduler_overrides, message
):
    monkeypatch.setattr("vllm.config.vllm.HAS_TRITON", True)
    monkeypatch.setattr("vllm.config.vllm.envs.VLLM_USE_V2_MODEL_RUNNER", True)
    with pytest.raises(ValueError, match=message):
        _make_vllm_uno_config(uno_config_factory, **scheduler_overrides)


def test_uno_rejects_forced_v1_runner(uno_config_factory, monkeypatch):
    monkeypatch.setattr("vllm.config.vllm.envs.VLLM_USE_V2_MODEL_RUNNER", False)
    with pytest.raises(ValueError, match="Model Runner V2"):
        _make_vllm_uno_config(uno_config_factory)


def test_uno_requires_lora_enabled(uno_config_factory, monkeypatch):
    monkeypatch.setattr("vllm.config.vllm.HAS_TRITON", True)
    monkeypatch.setattr("vllm.config.vllm.envs.VLLM_USE_V2_MODEL_RUNNER", True)
    scheduler = SchedulerConfig(
        max_model_len=2048,
        max_num_seqs=4,
        max_num_batched_tokens=256,
        is_encoder_decoder=False,
        async_scheduling=None,
    )
    with pytest.raises(ValueError, match="enable-lora"):
        VllmConfig(
            device_config=DeviceConfig(device="cpu"),
            speculative_config=uno_config_factory(),
            scheduler_config=scheduler,
        )


def test_uno_requires_two_lora_slots(uno_config_factory, monkeypatch):
    monkeypatch.setattr("vllm.config.vllm.HAS_TRITON", True)
    monkeypatch.setattr("vllm.config.vllm.envs.VLLM_USE_V2_MODEL_RUNNER", True)
    scheduler = SchedulerConfig(
        max_model_len=2048,
        max_num_seqs=4,
        max_num_batched_tokens=256,
        is_encoder_decoder=False,
        async_scheduling=None,
    )
    with pytest.raises(ValueError, match=r"max_loras >= 2"):
        VllmConfig(
            device_config=DeviceConfig(device="cpu"),
            speculative_config=uno_config_factory(),
            lora_config=LoRAConfig(max_loras=1, lora_dtype=torch.bfloat16),
            scheduler_config=scheduler,
        )


def test_uno_rejects_request_specific_lora(uno_config_factory):
    processor = InputProcessor.__new__(InputProcessor)
    processor.speculative_config = uno_config_factory()
    processor.lora_config = LoRAConfig(max_loras=2, lora_dtype=torch.bfloat16)

    with pytest.raises(VLLMValidationError, match="request-specific"):
        processor._validate_lora(MagicMock())
