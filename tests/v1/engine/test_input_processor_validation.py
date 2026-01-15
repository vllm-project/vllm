# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.config import CacheConfig, DeviceConfig, ModelConfig, VllmConfig
from vllm.sampling_params import SamplingParams
from vllm.v1.engine import input_processor as input_processor_mod
from vllm.v1.engine.input_processor import InputProcessor


def _mock_input_processor(monkeypatch, tokenizer=None) -> InputProcessor:
    """
    Create a Processor instance with minimal configuration suitable for unit
    tests without accessing external resources.
    """
    monkeypatch.setattr(
        ModelConfig, "try_get_generation_config", lambda self: {}, raising=True
    )
    monkeypatch.setattr(
        ModelConfig, "__post_init__", lambda self, *args: None, raising=True
    )
    monkeypatch.setattr(
        ModelConfig,
        "verify_with_parallel_config",
        lambda self, parallel_config: None,
        raising=True,
    )
    monkeypatch.setattr(
        input_processor_mod,
        "processor_cache_from_config",
        lambda vllm_config, mm_registry: None,
        raising=True,
    )

    monkeypatch.setattr(VllmConfig, "__post_init__", lambda self: None)

    model_config = ModelConfig(
        skip_tokenizer_init=True,
        max_model_len=128,
        mm_processor_cache_gb=4.0,
        generation_config="vllm",
        tokenizer="dummy",
    )

    class _MockMMConfig:
        mm_processor_cache_gb = 4.0

    model_config.multimodal_config = _MockMMConfig()
    vllm_config = VllmConfig(
        model_config=model_config,
        cache_config=CacheConfig(enable_prefix_caching=True),
        device_config=DeviceConfig(device="cpu"),
    )

    return InputProcessor(vllm_config, tokenizer=tokenizer)


def test_allowed_token_ids_empty_raises_vllm_validation_error(monkeypatch):
    """Verify empty allowed_token_ids raises VLLMValidationError with metadata."""
    from vllm.exceptions import VLLMValidationError

    input_processor = _mock_input_processor(monkeypatch)

    with pytest.raises(VLLMValidationError) as exc_info:
        input_processor._validate_sampling_params(SamplingParams(allowed_token_ids=[]))

    assert exc_info.value.parameter == "allowed_token_ids"
    assert exc_info.value.value is not None


def test_allowed_token_ids_oob_raises_vllm_validation_error(monkeypatch):
    """Verify OOB allowed_token_ids raises VLLMValidationError with metadata."""
    from vllm.exceptions import VLLMValidationError

    class MockTokenizer:
        max_token_id = 1000
        vocab_size = 1000

        def __len__(self):
            return self.vocab_size

    input_processor = _mock_input_processor(monkeypatch, tokenizer=MockTokenizer())

    with pytest.raises(VLLMValidationError) as exc_info:
        input_processor._validate_sampling_params(
            SamplingParams(allowed_token_ids=[10000000])
        )

    assert exc_info.value.parameter == "allowed_token_ids"
    assert exc_info.value.value is not None
