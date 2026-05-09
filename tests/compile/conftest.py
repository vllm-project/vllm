# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from contextlib import contextmanager
from typing import Any, Protocol
from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm.config import ModelConfig
from vllm.platforms.interface import DeviceCapability


class CompileModelConfigFactory(Protocol):
    def __call__(self, *, dtype: torch.dtype, **kwargs: Any) -> ModelConfig: ...


@pytest.fixture
def mock_cuda_platform():
    """
    Fixture that returns a factory for creating mocked CUDA platforms.

    Usage:
        def test_something(mock_cuda_platform):
            with mock_cuda_platform(is_cuda=True, capability=(9, 0)):
                # test code
    """

    @contextmanager
    def _mock_platform(is_cuda: bool = True, capability: tuple[int, int] | None = None):
        mock_platform = MagicMock()
        mock_platform.is_cuda.return_value = is_cuda
        device_capability = (
            DeviceCapability(*capability) if capability is not None else None
        )
        mock_platform.get_device_capability.return_value = device_capability

        def is_device_capability_family(
            requested_capability: int, device_id: int = 0
        ) -> bool:
            current_capability = mock_platform.get_device_capability(
                device_id=device_id
            )
            if current_capability is None:
                return False
            return current_capability.major == (requested_capability // 10)

        mock_platform.is_device_capability_family.side_effect = (
            is_device_capability_family
        )
        with patch("vllm.platforms.current_platform", mock_platform):
            yield mock_platform

    return _mock_platform


@pytest.fixture(scope="session")
def compile_test_qwen3_model_path(tmp_path_factory: pytest.TempPathFactory) -> str:
    model_dir = tmp_path_factory.mktemp("compile_test_qwen3_model")
    # Minimal local metadata fixture based on Qwen/Qwen3-0.6B's config.json.
    # Compile pass tests instantiate their own tiny modules and only need
    # ModelConfig metadata, not tokenizer files or weights.
    config = {
        "architectures": ["Qwen3ForCausalLM"],
        "attention_bias": False,
        "attention_dropout": 0.0,
        "bos_token_id": 151643,
        "eos_token_id": 151645,
        "head_dim": 128,
        "hidden_act": "silu",
        "hidden_size": 1024,
        "initializer_range": 0.02,
        "intermediate_size": 3072,
        "max_position_embeddings": 4096,
        "max_window_layers": 2,
        "model_type": "qwen3",
        "num_attention_heads": 16,
        "num_hidden_layers": 2,
        "num_key_value_heads": 8,
        "rms_norm_eps": 1e-06,
        "rope_scaling": None,
        "rope_theta": 1000000,
        "sliding_window": None,
        "tie_word_embeddings": True,
        "torch_dtype": "bfloat16",
        "transformers_version": "4.51.0",
        "use_cache": True,
        "use_sliding_window": False,
        "vocab_size": 151936,
    }
    (model_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
    return str(model_dir)


@pytest.fixture(scope="session")
def compile_test_llama_model_path(tmp_path_factory: pytest.TempPathFactory) -> str:
    model_dir = tmp_path_factory.mktemp("compile_test_llama_model")
    # Minimal local metadata fixture based on RedHatAI/Llama-3.2-1B-Instruct-FP8's
    # Llama shape. Distributed compile pass tests use this only to construct a
    # ModelConfig, not to load tokenizer files or weights.
    config = {
        "architectures": ["LlamaForCausalLM"],
        "attention_bias": False,
        "attention_dropout": 0.0,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "head_dim": 128,
        "hidden_act": "silu",
        "hidden_size": 1024,
        "initializer_range": 0.02,
        "intermediate_size": 3072,
        "max_position_embeddings": 4096,
        "model_type": "llama",
        "num_attention_heads": 16,
        "num_hidden_layers": 2,
        "num_key_value_heads": 8,
        "rms_norm_eps": 1e-06,
        "rope_theta": 1000000,
        "tie_word_embeddings": False,
        "torch_dtype": "bfloat16",
        "transformers_version": "4.51.0",
        "use_cache": True,
        "vocab_size": 32000,
    }
    (model_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
    return str(model_dir)


@pytest.fixture
def make_compile_test_model_config(
    compile_test_qwen3_model_path: str,
) -> CompileModelConfigFactory:
    """Create local model metadata for compile pass unit tests.

    The tests instantiate tiny hand-written modules and only need stable
    ModelConfig metadata. Use local Qwen3 metadata because it mirrors the
    ModelConfig default model without requiring HF access, and skip tokenizer
    init so tests do not depend on tokenizer files or downloads.
    """

    def make_model_config(*, dtype: torch.dtype, **kwargs: Any) -> ModelConfig:
        return ModelConfig(
            model=compile_test_qwen3_model_path,
            tokenizer=compile_test_qwen3_model_path,
            dtype=dtype,
            skip_tokenizer_init=True,
            **kwargs,
        )

    return make_model_config
