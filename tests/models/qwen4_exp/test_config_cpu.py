# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import sys
import types
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from vllm.model_executor.models.config import (
    Qwen3_5ForConditionalGenerationConfig,
    Qwen4ExpForConditionalGenerationConfig,
)
from vllm.models.qwen4_exp.cpu import runtime as cpu_runtime
from vllm.platforms import current_platform
from vllm.platforms.interface import CpuArchEnum
from vllm.transformers_utils.configs.qwen4_exp import Qwen4ExpTextConfig


def _text_config() -> Qwen4ExpTextConfig:
    return Qwen4ExpTextConfig(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        layer_types=["linear_attention", "full_attention"],
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        num_experts=0,
        hc_count=2,
        hc_lowrank=4,
        ple_layer_ids=[1],
        mtp_num_hidden_layers=1,
        mtp={"hybrid": True},
    )


def _vllm_config(restriction: str | None = None) -> SimpleNamespace:
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_text_config=_text_config(),
            hf_config=_text_config(),
            multimodal_config=SimpleNamespace(language_model_only=True),
            enforce_eager=restriction == "eager",
        ),
        parallel_config=SimpleNamespace(
            pipeline_parallel_size=1,
            tensor_parallel_size=1,
            enable_dbo=False,
            ubatch_size=1,
        ),
        speculative_config=None,
        lora_config=None,
        use_v2_model_runner=restriction != "model_runner",
    )
    if restriction == "tensor_parallel":
        config.parallel_config.tensor_parallel_size = 2
    elif restriction == "speculative":
        config.speculative_config = SimpleNamespace()
    elif restriction == "lora":
        config.lora_config = SimpleNamespace()
    elif restriction == "multimodal":
        config.model_config.multimodal_config.language_model_only = False
    return config


@pytest.mark.parametrize(
    ("restriction", "error", "message"),
    [
        ("architecture", NotImplementedError, "x86-64"),
        ("tensor_parallel", NotImplementedError, "tensor_parallel_size=1"),
        ("speculative", NotImplementedError, "speculative decoding"),
        ("lora", NotImplementedError, "LoRA"),
        ("multimodal", NotImplementedError, "text-only"),
        ("model_runner", ValueError, "Model Runner V2"),
        ("eager", ValueError, "compiled model execution"),
        ("triton", ValueError, "active CPU backend"),
    ],
)
def test_qwen4_exp_cpu_rejects_unsupported_runtime(
    restriction: str,
    error: type[Exception],
    message: str,
) -> None:
    cpu_arch = CpuArchEnum.ARM if restriction == "architecture" else CpuArchEnum.X86
    with (
        patch.object(
            Qwen3_5ForConditionalGenerationConfig,
            "verify_and_update_config",
        ),
        patch.object(current_platform, "is_cpu", return_value=True),
        patch.object(
            current_platform,
            "get_cpu_architecture",
            return_value=cpu_arch,
        ),
        patch.object(
            cpu_runtime,
            "has_active_triton_cpu_backend",
            return_value=restriction != "triton",
        ),
        pytest.raises(error, match=message),
    ):
        Qwen4ExpForConditionalGenerationConfig.verify_and_update_config(
            _vllm_config(restriction)
        )


def test_qwen4_exp_cpu_accepts_supported_runtime() -> None:
    with (
        patch.object(
            Qwen3_5ForConditionalGenerationConfig,
            "verify_and_update_config",
        ),
        patch.object(current_platform, "is_cpu", return_value=True),
        patch.object(
            current_platform,
            "get_cpu_architecture",
            return_value=CpuArchEnum.X86,
        ),
        patch.object(
            cpu_runtime,
            "has_active_triton_cpu_backend",
            return_value=True,
        ),
    ):
        Qwen4ExpForConditionalGenerationConfig.verify_and_update_config(_vllm_config())


@pytest.mark.parametrize(
    ("backend", "expected"),
    [
        pytest.param("cpu", True, id="cpu"),
        pytest.param("cuda", False, id="cuda"),
    ],
)
def test_active_triton_cpu_backend(backend: str, expected: bool) -> None:
    target = SimpleNamespace(backend=backend)
    driver = SimpleNamespace(active=SimpleNamespace(get_current_target=lambda: target))
    triton_runtime = types.ModuleType("triton.runtime")
    triton_runtime.__dict__["driver"] = driver
    with (
        patch.object(cpu_runtime, "HAS_TRITON", True),
        patch.dict(sys.modules, {"triton.runtime": triton_runtime}),
    ):
        assert cpu_runtime.has_active_triton_cpu_backend() is expected


def test_unavailable_triton_cpu_backend_fails_closed() -> None:
    with patch.object(cpu_runtime, "HAS_TRITON", False):
        assert not cpu_runtime.has_active_triton_cpu_backend()

    driver = SimpleNamespace(
        active=SimpleNamespace(
            get_current_target=Mock(side_effect=RuntimeError("no active driver"))
        )
    )
    triton_runtime = types.ModuleType("triton.runtime")
    triton_runtime.__dict__["driver"] = driver
    with (
        patch.object(cpu_runtime, "HAS_TRITON", True),
        patch.dict(sys.modules, {"triton.runtime": triton_runtime}),
    ):
        assert not cpu_runtime.has_active_triton_cpu_backend()
