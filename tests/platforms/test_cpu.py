# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.config import CacheConfig, VllmConfig
from vllm.platforms.cpu import CpuPlatform


class _FakeModel:
    @classmethod
    def get_mamba_state_dtype_from_config(
        cls, vllm_config: VllmConfig
    ) -> tuple[torch.dtype, torch.dtype]:
        state_dtype = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }.get(vllm_config.cache_config.mamba_ssm_cache_dtype, torch.float32)
        return torch.float32, state_dtype


class _FakeRegistry:
    @staticmethod
    def resolve_model_cls(
        architecture: str,
        model_config: SimpleNamespace,
    ) -> tuple[type[_FakeModel], str]:
        return _FakeModel, architecture


def _cpu_config(
    cache_config: CacheConfig,
    *,
    model_type: str,
    resolved_dtype: str,
    architecture: str | None = None,
    layer_types: tuple[str, ...] = ("linear_attention",),
) -> SimpleNamespace:
    model_config = SimpleNamespace(
        disable_cascade_attn=False,
        architecture=architecture or model_type,
        has_inner_state=True,
        use_mla=False,
        hf_text_config=SimpleNamespace(
            model_type=model_type,
            layer_types=layer_types,
        ),
        registry=_FakeRegistry,
    )
    config = VllmConfig(cache_config=cache_config)
    config.model_config = model_config
    cache_config.mamba_ssm_cache_dtype = resolved_dtype
    return config


@pytest.mark.parametrize(
    (
        "model_type",
        "requested_dtype",
        "resolved_dtype",
        "layer_types",
        "expected_dtype",
    ),
    [
        pytest.param(
            "qwen3_5",
            "bfloat16",
            "bfloat16",
            ("linear_attention",),
            "bfloat16",
            id="gdn-explicit-bf16",
        ),
        pytest.param(
            "qwen3_5",
            "float16",
            "float16",
            ("linear_attention",),
            "float16",
            id="gdn-explicit-fp16",
        ),
        pytest.param(
            "qwen3_5",
            "auto",
            "bfloat16",
            ("linear_attention",),
            "bfloat16",
            id="gdn-model-bf16",
        ),
        pytest.param(
            "qwen3_5",
            "auto",
            "float16",
            ("linear_attention",),
            "float16",
            id="gdn-model-fp16",
        ),
        pytest.param(
            "nemotron_h",
            "auto",
            "bfloat16",
            ("mamba",),
            "float32",
            id="unsupported-model-bf16",
        ),
        pytest.param(
            "nemotron_h",
            "auto",
            "float16",
            ("mamba",),
            "float32",
            id="unsupported-model-fp16",
        ),
        pytest.param(
            "nemotron_h",
            "bfloat16",
            "bfloat16",
            ("mamba",),
            "float32",
            id="unsupported-explicit-bf16",
        ),
        pytest.param(
            "nemotron_h",
            "float16",
            "float16",
            ("mamba",),
            "float32",
            id="unsupported-explicit-fp16",
        ),
    ],
)
def test_cpu_accelerated_gdn_dtype_policy(
    monkeypatch: pytest.MonkeyPatch,
    model_type: str,
    requested_dtype: str,
    resolved_dtype: str,
    layer_types: tuple[str, ...],
    expected_dtype: str,
) -> None:
    monkeypatch.setattr("torch.cpu._is_avx512_bf16_supported", lambda: True)
    cache_config = CacheConfig(mamba_ssm_cache_dtype=requested_dtype)
    config = _cpu_config(
        cache_config,
        model_type=model_type,
        resolved_dtype=resolved_dtype,
        layer_types=layer_types,
    )

    CpuPlatform.check_and_update_config(config)
    assert cache_config.mamba_ssm_cache_dtype == expected_dtype
