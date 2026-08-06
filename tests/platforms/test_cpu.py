# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

from vllm.config import CacheConfig, VllmConfig
from vllm.platforms.cpu import CpuPlatform


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
    )
    user_specified_dtype = cache_config.user_specified_mamba_ssm_cache_dtype
    config = VllmConfig(cache_config=cache_config)
    config.model_config = model_config
    cache_config.mamba_ssm_cache_dtype = resolved_dtype
    cache_config.user_specified_mamba_ssm_cache_dtype = user_specified_dtype
    return config


@pytest.mark.parametrize(
    ("requested_dtype", "resolved_dtype"),
    [
        pytest.param("float32", "float32", id="explicit-fp32"),
        pytest.param("bfloat16", "bfloat16", id="explicit-bf16"),
        pytest.param("auto", "bfloat16", id="explicit-auto-model-bf16"),
    ],
)
def test_cpu_amx_gdn_preserves_resolved_state_dtype(
    monkeypatch: pytest.MonkeyPatch,
    requested_dtype: str,
    resolved_dtype: str,
) -> None:
    monkeypatch.setattr("torch.cpu._is_amx_tile_supported", lambda: True)
    cache_config = CacheConfig(mamba_ssm_cache_dtype=requested_dtype)
    config = _cpu_config(
        cache_config,
        model_type="qwen3_5",
        resolved_dtype=resolved_dtype,
    )

    CpuPlatform.check_and_update_config(config)

    assert cache_config.mamba_ssm_cache_dtype == resolved_dtype


def test_cpu_amx_normalizes_model_selected_bf16_for_unsupported_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("torch.cpu._is_amx_tile_supported", lambda: True)
    cache_config = CacheConfig(mamba_ssm_cache_dtype="auto")
    config = _cpu_config(
        cache_config,
        model_type="nemotron_h",
        resolved_dtype="bfloat16",
        layer_types=("mamba",),
    )

    CpuPlatform.check_and_update_config(config)

    assert cache_config.mamba_ssm_cache_dtype == "float32"


def test_cpu_amx_rejects_bf16_for_unsupported_mamba_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("torch.cpu._is_amx_tile_supported", lambda: True)
    cache_config = CacheConfig(mamba_ssm_cache_dtype="bfloat16")
    config = _cpu_config(
        cache_config,
        model_type="nemotron_h",
        resolved_dtype="bfloat16",
        layer_types=("mamba",),
    )

    with pytest.raises(ValueError, match="unsupported for backend"):
        CpuPlatform.check_and_update_config(config)
