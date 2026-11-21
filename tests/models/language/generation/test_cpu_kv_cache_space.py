# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CI-runnable guard for the VLLM_CPU_KVCACHE_SPACE semantics.

The unit tests in tests/platforms/test_cpu.py are not picked up by any CI
job (see tools/pre_commit/test_tethering_allowlist.txt), so the same
semantics are guarded here with the cpu_model marker, which the CPU
hardware job runs (it exercises this directory with `-m cpu_model`).
"""

import logging
from types import SimpleNamespace

import pytest
import torch

from vllm.config import CacheConfig, VllmConfig
from vllm.platforms.cpu import CpuPlatform

pytestmark = pytest.mark.cpu_model


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


def _cpu_config(kv_cache_memory_bytes: int | None) -> VllmConfig:
    model_config = SimpleNamespace(
        architecture="qwen3_5",
        has_inner_state=True,
        use_mla=False,
        hf_config=SimpleNamespace(model_type="qwen3_5"),
        hf_text_config=SimpleNamespace(
            model_type="qwen3_5",
            layer_types=("linear_attention",),
        ),
        registry=_FakeRegistry,
    )
    cache_config = CacheConfig(kv_cache_memory_bytes=kv_cache_memory_bytes)
    config = VllmConfig(cache_config=cache_config)
    config.model_config = model_config
    config.cache_config.mamba_ssm_cache_dtype = "float32"
    return config


@pytest.mark.parametrize(
    (
        "env_value",
        "kv_cache_memory_bytes",
        "expected_bytes",
        "expected_warning",
    ),
    [
        pytest.param("0", None, None, None, id="env-zero-no-override"),
        pytest.param(
            "-1", None, None, "not a positive size", id="env-negative-ignored"
        ),
        pytest.param("4", None, 4 * 1024**3, None, id="env-positive-applied"),
        pytest.param(
            "4",
            50_000_000,
            50_000_000,
            "kv_cache_memory_bytes is explicitly set",
            id="env-loses-to-explicit-bytes",
        ),
    ],
)
def test_cpu_kv_cache_space_env_var(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    env_value: str,
    kv_cache_memory_bytes: int | None,
    expected_bytes: int | None,
    expected_warning: str | None,
) -> None:
    # Build the config before setting the env var: VllmConfig.__post_init__
    # already invokes current_platform.check_and_update_config, and a second
    # pass on a config that saw the env var would spuriously warn about
    # kv_cache_memory_bytes being "explicitly set".
    config = _cpu_config(kv_cache_memory_bytes)
    monkeypatch.setenv("VLLM_CPU_KVCACHE_SPACE", env_value)

    with caplog.at_level(logging.WARNING):
        CpuPlatform.check_and_update_config(config)
    assert config.cache_config.kv_cache_memory_bytes == expected_bytes

    kv_cache_warnings = [
        record.message
        for record in caplog.records
        if record.levelno >= logging.WARNING
        and "VLLM_CPU_KVCACHE_SPACE" in record.message
    ]
    if expected_warning is None:
        assert not kv_cache_warnings
    else:
        assert any(expected_warning in message for message in kv_cache_warnings)
