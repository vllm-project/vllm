# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import types

import pytest

from vllm.platforms.interface import Platform


def test_platform_supports_v2_model_runner_default():
    assert Platform.supports_v2_model_runner() is True

    class NoV2Platform(Platform):
        @classmethod
        def supports_v2_model_runner(cls) -> bool:
            return False

    assert NoV2Platform.supports_v2_model_runner() is False


def _platform_without_v2() -> types.SimpleNamespace:
    return types.SimpleNamespace(
        device_name="DummyDevice",
        supports_v2_model_runner=lambda: False,
    )


def test_use_v2_model_runner_respects_platform_capability(
    default_vllm_config, monkeypatch: pytest.MonkeyPatch
):
    from vllm import envs

    monkeypatch.setattr(envs, "VLLM_USE_V2_MODEL_RUNNER", None)
    monkeypatch.setattr("vllm.platforms.current_platform", _platform_without_v2())

    assert default_vllm_config.use_v2_model_runner is False


def test_explicit_v2_request_rejected_on_unsupported_platform(
    default_vllm_config, monkeypatch: pytest.MonkeyPatch
):
    from vllm import envs

    monkeypatch.setattr(envs, "VLLM_USE_V2_MODEL_RUNNER", True)
    monkeypatch.setattr("vllm.platforms.current_platform", _platform_without_v2())

    assert default_vllm_config.use_v2_model_runner is True
    with pytest.raises(ValueError, match="does not support the V2 model runner"):
        default_vllm_config._validate_v2_model_runner()
