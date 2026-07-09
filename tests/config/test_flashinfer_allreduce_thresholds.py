# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import patch

import vllm.envs as envs
from vllm.config import PassConfig


class _Capability:

    def __init__(self, value: int):
        self.value = value

    def to_int(self) -> int:
        return self.value


def _clear_env_cache() -> None:
    if hasattr(envs.__getattr__, "cache_clear"):
        envs.__getattr__.cache_clear()


def test_flashinfer_allreduce_threshold_env_override(monkeypatch):
    monkeypatch.setenv("VLLM_FLASHINFER_ALLREDUCE_FUSION_THRESHOLDS_MB",
                       '{"3": 64, "8": 2.5}')
    _clear_env_cache()

    with patch("vllm.config.compilation.current_platform") as platform:
        platform.is_cuda.return_value = True
        platform.get_device_capability.return_value = _Capability(120)

        thresholds = PassConfig.default_fi_allreduce_fusion_max_size_mb()
        max_size = PassConfig().flashinfer_max_size(3)

    assert thresholds[3] == 64
    assert thresholds[8] == 2.5
    assert max_size == 64 * 1024 * 1024
    assert isinstance(max_size, int)


def test_flashinfer_allreduce_threshold_default_sm120_tp3(monkeypatch):
    monkeypatch.delenv("VLLM_FLASHINFER_ALLREDUCE_FUSION_THRESHOLDS_MB",
                       raising=False)
    _clear_env_cache()

    with patch("vllm.config.compilation.current_platform") as platform:
        platform.is_cuda.return_value = True
        platform.get_device_capability.return_value = _Capability(120)

        thresholds = PassConfig.default_fi_allreduce_fusion_max_size_mb()

    assert thresholds[3] == 4
