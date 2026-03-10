# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.model_executor.layers.quantization import awq_marlin
from vllm.model_executor.layers.quantization.awq_marlin import AWQMarlinConfig


def _force_marlin_compatible(monkeypatch):
    monkeypatch.setattr(
        AWQMarlinConfig,
        "is_awq_marlin_compatible",
        classmethod(lambda cls, _: True),
    )


def test_awq_marlin_auto_falls_back_when_driver_older(monkeypatch):
    _force_marlin_compatible(monkeypatch)
    monkeypatch.setattr(awq_marlin, "_get_driver_cuda_version", lambda: (12, 8))
    monkeypatch.setattr(awq_marlin, "_parse_cuda_version", lambda _: (12, 9))
    monkeypatch.setattr(awq_marlin.envs, "VLLM_ENABLE_CUDA_COMPATIBILITY", False)

    assert AWQMarlinConfig.override_quantization_method({}, user_quant=None) is None


def test_awq_marlin_auto_kept_with_cuda_compatibility(monkeypatch):
    _force_marlin_compatible(monkeypatch)
    monkeypatch.setattr(awq_marlin, "_get_driver_cuda_version", lambda: (12, 8))
    monkeypatch.setattr(awq_marlin, "_parse_cuda_version", lambda _: (12, 9))
    monkeypatch.setattr(awq_marlin.envs, "VLLM_ENABLE_CUDA_COMPATIBILITY", True)

    assert (
        AWQMarlinConfig.override_quantization_method({}, user_quant=None)
        == "awq_marlin"
    )


def test_awq_marlin_explicit_request_not_overridden(monkeypatch):
    _force_marlin_compatible(monkeypatch)
    monkeypatch.setattr(awq_marlin, "_get_driver_cuda_version", lambda: (12, 8))
    monkeypatch.setattr(awq_marlin, "_parse_cuda_version", lambda _: (12, 9))
    monkeypatch.setattr(awq_marlin.envs, "VLLM_ENABLE_CUDA_COMPATIBILITY", False)

    assert (
        AWQMarlinConfig.override_quantization_method({}, user_quant="awq_marlin")
        == "awq_marlin"
    )
