# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A speculative method registered by a plugin is built by Model Runner V2."""

from types import SimpleNamespace

import pytest

from vllm import envs
from vllm.config import SpeculativeConfig, VllmConfig
from vllm.v1.spec_decode import registry
from vllm.v1.spec_decode.registry import (
    SpeculativeMethodSpec,
    register_speculative_method,
)
from vllm.v1.worker.gpu.spec_decode import init_speculator

NAME = "unit_test_runner_method"
SPEC_CLS = "tests.v1.spec_decode.test_method_registry_runner:RecordingSpeculator"


class RecordingSpeculator:
    def __init__(self, vllm_config, device):
        self.vllm_config = vllm_config
        self.device = device


@pytest.fixture
def registered_config(monkeypatch):
    monkeypatch.setattr(registry, "_REGISTRY", {})
    register_speculative_method(NAME, SpeculativeMethodSpec(SPEC_CLS))
    return SpeculativeConfig(method=NAME, num_speculative_tokens=3)


def test_init_speculator_builds_the_registered_class(registered_config):
    vllm_config = SimpleNamespace(speculative_config=registered_config)
    speculator = init_speculator(vllm_config, "cpu", req_states=object())
    assert isinstance(speculator, RecordingSpeculator)
    assert speculator.vllm_config is vllm_config
    assert speculator.device == "cpu"


def test_init_speculator_is_unchanged_for_builtin_methods(monkeypatch):
    import vllm.v1.worker.gpu.spec_decode.ngram.speculator as ngram_module

    class FakeNgram:
        def __init__(self, vllm_config, device, req_states):
            self.args = (vllm_config, device, req_states)

    monkeypatch.setattr(ngram_module, "NgramGPUSpeculator", FakeNgram)
    config = SpeculativeConfig(method="ngram", num_speculative_tokens=3)
    vllm_config = SimpleNamespace(speculative_config=config)
    speculator = init_speculator(vllm_config, "cpu", req_states="states")
    assert isinstance(speculator, FakeNgram)
    assert speculator.args == (vllm_config, "cpu", "states")


def _use_v2(config: SpeculativeConfig) -> bool:
    fake = SimpleNamespace(
        attention_config=SimpleNamespace(hisparse_config=None),
        speculative_config=config,
    )
    return VllmConfig.use_v2_model_runner.fget(fake)  # type: ignore[attr-defined]


def test_registered_method_selects_model_runner_v2(registered_config, monkeypatch):
    monkeypatch.setattr(envs, "VLLM_USE_V2_MODEL_RUNNER", None)
    assert _use_v2(registered_config) is True


def test_registered_method_rejects_forced_v1(registered_config, monkeypatch):
    monkeypatch.setattr(envs, "VLLM_USE_V2_MODEL_RUNNER", False)
    with pytest.raises(ValueError, match="requires Model Runner V2"):
        _use_v2(registered_config)


def test_builtin_method_keeps_the_env_choice(monkeypatch):
    config = SpeculativeConfig(method="ngram", num_speculative_tokens=3)
    monkeypatch.setattr(envs, "VLLM_USE_V2_MODEL_RUNNER", False)
    assert _use_v2(config) is False
