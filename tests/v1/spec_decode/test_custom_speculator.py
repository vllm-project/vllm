# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for speculator configs and custom V2 speculators registered with
`register_speculative_method`."""

import subprocess
import sys
import textwrap
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest
import torch

import vllm.envs as envs
from tests.v1.spec_decode.utils import fake_speculative_config
from vllm.config import SpeculativeConfig, VllmConfig
from vllm.model_executor.models.registry import ModelRegistry
from vllm.v1.worker.gpu.spec_decode import init_speculator
from vllm.v1.worker.gpu.spec_decode import speculator_config as speculator_config_module
from vllm.v1.worker.gpu.spec_decode.ngram.config import NgramGPUSpeculatorConfig
from vllm.v1.worker.gpu.spec_decode.speculator import BaseSpeculator
from vllm.v1.worker.gpu.spec_decode.speculator_config import SpeculatorConfig
from vllm.v1.worker.gpu.spec_decode.standalone_ar.config import (
    DraftModelSpeculatorConfig,
)
from vllm.v1.worker.gpu.spec_decode.v1_speculator_configs import (
    V1CustomProposerConfig,
)


def _qualname(cls: type) -> str:
    return f"{__name__}.{cls.__qualname__}"


class DummySpeculator(BaseSpeculator):
    def __init__(self, vllm_config: VllmConfig, device: torch.device, req_states):
        self.vllm_config = vllm_config
        self.req_states = req_states

    def init_cudagraph_manager(self, cudagraph_mode) -> None:
        pass

    def capture(self) -> None:
        pass

    def propose(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        raise NotImplementedError


class DummySpeculatorConfig(SpeculatorConfig):
    speculator_class = _qualname(DummySpeculator)


class DummyNgramGPUSpeculatorConfig(NgramGPUSpeculatorConfig):
    speculator_class = _qualname(DummySpeculator)


class DummyDraftModelSpeculatorConfig(DraftModelSpeculatorConfig):
    speculator_class = _qualname(DummySpeculator)


class OptionsSpeculatorConfig(DummySpeculatorConfig):
    supported_options = frozenset({"speculator_options"})

    @classmethod
    def apply_options(cls, spec: SpeculativeConfig) -> None:
        if (spec.speculator_options or {}).keys() - {"seed"}:
            raise ValueError("unknown option")


class NoSpeculatorClassConfig(SpeculatorConfig):
    pass


class NotASpeculatorConfig(SpeculatorConfig):
    speculator_class = _qualname(NoSpeculatorClassConfig)


@pytest.fixture
def register(monkeypatch):
    """Register custom speculative methods for a test only."""
    monkeypatch.setattr(speculator_config_module, "_CUSTOM_CONFIGS", {})
    return speculator_config_module.register_speculative_method


def _custom_spec_config(method: str = "dummy", **kwargs: Any) -> SpeculativeConfig:
    return SpeculativeConfig(method=method, num_speculative_tokens=3, **kwargs)


@pytest.mark.parametrize("lazy", [False, True])
def test_custom_speculator(register, lazy: bool):
    register(
        "dummy", _qualname(DummySpeculatorConfig) if lazy else DummySpeculatorConfig
    )
    spec_config = _custom_spec_config()

    # The configured name is kept.
    assert spec_config.method == "dummy"
    assert spec_config.base_method is None
    assert type(spec_config.speculator_config) is DummySpeculatorConfig

    config = VllmConfig()
    config.speculative_config = spec_config
    assert config.num_lookahead_tokens == 0
    with patch.object(envs, "VLLM_USE_V2_MODEL_RUNNER", None):
        assert config.use_v2_model_runner
    with (
        patch.object(envs, "VLLM_USE_V2_MODEL_RUNNER", False),
        pytest.raises(ValueError, match="'dummy' requires Model Runner V2"),
    ):
        _ = config.use_v2_model_runner


def test_custom_speculator_inherits_builtin_config(register):
    register("my_ngram", DummyNgramGPUSpeculatorConfig)
    spec_config = _custom_spec_config("my_ngram")

    # Set up and behaves as the built-in speculator, but builds the custom one.
    assert spec_config.method == "my_ngram"
    assert spec_config.base_method == "ngram_gpu"
    assert spec_config.use_ngram_gpu()
    assert spec_config.prompt_lookup_min == spec_config.prompt_lookup_max == 5
    assert spec_config.speculator_config.needs_dense_token_history
    assert type(spec_config.speculator_config) is DummyNgramGPUSpeculatorConfig

    # A draft model speculator needs a draft model.
    register("my_draft", DummyDraftModelSpeculatorConfig)
    with pytest.raises(ValueError, match="without speculative model"):
        _custom_spec_config("my_draft")


def test_speculator_options(register):
    register("dummy", DummySpeculatorConfig)
    register("with_options", OptionsSpeculatorConfig)

    spec_config = _custom_spec_config("with_options", speculator_options={"seed": 1})
    assert spec_config.speculator_options == {"seed": 1}
    other_seed = _custom_spec_config("with_options", speculator_options={"seed": 2})
    assert spec_config.compute_hash() != other_seed.compute_hash()

    with pytest.raises(ValueError, match="unknown option"):
        _custom_spec_config("with_options", speculator_options={"nope": 1})
    # Only speculators that support options accept them.
    for method in ("dummy", "ngram"):
        with pytest.raises(ValueError, match="speculator_options is not supported"):
            _custom_spec_config(method, speculator_options={"seed": 1})


def test_register_speculative_method(register):
    register("dummy", DummySpeculatorConfig)
    # Registering the same config again (e.g. by another process) is fine.
    register("dummy", DummySpeculatorConfig)
    assert speculator_config_module.registered_speculative_methods() == ["dummy"]

    with pytest.raises(ValueError, match="already registered"):
        register("dummy", DummyNgramGPUSpeculatorConfig)
    for name in ("eagle3", "deepseek_mtp"):
        with pytest.raises(ValueError, match="built-in speculative method"):
            register(name, DummySpeculatorConfig)
    with pytest.raises(ValueError, match=r"Unknown speculative method 'nope'.*dummy"):
        _custom_spec_config("nope")


@pytest.mark.parametrize(
    ("config", "kwargs", "error", "match"),
    [
        (_qualname(DummySpeculator), {}, TypeError, "must be a subclass of"),
        (NoSpeculatorClassConfig, {}, ValueError, "must set speculator_class"),
        (DummySpeculatorConfig, {"dspark_draft_topk": 4}, ValueError, "not supported"),
    ],
)
def test_custom_speculator_validation(register, config, kwargs, error, match):
    register("dummy", config)
    with pytest.raises(error, match=match):
        _custom_spec_config(**kwargs)


def test_v1_custom_proposer_unsupported_by_v2_model_runner():
    config = VllmConfig()
    config.speculative_config = SpeculativeConfig(
        method="custom_class", model="my_module.MyProposer", num_speculative_tokens=3
    )

    assert type(config.speculative_config.speculator_config) is V1CustomProposerConfig
    assert config._get_v2_model_runner_unsupported_features() == [
        "speculative method 'custom_class'"
    ]


def _vllm_config(config_cls: type[SpeculatorConfig], method: str = "dummy") -> Any:
    return SimpleNamespace(
        speculative_config=fake_speculative_config(config_cls, method=method)
    )


def test_init_speculator():
    vllm_config = _vllm_config(DummySpeculatorConfig)
    req_states = object()

    speculator = init_speculator(vllm_config, torch.device("cpu"), req_states)

    assert type(speculator) is DummySpeculator
    assert speculator.vllm_config is vllm_config
    assert speculator.req_states is req_states
    # Defaults the model runner relies on for any speculator.
    assert speculator.draft_logits is None
    assert not speculator.supports_mm_inputs
    speculator.prepare_watermarking(sampler=None, idx_mapping=torch.empty(0))


@pytest.mark.parametrize(
    ("vllm_config", "error", "match"),
    [
        (_vllm_config(NotASpeculatorConfig), TypeError, "must be a subclass of"),
        (
            SimpleNamespace(
                speculative_config=fake_speculative_config(method="suffix")
            ),
            NotImplementedError,
            "suffix is not supported",
        ),
    ],
)
def test_init_speculator_rejects(vllm_config, error, match):
    with pytest.raises(error, match=match):
        init_speculator(vllm_config, torch.device("cpu"), req_states=object())


def test_speculator_configs_do_not_import_speculators():
    """Speculator configs are used in every process (e.g. the scheduler's), so
    importing them must not import any speculator or the model runner."""
    script = textwrap.dedent(
        """
        import importlib, pkgutil, sys

        import vllm.v1.worker.gpu.spec_decode as pkg

        for module in pkgutil.walk_packages(pkg.__path__, pkg.__name__ + "."):
            if module.name.endswith(("config", "_configs")):
                importlib.import_module(module.name)

        suffixes = (".speculator", ".model_runner", ".extract_hidden_states")
        heavy = [
            name
            for name in sys.modules
            if name.startswith("vllm.v1.worker.gpu.") and name.endswith(suffixes)
        ]
        assert not heavy, heavy
        """
    )
    subprocess.run([sys.executable, "-c", script], check=True)


_SPEC = "vllm.v1.worker.gpu.spec_decode"
_DSPARK = f"{_SPEC}.dspark.config.DSparkSpeculatorConfig"
_LILICORR = f"{_SPEC}.lilicorr.config.LiLiCorrSpeculatorConfig"
_GEMMA4_MTP = f"{_SPEC}.gemma4.config.Gemma4MTPSpeculatorConfig"


def _checkpoint_spec(
    method: str | None, model: str, declared: str | None = None, model_type="llama"
) -> Any:
    return SimpleNamespace(
        method=method,
        draft_model_config=SimpleNamespace(
            model=model,
            # The speculator config class declared by the draft model.
            speculator_config_class=declared,
            hf_config=SimpleNamespace(model_type=model_type),
        ),
        use_gemma4_mtp=lambda: False,
        use_multi_module_mtp=lambda: False,
    )


@pytest.mark.parametrize(
    ("method", "model", "declared", "model_type", "expected"),
    [
        # No method configured: detected from the draft checkpoint.
        (None, "org/llama-draft", None, "llama", "draft_model"),
        (None, "yuhuili/EAGLE-LLaMA3-Instruct-8B", None, "llama", "eagle"),
        (None, "AngelSlim/Qwen3-8B_eagle3", None, "llama", "eagle3"),
        (None, "org/mtp", None, "deepseek_mtp", "mtp"),
        (None, "org/draft", _LILICORR, "x", "dflash"),
        # An explicitly configured method is never overridden.
        ("draft_model", "yuhuili/EAGLE-LLaMA3-Instruct-8B", None, "llama", None),
        ("mtp", "org/draft", _GEMMA4_MTP, "gemma4_mtp", None),
        ("eagle", "org/draft", _DSPARK, "llama", None),
    ],
)
def test_resolve_checkpoint_config_class(method, model, declared, model_type, expected):
    spec = _checkpoint_spec(method, model, declared, model_type)
    configured_cls = speculator_config_module.get_method_config_class(spec)

    resolved_cls = speculator_config_module._resolve_checkpoint_config_class(
        spec, configured_cls
    )

    assert resolved_cls.base_method == (expected or method)


@pytest.mark.parametrize(
    ("method", "model", "declared", "model_type", "match"),
    [
        ("mtp", "yuhuili/EAGLE-LLaMA3-Instruct-8B", None, "llama", "not recognized"),
        ("draft_model", "org/mtp", None, "deepseek_mtp", "looks like a 'mtp'"),
        ("draft_model", "org/draft", _DSPARK, "qwen3", "looks like a 'dspark'"),
    ],
)
def test_resolve_checkpoint_config_class_rejects_mismatch(
    method, model, declared, model_type, match
):
    spec = _checkpoint_spec(method, model, declared, model_type)
    configured_cls = speculator_config_module.get_method_config_class(spec)

    with pytest.raises(ValueError, match=match):
        speculator_config_module._resolve_checkpoint_config_class(spec, configured_cls)


@pytest.mark.parametrize(
    ("method", "declared", "multi_module", "expected"),
    [
        ("dflash", None, False, "DFlashSpeculatorConfig"),
        ("dflash", _LILICORR, False, "LiLiCorrSpeculatorConfig"),
        # A declared config of another speculator than configured is ignored.
        ("eagle", _DSPARK, False, "EagleSpeculatorConfig"),
        ("mtp", _GEMMA4_MTP, False, "Gemma4MTPSpeculatorConfig"),
        ("mtp", None, True, "MultiModuleMTPSpeculatorConfig"),
    ],
)
def test_speculator_config_selection(method, declared, multi_module, expected):
    spec = _checkpoint_spec(method, "org/draft", declared)
    spec.use_multi_module_mtp = lambda: multi_module

    config_cls = speculator_config_module.get_speculator_config_class(spec)

    assert config_cls.__name__ == expected


@pytest.mark.parametrize(
    ("architecture", "expected"),
    [
        ("MuseGlimmerAssistantModel", "DFlashSpeculatorConfig"),
        ("LiLiCorrDraftModel", "LiLiCorrSpeculatorConfig"),
        ("DFlashQwen3XPressModel", "XPressSpeculatorConfig"),
        ("MuseSparkDSparkModel", "DSparkSpeculatorConfig"),
        ("DFlash2DraftModel", "DFlash2SpeculatorConfig"),
        ("Qwen3OmniDSparkModel", "DSparkSpeculatorConfig"),
        ("Gemma4MTPModel", "Gemma4MTPSpeculatorConfig"),
        ("DeepSeekMTPModel", None),
    ],
)
def test_draft_models_declare_speculator(architecture, expected):
    model_info = ModelRegistry._try_inspect_model_cls(architecture)
    assert model_info is not None

    declared = model_info.speculator_config_class
    assert (declared and declared.rsplit(".", 1)[-1]) == expected
