# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for speculator configs and custom V2 speculators provided via
`speculator_config_class`."""

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
from vllm.v1.worker.gpu.spec_decode import init_speculator
from vllm.v1.worker.gpu.spec_decode.speculator import BaseSpeculator
from vllm.v1.worker.gpu.spec_decode.speculator_config import (
    SpeculatorConfig,
    UnsupportedSpeculatorConfig,
)
from vllm.v1.worker.gpu.spec_decode.standalone_ar.config import (
    DraftModelSpeculatorConfig,
)


def _qualname(cls: type) -> str:
    return f"{__name__}.{cls.__qualname__}"


class DummySpeculator(BaseSpeculator):
    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        self.vllm_config = vllm_config
        self.device = device

    def init_cudagraph_manager(self, cudagraph_mode) -> None:
        pass

    def capture(self) -> None:
        pass

    def propose(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        raise NotImplementedError


class DummyReqStatesSpeculator(DummySpeculator):
    def __init__(self, vllm_config: VllmConfig, device: torch.device, req_states):
        super().__init__(vllm_config, device)
        self.req_states = req_states


class NotASpeculator:
    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        pass


class DummySpeculatorConfig(SpeculatorConfig):
    speculator_class = _qualname(DummySpeculator)


class DummyReqStatesSpeculatorConfig(SpeculatorConfig):
    speculator_class = _qualname(DummyReqStatesSpeculator)


class NotASpeculatorConfig(SpeculatorConfig):
    speculator_class = _qualname(NotASpeculator)


class DummyDraftModelSpeculatorConfig(DraftModelSpeculatorConfig):
    speculator_class = _qualname(DummySpeculator)


class NoSpeculatorClassConfig(SpeculatorConfig):
    pass


def _custom_spec_config(**kwargs: Any) -> SpeculativeConfig:
    args: dict[str, Any] = dict(
        method="custom_class",
        speculator_config_class=_qualname(DummySpeculatorConfig),
        base_method="ngram_gpu",
        num_speculative_tokens=3,
    )
    args.update(kwargs)
    return SpeculativeConfig(**args)


def test_custom_speculator_with_base_method():
    spec_config = _custom_spec_config()

    # The draft config is created as for the base method...
    assert spec_config.method == "ngram_gpu"
    assert spec_config.prompt_lookup_min is not None
    assert spec_config.prompt_lookup_max is not None
    # ...but the speculator's behavior is described by its own config.
    assert type(spec_config.speculator_config) is DummySpeculatorConfig
    assert not spec_config.speculator_config.needs_dense_token_history


def test_custom_speculator_infers_custom_class_method():
    spec_config = _custom_spec_config(method=None)

    assert spec_config.method == "ngram_gpu"


@pytest.mark.parametrize("method", ["custom_class", None])
def test_custom_speculator_without_base_method(method: str | None):
    spec_config = _custom_spec_config(method=method, base_method=None)

    assert spec_config.method == "custom_class"
    assert type(spec_config.speculator_config) is DummySpeculatorConfig

    config = VllmConfig()
    config.speculative_config = spec_config
    assert config.num_lookahead_tokens == 0
    assert config.num_prefill_lookahead_tokens == 0
    assert config._get_v2_model_runner_unsupported_features() == []
    with patch.object(envs, "VLLM_USE_V2_MODEL_RUNNER", None):
        assert config.use_v2_model_runner


def test_custom_speculator_inherits_builtin_behavior():
    spec_config = _custom_spec_config(
        speculator_config_class=_qualname(DummyDraftModelSpeculatorConfig),
        base_method=None,
    )
    config = VllmConfig()
    config.speculative_config = spec_config

    # Draft KV lookahead slots, as for a draft model.
    assert config.num_lookahead_tokens == 3
    assert spec_config.max_num_new_slots_for_drafting == 1


@pytest.mark.parametrize(
    ("kwargs", "error", "match"),
    [
        (
            {"speculator_config_class": None},
            ValueError,
            "base_method requires speculator_config_class",
        ),
        ({"base_method": "custom_class"}, ValueError, "cannot be 'custom_class'"),
        (
            {"method": "ngram"},
            ValueError,
            "speculator_config_class requires method='custom_class'",
        ),
        (
            {"speculator_config_class": _qualname(DummySpeculator)},
            TypeError,
            "must be a subclass of",
        ),
        (
            {"speculator_config_class": _qualname(NoSpeculatorClassConfig)},
            ValueError,
            "must set speculator_class",
        ),
    ],
)
def test_custom_speculator_validation(
    kwargs: dict[str, Any], error: type[Exception], match: str
):
    with pytest.raises(error, match=match):
        _custom_spec_config(**kwargs)


def test_custom_speculator_requires_v2_model_runner():
    config = VllmConfig()
    config.speculative_config = _custom_spec_config()

    with patch.object(envs, "VLLM_USE_V2_MODEL_RUNNER", None):
        assert config.use_v2_model_runner
    with (
        patch.object(envs, "VLLM_USE_V2_MODEL_RUNNER", False),
        pytest.raises(ValueError, match="requires Model Runner V2"),
    ):
        _ = config.use_v2_model_runner


def test_v1_custom_proposer_unsupported_by_v2_model_runner():
    config = VllmConfig()
    config.speculative_config = SpeculativeConfig(
        method="custom_class", model="my_module.MyProposer", num_speculative_tokens=3
    )

    assert type(config.speculative_config.speculator_config) is (
        UnsupportedSpeculatorConfig
    )
    assert config._get_v2_model_runner_unsupported_features() == [
        "speculative method 'custom_class'"
    ]


def _vllm_config(config_cls: type[SpeculatorConfig]) -> VllmConfig:
    return SimpleNamespace(  # type: ignore[return-value]
        speculative_config=fake_speculative_config(config_cls, method="custom_class")
    )


def test_init_custom_speculator():
    vllm_config = _vllm_config(DummySpeculatorConfig)
    device = torch.device("cpu")

    speculator = init_speculator(vllm_config, device, req_states=object())

    assert type(speculator) is DummySpeculator
    assert speculator.vllm_config is vllm_config
    assert speculator.device == device
    # Defaults the model runner relies on for any speculator.
    assert speculator.draft_logits is None
    assert not speculator.supports_mm_inputs
    speculator.prepare_watermarking(sampler=None, idx_mapping=torch.empty(0))


def test_init_custom_speculator_passes_req_states():
    req_states = object()

    speculator = init_speculator(
        _vllm_config(DummyReqStatesSpeculatorConfig), torch.device("cpu"), req_states
    )

    assert isinstance(speculator, DummyReqStatesSpeculator)
    assert speculator.req_states is req_states


def test_init_speculator_rejects_non_speculator():
    with pytest.raises(TypeError, match="must be a subclass of"):
        init_speculator(
            _vllm_config(NotASpeculatorConfig), torch.device("cpu"), req_states=object()
        )


def test_init_speculator_rejects_unsupported_method():
    vllm_config = SimpleNamespace(
        speculative_config=fake_speculative_config(method="suffix")
    )
    with pytest.raises(NotImplementedError, match="suffix is not supported"):
        init_speculator(vllm_config, torch.device("cpu"), req_states=object())


def test_speculator_configs_do_not_import_speculators():
    """Speculator configs are used in every process (e.g. the scheduler's), so
    importing them must not import any speculator or the model runner."""
    script = textwrap.dedent(
        """
        import sys

        from vllm.utils.import_utils import resolve_obj_by_qualname
        from vllm.v1.worker.gpu.spec_decode import speculator_config

        qualnames = list(speculator_config._CONFIG_BY_METHOD.values()) + [
            speculator_config._builtin(module, name)
            for module, name in (
                ("lilicorr.config", "LiLiCorrSpeculatorConfig"),
                ("dflash2.config", "DFlash2SpeculatorConfig"),
                ("gemma4.config", "Gemma4MTPSpeculatorConfig"),
                ("multi_module_mtp.config", "MultiModuleMTPSpeculatorConfig"),
            )
        ]
        for qualname in qualnames:
            config_cls = resolve_obj_by_qualname(qualname)
            assert issubclass(config_cls, speculator_config.SpeculatorConfig)
            assert config_cls.speculator_class is not None, qualname

        pkg = "vllm.v1.worker.gpu."
        heavy = sorted(
            name
            for name in sys.modules
            if name.startswith(pkg)
            and (
                name.endswith(".speculator")
                or name
                in (
                    pkg + "model_runner",
                    pkg + "spec_decode.speculator",
                    pkg + "spec_decode.extract_hidden_states",
                )
            )
        )
        assert not heavy, heavy
        """
    )
    subprocess.run([sys.executable, "-c", script], check=True)
