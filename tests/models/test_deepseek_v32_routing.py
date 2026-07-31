# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Which DSA implementation and MTP draft an architecture resolves to.

The optimized implementation under ``deepseek_v32/nvidia`` is SM100-only;
everywhere else the package re-exports the generic ``deepseek_v2`` classes. The
draft model has to follow the target, and the DeepSeek-family MTP returns a
(logit_hidden, recycle_hidden) tuple that the proposer has to know about. None
of that needs a GPU to check, and the initialization tests skip these models
(``is_available_online=False``), so it would otherwise go uncovered.
"""

import importlib
import sys
from types import SimpleNamespace

import pytest

from vllm.config.speculative import SpeculativeConfig
from vllm.model_executor.models.registry import (
    _SPECULATIVE_DECODING_MODELS,
    _TEXT_GENERATION_MODELS,
)
from vllm.v1.spec_decode.llm_base_proposer import SpecDecodeBaseProposer


def _reload_package(
    monkeypatch: pytest.MonkeyPatch, *, sm100: bool, rocm: bool = False
):
    """Re-import the package with the platform forced, as at process start."""
    from vllm.platforms import current_platform

    monkeypatch.setattr(current_platform, "is_rocm", lambda: rocm)
    monkeypatch.setattr(current_platform, "is_xpu", lambda: False)
    monkeypatch.setattr(current_platform, "is_cuda", lambda: not rocm)
    monkeypatch.setattr(
        current_platform,
        "is_device_capability_family",
        lambda family: sm100 and not rocm and family == 100,
    )
    import vllm.models.deepseek_v32 as pkg

    return importlib.reload(pkg)


@pytest.fixture(autouse=True)
def _restore_package():
    """Drop the patched module so the next importer re-imports it for real.

    Reloading here instead would race monkeypatch teardown and could cache a
    module built against a faked platform.
    """
    yield
    sys.modules.pop("vllm.models.deepseek_v32", None)


def test_sm100_uses_the_optimized_implementation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pkg = _reload_package(monkeypatch, sm100=True)

    assert pkg.DeepseekV32ForCausalLM.__module__.endswith("deepseek_v32.nvidia.model")
    assert pkg.DeepseekV32MTP.__module__.endswith("deepseek_v32.nvidia.mtp")
    # GLM-5.2 reuses the same DSA module rather than getting its own.
    assert pkg.GlmMoeDsaForCausalLM is pkg.DeepseekV32ForCausalLM


def test_pre_sm100_cuda_falls_back_to_generic(monkeypatch: pytest.MonkeyPatch) -> None:
    """H100/CPU keep the deepseek_v2 path instead of failing to import."""
    pkg = _reload_package(monkeypatch, sm100=False)

    assert pkg.DeepseekV32ForCausalLM.__module__.endswith("models.deepseek_v2")
    assert pkg.GlmMoeDsaForCausalLM.__module__.endswith("models.deepseek_v2")
    assert pkg.DeepseekV32MTP.__module__.endswith("models.deepseek_mtp")


def test_rocm_keeps_its_own_dsa_port_but_generic_glm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """DeepSeek V3.2 has an AMD DSA port; GLM-5.2 does not, so it stays generic."""
    pkg = _reload_package(monkeypatch, sm100=False, rocm=True)

    assert pkg.DeepseekV32ForCausalLM.__module__.endswith("deepseek_v32.amd.model")
    assert pkg.DeepseekV32MTP.__module__.endswith("deepseek_v32.amd.mtp")
    assert pkg.GlmMoeDsaForCausalLM.__module__.endswith("models.deepseek_v2")


def test_both_architectures_are_registered() -> None:
    assert _TEXT_GENERATION_MODELS["GlmMoeDsaForCausalLM"] == (
        "vllm.models.deepseek_v32",
        "GlmMoeDsaForCausalLM",
    )
    assert _SPECULATIVE_DECODING_MODELS["DeepseekV32MTPModel"] == (
        "vllm.models.deepseek_v32",
        "DeepseekV32MTP",
    )


class _HfConfigStub:
    """The bits of PretrainedConfig that hf_config_override touches."""

    def __init__(self, model_type: str, architecture: str) -> None:
        self.architectures = [architecture]
        self.model_type = model_type
        self.num_nextn_predict_layers = 1

    def update(self, values: dict) -> None:
        self.__dict__.update(values)


def _mtp_arch(model_type: str, architecture: str = "SomeForCausalLM") -> list[str]:
    hf_config = _HfConfigStub(model_type, architecture)
    SpeculativeConfig.hf_config_override(hf_config)
    return hf_config.architectures


def test_glm_selects_the_sparse_mtp() -> None:
    assert _mtp_arch("glm_moe_dsa") == ["DeepseekV32MTPModel"]


@pytest.mark.parametrize("model_type", ["deepseek_v3", "deepseek_v32"])
def test_other_deepseek_keeps_the_original_mtp(model_type: str) -> None:
    """Only the DSA-sparse family gets the new draft; V3 must not move."""
    assert _mtp_arch(model_type) == ["DeepSeekMTPModel"]


@pytest.mark.parametrize(
    "architectures,expected",
    [
        (["DeepSeekMTPModel"], True),
        (["DeepseekV32MTPModel"], True),
        (["Glm4MoeMTPModel"], False),
        ([], False),
    ],
)
def test_tuple_return_contract_covers_the_new_architecture(
    architectures: list[str], expected: bool
) -> None:
    """The DSA MTP recycles the post-norm hidden, so it returns a 2-tuple.

    Miss this and the proposer feeds a tuple where a tensor is expected.
    """
    proposer = SimpleNamespace(
        method="mtp",
        draft_model_config=SimpleNamespace(
            hf_config=SimpleNamespace(architectures=architectures)
        ),
    )

    assert SpecDecodeBaseProposer.model_returns_tuple(proposer) is expected
