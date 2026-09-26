# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Configuration-level regression tests for Qwen4Exp GDN/PLE RecoverSSM selection.

``--use-replayssm`` with speculative decoding selects RecoverSSM for Qwen4Exp, as it
does for Kimi-K3 KDA. The MTP drafter's derived config shares ``cache_config`` and must
keep the choice. These tests run the real validator function against stand-in configs.
"""

from types import SimpleNamespace
from typing import Any, cast

import pytest

from vllm.config.mamba import MambaBackendEnum
from vllm.config.vllm import VllmConfig
from vllm.model_executor.layers.mamba.recoverssm_utils import uses_recoverssm
from vllm.v1.attention.backend import AttentionCGSupport

TARGET = "Qwen4ExpForConditionalGeneration"
DRAFTER = "Qwen4ExpMTP"


def _validator():
    validators = VllmConfig.__pydantic_decorators__.model_validators  # type: ignore[attr-defined]
    return validators["validate_mamba_cached_kernel"].func


def _config(
    arch: str,
    cache_config=None,
    num_spec: int = 3,
    use_replayssm: bool = True,
    supports_replayssm: bool = True,
    **overrides,
):
    cfg = SimpleNamespace(
        num_speculative_tokens=num_spec,
        model_config=SimpleNamespace(
            architecture=arch, supports_replayssm=supports_replayssm
        ),
        cache_config=cache_config
        or SimpleNamespace(
            use_recoverssm=False,
            use_replayssm=use_replayssm,
            mamba_cache_mode="align",
        ),
        mamba_config=SimpleNamespace(
            enable_stochastic_rounding=False, backend=MambaBackendEnum.TRITON
        ),
        use_v2_model_runner=True,
        parallel_config=SimpleNamespace(pipeline_parallel_size=1),
        kv_transfer_config=None,
    )
    for key, value in overrides.items():
        target, _, attr = key.partition("__")
        if attr:
            setattr(getattr(cfg, target), attr, value)
        else:
            setattr(cfg, target, value)
    return cfg


def test_validator_is_registered():
    assert "validate_mamba_cached_kernel" in (
        VllmConfig.__pydantic_decorators__.model_validators  # type: ignore[attr-defined]
    )


@pytest.mark.parametrize(
    "arch", ["Qwen4ExpForConditionalGeneration", "Qwen4ExpForCausalLM"]
)
def test_replayssm_with_spec_selects_recoverssm(arch):
    cfg = _config(arch)
    _validator()(cfg)
    assert cfg.cache_config.use_recoverssm is True
    assert uses_recoverssm(cfg.cache_config, 3)


def test_derived_drafter_keeps_the_choice_and_runs_the_checks():
    target = _config(TARGET)
    _validator()(target)
    drafter = _config(DRAFTER, cache_config=target.cache_config)
    _validator()(drafter)
    assert target.cache_config.use_recoverssm is True
    bad = _config(
        DRAFTER,
        cache_config=target.cache_config,
        parallel_config__pipeline_parallel_size=2,
    )
    with pytest.raises(ValueError, match="pipeline_parallel_size"):
        _validator()(bad)


@pytest.mark.parametrize(
    ("override", "value", "match"),
    [
        ("parallel_config__pipeline_parallel_size", 2, "pipeline_parallel_size"),
        ("mamba_config__enable_stochastic_rounding", True, "stochastic"),
        ("mamba_config__backend", "flashinfer", "mamba-backend triton"),
        ("use_v2_model_runner", False, "V2_MODEL_RUNNER"),
        ("cache_config__mamba_cache_mode", "all", "none and align"),
    ],
)
def test_runtime_checks_reject(override, value, match):
    cfg = _config(TARGET, **{override: value})
    with pytest.raises(ValueError, match=match):
        _validator()(cfg)


def test_flag_unset_keeps_the_stock_path():
    cfg = _config(TARGET, use_replayssm=False)
    _validator()(cfg)
    assert cfg.cache_config.use_recoverssm is False
    assert not uses_recoverssm(cfg.cache_config, 3)


def test_qwen4exp_replayssm_requires_speculative_decoding():
    with pytest.raises(ValueError, match="requires speculative decoding"):
        _validator()(_config(TARGET, num_spec=0))


def test_architecture_without_replayssm_support_is_rejected():
    cfg = _config("Qwen3NextForCausalLM", supports_replayssm=False)
    with pytest.raises(ValueError, match="not supported for architecture"):
        _validator()(cfg)


def test_uses_recoverssm_predicate():
    on = SimpleNamespace(use_recoverssm=True)
    assert uses_recoverssm(on, 3)
    assert not uses_recoverssm(on, 0)
    assert not uses_recoverssm(SimpleNamespace(use_recoverssm=False), 3)
    assert not uses_recoverssm(SimpleNamespace(), 3)


def test_builders_fall_back_to_piecewise():
    from vllm.v1.attention.backends.gdn_recoverssm import GDNRecoverSSMMetadataBuilder
    from vllm.v1.attention.backends.ple_recoverssm import PleRecoverSSMMetadataBuilder

    for builder in (GDNRecoverSSMMetadataBuilder, PleRecoverSSMMetadataBuilder):
        assert (
            builder.get_cudagraph_support(cast(Any, None), cast(Any, None))
            == AttentionCGSupport.NEVER
        )


@pytest.mark.parametrize("compile_mode", ["VLLM_COMPILE", "NONE"])
@pytest.mark.parametrize(("breakable", "expected"), [("1", "PIECEWISE"), ("0", "NONE")])
def test_never_support_falls_back_to_piecewise_with_breakable_cudagraphs(
    monkeypatch, compile_mode, breakable, expected
):
    """A backend without full-graph support must keep piecewise graphs when breakable
    CUDA graphs split at attention (splitting_ops is empty then, and torch.compile may
    be off)."""
    from vllm.config.compilation import (
        CompilationConfig,
        CompilationMode,
        CUDAGraphMode,
    )

    monkeypatch.setenv("VLLM_USE_BREAKABLE_CUDAGRAPH", breakable)
    compilation_config = CompilationConfig(
        mode=CompilationMode[compile_mode],
        cudagraph_mode=CUDAGraphMode.FULL_AND_PIECEWISE,
        splitting_ops=[],
    )
    resolved = compilation_config.resolve_cudagraph_mode_and_sizes(
        AttentionCGSupport.NEVER,
        "GDNRecoverSSMAttentionBackend",
        uniform_decode_query_len=4,
    )
    assert resolved.name == expected
