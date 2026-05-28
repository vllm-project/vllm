# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the `supports_swiglu_clamp_limit` oracle interface.

The MoE oracle (`FusedMoEExperts.is_supported_config`) skips any backend
that declares `supports_swiglu_clamp_limit() is False` when a model's
`FusedMoEConfig.swiglu_limit` is set. This guards against silently
selecting a backend that does not thread `gemm1_clamp_limit` through every
SwiGLU activation path (see Issue #41985 and the PR #42287 follow-up
review comment from @mgoin).
"""

from unittest.mock import MagicMock

from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.modular_kernel import (
    FusedMoEActivationFormat,
    FusedMoEExperts,
)


# ---------------------------------------------------------------------------
# 1. Default and inheritance of `supports_swiglu_clamp_limit`
# ---------------------------------------------------------------------------


def test_base_default_supports_swiglu_clamp_limit_false():
    """The base class defaults to False so every backend must opt in."""
    assert FusedMoEExperts.supports_swiglu_clamp_limit(MoEActivation.SILU) is False
    assert FusedMoEExperts.supports_swiglu_clamp_limit(MoEActivation.SWIGLUOAI) is False
    assert (
        FusedMoEExperts.supports_swiglu_clamp_limit(MoEActivation.SWIGLUSTEP) is False
    )


def test_marlin_inheritance_and_lora_override():
    """Marlin classes declare clamp support only on SILU (the only branch
    that threads `swiglu_limit_func`); SWIGLUOAI / SWIGLUSTEP fall through
    to `activation_func` and are unwired. MarlinExperts (LoRA mixin)
    additionally declares with_lora=False on every activation due to the
    `_fused_marlin_moe` clamp/activation mutex.
    """
    from vllm.model_executor.layers.fused_moe.experts.marlin_moe import (
        BatchedMarlinExperts,
        MarlinExperts,
        MarlinExpertsBase,
    )

    # SILU is the only SwiGLU branch with clamp threading.
    assert MarlinExpertsBase.supports_swiglu_clamp_limit(MoEActivation.SILU) is True
    assert MarlinExperts.supports_swiglu_clamp_limit(MoEActivation.SILU) is True
    assert BatchedMarlinExperts.supports_swiglu_clamp_limit(MoEActivation.SILU) is True
    # Other SwiGLU variants are NOT clamp-threaded by Marlin.
    assert (
        MarlinExpertsBase.supports_swiglu_clamp_limit(MoEActivation.SWIGLUOAI) is False
    )
    assert MarlinExperts.supports_swiglu_clamp_limit(MoEActivation.SWIGLUOAI) is False
    assert (
        BatchedMarlinExperts.supports_swiglu_clamp_limit(MoEActivation.SWIGLUSTEP)
        is False
    )
    # LoRA-specific override only on MarlinExperts: rejects every SwiGLU
    # activation because the clamp path bypasses `activation_with_lora`.
    assert (
        MarlinExpertsBase.supports_swiglu_clamp_limit_with_lora(MoEActivation.SILU)
        is True
    )
    assert (
        MarlinExperts.supports_swiglu_clamp_limit_with_lora(MoEActivation.SILU) is False
    )
    assert (
        MarlinExperts.supports_swiglu_clamp_limit_with_lora(MoEActivation.SWIGLUOAI)
        is False
    )
    assert (
        BatchedMarlinExperts.supports_swiglu_clamp_limit_with_lora(MoEActivation.SILU)
        is True
    )


# ---------------------------------------------------------------------------
# 2. Filter behavior in `FusedMoEExperts.is_supported_config`
# ---------------------------------------------------------------------------


class _DummyBase(FusedMoEExperts):
    """Shared scaffolding: satisfies every elif check in
    `FusedMoEExperts.is_supported_config` except the SwiGLU clamp filter,
    so subclasses isolate that single branch.
    """

    @staticmethod
    def supports_lora() -> bool:
        return True

    @staticmethod
    def _supports_current_device() -> bool:
        return True

    @staticmethod
    def _supports_no_act_and_mul() -> bool:
        return True

    @staticmethod
    def _supports_activation(activation) -> bool:
        return True

    @staticmethod
    def _supports_quant_scheme(weight_key, activation_key) -> bool:
        return True

    @staticmethod
    def _supports_parallel_config(parallel_config) -> bool:
        return True

    @staticmethod
    def activation_format() -> FusedMoEActivationFormat:
        return FusedMoEActivationFormat.Standard


class _SupportingDummy(_DummyBase):
    """Backend that declares it threads clamp through every SwiGLU path."""

    @staticmethod
    def supports_swiglu_clamp_limit(activation) -> bool:
        return True


class _RejectingDummy(_DummyBase):
    """Backend that declares it does NOT support clamp."""

    @staticmethod
    def supports_swiglu_clamp_limit(activation) -> bool:
        return False


class _SiluOnlyDummy(_DummyBase):
    """Backend that declares clamp support only on SILU. Models the new
    per-activation contract: future-proofs against non-SILU SwiGLU
    variants reaching a backend whose clamp threading is SILU-only.
    """

    @staticmethod
    def supports_swiglu_clamp_limit(activation) -> bool:
        return activation == MoEActivation.SILU


def _make_mock_config(
    swiglu_limit: float | None = None,
    is_lora_enabled: bool = False,
    activation: MoEActivation = MoEActivation.SILU,
) -> MagicMock:
    """Minimal stand-in for `FusedMoEConfig` covering only the fields read
    by `is_supported_config`.
    """
    return MagicMock(
        is_act_and_mul=True,
        activation=activation,
        moe_parallel_config=None,
        routing_method=None,
        router_logits_dtype=None,
        hidden_dim=128,
        is_lora_enabled=is_lora_enabled,
        swiglu_limit=swiglu_limit,
    )


def test_filter_passes_when_swiglu_limit_unset():
    """`swiglu_limit is None` -> filter never trips regardless of backend."""
    config = _make_mock_config(swiglu_limit=None)
    supported, reason = FusedMoEExperts.is_supported_config(
        _RejectingDummy, config, None, None, FusedMoEActivationFormat.Standard
    )
    assert supported is True
    assert reason is None


def test_filter_passes_when_backend_supports_clamp():
    """`swiglu_limit` set, backend declares support -> filter passes."""
    config = _make_mock_config(swiglu_limit=7.0)
    supported, reason = FusedMoEExperts.is_supported_config(
        _SupportingDummy, config, None, None, FusedMoEActivationFormat.Standard
    )
    assert supported is True
    assert reason is None


def test_filter_rejects_when_backend_lacks_clamp_support():
    """`swiglu_limit` set, backend declares no support -> filter rejects
    with a clamp-specific reason string."""
    config = _make_mock_config(swiglu_limit=7.0)
    supported, reason = FusedMoEExperts.is_supported_config(
        _RejectingDummy, config, None, None, FusedMoEActivationFormat.Standard
    )
    assert supported is False
    assert reason is not None
    assert "SwiGLU clamp limit" in reason


def test_filter_passes_when_swiglu_limit_zero_regardless_of_backend():
    """`swiglu_limit=0.0` is treated as no clamp (matches the
    `swiglu_limit_func` guard `if swiglu_limit > 0` in utils.py); filter
    never trips even against an unsupporting backend."""
    config = _make_mock_config(swiglu_limit=0.0)
    supported, reason = FusedMoEExperts.is_supported_config(
        _RejectingDummy, config, None, None, FusedMoEActivationFormat.Standard
    )
    assert supported is True
    assert reason is None


class _SupportingButNotWithLoraDummy(_DummyBase):
    """Mimics MarlinExperts post-fix: declares clamp True but
    with_lora False."""

    @staticmethod
    def supports_swiglu_clamp_limit(activation) -> bool:
        return True

    @staticmethod
    def supports_swiglu_clamp_limit_with_lora(activation) -> bool:
        return False


def test_filter_passes_lora_disabled_with_clamp_backend_no_lora_support():
    """Backend supports clamp but not with-LoRA: filter passes when LoRA off."""
    config = _make_mock_config(swiglu_limit=7.0, is_lora_enabled=False)
    supported, _ = FusedMoEExperts.is_supported_config(
        _SupportingButNotWithLoraDummy,
        config,
        None,
        None,
        FusedMoEActivationFormat.Standard,
    )
    assert supported is True


def test_filter_rejects_lora_enabled_with_clamp_backend_no_lora_support():
    """Backend supports clamp but not with-LoRA: filter rejects when LoRA on."""
    config = _make_mock_config(swiglu_limit=7.0, is_lora_enabled=True)
    supported, reason = FusedMoEExperts.is_supported_config(
        _SupportingButNotWithLoraDummy,
        config,
        None,
        None,
        FusedMoEActivationFormat.Standard,
    )
    assert supported is False
    assert "LoRA" in reason


def test_filter_passes_silu_only_backend_with_silu_config():
    """A SILU-only backend (e.g. Marlin post-refactor) passes when the
    model's activation is SILU and swiglu_limit is set."""
    config = _make_mock_config(swiglu_limit=7.0, activation=MoEActivation.SILU)
    supported, reason = FusedMoEExperts.is_supported_config(
        _SiluOnlyDummy, config, None, None, FusedMoEActivationFormat.Standard
    )
    assert supported is True
    assert reason is None


def test_filter_rejects_silu_only_backend_with_non_silu_clamp_config():
    """A SILU-only backend is rejected when the model's activation is a
    non-SILU SwiGLU variant (SWIGLUOAI / SWIGLUSTEP) and swiglu_limit is
    set. This is the future-proofing path the per-activation contract
    was introduced for."""
    for non_silu in (MoEActivation.SWIGLUOAI, MoEActivation.SWIGLUSTEP):
        config = _make_mock_config(swiglu_limit=7.0, activation=non_silu)
        supported, reason = FusedMoEExperts.is_supported_config(
            _SiluOnlyDummy, config, None, None, FusedMoEActivationFormat.Standard
        )
        assert supported is False, f"expected rejection for {non_silu}"
        assert reason is not None
        assert "SwiGLU clamp limit" in reason


def test_filter_rejects_lora_plus_swiglu_combo():
    """When `is_lora_enabled=True` and `swiglu_limit` is set, a backend
    that declares `supports_swiglu_clamp_limit=False` (the Marlin-post-fix
    pattern) is filtered out by the SwiGLU filter — *not* the LoRA filter,
    since `_RejectingDummy.supports_lora()` returns True. This is the
    safety net for Marlin-style mutex bugs where the clamp path silently
    bypasses LoRA injection (see gemini-code-assist comment on #42287).

    Note: we mimic Marlin's post-fix structure with `_RejectingDummy`
    instead of importing real `MarlinExperts`, because real backends fail
    earlier `_supports_current_device()` checks on CPU-only test hosts.
    """
    config = _make_mock_config(swiglu_limit=7.0, is_lora_enabled=True)
    supported, reason = FusedMoEExperts.is_supported_config(
        _RejectingDummy, config, None, None, FusedMoEActivationFormat.Standard
    )
    assert supported is False
    assert "SwiGLU clamp limit" in reason
    # Verify the SwiGLU filter rejected, not the LoRA filter, because
    # `_RejectingDummy.supports_lora()` returns True.
    assert "LoRA" not in reason
