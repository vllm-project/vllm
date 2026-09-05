# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Capability-check tests for the FlashInfer fused GDN decode route.

The route must only activate when the installed FlashInfer exports both
``gdn_fused_decode_step`` and a routing probe that understands
``conv_state_layout``. Anything older keeps the stock path bit-for-bit
unchanged, and there are three distinct "older or broken" cases the resolver
has to survive, all of which must fail CLOSED:

* a FlashInfer that predates the op — ``import flashinfer`` succeeds and the
  **attribute is missing**, which is the whole reason the resolver reads it
  with a ``getattr`` default and then checks for ``None``;
* a FlashInfer whose probe predates conv-state-layout awareness;
* a FlashInfer that cannot be imported at all (a broken/partial install),
  which must not propagate out of layer construction.

The route adds no backend name and no config surface: support is the only
gate, and it is asked per step, so a probe that declines every geometry must
NOT stop the route from resolving. The single exception is the fallback
control ``VLLM_ENABLE_QWEN_GDN_FUSED_DECODE=0``, which holds the route off
entirely so the layer runs the pre-integration decode op -- that is what
makes a one-build A/B possible, and it must log a marker distinct from
"unavailable" so an off arm cannot be confused with a missing install.

The custom op wrapping the route must be registered as a piecewise-cudagraph
splitting op.
"""

from __future__ import annotations

import sys
import types
from unittest.mock import patch

from vllm import envs
from vllm.config import CompilationConfig
from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn as mod


class _Cfg:
    """Duck-typed VllmConfig: the resolver only reads additional_config."""

    def __init__(self, additional_config: dict | None = None):
        self.additional_config = additional_config or {}


def _probe_with_layout(batch_size, *, conv_state_layout="SD", **kwargs):
    return True


def _probe_without_layout(batch_size, hidden_size=0, device=None):
    return True


def _declining_probe(batch_size, *, conv_state_layout="SD", **kwargs):
    """What a layout-aware probe returns for an unsupported geometry."""
    return False


def _fake_flashinfer(with_api: bool = True, probe=_probe_with_layout) -> dict:
    """sys.modules patch for a fake flashinfer install.

    FlashInfer exports the op at the top level, so ``with_api=False`` models
    a FlashInfer that predates it: the package imports fine and the two
    attributes are simply absent.
    """
    root = types.ModuleType("flashinfer")
    if with_api:
        root.gdn_fused_decode_step = lambda *args, **kwargs: None
        root.gdn_fused_decode_step_supported = probe
    return {"flashinfer": root}


@patch.object(mod.current_platform, "is_cuda", return_value=True)
def test_resolver_requires_the_flashinfer_api(_):
    """A FlashInfer without the op -> stock path, via a MISSING ATTRIBUTE.

    This is the shape of "too old" now that the op is a top-level export:
    ``import flashinfer`` succeeds on every FlashInfer ever released, so the
    import cannot be the gate. The resolver reads the two names with a
    ``getattr`` default and returns ``None`` when either is absent — drop
    that check and an old install would resolve to ``(None, None)`` and blow
    up on the first decode step instead of keeping the stock chain.
    """
    with patch.dict(sys.modules, _fake_flashinfer(with_api=False), clear=False):
        assert mod._resolve_gdn_fused_decode_step(_Cfg()) is None
    with patch.dict(sys.modules, _fake_flashinfer(), clear=False):
        resolved = mod._resolve_gdn_fused_decode_step(_Cfg())
        assert resolved is not None
        step, supported = resolved
        assert callable(step)
        assert callable(supported)


@patch.object(mod.current_platform, "is_cuda", return_value=True)
def test_resolver_requires_both_names(_):
    """Half an API is not an API: either name missing keeps the stock path."""
    for present in ("gdn_fused_decode_step", "gdn_fused_decode_step_supported"):
        root = types.ModuleType("flashinfer")
        setattr(root, present, _probe_with_layout)
        with patch.dict(sys.modules, {"flashinfer": root}, clear=False):
            assert mod._resolve_gdn_fused_decode_step(_Cfg()) is None, present


@patch.object(mod.current_platform, "is_cuda", return_value=True)
def test_resolver_survives_an_unimportable_flashinfer(_):
    """A build where ``import flashinfer`` itself raises keeps the stock path.

    Rare, but it is the branch the ``try`` still exists for: a partial or
    broken install must land on the stock chain, not propagate an exception
    out of layer construction. ``None`` in ``sys.modules`` is CPython's way
    of making an import raise ``ImportError``.
    """
    with patch.dict(sys.modules, {"flashinfer": None}, clear=False):
        assert mod._resolve_gdn_fused_decode_step(_Cfg()) is None


@patch.object(mod.current_platform, "is_cuda", return_value=True)
def test_resolver_adds_no_config_surface(_):
    """No backend name, no knob: an unrelated additional_config cannot turn
    the route on or off. Support is the gate."""
    with patch.dict(sys.modules, _fake_flashinfer(), clear=False):
        for additional_config in ({}, {"gdn_decode_backend": "triton"}, None):
            resolved = mod._resolve_gdn_fused_decode_step(_Cfg(additional_config))
            assert resolved is not None


@patch.object(mod.current_platform, "is_cuda", return_value=True)
def test_fallback_control_holds_the_route_off(_):
    """VLLM_ENABLE_QWEN_GDN_FUSED_DECODE=0 keeps the stock chain on a build
    that is fully capable -- the one-build A/B baseline, and the field
    fallback for a bad fusion."""
    with (
        patch.dict(sys.modules, _fake_flashinfer(), clear=False),
        patch.object(envs, "VLLM_ENABLE_QWEN_GDN_FUSED_DECODE", False),
    ):
        assert mod._resolve_gdn_fused_decode_step(_Cfg()) is None
    # ...and the default leaves it on, so the control cannot silently be
    # the reason a measurement saw no dispatch.
    with (
        patch.dict(sys.modules, _fake_flashinfer(), clear=False),
        patch.object(envs, "VLLM_ENABLE_QWEN_GDN_FUSED_DECODE", True),
    ):
        assert mod._resolve_gdn_fused_decode_step(_Cfg()) is not None


def test_resolver_requires_cuda():
    with (
        patch.object(mod.current_platform, "is_cuda", return_value=False),
        patch.dict(sys.modules, _fake_flashinfer(), clear=False),
    ):
        assert mod._resolve_gdn_fused_decode_step(_Cfg()) is None


@patch.object(mod.current_platform, "is_cuda", return_value=True)
def test_resolver_requires_layout_aware_probe(_):
    """A probe that predates conv_state_layout assumes a dim-first conv pool
    and must keep the route unavailable (vLLM defaults to state-first)."""
    with patch.dict(
        sys.modules, _fake_flashinfer(probe=_probe_without_layout), clear=False
    ):
        assert mod._resolve_gdn_fused_decode_step(_Cfg()) is None


@patch.object(mod.current_platform, "is_cuda", return_value=True)
def test_resolver_is_capability_only(_):
    """A probe that declines every geometry still resolves: the route is
    chosen once per layer, and each decode step asks the probe again before
    dispatching, falling back to the stock path. Support is per step, so it
    cannot be answered once at construction."""
    with patch.dict(sys.modules, _fake_flashinfer(probe=_declining_probe), clear=False):
        resolved = mod._resolve_gdn_fused_decode_step(_Cfg())
        assert resolved is not None
        assert resolved[1](1, conv_state_layout="SD") is False


def test_fused_core_op_is_a_splitting_op():
    """qwen_gdn_attention_core_fi reads per-step forward context/metadata on
    the host and must never be inlined into PIECEWISE cudagraph pieces."""
    assert "vllm::qwen_gdn_attention_core_fi" in CompilationConfig._attention_ops
