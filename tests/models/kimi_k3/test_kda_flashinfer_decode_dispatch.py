# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only tests for how the Kimi-K3 KDA layer calls FlashInfer's fused decode.

When the resolved KDA decode backend is ``flashinfer`` and the installed
FlashInfer ``fused_kda_decode`` accepts ``backend`` and ``state_indices_mode``
(FlashInfer >= 0.7.1rc1), the layer passes ``backend="auto"`` and
``state_indices_mode="unique_or_null"`` so FlashInfer selects the kernel itself;
otherwise (the pinned 0.7.0.post1) the call is unchanged.
"""

import ast
import inspect
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from vllm.models.kimi_k3.nvidia import kda as kda_mod
from vllm.utils import flashinfer as flashinfer_utils

pytestmark = pytest.mark.cpu_test

SELECTING_KWARGS = {"backend": "auto", "state_indices_mode": "unique_or_null"}


def _accepting_fused_kda_decode(x, output, *, backend="auto", state_indices_mode=None):
    pass


def _legacy_fused_kda_decode(x, output):
    pass


def _probe_with(monkeypatch, fused_kda_decode, available: bool = True) -> bool:
    """Run the cached probe as if FlashInfer exported ``fused_kda_decode``."""
    monkeypatch.setattr(
        flashinfer_utils, "has_flashinfer_fused_kda_decode", lambda: available
    )
    monkeypatch.setattr(
        flashinfer_utils,
        "_get_submodule",
        lambda name: (
            SimpleNamespace(fused_kda_decode=fused_kda_decode)
            if name == "flashinfer.kda_decode"
            else None
        ),
    )
    probe = flashinfer_utils.flashinfer_fused_kda_decode_selects_backend
    probe.cache_clear()
    try:
        return probe()
    finally:
        probe.cache_clear()


def test_signature_probe_requires_backend_and_state_indices_mode():
    accepts = flashinfer_utils._fused_kda_decode_accepts_backend
    assert accepts(_accepting_fused_kda_decode)
    assert not accepts(_legacy_fused_kda_decode)
    assert not accepts(object())  # no signature at all


def test_selects_backend_probe_follows_the_installed_signature(monkeypatch):
    assert _probe_with(monkeypatch, _accepting_fused_kda_decode)
    assert not _probe_with(monkeypatch, _legacy_fused_kda_decode)
    # No fused decode at all: nothing to select, whatever the signature says.
    assert not _probe_with(monkeypatch, _accepting_fused_kda_decode, available=False)


@pytest.mark.parametrize("selects_backend", [True, False], ids=["auto", "legacy"])
def test_fused_decode_passes_auto_kwargs_only_when_flashinfer_accepts_them(
    monkeypatch, selects_backend
):
    monkeypatch.setattr(
        kda_mod,
        "flashinfer_fused_kda_decode_selects_backend",
        lambda: selects_backend,
    )
    kernel = MagicMock()
    monkeypatch.setattr(kda_mod, "flashinfer_fused_kda_decode", kernel)

    layer = object.__new__(kda_mod.KimiK3DeltaAttention)
    attrs = {
        # As __init__ sets it once kda_decode_backend resolved to "flashinfer".
        "flashinfer_kda_decode_kwargs": kda_mod._flashinfer_kda_decode_kwargs(
            "flashinfer"
        ),
        "decode_conv1d_weight": torch.zeros(3, 4, 8),
        "A_log": torch.zeros(1),
        "dt_bias": torch.zeros(1),
        "decode_norm_weight": torch.ones(128),
        "gate_lower_bound": -5.0,
        "o_norm": SimpleNamespace(eps=1e-6),
    }
    for name, value in attrs.items():
        object.__setattr__(layer, name, value)
    tensors = [torch.zeros(1) for _ in range(8)]

    layer._flashinfer_fused_kda_decode(*tensors)

    kwargs = kernel.call_args.kwargs
    assert kwargs["x"] is tensors[0]
    assert kwargs["output"] is tensors[-1]
    assert kwargs["weight"] is attrs["decode_conv1d_weight"]
    if selects_backend:
        assert kwargs["backend"] == "auto"
        assert kwargs["state_indices_mode"] == "unique_or_null"
    else:
        assert "backend" not in kwargs
        assert "state_indices_mode" not in kwargs


@pytest.mark.parametrize("selects_backend", [True, False], ids=["auto", "legacy"])
@pytest.mark.parametrize("backend", ["native", "flashinfer", "triton"])
def test_init_decision_adds_kwargs_only_for_the_flashinfer_backend(
    monkeypatch, backend, selects_backend
):
    monkeypatch.setattr(
        kda_mod,
        "flashinfer_fused_kda_decode_selects_backend",
        lambda: selects_backend,
    )
    expected = SELECTING_KWARGS if backend == "flashinfer" and selects_backend else {}
    assert kda_mod._flashinfer_kda_decode_kwargs(backend) == expected


def test_flashinfer_selection_announcement_renders_the_kwargs():
    """The once-only announcement must hash: it receives a rendered string, not the
    dict (``logger.info_once`` hashes its arguments; a dict raised ``TypeError``
    while constructing every KDA layer)."""
    assert (
        kda_mod._format_kda_decode_kwargs(SELECTING_KWARGS)
        == "backend=auto, state_indices_mode=unique_or_null"
    )
    # The real logger and the production kwargs: must not raise.
    kda_mod._announce_flashinfer_kda_decode(dict(SELECTING_KWARGS))
    kda_mod._announce_flashinfer_kda_decode(dict(SELECTING_KWARGS))


def test_forward_keeps_the_eager_break_decorator():
    # The decorator is the identity unless breakable CUDA-graph capture is
    # enabled, so a lost decorator would be invisible at runtime.
    tree = ast.parse(inspect.getsource(kda_mod))
    layer = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef) and node.name == "KimiK3DeltaAttention"
    )
    forward = next(
        node
        for node in layer.body
        if isinstance(node, ast.FunctionDef) and node.name == "_forward"
    )
    assert any(
        getattr(decorator, "id", None) == "eager_break_during_capture"
        for decorator in forward.decorator_list
    )
