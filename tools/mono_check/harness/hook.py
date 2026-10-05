# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Debug-harness install of the GLM-5.2 MonoKernel.

Patches ``DeepseekV32DecoderLayer.forward`` process-wide and routes the mono layers to
a ``MonoLive``. The model-integrated dispatch (``vllm/.../mono/dispatch.py``) needs
none of this; it exists for out-of-tree experiments on an unmodified model.
"""

from __future__ import annotations

_ORIG: dict = dict(forward=None)
_LIVE: dict = dict(lv=None)


def orig_forward(layer, positions, hidden_states, residual):
    """The unpatched ``DeepseekV32DecoderLayer.forward`` of vLLM."""
    fn = _ORIG["forward"]
    if fn is None:
        from vllm.models.deepseek_v32.amd.model import DeepseekV32DecoderLayer

        fn = DeepseekV32DecoderLayer.forward
    return fn(layer, positions, hidden_states, residual)


def _hooked_forward(self, positions, hidden_states, residual):
    owner = getattr(self, "_mono_owner", None)
    if owner is None:
        return _ORIG["forward"](self, positions, hidden_states, residual)
    return owner.forward(self, positions, hidden_states, residual)


def install_layer_hook(layers, owner) -> None:
    """Patch the decoder-layer class once and route ``layers`` to
    ``owner.forward(layer, positions, hidden_states, residual)``."""
    from vllm.models.deepseek_v32.amd.model import DeepseekV32DecoderLayer

    if _ORIG["forward"] is None:
        _ORIG["forward"] = DeepseekV32DecoderLayer.forward
        DeepseekV32DecoderLayer.forward = _hooked_forward
    for layer in layers:
        layer._mono_owner = owner


class LiveOwner:
    """A hooked layer's step logic: what ``Glm5MonoDecode.forward_layer`` does,
    without the custom op."""

    def __init__(self, lv):
        self.lv = lv

    def forward(self, layer, positions, hidden_states, residual):
        lv = self.lv
        if layer.layer_idx == lv.first:
            lv._begin_step(layer, positions, hidden_states, residual)
        if not lv.active:
            return orig_forward(layer, positions, hidden_states, residual)
        return lv.mono_forward(layer, positions, hidden_states, residual)


def install(model, cfg, vllm_config=None):
    """Build a ``MonoLive`` on the loaded model and hook its layers."""
    from vllm.models.deepseek_v32.amd.mono import dispatch, live
    from vllm.models.deepseek_v32.amd.mono.guards import (
        check_after_install,
        check_before_install,
    )

    if dispatch._ACTIVE["obj"] is not None or _LIVE["lv"] is not None:
        raise RuntimeError("mono harness: a MonoKernel is already installed")
    check_before_install(model, cfg, vllm_config)
    lv = live.MonoLive(model, cfg, vllm_config)
    install_layer_hook(lv.layers.values(), LiveOwner(lv))
    check_after_install(lv, vllm_config)
    _LIVE["lv"] = lv
    return lv


def get_live():
    """The harness-installed ``MonoLive``, else the dispatch path's, else None."""
    if _LIVE["lv"] is not None:
        return _LIVE["lv"]
    from vllm.models.deepseek_v32.amd.mono import dispatch

    obj = dispatch._ACTIVE["obj"]
    return None if obj is None else obj.lv
