# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for Quark's online-requant overlay routing.

Routing-only tests for the ``--quantization-config`` overlay that
re-quantizes a Quark mixed-precision checkpoint's unquantized layers via the
shared online methods. Kept in a dedicated module so they run without the
heavy (``lm_eval`` / network) imports in ``test_quark.py``.

The overlay's online method (``Fp8PtpcOnlineLinearMethod``) is not instantiated
directly: its ``__init__`` reads ``get_current_vllm_config().model_config.dtype``
which needs a real model path. So the "routed" case patches the method map with
a sentinel and asserts the routing/ignore logic instead (mirrors the note in
``test_modelopt.py::test_modelopt_mixed_precision_dispatches_w4a16_layer``).

Run `pytest tests/quantization/test_quark_online_overlay.py`.
"""

from unittest.mock import patch

from vllm.config.quantization import QuantizationConfigArgs
from vllm.model_executor.layers.quantization.online.base import (
    _ONLINE_LINEAR_METHODS,
)
from vllm.model_executor.layers.quantization.online.fp8 import (
    Fp8PtpcOnlineLinearMethod,
)
from vllm.model_executor.layers.quantization.quark.quark import QuarkConfig
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kFp8StaticChannelSym,
)

FUSED_MAPPING = {
    "qkv_proj": ["q_proj", "k_proj", "v_proj"],
    "gate_up_proj": ["gate_proj", "up_proj"],
}


class _SentinelMethod:
    """Stand-in for the online method so routing can be tested without a real
    vLLM config context (the real method's ``__init__`` reads model_config)."""


def _config() -> QuarkConfig:
    config = QuarkConfig(quant_config={})
    config.packed_modules_mapping = FUSED_MAPPING
    return config


def test_fp8_per_channel_shorthand_maps_to_ptpc():
    # Contract: the ``fp8_per_channel`` overlay resolves to online PTPC FP8.
    overlay = QuantizationConfigArgs(linear="fp8_per_channel")
    assert overlay.linear.weight is kFp8StaticChannelSym
    assert _ONLINE_LINEAR_METHODS[kFp8StaticChannelSym] is Fp8PtpcOnlineLinearMethod


def test_overlay_routes_non_ignored_linear():
    config = _config()
    overlay = QuantizationConfigArgs(
        linear="fp8_per_channel",
        ignore=["lm_head", "re:.*kv_b_proj.*"],
    )
    with patch.dict(
        _ONLINE_LINEAR_METHODS, {kFp8StaticChannelSym: _SentinelMethod}
    ):
        method = config._online_requant_method(
            overlay, "model.layers.0.self_attn.q_proj"
        )
    assert isinstance(method, _SentinelMethod)


def test_overlay_ignore_leaves_layer_unquantized():
    config = _config()
    overlay = QuantizationConfigArgs(
        linear="fp8_per_channel",
        ignore=["lm_head", "re:.*kv_b_proj.*"],
    )
    # ``kv_b_proj`` must stay bf16 (MLA BMM consumes the raw weight), and the
    # exact ``lm_head`` name is excluded too. Ignored => None (no method built,
    # so no config context is required).
    assert (
        config._online_requant_method(
            overlay, "model.layers.0.self_attn.kv_b_proj"
        )
        is None
    )
    assert config._online_requant_method(overlay, "lm_head") is None


def test_overlay_without_linear_spec_is_inactive():
    # No ``linear`` spec => no overlay; default Quark behavior is unchanged.
    overlay = QuantizationConfigArgs(ignore=["lm_head"])
    assert overlay.linear is None
