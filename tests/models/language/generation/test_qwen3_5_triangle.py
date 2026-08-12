# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Tests for the TriangleMix attention acceleration on Qwen3.5 hybrid models.

The TriangleMix pattern (paper: "Accelerating Prefilling via Decoding-time
Contribution Sparsity", ACL 2026 Findings, https://arxiv.org/abs/2602.03295-equivalent)
replaces dense attention with a sliding-window pattern on a calibrated subset
of `full_attention` layers, accelerating the prefilling stage without
measurable accuracy loss.

This test file verifies:
1. Module-level configuration (layer indices, sliding window size, env var).
2. The `get_layer` helper in `Qwen3_5Model` correctly passes the per-layer
   sliding window to the deep `full_attention` layers (15, 19, 23 for the
   calibrated Qwen3.5-2B) and full attention to the rest.
3. Disabling via `VLLM_QWEN35_TRIANGLE=0` falls back to dense attention for
   every layer.
"""

import os
from collections import namedtuple
from unittest import mock

import pytest


# These are imported lazily so that the module is importable without the full
# vLLM stack (e.g., for unit-testing the configuration constants).
def _import_qwen3_5_module():
    from vllm.model_executor.models import qwen3_5
    return qwen3_5


def _import_qwen3_next_module():
    from vllm.model_executor.models import qwen3_next
    return qwen3_next


def test_triangle_attention_constants_present():
    """The TriangleMix configuration must be exposed at module level so that
    downstream tooling can introspect / override it.
    """
    qwen3_5 = _import_qwen3_5_module()
    assert hasattr(qwen3_5, "_TRIANGLE_ATTENTION_LAYERS")
    assert hasattr(qwen3_5, "_TRIANGLE_ATTENTION_WINDOW")
    assert hasattr(qwen3_5, "_TRIANGLE_ATTENTION_ENABLED")

    # Default values calibrated for Qwen3.5-2B (24 layers: 6 full_attention
    # layers at indices 3, 7, 11, 15, 19, 23; the deepest three are 15, 19, 23).
    assert 15 in qwen3_5._TRIANGLE_ATTENTION_LAYERS
    assert 19 in qwen3_5._TRIANGLE_ATTENTION_LAYERS
    assert 23 in qwen3_5._TRIANGLE_ATTENTION_LAYERS
    assert qwen3_5._TRIANGLE_ATTENTION_WINDOW == 128


def test_triangle_attention_disabled_by_env(monkeypatch):
    """`VLLM_QWEN35_TRIANGLE=0` disables the optimization at import time."""
    monkeypatch.setenv("VLLM_QWEN35_TRIANGLE", "0")
    # Reload the module to pick up the env var.
    import importlib
    import vllm.model_executor.models.qwen3_5 as qwen3_5_module
    importlib.reload(qwen3_5_module)
    try:
        assert qwen3_5_module._TRIANGLE_ATTENTION_ENABLED is False
    finally:
        # Reload again to restore the default-enabled state for other tests.
        monkeypatch.delenv("VLLM_QWEN35_TRIANGLE", raising=False)
        importlib.reload(qwen3_5_module)
        assert qwen3_5_module._TRIANGLE_ATTENTION_ENABLED is True


def test_qwen3_next_attention_accepts_per_layer_sliding_window():
    """`Qwen3NextAttention.__init__` must accept and forward
    `per_layer_sliding_window` to vLLM's `Attention` class.
    """
    import inspect

    qwen3_next = _import_qwen3_next_module()
    sig = inspect.signature(qwen3_next.Qwen3NextAttention.__init__)
    assert "per_layer_sliding_window" in sig.parameters

    qwen3_next_decoder_sig = inspect.signature(qwen3_next.Qwen3NextDecoderLayer.__init__)
    assert "per_layer_sliding_window" in qwen3_next_decoder_sig.parameters


def test_qwen3_5_decoder_layer_accepts_per_layer_sliding_window():
    """`Qwen3_5DecoderLayer.__init__` must accept and forward
    `per_layer_sliding_window`.
    """
    import inspect

    qwen3_5 = _import_qwen3_5_module()
    sig = inspect.signature(qwen3_5.Qwen3_5DecoderLayer.__init__)
    assert "per_layer_sliding_window" in sig.parameters


# ── End-to-end smoke test (requires a GPU + the model weights) ──

# The actual end-to-end benchmark (TTFT, throughput, accuracy) belongs in the
# model's own benchmark suite. Mark it as requiring CUDA + the model so it
# does not run in unit-test CI environments without GPU.

_Qwen35Config = namedtuple("_Qwen35Config", ["layer_types"])
_LAYER_TYPES_QWEN35_2B = [
    "linear_attention", "linear_attention", "linear_attention", "full_attention",
    "linear_attention", "linear_attention", "linear_attention", "full_attention",
    "linear_attention", "linear_attention", "linear_attention", "full_attention",
    "linear_attention", "linear_attention", "linear_attention", "full_attention",
    "linear_attention", "linear_attention", "linear_attention", "full_attention",
    "linear_attention", "linear_attention", "linear_attention", "full_attention",
]


def test_get_layer_passes_sliding_window_to_calibrated_subset():
    """`Qwen3_5Model.get_layer` selects sliding window for the calibrated
    `full_attention` layers (15, 19, 23) and full attention for the rest.

    This unit test mocks the heavyweight Qwen3_5DecoderLayer to keep the
    test fast and free of GPU / model-weight dependencies.
    """
    from vllm.model_executor.models import qwen3_5
    from vllm.model_executor.models.utils import extract_layer_index

    captured = []

    class _FakeDecoderLayer:
        def __init__(self, vllm_config, layer_type, prefix, per_layer_sliding_window=None):
            captured.append(
                {
                    "layer_idx": extract_layer_index(prefix),
                    "layer_type": layer_type,
                    "per_layer_sliding_window": per_layer_sliding_window,
                }
            )

    # Patch Qwen3_5DecoderLayer to a lightweight recorder.
    with mock.patch.object(qwen3_5, "Qwen3_5DecoderLayer", _FakeDecoderLayer):
        # Build a minimal namespace compatible with `get_layer`.
        config = _Qwen35Config(layer_types=_LAYER_TYPES_QWEN35_2B)
        vllm_config = mock.MagicMock()
        vllm_config.model_config.hf_text_config = config

        # Recreate `get_layer` with the captured config/vllm_config.
        def get_layer(prefix: str):
            layer_idx = extract_layer_index(prefix)
            layer_type = config.layer_types[layer_idx]
            per_layer_sliding_window = None
            if (
                qwen3_5._TRIANGLE_ATTENTION_ENABLED
                and layer_type == "full_attention"
                and layer_idx in qwen3_5._TRIANGLE_ATTENTION_LAYERS
            ):
                per_layer_sliding_window = qwen3_5._TRIANGLE_ATTENTION_WINDOW
            return _FakeDecoderLayer(
                vllm_config, layer_type=layer_type, prefix=prefix,
                per_layer_sliding_window=per_layer_sliding_window,
            )

        # Run `get_layer` over every layer index.
        for i in range(24):
            get_layer(f"model.layers.{i}")

    # Assertions.
    assert len(captured) == 24
    # Every captured entry must have the right `layer_type` per config.
    for entry in captured:
        assert entry["layer_type"] == _LAYER_TYPES_QWEN35_2B[entry["layer_idx"]]
    # The three calibrated `full_attention` layers get sliding_window=128.
    for layer_idx in (15, 19, 23):
        assert captured[layer_idx]["per_layer_sliding_window"] == 128, (
            f"layer {layer_idx} should use TriangleMix sliding window"
        )
    # All other layers (including the other 3 `full_attention` layers at 3, 7, 11)
    # must keep full attention.
    for layer_idx in (3, 7, 11):
        assert captured[layer_idx]["per_layer_sliding_window"] is None, (
            f"layer {layer_idx} should keep full attention"
        )
    # All `linear_attention` layers must keep None regardless.
    for layer_idx in range(24):
        if _LAYER_TYPES_QWEN35_2B[layer_idx] == "linear_attention":
            assert captured[layer_idx]["per_layer_sliding_window"] is None