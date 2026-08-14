# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression test for FP8 KV-cache scale preservation across sleep/wake.

Background
----------
``GPUModelRunner.init_fp8_kv_scales`` is invoked on every wake from sleep
(``post_kv_cache_wake_up``). The KV-cache memory pool is re-allocated on wake,
which zeroes the device-resident ``_k_scale`` / ``_v_scale`` tensors. The
method must restore those tensors so the cache doesn't read as all-zeros.

The bug this guards against: the method used to *unconditionally* fill the
scales with ``1.0``. For models whose checkpoint carries calibrated per-tensor
FP8-KV scales (e.g. produced by llm-compressor), that discarded the calibrated
values on every wake — silently degrading accuracy after the first sleep/wake
cycle. The calibrated values live on the host in ``_k_scale_float`` /
``_v_scale_float`` (set once at weight-load time, never freed by the sleep
allocator), so the wake path must restore from them.

These tests exercise the real ``GPUModelRunner.init_fp8_kv_scales`` logic
against a lightweight stub ``self`` — no GPU, no full runner construction.
"""

from types import SimpleNamespace

import torch

import vllm.v1.worker.gpu_model_runner as gpu_model_runner_module
from vllm.v1.worker.gpu_model_runner import GPUModelRunner


class _StubAttention(torch.nn.Module):
    """Minimal stand-in for an Attention layer carrying FP8-KV scales.

    Mirrors the real layer's relevant surface: device-resident ``_k_scale`` /
    ``_v_scale`` tensors plus the host-side ``_k_scale_float`` /
    ``_v_scale_float`` floats that survive sleep/wake.
    """

    def __init__(self, k_calib: float, v_calib: float):
        super().__init__()
        # Host-side calibrated values, set at weight-load time. These persist
        # across sleep/wake (plain Python floats, not freed by the allocator).
        self._k_scale_float = k_calib
        self._v_scale_float = v_calib
        # Device tensors. After wake_up the re-allocated pool leaves these at
        # 0.0; simulate that post-wake state.
        self._k_scale = torch.zeros(1, dtype=torch.float32)
        self._v_scale = torch.zeros(1, dtype=torch.float32)


def _make_runner_stub(layers: dict, cache_dtype: str = "fp8"):
    """Build a stub object carrying only what init_fp8_kv_scales reads."""
    return SimpleNamespace(
        cache_config=SimpleNamespace(cache_dtype=cache_dtype),
        kv_caches=[],  # no real cache tensors needed for the scale path
        compilation_config=SimpleNamespace(static_forward_context=layers),
    )


def _run_wake(runner_stub):
    """Invoke the real production wake-init logic on the stub."""
    GPUModelRunner.init_fp8_kv_scales(runner_stub)


def test_calibrated_scales_preserved_across_wake(monkeypatch):
    """A model WITH calibrated scales != 1.0 must keep them after wake.

    Pre-fix this FAILS: scales come back as 1.0 (calibration discarded).
    Post-fix this PASSES: scales restored to the calibrated host floats.
    """
    # Make our stub pass the real isinstance(module, (Attention, MLAAttention))
    # check inside init_fp8_kv_scales.
    monkeypatch.setattr(gpu_model_runner_module, "Attention", _StubAttention)
    monkeypatch.setattr(gpu_model_runner_module, "MLAAttention", _StubAttention)

    k_calib, v_calib = 0.0625, 0.125
    layer = _StubAttention(k_calib=k_calib, v_calib=v_calib)
    runner = _make_runner_stub({"layer.0.attn": layer})

    # Sanity: post-wake device tensors are zeroed (the dangerous state).
    assert layer._k_scale.item() == 0.0
    assert layer._v_scale.item() == 0.0

    _run_wake(runner)

    # The calibrated scales must be restored onto the device tensors — NOT 1.0.
    assert layer._k_scale.item() == k_calib, (
        f"calibrated k_scale clobbered: got {layer._k_scale.item()}, "
        f"expected {k_calib}"
    )
    assert layer._v_scale.item() == v_calib, (
        f"calibrated v_scale clobbered: got {layer._v_scale.item()}, "
        f"expected {v_calib}"
    )


def test_uncalibrated_scales_default_to_one(monkeypatch):
    """A model WITHOUT calibration (host float == 1.0) still wakes to 1.0.

    Guards backward compatibility: the on-the-fly fp8 quant path is unchanged.
    """
    monkeypatch.setattr(gpu_model_runner_module, "Attention", _StubAttention)
    monkeypatch.setattr(gpu_model_runner_module, "MLAAttention", _StubAttention)

    layer = _StubAttention(k_calib=1.0, v_calib=1.0)
    runner = _make_runner_stub({"layer.0.attn": layer})

    _run_wake(runner)

    assert layer._k_scale.item() == 1.0
    assert layer._v_scale.item() == 1.0


def test_non_quantized_cache_is_noop(monkeypatch):
    """When the KV cache isn't quantized, the scale path is skipped entirely."""
    monkeypatch.setattr(gpu_model_runner_module, "Attention", _StubAttention)
    monkeypatch.setattr(gpu_model_runner_module, "MLAAttention", _StubAttention)

    layer = _StubAttention(k_calib=0.5, v_calib=0.5)
    runner = _make_runner_stub({"layer.0.attn": layer}, cache_dtype="auto")

    _run_wake(runner)

    # Untouched: still the zeroed post-wake state, because we returned early.
    assert layer._k_scale.item() == 0.0
    assert layer._v_scale.item() == 0.0
