# SPDX-License-Identifier: Apache-2.0
"""TDD for PN62 — text-only ViT scratch skip on qwen3_vl.

Tests the wrapper logic in isolation. Live cross-rig validation pending
apnar's NVFP4 RTX 5090 setup with qwen3_vl checkpoint.
"""
from __future__ import annotations

import pytest

from vllm._genesis.wiring.memory.patch_N62_text_only_vit_skip import (
    _is_text_only_mode,
    _wrap_dummy_run,
)


# ─── _is_text_only_mode ────────────────────────────────────────────────


class _FakeRunner:
    def __init__(self, lmo=False, mm_limits=None):
        self.vllm_config = type("C", (), {})()
        self.vllm_config.language_model_only = lmo
        self.vllm_config.limit_mm_per_prompt = mm_limits


class TestTextOnlyModeDetection:
    def test_no_lmo_returns_false(self):
        r = _FakeRunner(lmo=False)
        assert _is_text_only_mode(r) is False

    def test_lmo_with_zero_limits_returns_true(self):
        r = _FakeRunner(lmo=True, mm_limits={"image": 0, "video": 0})
        assert _is_text_only_mode(r) is True

    def test_lmo_with_no_limits_returns_true(self):
        """Empty mm_limits dict + lmo → considered text-only."""
        r = _FakeRunner(lmo=True, mm_limits={})
        assert _is_text_only_mode(r) is True

    def test_lmo_with_nonzero_limits_returns_false(self):
        r = _FakeRunner(lmo=True, mm_limits={"image": 1})
        assert _is_text_only_mode(r) is False

    def test_no_config_returns_false(self):
        r = type("R", (), {})()
        assert _is_text_only_mode(r) is False


# ─── _wrap_dummy_run ────────────────────────────────────────────────────


class TestDummyRunWrapper:
    def test_text_only_sets_skip_marker_during_call(self):
        captured = {}

        def fake_dummy_run(self, *args, **kwargs):
            captured["marker_during_call"] = getattr(
                self, "_pn62_skip_vit_scratch", False
            )

        runner = _FakeRunner(lmo=True, mm_limits={"image": 0})
        wrapped = _wrap_dummy_run(fake_dummy_run)
        wrapped(runner)
        assert captured["marker_during_call"] is True
        # Marker is cleaned up after call
        assert not hasattr(runner, "_pn62_skip_vit_scratch")

    def test_non_text_only_does_not_set_marker(self):
        captured = {}

        def fake_dummy_run(self, *args, **kwargs):
            captured["marker_during_call"] = getattr(
                self, "_pn62_skip_vit_scratch", False
            )

        runner = _FakeRunner(lmo=False)
        wrapped = _wrap_dummy_run(fake_dummy_run)
        wrapped(runner)
        assert captured["marker_during_call"] is False

    def test_propagates_inner_exception(self):
        def fake_dummy_run(self, *args, **kwargs):
            raise RuntimeError("inner failure")

        runner = _FakeRunner(lmo=True, mm_limits={})
        wrapped = _wrap_dummy_run(fake_dummy_run)
        with pytest.raises(RuntimeError, match="inner failure"):
            wrapped(runner)
        # Marker cleanup happens even on exception
        assert not hasattr(runner, "_pn62_skip_vit_scratch")

    def test_idempotency_marker_attached(self):
        wrapped = _wrap_dummy_run(lambda self: None)
        assert getattr(wrapped, "__pn62_wrapped__", False) is True


# ─── apply() integration ────────────────────────────────────────────────


class TestApplyFunction:
    def test_apply_skipped_when_env_disabled(self, monkeypatch):
        from vllm._genesis.wiring.memory import patch_N62_text_only_vit_skip as p
        monkeypatch.delenv("GENESIS_ENABLE_PN62", raising=False)
        status, reason = p.apply()
        assert status == "skipped"

    def test_apply_skipped_when_runner_module_absent(self, monkeypatch):
        from vllm._genesis.wiring.memory import patch_N62_text_only_vit_skip as p
        monkeypatch.setenv("GENESIS_ENABLE_PN62", "1")
        import sys
        monkeypatch.setitem(sys.modules,
                            "vllm.v1.worker.gpu_model_runner", None)
        status, reason = p.apply()
        assert status == "skipped"
        assert "not importable" in reason or "GPUModelRunner" in reason
