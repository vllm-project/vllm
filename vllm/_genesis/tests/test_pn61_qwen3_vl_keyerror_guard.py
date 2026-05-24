# SPDX-License-Identifier: Apache-2.0
"""TDD for PN61 — qwen3_vl loader KeyError → text-only auto-fallback.

Tests the wrapper logic in isolation (without requiring a real qwen3_vl
checkpoint or vllm install). Live cross-rig validation pending apnar's
NVFP4 RTX 5090 setup.
"""
from __future__ import annotations

import pytest

from vllm._genesis.wiring.loader.patch_N61_qwen3_vl_keyerror_guard import (
    _is_vit_keyerror,
    _wrap_load_weights,
)


# ─── ViT key recognition ────────────────────────────────────────────────


class TestVitKeyRecognition:
    def test_vit_blocks_key_recognized(self):
        assert _is_vit_keyerror(KeyError("blocks.0.attn.proj.weight"))

    def test_vision_tower_key_recognized(self):
        assert _is_vit_keyerror(KeyError("vision_tower.embed.weight"))

    def test_visual_prefix_recognized(self):
        assert _is_vit_keyerror(KeyError("visual.encoder.layer.0.attn.q.weight"))

    def test_vit_prefix_recognized(self):
        assert _is_vit_keyerror(KeyError("vit.layer.5.mlp.fc2.weight"))

    def test_language_model_key_NOT_treated_as_vit(self):
        assert not _is_vit_keyerror(
            KeyError("model.layers.0.self_attn.q_proj.weight")
        )

    def test_empty_keyerror_returns_false(self):
        assert not _is_vit_keyerror(KeyError())


# ─── Wrapper behavior ───────────────────────────────────────────────────


class _FakeConfig:
    def __init__(self):
        self.language_model_only = False


class _FakeModel:
    def __init__(self, raise_key=None):
        self.config = _FakeConfig()
        self._raise_key = raise_key

    def real_load_weights(self, weights):
        if self._raise_key is not None:
            raise KeyError(self._raise_key)
        return 42  # number of loaded params


class TestWrapperBehavior:
    def test_no_exception_passes_through(self):
        m = _FakeModel(raise_key=None)
        wrapped = _wrap_load_weights(m.real_load_weights.__func__)
        result = wrapped(m, weights=None)
        assert result == 42
        assert m.config.language_model_only is False  # untouched

    def test_vit_keyerror_caught_and_warns(self, caplog):
        import logging
        m = _FakeModel(raise_key="blocks.0.attn.proj.weight")
        wrapped = _wrap_load_weights(m.real_load_weights.__func__)
        with caplog.at_level(logging.WARNING,
                             logger="genesis.wiring.pn61_qwen3_vl_keyerror_guard"):
            result = wrapped(m, weights=None)
        assert result == 0  # zero loaded params for absent ViT
        assert m.config.language_model_only is True  # auto-set
        assert any("ViT KeyError" in r.message for r in caplog.records)

    def test_non_vit_keyerror_propagates(self):
        m = _FakeModel(raise_key="model.layers.0.self_attn.q_proj.weight")
        wrapped = _wrap_load_weights(m.real_load_weights.__func__)
        with pytest.raises(KeyError):
            wrapped(m, weights=None)

    def test_idempotency_marker_attached(self):
        wrapped = _wrap_load_weights(lambda self, weights: None)
        assert getattr(wrapped, "__pn61_wrapped__", False) is True
        assert getattr(wrapped, "__wrapped__", None) is not None


# ─── apply() integration ────────────────────────────────────────────────


class TestApplyFunction:
    def test_apply_skipped_when_env_disabled(self, monkeypatch):
        from vllm._genesis.wiring.loader import patch_N61_qwen3_vl_keyerror_guard as p
        monkeypatch.delenv("GENESIS_ENABLE_PN61", raising=False)
        status, reason = p.apply()
        # No env flag → opt-in default-OFF → skipped
        assert status == "skipped"

    def test_apply_skipped_when_qwen3_vl_module_absent(self, monkeypatch):
        from vllm._genesis.wiring.loader import patch_N61_qwen3_vl_keyerror_guard as p
        monkeypatch.setenv("GENESIS_ENABLE_PN61", "1")
        # Force ImportError by monkeypatching sys.modules
        import sys
        monkeypatch.setitem(sys.modules,
                            "vllm.model_executor.models.qwen3_vl", None)
        status, reason = p.apply()
        assert status == "skipped"
        assert "not importable" in reason or "qwen3_vl" in reason
