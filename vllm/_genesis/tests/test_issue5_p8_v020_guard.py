# SPDX-License-Identifier: Apache-2.0
"""Regression tests for Genesis Issue #5 — P8 ImportError on vLLM v0.20.0.

Issue #5 (noonghunna 2026-04-29): Genesis v7.54+ on `vllm/vllm-openai:v0.20.0`
crashes at engine boot with:

    File ".../vllm/v1/core/sched/scheduler.py", line 43
      from vllm.v1.core.kv_cache_utils import token_capacity_kv_cache_groups  # [Genesis P8]
    ImportError: cannot import name 'token_capacity_kv_cache_groups'

P8 reports clean (30 applied / 33 skipped / 0 failed) but the helper
symbol it injects into `kv_cache_utils.py` doesn't end up at module
scope under v0.20.0's reorganized layout. The scheduler.py sub-patch
then injects an `import` line that crashes the engine.

Fix: P8.apply() now does a post-apply import probe. If the helper is
not importable from the patched `kv_cache_utils.py`, scheduler.py is
NOT patched and the engine boots with the vanilla scheduler. The
operator gets a clear log message + skip-reason.
"""
from __future__ import annotations

import sys
import types



class TestIssue5PostApplyImportGuard:
    """The new guard must trigger when kv_cache_utils.py reports
    APPLIED but the helper symbol is not actually exposed."""

    def test_guard_skips_when_helper_not_importable(self, monkeypatch):
        """Simulate the v0.20.0 case: kv_cache_utils.py is importable
        but does NOT expose token_capacity_kv_cache_groups at module
        scope. P8 must SKIP scheduler.py and return a clean skip
        status — not 'applied'."""
        from vllm._genesis.wiring.legacy import patch_8_kv_hybrid_reporting

        # Build a fake kv_cache_utils module without the helper
        fake_mod = types.ModuleType("vllm.v1.core.kv_cache_utils")
        # NOTE: deliberately omit `token_capacity_kv_cache_groups`
        monkeypatch.setitem(sys.modules, "vllm.v1.core.kv_cache_utils",
                            fake_mod)

        # Simulate "kv_cache_utils.py text-patch reports APPLIED" by
        # monkey-patching _patcher_kv to a stub.
        from vllm._genesis.wiring.text_patch import TextPatchResult

        def _fake_patcher_kv():
            class _Stub:
                def apply(self):
                    return TextPatchResult.APPLIED, None
            return _Stub()

        # Also stub _patcher_sched so we can detect whether it gets
        # called (it should NOT)
        sched_called = {"value": False}
        def _fake_patcher_sched():
            sched_called["value"] = True
            class _Stub:
                def apply(self):
                    return TextPatchResult.APPLIED, None
            return _Stub()

        # Bypass the vllm_install_root check
        monkeypatch.setattr(
            patch_8_kv_hybrid_reporting,
            "vllm_install_root",
            lambda: "/fake/vllm/install",
        )
        monkeypatch.setattr(
            patch_8_kv_hybrid_reporting, "_patcher_kv", _fake_patcher_kv
        )
        monkeypatch.setattr(
            patch_8_kv_hybrid_reporting, "_patcher_sched", _fake_patcher_sched
        )

        status, reason = patch_8_kv_hybrid_reporting.apply()

        assert status == "skipped", (
            f"Issue #5 guard must SKIP when helper not importable; "
            f"got {status} ({reason})"
        )
        assert "Issue #5" in reason or "v0.20.0" in reason or "ImportError" in reason
        assert not sched_called["value"], (
            "Issue #5 guard must NOT call _patcher_sched when "
            "kv_cache_utils helper is missing — that's the bug it "
            "exists to prevent."
        )

    def test_guard_proceeds_when_helper_importable(self, monkeypatch):
        """Sanity: when the helper IS importable (normal pre-v0.20.0
        case), P8 proceeds to scheduler.py as before."""
        from vllm._genesis.wiring.legacy import patch_8_kv_hybrid_reporting
        from vllm._genesis.wiring.text_patch import TextPatchResult

        # Fake kv_cache_utils WITH the helper symbol
        fake_mod = types.ModuleType("vllm.v1.core.kv_cache_utils")
        fake_mod.token_capacity_kv_cache_groups = lambda *a, **kw: []
        monkeypatch.setitem(sys.modules, "vllm.v1.core.kv_cache_utils",
                            fake_mod)

        sched_called = {"value": False}

        def _fake_patcher_kv():
            class _Stub:
                def apply(self):
                    return TextPatchResult.APPLIED, None
            return _Stub()

        def _fake_patcher_sched():
            sched_called["value"] = True
            class _Stub:
                def apply(self):
                    return TextPatchResult.APPLIED, None
            return _Stub()

        monkeypatch.setattr(
            patch_8_kv_hybrid_reporting,
            "vllm_install_root",
            lambda: "/fake/vllm/install",
        )
        monkeypatch.setattr(
            patch_8_kv_hybrid_reporting, "_patcher_kv", _fake_patcher_kv
        )
        monkeypatch.setattr(
            patch_8_kv_hybrid_reporting, "_patcher_sched", _fake_patcher_sched
        )

        status, reason = patch_8_kv_hybrid_reporting.apply()

        assert sched_called["value"], (
            "When helper IS importable, scheduler.py P8 should still "
            "be applied"
        )
        assert status == "applied", (
            f"Normal case must report applied; got {status} ({reason})"
        )

    def test_guard_skips_when_module_import_raises(self, monkeypatch):
        """If `import vllm.v1.core.kv_cache_utils` itself raises (e.g.
        ANY exception during import — even before our injection),
        we MUST refuse to proceed."""
        from vllm._genesis.wiring.legacy import patch_8_kv_hybrid_reporting
        from vllm._genesis.wiring.text_patch import TextPatchResult

        # Make import_module raise
        import importlib
        original = importlib.import_module

        def _raising(name, *a, **kw):
            if name == "vllm.v1.core.kv_cache_utils":
                raise RuntimeError("simulated import failure")
            return original(name, *a, **kw)

        monkeypatch.setattr(importlib, "import_module", _raising)

        sched_called = {"value": False}

        def _fake_patcher_kv():
            class _Stub:
                def apply(self):
                    return TextPatchResult.APPLIED, None
            return _Stub()

        def _fake_patcher_sched():
            sched_called["value"] = True
            class _Stub:
                def apply(self):
                    return TextPatchResult.APPLIED, None
            return _Stub()

        monkeypatch.setattr(
            patch_8_kv_hybrid_reporting,
            "vllm_install_root",
            lambda: "/fake/vllm/install",
        )
        monkeypatch.setattr(
            patch_8_kv_hybrid_reporting, "_patcher_kv", _fake_patcher_kv
        )
        monkeypatch.setattr(
            patch_8_kv_hybrid_reporting, "_patcher_sched", _fake_patcher_sched
        )

        status, _reason = patch_8_kv_hybrid_reporting.apply()

        assert status == "skipped"
        assert not sched_called["value"], (
            "If kv_cache_utils import probe raises, scheduler.py P8 "
            "MUST NOT proceed."
        )
