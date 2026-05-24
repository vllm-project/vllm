# SPDX-License-Identifier: Apache-2.0
"""TDD tests for PN34 — WorkspaceManager runtime lock relaxation.

Companion to PN33 — same bug class (workspace under-counted) but on
the runtime decode path. Direct port of noonghunna's club-3090 setup-
time sidecar patch_workspace_lock_disable.py promoted into Genesis.

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
Port credit: noonghunna club-3090 (commit 2b5ab4d).
"""
from __future__ import annotations



def test_pn34_wiring_imports():
    from vllm._genesis.wiring.perf_hotfix import (
        patch_N34_workspace_lock_runtime_relax as mod,
    )
    assert hasattr(mod, "apply")
    assert hasattr(mod, "GENESIS_PN34_MARKER")
    assert hasattr(mod, "PN34_ANCHOR")
    assert hasattr(mod, "PN34_REPLACEMENT")


def test_pn34_dispatcher_registry():
    from vllm._genesis.dispatcher import PATCH_REGISTRY
    assert "PN34" in PATCH_REGISTRY
    e = PATCH_REGISTRY["PN34"]
    assert e["env_flag"] == "GENESIS_ENABLE_PN34_WORKSPACE_LOCK_RELAX"
    assert e["default_on"] is False  # opt-in (relaxes strict assertion)
    assert e["requires_patches"] == ["PN33"]


def test_pn34_anchor_targets_strict_assertion():
    from vllm._genesis.wiring.perf_hotfix.patch_N34_workspace_lock_runtime_relax import (
        PN34_ANCHOR,
    )
    assert "if self._locked:" in PN34_ANCHOR
    assert "raise AssertionError" in PN34_ANCHOR
    assert "Workspace is locked" in PN34_ANCHOR
    assert "Workspace growth is not allowed after locking" in PN34_ANCHOR


def test_pn34_replacement_relaxes_to_warn_plus_grow():
    from vllm._genesis.wiring.perf_hotfix.patch_N34_workspace_lock_runtime_relax import (
        PN34_REPLACEMENT,
    )
    # Relaxation: still detects locked state but warns instead of asserts
    assert "if self._locked:" in PN34_REPLACEMENT
    # No AssertionError raise in replacement (regression guard —
    # would re-introduce the strict failure mode)
    assert "raise AssertionError" not in PN34_REPLACEMENT
    # One-shot warning at WARNING level (visible in default log levels)
    assert "logger.warning" in PN34_REPLACEMENT
    # Module-level flag prevents log spam
    assert "_GENESIS_PN34_WORKSPACE_LOCK_WARNED" in PN34_REPLACEMENT
    assert "global _GENESIS_PN34_WORKSPACE_LOCK_WARNED" in PN34_REPLACEMENT
    # Defensive NameError guard for first call
    assert "NameError" in PN34_REPLACEMENT


def test_pn34_skips_when_env_off(monkeypatch):
    monkeypatch.delenv(
        "GENESIS_ENABLE_PN34_WORKSPACE_LOCK_RELAX", raising=False
    )
    from vllm._genesis.wiring.perf_hotfix.patch_N34_workspace_lock_runtime_relax import (
        apply,
    )
    status, reason = apply()
    assert status == "skipped"
    assert "opt-in" in reason.lower()


def test_pn34_register_in_apply_all():
    from vllm._genesis.patches.apply_all import (
        PATCH_REGISTRY as APPLY_REGISTRY,
    )
    names = [name for name, _ in APPLY_REGISTRY]
    pn34 = [n for n in names if "PN34" in n]
    assert len(pn34) == 1, (
        f"PN34 not registered in apply_all, names: {names[:5]}"
    )


def test_pn34_marker_unique():
    from vllm._genesis.wiring.perf_hotfix.patch_N34_workspace_lock_runtime_relax import (
        GENESIS_PN34_MARKER,
    )
    assert "PN34" in GENESIS_PN34_MARKER
    assert "v7.68" in GENESIS_PN34_MARKER


def test_pn34_documents_pn33_companion_relationship():
    """Source must document that PN34 is companion to PN33 — both
    address the same root cause (workspace under-counted) but at
    different layers (boot vs runtime decode)."""
    import inspect
    from vllm._genesis.wiring.perf_hotfix import (
        patch_N34_workspace_lock_runtime_relax as mod,
    )
    src = inspect.getsource(mod)
    assert "PN33" in src
    assert "companion" in src.lower()
    assert "boot" in src.lower() or "_dummy_sampler_run" in src
    assert "runtime decode" in src.lower() or "_decode_attention" in src


def test_pn34_documents_noonghunna_credit():
    import inspect
    from vllm._genesis.wiring.perf_hotfix import (
        patch_N34_workspace_lock_runtime_relax as mod,
    )
    src = inspect.getsource(mod)
    assert "noonghunna" in src
    assert "club-3090" in src
    assert "patch_workspace_lock_disable" in src


def test_pn34_documents_upstream_retirement_path():
    """Source must reference the upstream PR that obsoletes PN34."""
    import inspect
    from vllm._genesis.wiring.perf_hotfix import (
        patch_N34_workspace_lock_runtime_relax as mod,
    )
    src = inspect.getsource(mod)
    assert "40706" in src  # vllm#40706 = TQ scratch dedup
