# SPDX-License-Identifier: Apache-2.0
"""TDD tests for PN31 — FA varlen persistent out buffer (issue #15).

Author: Sandermage(Sander) Barzov Aleksandr, Ukraine, Odessa.
Bug: github.com/Sandermage/genesis-vllm-patches/issues/15
"""
from __future__ import annotations



def test_pn31_wiring_imports():
    from vllm._genesis.wiring.perf_hotfix import (
        patch_N31_fa_varlen_persistent_out as mod,
    )
    assert hasattr(mod, "apply")
    assert hasattr(mod, "GENESIS_PN31_MARKER")


def test_pn31_dispatcher_registry():
    from vllm._genesis.dispatcher import PATCH_REGISTRY
    assert "PN31" in PATCH_REGISTRY
    e = PATCH_REGISTRY["PN31"]
    assert e["env_flag"] == "GENESIS_ENABLE_PN31_FA_VARLEN_PERSISTENT_OUT"
    assert e["default_on"] is False


def test_pn31_skips_when_env_off(monkeypatch):
    monkeypatch.delenv(
        "GENESIS_ENABLE_PN31_FA_VARLEN_PERSISTENT_OUT", raising=False
    )
    from vllm._genesis.wiring.perf_hotfix.patch_N31_fa_varlen_persistent_out import apply
    status, reason = apply()
    assert status == "skipped"
    assert "opt-in" in reason.lower()


def test_pn31_anchor_targets_flash_attn_varlen_func_calls():
    from vllm._genesis.wiring.perf_hotfix.patch_N31_fa_varlen_persistent_out import (
        PN31_ANCHOR_NO_VER, PN31_ANCHOR_WITH_VER,
        PN31_REPLACEMENT_NO_VER, PN31_REPLACEMENT_WITH_VER,
    )
    # Both anchors must target flash_attn_varlen_func calls
    assert "flash_attn_varlen_func" in PN31_ANCHOR_NO_VER
    assert "flash_attn_varlen_func" in PN31_ANCHOR_WITH_VER
    # Anchors target different call sites — only "no_ver" anchor matches
    # the if-branch where fa_version IS None (so call doesn't pass fa_version
    # as kwarg). The if condition itself contains "fa_version" but the
    # call inside the branch does not.
    assert "fa_version=self.fa_version" not in PN31_ANCHOR_NO_VER
    assert "fa_version=self.fa_version" in PN31_ANCHOR_WITH_VER
    # Both replacements pass out= parameter
    assert "out=_genesis_pn31_out" in PN31_REPLACEMENT_NO_VER
    assert "out=_genesis_pn31_out" in PN31_REPLACEMENT_WITH_VER


def test_pn31_replacement_uses_buffer_acquire_pattern():
    from vllm._genesis.wiring.perf_hotfix.patch_N31_fa_varlen_persistent_out import (
        PN31_REPLACEMENT_NO_VER,
    )
    # Buffer acquire keyed by shape
    assert "_genesis_pn31_buf_key" in PN31_REPLACEMENT_NO_VER
    assert "_genesis_pn31_out_bufs" in PN31_REPLACEMENT_NO_VER
    # Reallocate-on-grow logic
    assert "is None" in PN31_REPLACEMENT_NO_VER or ".dtype" in PN31_REPLACEMENT_NO_VER
    # Drift marker
    assert "Genesis PN31" in PN31_REPLACEMENT_NO_VER
    assert "issue #15" in PN31_REPLACEMENT_NO_VER


def test_pn31_register_in_apply_all():
    from vllm._genesis.patches.apply_all import (
        PATCH_REGISTRY as APPLY_REGISTRY,
    )
    names = [name for name, _ in APPLY_REGISTRY]
    pn31 = [n for n in names if "PN31" in n]
    assert len(pn31) == 1, f"PN31 not registered, names: {names[:5]}"


def test_pn31_marker_unique():
    from vllm._genesis.wiring.perf_hotfix.patch_N31_fa_varlen_persistent_out import (
        GENESIS_PN31_MARKER,
    )
    assert "PN31" in GENESIS_PN31_MARKER
    assert "issue #15" in GENESIS_PN31_MARKER


def test_pn31_documents_sister_relationship_to_p38():
    """PN31 docs note this is sister patch to P38."""
    import inspect
    from vllm._genesis.wiring.perf_hotfix import (
        patch_N31_fa_varlen_persistent_out as mod,
    )
    src = inspect.getsource(mod)
    assert "P38" in src
    assert "sister" in src.lower() or "analogous" in src.lower()


def test_pn31_buffer_keyed_by_shape():
    """Buffer dict keyed by (total_q, n_heads, head_dim) tuple."""
    from vllm._genesis.wiring.perf_hotfix.patch_N31_fa_varlen_persistent_out import (
        PN31_REPLACEMENT_NO_VER,
    )
    # Buffer key contains the 3 shape dims
    assert "total_q" in PN31_REPLACEMENT_NO_VER
    assert "n_heads" in PN31_REPLACEMENT_NO_VER
    assert "head_dim" in PN31_REPLACEMENT_NO_VER
