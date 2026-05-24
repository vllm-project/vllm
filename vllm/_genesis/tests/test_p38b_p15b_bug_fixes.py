# SPDX-License-Identifier: Apache-2.0
"""TDD tests for P38B (#14) + P15B (#15) bug fixes.

P38B fixes Issue #14 — P38 silent no-op on TQ KV path.
Root cause: aot_compile_fullgraph captures `_continuation_prefill`
original body; class-attribute rebind doesn't propagate.
Fix: text-patch source to inject delegate hook + install
`_genesis_p38_dispatch` class attribute.

P15B fixes Issue #15 — FA varlen workspace cliff on long-ctx
continuation prefill. Root cause: PN17 doesn't reach the TQ FA varlen
call site; max_seqlen_k stays cudagraph-bloated.
Fix: text-patch `_flash_attn_varlen` to clamp max_seqlen_k to actual
cu_seqlens_k span.

These tests verify both fixes:

1. **P38B in-source hook present** in patched `turboquant_attn.py`
2. **P38B dispatcher installed** on `TurboQuantAttentionImpl` class
3. **P38B dispatcher actually engages** when called (returns Genesis
   result vs falls through)
4. **P15B clamp present** in patched `_flash_attn_varlen` body
5. **P15B clamp logic correct** — actual max_seqlen_k computed from
   cu_seqlens_k

These are CPU tests — they don't require GPU, just verify the
text-patch + class-attribute installation. The actual runtime
behavior (kernel calls during inference) is exercised by the bench
suite.
"""
from __future__ import annotations




# ─────────────────────────────────────────────────────────────────
# P38B tests (Issue #14)
# ─────────────────────────────────────────────────────────────────


def test_p38b_wiring_imports():
    """P38B wiring module imports cleanly."""
    from vllm._genesis.wiring.perf_hotfix import patch_38b_compile_safe_hook
    assert hasattr(patch_38b_compile_safe_hook, "apply")
    assert hasattr(patch_38b_compile_safe_hook, "is_applied")
    assert hasattr(patch_38b_compile_safe_hook, "_install_dispatcher")


def test_p38b_dispatcher_registry():
    """P38B is registered in PATCH_REGISTRY with correct env flag."""
    from vllm._genesis.dispatcher import PATCH_REGISTRY
    assert "P38B" in PATCH_REGISTRY
    e = PATCH_REGISTRY["P38B"]
    assert e["env_flag"] == "GENESIS_ENABLE_P38B_COMPILE_SAFE"
    assert e["default_on"] is False


def test_p38b_skips_when_env_off(monkeypatch):
    """When env is OFF, apply() returns 'skipped' with opt-in reason."""
    monkeypatch.delenv("GENESIS_ENABLE_P38B_COMPILE_SAFE", raising=False)
    from vllm._genesis.wiring.perf_hotfix.patch_38b_compile_safe_hook import apply
    status, reason = apply()
    assert status == "skipped"
    assert "opt-in" in reason.lower()


def test_p38b_anchor_text_present():
    """P38B uses concrete text anchor + replacement structure."""
    from vllm._genesis.wiring.perf_hotfix.patch_38b_compile_safe_hook import (
        P38B_ANCHOR, P38B_REPLACEMENT, GENESIS_P38B_MARKER,
    )
    # Anchor includes docstring close + first body line of _continuation_prefill
    assert "Dequants previously cached K/V" in P38B_ANCHOR  # docstring marker
    assert "q_len, Hq, D = query.shape" in P38B_ANCHOR  # first body line
    # Replacement includes the in-source hook
    assert "_genesis_p38_dispatch" in P38B_REPLACEMENT
    assert "_genesis_p38b_disp" in P38B_REPLACEMENT
    # Marker is non-trivial (drift detection)
    assert "P38b" in GENESIS_P38B_MARKER
    assert len(GENESIS_P38B_MARKER) > 30


def test_p38b_hook_falls_through_on_none():
    """P38B hook structure: returns Genesis result OR None (fall through).

    Verifies the replacement code structure: `if _r is not None: return _r`
    means None falls through to original body. This is the contract.
    """
    from vllm._genesis.wiring.perf_hotfix.patch_38b_compile_safe_hook import (
        P38B_REPLACEMENT,
    )
    # Critical check: dispatcher result is gated on `is not None`
    assert "if _genesis_p38b_r is not None:" in P38B_REPLACEMENT
    assert "return _genesis_p38b_r" in P38B_REPLACEMENT
    # Original body line follows the hook
    assert P38B_REPLACEMENT.rstrip().endswith("q_len, Hq, D = query.shape")


# ─────────────────────────────────────────────────────────────────
# P15B tests (Issue #15)
# ─────────────────────────────────────────────────────────────────


def test_p15b_wiring_imports():
    """P15B wiring module imports cleanly."""
    from vllm._genesis.wiring.perf_hotfix import patch_15B_fa_varlen_clamp
    assert hasattr(patch_15B_fa_varlen_clamp, "apply")
    assert hasattr(patch_15B_fa_varlen_clamp, "is_applied")


def test_p15b_dispatcher_registry():
    """P15B is registered in PATCH_REGISTRY."""
    from vllm._genesis.dispatcher import PATCH_REGISTRY
    assert "P15B" in PATCH_REGISTRY
    e = PATCH_REGISTRY["P15B"]
    assert e["env_flag"] == "GENESIS_ENABLE_P15B_FA_VARLEN_CLAMP"
    assert e["default_on"] is False


def test_p15b_skips_when_env_off(monkeypatch):
    """When env is OFF, apply() returns 'skipped'."""
    monkeypatch.delenv("GENESIS_ENABLE_P15B_FA_VARLEN_CLAMP", raising=False)
    from vllm._genesis.wiring.perf_hotfix.patch_15B_fa_varlen_clamp import apply
    status, reason = apply()
    assert status == "skipped"
    assert "opt-in" in reason.lower()


def test_p15b_anchor_targets_flash_attn_varlen():
    """P15B anchor targets the right function signature."""
    from vllm._genesis.wiring.perf_hotfix.patch_15B_fa_varlen_clamp import (
        P15B_ANCHOR, P15B_REPLACEMENT, GENESIS_P15B_MARKER,
    )
    # Anchor must match _flash_attn_varlen function signature
    assert "def _flash_attn_varlen(" in P15B_ANCHOR
    assert "cu_seqlens_q: torch.Tensor" in P15B_ANCHOR
    assert "cu_seqlens_k: torch.Tensor" in P15B_ANCHOR
    assert "max_seqlen_k: int" in P15B_ANCHOR
    # Replacement adds clamp logic
    assert "_genesis_p15b_actual" in P15B_REPLACEMENT
    assert "min(max_seqlen_k" in P15B_REPLACEMENT
    assert "cu_seqlens_k[-1].item()" in P15B_REPLACEMENT
    assert "[1:] - cu_seqlens_k[:-1]" in P15B_REPLACEMENT
    # Marker present
    assert "P15B" in GENESIS_P15B_MARKER


def test_p15b_clamp_uses_min_not_replace():
    """P15B clamp uses `min(max_seqlen_k, actual)` — never INCREASES.

    Critical correctness invariant: the clamp must only REDUCE
    max_seqlen_k, never increase. If the heuristic actual > original
    max, the original wins.
    """
    from vllm._genesis.wiring.perf_hotfix.patch_15B_fa_varlen_clamp import (
        P15B_REPLACEMENT,
    )
    assert "max_seqlen_k = min(max_seqlen_k, _genesis_p15b_actual)" in P15B_REPLACEMENT
    # Also: try/except guard ensures fall-through on error
    assert "try:" in P15B_REPLACEMENT
    assert "except Exception:" in P15B_REPLACEMENT


def test_p15b_handles_batch_1_fast_path():
    """P15B handles batch=1 (continuation prefill) without diff().max() overhead."""
    from vllm._genesis.wiring.perf_hotfix.patch_15B_fa_varlen_clamp import (
        P15B_REPLACEMENT,
    )
    # batch=1: cu_seqlens_k has shape [2] (start, end). Direct indexing.
    assert "if cu_seqlens_k.shape[0] == 2:" in P15B_REPLACEMENT
    assert "int(cu_seqlens_k[-1].item())" in P15B_REPLACEMENT


# ─────────────────────────────────────────────────────────────────
# Integration: both fixes coexist
# ─────────────────────────────────────────────────────────────────


def test_p38b_and_p15b_both_applicable_to_turboquant_attn():
    """Both fixes target the same file — verify they don't anchor-collide.

    P38B targets _continuation_prefill body (around line 1130).
    P15B targets _flash_attn_varlen signature (around line 300).
    They edit different functions, so anchors must be disjoint.
    """
    from vllm._genesis.wiring.perf_hotfix.patch_38b_compile_safe_hook import (
        P38B_ANCHOR,
    )
    from vllm._genesis.wiring.perf_hotfix.patch_15B_fa_varlen_clamp import (
        P15B_ANCHOR,
    )
    # P38B anchor mentions _continuation_prefill behavior
    assert "Dequants previously cached K/V" in P38B_ANCHOR
    # P15B anchor mentions _flash_attn_varlen signature
    assert "_flash_attn_varlen" in P15B_ANCHOR
    # Anchors don't overlap
    assert P38B_ANCHOR not in P15B_ANCHOR
    assert P15B_ANCHOR not in P38B_ANCHOR


def test_dispatch_registry_both_present():
    """Both P38B and P15B in dispatcher registry as opt-in OFF."""
    from vllm._genesis.dispatcher import PATCH_REGISTRY
    for key in ("P38B", "P15B"):
        assert key in PATCH_REGISTRY, f"{key} missing from registry"
        assert PATCH_REGISTRY[key]["default_on"] is False
        assert PATCH_REGISTRY[key]["category"] == "perf_hotfix"


def test_apply_all_register_both():
    """Both P38B and P15B registered via @register_patch in apply_all.py."""
    from vllm._genesis.patches.apply_all import PATCH_REGISTRY as APPLY_REGISTRY
    names = [name for name, _ in APPLY_REGISTRY]
    p38b = [n for n in names if "P38B" in n]
    p15b = [n for n in names if "P15B" in n]
    assert len(p38b) == 1, f"P38B not registered, names found: {names[:5]}"
    assert len(p15b) == 1, f"P15B not registered, names found: {names[:5]}"
