# SPDX-License-Identifier: Apache-2.0
"""TDD tests for PN33 — spec-decode warmup K-aware sizing.

Backport of vllm-project/vllm#37521 extended to ALL spec-decode methods
(EAGLE + MTP + ngram + draft-model). Closes ampersandru mid-stream OOM
+ noonghunna workspace-lock AssertionError. Both share root cause: the
warmup path uses dummy K=1 instead of real num_speculative_tokens.

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
"""
from __future__ import annotations



def test_pn33_wiring_imports():
    from vllm._genesis.wiring.spec_decode import patch_N33_spec_decode_warmup_k as mod
    assert hasattr(mod, "apply")
    assert hasattr(mod, "GENESIS_PN33_MARKER")
    assert hasattr(mod, "PN33_ANCHOR")
    assert hasattr(mod, "PN33_REPLACEMENT")


def test_pn33_dispatcher_registry():
    from vllm._genesis.dispatcher import PATCH_REGISTRY
    assert "PN33" in PATCH_REGISTRY
    e = PATCH_REGISTRY["PN33"]
    assert e["env_flag"] == "GENESIS_ENABLE_PN33_SPEC_DECODE_WARMUP_K"
    # Default ON when spec-decode is active — this is a real correctness
    # fix, not experimental. Operators can disable via the DISABLE env.
    assert e["default_on"] is True
    assert e["category"] == "spec_decode"


def test_pn33_anchor_matches_canonical_upstream():
    from vllm._genesis.wiring.spec_decode.patch_N33_spec_decode_warmup_k import (
        PN33_ANCHOR,
    )
    # Anchor is the original 1-draft-token warmup line in
    # gpu_model_runner._dummy_sampler_run(). Canonical formatting.
    assert "if self.speculative_config:" in PN33_ANCHOR
    assert "draft_token_ids = [[0] for _ in range(num_reqs)]" in PN33_ANCHOR


def test_pn33_replacement_uses_num_speculative_tokens():
    from vllm._genesis.wiring.spec_decode.patch_N33_spec_decode_warmup_k import (
        PN33_REPLACEMENT,
    )
    # Reads num_speculative_tokens from speculative_config (defensive)
    assert "num_speculative_tokens" in PN33_REPLACEMENT
    assert "getattr(" in PN33_REPLACEMENT
    # Distinct token IDs (matches upstream PR's choice — avoids dedup
    # under-counting in some sampler paths)
    assert "list(range(_genesis_pn33_K))" in PN33_REPLACEMENT


def test_pn33_replacement_extends_beyond_eagle_only():
    """Genesis extends upstream #37521 beyond use_eagle() to cover
    MTP/ngram. Replacement must NOT gate on use_eagle()."""
    from vllm._genesis.wiring.spec_decode.patch_N33_spec_decode_warmup_k import (
        PN33_REPLACEMENT,
    )
    # Upstream PR gates on use_eagle(); Genesis covers all methods so
    # this gate must be absent
    assert "use_eagle()" not in PN33_REPLACEMENT, (
        "PN33 must NOT gate on use_eagle() — upstream's EAGLE-only fix "
        "doesn't cover ampersandru's MTP K=3 case. Genesis extends to "
        "all spec-decode methods uniformly."
    )


def test_pn33_replacement_falls_through_when_K_is_1():
    """When num_speculative_tokens <= 1, replacement preserves
    original [0] behavior (no regression for non-spec-decode or K=1)."""
    from vllm._genesis.wiring.spec_decode.patch_N33_spec_decode_warmup_k import (
        PN33_REPLACEMENT,
    )
    # The else branch falls through to [0] — verify presence
    assert "_genesis_pn33_dummy_tokens = [0]" in PN33_REPLACEMENT
    # Guard: K > 1 (not >= 1; K=1 is degenerate same as [0])
    assert "_genesis_pn33_K > 1" in PN33_REPLACEMENT


def test_pn33_replacement_honors_disable_env():
    """Operator can disable PN33 via env if K-sized warmup OOMs on a
    tight rig. Disable env reverts to original [0]."""
    from vllm._genesis.wiring.spec_decode.patch_N33_spec_decode_warmup_k import (
        PN33_REPLACEMENT,
    )
    assert "GENESIS_DISABLE_PN33_SPEC_DECODE_WARMUP_K" in PN33_REPLACEMENT
    # Honored values include common truthy strings
    assert "'1'" in PN33_REPLACEMENT
    assert "'true'" in PN33_REPLACEMENT


def test_pn33_skips_when_env_off(monkeypatch):
    monkeypatch.delenv(
        "GENESIS_ENABLE_PN33_SPEC_DECODE_WARMUP_K", raising=False
    )
    # Force-skip: dispatcher's should_apply is the gate. When the
    # ENABLE env is unset AND default_on=True, it still applies.
    # When the env is explicitly set to "0", it should skip.
    monkeypatch.setenv("GENESIS_ENABLE_PN33_SPEC_DECODE_WARMUP_K", "0")
    from vllm._genesis.wiring.spec_decode.patch_N33_spec_decode_warmup_k import apply
    status, _reason = apply()
    assert status == "skipped"


def test_pn33_register_in_apply_all():
    from vllm._genesis.patches.apply_all import (
        PATCH_REGISTRY as APPLY_REGISTRY,
    )
    names = [name for name, _ in APPLY_REGISTRY]
    # Match PN33 as standalone token (not substring) — PN34's title
    # references PN33 ("companion to PN33"), so plain `"PN33" in n`
    # would also match PN34's entry.
    pn33 = [n for n in names if n.startswith("PN33 ")]
    assert len(pn33) == 1, (
        f"PN33 not registered in apply_all, "
        f"matching entries: {pn33}, all names: {names[:10]}"
    )


def test_pn33_marker_unique():
    from vllm._genesis.wiring.spec_decode.patch_N33_spec_decode_warmup_k import (
        GENESIS_PN33_MARKER,
    )
    assert "PN33" in GENESIS_PN33_MARKER
    # Version pin so re-applies after kernel rewrites don't no-op
    # against a stale marker
    assert "v7.65" in GENESIS_PN33_MARKER


def test_pn33_documents_root_cause_coupling():
    """Source must document that PN33 closes BOTH ampersandru's
    mid-stream OOM AND noonghunna's workspace-lock bug — they share
    one root cause (warmup undercounted)."""
    import inspect
    from vllm._genesis.wiring.spec_decode import patch_N33_spec_decode_warmup_k as mod
    src = inspect.getsource(mod)
    assert "ampersandru" in src
    assert "noonghunna" in src
    assert "root cause" in src.lower()
    # Both bug references must be present
    assert "mid-stream" in src.lower() or "propose_draft_token_ids" in src
    assert "workspace-lock" in src.lower() or "WorkspaceManager" in src


def test_pn33_documents_upstream_relation():
    """Source must reference upstream PR #37521 + explain the Genesis
    extension (EAGLE-only → all spec-decode methods)."""
    import inspect
    from vllm._genesis.wiring.spec_decode import patch_N33_spec_decode_warmup_k as mod
    src = inspect.getsource(mod)
    assert "37521" in src
    assert "EAGLE" in src or "use_eagle" in src
    assert "MTP" in src and "ngram" in src
