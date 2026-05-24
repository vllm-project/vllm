# SPDX-License-Identifier: Apache-2.0
"""Unit tests for P82 — SGLang threshold_single OR-clause acceptance.

Pure-Python text-patch generator. No GPU / vLLM dependency — exercises
threshold parsing, replacement-builder, and apply() decision tree.

Covers:
  - threshold parsing: empty → default 0.3, valid float, garbage → default,
    out-of-range → clamped to [0.0, 1.0]
  - replacement contains BOTH the vanilla rule AND the threshold OR-clause
  - replacement preserves the upstream NOTE comment (anchor uniqueness)
  - threshold is baked as a Python float literal (not a runtime env read)
  - apply() returns 'skipped' when env not set or threshold == 0.0
  - marker is versioned so re-applies after bump aren't no-ops
"""
from __future__ import annotations

import re

import pytest

from vllm._genesis.wiring.spec_decode.patch_82_sglang_acceptance_threshold import (
    GENESIS_P82_MARKER_PREFIX,
    P82_OLD,
    _build_replacement,
    _marker_for,
    _read_threshold,
    _DEFAULT_THRESHOLD,
    apply,
)


# ─── threshold parsing ──────────────────────────────────────────────────


def test_threshold_default_when_unset(monkeypatch):
    monkeypatch.delenv("GENESIS_P82_THRESHOLD_SINGLE", raising=False)
    assert _read_threshold() == _DEFAULT_THRESHOLD


def test_threshold_default_when_empty(monkeypatch):
    monkeypatch.setenv("GENESIS_P82_THRESHOLD_SINGLE", "")
    assert _read_threshold() == _DEFAULT_THRESHOLD


def test_threshold_default_when_garbage(monkeypatch):
    monkeypatch.setenv("GENESIS_P82_THRESHOLD_SINGLE", "not_a_float")
    assert _read_threshold() == _DEFAULT_THRESHOLD


def test_threshold_valid_float(monkeypatch):
    monkeypatch.setenv("GENESIS_P82_THRESHOLD_SINGLE", "0.42")
    assert _read_threshold() == pytest.approx(0.42)


def test_threshold_clamped_above_one(monkeypatch):
    monkeypatch.setenv("GENESIS_P82_THRESHOLD_SINGLE", "1.5")
    assert _read_threshold() == 1.0


def test_threshold_clamped_below_zero(monkeypatch):
    monkeypatch.setenv("GENESIS_P82_THRESHOLD_SINGLE", "-0.1")
    assert _read_threshold() == 0.0


def test_threshold_at_bounds(monkeypatch):
    monkeypatch.setenv("GENESIS_P82_THRESHOLD_SINGLE", "0.0")
    assert _read_threshold() == 0.0
    monkeypatch.setenv("GENESIS_P82_THRESHOLD_SINGLE", "1.0")
    assert _read_threshold() == 1.0


# ─── replacement builder ────────────────────────────────────────────────


def test_replacement_preserves_upstream_NOTE_comment():
    """Anchor uniqueness depends on keeping the NOTE(woosuk) comment."""
    rep = _build_replacement(0.3)
    assert "NOTE(woosuk)" in rep, (
        "replacement must preserve upstream NOTE for anchor uniqueness"
    )
    assert "draft probability should never be 0" in rep


def test_replacement_contains_vanilla_rule():
    """v2 (2026-04-30): the vanilla rule's draft_prob guard tightened
    from `draft_prob > 0` to `draft_prob >= 1e-20` for fp32 denormal
    safety. The ratio check (`target_prob / draft_prob >= uniform_prob`)
    is unchanged."""
    rep = _build_replacement(0.3)
    assert "target_prob / draft_prob >= uniform_prob" in rep, (
        "vanilla rejection-sample ratio check must be preserved in "
        "OR-clause"
    )
    # v2 numerical guard
    assert "draft_prob >= 1e-20" in rep, (
        "v2 fp32-denormal guard must be present (was 'draft_prob > 0' "
        "in v1; new guard prevents inf/NaN on tiny draft probs)"
    )


def test_replacement_contains_threshold_clause():
    rep = _build_replacement(0.42)
    # v2: threshold clause is a multi-line block with explicit
    # `target_prob > 0` defensive check before the threshold compare.
    assert "_genesis_p82_threshold = (" in rep, (
        "v2 threshold clause must be a multi-line tuple-style block"
    )
    assert "target_prob > 0 and target_prob >= 0.42" in rep, (
        "v2 threshold side: defensive `target_prob > 0` AND threshold-"
        "compare must both appear; threshold value 0.42 baked as literal"
    )


def test_replacement_uses_OR_combinator():
    rep = _build_replacement(0.3)
    # Final acceptance is `vanilla OR threshold`
    assert re.search(
        r"accepted\s*=\s*_genesis_p82_vanilla\s+or\s+_genesis_p82_threshold",
        rep,
    ), "final 'accepted' assignment must combine vanilla OR threshold"


def test_replacement_carries_genesis_breadcrumb():
    rep = _build_replacement(0.3)
    assert "[Genesis P82" in rep, (
        "replacement must include `[Genesis P82` breadcrumb for drift detection"
    )


def test_replacement_threshold_is_baked_not_env_read():
    """Threshold must be a literal, not an env read at runtime (perf)."""
    rep = _build_replacement(0.3)
    assert "os.environ" not in rep, (
        "replacement must NOT contain runtime env reads — threshold is baked at apply()"
    )


# ─── threshold values produce different literals ─────────────────────────


@pytest.mark.parametrize("threshold", [0.1, 0.25, 0.3, 0.5, 0.7])
def test_replacement_distinct_per_threshold(threshold):
    rep = _build_replacement(threshold)
    # The numeric repr should appear; we don't lock to Python's exact repr
    # form, just that the rounded value shows up.
    rounded = f"{threshold:.4f}".rstrip("0").rstrip(".")
    assert (
        repr(threshold) in rep or rounded in rep
    ), f"threshold {threshold!r} should appear as a literal in the replacement"


# ─── anchor invariants ──────────────────────────────────────────────────


def test_anchor_is_three_lines_with_NOTE():
    """The 3-line anchor including NOTE comment + assignment is needed for
    uniqueness in rejection_sampler.py.
    """
    lines = P82_OLD.split("\n")
    # 3 logical lines + trailing newline = 4 entries when split by \n
    assert "NOTE(woosuk)" in lines[0]
    assert "we check it to avoid NaNs" in lines[1]
    assert "draft_prob > 0 and target_prob / draft_prob >= uniform_prob" in lines[2]


def test_anchor_long_enough_for_uniqueness():
    """Heuristic: an anchor under 100 chars risks ambiguity."""
    assert len(P82_OLD) >= 100, (
        f"anchor too short ({len(P82_OLD)}) — risk of multi-match"
    )


# ─── marker invariants ──────────────────────────────────────────────────


def test_marker_prefix_versioned():
    """Prefix should embed a version token. v7.63.x v2 (2026-04-30)
    bumped from v7.62.11 → v7.63.x to match the v2 hardening
    (numerical guard + min_draft_pos + threshold==1.0 skip).
    """
    # Watch for ANY versioned token; specific expectation is current
    # release line.
    assert any(
        v in GENESIS_P82_MARKER_PREFIX
        for v in ("v7.62.11", "v7.63", "v7.64")
    ), (
        f"P82 marker prefix {GENESIS_P82_MARKER_PREFIX!r} should embed "
        "a v7.6x version token"
    )
    assert "Genesis P82" in GENESIS_P82_MARKER_PREFIX


# ─── B3 fix: marker encodes threshold ─────────────────────────────────


def test_marker_for_encodes_threshold():
    """The marker must include the threshold so a different bake forces
    re-apply (not silent IDEMPOTENT skip)."""
    m1 = _marker_for(0.30)
    m2 = _marker_for(0.50)
    assert m1 != m2, (
        "Markers for different thresholds must differ (B3 fix). "
        f"Got identical {m1!r} for both 0.30 and 0.50"
    )
    assert "thresh=0.3000" in m1
    assert "thresh=0.5000" in m2


def test_marker_for_stable_to_float_repr():
    """0.30000000000000004 should produce the same marker as 0.3 — round to
    4 decimals avoids spurious re-applies from float representation noise.
    """
    a = _marker_for(0.3)
    b = _marker_for(0.3 + 1e-16)
    assert a == b, (
        f"Marker should be float-repr stable (round to 4 decimals). "
        f"Got {a!r} vs {b!r}"
    )


def test_marker_for_starts_with_prefix():
    """Drift detection still works: the prefix is stable across thresholds."""
    for th in (0.1, 0.3, 0.5, 0.7, 0.999):
        m = _marker_for(th)
        assert m.startswith(GENESIS_P82_MARKER_PREFIX), (
            f"_marker_for({th}) = {m!r} must start with {GENESIS_P82_MARKER_PREFIX!r}"
        )


# ─── apply() decision tree ──────────────────────────────────────────────


def test_apply_skipped_when_disabled(monkeypatch):
    """Without GENESIS_ENABLE_P82=1, dispatcher should reject the apply."""
    monkeypatch.delenv("GENESIS_ENABLE_P82", raising=False)
    status, reason = apply()
    assert status == "skipped", (
        f"P82 should skip when GENESIS_ENABLE_P82 unset; got {status!r}: {reason}"
    )


def test_apply_skipped_when_threshold_zero(monkeypatch):
    """threshold=0.0 → OR-clause never fires → skip patch entirely (keep
    source vanilla). Avoids paying patch overhead for a no-op rule.

    Skipped on dev hosts without vllm installed (apply() short-circuits
    earlier on `vllm_install_root() is None`).
    """
    from vllm._genesis.guards import vllm_install_root
    if vllm_install_root() is None:
        pytest.skip("vllm not installed — apply() short-circuits before threshold check")
    monkeypatch.setenv("GENESIS_ENABLE_P82", "1")
    monkeypatch.setenv("GENESIS_P82_THRESHOLD_SINGLE", "0.0")
    status, reason = apply()
    assert status == "skipped", (
        f"P82 should skip when threshold==0.0; got {status!r}: {reason}"
    )
    assert "0.0" in reason or "OR clause would never fire" in reason


# ─── P82 v2 (2026-04-30) regression coverage ──────────────────────────────


class TestP82V2NumericalGuard:
    """v2: `draft_prob >= 1e-20` instead of `> 0` to prevent fp32
    denormal-zone overflow in `target_prob / draft_prob`."""

    def test_replacement_uses_fp32_safe_eps(self):
        rep = _build_replacement(0.3)
        # New guard
        assert "draft_prob >= 1e-20" in rep
        # Old guard (UNSAFE) must be GONE — it only checks > 0 which
        # admits denormals 1e-300 that overflow on division
        assert "draft_prob > 0 and target_prob" not in rep, (
            "v2 must NOT keep the old `draft_prob > 0` guard — that "
            "form admits fp32 denormals that produce inf/NaN when "
            "divided"
        )

    def test_eps_constant_is_fp32_safe(self):
        from vllm._genesis.wiring.spec_decode.patch_82_sglang_acceptance_threshold import (
            GENESIS_P82_DRAFT_PROB_EPS,
        )
        # Must be well within fp32 normal range (~1e-38 minimum)
        assert GENESIS_P82_DRAFT_PROB_EPS >= 1e-30
        # Must be small enough not to reject realistic softmax outputs
        # (typical low-confidence draft probs are 1e-6 to 1e-3)
        assert GENESIS_P82_DRAFT_PROB_EPS <= 1e-15


class TestP82V2DefensiveTargetProbCheck:
    """v2: explicit `target_prob > 0` check on threshold side guards
    against malformed softmax (impossible in practice but defensive)."""

    def test_threshold_clause_includes_target_prob_positive_check(self):
        rep = _build_replacement(0.3)
        # The threshold clause must include both:
        # - target_prob > 0 (defensive)
        # - target_prob >= threshold (the actual rule)
        assert "target_prob > 0 and target_prob >= 0.3" in rep, (
            "v2 threshold clause must combine defensive target_prob > 0 "
            "with the threshold compare"
        )


class TestP82V2MinDraftPosGuard:
    """v2: `GENESIS_P82_MIN_DRAFT_POS` opt-in env var restricts the
    OR-clause to draft positions >= N. Default 0 = current behavior."""

    def test_default_is_zero(self, monkeypatch):
        monkeypatch.delenv("GENESIS_P82_MIN_DRAFT_POS", raising=False)
        from vllm._genesis.wiring.spec_decode.patch_82_sglang_acceptance_threshold import (
            _read_min_draft_pos,
        )
        assert _read_min_draft_pos() == 0

    def test_invalid_falls_back_to_zero(self, monkeypatch):
        monkeypatch.setenv("GENESIS_P82_MIN_DRAFT_POS", "not-an-int")
        from vllm._genesis.wiring.spec_decode.patch_82_sglang_acceptance_threshold import (
            _read_min_draft_pos,
        )
        assert _read_min_draft_pos() == 0

    def test_negative_clamped_to_zero(self, monkeypatch):
        monkeypatch.setenv("GENESIS_P82_MIN_DRAFT_POS", "-5")
        from vllm._genesis.wiring.spec_decode.patch_82_sglang_acceptance_threshold import (
            _read_min_draft_pos,
        )
        assert _read_min_draft_pos() == 0

    def test_clamped_at_max_spec_len(self, monkeypatch):
        monkeypatch.setenv("GENESIS_P82_MIN_DRAFT_POS", "200")
        from vllm._genesis.wiring.spec_decode.patch_82_sglang_acceptance_threshold import (
            _read_min_draft_pos,
        )
        # MAX_SPEC_LEN = 128 in upstream; we clamp at 127 (last valid pos)
        assert _read_min_draft_pos() == 127

    def test_replacement_omits_position_guard_when_zero(self):
        """When min_draft_pos==0, replacement must NOT include the
        `pos >= N` guard so kernel disasm stays bit-equivalent to v1
        for current PROD users (who don't set this env var)."""
        rep = _build_replacement(0.3, min_draft_pos=0)
        assert "pos >=" not in rep, (
            "min_draft_pos=0 (default) MUST NOT inject a position guard "
            "into the kernel — backward-compat with v1 PROD"
        )

    def test_replacement_includes_position_guard_when_nonzero(self):
        rep = _build_replacement(0.3, min_draft_pos=2)
        assert "pos >= 2" in rep, (
            "min_draft_pos=2 must inject `pos >= 2` into the threshold "
            "clause"
        )

    def test_marker_encodes_min_draft_pos(self):
        """Two different min_draft_pos values must produce different
        markers so a config change forces re-apply."""
        from vllm._genesis.wiring.spec_decode.patch_82_sglang_acceptance_threshold import (
            _marker_for,
        )
        m_default = _marker_for(0.3, min_draft_pos=0)
        m_pos2 = _marker_for(0.3, min_draft_pos=2)
        assert m_default != m_pos2
        # Backward-compat: when min_draft_pos==0 the marker must NOT
        # include `mdp=` so existing v1 markers in source still match
        assert "mdp=" not in m_default
        assert "mdp=2" in m_pos2


class TestP82V2ThresholdOneIsSkip:
    """v2: threshold=1.0 is operator UX skip — only fires on argmax-tier
    confidence which is essentially never. Avoid patch overhead."""

    def test_apply_skipped_when_threshold_one(self, monkeypatch):
        from vllm._genesis.guards import vllm_install_root
        if vllm_install_root() is None:
            pytest.skip("vllm not installed — apply() short-circuits earlier")
        monkeypatch.setenv("GENESIS_ENABLE_P82", "1")
        monkeypatch.setenv("GENESIS_P82_THRESHOLD_SINGLE", "1.0")
        status, reason = apply()
        assert status == "skipped"
        assert "1.0" in reason
        # Operator UX: must mention it's effectively a no-op + suggest
        # a useful range
        assert "no-op" in reason or "argmax" in reason
        assert "0.7" in reason or "0.95" in reason
