# SPDX-License-Identifier: Apache-2.0
"""Wiring for Patch 82 — SGLang-style threshold_single OR-clause acceptance.

Backport of the per-token acceptance rule from SGLang
(`sgl-kernel/csrc/speculative/speculative_sampling.cuh` ~line 107):

    if (coin <= prob_acc / threshold_acc || target_prob_single >= threshold_single):
        accept

vs vLLM's vanilla rule (`vllm/v1/sample/rejection_sampler.py:797`):

    accepted = draft_prob > 0 and target_prob / draft_prob >= uniform_prob

P82 inserts the OR-clause: accept if EITHER vanilla rejection passes OR
the target's confidence in the drafted token meets a threshold. Targets
the structural ceiling `clean_rate ≈ accept_rate^num_spec` identified in
the v7.13 strict-ngram analysis.

================================================================
TRADE-OFF — READ THIS BEFORE ENABLING
================================================================

The threshold rule is **biased** — it loses the unbiased-sampling guarantee
of canonical rejection sampling. SGLang accepts this trade-off explicitly.
For greedy / low-temperature tool-call workloads (our case), the bias
short-circuits in favor of higher-prob target tokens, which is the right
direction. For temperature ≥ 1.0 creative-writing workloads the bias
could compress diversity. WE DO NOT SHIP THIS WITHOUT EMPIRICAL
VALIDATION (`genesis_quality_harness.py` ≥ 30/31 + `genesis_bench_v3.py`
TPS sweep).

================================================================
DESIGN
================================================================

- Text-patch on `vllm/v1/sample/rejection_sampler.py` inside the random
  sampling Triton kernel `rejection_random_sample_kernel`.
- The threshold is baked as a fp32 LITERAL at apply() time from env
  `GENESIS_P82_THRESHOLD_SINGLE` (default 0.3 — SGLang's typical default).
  Changing the threshold requires server restart.
- Greedy path is untouched (greedy already accepts on argmax-match;
  threshold doesn't apply to T=0).
- Synthetic mode is untouched (synthetic acceptance has its own rule).

================================================================
SAFETY MODEL
================================================================

- If env GENESIS_ENABLE_P82 is unset/0 → patch is SKIPPED, source stays
  vanilla. No runtime fall-through path needed.
- If anchor missing (upstream rewrote the line) → SKIPPED with clear
  reason; server boots on vanilla rule.
- Drift markers catch upstream's own threshold patch if/when it lands.

Status: opt-in via `GENESIS_ENABLE_P82=1`. Default OFF.

Tunable knobs
-------------
- `GENESIS_ENABLE_P82` (default unset/0): master switch
- `GENESIS_P82_THRESHOLD_SINGLE` (default 0.3): float in [0.0, 1.0]
  - 0.0 → disables the OR clause (equivalent to OFF, but with overhead)
  - 0.2-0.3 → SGLang typical range, light bias
  - ≥0.5 → aggressive, expect quality regression on diverse outputs

Compatibility
-------------
- All draft methods (ngram, MTP/EAGLE, suffix) — affects only the
  acceptance comparison, not the draft generation.
- Cudagraph: unaffected (rejection sampler runs OUTSIDE the captured graph).
- P71 (block-verify): mutually exclusive in practice — P71 takes the
  block-verify branch BEFORE this point if eligible. P82 fires on the
  per-token fall-through path. Safe to enable both.

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
Reference algorithm: SGLang team (sgl-project/sglang).
"""
from __future__ import annotations

import logging
import os

from vllm._genesis.guards import resolve_vllm_file, vllm_install_root
from vllm._genesis.wiring.text_patch import (
    TextPatcher,
    TextPatchResult,
    TextPatch,
)

log = logging.getLogger("genesis.wiring.p82_sglang_acceptance_threshold")


# NOTE: marker is built dynamically from the threshold so that operators
# changing GENESIS_P82_THRESHOLD_SINGLE between restarts cause apply() to
# re-patch (not silently skip on idempotency). See `_marker_for(threshold)`.
GENESIS_P82_MARKER_PREFIX = "Genesis P82 SGLang-style threshold_single OR-clause v7.63.x"


def _marker_for(threshold: float, min_draft_pos: int = 0) -> str:
    """Build the marker for a specific baked (threshold, min_draft_pos) tuple.

    v7.62.11 fix (B3 from hidden bug audit): previous marker was constant
    `"...v7.53"` regardless of `_BAKED_THRESHOLD`. Operator changes
    `GENESIS_P82_THRESHOLD_SINGLE` and restarts → marker check matches the
    OLD bake, returns IDEMPOTENT, **previously-baked threshold stays in
    source**. Threshold change silently ignored unless container fs reset.

    v7.63.x v2 (2026-04-30) extension: marker now also encodes
    `min_draft_pos`. Same forced-re-apply logic when EITHER value
    changes. Backward-compat: when `min_draft_pos == 0` the marker
    omits the `mdp=` segment entirely so existing v1 marker text in
    source is still recognized (no-op upgrade for current PROD users).
    """
    # Round to 4 decimals for marker stability (avoid 0.30000000000000004
    # vs 0.3 mismatches in apparently-equal env values)
    base = f"{GENESIS_P82_MARKER_PREFIX} thresh={float(threshold):.4f}"
    if min_draft_pos > 0:
        return f"{base} mdp={int(min_draft_pos)}"
    return base


# Back-compat alias (old constant name still imported by tests)
GENESIS_P82_MARKER = GENESIS_P82_MARKER_PREFIX


# ─── Threshold parsing (with bounds + fallback) ────────────────────────────

_DEFAULT_THRESHOLD = 0.3


def _read_threshold() -> float:
    raw = os.environ.get("GENESIS_P82_THRESHOLD_SINGLE", "").strip()
    if not raw:
        return _DEFAULT_THRESHOLD
    try:
        v = float(raw)
    except ValueError:
        log.warning(
            "[P82] GENESIS_P82_THRESHOLD_SINGLE=%r not parseable as float; using default %.2f",
            raw, _DEFAULT_THRESHOLD,
        )
        return _DEFAULT_THRESHOLD
    if not (0.0 <= v <= 1.0):
        log.warning(
            "[P82] threshold %.4f out of [0.0, 1.0]; clamping",
            v,
        )
        v = max(0.0, min(1.0, v))
    return v


# ─── v2: min draft-position guard (opt-in) ─────────────────────────────────

_DEFAULT_MIN_DRAFT_POS = 0  # 0 = current behavior (OR-clause fires at all positions)


def _read_min_draft_pos() -> int:
    """v2 (2026-04-30): operator can restrict the OR-clause to draft
    positions >= N. Earlier positions cascade-affect more output tokens,
    so biasing later positions is "safer" if quality drift is observed.

    Default 0 = current behavior (clause fires at every position).
    Recommended for ngram with low `prompt_lookup_min`: try =1 or =2 to
    reduce cascade impact while keeping the OR-clause's TPS win at later
    positions.

    Bounds: clamped to [0, MAX_SPEC_LEN-1] = [0, 127] so the clause can
    always fire on at least one position.
    """
    raw = os.environ.get("GENESIS_P82_MIN_DRAFT_POS", "").strip()
    if not raw:
        return _DEFAULT_MIN_DRAFT_POS
    try:
        v = int(raw)
    except ValueError:
        log.warning(
            "[P82] GENESIS_P82_MIN_DRAFT_POS=%r not parseable as int; "
            "using default %d", raw, _DEFAULT_MIN_DRAFT_POS,
        )
        return _DEFAULT_MIN_DRAFT_POS
    if v < 0:
        log.warning("[P82] min_draft_pos %d negative; clamping to 0", v)
        v = 0
    if v > 127:  # MAX_SPEC_LEN constant in upstream rejection_sampler
        log.warning(
            "[P82] min_draft_pos %d exceeds MAX_SPEC_LEN-1=127; clamping",
            v,
        )
        v = 127
    return v


# ─── v2: numerical-stability epsilon for draft_prob guard ──────────────────

# fp32 normal-range minimum is ~1e-38; we use 1e-20 as a safety margin
# that still covers ~99.999...% of realistic softmax outputs while
# guarding against denormal-zone instability in `target_prob / draft_prob`.
GENESIS_P82_DRAFT_PROB_EPS = 1e-20


# ─── Anchor: 3-line block including upstream NOTE comment for uniqueness ───

P82_OLD = (
    "                # NOTE(woosuk): While the draft probability should never be 0,\n"
    "                # we check it to avoid NaNs. If it happens to be 0, we reject.\n"
    "                accepted = draft_prob > 0 and target_prob / draft_prob >= uniform_prob\n"
)


def _build_replacement(threshold: float, min_draft_pos: int = 0) -> str:
    """Build the Triton-side replacement block.

    v2 improvements (2026-04-30):
    - Numerical-stability guard: replaces `draft_prob > 0` with
      `draft_prob >= 1e-20` to prevent fp32 denormal-zone overflow in
      `target_prob / draft_prob` (denormals can produce inf/NaN even
      though the value is "non-zero").
    - Defensive `target_prob > 0` check on the threshold-clause side:
      guards against malformed input where target softmax somehow
      produces a non-positive probability (impossible in practice but
      defensive).
    - `min_draft_pos` runtime guard: when `min_draft_pos > 0`, the
      OR-clause fires only on positions `pos >= min_draft_pos`. Earlier
      positions cascade-affect more output tokens; restricting the bias
      to later positions reduces quality drift while keeping the TPS
      win where it matters most. Default 0 = current behavior.

    All v2 changes preserve bit-equivalence with v1 when invoked with
    default args (threshold > 0, min_draft_pos = 0): the new guards
    only EXCLUDE acceptances v1 would have made on numerically-degenerate
    input, which is the desired safety direction.
    """
    # Bake threshold as a fp32-precision literal (Python repr of float is
    # round-trip safe, sufficient for Triton constexpr coercion).
    threshold_literal = repr(float(threshold))
    eps_literal = repr(float(GENESIS_P82_DRAFT_PROB_EPS))

    # Build the threshold-clause guard. When min_draft_pos == 0 (default)
    # we omit the position guard entirely so the kernel disasm stays
    # identical to v1 — important because operators may have validated
    # P82 v1 on PROD and we don't want to introduce silent reordering.
    if min_draft_pos > 0:
        position_guard = f" and pos >= {min_draft_pos}"
        position_doc = (
            f"                # [Genesis P82 v2] OR-clause restricted to "
            f"draft pos >= {min_draft_pos} (min_draft_pos guard); earlier\n"
            f"                # positions use vanilla rule only. Cascade-impact "
            f"reduction.\n"
        )
    else:
        position_guard = ""
        position_doc = ""

    return (
        "                # NOTE(woosuk): While the draft probability should never be 0,\n"
        "                # we check it to avoid NaNs. If it happens to be 0, we reject.\n"
        "                # ════════════════════════════════════════════════════════════════\n"
        "                # [Genesis P82 v2 SGLang-style] threshold_single OR-clause\n"
        "                # accept if EITHER vanilla rejection passes OR target's confidence\n"
        "                # in the drafted token meets the configured threshold. Bias trade-off:\n"
        "                # loses unbiased-sampling guarantee; chosen for low-temp tool-call.\n"
        "                # Threshold baked from env GENESIS_P82_THRESHOLD_SINGLE at server start.\n"
        "                #\n"
        "                # v2 (2026-04-30) hardening:\n"
        f"                #   - draft_prob guard tightened: > 0  →  >= {eps_literal}\n"
        "                #     (fp32 denormal-zone protection; prevents inf/NaN ratio)\n"
        "                #   - target_prob > 0 explicit (defensive vs malformed softmax)\n"
        + (f"                #   - min_draft_pos = {min_draft_pos} (OR-clause restricted)\n"
           if min_draft_pos > 0 else "")
        + "                # ════════════════════════════════════════════════════════════════\n"
        + position_doc
        + "                _genesis_p82_vanilla = (\n"
        f"                    draft_prob >= {eps_literal} and target_prob / draft_prob >= uniform_prob\n"
        "                )\n"
        f"                _genesis_p82_threshold = (\n"
        f"                    target_prob > 0 and target_prob >= {threshold_literal}{position_guard}\n"
        f"                )\n"
        "                accepted = _genesis_p82_vanilla or _genesis_p82_threshold\n"
    )


def _make_patcher(threshold: float, min_draft_pos: int = 0) -> TextPatcher | None:
    target = resolve_vllm_file("v1/sample/rejection_sampler.py")
    if target is None:
        return None
    return TextPatcher(
        patch_name=(
            "P82 v2 v1/sample/rejection_sampler.py — SGLang threshold_single "
            f"OR-clause (threshold={threshold:.4f}, min_draft_pos={min_draft_pos})"
        ),
        target_file=str(target),
        # B3 fix: marker now embeds threshold + min_draft_pos so a config
        # change forces re-apply instead of silently passing IDEMPOTENT.
        marker=_marker_for(threshold, min_draft_pos),
        sub_patches=[
            TextPatch(
                name="p82_threshold_or_clause",
                anchor=P82_OLD,
                replacement=_build_replacement(threshold, min_draft_pos),
                required=True,
            ),
        ],
        upstream_drift_markers=[
            "[Genesis P82",
            "_genesis_p82_threshold",
            # Upstream-side markers: if vLLM ever ships its own threshold_single
            # arg in this kernel, we should bow out and let upstream handle it.
            "threshold_single",
            "speculative-accept-threshold",
            # PR #40819 markers (block-wise verification rule, OPEN as of
            # 2026-04-29). #40819 inserts a separate `if use_block_verify:`
            # branch BEFORE the per-token kernel; control flow only reaches
            # P82's anchor when verify_method != "block", so the two are
            # complementary at runtime. We still drift-detect the merge so
            # we know upstream now has the canonical SGLang block-verify
            # rule for ≥3 spec tokens + real draft probs (P82 keeps the
            # OR-clause for ngram + short-spec paths). Authors:
            # masterFoad / z00918512.
            "use_block_verify",
            "verify_method",
            "_BLOCK_VERIFY_VOCAB_BLOCK",
            "_block_verify_kernel",
            # vllm/config/speculative.py marker for the new SpecVerifyMethod
            # field added by #40819:
            "SpecVerifyMethod",
            # PR #41258 (masterFoad, OPEN 2026-04-30): "Lazy recovery
            # evaluation for spec rejection sampling" — removes the
            # eager `sample_recovered_tokens_kernel` (full-vocab scan
            # for ALL draft positions) and computes recovered tokens
            # lazily inside `rejection_random_sample_kernel` only at
            # the rejected position. If this lands, the kernel that
            # P82's baked threshold reads from is restructured —
            # P82 will need a redesign around the lazy path. Watch
            # for the kernel signature change:
            "sample_recovered_tokens_kernel",
            "_lazy_recovered_token",
            "lazy_recovery",
        ],
    )


def apply() -> tuple[str, str]:
    """Apply P82 — SGLang threshold_single OR-clause acceptance."""
    from vllm._genesis.dispatcher import should_apply, log_decision
    decision, reason = should_apply("P82")
    log_decision("P82", decision, reason)
    if not decision:
        return "skipped", reason

    if vllm_install_root() is None:
        return "skipped", "vllm install root not discoverable"

    threshold = _read_threshold()
    if threshold == 0.0:
        # Equivalent to OFF (OR-clause never fires) but with patch overhead;
        # explicitly skip to keep the source vanilla.
        return "skipped", (
            "GENESIS_P82_THRESHOLD_SINGLE=0.0 — OR clause would never fire; "
            "skipping patch to keep source vanilla"
        )

    # v2 (2026-04-30): explicit skip when threshold is 1.0 — operator UX.
    # `target_prob >= 1.0` only fires for argmax-tier confidence which is
    # essentially never. Patch overhead would add no value.
    if threshold >= 1.0:
        return "skipped", (
            "GENESIS_P82_THRESHOLD_SINGLE=1.0 — OR clause would only fire on "
            "100%-confident target prob (argmax cases). Effectively a no-op; "
            "skipping patch. Set threshold to 0.7-0.95 for meaningful TPS gain."
        )

    min_draft_pos = _read_min_draft_pos()

    patcher = _make_patcher(threshold, min_draft_pos)
    if patcher is None:
        return "skipped", "vllm/v1/sample/rejection_sampler.py not found"

    if not os.path.isfile(patcher.target_file):
        return "skipped", f"target disappeared: {patcher.target_file}"
    with open(patcher.target_file) as f:
        content = f.read()
    if patcher.marker in content:
        log.info("[P82] marker present (current threshold) — skip (idempotent)")
        return "applied", "idempotent (marker present, threshold unchanged)"
    # B3 fix: detect stale P82 marker from a different threshold bake.
    # If a different P82 prefix marker is present, the source has the OLD
    # threshold baked. We can't safely re-patch because the original anchor
    # is now consumed. Operator must `docker compose down && up -d` to reset
    # the container fs first. Surface this clearly instead of silent skip.
    if GENESIS_P82_MARKER_PREFIX in content:
        return (
            "skipped",
            f"P82 stale marker present (different threshold). Container fs has a previous "
            f"P82 bake; current threshold={threshold:.4f} cannot be applied without resetting "
            f"the source. Reset via `docker compose down && up -d` (NOT just stop/start)."
        )
    for m in patcher.upstream_drift_markers:
        if m == "[Genesis P82" and m in content:
            continue  # our marker; handled above
        if m in content:
            return (
                "skipped",
                f"upstream drift marker {m!r} in {patcher.target_file} — "
                "upstream may have absorbed this fix or independent threshold patch",
            )

    result, failure = patcher.apply()
    # Audit P1 fix 2026-05-05: route SKIPPED to "skipped" (was masked as "applied")
    # via the centralized helper that already lives in text_patch.py.
    from vllm._genesis.wiring.text_patch import result_to_wiring_status
    pos_note = (
        f", min_draft_pos={min_draft_pos}"
        if min_draft_pos > 0
        else ""
    )
    applied_msg = (
        f"P82 v2 applied: SGLang threshold_single OR-clause installed at "
        f"threshold={threshold:.4f}{pos_note}. v2 hardening: fp32 denormal guard "
        "(draft_prob >= 1e-20), explicit target_prob > 0 check"
        + (f", OR-clause restricted to draft pos >= {min_draft_pos}"
           if min_draft_pos > 0 else "")
        + ". Activates on random-sample path (greedy / synthetic untouched). "
        "BIASED rule — validate with genesis_quality_harness before prod."
    )
    return result_to_wiring_status(
        result, failure,
        applied_message=applied_msg,
        patch_name=patcher.patch_name,
    )
