# SPDX-License-Identifier: Apache-2.0
"""Wiring for Patch 70 — Auto-strict-ngram (env-gated min>=8 enforcement).

Genesis-original — addresses the empirically-confirmed acceptance-rate
heuristic from vllm#40875 / Genesis v7.13 BREAKTHROUGH discovery.

================================================================
WHAT THIS DOES

When `--speculative-config '{"method":"ngram","prompt_lookup_min":N}'`
is set with N<8, Genesis ngram acceptance heuristic shows tool-call
output corruption (model produces `<<`, `parameter=parameter`,
`<argname>` patterns due to spurious draft acceptance from tool
schema lookups).

Empirical study in Genesis v7.13 found `prompt_lookup_min=8` is the
sweet spot for clean tool-call rate (100% clean vs ~20% baseline at
default min=5).

P70 hooks `SpeculativeConfig.__post_init__` to AUTO-FORCE
prompt_lookup_min>=8 if env flag set. Operator can disable to keep
explicit lower min for non-tool workloads.

Status: opt-in via `GENESIS_ENABLE_P70_AUTO_STRICT_NGRAM=1`.

================================================================

Compatibility
-------------
- Affects ONLY when `speculative_config.method == "ngram"` (or "ngram_gpu")
- No-op when env flag off
- No-op when min already >= 8
- Auto-bumps prompt_lookup_max if needed to satisfy invariant min <= max
- Logs WARNING with original value so operator can see what was changed
- Idempotent (marker check)

Tradeoff
--------
Higher min = stricter matching = lower acceptance rate but higher
correctness. For tool-call workloads correctness >> speed (since
broken tool calls fail downstream agents). For pure plain-text
workloads default min=5 is fine — operator should NOT enable P70.

References
----------
- vllm#40875 (Genesis report — config-only fix achieves 100% clean)
- Genesis v7.13 BREAKTHROUGH section in README
- Genesis_Doc/REPORT_v7_13_BREAKTHROUGH_STRICT_NGRAM_RU.md (internal)

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
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

log = logging.getLogger("genesis.wiring.p70_auto_strict_ngram")

GENESIS_P70_MARKER = "Genesis P70 auto-strict-ngram min>=8 enforcement v7.15"


# ─── Sub-patch: inject auto-force hook into SpeculativeConfig.__post_init__ ─
# Anchor on the closing of the validation block (lines 442-447 in upstream).
# Insert AFTER the validation but BEFORE the draft_model_config assignment.
#
# NOTE: the trailing "# TODO: current we still need extract vocab_size from
# target model" line in P70_OLD/P70_NEW below is a verbatim copy from upstream
# `vllm/config/speculative.py`. It is **load-bearing** for TextPatcher
# anchor matching — do not "clean up" the typo or rewrite the comment.
# If upstream fixes that comment, this patch will need a corresponding
# anchor refresh.

P70_OLD = (
    "            # Validate values\n"
    "            if self.prompt_lookup_min > self.prompt_lookup_max:\n"
    "                raise ValueError(\n"
    "                    f\"prompt_lookup_min={self.prompt_lookup_min} must \"\n"
    "                    f\"be <= prompt_lookup_max={self.prompt_lookup_max}\"\n"
    "                )\n"
    "\n"
    "            # TODO: current we still need extract vocab_size from target model\n"
)

P70_NEW = (
    "            # Validate values\n"
    "            if self.prompt_lookup_min > self.prompt_lookup_max:\n"
    "                raise ValueError(\n"
    "                    f\"prompt_lookup_min={self.prompt_lookup_min} must \"\n"
    "                    f\"be <= prompt_lookup_max={self.prompt_lookup_max}\"\n"
    "                )\n"
    "\n"
    "            # [Genesis P70 auto-strict-ngram] When env flag is set, force\n"
    "            # prompt_lookup_min >= 8 to avoid the spurious-acceptance bug\n"
    "            # class identified in vllm#40875 (Genesis investigation 2026-04-25).\n"
    "            # At min<8 ngram lookup matches tool-schema fragments and produces\n"
    "            # degenerate tool-call output. min>=8 = matched-only acceptance\n"
    "            # proven 100% clean. Operator can disable for plain-text workloads.\n"
    "            import os as _genesis_p70_os\n"
    "            if _genesis_p70_os.environ.get(\n"
    "                \"GENESIS_ENABLE_P70_AUTO_STRICT_NGRAM\", \"\"\n"
    "            ).strip().lower() in (\"1\", \"true\", \"yes\", \"on\"):\n"
    "                if self.prompt_lookup_min < 8:\n"
    "                    import logging as _genesis_p70_logging\n"
    "                    _genesis_p70_orig_min = self.prompt_lookup_min\n"
    "                    _genesis_p70_orig_max = self.prompt_lookup_max\n"
    "                    self.prompt_lookup_min = 8\n"
    "                    if self.prompt_lookup_max < 8:\n"
    "                        self.prompt_lookup_max = 8\n"
    "                    _genesis_p70_logging.getLogger(\n"
    "                        \"genesis.p70_strict_ngram\"\n"
    "                    ).warning(\n"
    "                        \"[Genesis P70] Auto-tuned ngram: \"\n"
    "                        \"prompt_lookup_min %d -> 8, prompt_lookup_max %d -> %d. \"\n"
    "                        \"Strict mode for tool-call correctness (see vllm#40875). \"\n"
    "                        \"Set GENESIS_ENABLE_P70_AUTO_STRICT_NGRAM=0 to disable.\",\n"
    "                        _genesis_p70_orig_min, _genesis_p70_orig_max,\n"
    "                        self.prompt_lookup_max,\n"
    "                    )\n"
    "\n"
    "            # TODO: current we still need extract vocab_size from target model\n"
)


def _make_patcher() -> TextPatcher | None:
    target = resolve_vllm_file("config/speculative.py")
    if target is None:
        return None
    return TextPatcher(
        patch_name="P70 config/speculative.py — auto-strict-ngram min>=8",
        target_file=str(target),
        marker=GENESIS_P70_MARKER,
        sub_patches=[
            TextPatch(
                name="p70_auto_strict_ngram_force",
                anchor=P70_OLD,
                replacement=P70_NEW,
                required=True,
            ),
        ],
        upstream_drift_markers=[
            "[Genesis P70",
            "_genesis_p70_orig_min",
            "auto-strict-ngram",
        ],
    )


def apply() -> tuple[str, str]:
    """Apply P70 — auto-strict-ngram min>=8 enforcement."""
    from vllm._genesis.dispatcher import should_apply, log_decision
    decision, reason = should_apply("P70")
    log_decision("P70", decision, reason)
    if not decision:
        return "skipped", reason

    if vllm_install_root() is None:
        return "skipped", "vllm install root not discoverable"

    patcher = _make_patcher()
    if patcher is None:
        return "skipped", "vllm/config/speculative.py not found"

    if not os.path.isfile(patcher.target_file):
        return "skipped", f"target disappeared: {patcher.target_file}"
    with open(patcher.target_file) as f:
        content = f.read()
    if patcher.marker in content:
        pass  # idempotent
    else:
        for m in patcher.upstream_drift_markers:
            if m in content:
                return (
                    "skipped",
                    f"upstream drift marker {m!r} in {patcher.target_file} — "
                    "auto-strict ngram likely already merged upstream.",
                )
        if patcher.sub_patches[0].anchor not in content:
            return (
                "skipped",
                "required anchor (prompt_lookup_min validation block) not "
                "found — P70 cannot apply (upstream drift).",
            )

    result, failure = patcher.apply()
    # Audit P1 fix 2026-05-05: surface SKIPPED as skipped (was masked as applied)
    if result == TextPatchResult.SKIPPED:
        _r = failure.reason if failure else "anchor drift / not eligible"
        _d = f" ({failure.detail})" if (failure and failure.detail) else ""
        return "skipped", f"{patcher.patch_name}: {_r}{_d}"
    if result == TextPatchResult.FAILED:
        return "failed", (
            f"{patcher.patch_name}: {failure.reason if failure else 'unknown'} "
            f"({failure.detail if failure else ''})"
        )

    return "applied", (
        "P70 applied: when GENESIS_ENABLE_P70_AUTO_STRICT_NGRAM=1, "
        "prompt_lookup_min and prompt_lookup_max are auto-bumped to >=8 "
        "for ngram method. Disable env flag to opt-out per workload."
    )
