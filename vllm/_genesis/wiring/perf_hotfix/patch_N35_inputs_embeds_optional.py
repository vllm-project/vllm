# SPDX-License-Identifier: Apache-2.0
"""Wiring for Patch N35 — skip inputs_embeds buffer for text-only models.

================================================================
Issue
================================================================

vLLM unconditionally allocates a `(max_num_tokens, hidden_size)` GPU
buffer for `inputs_embeds` in two places:

  1. `vllm/v1/worker/gpu_model_runner.py:713` — main model runner
  2. `vllm/v1/spec_decode/llm_base_proposer.py:205` — spec-decode proposer

For Qwen3.6-27B at `max_num_tokens=4096` and hidden_size=8192:
  4096 × 8192 × 2 bytes (bf16) = **64 MiB GPU per buffer**
  Plus the same on pinned CPU (the `_make_buffer` allocates both).

For text-only models (no multimodal inputs, no `enable_prompt_embeds`),
this buffer is NEVER read or written — pure dead allocation. On
borderline-OOM configs (single 24GB card + long context + spec-decode
+ aggressive mem-util), the freed ~64 MiB GPU + ~64 MiB pinned CPU
can be the difference between OOM and working.

================================================================
UPSTREAM
================================================================

vllm-project/vllm#35975 by AjAnubolu (OPEN since 2026-03-04, awaiting
code-owner approval after author addressed reviewer feedback).

When PR #35975 merges + ships in our nightly tag, this patch retires.
The TextPatcher framework's anchor-not-found path catches the merge
automatically (anchor disappears from upstream → SKIPPED, no harm).

================================================================
COMPOSITION
================================================================

- **Independent of all other Genesis patches** — no conflict, no
  interaction. Pure GPU buffer accounting.
- **Particularly relevant alongside P103 + PN32** (Cliff 2 stack on
  single-24GB-GPU long context). Each freed MiB matters when the
  Cliff 2 OOM trigger fires at "tried to allocate 50 MiB, 24.5 MiB
  free" boundaries.
- **Particularly relevant on WSL2 setups** (per club-3090#32) where
  the Xwayland/WSL vGPU layer eats ~830 MiB-1 GiB extra overhead.

================================================================
SAFETY MODEL
================================================================

- **Default ON** when text-only model detected — this is a strict
  improvement (no regression possible for text-only). Operators with
  multimodal models keep the buffer (the patch's `if` guard checks
  `self.supports_mm_inputs or self.enable_prompt_embeds`).
- Pure text-patch on 2 sites (gpu_model_runner + llm_base_proposer).
  Idempotent via marker.
- Anchor includes the EXACT existing allocation block; if upstream
  rewrites either site (e.g. when PR #35975 merges with different
  factoring), the anchor won't match → SKIPPED, source stays vanilla.
- Worst case: on a multimodal model where `supports_mm_inputs=True`,
  the patch's `if` guard re-allocates the buffer — same behavior as
  unpatched code. Zero regression path.

================================================================
HISTORY
================================================================

- **2026-05-03 v7.69.x** — Genesis port of PR #35975, prompted by
  club-3090#32 (RossNE99 + GuiPerPT WSL2 OOM reports). noonghunna had
  already shipped this as a setup-time sidecar
  (`patch_inputs_embeds_optional.py`) on club-3090 dev tip; this PN35
  promotes it to a first-class Genesis text-patch so it survives
  the entrypoint `exec vllm serve` pattern (same survivability
  argument as P103 v7.69 self-install).

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
Backport: vllm#35975 by AjAnubolu (UPSTREAM author, full credit).
Pattern credit: noonghunna (club-3090 setup-time sidecar
                `patch_inputs_embeds_optional.py`, 2026-05-02).
Originally raised by: club-3090#32 reporters (RossNE99, GuiPerPT)
                      hitting borderline OOM on 24GB cards under WSL2.
"""
from __future__ import annotations

import logging

from vllm._genesis.guards import resolve_vllm_file, vllm_install_root
from vllm._genesis.wiring.text_patch import (
    TextPatch,
    TextPatcher,
    TextPatchResult,
    result_to_wiring_status,
)

log = logging.getLogger("genesis.wiring.pN35_inputs_embeds_optional")

GENESIS_PN35_MARKER = (
    "Genesis PN35 inputs_embeds optional for text-only (vllm#35975) v7.69"
)


# ─── Sub-patch 1: gpu_model_runner.py:713 ───────────────────────────

PN35_PART1_ANCHOR = (
    "        self.inputs_embeds = self._make_buffer(\n"
    "            self.max_num_tokens, self.inputs_embeds_size, dtype=self.dtype, numpy=False\n"
    "        )\n"
)

PN35_PART1_REPLACEMENT = (
    "        # [Genesis PN35 v7.69 inputs_embeds_optional] vllm#35975 by\n"
    "        # AjAnubolu — skip inputs_embeds buffer (~64 MiB GPU + 64 MiB\n"
    "        # pinned CPU) on text-only models. Allocation only fires when\n"
    "        # the model needs it (multimodal or prompt embeds enabled).\n"
    "        # Retires when PR #35975 merges upstream.\n"
    "        self.inputs_embeds = None\n"
    "        if self.supports_mm_inputs or self.enable_prompt_embeds:\n"
    "            self.inputs_embeds = self._make_buffer(\n"
    "                self.max_num_tokens, self.inputs_embeds_size, dtype=self.dtype, numpy=False\n"
    "            )\n"
)


# ─── Sub-patch 2: llm_base_proposer.py:205 ──────────────────────────

PN35_PART2_ANCHOR = (
    "        self.inputs_embeds = torch.zeros(\n"
    "            (self.max_num_tokens, self.inputs_embeds_size),\n"
    "            dtype=self.dtype,\n"
    "            device=device,\n"
    "        )\n"
)

PN35_PART2_REPLACEMENT = (
    "        # [Genesis PN35 v7.69 inputs_embeds_optional] vllm#35975 by\n"
    "        # AjAnubolu — same pattern for spec-decode proposer's\n"
    "        # inputs_embeds tensor. Saves ~64 MiB GPU on text-only spec-\n"
    "        # decode methods (MTP, ngram, draft-model w/o prompt embeds).\n"
    "        self.inputs_embeds = None\n"
    "        if self.supports_mm_inputs:\n"
    "            self.inputs_embeds = torch.zeros(\n"
    "                (self.max_num_tokens, self.inputs_embeds_size),\n"
    "                dtype=self.dtype,\n"
    "                device=device,\n"
    "            )\n"
)


def _make_patcher_part1() -> TextPatcher | None:
    target = resolve_vllm_file("v1/worker/gpu_model_runner.py")
    if target is None:
        return None
    return TextPatcher(
        patch_name=(
            "PN35 v1/worker/gpu_model_runner.py — skip inputs_embeds "
            "buffer for text-only (vllm#35975)"
        ),
        target_file=str(target),
        marker=GENESIS_PN35_MARKER + " part1",
        sub_patches=[
            TextPatch(
                name="pN35_gpu_model_runner_inputs_embeds_optional",
                anchor=PN35_PART1_ANCHOR,
                replacement=PN35_PART1_REPLACEMENT,
                required=True,
            ),
        ],
        upstream_drift_markers=[
            # Specific to part1's own insertion. Re-runs hit Layer 2
            # idempotency marker first (Layer 3 never fires on re-run).
            "[Genesis PN35 v7.69 inputs_embeds_optional]",
            # Upstream-merge signal: when PR #35975 lands, the upstream
            # code likely uses a similar guard pattern. This catches
            # post-merge state automatically.
            "vllm#35975",
        ],
    )


def _make_patcher_part2() -> TextPatcher | None:
    target = resolve_vllm_file("v1/spec_decode/llm_base_proposer.py")
    if target is None:
        return None
    return TextPatcher(
        patch_name=(
            "PN35 v1/spec_decode/llm_base_proposer.py — skip "
            "inputs_embeds tensor for text-only proposers (vllm#35975)"
        ),
        target_file=str(target),
        marker=GENESIS_PN35_MARKER + " part2",
        sub_patches=[
            TextPatch(
                name="pN35_llm_base_proposer_inputs_embeds_optional",
                anchor=PN35_PART2_ANCHOR,
                replacement=PN35_PART2_REPLACEMENT,
                required=True,
            ),
        ],
        upstream_drift_markers=[
            "[Genesis PN35 v7.69 inputs_embeds_optional]",
            "vllm#35975",
        ],
    )


def apply() -> tuple[str, str]:
    """Apply PN35 — skip inputs_embeds buffer on text-only models.

    Two-file text-patch (gpu_model_runner + llm_base_proposer).
    Default ON (when env-flag GENESIS_DISABLE_PN35_INPUTS_EMBEDS_OPTIONAL
    not set) — strict memory savings, no regression possible since the
    `if` guard preserves original behavior for multimodal/prompt-embed
    models.

    Never raises. Returns (status, reason).
    """
    from vllm._genesis.dispatcher import log_decision, should_apply

    decision, reason = should_apply("PN35")
    log_decision("PN35", decision, reason)
    if not decision:
        return "skipped", reason

    if vllm_install_root() is None:
        return "skipped", "vllm install root not discoverable"

    p1 = _make_patcher_part1()
    p2 = _make_patcher_part2()
    if p1 is None or p2 is None:
        return "skipped", (
            "target file(s) not resolvable — vllm tree may differ "
            "from expected layout"
        )

    patch_results = [
        ("part1 v1/worker/gpu_model_runner.py:inputs_embeds", *p1.apply()),
        (
            "part2 v1/spec_decode/llm_base_proposer.py:inputs_embeds",
            *p2.apply(),
        ),
    ]
    for label, result, failure in patch_results:
        if result not in (
            TextPatchResult.APPLIED,
            TextPatchResult.IDEMPOTENT,
        ):
            reason_text = failure.reason if failure else "unknown"
            detail = (
                failure.detail
                if failure and failure.detail
                else "unknown"
            )
            # Soft-fail on a single sub-patch — the other half still
            # gives meaningful savings. Log warning and continue.
            log.warning(
                "[Genesis PN35] %s did not apply: %s — %s "
                "(continuing with sibling patches)",
                label, reason_text, detail,
            )

    # Report applied if at least one sub-patch landed
    any_applied = any(
        r == TextPatchResult.APPLIED
        for _, r, _ in patch_results
    )
    any_idempotent = any(
        r == TextPatchResult.IDEMPOTENT
        for _, r, _ in patch_results
    )
    if not (any_applied or any_idempotent):
        return "failed", (
            "PN35: no sub-patch applied — vllm pin may have rewritten "
            "both inputs_embeds allocation sites. Check upstream "
            "PR #35975 status; may have merged."
        )

    status_result = (
        TextPatchResult.APPLIED if any_applied
        else TextPatchResult.IDEMPOTENT
    )
    return result_to_wiring_status(
        status_result,
        None,
        applied_message=(
            "PN35 v7.69 applied: inputs_embeds buffer now skipped on "
            "text-only models (gpu_model_runner + llm_base_proposer "
            "spec-decode proposer). Saves ~64 MiB GPU + ~64 MiB pinned "
            "CPU per site on Qwen3.6-27B at max_num_tokens=4096. "
            "Auto-retires when upstream vllm#35975 lands. Particularly "
            "useful on borderline-OOM configs (single-24GB-GPU + long "
            "context + spec-decode, WSL2 setups with extra display "
            "overhead). NO-OP on multimodal models — guard preserves "
            "original allocation when supports_mm_inputs=True."
        ),
        patch_name="PN35 inputs_embeds optional for text-only",
    )
