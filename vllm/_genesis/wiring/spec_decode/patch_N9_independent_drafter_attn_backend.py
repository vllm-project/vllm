# SPDX-License-Identifier: Apache-2.0
"""Wiring for Patch N9 — independent drafter attention backend.

================================================================
Source PR
================================================================
https://github.com/vllm-project/vllm/pull/39930
"[Attention][Spec Decode] Allow independent drafter attention backend
selection" by @MatthewBonanni (Matthew Bonanni), MERGED upstream.

================================================================
WHAT IT DOES (one paragraph)
================================================================

Currently the spec-decode drafter inherits the target model's attention
backend. This breaks for drafters with incompatible requirements
(e.g. DFlash needs non-causal attention support, which TRITON_ATTN
does not provide → ValueError on boot). Upstream PR #39930 adds a
`--speculative-config.attention_backend` CLI flag and modifies
`LLMBaseProposer._create_draft_vllm_config()` to ALWAYS reset the
drafter's attention backend (None = auto-select independently from
target). This unblocks the DFlash spike sprint task without requiring
a full pin bump (which would drag in #40860 mega-merge risk).

The minimal Genesis backport text-patches `_create_draft_vllm_config`
to replace `attention_config.backend` with a value chosen by env
`GENESIS_PN9_DRAFTER_BACKEND`. Unset (default) → None → drafter
auto-selects whatever backend matches its requirements, ignoring the
target's choice. Set to a backend name (e.g. "FLASH_ATTN", "FLASHINFER",
"TRITON_ATTN") → drafter pinned to that backend.

We do NOT add the new pydantic field on `SpeculativeConfig` (PR #39930
adds `attention_backend: AttentionBackendEnum | None`). That requires
modifying a frozen dataclass + adding a `field_validator` — too invasive
for a runtime text-patch and would risk pydantic validation drift
across vllm versions. The env-driven control is functionally equivalent
for our single-operator deployment model.

================================================================
SAFETY MODEL
================================================================

- Default OFF (opt-in via `GENESIS_ENABLE_PN9_INDEPENDENT_DRAFTER_ATTN=1`).
- Text-patch on `vllm/v1/spec_decode/llm_base_proposer.py` —
  `LLMBaseProposer._create_draft_vllm_config()` body.
- Idempotent via marker `Genesis PN9 ...`.
- Drift-aware: if upstream PR #39930 (or equivalent) appears in the
  target file (detect via "spec_cfg.attention_backend" reference in
  the function body), apply() returns SKIPPED with a "self-retired"
  reason. The merged upstream behavior is a strict superset of ours.
- Anchor missing → SKIPPED, source stays vanilla. No regression.
- Worst-case regression: if the auto-selected drafter backend turns
  out to be INCOMPATIBLE with the target's KV cache layout (e.g. the
  drafter picks FLASH_ATTN while the target uses FLASHINFER, and the
  KV cache page format differs), the drafter init will raise. To
  recover: set `GENESIS_PN9_DRAFTER_BACKEND=<target's backend name>`
  to pin both to the same backend — restores upstream pre-PR behavior.

Predicates / activation
-----------------------
The patch is unconditional once env enabled — but only fires when
spec-decode is active (the patched function is only called from the
draft-model load path). For non-spec-decode boots, the patch sits as
unused source modification.

We deliberately avoid spec-method gating at the apply() site because:
  (a) `model_detect` does not currently expose live spec_method;
  (b) the patched function preserves the moe_backend branch verbatim;
  (c) the new attention_backend reset only takes effect when
      spec-decode is engaged.

Author backport: Sandermage(Sander) Barzov Aleksandr, Ukraine, Odessa.
Source PR: vllm-project/vllm#39930 by @MatthewBonanni.
"""
from __future__ import annotations

import logging
import os

from vllm._genesis.guards import resolve_vllm_file, vllm_install_root
from vllm._genesis.wiring.text_patch import (
    TextPatch,
    TextPatcher,
    TextPatchResult,
)

log = logging.getLogger("genesis.wiring.pN9_independent_drafter_attn_backend")

GENESIS_PN9_MARKER_PREFIX = (
    "Genesis PN9 independent drafter attention backend (vllm#39930) v7.62.x"
)


def _read_drafter_backend() -> str | None:
    """Return the operator-chosen drafter backend name, or None for auto.

    Recognized values (case-insensitive): FLASH_ATTN, FLASHINFER,
    TRITON_ATTN, FLEX_ATTENTION, TORCH_SDPA, ROCM_FLASH, plus any other
    `AttentionBackendEnum` member. Unknown values are passed through;
    AttentionBackendEnum will raise at draft init if invalid (deferred
    validation — keeps this patch independent of vllm enum churn).
    """
    raw = os.environ.get("GENESIS_PN9_DRAFTER_BACKEND", "").strip()
    if not raw or raw.lower() in ("auto", "none", ""):
        return None
    return raw.upper()


def _marker_for(backend: str | None) -> str:
    """Build a marker that embeds the chosen backend.

    Same B3 lesson from P82: if the operator changes
    GENESIS_PN9_DRAFTER_BACKEND between restarts, idempotency check
    must catch the mismatch and re-apply (otherwise the previous bake
    silently sticks). Embedding the value in the marker triggers the
    re-apply path naturally. Container fs reset is required because
    the original anchor was already consumed by the previous bake.
    """
    return f"{GENESIS_PN9_MARKER_PREFIX} backend={backend or 'auto'}"


# ─── Anchor: full _create_draft_vllm_config body ─────────────────────────
# This anchor includes the moe_backend branch verbatim so we don't
# lose that behavior. Replacement preserves moe_backend handling, then
# resets attention_config.backend at the END (mirrors PR #39930).

PN9_OLD = (
    "    def _create_draft_vllm_config(self) -> VllmConfig:\n"
    "        \"\"\"Return a VllmConfig with kernel-level overrides for the proposer.\n"
    "        Subclasses may override to apply additional config changes.\n"
    "        \"\"\"\n"
    "        spec_cfg = self.speculative_config\n"
    "        if spec_cfg.moe_backend is not None:\n"
    "            return replace(\n"
    "                self.vllm_config,\n"
    "                kernel_config=replace(\n"
    "                    self.vllm_config.kernel_config,\n"
    "                    moe_backend=spec_cfg.moe_backend,\n"
    "                ),\n"
    "            )\n"
    "        return self.vllm_config\n"
)


def _build_replacement(backend: str | None) -> str:
    """Build the patched body. backend=None → drafter auto-selects."""
    if backend is None:
        backend_literal = "None"
    else:
        # Defer enum resolution to runtime — avoids hard-import of
        # AttentionBackendEnum at module load (which may not exist on
        # older vllm builds).
        backend_literal = (
            "_genesis_pn9_resolve("
            f"{backend!r}"
            ")"
        )
    return (
        "    def _create_draft_vllm_config(self) -> VllmConfig:\n"
        "        \"\"\"Return a VllmConfig with kernel-level overrides for the proposer.\n"
        "        Subclasses may override to apply additional config changes.\n"
        "\n"
        "        [Genesis PN9 vllm#39930 backport] Always reset the drafter's\n"
        "        attention backend so it is auto-selected (or pinned via env)\n"
        "        independently from the target. Mirrors upstream PR #39930\n"
        "        behavior without requiring the new SpeculativeConfig field.\n"
        "        \"\"\"\n"
        "        spec_cfg = self.speculative_config\n"
        "        # Helper: deferred enum resolution (avoids module-load import)\n"
        "        def _genesis_pn9_resolve(name):\n"
        "            try:\n"
        "                from vllm.v1.attention.backends.registry import (\n"
        "                    AttentionBackendEnum as _AB,\n"
        "                )\n"
        "                return _AB[name]\n"
        "            except Exception as _e:  # pragma: no cover\n"
        "                import logging as _genesis_pn9_logging\n"
        "                _genesis_pn9_logging.getLogger(\n"
        "                    \"genesis.wiring.pN9\"\n"
        "                ).warning(\n"
        "                    \"[Genesis PN9] could not resolve backend %r: %s; \"\n"
        "                    \"falling back to auto-select (None)\",\n"
        "                    name, _e,\n"
        "                )\n"
        "                return None\n"
        "\n"
        "        # Preserve upstream moe_backend branch — apply our backend reset\n"
        "        # ON TOP of the moe_backend-modified config.\n"
        "        base = self.vllm_config\n"
        "        if spec_cfg.moe_backend is not None:\n"
        "            base = replace(\n"
        "                base,\n"
        "                kernel_config=replace(\n"
        "                    base.kernel_config,\n"
        "                    moe_backend=spec_cfg.moe_backend,\n"
        "                ),\n"
        "            )\n"
        "\n"
        "        # [Genesis PN9] Reset drafter's attention backend.\n"
        "        # Baked from env GENESIS_PN9_DRAFTER_BACKEND at server start.\n"
        f"        _genesis_pn9_drafter_backend = {backend_literal}\n"
        "        try:\n"
        "            base = replace(\n"
        "                base,\n"
        "                attention_config=replace(\n"
        "                    base.attention_config,\n"
        "                    backend=_genesis_pn9_drafter_backend,\n"
        "                ),\n"
        "            )\n"
        "        except Exception as _genesis_pn9_exc:  # pragma: no cover\n"
        "            # If attention_config replace fails (older vllm without\n"
        "            # AttentionConfig.backend field, or pydantic rejects the\n"
        "            # value), fall through to the unmodified base config.\n"
        "            import logging as _genesis_pn9_logging\n"
        "            _genesis_pn9_logging.getLogger(\n"
        "                \"genesis.wiring.pN9\"\n"
        "            ).warning(\n"
        "                \"[Genesis PN9] attention_config replace failed: %s; \"\n"
        "                \"drafter will inherit target backend (vanilla path)\",\n"
        "                _genesis_pn9_exc,\n"
        "            )\n"
        "        return base\n"
    )


def _make_patcher(backend: str | None) -> TextPatcher | None:
    target = resolve_vllm_file("v1/spec_decode/llm_base_proposer.py")
    if target is None:
        return None
    return TextPatcher(
        patch_name=(
            "PN9 v1/spec_decode/llm_base_proposer.py — independent drafter "
            f"attention backend (vllm#39930) [backend={backend or 'auto'}]"
        ),
        target_file=str(target),
        marker=_marker_for(backend),
        sub_patches=[
            TextPatch(
                name="pN9_create_draft_vllm_config",
                anchor=PN9_OLD,
                replacement=_build_replacement(backend),
                required=True,
            ),
        ],
        upstream_drift_markers=[
            "[Genesis PN9",
            # Upstream PR #39930 specific markers — when these appear the
            # PR has merged into the pin we are running and we self-retire.
            "spec_cfg.attention_backend",
            "_create_draft_vllm_config",  # weak signal (always present); checked separately
        ],
    )


def apply() -> tuple[str, str]:
    """Apply PN9 — independent drafter attention backend (text-patch)."""
    from vllm._genesis.dispatcher import log_decision, should_apply

    decision, reason = should_apply("PN9")
    log_decision("PN9", decision, reason)
    if not decision:
        return "skipped", reason

    if vllm_install_root() is None:
        return "skipped", "vllm install root not discoverable"

    backend = _read_drafter_backend()
    patcher = _make_patcher(backend)
    if patcher is None:
        return "skipped", "vllm/v1/spec_decode/llm_base_proposer.py not found"

    if not os.path.isfile(patcher.target_file):
        return "skipped", f"target disappeared: {patcher.target_file}"
    with open(patcher.target_file) as f:
        content = f.read()

    # Idempotency check (current backend bake)
    if patcher.marker in content:
        log.info("[PN9] marker present (current backend) — skip (idempotent)")
        return "applied", "idempotent (marker present, backend unchanged)"

    # Stale-marker detection: a previous bake of PN9 with a different
    # backend value is already in the source. The original anchor has
    # been consumed; we cannot safely re-patch without resetting the
    # container fs. Surface clearly instead of silent skip.
    if GENESIS_PN9_MARKER_PREFIX in content:
        return (
            "skipped",
            f"PN9 stale marker present (different backend). Container fs has a "
            f"previous PN9 bake; current backend={backend or 'auto'} cannot be "
            f"applied without resetting the source. Reset via "
            f"`docker compose down && up -d` (NOT just stop/start).",
        )

    # Upstream drift: if PR #39930 lands, vanilla source contains a
    # reference to spec_cfg.attention_backend inside the patched function.
    # Detect by simple substring (cheap).
    if (
        "spec_cfg.attention_backend" in content
        and "[Genesis PN9" not in content
    ):
        return (
            "skipped",
            "upstream drift: 'spec_cfg.attention_backend' present in "
            "llm_base_proposer.py — PR #39930 (or equivalent) appears merged; "
            "PN9 self-retires (use --speculative-config.attention_backend instead)",
        )

    result, failure = patcher.apply()
    # Audit P1 fix 2026-05-05: surface SKIPPED as skipped (was masked as applied)
    if result == TextPatchResult.SKIPPED:
        _r = failure.reason if failure else "anchor drift / not eligible"
        _d = f" ({failure.detail})" if (failure and failure.detail) else ""
        return "skipped", f"{patcher.patch_name}: {_r}{_d}"
    if result == TextPatchResult.FAILED:
        return "failed", (
            f"{patcher.patch_name}: "
            f"{failure.reason if failure else 'unknown'} "
            f"({failure.detail if failure else ''})"
        )

    return (
        "applied",
        f"PN9 applied: drafter attention backend reset to "
        f"{backend or 'auto-select (None)'}. Drafter is now independent of "
        "target's backend choice. Unblocks DFlash + non-causal-required "
        "drafters. No effect when spec-decode is OFF."
    )


def is_applied() -> bool:
    """Return True iff our PN9 marker prefix is present in the target file."""
    if vllm_install_root() is None:
        return False
    target = resolve_vllm_file("v1/spec_decode/llm_base_proposer.py")
    if target is None:
        return False
    try:
        with open(target) as f:
            return GENESIS_PN9_MARKER_PREFIX in f.read()
    except Exception:
        return False
