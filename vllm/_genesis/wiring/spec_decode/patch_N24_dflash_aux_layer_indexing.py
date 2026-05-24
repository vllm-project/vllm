# SPDX-License-Identifier: Apache-2.0
"""Wiring for Patch N24 — DFlash aux layer indexing off-by-one fix.

Backport of [vllm#40727](https://github.com/vllm-project/vllm/pull/40727)
(benchislett, OPEN as of 2026-05-01). One-line semantic fix in
`gpu_model_runner._get_eagle3_aux_layers_from_config` — adds `+1` to
DFlash's `target_layer_ids` to convert the DFlash aux-layer-id semantics
to the 1-indexed semantics expected downstream.

================================================================
WHY THIS IS NEEDED
================================================================

DFlash config stores `target_layer_ids` as 0-indexed layer numbers
(matching the model's internal layer indexing). The Eagle3 aux-layer
machinery in `gpu_model_runner` expects 1-indexed layer ids (because
layer 0 is the embedding lookup, not an attention layer). Without the
+1 shift, every aux hidden state is read from the WRONG layer —
specifically, layer N's output is grabbed when layer N+1 was intended.

Empirical impact (per PR description): DFlash + GSM8K acceptance length
improved from 6.18 → 6.42 with the fix. Small but measurable.

================================================================
WHEN IT FIRES
================================================================

- DFlash speculative decoding with `aux_hidden_state_layer_ids` in config
- Specifically the codepath `_get_eagle3_aux_layers_from_config`
- Falls back to original (no shift) if `dflash_config` is None or absent
  (e.g. running pure Eagle3 instead of DFlash)

================================================================
SAFETY MODEL
================================================================

- env: `GENESIS_ENABLE_PN24_DFLASH_AUX_LAYER_FIX=1`
- default OFF; opt-in
- Idempotent (marker check)
- Falls through cleanly if upstream renamed the function or shifted lines
  (anchor missing → SKIPPED, not crash).
- Auto-no-op once vllm#40727 merges (drift marker).

Author: backport for Genesis from benchislett's vllm#40727.
"""
from __future__ import annotations

import logging
import os

from vllm._genesis.guards import resolve_vllm_file, vllm_install_root
from vllm._genesis.wiring.text_patch import (
    TextPatch,
    TextPatcher,
    result_to_wiring_status,
)

log = logging.getLogger("genesis.wiring.pn24_dflash_aux_layer_indexing")

GENESIS_PN24_MARKER = "Genesis PN24 DFlash aux layer indexing +1 shift v7.65"

PN24_ANCHOR = (
    "        if not layer_ids:\n"
    "            dflash_config = getattr(hf_config, \"dflash_config\", None)\n"
    "            if dflash_config and isinstance(dflash_config, dict):\n"
    "                layer_ids = dflash_config.get(\"target_layer_ids\")\n"
)

PN24_REPLACEMENT = (
    "        if not layer_ids:\n"
    "            dflash_config = getattr(hf_config, \"dflash_config\", None)\n"
    "            if dflash_config and isinstance(dflash_config, dict):\n"
    "                # [Genesis PN24] vllm#40727 backport — DFlash target_layer_ids\n"
    "                # are 0-indexed; downstream Eagle3 aux machinery expects 1-indexed\n"
    "                # (layer 0 = embedding). Add 1 to convert semantics.\n"
    "                layer_ids = [i + 1 for i in dflash_config.get(\"target_layer_ids\", [])]\n"
)


def apply() -> tuple[str, str]:
    """Apply PN24 — DFlash aux layer +1 shift (vllm#40727)."""
    from vllm._genesis.dispatcher import should_apply, log_decision
    decision, reason = should_apply("PN24")
    log_decision("PN24", decision, reason)
    if not decision:
        return "skipped", reason

    if vllm_install_root() is None:
        return "skipped", "vllm install root not discoverable"

    target = resolve_vllm_file("v1/worker/gpu_model_runner.py")
    if target is None or not os.path.isfile(str(target)):
        return "skipped", "gpu_model_runner.py not found"

    patcher = TextPatcher(
        patch_name="PN24 gpu_model_runner.py — DFlash aux layer +1 shift (vllm#40727)",
        target_file=str(target),
        marker=GENESIS_PN24_MARKER,
        sub_patches=[
            TextPatch(
                name="pn24_aux_layer_shift",
                anchor=PN24_ANCHOR,
                replacement=PN24_REPLACEMENT,
                required=True,
            ),
        ],
        upstream_drift_markers=[
            "[Genesis PN24]",
            "i + 1 for i in dflash_config.get",
        ],
    )
    result, failure = patcher.apply()
    return result_to_wiring_status(
        result,
        failure,
        applied_message=(
            "PN24 applied: DFlash aux_layer_ids now shifted +1 in "
            "_get_eagle3_aux_layers_from_config (vllm#40727 backport). "
            "AL gsm8k 6.18→6.42 per PR author measurement."
        ),
        patch_name="PN24 DFlash aux layer indexing +1 shift",
    )


def is_applied() -> bool:
    target = resolve_vllm_file("v1/worker/gpu_model_runner.py")
    if target is None: return False
    try:
        with open(str(target)) as f:
            return GENESIS_PN24_MARKER in f.read()
    except OSError:
        return False
