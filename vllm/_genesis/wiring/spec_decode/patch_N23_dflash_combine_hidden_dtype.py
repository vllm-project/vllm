# SPDX-License-Identifier: Apache-2.0
"""Wiring for Patch N23 — DFlash combine_hidden_states dtype mismatch fix.

Backport of [vllm#40334](https://github.com/vllm-project/vllm/pull/40334)
(ciphernaut, OPEN as of 2026-05-01). Six-line defensive cast added to
`Qwen3DFlashModel.combine_hidden_states` to cast hidden_states to
`self.model.fc.params_dtype` before passing to the FC layer.

================================================================
WHY THIS IS NEEDED
================================================================

DFlash uses an auxiliary FC layer to combine hidden states from multiple
target layers into a single representation for drafting. Under mixed
precision targets (e.g. AWQ-quantized attention but BF16/FP32 outputs
for some layers), `hidden_states.dtype` may not match `fc.params_dtype`,
causing `RuntimeError: expected scalar type X but found Y`.

The fix: explicit `.to(params_dtype)` cast before the `fc()` call.
`params_dtype` is the intended compute dtype; `weight.dtype` may be a
packed integer type under quantization.

================================================================
WHEN IT FIRES
================================================================

- DFlash speculative decoding (`method=dflash`)
- Mixed-precision target (AWQ + non-quantized layers, FP8 + BF16 mix, etc)
- Specifically the `combine_hidden_states` codepath

================================================================
SAFETY MODEL
================================================================

- env: `GENESIS_ENABLE_PN23_DFLASH_DTYPE_FIX=1`
- default OFF; opt-in.
- Idempotent (marker check)
- Falls through cleanly if `combine_hidden_states` was renamed upstream
  (anchor missing → SKIPPED, not crash).
- Auto-no-op once vllm#40334 merges.

Author: backport for Genesis from ciphernaut's vllm#40334.
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

log = logging.getLogger("genesis.wiring.pn23_dflash_combine_hidden_dtype")

GENESIS_PN23_MARKER = "Genesis PN23 DFlash combine_hidden_states dtype cast v7.65"

PN23_ANCHOR = (
    "        needs_squeeze = hidden_states.dim() == 1\n"
    "        if needs_squeeze:\n"
    "            hidden_states = hidden_states.unsqueeze(0)\n"
    "        result = self.model.fc(hidden_states)\n"
)

PN23_REPLACEMENT = (
    "        needs_squeeze = hidden_states.dim() == 1\n"
    "        if needs_squeeze:\n"
    "            hidden_states = hidden_states.unsqueeze(0)\n"
    "        # [Genesis PN23] vllm#40334 backport — cast to fc params_dtype to\n"
    "        # handle mixed-precision targets (AWQ + non-quantized, FP8 + BF16 mix).\n"
    "        # params_dtype is the intended compute dtype; weight.dtype may be a\n"
    "        # packed integer type under quantization.\n"
    "        if hidden_states.dtype != self.model.fc.params_dtype:\n"
    "            hidden_states = hidden_states.to(self.model.fc.params_dtype)\n"
    "        result = self.model.fc(hidden_states)\n"
)


def apply() -> tuple[str, str]:
    """Apply PN23 — DFlash combine_hidden_states dtype cast (vllm#40334)."""
    from vllm._genesis.dispatcher import should_apply, log_decision
    decision, reason = should_apply("PN23")
    log_decision("PN23", decision, reason)
    if not decision:
        return "skipped", reason

    if vllm_install_root() is None:
        return "skipped", "vllm install root not discoverable"

    target = resolve_vllm_file("model_executor/models/qwen3_dflash.py")
    if target is None or not os.path.isfile(str(target)):
        return "skipped", "qwen3_dflash.py not found (DFlash not in this vllm pin?)"

    patcher = TextPatcher(
        patch_name="PN23 qwen3_dflash.py — combine_hidden_states dtype cast (vllm#40334)",
        target_file=str(target),
        marker=GENESIS_PN23_MARKER,
        sub_patches=[
            TextPatch(
                name="pn23_dtype_cast",
                anchor=PN23_ANCHOR,
                replacement=PN23_REPLACEMENT,
                required=True,
            ),
        ],
        upstream_drift_markers=[
            "[Genesis PN23]",
            # If vllm#40334 lands as-is, the upstream replacement will also
            # contain `params_dtype` near this anchor — auto-no-op.
            "self.model.fc.params_dtype",
        ],
    )
    result, failure = patcher.apply()
    return result_to_wiring_status(
        result,
        failure,
        applied_message=(
            "PN23 applied: DFlash combine_hidden_states now casts hidden_states "
            "to fc.params_dtype before FC call (vllm#40334 backport). Fixes "
            "RuntimeError on mixed-precision DFlash configs (AWQ + non-quant)."
        ),
        patch_name="PN23 DFlash combine_hidden_states dtype cast",
    )


def is_applied() -> bool:
    target = resolve_vllm_file("model_executor/models/qwen3_dflash.py")
    if target is None: return False
    try:
        with open(str(target)) as f:
            return GENESIS_PN23_MARKER in f.read()
    except OSError:
        return False
