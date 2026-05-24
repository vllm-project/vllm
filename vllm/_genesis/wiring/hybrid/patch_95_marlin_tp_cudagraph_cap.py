# SPDX-License-Identifier: Apache-2.0
"""Wiring for Patch 95 — Marlin TP cudagraph cap on Ampere.

Backport of [vllm#40385](https://github.com/vllm-project/vllm/pull/40385)
("compilation: cap cudagraph sizes for TP Marlin on Ampere"). PR is OPEN
upstream as of 2026-04-28; we backport defensively because the bug
(#40121 "illegal memory access during CUDA graph replay") matches our
exact hardware profile (A5000 sm86, TP=2, Lorbus INT4 + Minachist gs128
both routing to Marlin).

================================================================
WHAT THIS PATCH DOES
================================================================

In `vllm/config/vllm.py::_set_cudagraph_sizes`, after the initial
`max_cudagraph_capture_size` is computed, add a defensive cap to 8 IF:

  - user did NOT set `cudagraph_capture_sizes` (auto-derived path)
  - user did NOT set `max_cudagraph_capture_size` (auto-derived path)
  - `tensor_parallel_size > 1`
  - CUDA + Ampere (sm_80 family — covers SM 8.0 / 8.6 / 8.9)
  - `model_config.quantization` ends with `"_marlin"`

Effect: prevents `cudaErrorIllegalAddress` during CG replay on Ampere
TP>1 Marlin. Operator can override with `--compilation-config` (matches
the user-set guard above).

================================================================
WHO IS HIT IN OUR FLEET
================================================================

PROD v759 (Qwen3.6-35B-A3B-FP8) — quantization='fp8', NOT '_marlin' →
patch is NO-OP for prod. Safe to enable.

Test variants:
- v770 INT8 main (Minachist W8A16 group_size=-1) → AllSpark, NOT Marlin → NO-OP
- v771 INT4 Lorbus (W4A16 group_size=128) → Marlin path → **CAP APPLIES**
- v764f gs128 (W8A16 group_size=128 + force-Marlin via VLLM_DISABLED_KERNELS)
  → Marlin path → **CAP APPLIES**

The cap is conservative — `max_cudagraph_capture_size=8` matches the
known-stable workaround. If we measure regression on Lorbus or gs128 at
batch>8, override with `--compilation-config '{"max_cudagraph_capture_size": 32}'`.

================================================================
SAFETY MODEL
================================================================

- Default OFF; opt-in via `GENESIS_ENABLE_P95=1`
- Idempotent via marker
- Drift detection: 2 anchor checks (line before insertion + line after)
- When inactive (env unset / user override / non-Marlin), behaviour
  is upstream byte-for-byte
- Pure additive — does not delete or rewrite any existing logic

Author backport: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
Original PR: vllm#40385 (OPEN). Issue: vllm#40121.
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

log = logging.getLogger("genesis.wiring.p95_marlin_tp_cudagraph_cap")


GENESIS_P95_MARKER = (
    "Genesis P95 Marlin TP cudagraph cap on Ampere (vllm#40385) v7.62.12"
)


# ─── Sub-patch 1: import current_platform inside _set_cudagraph_sizes ───
# We anchor on the surrounding context (the `if` condition that gates
# the whole block). Keep enough lines to be unique against drift.

P95_IMPORT_OLD = (
    "            and not self.model_config.enforce_eager\n"
    "            and self.compilation_config.cudagraph_mode != CUDAGraphMode.NONE\n"
    "        ):\n"
    "            # determine the initial max_cudagraph_capture_size\n"
)

P95_IMPORT_NEW = (
    "            and not self.model_config.enforce_eager\n"
    "            and self.compilation_config.cudagraph_mode != CUDAGraphMode.NONE\n"
    "        ):\n"
    "            # [Genesis P95 vllm#40385 backport] need current_platform here\n"
    "            from vllm.platforms import current_platform as _genesis_p95_platform\n"
    "\n"
    "            # determine the initial max_cudagraph_capture_size\n"
)


# ─── Sub-patch 2: insert the Marlin cap right before the assert ─────────
# Anchor is the assert line — we insert the cap logic immediately before.

P95_CAP_OLD = (
    "            max_num_tokens = self.scheduler_config.max_num_batched_tokens\n"
    "            max_cudagraph_capture_size = min(max_num_tokens, max_cudagraph_capture_size)\n"
    "\n"
    "            assert max_cudagraph_capture_size >= 1, (\n"
)

P95_CAP_NEW = (
    "            max_num_tokens = self.scheduler_config.max_num_batched_tokens\n"
    "            max_cudagraph_capture_size = min(max_num_tokens, max_cudagraph_capture_size)\n"
    "\n"
    "            # ════════════════════════════════════════════════════════════════\n"
    "            # [Genesis P95 vllm#40385 backport] Mitigation for illegal memory\n"
    "            # access during CUDA graph replay in TP>1 + Marlin-quantized configs\n"
    "            # on Ampere GPUs (issue vllm#40121). Caps to 8 ONLY when the user\n"
    "            # has not specified their own cudagraph sizing — explicit operator\n"
    "            # overrides via --compilation-config bypass this entirely.\n"
    "            # ════════════════════════════════════════════════════════════════\n"
    "            if (\n"
    "                self.compilation_config.cudagraph_capture_sizes is None\n"
    "                and self.compilation_config.max_cudagraph_capture_size is None\n"
    "                and self.parallel_config.tensor_parallel_size > 1\n"
    "                and _genesis_p95_platform.is_cuda()\n"
    "                and _genesis_p95_platform.is_device_capability_family(80)\n"
    "            ):\n"
    "                _genesis_p95_quant = getattr(self.model_config, 'quantization', None)\n"
    "                if isinstance(_genesis_p95_quant, str) and _genesis_p95_quant.endswith('_marlin'):\n"
    "                    if max_cudagraph_capture_size > 8:\n"
    "                        logger.warning_once(\n"
    "                            '[Genesis P95] Capping max_cudagraph_capture_size to 8 '\n"
    "                            'for TP>1 with %s on Ampere GPUs (mitigates vllm#40121 '\n"
    "                            'illegal memory access). Override with '\n"
    "                            '--compilation-config if needed.',\n"
    "                            _genesis_p95_quant,\n"
    "                        )\n"
    "                    max_cudagraph_capture_size = min(max_cudagraph_capture_size, 8)\n"
    "\n"
    "            assert max_cudagraph_capture_size >= 1, (\n"
)


def _make_patcher() -> TextPatcher | None:
    target = resolve_vllm_file("config/vllm.py")
    if target is None:
        return None
    return TextPatcher(
        patch_name="P95 vllm/config/vllm.py — Marlin TP cudagraph cap (vllm#40385)",
        target_file=str(target),
        marker=GENESIS_P95_MARKER,
        sub_patches=[
            TextPatch(
                name="p95_import_current_platform",
                anchor=P95_IMPORT_OLD,
                replacement=P95_IMPORT_NEW,
                required=True,
            ),
            TextPatch(
                name="p95_marlin_cap_block",
                anchor=P95_CAP_OLD,
                replacement=P95_CAP_NEW,
                required=True,
            ),
        ],
        upstream_drift_markers=[
            "[Genesis P95",
            # Upstream-side markers if PR #40385 (or equivalent) merges:
            "_genesis_p95_platform",
            "Capping max_cudagraph_capture_size to 8 for TP",
            "mitigate potential CUDA graph replay",
        ],
    )


def apply() -> tuple[str, str]:
    """Apply P95 — Marlin TP cudagraph cap on Ampere."""
    from vllm._genesis.dispatcher import log_decision, should_apply

    decision, reason = should_apply("P95")
    log_decision("P95", decision, reason)
    if not decision:
        return "skipped", reason

    if vllm_install_root() is None:
        return "skipped", "vllm install root not discoverable"

    patcher = _make_patcher()
    if patcher is None:
        return "skipped", "config/vllm.py not found"

    if not os.path.isfile(patcher.target_file):
        return "skipped", f"target disappeared: {patcher.target_file}"
    with open(patcher.target_file) as f:
        content = f.read()
    if patcher.marker in content:
        log.info("[P95] marker present — skip (idempotent)")
        return "applied", "idempotent (marker present)"
    for m in patcher.upstream_drift_markers:
        if m.startswith("[Genesis"):
            continue
        if m in content:
            return (
                "skipped",
                f"upstream drift marker {m!r} in {patcher.target_file} "
                "— upstream PR #40385 (or equivalent) appears merged",
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
        "P95 v7.62.12 applied: vllm/config/vllm.py _set_cudagraph_sizes now "
        "auto-caps max_cudagraph_capture_size to 8 when TP>1 + Marlin + Ampere "
        "(SM 8.0 family) AND user has not explicitly sized cudagraphs. NO-OP for "
        "non-Marlin quants (FP8 PROD unaffected)."
    )


def is_applied() -> bool:
    """Return True iff our marker is present in the target file."""
    if vllm_install_root() is None:
        return False
    patcher = _make_patcher()
    if patcher is None:
        return False
    try:
        with open(patcher.target_file) as f:
            return patcher.marker in f.read()
    except Exception:
        return False
