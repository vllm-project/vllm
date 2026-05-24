# SPDX-License-Identifier: Apache-2.0
"""Wiring for Patch N8 — MTP / draft model online-quant propagation.

================================================================
Source PR
================================================================
https://github.com/vllm-project/vllm/pull/40849
"[Spec Decode] Inherit online quantization for draft models"
by @bhoomit (Bhoomit), OPEN as of 2026-04-29.

================================================================
WHAT IT DOES (one paragraph)
================================================================

When a target model is loaded with online quantization (e.g.
`--quantization fp8_per_tensor`), the spec-decode draft model
(Eagle / Eagle3 / Medusa / MTP-as-external-draft) currently loads in
BF16 because `get_draft_quant_config()` only consults the draft model's
own quantization field — which is None for any draft that doesn't ship
quant metadata in its checkpoint. Result: a 530M Eagle3 draft uses
1.45 GiB BF16 instead of 0.88 GiB FP8, wasting ~600 MiB that could
otherwise feed KV cache. PR #40849 modifies `get_draft_quant_config()`
in `vllm/model_executor/models/utils.py` so that, when no draft-side
quant config is found, it inherits the target's `OnlineQuantizationConfig`
directly. The fallback path also catches `ValueError`/`FileNotFoundError`
from `get_quantization_config()` (which crashes for online-quant methods
because `hf_overrides` is a callable, not a dict).

For Genesis: Lorbus/Minachist 27B-INT4 prod targets do not currently
run online-quant + external-draft, so this patch is INFORMATIONAL for
prod. Direct value lands when we deploy Eagle3 / DFlash drafters with
an FP8 target — frees ~600 MiB of KV cache headroom on the worker.

================================================================
SAFETY MODEL
================================================================

- Default OFF (opt-in via `GENESIS_ENABLE_PN8_MTP_DRAFT_ONLINE_QUANT=1`).
- Text-patch on `vllm/model_executor/models/utils.py` —
  `get_draft_quant_config()` body.
- Idempotent via marker `Genesis PN8 ...`.
- Drift-aware: if upstream merges PR #40849 (or any equivalent), the
  marker `OnlineQuantizationConfig` import in this exact site is
  detected and apply() returns SKIPPED with a "self-retired" reason.
- Anchor missing → SKIPPED, source stays vanilla. No behavioral
  regression possible because the original return path is preserved
  when the new fallback doesn't fire.
- Worst-case regression: if the inherited `OnlineQuantizationConfig`
  somehow rejects the draft model's tensor shapes at load time, the
  draft load will raise. To recover: set `GENESIS_ENABLE_PN8...=0` and
  restart — vanilla path resumes.

Predicates / activation
-----------------------
The patch is *unconditional* once the env flag is set. The runtime
predicate (spec method == "mtp"/"qwen3_next_mtp"/"eagle"/etc. AND main
model has online-quant config) is enforced naturally by the patched
function — when those conditions are not met, the new branch falls
through to the original `return None` path, identical to upstream.

We do NOT additionally gate at apply() time on detected spec method
because:
  (a) `model_detect` does not currently expose spec_method;
  (b) the patch is a strict superset of upstream behavior (every
      vanilla code path is preserved); a wrong-context apply is a no-op.

Author backport: Sandermage(Sander) Barzov Aleksandr, Ukraine, Odessa.
Source PR: vllm-project/vllm#40849 by @bhoomit.
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

log = logging.getLogger("genesis.wiring.pN8_mtp_draft_online_quant_propagation")

GENESIS_PN8_MARKER = (
    "Genesis PN8 MTP/draft online-quant propagation (vllm#40849) v7.62.x"
)


# ─── Sub-patch 1: imports — add OnlineQuantizationConfig ─────────────────

PN8_IMPORTS_OLD = (
    "from vllm.model_executor.layers.quantization.base_config import (\n"
    "    QuantizationConfig,\n"
    ")\n"
)

PN8_IMPORTS_NEW = (
    "from vllm.model_executor.layers.quantization.base_config import (\n"
    "    QuantizationConfig,\n"
    ")\n"
    "# [Genesis PN8 vllm#40849 backport] OnlineQuantizationConfig import\n"
    "# Wrapped in try/except — older vllm builds without the online subpkg\n"
    "# silently fall back to vanilla behavior (no inheritance).\n"
    "try:\n"
    "    from vllm.model_executor.layers.quantization.online.base import (\n"
    "        OnlineQuantizationConfig as _GENESIS_PN8_OnlineQuantizationConfig,\n"
    "    )\n"
    "except Exception:  # pragma: no cover - module shape variance across vllm builds\n"
    "    _GENESIS_PN8_OnlineQuantizationConfig = None\n"
)


# ─── Sub-patch 2: replace get_draft_quant_config() body ──────────────────

PN8_BODY_OLD = (
    "    draft_model_config = vllm_config.speculative_config.draft_model_config\n"
    "    draft_load_config = vllm_config.load_config\n"
    "\n"
    "    return (\n"
    "        VllmConfig.get_quantization_config(draft_model_config, draft_load_config)\n"
    "        if draft_model_config\n"
    "        else None\n"
    "    )\n"
)

PN8_BODY_NEW = (
    "    # ════════════════════════════════════════════════════════════════\n"
    "    # [Genesis PN8 vllm#40849 backport] online-quant propagation\n"
    "    # If the draft model has its own quantization, use it (orig path).\n"
    "    # Else, inherit the target model's OnlineQuantizationConfig so the\n"
    "    # draft loads at the same precision (frees ~600 MiB on FP8 + Eagle3).\n"
    "    # ════════════════════════════════════════════════════════════════\n"
    "    draft_model_config = vllm_config.speculative_config.draft_model_config\n"
    "    if not draft_model_config:\n"
    "        return None\n"
    "\n"
    "    draft_load_config = vllm_config.load_config\n"
    "\n"
    "    # Path A: draft has explicit quantization → use it.\n"
    "    if getattr(draft_model_config, \"quantization\", None):\n"
    "        try:\n"
    "            return VllmConfig.get_quantization_config(\n"
    "                draft_model_config, draft_load_config\n"
    "            )\n"
    "        except (ValueError, FileNotFoundError) as _genesis_pn8_exc:\n"
    "            # Online-quant methods crash through checkpoint config path\n"
    "            # because hf_overrides is a callable. Fall through to inherit.\n"
    "            try:\n"
    "                logger.warning(\n"
    "                    \"[Genesis PN8] Draft model has quantization=%s but \"\n"
    "                    \"get_quantization_config failed: %s. \"\n"
    "                    \"Falling back to target model's quant config.\",\n"
    "                    getattr(draft_model_config, \"quantization\", None),\n"
    "                    _genesis_pn8_exc,\n"
    "                )\n"
    "            except Exception:\n"
    "                pass\n"
    "\n"
    "    # Path B: inherit target's OnlineQuantizationConfig if present.\n"
    "    if (\n"
    "        _GENESIS_PN8_OnlineQuantizationConfig is not None\n"
    "        and isinstance(\n"
    "            getattr(vllm_config, \"quant_config\", None),\n"
    "            _GENESIS_PN8_OnlineQuantizationConfig,\n"
    "        )\n"
    "    ):\n"
    "        return vllm_config.quant_config\n"
    "\n"
    "    return None\n"
)


def _make_patcher() -> TextPatcher | None:
    target = resolve_vllm_file("model_executor/models/utils.py")
    if target is None:
        return None
    return TextPatcher(
        patch_name=(
            "PN8 model_executor/models/utils.py — get_draft_quant_config "
            "online-quant propagation (vllm#40849)"
        ),
        target_file=str(target),
        marker=GENESIS_PN8_MARKER,
        sub_patches=[
            TextPatch(
                name="pN8_imports",
                anchor=PN8_IMPORTS_OLD,
                replacement=PN8_IMPORTS_NEW,
                required=True,
            ),
            TextPatch(
                name="pN8_body",
                anchor=PN8_BODY_OLD,
                replacement=PN8_BODY_NEW,
                required=True,
            ),
        ],
        upstream_drift_markers=[
            "[Genesis PN8",
            # If upstream PR #40849 (or any equivalent) merges, these
            # marker strings will appear in vanilla source and we self-retire.
            "_GENESIS_PN8_OnlineQuantizationConfig",
            "OnlineQuantizationConfig",
        ],
    )


def apply() -> tuple[str, str]:
    """Apply PN8 — MTP/draft online-quant propagation (text-patch)."""
    from vllm._genesis.dispatcher import log_decision, should_apply

    decision, reason = should_apply("PN8")
    log_decision("PN8", decision, reason)
    if not decision:
        return "skipped", reason

    if vllm_install_root() is None:
        return "skipped", "vllm install root not discoverable"

    patcher = _make_patcher()
    if patcher is None:
        return "skipped", "vllm/model_executor/models/utils.py not found"

    if not os.path.isfile(patcher.target_file):
        return "skipped", f"target disappeared: {patcher.target_file}"
    with open(patcher.target_file) as f:
        content = f.read()
    if patcher.marker in content:
        log.info("[PN8] marker present — skip (idempotent)")
        return "applied", "idempotent (marker present)"
    # Drift detection: if upstream's own import of OnlineQuantizationConfig
    # appears, PR #40849 (or equivalent) has merged — self-retire.
    if (
        "from vllm.model_executor.layers.quantization.online.base import" in content
        and "OnlineQuantizationConfig" in content
        and "[Genesis PN8" not in content
    ):
        return (
            "skipped",
            "upstream drift: OnlineQuantizationConfig already imported in "
            "utils.py — PR #40849 (or equivalent) appears merged; PN8 self-retires",
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
        "PN8 applied: get_draft_quant_config() now inherits target's "
        "OnlineQuantizationConfig when draft has no explicit quant. "
        "Frees ~600 MiB on FP8-target + external-draft (Eagle3/DFlash/MTP). "
        "No-op when target is not online-quantized."
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
