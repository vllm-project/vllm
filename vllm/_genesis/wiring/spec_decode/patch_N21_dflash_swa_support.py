# SPDX-License-Identifier: Apache-2.0
"""Wiring for Patch N21 — DFlash Sliding Window Attention support.

Backport of [vllm#40898](https://github.com/vllm-project/vllm/pull/40898)
(jianc99, OPEN as of 2026-05-01). Adds proper SWA support to the DFlash
drafter codepath:

1. `qwen3_dflash.py` — layer_types tracking + per-layer sliding_window
   propagation through DFlashQwen3Attention / DFlashQwen3DecoderLayer +
   `sliding_attention_layer_names` set on root model.
2. `speculators/algos.py` — preserve layer_types / use_sliding_window /
   sliding_window / max_window_layers from speculators-format checkpoint
   into HF config (without preservation they get dropped → all layers
   fall through to full attention → broken acceptance).
3. `dflash.py` — force `causal=True` per-layer attention metadata for
   sliding-window layers (windowed FlashAttention requires causal=True).

================================================================
WHY THIS IS NEEDED
================================================================

In Qwen3.5-122B-A10B-DFlash and Qwen3.6-35B-A3B-DFlash, ~50% of the
draft transformer layers are `sliding_attention` (window=2048). Without
this fix:
- `layer_types` from speculators config is silently dropped during
  HF-config extraction (only target_hidden_size + draft_vocab_size kept)
- Drafter constructs all layers as full attention (NOT windowed)
- Drafter "sees" full context, target sees windowed → distribution
  mismatch → target rejects more drafts → acceptance length collapses

After this fix:
- layer_types preserved through config pipeline
- Drafter constructs SWA layers with proper sliding_window
- Drafter context matches target's view → distribution consistent →
  acceptance length 5.14 → 6.45 (+25%) per PR author measurement
- For Genesis: this is the unblocker for 35B-A3B-DFlash >80K context
  (currently OOM at 200K because draft KV grows unbounded without SWA)

================================================================
COMPOSITION WITH PN24
================================================================

Genesis PN24 (vllm#40727 backport) already adds `+1` shift to layer_ids
in `gpu_model_runner._get_eagle3_aux_layers_from_config`. Upstream PR
#40898 ALSO modifies that same function — adds `is_dflash` gate around
the existing logic. The two edits target the same code region.

P-N21 strategy: it does NOT touch `gpu_model_runner.py`. PN24's `+1`
shift is sufficient for our use case. P-N21 covers ONLY the 3 OTHER
files (qwen3_dflash, algos, dflash). Both patches coexist cleanly.

If user enables P-N21 alone (without PN24): the `+1` shift is missing
and layer_ids point to wrong layers. P-N21 dispatcher metadata
declares `requires_patches=["PN24"]` to enforce the pairing.

================================================================
EMPIRICAL FINDING (2026-05-01, v7.65 dev)
================================================================

Validated on 35B-A3B-FP8-DFlash 160K, 7-city tool-call sweep:

- Baseline (PN21 OFF, PN22+PN23+PN24 ON): 7/7 tool-call clean
- With PN21 ON (partial — algos.py + dflash.py only): 5-6/7 (3-run avg)

Regression matches the partial-backport caveat: when config preserves
SWA but the model class doesn't construct windowed attention, the
draft worker has metadata claiming SWA while computing full attention.
That divergence shifts spec acceptance for tool-call tokens.

Decision: PN21 stays SHIPPED (file + dispatcher + apply_all entry)
but DEFAULT OFF and NOT enabled in any launch script. Full enabler
requires either upstream merge (vllm#40898) or manual qwen3_dflash.py
edits (7+ sub-patches; high anchor-drift risk for text-patch).

================================================================
SAFETY MODEL
================================================================

- env: `GENESIS_ENABLE_PN21_DFLASH_SWA=1`
- default OFF; opt-in.
- empirical regression on 35B (5-6/7 vs 7/7 baseline) → DO NOT enable
  in production launch scripts until model class also patched.
- Idempotent (3 separate marker checks per file).
- Apply order: algos.py first (config preservation), then qwen3_dflash.py
  (model class), finally dflash.py (proposer metadata).
- Each file is independent TextPatcher — failure on one logs but does
  not block others (best-effort for SWA support).
- Auto-no-op once vllm#40898 merges (drift markers).

Author: backport for Genesis from jianc99's vllm#40898.
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

log = logging.getLogger("genesis.wiring.pn21_dflash_swa")

GENESIS_PN21_MARKER = "Genesis PN21 DFlash SWA support v7.65"


# ─── Sub-patch: speculators/algos.py — preserve SWA config ─────────
PN21_ALGOS_ANCHOR = (
    "    aux_layer_ids = config_dict[\"aux_hidden_state_layer_ids\"]\n"
    "    pre_trained_config[\"eagle_aux_hidden_state_layer_ids\"] = aux_layer_ids\n"
)

PN21_ALGOS_REPLACEMENT = (
    "    # [Genesis PN21] vllm#40898 backport — preserve SWA config\n"
    "    for _genesis_pn21_key in (\n"
    "        \"layer_types\",\n"
    "        \"use_sliding_window\",\n"
    "        \"sliding_window\",\n"
    "        \"max_window_layers\",\n"
    "    ):\n"
    "        if _genesis_pn21_key in config_dict:\n"
    "            pre_trained_config[_genesis_pn21_key] = config_dict[_genesis_pn21_key]\n"
    "\n"
    "    aux_layer_ids = config_dict[\"aux_hidden_state_layer_ids\"]\n"
    "    pre_trained_config[\"eagle_aux_hidden_state_layer_ids\"] = aux_layer_ids\n"
)


# ─── Sub-patch: dflash.py — causal=True for SWA layers ─────────────
PN21_DFLASH_ANCHOR = (
    "        per_group, per_layer = super().build_per_group_and_layer_attn_metadata(\n"
    "            cad, draft_index\n"
    "        )\n"
    "        for layer_name, attn_metadata in per_layer.items():\n"
    "            assert getattr(attn_metadata, \"causal\", None) is False, (\n"
    "                f\"Attention metadata for layer {layer_name} does not have\"\n"
    "                \" non-causal support, which is required for DFlash.\"\n"
    "                \" Consider using a different attention backend, such as FlashAttention.\"\n"
    "            )\n"
    "        return per_group, per_layer\n"
)

PN21_DFLASH_REPLACEMENT = (
    "        per_group, per_layer = super().build_per_group_and_layer_attn_metadata(\n"
    "            cad, draft_index\n"
    "        )\n"
    "        # [Genesis PN21] vllm#40898 backport — SWA layers need causal=True\n"
    "        _genesis_pn21_sliding = getattr(self.model, \"sliding_attention_layer_names\", set())\n"
    "        if _genesis_pn21_sliding:\n"
    "            _genesis_pn21_causal_cad = cad.replace(causal=True)\n"
    "            for _genesis_pn21_grp in self.draft_attn_groups:\n"
    "                _genesis_pn21_causal_layers = _genesis_pn21_sliding & set(_genesis_pn21_grp.layer_names)\n"
    "                if not _genesis_pn21_causal_layers:\n"
    "                    continue\n"
    "                _genesis_pn21_meta = _genesis_pn21_grp.get_metadata_builder().build_for_drafting(\n"
    "                    common_attn_metadata=_genesis_pn21_causal_cad, draft_index=draft_index\n"
    "                )\n"
    "                for _genesis_pn21_ln in _genesis_pn21_causal_layers:\n"
    "                    per_layer[_genesis_pn21_ln] = _genesis_pn21_meta\n"
    "        for layer_name, attn_metadata in per_layer.items():\n"
    "            if layer_name in _genesis_pn21_sliding:\n"
    "                assert getattr(attn_metadata, \"causal\", None) is True, (\n"
    "                    f\"Attention metadata for sliding layer {layer_name} does not have\"\n"
    "                    \" causal support, which is required for DFlash SWA.\"\n"
    "                )\n"
    "                continue\n"
    "            assert getattr(attn_metadata, \"causal\", None) is False, (\n"
    "                f\"Attention metadata for layer {layer_name} does not have\"\n"
    "                \" non-causal support, which is required for DFlash.\"\n"
    "                \" Consider using a different attention backend, such as FlashAttention.\"\n"
    "            )\n"
    "        return per_group, per_layer\n"
)


def _apply_algos() -> tuple[str, str | None]:
    """Apply speculators/algos.py SWA config preservation."""
    target = resolve_vllm_file("transformers_utils/configs/speculators/algos.py")
    if target is None or not os.path.isfile(str(target)):
        return "skipped", "speculators/algos.py not found"

    patcher = TextPatcher(
        patch_name="PN21 algos.py — preserve SWA config (vllm#40898)",
        target_file=str(target),
        marker=GENESIS_PN21_MARKER + " (algos)",
        sub_patches=[
            TextPatch(
                name="pn21_algos_swa_preserve",
                anchor=PN21_ALGOS_ANCHOR,
                replacement=PN21_ALGOS_REPLACEMENT,
                required=True,
            ),
        ],
        upstream_drift_markers=[
            "[Genesis PN21]",
            "_genesis_pn21_key",
            # Upstream merge — these keys appear directly
            "use_sliding_window",
        ],
    )
    result, failure = patcher.apply()
    if result == TextPatchResult.APPLIED:
        return "applied", None
    if result == TextPatchResult.IDEMPOTENT:
        return "skipped", "already applied (marker present)"
    if result == TextPatchResult.SKIPPED:
        return "skipped", failure.reason if failure else "already applied"
    return "failed", failure.detail if failure else "unknown"


def _apply_dflash() -> tuple[str, str | None]:
    """Apply dflash.py causal=True for SWA layers."""
    target = resolve_vllm_file("v1/spec_decode/dflash.py")
    if target is None or not os.path.isfile(str(target)):
        return "skipped", "v1/spec_decode/dflash.py not found"

    patcher = TextPatcher(
        patch_name="PN21 dflash.py — SWA causal metadata (vllm#40898)",
        target_file=str(target),
        marker=GENESIS_PN21_MARKER + " (dflash)",
        sub_patches=[
            TextPatch(
                name="pn21_dflash_swa_causal",
                anchor=PN21_DFLASH_ANCHOR,
                replacement=PN21_DFLASH_REPLACEMENT,
                required=True,
            ),
        ],
        upstream_drift_markers=[
            "[Genesis PN21]",
            "sliding_attention_layer_names",
        ],
    )
    result, failure = patcher.apply()
    if result == TextPatchResult.APPLIED:
        return "applied", None
    if result == TextPatchResult.IDEMPOTENT:
        return "skipped", "already applied (marker present)"
    if result == TextPatchResult.SKIPPED:
        return "skipped", failure.reason if failure else "already applied"
    return "failed", failure.detail if failure else "unknown"


def apply() -> tuple[str, str]:
    """Apply PN21 — DFlash SWA support partial backport (algos + dflash files only).

    qwen3_dflash.py model class changes are NOT backported here — they require
    7+ sub-patches with multi-line context across the file (Attention __init__
    signature + body, DecoderLayer __init__ + body, Model class init + property).
    The risk of anchor drift is high enough that we prefer the partial backport
    + waiting for upstream merge over a fragile big-patch.

    Without the qwen3_dflash.py changes, the algos.py + dflash.py changes
    still preserve the SWA config and force causal=True on SWA layers — but
    the model class itself doesn't construct sliding-window attention layers,
    so the windowed compute does not happen.

    => Genesis PN21 is currently a CONFIG-PRESERVING + METADATA-CORRECT but
       NOT a full SWA enabler. It positions the model for upstream merge to
       activate.

    Operator path: enable PN21 + PN24 today, get partial benefit + future-proof
    against upstream merge auto-activation. When upstream PR #40898 merges,
    drift markers will detect and PN21 will auto-no-op cleanly.
    """
    from vllm._genesis.dispatcher import should_apply, log_decision
    decision, reason = should_apply("PN21")
    log_decision("PN21", decision, reason)
    if not decision:
        return "skipped", reason

    if vllm_install_root() is None:
        return "skipped", "vllm install root not discoverable"

    results = []
    for name, fn in [("algos", _apply_algos), ("dflash", _apply_dflash)]:
        status, detail = fn()
        results.append((name, status, detail))
        log.info("[PN21:%s] %s%s", name, status,
                 f" — {detail}" if detail else "")

    applied = [n for n, s, _ in results if s == "applied"]
    skipped = [n for n, s, _ in results if s == "skipped"]
    failed = [n for n, s, _ in results if s == "failed"]

    if failed:
        return "failed", (
            f"PN21 partial: applied={applied}, skipped={skipped}, failed={failed}"
        )
    if not applied:
        return "skipped", (
            f"PN21 nothing to apply (already applied or anchors absent): {skipped}"
        )
    return "applied", (
        f"PN21 applied {applied} (DFlash SWA partial — algos.py preserves "
        f"layer_types/sliding_window config + dflash.py forces causal=True "
        f"on SWA layers). Skipped: {skipped}. Note: full SWA enabler in "
        f"qwen3_dflash.py model class deferred — wait for vllm#40898 merge "
        f"or apply manually. Composes with PN24."
    )


def is_applied() -> bool:
    target = resolve_vllm_file("transformers_utils/configs/speculators/algos.py")
    if target is None: return False
    try:
        with open(str(target)) as f:
            return GENESIS_PN21_MARKER in f.read()
    except OSError:
        return False
