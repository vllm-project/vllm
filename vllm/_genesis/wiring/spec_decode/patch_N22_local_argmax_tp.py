# SPDX-License-Identifier: Apache-2.0
"""Wiring for Patch N22 — Local argmax for TP draft (vllm#39419 backport).

Backport of [vllm#39419](https://github.com/vllm-project/vllm/pull/39419)
(EanWang, OPEN as of 2026-05-01). Adds `get_top_tokens()` plumbing methods
to Qwen3 + Qwen3-DFlash model classes, enabling vocab-parallel argmax
without all-gathering full logits across TP ranks.

================================================================
WHY THIS IS NEEDED
================================================================

Old draft path: each TP shard computes its slice of logits[batch,
vocab/tp_size], then all-gather assembles global_logits[batch, vocab],
then argmax. Communication = O(batch × vocab) per draft step. For
PARD-Qwen3-0.6B vocab=40960, batch=32, fp16 → 32×40960×2 = 2.5 MB per
step over PCIe Gen4 — ~1ms latency on dual 4090/A5000.

New draft path: each rank computes local argmax → (max_value, local_index).
Gather only pairs O(batch × 2 × tp_size) = ~1 KB. Reduce: global argmax
= argmax(max_values across ranks), global_index = local_index +
rank × shard_size. Bit-exact equivalent on identical shards.

Empirical (PR author): +9.4% to +30.6% throughput on TP=2 + draft model
(Qwen3-8B + DFlash, max-num-seqs=1, max-num-batched-tokens=8192).

================================================================
SCOPE
================================================================

Genesis backport covers our two production model classes:
- `qwen3.py` — main 27B / 35B-A3B
- `qwen3_dflash.py` — DFlash drafter

Llama (`llama.py`) and Eagle3 (`llama_eagle3.py`) parts of upstream PR are
NOT backported here — Genesis does not run those models in production.
If a user needs them, they can copy this pattern.

`LogitsProcessor.get_top_tokens()` itself already exists in our pin
(verified at line 106 of `vllm/model_executor/layers/logits_processor.py`).
The PR is pure plumbing — wiring model classes through to that method.

================================================================
SAFETY MODEL
================================================================

- env: `GENESIS_ENABLE_PN22_LOCAL_ARGMAX_TP=1`
- default OFF; opt-in.
- Idempotent (marker check)
- Falls through cleanly if anchor missed (SKIPPED, not crash).
- NO callsite swap: this patch only adds the methods. The proposer
  in `vllm/v1/spec_decode/llm_base_proposer.py` already calls
  `get_top_tokens()` when available (PR #34049 plumbing already merged
  in our pin). So enabling PN22 == enabling vocab-parallel argmax.
- Auto-no-op once vllm#39419 merges (drift markers).

Author: backport for Genesis from EanWang's vllm#39419.
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

log = logging.getLogger("genesis.wiring.pn22_local_argmax_tp")

GENESIS_PN22_MARKER = "Genesis PN22 local argmax TP for spec-decode draft v7.65"

# ─── Sub-patch 1: qwen3.py ──────────────────────────────────────────
PN22_QWEN3_ANCHOR = (
    "    def compute_logits(\n"
    "        self,\n"
    "        hidden_states: torch.Tensor,\n"
    "    ) -> torch.Tensor | None:\n"
    "        logits = self.logits_processor(self.lm_head, hidden_states)\n"
    "        return logits\n"
)

PN22_QWEN3_REPLACEMENT = (
    "    def compute_logits(\n"
    "        self,\n"
    "        hidden_states: torch.Tensor,\n"
    "    ) -> torch.Tensor | None:\n"
    "        logits = self.logits_processor(self.lm_head, hidden_states)\n"
    "        return logits\n"
    "\n"
    "    def get_top_tokens(\n"
    "        self,\n"
    "        hidden_states: torch.Tensor,\n"
    "    ) -> torch.Tensor:\n"
    "        # [Genesis PN22] vllm#39419 backport — vocab-parallel argmax,\n"
    "        # avoids all-gather of full logits across TP ranks.\n"
    "        # Wins +9-30% TPS on TP>=2 with draft models (DFlash/MTP).\n"
    "        return self.logits_processor.get_top_tokens(self.lm_head, hidden_states)\n"
)

# ─── Sub-patch 2: qwen3_dflash.py (DFlash draft) ─────────────────────
PN22_DFLASH_ANCHOR = (
    "        logits_new[:, targets] = logits\n"
    "        return logits_new\n"
    "\n"
    "    def precompute_and_store_context_kv(\n"
)

PN22_DFLASH_REPLACEMENT = (
    "        logits_new[:, targets] = logits\n"
    "        return logits_new\n"
    "\n"
    "    def get_top_tokens(\n"
    "        self,\n"
    "        hidden_states: torch.Tensor,\n"
    "    ) -> torch.Tensor:\n"
    "        # [Genesis PN22] vllm#39419 backport — vocab-parallel argmax for DFlash\n"
    "        # draft. Falls back to full logits when draft_id_to_target_id remap\n"
    "        # is active (draft predicts over draft_vocab_size, target expects\n"
    "        # target vocab ids — remap can't be done on local indices).\n"
    "        if self.draft_id_to_target_id is not None:\n"
    "            return self.compute_logits(hidden_states).argmax(dim=-1)\n"
    "        return self.logits_processor.get_top_tokens(self.lm_head, hidden_states)\n"
    "\n"
    "    def precompute_and_store_context_kv(\n"
)


def apply() -> tuple[str, str]:
    """Apply PN22 — vocab-parallel argmax plumbing (vllm#39419)."""
    from vllm._genesis.dispatcher import should_apply, log_decision
    decision, reason = should_apply("PN22")
    log_decision("PN22", decision, reason)
    if not decision:
        return "skipped", reason

    if vllm_install_root() is None:
        return "skipped", "vllm install root not discoverable"

    # Apply qwen3.py first (always present). Then qwen3_dflash.py (DFlash optional).
    qwen3 = resolve_vllm_file("model_executor/models/qwen3.py")
    if qwen3 is None or not os.path.isfile(str(qwen3)):
        return "skipped", "qwen3.py not found"

    patcher_qwen3 = TextPatcher(
        patch_name="PN22 qwen3.py — get_top_tokens (vllm#39419)",
        target_file=str(qwen3),
        marker=GENESIS_PN22_MARKER,
        sub_patches=[
            TextPatch(
                name="pn22_qwen3_get_top_tokens",
                anchor=PN22_QWEN3_ANCHOR,
                replacement=PN22_QWEN3_REPLACEMENT,
                required=True,
            ),
        ],
        upstream_drift_markers=[
            "[Genesis PN22]",
            "def get_top_tokens(",
        ],
    )
    # Audit G-POST-03 fix 2026-05-05 (genesis_post_fix_rescan_audit):
    # SKIPPED was being masked as final "applied" — surface it honestly.
    r1, f1 = patcher_qwen3.apply()
    if r1 == TextPatchResult.SKIPPED:
        _r = f1.reason if f1 else "anchor drift / not eligible"
        _d = f" ({f1.detail})" if (f1 and f1.detail) else ""
        return "skipped", f"qwen3.py: {_r}{_d}"
    if r1 == TextPatchResult.FAILED:
        return "failed", f"qwen3.py: {f1.reason if f1 else 'unknown'}"

    # Now qwen3_dflash.py (only present if DFlash supported)
    dflash = resolve_vllm_file("model_executor/models/qwen3_dflash.py")
    if dflash is not None and os.path.isfile(str(dflash)):
        patcher_dflash = TextPatcher(
            patch_name="PN22 qwen3_dflash.py — get_top_tokens (vllm#39419)",
            target_file=str(dflash),
            marker=GENESIS_PN22_MARKER + " (dflash)",
            sub_patches=[
                TextPatch(
                    name="pn22_dflash_get_top_tokens",
                    anchor=PN22_DFLASH_ANCHOR,
                    replacement=PN22_DFLASH_REPLACEMENT,
                    required=True,
                ),
            ],
            upstream_drift_markers=[
                "[Genesis PN22]",
                "def get_top_tokens(",
            ],
        )
        r2, f2 = patcher_dflash.apply()
        if r2 == TextPatchResult.SKIPPED:
            _r = f2.reason if f2 else "anchor drift / not eligible"
            _d = f" ({f2.detail})" if (f2 and f2.detail) else ""
            return "skipped", (
                f"qwen3_dflash.py: {_r}{_d} (qwen3.py applied, but DFlash "
                "subpatch skipped — re-apply needed for matching pair)"
            )
        if r2 == TextPatchResult.FAILED:
            return "failed", f"qwen3_dflash.py: {f2.reason if f2 else 'unknown'}"

    return "applied", (
        "PN22 applied: get_top_tokens() added to qwen3.py + qwen3_dflash.py "
        "(vllm#39419 backport). Enables vocab-parallel argmax in spec-decode "
        "draft path; +9-30% TPS on TP>=2 per PR author."
    )


def is_applied() -> bool:
    qwen3 = resolve_vllm_file("model_executor/models/qwen3.py")
    if qwen3 is None: return False
    try:
        with open(str(qwen3)) as f:
            return GENESIS_PN22_MARKER in f.read()
    except OSError:
        return False
