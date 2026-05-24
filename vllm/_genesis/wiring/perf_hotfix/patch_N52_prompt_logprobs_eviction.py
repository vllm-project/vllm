# SPDX-License-Identifier: Apache-2.0
"""Wiring for PN52 — vllm#41411 backport: prompt_logprobs eviction fix.

Backport of upstream PR vllm-project/vllm#41411 (MERGED 2026-05-04 18:46
UTC by Joachim Studnia, Mistral). NOT in our pin (01d4d1ad3, ahead by 10
commits). Bug-fix; default OFF until live PROD verifies prompt_logprobs
+ chunked-prefill workload doesn't regress.

What this fixes
---------------
Two bugs in the v1 gpu_worker prompt_logprobs path:

1. **`prompt_logprob.py:61`** — overly aggressive `-1` in
   `includes_prompt = computed_prefill < prompt_lens - 1`. This skipped
   the LAST prompt token's logprob when chunked-prefill boundary fell
   exactly on `prompt_lens - 1`. Fix: drop the `-1`.

2. **Eviction-survival of accumulator** — `in_progress_prompt_logprobs_cpu`
   was stored on `input_batch` (per-batch dict). When a request was
   evicted during chunked prefill (e.g. memory pressure), the dict entry
   was discarded. On re-schedule, accumulation restarted from scratch
   while the prefill state showed `computed_prefill > 0` → silent
   corruption / IndexError on the wrong slice.

   Fix: move `in_progress_prompt_logprobs_cpu` to `CachedRequestState`
   (per-request, survives eviction).

Why we need it on Genesis
-------------------------
- All Genesis configs run with `--enable-chunked-prefill` (start scripts).
- 27B PROD runs MTP K=3 spec-decode (request preemption common under
  long-context burst).
- Open WebUI / LibreChat sometimes pass `prompt_logprobs=N` for token-
  weighting display; bug → silent broken display or 500.

Multi-file text patch
---------------------
PN52 mutates 3 files. Each sub-patch is `required=True`. As of v7.70 the
apply path uses `MultiFilePatchTransaction` (validate-all-then-write-all):
all 3 files dry-run first; if any required anchor is missing or has
already-applied marker drift, NO file is modified.

**Caveat (audit P1, genesis_deep_cross_audit_2026-05-05):** the current
`MultiFilePatchTransaction.dry_run()` does NOT check anchor uniqueness
(`content.count(anchor) == 1`) nor model sequential replacement, so a
file with a duplicated required anchor can pass dry-run and then return
SKIPPED at commit phase — leaving prior files modified. True rollback
is not yet implemented (only marker-based reverse-search best-effort).
Tracked for follow-up sprint; current rollback policy = log + skip.

Anchor stability
----------------
All anchors verified against pristine upstream (pre-#41411) AND our
live container at `01d4d1ad3` — both match. Once #41411 lands in our
next pin bump, anchors will drift → patch reports SKIPPED, not failure.

Author: Sandermage (Sander) Barzov Aleksandr backport.
Original fix: Joachim Studnia (Mistral), vllm#41411.
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

log = logging.getLogger("genesis.wiring.pn52_prompt_logprobs_eviction")

GENESIS_PN52_MARKER = "Genesis PN52 prompt_logprobs eviction fix v7.66 (vllm#41411 backport)"


def _is_enabled() -> bool:
    return os.environ.get(
        "GENESIS_ENABLE_PN52_PROMPT_LOGPROBS_EVICTION", ""
    ).strip().lower() in ("1", "true", "yes", "on")


# ─── Sub-patch 1: prompt_logprob.py — drop overly-aggressive -1 ─────────────
PROMPT_LOGPROB_OLD = (
    "        # NOTE(woosuk): -1 because the last prompt token's hidden state is not\n"
    "        # needed for prompt logprobs.\n"
    "        computed_prefill = num_computed_prefill_tokens[idx_mapping_np]\n"
    "        includes_prompt = computed_prefill < prompt_lens - 1"
)
PROMPT_LOGPROB_NEW = (
    "        # [Genesis PN52 vllm#41411] Drop overly-aggressive -1 — caused\n"
    "        # last prompt token's logprob to be skipped when chunked-prefill\n"
    "        # boundary fell exactly on prompt_lens - 1.\n"
    "        computed_prefill = num_computed_prefill_tokens[idx_mapping_np]\n"
    "        includes_prompt = computed_prefill < prompt_lens"
)


# ─── Sub-patch 2a: gpu_input_batch.py — add accumulator to CachedRequestState ─
INPUT_BATCH_FIELD_OLD = (
    "    lora_request: LoRARequest | None = None\n"
    "    prompt_embeds: torch.Tensor | None = None"
)
INPUT_BATCH_FIELD_NEW = (
    "    lora_request: LoRARequest | None = None\n"
    "    prompt_embeds: torch.Tensor | None = None\n"
    "    # [Genesis PN52 vllm#41411] Per-request accumulator for prompt logprobs\n"
    "    # tensor chunks across prefill steps. Survives request eviction; was\n"
    "    # previously stored on input_batch dict and lost on remove_request.\n"
    "    in_progress_prompt_logprobs_cpu: \"LogprobsTensors | None\" = None"
)

# ─── Sub-patch 2b: gpu_input_batch.py — remove old dict init ────────────────
INPUT_BATCH_INIT_OLD = (
    "        # To accumulate prompt logprobs tensor chunks across prefill steps.\n"
    "        self.in_progress_prompt_logprobs_cpu: dict[str, LogprobsTensors] = {}\n"
)
INPUT_BATCH_INIT_NEW = (
    "        # [Genesis PN52 vllm#41411] in_progress_prompt_logprobs_cpu moved\n"
    "        # to CachedRequestState (survives request eviction).\n"
)

# ─── Sub-patch 2c: gpu_input_batch.py — remove old dict pop on remove ───────
INPUT_BATCH_REMOVE_OLD = (
    "        self.logprob_token_ids.pop(req_id, None)\n"
    "        self.in_progress_prompt_logprobs_cpu.pop(req_id, None)\n"
)
INPUT_BATCH_REMOVE_NEW = (
    "        self.logprob_token_ids.pop(req_id, None)\n"
    "        # [Genesis PN52 vllm#41411] dict pop removed — accumulator now\n"
    "        # lives on CachedRequestState and survives eviction.\n"
)

# ─── Sub-patch 3a: gpu_model_runner.py — drop in_progress_dict alias ────────
RUNNER_DICT_ALIAS_OLD = (
    "        if not num_prompt_logprobs_dict:\n"
    "            return {}\n"
    "\n"
    "        in_progress_dict = self.input_batch.in_progress_prompt_logprobs_cpu\n"
    "        prompt_logprobs_dict: dict[str, LogprobsTensors | None] = {}"
)
RUNNER_DICT_ALIAS_NEW = (
    "        if not num_prompt_logprobs_dict:\n"
    "            return {}\n"
    "\n"
    "        # [Genesis PN52 vllm#41411] in_progress_dict alias removed —\n"
    "        # accumulator now read from request.in_progress_prompt_logprobs_cpu.\n"
    "        prompt_logprobs_dict: dict[str, LogprobsTensors | None] = {}"
)

# ─── Sub-patch 3b: gpu_model_runner.py — read/write per-request accumulator ─
RUNNER_GET_OLD = (
    "            # Set up target LogprobsTensors object.\n"
    "            logprobs_tensors = in_progress_dict.get(req_id)\n"
    "            if not logprobs_tensors:\n"
    "                # Create empty logprobs CPU tensors for the entire prompt.\n"
    "                # If chunked, we'll copy in slice by slice.\n"
    "                logprobs_tensors = LogprobsTensors.empty_cpu(\n"
    "                    num_prompt_tokens - 1, num_prompt_logprobs + 1\n"
    "                )\n"
    "                in_progress_dict[req_id] = logprobs_tensors"
)
RUNNER_GET_NEW = (
    "            # Set up target LogprobsTensors object.\n"
    "            # [Genesis PN52 vllm#41411] read accumulator from request,\n"
    "            # not from per-batch dict (eviction survival).\n"
    "            logprobs_tensors = request.in_progress_prompt_logprobs_cpu\n"
    "            if logprobs_tensors is None:\n"
    "                # Create empty logprobs CPU tensors for the entire prompt.\n"
    "                # If chunked, we'll copy in slice by slice.\n"
    "                logprobs_tensors = LogprobsTensors.empty_cpu(\n"
    "                    num_prompt_tokens - 1, num_prompt_logprobs + 1\n"
    "                )\n"
    "                request.in_progress_prompt_logprobs_cpu = logprobs_tensors"
)

# ─── Sub-patch 3c: gpu_model_runner.py — clear accumulator on prefill done ──
RUNNER_DEL_OLD = (
    "        for req_id in completed_prefill_reqs:\n"
    "            del num_prompt_logprobs_dict[req_id]\n"
    "            del in_progress_dict[req_id]"
)
RUNNER_DEL_NEW = (
    "        for req_id in completed_prefill_reqs:\n"
    "            del num_prompt_logprobs_dict[req_id]\n"
    "            # [Genesis PN52 vllm#41411] clear accumulator on request, not dict\n"
    "            self.requests[req_id].in_progress_prompt_logprobs_cpu = None"
)


def _make_patcher_prompt_logprob() -> TextPatcher | None:
    target = resolve_vllm_file("v1/worker/gpu/sample/prompt_logprob.py")
    if target is None:
        return None
    return TextPatcher(
        patch_name="PN52 prompt_logprob.py (drop -1)",
        target_file=str(target),
        marker=GENESIS_PN52_MARKER,
        sub_patches=[TextPatch(
            name="pn52_drop_minus1", anchor=PROMPT_LOGPROB_OLD,
            replacement=PROMPT_LOGPROB_NEW, required=True)],
    )


def _make_patcher_input_batch() -> TextPatcher | None:
    target = resolve_vllm_file("v1/worker/gpu_input_batch.py")
    if target is None:
        return None
    return TextPatcher(
        patch_name="PN52 gpu_input_batch.py (move accumulator)",
        target_file=str(target),
        marker=GENESIS_PN52_MARKER,
        sub_patches=[
            TextPatch(name="pn52_field", anchor=INPUT_BATCH_FIELD_OLD,
                      replacement=INPUT_BATCH_FIELD_NEW, required=True),
            TextPatch(name="pn52_init_remove", anchor=INPUT_BATCH_INIT_OLD,
                      replacement=INPUT_BATCH_INIT_NEW, required=True),
            TextPatch(name="pn52_remove_pop", anchor=INPUT_BATCH_REMOVE_OLD,
                      replacement=INPUT_BATCH_REMOVE_NEW, required=True),
        ],
    )


def _make_patcher_model_runner() -> TextPatcher | None:
    target = resolve_vllm_file("v1/worker/gpu_model_runner.py")
    if target is None:
        return None
    return TextPatcher(
        patch_name="PN52 gpu_model_runner.py (per-request accumulator)",
        target_file=str(target),
        marker=GENESIS_PN52_MARKER,
        sub_patches=[
            TextPatch(name="pn52_dict_alias", anchor=RUNNER_DICT_ALIAS_OLD,
                      replacement=RUNNER_DICT_ALIAS_NEW, required=True),
            TextPatch(name="pn52_get_set", anchor=RUNNER_GET_OLD,
                      replacement=RUNNER_GET_NEW, required=True),
            TextPatch(name="pn52_del", anchor=RUNNER_DEL_OLD,
                      replacement=RUNNER_DEL_NEW, required=True),
        ],
    )


def apply() -> tuple[str, str]:
    from vllm._genesis.dispatcher import log_decision, should_apply

    decision, reason = should_apply("PN52")
    log_decision("PN52", decision, reason)
    if not decision:
        return "skipped", reason

    if vllm_install_root() is None:
        return "skipped", "vllm install root not discoverable"

    # [Audit A-03/A-05 fix 2026-05-05] Use MultiFilePatchTransaction —
    # validate-all-then-write-all atomicity. Phase 1 dry-run catches
    # anchor drift in any file before any write happens.
    from vllm._genesis.wiring.text_patch import MultiFilePatchTransaction

    patchers = [
        _make_patcher_prompt_logprob(),
        _make_patcher_input_batch(),
        _make_patcher_model_runner(),
    ]
    txn = MultiFilePatchTransaction(patchers, name="PN52")
    return txn.apply_or_skip()
