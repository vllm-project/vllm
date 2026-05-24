# SPDX-License-Identifier: Apache-2.0
"""Wiring for Patch 74 — auto chunk-clamp for prefill chunks (companion to P72).

Genesis-original — addresses the prealloc buffer overflow class bug exposed
by P72 (`--max-num-batched-tokens > 4096` on MoE).

================================================================
WHY THIS EXISTS
================================================================

When P72 unlocks `--max-num-batched-tokens=8192`, the chunked-prefill
scheduler is free to dispatch prefill chunks up to 8192 tokens. But our
prealloc patches (P28 GDN core_attn_out, P26 TQ prefill output, P44 TQ
mixed-batch attn_out) sized their pools at 4096 → buffer overflow on
long-context (180K) requests:

    RuntimeError: setStorage: sizes [5664, 16, 128] requiring 23199744 bytes
    out of bounds for storage of size 16777216 (= 4096 × 16 × 128 × 2)

vLLM has a built-in mechanism to cap prefill chunk size:
`SchedulerConfig.long_prefill_token_threshold`. When > 0, the scheduler
limits any prefill chunk to this value (see
`vllm/v1/core/sched/scheduler.py:420-421`):

    if 0 < self.scheduler_config.long_prefill_token_threshold < num_new_tokens:
        num_new_tokens = self.scheduler_config.long_prefill_token_threshold

P74 auto-sets this threshold to the resolved Genesis P73 token budget
when the operator hasn't set it explicitly — providing a zero-VRAM-cost
safety net against prealloc overflow.

================================================================
RESOLUTION ORDER
================================================================

1. If `long_prefill_token_threshold > 0` is already set by user/upstream,
   honor it (no override).
2. Otherwise, use `GENESIS_PREALLOC_TOKEN_BUDGET` env (preferred).
3. Otherwise, default to 4096 (preserves prealloc safety for legacy patches).

Bonus: also caps `max_long_partial_prefills` interaction so single 8192
batch can still pack many small decodes (multi-seq parallelism preserved).

================================================================
PERF IMPACT
================================================================

For our typical 2-seq interactive workload:
- Decode steps: still up to `max_num_batched_tokens` (8192) — multi-seq
  packing works as designed.
- Prefill chunks: capped to threshold (4096 default) — adds ~1 extra
  scheduler step per long-context (180K) request. Throughput cost ≈ 0.

For multi-seq throughput-heavy workload:
- Decode parallelism preserved.
- Long-context concurrent prefills serialized at threshold granularity.

Status: opt-in via `GENESIS_ENABLE_P74_CHUNK_CLAMP=1`. Recommended ON
when `GENESIS_ENABLE_P72_PROFILE_RUN_CAP=1` AND
`--max-num-batched-tokens > 4096`.

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

log = logging.getLogger("genesis.wiring.p74_chunk_clamp")

GENESIS_P74_MARKER = "Genesis P74 auto chunk-clamp via long_prefill_token_threshold v7.42"


# ─── Sub-patch: inject auto-clamp BEFORE the existing partial_prefills branch ─
# Anchor on `if self.max_num_partial_prefills > 1:` block which is unique
# inside SchedulerConfig.__post_init__.

P74_OLD = (
    "        if self.enable_chunked_prefill:\n"
    "            logger.info_once(\n"
    "                \"Chunked prefill is enabled with max_num_batched_tokens=%d.\",\n"
    "                self.max_num_batched_tokens,\n"
    "            )\n"
    "\n"
    "        if self.max_num_partial_prefills > 1:\n"
)

P74_NEW = (
    "        if self.enable_chunked_prefill:\n"
    "            logger.info_once(\n"
    "                \"Chunked prefill is enabled with max_num_batched_tokens=%d.\",\n"
    "                self.max_num_batched_tokens,\n"
    "            )\n"
    "\n"
    "        # ════════════════════════════════════════════════════════════════\n"
    "        # [Genesis P74 v7.42] Auto chunk-clamp companion to P72 unblock.\n"
    "        # When prealloc patches (P28/P26/P44) are sized at smaller budget\n"
    "        # than max_num_batched_tokens, scheduler dispatching a larger\n"
    "        # prefill chunk would overflow them. Cap the threshold to the\n"
    "        # resolved Genesis prealloc budget if user didn't set it explicitly.\n"
    "        # Decode steps (which respect total max_num_batched_tokens) are\n"
    "        # unaffected — only prefill chunks get clamped.\n"
    "        # ════════════════════════════════════════════════════════════════\n"
    "        try:\n"
    "            import os as _genesis_p74_os\n"
    "            if _genesis_p74_os.environ.get(\n"
    "                'GENESIS_ENABLE_P74_CHUNK_CLAMP', '').strip().lower() in (\n"
    "                '1', 'true', 'yes', 'on'):\n"
    "                if self.long_prefill_token_threshold == 0:\n"
    "                    # User didn't set explicit threshold — auto-clamp to budget\n"
    "                    _genesis_p74_budget = int(_genesis_p74_os.environ.get(\n"
    "                        'GENESIS_PREALLOC_TOKEN_BUDGET',\n"
    "                        _genesis_p74_os.environ.get(\n"
    "                            'GENESIS_GDN_MAX_BATCHED_TOKENS', '4096')))\n"
    "                    if (\n"
    "                        _genesis_p74_budget > 0\n"
    "                        and _genesis_p74_budget < self.max_num_batched_tokens\n"
    "                    ):\n"
    "                        self.long_prefill_token_threshold = _genesis_p74_budget\n"
    "                        logger.info(\n"
    "                            '[Genesis P74] Auto-clamped long_prefill_token_threshold '\n"
    "                            'to %d (vs max_num_batched_tokens=%d) to keep prefill '\n"
    "                            'chunks within prealloc budgets (P28/P26/P44). '\n"
    "                            'Decode batches still up to %d. Disable via '\n"
    "                            'GENESIS_ENABLE_P74_CHUNK_CLAMP=0 or set explicit '\n"
    "                            '--long-prefill-token-threshold.',\n"
    "                            _genesis_p74_budget,\n"
    "                            self.max_num_batched_tokens,\n"
    "                            self.max_num_batched_tokens,\n"
    "                        )\n"
    "        except Exception as _genesis_p74_err:\n"
    "            logger.warning(\n"
    "                '[Genesis P74] auto-clamp failed (%s); upstream behavior preserved.',\n"
    "                _genesis_p74_err,\n"
    "            )\n"
    "\n"
    "        if self.max_num_partial_prefills > 1:\n"
)


def _make_patcher() -> TextPatcher | None:
    target = resolve_vllm_file("config/scheduler.py")
    if target is None:
        return None
    return TextPatcher(
        patch_name="P74 config/scheduler.py — auto chunk-clamp via long_prefill_token_threshold",
        target_file=str(target),
        marker=GENESIS_P74_MARKER,
        sub_patches=[
            TextPatch(
                name="p74_auto_clamp",
                anchor=P74_OLD,
                replacement=P74_NEW,
                required=True,
            ),
        ],
        upstream_drift_markers=[
            "[Genesis P74]",
            "GENESIS_ENABLE_P74_CHUNK_CLAMP",
        ],
    )


def apply() -> tuple[str, str]:
    """Apply P74 — auto chunk-clamp."""
    from vllm._genesis.dispatcher import should_apply, log_decision
    decision, reason = should_apply("P74")
    log_decision("P74", decision, reason)
    if not decision:
        return "skipped", reason

    if vllm_install_root() is None:
        return "skipped", "vllm install root not discoverable"

    patcher = _make_patcher()
    if patcher is None:
        return "skipped", "vllm/config/scheduler.py not found"

    if not os.path.isfile(patcher.target_file):
        return "skipped", f"target disappeared: {patcher.target_file}"
    with open(patcher.target_file) as f:
        content = f.read()
    if patcher.marker in content:
        log.info("[P74] marker present — skip (idempotent)")
        return "applied", "idempotent (marker present)"
    for m in patcher.upstream_drift_markers:
        if m == "[Genesis P74]" and m in content:
            continue
        if m in content:
            return (
                "skipped",
                f"upstream drift marker {m!r} in {patcher.target_file} — "
                "upstream may have absorbed this fix",
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
        "P74 applied: SchedulerConfig auto-clamps long_prefill_token_threshold "
        "to GENESIS_PREALLOC_TOKEN_BUDGET when batched_tokens > budget. "
        "Prevents prealloc buffer overflow when running with batched=8192."
    )
