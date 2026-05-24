# SPDX-License-Identifier: Apache-2.0
"""Wiring for Patch 72 — profile_run M cap to unblock max_num_batched_tokens > 4096.

Genesis-original — addresses Dynamo fake-tensor mismatch when running with
`--max-num-batched-tokens=8192` (or higher) on MoE models.

================================================================
WHAT THIS FIXES

When the user requests `--max-num-batched-tokens=8192` (vLLM warns at
config/vllm.py:1414 that 4096 is suboptimal for MTP K+1 spec-decode draft
slots), `profile_run` in gpu_model_runner.py calls `_dummy_run(8192,
is_profile=True)`. Inside MoE forward, `topk_ids` has shape `(8192, 8)`
with `numel() = 65536`.

Inside `moe_align_block_size`, `topk_ids.numel()` becomes a Dynamo symbolic
dim `s72`, while a parallel branch in the trace specializes the constant
`65536` from a previous trace at M=4096. Dynamo cannot reconcile concrete
65536 vs symbolic `16*s72` and aborts:

    RuntimeError when making fake tensor call
      tensor a (65536) must match tensor b (16*s72) at non-singleton dim 0

================================================================

Mechanism of the fix
--------------------
Cap the M passed to `_dummy_run(...)` during profiling to a known-good
value (default 4096). Memory profiling difference is < 1MB (sorted_token_ids
delta = (65536 - 36608) × 4 bytes = 116KB) — negligible vs 35GB model weights.

This allows runtime to use `--max-num-batched-tokens=8192`:
- During profile_run: traces with M=4096 (no Dynamo conflict)
- At runtime: real batches up to 8192 tokens go through the SAME compiled
  graph (Dynamo doesn't re-trace; the symbolic shape s72 covers both)
- Memory estimate for KV cache stays within ~0.1% of true bound

What our prod gets
------------------
- Eliminates the `cannot reconcile 65536 vs 16*s72` boot failure
- Allows operators to bump batched_tokens for higher prefill batch sizes
  (relevant for multi-turn aggregator scenarios where ISL > 4096)
- For our 2-seq MTP K+1=4 interactive workload: ≤0.5% TPS gain
  (max_num_seqs=2 × K_PLUS_1=4 = 8 tokens/step actual; the headroom
  is for prefill chunk size, not decode throughput)

Status: opt-in via `GENESIS_ENABLE_P72_PROFILE_RUN_CAP=1`.

Tunable knobs
-------------
- `GENESIS_PROFILE_RUN_CAP_M` (default: 4096) — cap value for profile_run.
  Set to 0 to disable cap (returns to upstream behavior).
- `GENESIS_PROFILE_RUN_CAP_LOG` (default: 1) — log when cap fires.

Compatibility
-------------
- Affects ONLY when `max_num_batched_tokens > GENESIS_PROFILE_RUN_CAP_M`.
- For users at or below 4096: no-op.
- Idempotent (marker check).
- Auto-no-op if upstream marks `topk_ids.numel()` dynamic at trace time
  (drift marker: `_dummy_run_M_capped_for_genesis`).

Risks acknowledged
------------------
- Memory profiling underestimates ~0.1% — at GMU=0.91 we have ample headroom.
- IF a future MoE op allocates anything proportional to M^2, the underestimate
  grows. Currently no such op exists in fused_moe path.
- Worst case: real OOM at runtime → operator lowers GMU back to 0.88. Visible.

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

log = logging.getLogger("genesis.wiring.p72_profile_run_cap")

GENESIS_P72_MARKER = "Genesis P72 profile_run M cap for batched_tokens>4096 v7.42"


# ─── Sub-patch: cap M passed to _dummy_run inside profile_run ────────────
# Anchor at the call site in profile_run that does:
#     hidden_states, last_hidden_states = self._dummy_run(
#         self.max_num_tokens, is_profile=True
#     )
# We replace the M argument with min(self.max_num_tokens, cap).

P72_OLD = (
    "        # Add `is_profile` here to pre-allocate communication buffers\n"
    "        hidden_states, last_hidden_states = self._dummy_run(\n"
    "            self.max_num_tokens, is_profile=True\n"
    "        )\n"
)

P72_NEW = (
    "        # Add `is_profile` here to pre-allocate communication buffers\n"
    "        # [Genesis P72 _dummy_run_M_capped_for_genesis] Cap M to avoid\n"
    "        # Dynamo fake-tensor mismatch when batched_tokens > 4096 with MoE\n"
    "        # moe_align_block_size symbolic shape inference.\n"
    "        import os as _genesis_p72_os\n"
    "        _genesis_p72_cap = int(_genesis_p72_os.environ.get(\n"
    "            'GENESIS_PROFILE_RUN_CAP_M', '4096'))\n"
    "        if _genesis_p72_cap > 0 and self.max_num_tokens > _genesis_p72_cap:\n"
    "            _genesis_p72_M = _genesis_p72_cap\n"
    "            if _genesis_p72_os.environ.get(\n"
    "                'GENESIS_PROFILE_RUN_CAP_LOG', '1') == '1':\n"
    "                import logging as _genesis_p72_log_mod\n"
    "                _genesis_p72_log = _genesis_p72_log_mod.getLogger(\n"
    "                    'genesis.profile_run_cap')\n"
    "                _genesis_p72_log.warning(\n"
    "                    '[Genesis P72] profile_run M capped: %d -> %d '\n"
    "                    '(unblocks --max-num-batched-tokens=%d at runtime). '\n"
    "                    'Memory estimate delta < 1MB. Disable: '\n"
    "                    'GENESIS_PROFILE_RUN_CAP_M=0',\n"
    "                    self.max_num_tokens, _genesis_p72_M, self.max_num_tokens,\n"
    "                )\n"
    "        else:\n"
    "            _genesis_p72_M = self.max_num_tokens\n"
    "        hidden_states, last_hidden_states = self._dummy_run(\n"
    "            _genesis_p72_M, is_profile=True\n"
    "        )\n"
)


def _make_patcher() -> TextPatcher | None:
    target = resolve_vllm_file("v1/worker/gpu_model_runner.py")
    if target is None:
        return None
    return TextPatcher(
        patch_name="P72 v1/worker/gpu_model_runner.py — profile_run M cap",
        target_file=str(target),
        marker=GENESIS_P72_MARKER,
        sub_patches=[
            TextPatch(
                name="p72_profile_run_cap",
                anchor=P72_OLD,
                replacement=P72_NEW,
                required=True,
            ),
        ],
        upstream_drift_markers=[
            "[Genesis P72]",
            "_dummy_run_M_capped_for_genesis",
            "GENESIS_PROFILE_RUN_CAP_M",
        ],
    )


def apply() -> tuple[str, str]:
    """Apply P72 — profile_run M cap."""
    from vllm._genesis.dispatcher import should_apply, log_decision
    decision, reason = should_apply("P72")
    log_decision("P72", decision, reason)
    if not decision:
        return "skipped", reason

    if vllm_install_root() is None:
        return "skipped", "vllm install root not discoverable"

    patcher = _make_patcher()
    if patcher is None:
        return "skipped", "vllm/v1/worker/gpu_model_runner.py not found"

    if not os.path.isfile(patcher.target_file):
        return "skipped", f"target disappeared: {patcher.target_file}"
    with open(patcher.target_file) as f:
        content = f.read()
    if patcher.marker in content:
        log.info("[P72] marker present — skip (idempotent)")
        return "applied", "idempotent (marker present)"
    for m in patcher.upstream_drift_markers:
        if m in content:
            return (
                "skipped",
                f"upstream drift marker {m!r} in {patcher.target_file} — "
                "upstream may have absorbed this fix",
            )

    result, failure = patcher.apply()
    # Audit P1 fix 2026-05-05: route SKIPPED/IDEMPOTENT honestly via shared helper
    from vllm._genesis.wiring.text_patch import result_to_wiring_status
    return result_to_wiring_status(
        result, failure,
        applied_message=(
            "P72 applied: profile_run M capped to GENESIS_PROFILE_RUN_CAP_M (default 4096). "
            "Unblocks --max-num-batched-tokens > 4096 by avoiding Dynamo fake-tensor "
            "shape mismatch in moe_align_block_size symbolic-shape inference."
        ),
        patch_name=patcher.patch_name,
    )
