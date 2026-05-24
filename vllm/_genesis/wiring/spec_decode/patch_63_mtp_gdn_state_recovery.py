# SPDX-License-Identifier: Apache-2.0
"""Wiring for Patch 63 — MTP/Eagle drafter forward GDN state recovery.

Genesis-original investigation (no upstream PR yet at time of writing).

================================================================
DEPENDS ON P60 (Phase 1) being applied. P60 fixes the MAIN model's
post-verify decode path. P63 covers the same class of bug for the
DRAFTER forward path that MTP/Eagle methods take.
================================================================

Bug class
---------
For hybrid GDN models (Qwen3-Next family) running with MTP/Eagle
spec-decode:

  1. Step N main verify runs `GDNAttentionMetadataBuilder.build()` with
     `num_accepted_tokens` from gpu_model_runner. Spec branch fires (line
     400-404 upstream): cached buffer `self.num_accepted_tokens[:batch]`
     is populated.
  2. Sampling computes which draft tokens were accepted.
  3. Drafter forward fires via `proposer.propose()` → calls
     `build_per_group_and_layer_attn_metadata()` → calls
     `attn_group.get_metadata_builder().build_for_drafting()`.
  4. The base `AttentionMetadataBuilder.build_for_drafting()` impl
     (`vllm/v1/attention/backend.py:611-631`) defaults to
     `self.build(common_prefix_len=0, common_attn_metadata=cad,
     fast_build=True)` — **WITHOUT** `num_accepted_tokens`.
  5. Inside GDN's `build()`, the non-spec branch sees
     `num_accepted_tokens is None` → P60's recovery logic guard
     (`if self.use_spec_decode and num_accepted_tokens is not None
     and num_decodes > 0`) fails → `spec_decode_src_indices = None`.
  6. In `_forward_core` (gdn_linear_attn.py), the SSM state pre-copy
     guard sees `spec_decode_src_indices is None` → no recovery happens
     → drafter reads SSM state from block[0] which is stale (still
     contains state from before the spec round).
  7. Drafter generates corrupt next-token drafts; eventually the
     pollution leaks into target model's KV cache on the next step.

Symptoms (reproduced by @noonghunna's Probe 9, 2026-04-25):
  - Tool calls empty (`tool_calls: []` in API response)
  - First-token truncation visible at ~10K context
  - Only when MTP n=3 + cudagraph ON + TurboQuant KV — separately
    correlated with `prompt_lookup_min=2` ngram bug class but
    DIFFERENT root cause path

Why P60 doesn't cover this
--------------------------
- PR #40738 (tdoublep) fixes `gpu_model_runner._build_attn_group_metadata`
  to pass `num_accepted_tokens` to the builder for MAIN model non-spec
  steps. That code path is `gpu_model_runner.execute_model` → `_build_attn_group_metadata`.
- The DRAFTER forward path bypasses gpu_model_runner. It goes:
  `gpu_model_runner.execute_model` → sample → `drafter.propose()` →
  `build_per_group_and_layer_attn_metadata()` → `build_for_drafting()`.
- The `build_for_drafting()` route doesn't have the gpu_model_runner
  passthrough fix.

P63 fix design
--------------
Two sub-patches in `vllm/v1/attention/backends/gdn_attn.py`:

**Sub-patch A** — Mark cache fresh after spec branch populates it.
After the existing `num_accepted_tokens[num_spec_decodes:].fill_(1)`
line in the spec branch, set `self._p63_num_accepted_fresh = True` so
`build_for_drafting()` knows the buffer holds valid current-step
acceptance counts.

**Sub-patch B** — Insert a `build_for_drafting()` override on
`GDNAttentionMetadataBuilder` that:
  - Reads cached `num_accepted_tokens` from `self.num_accepted_tokens`
    buffer IF the fresh flag is set
  - Calls `self.build()` with that num_accepted (engaging P60's
    spec_decode_src_indices computation + SSM state pre-copy)
  - Resets the fresh flag after consume (one-shot semantics)
  - Falls back to base behavior (no num_accepted) if fresh flag is
    unset — happens on cold start or when no prior verify ran

Critical invariants
-------------------
1. **P60 must be applied first.** P63 relies on P60's
   `spec_decode_src_indices` field on the metadata + `build()`
   recovery logic in the non-spec branch. Without P60, P63 is a no-op
   (passes num_accepted but build() doesn't compute the recovery).
2. **One-shot consume.** Each fresh flag set is consumed by ONE
   build_for_drafting call. Subsequent calls fall back to base
   behavior until the next spec verify refreshes the buffer.
3. **No cross-step contamination.** If a non-spec build runs between
   a spec verify and the drafter forward (unusual but possible in
   pipelined async modes), the cache MAY be stale. The fresh flag
   protects against this — only set in spec branch.

Risk assessment
---------------
- **No prod test rig for MTP** at our setup (Sander's prod uses
  ngram, not MTP). P63 cannot be empirically verified on our hardware.
  Validation paths:
    1. @noonghunna's Probe 9 rig (1× 3090 + Qwen3.6-27B int4 + MTP n=3)
       — he can apply P63 and re-run his MTP test
    2. Gemini-side / Qwen team CI if they pick up the fix upstream
- **Race condition timing.** Concern: between spec verify build and
  drafter forward, does any OTHER build call mutate `self.num_accepted_tokens`?
  Analysis of code path says NO: gpu_model_runner.execute_model
  finishes main forward, calls drafter.propose(). Drafter's
  build_for_drafting fires before any new main verify build. So the
  cache should be fresh at consume time.
- **Mode interaction with cudagraph.** GDN `build_for_cudagraph_capture`
  also calls `self.build()` (without num_accepted). Capture-time call
  shouldn't engage P63 because spec branch hasn't run yet. The
  fresh-flag default is False, so capture uses base path. ✓
- **Override vs base.** P63 changes `build_for_drafting` from
  `self.build(common_prefix_len=0, common_attn_metadata=cad, fast_build=True)`
  to `self.build(common_prefix_len=0, common_attn_metadata=cad,
  num_accepted_tokens=cached_or_None, fast_build=True)`. When cached
  is None, behavior is identical to base (modulo the kwarg's signature).

Compatibility / drift markers
-----------------------------
- Idempotent (marker check)
- Auto-no-op once upstream merges equivalent fix (drift marker on
  the new method's `_p63_num_accepted_fresh` token; if upstream calls
  it `_drafter_state_recovery_fresh` or similar, our text-patch will
  cleanly skip via marker miss)

Status: opt-in via `GENESIS_ENABLE_P63_MTP_GDN_STATE_RECOVERY=1`.
Default OFF until validated on a real MTP test rig. Operators with
MTP setups should manually enable + re-test.

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

log = logging.getLogger("genesis.wiring.p63_mtp_gdn_state_recovery")

GENESIS_P63_MARKER = "Genesis P63 MTP/Eagle drafter GDN state recovery v7.13"


# ─── Sub-patch A: mark fresh flag after spec branch populates buffer ────────
# Anchor targets the closing of the spec branch in build() — specifically
# the line that fills num_accepted_tokens with 1 for non-spec sequences.
# This region is NOT touched by P60, so the anchor is stable post-P60.

GDN_FRESH_FLAG_OLD = (
    "            self.num_accepted_tokens[:num_spec_decodes].copy_(\n"
    "                num_accepted_tokens, non_blocking=True\n"
    "            )\n"
    "            num_accepted_tokens = self.num_accepted_tokens[:batch_size]\n"
    "            num_accepted_tokens[num_spec_decodes:].fill_(1)\n"
)

GDN_FRESH_FLAG_NEW = (
    "            self.num_accepted_tokens[:num_spec_decodes].copy_(\n"
    "                num_accepted_tokens, non_blocking=True\n"
    "            )\n"
    "            num_accepted_tokens = self.num_accepted_tokens[:batch_size]\n"
    "            num_accepted_tokens[num_spec_decodes:].fill_(1)\n"
    "            # [Genesis P63 MTP-GDN-recovery] cache is fresh — next\n"
    "            # build_for_drafting() may consume it for state recovery.\n"
    "            self._p63_num_accepted_fresh = True\n"
)


# ─── Sub-patch B: insert build_for_drafting() override ──────────────────────
# Anchor targets the start of build_for_cudagraph_capture (stable, not
# touched by P60). We insert the new method right before it.

GDN_DRAFTING_OVERRIDE_OLD = (
    "    def build_for_cudagraph_capture(\n"
    "        self, common_attn_metadata: CommonAttentionMetadata\n"
    "    ):\n"
)

GDN_DRAFTING_OVERRIDE_NEW = (
    "    def build_for_drafting(\n"
    "        self,\n"
    "        common_attn_metadata: CommonAttentionMetadata,\n"
    "        draft_index: int,\n"
    "    ) -> GDNAttentionMetadata:\n"
    "        \"\"\"[Genesis P63] Override base build_for_drafting to pass cached\n"
    "        num_accepted_tokens through to build(). Without this, MTP/Eagle\n"
    "        drafter forwards drop into the non-spec branch with\n"
    "        num_accepted=None → P60's spec_decode_src_indices stays None →\n"
    "        SSM state pre-copy in gdn_linear_attn._forward_core never fires →\n"
    "        drafter reads stale SSM state from block[0] → corrupt drafts.\n"
    "\n"
    "        We use the builder's own self.num_accepted_tokens buffer, which\n"
    "        was populated in the spec branch of the most recent build() call\n"
    "        (see Sub-patch A: _p63_num_accepted_fresh flag). The flag is\n"
    "        consumed once per drafter call to avoid cross-step staleness.\n"
    "\n"
    "        Depends on P60 being applied (build() must accept num_accepted\n"
    "        kwarg + compute spec_decode_src_indices in non-spec branch).\n"
    "        \"\"\"\n"
    "        # [Genesis P63 trace] log invocation for diagnostic tracing\n"
    "        import logging as _p63_logging\n"
    "        _p63_logging.getLogger(\"genesis.p63\").debug(\n"
    "            \"build_for_drafting fired: draft_index=%d num_reqs=%d fresh=%s\",\n"
    "            draft_index, common_attn_metadata.num_reqs,\n"
    "            getattr(self, \"_p63_num_accepted_fresh\", False))\n"
    "        # [Genesis P63b cudagraph-aware] ALWAYS pass a num_accepted tensor\n"
    "        # even at warm-up / cold start, so the cudagraph capture INCLUDES\n"
    "        # the spec_decode_src_indices recovery branch (otherwise the\n"
    "        # captured graph has spec_decode_src_indices=None baked in and\n"
    "        # the recovery branch never fires at replay time).\n"
    "        # When fresh cache exists → use real values (real recovery).\n"
    "        # When no cache → synthetic all-1 (= identity copy = no-op,\n"
    "        # but the recovery KERNEL is still captured for replay).\n"
    "        import torch as _p63_torch\n"
    "        batch_size = common_attn_metadata.num_reqs\n"
    "        if getattr(self, \"_p63_num_accepted_fresh\", False):\n"
    "            cached_n = self.num_accepted_tokens[:batch_size]\n"
    "            # consume-once: prevent stale carry-over to next call\n"
    "            self._p63_num_accepted_fresh = False\n"
    "        else:\n"
    "            # Synthetic all-1: makes spec_decode_src_indices = block[:,0]\n"
    "            # which equals non_spec_state_indices_tensor (identity copy)\n"
    "            cached_n = _p63_torch.ones(\n"
    "                batch_size,\n"
    "                dtype=self.num_accepted_tokens.dtype,\n"
    "                device=self.num_accepted_tokens.device,\n"
    "            )\n"
    "        return self.build(\n"
    "            common_prefix_len=0,\n"
    "            common_attn_metadata=common_attn_metadata,\n"
    "            num_accepted_tokens=cached_n,\n"
    "            fast_build=True,\n"
    "        )\n"
    "\n"
    "    def build_for_cudagraph_capture(\n"
    "        self, common_attn_metadata: CommonAttentionMetadata\n"
    "    ):\n"
)


def _make_patcher() -> TextPatcher | None:
    target = resolve_vllm_file("v1/attention/backends/gdn_attn.py")
    if target is None:
        return None
    return TextPatcher(
        patch_name="P63 gdn_attn.py — MTP/Eagle drafter state recovery",
        target_file=str(target),
        marker=GENESIS_P63_MARKER,
        sub_patches=[
            TextPatch(
                name="p63_fresh_flag",
                anchor=GDN_FRESH_FLAG_OLD,
                replacement=GDN_FRESH_FLAG_NEW,
                required=True,
            ),
            TextPatch(
                name="p63_build_for_drafting_override",
                anchor=GDN_DRAFTING_OVERRIDE_OLD,
                replacement=GDN_DRAFTING_OVERRIDE_NEW,
                required=True,
            ),
        ],
        # Drift markers — distinctive Genesis-only phrasing. If upstream
        # adds an equivalent override they'll likely use a different name
        # for the cache flag (e.g., `_drafter_recovery_fresh`); our marker
        # is `_p63_num_accepted_fresh` which is Genesis-unique.
        upstream_drift_markers=[
            "MTP/Eagle drafter forward GDN state recovery",
            "_p63_num_accepted_fresh",
        ],
    )


def apply() -> tuple[str, str]:
    """Apply P63 (MTP/Eagle drafter GDN state recovery) text-patch."""
    from vllm._genesis.dispatcher import should_apply, log_decision
    decision, reason = should_apply("P63")
    log_decision("P63", decision, reason)
    if not decision:
        return "skipped", reason

    if vllm_install_root() is None:
        return "skipped", "vllm install root not discoverable"

    patcher = _make_patcher()
    if patcher is None:
        return "skipped", "gdn_attn.py not found"

    # Pre-flight: confirm anchors before write
    if not os.path.isfile(patcher.target_file):
        return "skipped", f"target disappeared: {patcher.target_file}"
    with open(patcher.target_file) as f:
        content = f.read()
    if patcher.marker in content:
        # idempotent
        pass
    else:
        for m in patcher.upstream_drift_markers:
            if m in content:
                return (
                    "skipped",
                    f"upstream drift marker {m!r} in {patcher.target_file} — "
                    "MTP-GDN drafter state recovery likely already merged or "
                    "backported by another patch.",
                )
        for sp in patcher.sub_patches:
            if sp.required and sp.anchor not in content:
                return (
                    "skipped",
                    f"required anchor for {sp.name!r} not found in "
                    f"{patcher.target_file} — anchor drifted, P63 cannot apply. "
                    "Most likely cause: P60 not yet applied (P63 depends on "
                    "P60's spec_decode_src_indices logic in build()).",
                )

    result, failure = patcher.apply()
    if result == TextPatchResult.FAILED:
        return "failed", (
            f"{patcher.patch_name}: {failure.reason if failure else 'unknown'} "
            f"({failure.detail if failure else ''}) — partial state risk; "
            "container should be torn down (compose down + up -d)."
        )
    if result == TextPatchResult.SKIPPED:
        return "skipped", "anchor drift on sub-patch — no changes written"

    return "applied", (
        "P63 applied: GDNAttentionMetadataBuilder.build_for_drafting() now "
        "passes cached num_accepted_tokens through to build(), engaging P60's "
        "SSM state recovery for MTP/Eagle drafter forward path. Requires P60 "
        "+ P60b for full correctness."
    )
