# SPDX-License-Identifier: Apache-2.0
"""Wiring for Patch N14 — TurboQuant decode IOOB clamp via safe_page_idx.

================================================================
Source PR
================================================================
https://github.com/vllm-project/vllm/pull/40074
"[Bugfix] Fix TurboQuant KV cache index-out-of-bounds in Triton decode kernel"
by @devarakondasrikanth (Adobe), OPEN as of 2026-04-29.

Fixes upstream issue #39998 — the same OOB assertion class that
@jhsmith409 confirmed "applies cleanly" while stacking on top of #39931.

================================================================
WHAT IT DOES
================================================================

`_tq_decode_stage1` in `vllm/v1/attention/ops/triton_turboquant_decode.py`
computes `page_idx = kv_offs // BLOCK_SIZE` and uses it directly as an
index into `Block_table_ptr`:

    page_idx = kv_offs // BLOCK_SIZE
    page_off = kv_offs % BLOCK_SIZE
    block_nums = tl.load(
        Block_table_ptr + bt_base + page_idx,
        mask=kv_mask,
        other=0,
    ).to(tl.int64)

The `mask=kv_mask` argument prevents the LOADED VALUE from being read on
masked-out lanes, but it does NOT suppress the address arithmetic. Triton's
bounds checker fires on `Block_table_ptr + bt_base + page_idx` whenever a
masked-out lane points past the block table — which happens routinely on
long sequences (>32k) where `kv_offs` runs past `split_end`.

The fix is a 4-line change: clamp masked-out lanes to page_idx=0 BEFORE
the address computation, so the pointer arithmetic stays in-bounds even
on lanes whose result will be discarded:

    safe_page_idx = tl.where(kv_mask, page_idx, 0)
    block_nums = tl.load(
        Block_table_ptr + bt_base + safe_page_idx,
        mask=kv_mask,
        other=0,
    ).to(tl.int64)

Zero perf cost — one extra `tl.where` per lane, all lanes are in registers.
Defensive correctness — the assertion can never fire even if Triton's
bounds checker becomes stricter in future releases.

================================================================
APPLICABILITY TO GENESIS
================================================================

Hardware tier where the bug is RECONSTRUCTIBLE: 4090 (sm_89). On 5090,
H20, R6000 the assertion does not fire (jhsmith409 ran 8-concurrent ×
31632-token-prompt stress, no assertion either side of the patch).

Genesis prod hardware: 2× A5000 (sm_86) — older than 4090. We have not
seen the assertion in production, and based on the pattern (Triton bounds
checker stricter on newer arch) we likely won't unless we migrate to
sm_89/Blackwell. However:

1. The clamp is **provably correct** regardless of whether the assertion
   fires — it can never make things worse.
2. Sander's planned RTX PRO 6000 Blackwell upgrade (Q3 2026) sits on
   sm_120, where the bounds-checker may be stricter still. Defensive
   patch removes a potential future blocker at near-zero cost.
3. Future model swaps to >32k sequence-length workloads (e.g., the
   long-context tool-call path on 27B that lands at 256K) push the
   `kv_offs >= split_end` regime where the assertion was originally
   reported in #39998.

The standard upstream `_tq_decode_stage1` kernel runs in Genesis prod
when:
- spec-decode is OFF entirely, OR
- spec-decode is ON but a particular step proposed K=1 (no MTP draft
  accepted that step), OR
- the P67 multi-query kernel dispatch returns False (shape outside
  envelope: Hq<8, D not in {128,256}, GQA<2, etc.)

So we DO hit this code path in production despite running spec-decode.

================================================================
SAFETY MODEL
================================================================

- Default OFF (opt-in via `GENESIS_ENABLE_PN14_TQ_DECODE_OOB_CLAMP=1`).
- Pure text-patch, idempotent via marker.
- Drift-aware: when upstream PR #40074 merges, the new `safe_page_idx`
  string appears in vanilla source and our anchor (the original 4-line
  block without `safe_page_idx`) won't match → patch is auto-skipped via
  `upstream_compat.py` marker `safe_page_idx`.
- Anchor missing → SKIPPED, source stays vanilla. Zero regression risk.
- Worst case: bug never fires on a particular workload = no-op runtime
  cost (one `tl.where` per lane, in registers).

Author backport: Sandermage(Sander) Barzov Aleksandr, Ukraine, Odessa.
Source PR: vllm-project/vllm#40074 by @devarakondasrikanth (Adobe).
"""
from __future__ import annotations

import logging

from vllm._genesis.guards import resolve_vllm_file, vllm_install_root
from vllm._genesis.wiring.text_patch import (
    TextPatch,
    TextPatcher,
    result_to_wiring_status,
)

log = logging.getLogger("genesis.wiring.pN14_tq_decode_oob_clamp")

GENESIS_PN14_MARKER = (
    "Genesis PN14 TQ decode safe_page_idx clamp (vllm#40074) v7.62.x"
)


# ─── Sub-patch: add safe_page_idx clamp before block_nums load ────────────

PN14_ANCHOR = (
    "        page_idx = kv_offs // BLOCK_SIZE\n"
    "        page_off = kv_offs % BLOCK_SIZE\n"
    "        block_nums = tl.load(\n"
    "            Block_table_ptr + bt_base + page_idx,\n"
    "            mask=kv_mask,\n"
    "            other=0,\n"
    "        ).to(tl.int64)\n"
)

PN14_REPLACEMENT = (
    "        page_idx = kv_offs // BLOCK_SIZE\n"
    "        page_off = kv_offs % BLOCK_SIZE\n"
    "        # [Genesis PN14 vllm#40074 backport] Clamp OOB lanes to page_idx=0\n"
    "        # before pointer arithmetic so Triton's bounds checker does not\n"
    "        # fire on masked-out lanes (mask= guards the loaded VALUE, not\n"
    "        # the address computation). Bug originally reported on 4090 with\n"
    "        # >32k sequences in upstream issue #39998.\n"
    "        safe_page_idx = tl.where(kv_mask, page_idx, 0)\n"
    "        block_nums = tl.load(\n"
    "            Block_table_ptr + bt_base + safe_page_idx,\n"
    "            mask=kv_mask,\n"
    "            other=0,\n"
    "        ).to(tl.int64)\n"
)


def _make_patcher() -> TextPatcher | None:
    target = resolve_vllm_file("v1/attention/ops/triton_turboquant_decode.py")
    if target is None:
        return None
    return TextPatcher(
        patch_name=(
            "PN14 triton_turboquant_decode.py — TQ decode IOOB safe_page_idx "
            "clamp (vllm#40074)"
        ),
        target_file=str(target),
        marker=GENESIS_PN14_MARKER,
        sub_patches=[
            TextPatch(
                name="pN14_safe_page_idx",
                anchor=PN14_ANCHOR,
                replacement=PN14_REPLACEMENT,
                required=True,
            ),
        ],
        upstream_drift_markers=[
            "[Genesis PN14",
            # If upstream PR #40074 lands, `safe_page_idx` appears in vanilla
            # source and our anchor (without safe_page_idx) won't match.
            "safe_page_idx = tl.where(kv_mask, page_idx, 0)",
        ],
    )


def apply() -> tuple[str, str]:
    """Apply PN14 — TQ decode IOOB safe_page_idx clamp (text-patch)."""
    from vllm._genesis.dispatcher import log_decision, should_apply

    decision, reason = should_apply("PN14")
    log_decision("PN14", decision, reason)
    if not decision:
        return "skipped", reason

    if vllm_install_root() is None:
        return "skipped", "vllm install root not discoverable"

    patcher = _make_patcher()
    if patcher is None:
        return "skipped", "target file not resolvable"

    result, failure = patcher.apply()
    return result_to_wiring_status(
        result, failure,
        applied_message=(
            "PN14 applied: _tq_decode_stage1 now clamps masked-out lanes to "
            "page_idx=0 before block-table pointer arithmetic (vllm#40074). "
            "Defensive fix — prevents Triton bounds-checker assertion class "
            "originally reported on 4090 with >32k sequences in upstream #39998."
        ),
        patch_name=patcher.patch_name,
    )
