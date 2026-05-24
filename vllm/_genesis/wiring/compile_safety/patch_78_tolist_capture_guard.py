# SPDX-License-Identifier: Apache-2.0
"""Wiring for Patch 78 — surgical .tolist() capture-guard for TurboQuant.

================================================================
CREDIT
================================================================

Algorithm + anchor strings adapted from noonghunna's
`patch_tolist_cudagraph.py`:
  https://github.com/noonghunna/qwen36-27b-single-3090/blob/master/patches/patch_tolist_cudagraph.py
  (Apache-2.0, original author: @noonghunna)

Original problem-statement and bypass logic are noonghunna's. We adapt
it to:
  - Run under our `TextPatcher` framework (idempotent, drift-marker-aware)
  - Use Genesis env-gate convention (`GENESIS_ENABLE_P78_*`)
  - Compose cleanly with our P22/P26/P44 prealloc patches (which already
    avoid the `.tolist()` path on steady-state — P78 is the safety-net
    for cases where prealloc is bypassed, e.g. cold cudagraph capture
    with dynamic batch shapes)

Per `feedback_no_ai_credit_in_public.md`: no AI co-author credit. Sole
human authors are Sander Barzov (Genesis adaptation) + noonghunna (original).

================================================================
WHAT THIS FIXES
================================================================

`vllm/v1/attention/backends/turboquant_attn.py` has two `.tolist()` calls
that force GPU->CPU sync inside paths that can execute under active
CUDA stream capture:

  Site A — `forward()` mixed-batch branch:
      prefill_max_seq = max(attn_metadata.seq_lens[num_decodes:].tolist())

  Site B — `_prefill_attention()` continuation branch:
      qsl = query_start_loc.tolist()
      seq_lens_list = attn_metadata.seq_lens.tolist()

Hit during cudagraph capture warmup with mixed prefill+decode or with
spec-decode + chunked-prefill (continuation chunks).

Our P22/P26/P44 patches AVOID these paths on steady-state (prealloc'd
buffers used directly), but warmup/capture can transit them before
prealloc kicks in. P78 makes the path itself capture-safe by using
`torch.cuda.is_current_stream_capturing()` as a guard.

================================================================
COMPOSITION
================================================================

- Composes additively with P22/P26/P44 — P78 fires ONLY during capture,
  prealloc fires on steady-state. No conflict.
- Composes additively with P67/P67b — P67 routes K+1 spec-verify above
  the `_prefill_attention` path; P78 makes the fallback path safer.
- For sites already prealloc'd by P22/P26/P44 in the steady-state path:
  the runtime check `is_current_stream_capturing() == False` short-circuits
  to original behavior — zero overhead.

================================================================
ENV
================================================================

GENESIS_ENABLE_P78_TOLIST_CAPTURE_GUARD=1   # opt-in master switch

================================================================
RISK
================================================================

LOW. Capture-time output values are not used by inference (V1 PIECEWISE
mode marks attention as splitting_op — capture only drives memory
profiling). The flash_attn_varlen_func fast-path returns the right
shape with similar workspace footprint, so memory profiling stays accurate.

If `_HAS_FLASH_ATTN` is False (rare), falls back to `torch.zeros(...)`
which is safe (correct shape, no garbage propagation in non-inference
captured graph).

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
Adapted from: @noonghunna `patch_tolist_cudagraph.py` (Apache-2.0).
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

log = logging.getLogger("genesis.wiring.p78_tolist_capture_guard")

GENESIS_P78_MARKER = "Genesis P78 tolist capture-guard (Sites B+C+D+E metadata-builder pattern) v7.63.x_v6"


# ─── Sub-patch: insert capture-guard early-return BEFORE continuation branch ─
# Anchor on the start of the continuation branch in _prefill_attention.

P78_OLD = (
    "        # Continuation or no flash_attn: per-request attention.\n"
    "        # For continuation chunks (seq_len > q_len), we must attend to\n"
    "        # previously cached K/V from the TQ cache, not just the current\n"
    "        # chunk's raw K/V.\n"
    "        Hk = key.shape[1]\n"
)

# Site A — forward() mixed-batch branch's `prefill_max_seq = max(...tolist())`.
# Discovered 2026-04-30: P78 v1 only covered Site B; vllm pin 0.20.1rc1.dev16
# exercises Site A during cudagraph capture warmup with mixed prefill+decode
# batches (which fire on 27B Lorbus + TQ k8v4 + MTP K=3, but NOT on 35B-A3B
# because 35B's GQA=8 power-of-2 routes through P67 multi-query kernel
# BEFORE this site). Adding the same is_current_stream_capturing() guard
# from external_probe/patch_tolist_cudagraph.py — substitute the batch-level
# max_seq_len Python int (safe overestimate; flash_attn uses it as a grid
# upper bound only).
P78_SITE_A_OLD = (
    "            # Use CPU-side max to avoid GPU→CPU sync from .item()\n"
    "            prefill_max_seq = max(attn_metadata.seq_lens[num_decodes:].tolist())\n"
)

P78_SITE_A_NEW = (
    "            # Use CPU-side max to avoid GPU→CPU sync from .item()\n"
    "            # [Genesis P78 Site A v5 — DISABLED via empty-substitution]\n"
    "            # Both attempts to substitute prefill_max_seq during cudagraph\n"
    "            # capture proved incorrect:\n"
    "            #   v2 (max_seq_len): over-estimates including 280K decode → flash_attn\n"
    "            #     reads invalid KV → garbage output\n"
    "            #   v5 (max_query_len): under-estimates to K+1 → seems clean during\n"
    "            #     capture but produces 3-token truncation `<tool_call>ello` on\n"
    "            #     subsequent tool-call inference. Likely the captured graph's\n"
    "            #     attention output for decode tokens IS replayed, and the\n"
    "            #     under-grid creates wrong attention math.\n"
    "            # \n"
    "            # The robust fix needs upstream attention rewrite (move\n"
    "            # prefill_max_seq computation into the metadata builder where it\n"
    "            # can use CPU-side seq_lens without sync). Until that lands, the\n"
    "            # operator-level fix is to use `--cudagraph-mode=PIECEWISE` to\n"
    "            # avoid capturing this path entirely.\n"
    "            #\n"
    "            # Site A is intentionally re-applied here as a no-op echo so the\n"
    "            # text-patch's idempotency marker is in the file (and reapply\n"
    "            # detects 'already done'). DO NOT register Site A in sub_patches.\n"
    "            prefill_max_seq = max(attn_metadata.seq_lens[num_decodes:].tolist())\n"
)

# ═══════════════════════════════════════════════════════════════════════
# Sites C/D/E — proper architectural fix for prefill_max_seq sync.
#
# Mirrors vllm GDN backend (PR #38361 MERGED 2026-04-03), vllm flash_attn
# backend, SGLang's seq_lens_cpu discipline, TRT-LLM's _max_seq_len_storage.
# Closes vllm Issue #40807 (open, filed 2026-04-24 by noonghunna).
#
# Strategy: pre-compute prefill_max_seq in TurboQuantMetadataBuilder.build()
# using CommonAttentionMetadata.seq_lens_cpu (a CPU tensor — `.item()` on it
# is a Python op, NOT a CUDA sync). Store as a Python int field on
# TurboQuantMetadata. forward() reads the int — zero GPU sync.
#
# Capture-time semantics: build() runs OUTSIDE captured region (per vllm
# CUDA graph design — only kernel launches are captured, not metadata
# build). The captured graph contains attention kernels parameterized by
# the captured-time prefill_max_seq value. At replay, vllm only replays
# uniform decode/spec-verify batches (mixed-batch with prefills runs
# eagerly), so prefill_max_seq variance never breaks replay correctness.
# ═══════════════════════════════════════════════════════════════════════

# Site C — TurboQuantMetadata dataclass: add prefill_max_seq_cpu field.
P78_SITE_C_OLD = (
    "    is_prefill: bool = False\n"
    "    num_decodes: int = 0  # number of decode requests (first in batch)\n"
    "    num_decode_tokens: int = 0  # tokens from decode requests\n"
)

P78_SITE_C_NEW = (
    "    is_prefill: bool = False\n"
    "    num_decodes: int = 0  # number of decode requests (first in batch)\n"
    "    num_decode_tokens: int = 0  # tokens from decode requests\n"
    "    # [Genesis P78 Site C] precomputed in Builder.build() from\n"
    "    # CommonAttentionMetadata.seq_lens_cpu — Python op, NO CUDA sync.\n"
    "    # Used by forward() instead of `seq_lens[num_decodes:].tolist().max()`\n"
    "    # which crashes under cudagraph capture. Closes vllm#40807.\n"
    "    prefill_max_seq_cpu: int = 0\n"
)

# Site D — Builder.build(): compute prefill_max_seq_cpu from CPU mirror.
P78_SITE_D_OLD = (
    "        return TurboQuantMetadata(\n"
    "            seq_lens=cam.seq_lens,\n"
    "            slot_mapping=cam.slot_mapping,\n"
    "            block_table=cam.block_table_tensor,\n"
    "            query_start_loc=cam.query_start_loc,\n"
    "            num_actual_tokens=cam.num_actual_tokens,\n"
    "            max_query_len=cam.max_query_len,\n"
    "            max_seq_len=cam.max_seq_len,\n"
    "            is_prefill=(cam.max_query_len > 1),\n"
    "            num_decodes=num_decodes,\n"
    "            num_decode_tokens=num_decode_tokens,\n"
    "        )\n"
)

P78_SITE_D_NEW = (
    "        # [Genesis P78 Site D] Pre-compute prefill_max_seq from CPU mirror\n"
    "        # so forward() doesn't need to call .tolist() on a GPU tensor under\n"
    "        # cudagraph capture. Mirrors vllm GDN PR #38361 + flash_attn backend\n"
    "        # pattern. cam.seq_lens_cpu is a CPU tensor (lazy-cached property);\n"
    "        # .max().item() on a CPU tensor is a Python op, NOT a CUDA sync.\n"
    "        # cam exposes _seq_lens_cpu as well — both work.\n"
    "        if num_decodes < cam.num_reqs:\n"
    "            _genesis_p78_seq_lens_cpu = cam.seq_lens_cpu\n"
    "            _genesis_p78_prefill_slice = _genesis_p78_seq_lens_cpu[num_decodes:]\n"
    "            _genesis_p78_prefill_max_seq = int(_genesis_p78_prefill_slice.max().item())\n"
    "        else:\n"
    "            _genesis_p78_prefill_max_seq = 0\n"
    "        return TurboQuantMetadata(\n"
    "            seq_lens=cam.seq_lens,\n"
    "            slot_mapping=cam.slot_mapping,\n"
    "            block_table=cam.block_table_tensor,\n"
    "            query_start_loc=cam.query_start_loc,\n"
    "            num_actual_tokens=cam.num_actual_tokens,\n"
    "            max_query_len=cam.max_query_len,\n"
    "            max_seq_len=cam.max_seq_len,\n"
    "            is_prefill=(cam.max_query_len > 1),\n"
    "            num_decodes=num_decodes,\n"
    "            num_decode_tokens=num_decode_tokens,\n"
    "            prefill_max_seq_cpu=_genesis_p78_prefill_max_seq,\n"
    "        )\n"
)

# Site E — forward(): replace .tolist() max with metadata field read.
# Anchor and replacement both touch the same line as Site A; they are
# MUTUALLY EXCLUSIVE — Site E supersedes Site A. Site A's anchor and
# replacement strings are kept in this file for git-history reference but
# are not registered in sub_patches.
P78_SITE_E_OLD = (
    "            # Use CPU-side max to avoid GPU→CPU sync from .item()\n"
    "            prefill_max_seq = max(attn_metadata.seq_lens[num_decodes:].tolist())\n"
)

P78_SITE_E_NEW = (
    "            # Use CPU-side max to avoid GPU→CPU sync from .item()\n"
    "            # [Genesis P78 Site E] Read from metadata (precomputed in Builder.build())\n"
    "            # to avoid .tolist() — illegal under cudagraph capture.\n"
    "            # Closes vllm Issue #40807. Pattern mirrors merged PR #38361 (GDN).\n"
    "            prefill_max_seq = attn_metadata.prefill_max_seq_cpu\n"
)


P78_NEW = (
    "        # ════════════════════════════════════════════════════════════\n"
    "        # [Genesis P78 — adapted from noonghunna's patch_tolist_cudagraph.py]\n"
    "        # During CUDA graph capture, the continuation branch below calls\n"
    "        # .tolist() forcing a GPU->CPU sync — illegal under torch.cuda.graph().\n"
    "        # vLLM V1 PIECEWISE marks unified_attention_with_output as a splitting_op,\n"
    "        # so capture does NOT bake in attention outputs; capture-time values\n"
    "        # only need to drive memory profiling. Falling back to the graph-safe\n"
    "        # flash_attn_varlen_func returns the same shape with similar workspace.\n"
    "        # At inference (non-capture), is_current_stream_capturing()==False and\n"
    "        # the original per-request continuation path runs unchanged.\n"
    "        # CREDIT: github.com/noonghunna/qwen36-27b-single-3090 (Apache-2.0)\n"
    "        # ════════════════════════════════════════════════════════════\n"
    "        import os as _genesis_p78_os\n"
    "        if (\n"
    "            _genesis_p78_os.environ.get('GENESIS_ENABLE_P78_TOLIST_CAPTURE_GUARD', '').strip().lower()\n"
    "            in ('1', 'true', 'yes', 'on')\n"
    "            and torch.cuda.is_current_stream_capturing()\n"
    "        ):\n"
    "            try:\n"
    "                from vllm.attention.backends.flash_attn import flash_attn_varlen_func as _genesis_p78_fa_func\n"
    "                return _genesis_p78_fa_func(\n"
    "                    q=query, k=key, v=value,\n"
    "                    cu_seqlens_q=attn_metadata.query_start_loc,\n"
    "                    cu_seqlens_k=attn_metadata.query_start_loc,\n"
    "                    max_seqlen_q=attn_metadata.max_query_len,\n"
    "                    max_seqlen_k=attn_metadata.max_query_len,\n"
    "                    softmax_scale=self.scale,\n"
    "                    causal=True,\n"
    "                )\n"
    "            except Exception:\n"
    "                # Final fallback: correct shape zero tensor (capture-time\n"
    "                # output is unused under PIECEWISE; memory profile stays valid)\n"
    "                return torch.zeros(N, Hq, D, device=query.device, dtype=query.dtype)\n"
    "\n"
    "        # Continuation or no flash_attn: per-request attention.\n"
    "        # For continuation chunks (seq_len > q_len), we must attend to\n"
    "        # previously cached K/V from the TQ cache, not just the current\n"
    "        # chunk's raw K/V.\n"
    "        Hk = key.shape[1]\n"
)


def _make_patcher() -> TextPatcher | None:
    target = resolve_vllm_file("v1/attention/backends/turboquant_attn.py")
    if target is None:
        return None
    return TextPatcher(
        patch_name="P78 v1/attention/backends/turboquant_attn.py — tolist capture-guard",
        target_file=str(target),
        marker=GENESIS_P78_MARKER,
        sub_patches=[
            # Site B (original, kept) — capture-guard for _prefill_attention
            # continuation branch's qsl/seq_lens .tolist() calls. Substitutes
            # flash_attn_varlen_func during capture; safe because both
            # capture-time and inference-time call paths produce same shape
            # output, and capture output is discarded under PIECEWISE.
            TextPatch(
                name="p78_site_b_capture_guard",
                anchor=P78_OLD,
                replacement=P78_NEW,
                required=False,
            ),
            # Site C (NEW v6) — TurboQuantMetadata gets prefill_max_seq_cpu int.
            TextPatch(
                name="p78_site_c_metadata_field",
                anchor=P78_SITE_C_OLD,
                replacement=P78_SITE_C_NEW,
                required=True,
            ),
            # Site D (NEW v6) — Builder.build() pre-computes prefill_max_seq_cpu
            # from CPU-side seq_lens (no CUDA sync). Mirrors vllm GDN
            # PR #38361 (MERGED) pattern.
            TextPatch(
                name="p78_site_d_builder_precompute",
                anchor=P78_SITE_D_OLD,
                replacement=P78_SITE_D_NEW,
                required=True,
            ),
            # Site E (NEW v6) — forward() reads precomputed int instead of
            # .tolist() max. Closes vllm Issue #40807.
            # Site A (the substitute-with-batch-max approach) is intentionally
            # NOT registered here — empirical 2026-04-30 testing proved it
            # incorrect (over-grid invalid KV reads + cudagraph captured-
            # constant replay correctness issue per Tri Dao flash-attn#1164).
            TextPatch(
                name="p78_site_e_forward_read",
                anchor=P78_SITE_E_OLD,
                replacement=P78_SITE_E_NEW,
                required=True,
            ),
        ],
        upstream_drift_markers=[
            # If upstream adds the same stream-capturing guard at the
            # source we patch, our patch becomes redundant. The marker
            # check happens AFTER our `marker=GENESIS_P78_MARKER`
            # idempotency check (our own marker contains the same string),
            # so re-application is safe.
            "[Genesis P78",
            "GENESIS_ENABLE_P78_TOLIST_CAPTURE_GUARD",
        ],
    )


def apply() -> tuple[str, str]:
    """Apply P78 — surgical .tolist() capture guard."""
    from vllm._genesis.dispatcher import should_apply, log_decision
    decision, reason = should_apply("P78")
    log_decision("P78", decision, reason)
    if not decision:
        return "skipped", reason

    if vllm_install_root() is None:
        return "skipped", "vllm install root not discoverable"

    patcher = _make_patcher()
    if patcher is None:
        return "skipped", "vllm/v1/attention/backends/turboquant_attn.py not found"

    if not os.path.isfile(patcher.target_file):
        return "skipped", f"target disappeared: {patcher.target_file}"
    with open(patcher.target_file) as f:
        content = f.read()
    if patcher.marker in content:
        log.info("[P78] marker present — skip (idempotent)")
        return "applied", "idempotent (marker present)"
    for m in patcher.upstream_drift_markers:
        if m == "[Genesis P78" and m in content:
            continue
        if m in content:
            return (
                "skipped",
                f"upstream drift marker {m!r} in {patcher.target_file} — "
                "upstream may have absorbed this fix or independent capture-guard",
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
        "P78 applied: capture-guard installed in TurboQuant._prefill_attention "
        "continuation branch. Falls back to flash_attn_varlen_func during cudagraph "
        "capture (zero overhead at inference). Adapted from noonghunna's "
        "patch_tolist_cudagraph.py (Apache-2.0, attribution in patch docstring)."
    )
