# SPDX-License-Identifier: Apache-2.0
"""Wiring for Patch N17 — FA2 softmax_lse runtime clamp (Cliff 1 mechanism A).

================================================================
WHAT THIS PATCH DOES
================================================================

Replaces the `max_seqlen_k = attn_metadata.max_seq_len` assignment in
`vllm/v1/attention/backends/flash_attn.py` with a runtime clamp to the
actual max-per-sequence value (computed from `seqused_k`), but ONLY
when CUDA stream is NOT capturing a graph. During cudagraph capture,
falls back to `attn_metadata.max_seq_len` for shape stability (the
upstream behavior).

================================================================
ROOT CAUSE
================================================================

FA2's `flash_attn_varlen_func` allocates an internal `softmax_lse`
buffer of shape `[num_seqs, num_heads, max_seqlen_k]` — sized by the
`max_seqlen_k` argument the caller passes, NOT by the actual sequence
lengths in `cu_seqlens_k` / `seqused_k`. Reference: Dao-AILab/
flash-attention#1011 (open since 2024).

vLLM's `gpu_model_runner.py` sets `attn_metadata.max_seq_len =
self.max_model_len` during cudagraph capture for shape stability
(also confirmed by upstream PR vllm#40961 for SWA models). This
choice leaks into runtime decode/prefill: at `--max-model-len=205000`
even a 25K-token chunk reserves softmax_lse for 205K tokens →
unnecessary 50-100 MiB allocation.

Empirical Cliff 1 mechanism A (noonghunna 2026-04-29 cross-rig on
RTX 3090):

  205K + 0.98 + TQ3 + no-vision: FA2 softmax_lse OOM, 50 MiB / 50 MiB free
  identical-prefill on 48K + 0.92: passes cleanly

Closing this mechanism widens the safe envelope for `long-text`
no-vision configs to ~205K. (The dual mechanism B — FFN intermediate
buffer cliff at 138 MiB on `long-vision` configs — is OUT OF SCOPE
for this patch and requires upstream-FFN changes; see Genesis
[Issue #11](https://github.com/Sandermage/genesis-vllm-patches/issues/11)
discussion thread for full analysis.)

================================================================
SAFETY MODEL
================================================================

- Cudagraph guard: only clamps in eager mode
  (`torch.cuda.is_current_stream_capturing()` returns False). During
  capture, behavior is identical to upstream (max_model_len padding).
  This preserves cudagraph shape stability.

- Per-rank guard: `seqused_k` is a tensor; `.max()` is a single
  GPU→CPU sync that fires once per FA2 call. Cost is one int read,
  amortized across the kernel work. Probed cost on Ampere: ~3-5 us
  per call → noise relative to FA2 kernel runtime (~ms).

- Idempotent via marker; drift detection on the upstream anchor.

- Default OFF; opt-in via `GENESIS_ENABLE_PN17_FA2_LSE_CLAMP=1`.
  Recommend enabling on `long-text-no-vision.yml` configs only;
  for `long-vision.yml` the FFN cliff dominates and PN17 is no-op.

================================================================
ANCHOR / REPLACEMENT
================================================================

The anchor block is the variable assignment lines just before the
`flash_attn_varlen_func` call in the non-cascade path of
`FlashAttentionImpl.forward`:

    if not attn_metadata.use_cascade:
        cu_seqlens_q = attn_metadata.query_start_loc
        seqused_k = attn_metadata.seq_lens
        max_seqlen_q = attn_metadata.max_query_len
        max_seqlen_k = attn_metadata.max_seq_len   ← THIS

We replace the last line with a conditional that consults
`seqused_k.max().item()` outside cudagraph capture.

Author backport: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
Diagnosis credit: noonghunna (cross-rig RTX 3090, Genesis Issue #11
follow-up 2026-04-29).
"""
from __future__ import annotations

import logging
import os

from vllm._genesis.guards import resolve_vllm_file, vllm_install_root
from vllm._genesis.wiring.text_patch import (
    TextPatch,
    TextPatcher,
)

log = logging.getLogger("genesis.wiring.pn17_fa2_softmax_lse_clamp")


GENESIS_PN17_MARKER = "Genesis PN17 FA2 softmax_lse runtime clamp v1"


# Drift markers: if upstream changes the anchor block (e.g. variable
# rename, refactored cascade gate), our text-patch won't apply
# silently in the wrong place.
UPSTREAM_DRIFT_MARKERS = [
    GENESIS_PN17_MARKER,
    # If upstream natively clamps softmax_lse:
    "max_seqlen_k = int(seqused_k.max",
    # If upstream issue Dao-AILab/flash-attention#1011 lands a fix:
    "softmax_lse_clamped",
]


# Anchor: the 4-line block of attn-metadata reads just before the
# flash_attn_varlen_func call. Sized to be unique within the file.
PN17_OLD = (
    "        if not attn_metadata.use_cascade:\n"
    "            cu_seqlens_q = attn_metadata.query_start_loc\n"
    "            seqused_k = attn_metadata.seq_lens\n"
    "            max_seqlen_q = attn_metadata.max_query_len\n"
    "            max_seqlen_k = attn_metadata.max_seq_len\n"
)


PN17_NEW = (
    "        if not attn_metadata.use_cascade:\n"
    "            cu_seqlens_q = attn_metadata.query_start_loc\n"
    "            seqused_k = attn_metadata.seq_lens\n"
    "            max_seqlen_q = attn_metadata.max_query_len\n"
    "            # [Genesis PN17 FA2 softmax_lse runtime clamp v1]\n"
    "            # FA2 varlen allocates softmax_lse[num_seqs, heads, max_seqlen_k]\n"
    "            # — sized by THIS arg, not by actual seqused_k. Upstream sets\n"
    "            # attn_metadata.max_seq_len = max_model_len during cudagraph\n"
    "            # capture for shape stability; that value leaks into runtime\n"
    "            # decode/prefill, causing 50-100 MiB over-allocation at long\n"
    "            # context (Cliff 1 mechanism A; ref Genesis Issue #11). Eager-\n"
    "            # mode runtime: clamp to actual chunk max from seqused_k.\n"
    "            # Capture mode: keep max_model_len for shape stability.\n"
    "            import torch as _genesis_pn17_torch\n"
    "            try:\n"
    "                _genesis_pn17_capturing = (\n"
    "                    _genesis_pn17_torch.cuda.is_available()\n"
    "                    and _genesis_pn17_torch.cuda.is_current_stream_capturing()\n"
    "                )\n"
    "            except Exception:\n"
    "                _genesis_pn17_capturing = False\n"
    "            if _genesis_pn17_capturing:\n"
    "                max_seqlen_k = attn_metadata.max_seq_len\n"
    "            else:\n"
    "                try:\n"
    "                    max_seqlen_k = int(seqused_k.max().item())\n"
    "                    # Defensive lower bound: should not exceed upstream's\n"
    "                    # max_model_len cap regardless of metadata corruption.\n"
    "                    if max_seqlen_k > attn_metadata.max_seq_len:\n"
    "                        max_seqlen_k = attn_metadata.max_seq_len\n"
    "                except Exception:\n"
    "                    max_seqlen_k = attn_metadata.max_seq_len\n"
)


def _patcher() -> TextPatcher | None:
    target = resolve_vllm_file("v1/attention/backends/flash_attn.py")
    if target is None:
        return None
    return TextPatcher(
        patch_name="PN17 FA2 softmax_lse runtime clamp",
        target_file=target,
        marker=GENESIS_PN17_MARKER,
        sub_patches=[
            TextPatch(
                name="pn17_clamp",
                anchor=PN17_OLD,
                replacement=PN17_NEW,
                required=True,
            ),
        ],
        upstream_drift_markers=UPSTREAM_DRIFT_MARKERS,
    )


def _is_enabled() -> bool:
    return os.environ.get(
        "GENESIS_ENABLE_PN17_FA2_LSE_CLAMP", ""
    ).strip().lower() in ("1", "true", "yes", "on")


def apply() -> tuple[str, str]:
    """Apply PN17. Default OFF. Opt-in via env flag.

    See module docstring for safety model + per-config recommendation:
    enable on long-text-no-vision configs (closes FA2 softmax_lse
    cliff). No-op on long-vision configs (FFN buffer dominates;
    out-of-scope upstream-FFN problem per Issue #11 dual-mechanism
    analysis).
    """
    if not _is_enabled():
        return "skipped", (
            "GENESIS_ENABLE_PN17_FA2_LSE_CLAMP not set; default OFF. "
            "Enable on long-text-no-vision configs to close Cliff 1 "
            "mechanism A (FA2 softmax_lse over-allocation at long ctx). "
            "Diagnosis credit: noonghunna, Genesis Issue #11."
        )

    if vllm_install_root() is None:
        return "skipped", "vllm install root not discoverable"

    p = _patcher()
    if p is None:
        return "skipped", "v1/attention/backends/flash_attn.py not found"

    result, failure = p.apply()
    from vllm._genesis.wiring.text_patch import result_to_wiring_status
    return result_to_wiring_status(result, failure, applied_message='PN17 applied: FA2 softmax_lse buffer now clamped to actual seqused_k at runtime, freeing 50-100 MiB on long-ctx (Cliff 1 mechanism A fix per noonghunna Genesis Issue #11).', patch_name='PN17 FA2 softmax_lse runtime clamp')


def is_applied() -> bool:
    """Reporter for verify_live_rebinds in apply_all.py."""
    if vllm_install_root() is None:
        return False
    p = _patcher()
    if p is None:
        return False
    try:
        with open(p.target_file) as f:
            return GENESIS_PN17_MARKER in f.read()
    except Exception:
        return False
