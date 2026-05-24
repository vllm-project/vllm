# SPDX-License-Identifier: Apache-2.0
"""Wiring for Patch 36 — Shared TurboQuant decode intermediate buffers.

Mirrors upstream open PR #40655 by @bhoomit.

Problem
-------
`Attention._init_turboquant_buffers` calls `self.register_buffer(...)`
three times per TQ attention layer:

  - `_tq_mid_o_buf`:   (B, Hq, S, D+1) fp32  — KV-split accumulator
  - `_tq_output_buf`:  (B, Hq, D) fp32       — final per-head output
  - `_tq_lse_buf`:     (B, Hq) fp32          — logsumexp scratch

For Qwen3-32B (60 TQ layers) this is 180 `register_buffer` calls at
model init — creating ~16 GiB of direct allocations PLUS ~45 GiB of
PyTorch allocator fragmentation (model load goes from 62 GiB expected
to 123 GiB observed). Upstream PR #40655 body: «baseline OOMs on a
single H200 (143 GB) at 0.90 util; with this patch 588K KV tokens,
52.7 tok/s rock-steady across 5 runs».

For our Qwen3.6-35B-A3B (10 TQ attention layers hybrid), the direct
savings are smaller (~9 MiB), BUT:

  1. `expandable_segments:True` in our prod env already mitigates most
     fragmentation — but not all.
  2. We hit OOM at 50k prefill with only **21 MiB free** headroom
     (v7.0 integration run 2026-04-24). Any freed MiB is load-bearing.
  3. `register_buffer` at init creates allocator slab traffic that
     competes with weight-load slabs → reduces contiguous free space.
  4. Shipping this patch now means when we upgrade to a bigger hybrid
     model (Qwen3-Next-80B-AWQ on roadmap, 40+ TQ layers projected),
     the mechanism is already in place.

Invariant justifying sharing
----------------------------
All TQ attention layers in a given model use identical `(max_num_seqs,
num_q_heads, tq_max_kv_splits, head_size)` config. The buffers are
scratchpads — fresh each forward call, no cross-layer state. TQ layers
execute SEQUENTIALLY per step (`forward()` is not parallelised across
layers). Therefore one shared set of buffers is functionally equivalent
to per-layer copies.

Pointer stability
-----------------
The shared buffers are allocated ONCE at first `_init_turboquant_buffers`
call on a given (config, device) key, then reused. `data_ptr()` stays
constant — CUDA graph capture safe.

Platform compatibility
----------------------
  - NVIDIA CUDA SM ≥ 8.0: shared path engaged.
  - AMD / XPU / CPU: `should_apply()` returns False → manager returns
    None → we fall back to upstream per-layer `register_buffer` path.

Upstream drift detection
------------------------
If the file already contains a shared-buffer implementation
(`_tq_shared_mid_o_buf`, `_tq_decode_buffer_manager`, PR #40655 landed)
we skip.

Author: Sandermage(Sander)-Barzov Aleksandr, Ukraine, Odessa
Reference: https://github.com/vllm-project/vllm/pull/40655 (@bhoomit)
           https://github.com/vllm-project/vllm/pull/40748 (@anonsubmitter, alt.)
"""
from __future__ import annotations

import logging

from vllm._genesis.guards import resolve_vllm_file, vllm_install_root
from vllm._genesis.wiring.text_patch import (
    TextPatch, TextPatcher, TextPatchResult,
)

log = logging.getLogger("genesis.wiring.p36_tq_shared_decode_buffers")

GENESIS_P36_MARKER = "Genesis P36 TQ shared decode buffers v7.0"

UPSTREAM_DRIFT_MARKERS = [
    # PR #40655 signatures
    "_tq_shared_mid_o_buf",
    "_tq_decode_buffer_manager",
    "get_shared_decode_buffers(",
    # Alternative PR #40748 signatures (WorkspaceManager variant)
    "reserve_turboquant_decode_workspace",
    "WorkspaceManager.reserve_turboquant",
]


# Anchor matches the 3 consecutive `register_buffer` calls for the
# transient decode buffers. The preceding `_tq_centroids` + comment
# are NOT touched — centroids are per-config state, not scratch.
_OLD = (
    "        self.register_buffer(\n"
    "            \"_tq_mid_o_buf\",\n"
    "            torch.empty(B, Hq, S, D + 1, dtype=torch.float32),\n"
    "            persistent=False,\n"
    "        )\n"
    "        self.register_buffer(\n"
    "            \"_tq_output_buf\",\n"
    "            torch.empty(B, Hq, D, dtype=torch.float32),\n"
    "            persistent=False,\n"
    "        )\n"
    "        self.register_buffer(\n"
    "            \"_tq_lse_buf\",\n"
    "            torch.empty(B, Hq, dtype=torch.float32),\n"
    "            persistent=False,\n"
    "        )"
)

_NEW = (
    "        # [Genesis P36] Shared decode buffers across all TQ attention\n"
    "        # layers (mirrors upstream PR #40655). All TQ layers use identical\n"
    "        # (B, Hq, S, D) config and execute sequentially per step, so one\n"
    "        # shared buffer pool is safe. Saves 3× register_buffer calls per\n"
    "        # layer → reduced allocator slab fragmentation. Falls back to the\n"
    "        # original per-layer register_buffer path on non-NVIDIA / pre-Ampere\n"
    "        # (manager returns None).\n"
    "        try:\n"
    "            from vllm._genesis.kernels.dequant_buffer import (\n"
    "                TurboQuantBufferManager as _GenesisTQBuf,\n"
    "            )\n"
    "            _target_device = (\n"
    "                torch.device(f\"cuda:{torch.cuda.current_device()}\")\n"
    "                if torch.cuda.is_available() else torch.device(\"cpu\")\n"
    "            )\n"
    "            _mid_o = _GenesisTQBuf.get_shared_decode_mid_o(\n"
    "                max_num_seqs=B, num_q_heads=Hq,\n"
    "                tq_max_kv_splits=S, head_size=D,\n"
    "                device=_target_device, dtype=torch.float32,\n"
    "            )\n"
    "            _output = _GenesisTQBuf.get_shared_decode_output(\n"
    "                max_num_seqs=B, num_q_heads=Hq, head_size=D,\n"
    "                device=_target_device, dtype=torch.float32,\n"
    "            )\n"
    "            _lse = _GenesisTQBuf.get_shared_decode_lse(\n"
    "                max_num_seqs=B, num_q_heads=Hq,\n"
    "                device=_target_device, dtype=torch.float32,\n"
    "            )\n"
    "        except Exception:\n"
    "            _mid_o = _output = _lse = None\n"
    "        if _mid_o is not None and _output is not None and _lse is not None:\n"
    "            # Use setattr (not register_buffer) — these are scratchpads,\n"
    "            # not model state, and we've already placed them on the\n"
    "            # target device so `.to(device)` migration isn't needed.\n"
    "            self._tq_mid_o_buf = _mid_o\n"
    "            self._tq_output_buf = _output\n"
    "            self._tq_lse_buf = _lse\n"
    "        else:\n"
    "            # Fallback: upstream per-layer path (non-NVIDIA / pre-Ampere).\n"
    "            self.register_buffer(\n"
    "                \"_tq_mid_o_buf\",\n"
    "                torch.empty(B, Hq, S, D + 1, dtype=torch.float32),\n"
    "                persistent=False,\n"
    "            )\n"
    "            self.register_buffer(\n"
    "                \"_tq_output_buf\",\n"
    "                torch.empty(B, Hq, D, dtype=torch.float32),\n"
    "                persistent=False,\n"
    "            )\n"
    "            self.register_buffer(\n"
    "                \"_tq_lse_buf\",\n"
    "                torch.empty(B, Hq, dtype=torch.float32),\n"
    "                persistent=False,\n"
    "            )"
)


def _make_patcher() -> TextPatcher | None:
    target = resolve_vllm_file("model_executor/layers/attention/attention.py")
    if target is None:
        return None
    return TextPatcher(
        patch_name="P36 TQ shared decode buffers",
        target_file=target,
        marker=GENESIS_P36_MARKER,
        sub_patches=[
            TextPatch(
                name="p36_shared_decode_bufs",
                anchor=_OLD,
                replacement=_NEW,
                required=True,
            ),
        ],
        upstream_drift_markers=UPSTREAM_DRIFT_MARKERS,
    )


def apply() -> tuple[str, str]:
    """Apply P36 wiring. Never raises.

    v7.12: consults `config_detect.should_apply("P36")` first.
    Skipped automatically if upstream PR #40798 active OR if
    `max_num_seqs < 8` (memory benefit marginal).
    Override via `GENESIS_FORCE_APPLY_P36=1`.
    """
    try:
        from vllm._genesis import config_detect
        ok, reason = config_detect.should_apply("P36")
        if not ok:
            return "skipped", reason
    except Exception as e:
        log.debug("[P36] config_detect probe failed (proceeding): %s", e)

    if vllm_install_root() is None:
        return "skipped", "vllm install root not discoverable"

    patcher = _make_patcher()
    if patcher is None:
        return "skipped", "model_executor/layers/attention/attention.py not found"

    result, failure = patcher.apply()
    if result == TextPatchResult.APPLIED:
        return (
            "applied",
            "decode mid_o + output + lse buffers shared across TQ attention layers",
        )
    if result == TextPatchResult.IDEMPOTENT:
        return "applied", "already applied this image layer (idempotent)"
    if result == TextPatchResult.SKIPPED:
        return "skipped", failure.reason if failure else "unknown skip"
    return "failed", failure.reason if failure else "unknown failure"
