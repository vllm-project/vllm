# SPDX-License-Identifier: Apache-2.0
"""Wiring for Patch 44 — TQ mixed-batch attn_out zero pool.

The mixed decode+prefill branch of `TurboQuantAttentionImpl._forward`
(dev134 `turboquant_attn.py:438`) does
    attn_out = torch.zeros(N, self.num_heads, self.head_size,
                           device=device, dtype=q.dtype)
on every forward that lands in a mixed batch. `N` up to
`max_num_batched_tokens` (4096 in our config) × 40 heads × 256 dim ×
2 bytes = ~80 MB zero-init **per mixed-batch forward**.

On single-user VM 100 this branch fires rarely but when it does the
zero-init cost is non-trivial at long context. On multi-user setups
(typical production serving) this is the hot path.

P26 covers the prefill-only branch (line 566). This patch extends
the same infrastructure (`TurboQuantBufferManager`) to the mixed
branch via `acquire_mixed_attn_out(...)`.

Byte-exact output
-----------------
`.zero_()` on the returned view matches `torch.zeros(...)`
semantics exactly. Pool is keyed by (max_batched, Hq, D, dtype,
device); grow via the standard `acquire` fallback to `torch.zeros`
when overflow.

Author: Sandermage(Sander)-Barzov Aleksandr, Ukraine, Odessa
Status: v7.7 (default-on NVIDIA SM 8.0+)
"""
from __future__ import annotations

import logging

from vllm._genesis.guards import (
    is_nvidia_cuda, is_sm_at_least, resolve_vllm_file, vllm_install_root,
)
from vllm._genesis.wiring.text_patch import (
    TextPatch, TextPatcher, TextPatchResult,
)

log = logging.getLogger("genesis.wiring.p44_tq_mixed_attn_out")

GENESIS_P44_MARKER = "Genesis P44 TQ mixed-batch attn_out pool v7.7"

UPSTREAM_DRIFT_MARKERS = [
    "acquire_mixed_attn_out",
    "_tq_mixed_attn_out_pool",
    "mixed_attn_out_persistent",
]


_OLD_MIXED_ATTN_OUT = (
    "            # Mixed batch: decodes first (guaranteed by reorder_batch).\n"
    "            attn_out = torch.zeros(\n"
    "                N, self.num_heads, self.head_size, device=device, dtype=q.dtype\n"
    "            )"
)
_NEW_MIXED_ATTN_OUT = (
    "            # Mixed batch: decodes first (guaranteed by reorder_batch).\n"
    "            # [Genesis P44] Shared, profiler-visible mixed-batch output pool.\n"
    "            # Scheduler budget resolved at `_ensure_on_device` time, stashed\n"
    "            # on the impl as `_max_num_batched_tokens`. Fallback to 4096\n"
    "            # if the attr isn't populated yet (pre-warmup path).\n"
    "            from vllm._genesis.kernels.dequant_buffer import (\n"
    "                TurboQuantBufferManager as _GenesisTQBufP44,\n"
    "            )\n"
    "            attn_out = _GenesisTQBufP44.acquire_mixed_attn_out(\n"
    "                num_tokens=N,\n"
    "                num_q_heads=self.num_heads,\n"
    "                head_size=self.head_size,\n"
    "                device=device, dtype=q.dtype,\n"
    "                max_batched_tokens=getattr(\n"
    "                    self, '_max_num_batched_tokens', None,\n"
    "                ),\n"
    "            )"
)


def _make_patcher() -> TextPatcher | None:
    target = resolve_vllm_file("v1/attention/backends/turboquant_attn.py")
    if target is None:
        return None
    return TextPatcher(
        patch_name="P44 TQ mixed-batch attn_out pool",
        target_file=target,
        marker=GENESIS_P44_MARKER,
        sub_patches=[
            TextPatch(
                name="p44_mixed_attn_out_alloc",
                anchor=_OLD_MIXED_ATTN_OUT,
                replacement=_NEW_MIXED_ATTN_OUT,
                required=True,
            ),
        ],
        upstream_drift_markers=UPSTREAM_DRIFT_MARKERS,
    )


def should_apply() -> bool:
    if not is_nvidia_cuda():
        return False
    if not is_sm_at_least(8, 0):
        return False
    return True


def apply() -> tuple[str, str]:
    if not is_nvidia_cuda():
        return "skipped", "non-NVIDIA: TurboQuant is CUDA-only"
    if not is_sm_at_least(8, 0):
        return "skipped", "SM < 8.0"
    if vllm_install_root() is None:
        return "skipped", "vllm install root not discoverable"
    patcher = _make_patcher()
    if patcher is None:
        return "skipped", "turboquant_attn.py not found"
    result, failure = patcher.apply()
    if result == TextPatchResult.APPLIED:
        return "applied", (
            "text-patch applied — mixed-batch attn_out routed through "
            "TurboQuantBufferManager pool (~80 MB zero-init eliminated "
            "per mixed-batch forward in multi-user serving)"
        )
    if result == TextPatchResult.IDEMPOTENT:
        return "applied", "already patched this image layer (idempotent)"
    if result == TextPatchResult.SKIPPED:
        return "skipped", failure.reason if failure else "unknown skip"
    return "failed", failure.reason if failure else "unknown failure"


def is_applied() -> bool:
    target = resolve_vllm_file("v1/attention/backends/turboquant_attn.py")
    if target is None or not target.exists():
        return False
    try:
        return GENESIS_P44_MARKER in target.read_text()
    except Exception:
        return False


def revert() -> bool:
    """Text-patch — no runtime revert."""
    return False
