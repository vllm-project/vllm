# SPDX-License-Identifier: Apache-2.0
"""Wiring for Patch 15B — extend PN17-style clamp to TQ FA varlen call.

Fixes Genesis Issue #15 (noonghunna 2026-05-01):
https://github.com/Sandermage/genesis-vllm-patches/issues/15

================================================================
PROBLEM (root cause)
================================================================

PN17 patches `vllm/v1/attention/backends/flash_attn.py` to clamp
`max_seqlen_k` from cudagraph-capture-bloated `max_model_len` to actual
runtime span. This prevents `softmax_lse` over-allocation in the FA2
backend.

But PN17's coverage doesn't reach the **TurboQuant code path**. When
`_continuation_prefill` (TQ k8v4 with chunked prefill at long context)
calls `self._flash_attn_varlen(...)` (turboquant_attn.py:394), the
`max_seqlen_k` passed in is `seq_len` from the metadata — which on
cudagraph-captured runtime can also be bloated to `max_model_len`.

The trace from noonghunna's repro:
```
turboquant_attn.py:909 _continuation_prefill
turboquant_attn.py:394 _flash_attn_varlen
flash_attn_interface.py:300 flash_attn_varlen_func
torch._ops:1269 → C extension allocates ~50 MiB workspace based on max_seqlen_k
torch.OutOfMemoryError
```

================================================================
FIX DESIGN
================================================================

Text-patch `turboquant_attn.py:_flash_attn_varlen` to clamp
`max_seqlen_k` to the ACTUAL maximum sequence length, computed from
`cu_seqlens_k`:

- For batch=1 (continuation prefill case): `cu_seqlens_k[-1] == seq_len`
  is the actual max, single tensor access.
- For batch>1: `(cu_seqlens_k[1:] - cu_seqlens_k[:-1]).max()` gives the
  actual max across batch elements. One reduction + sync.

The clamp adds ONE GPU→CPU sync per `_flash_attn_varlen` call. On the
continuation prefill path this is tolerable: each call already triggers
synchronous FA kernel invocation, and the path itself is infrequent
(once per chunked prefill rollover, not per decode token).

PN17's design avoided sync via CPU-resident metadata, but on this
path we don't have CPU-resident max_seq_len. The fallback sync is the
pragmatic choice given the alternative is silent OOM.

================================================================
SAFETY MODEL
================================================================

- Default OFF (opt-in via `GENESIS_ENABLE_P15B_FA_VARLEN_CLAMP=1`).
- Idempotent (marker-checked).
- Drift-aware: if upstream rewrites `_flash_attn_varlen` signature or
  body, anchor misses → SKIPPED, source stays vanilla.
- Try/except guard: if clamp computation raises (degenerate input),
  falls through to original `max_seqlen_k`. No crash.
- Sync added only ONCE per call (not per layer) — already-sync codepath.

Composition:

- P38 + P38B (Issue #14): orthogonal — they fix `_continuation_prefill`
  upstream alloc; P15B fixes the FA varlen wrapper alloc downstream.
- PN17: orthogonal — different file, different code path. Together they
  cover both FA backends.

Author: Sandermage(Sander) Barzov Aleksandr, Ukraine, Odessa
Origin: noonghunna Issue #15 — direct fix per their suggestion path 1
"""
from __future__ import annotations

import logging

from vllm._genesis.guards import resolve_vllm_file, vllm_install_root
from vllm._genesis.wiring.text_patch import (
    TextPatch,
    TextPatcher,
    result_to_wiring_status,
)

log = logging.getLogger("genesis.wiring.p15B_fa_varlen_clamp")

GENESIS_P15B_MARKER = "Genesis P15B FA varlen max_seqlen_k clamp (Issue #15) v7.65"


# Anchor: the function signature. We insert clamp logic right at the
# top of the body, before any other logic.
P15B_ANCHOR = (
    "    def _flash_attn_varlen(\n"
    "        self,\n"
    "        q: torch.Tensor,\n"
    "        k: torch.Tensor,\n"
    "        v: torch.Tensor,\n"
    "        cu_seqlens_q: torch.Tensor,\n"
    "        cu_seqlens_k: torch.Tensor,\n"
    "        max_seqlen_q: int,\n"
    "        max_seqlen_k: int,\n"
    "    ) -> torch.Tensor:\n"
    "        # fa_utils.get_flash_attn_version() returns None on backends that\n"
)

P15B_REPLACEMENT = (
    "    def _flash_attn_varlen(\n"
    "        self,\n"
    "        q: torch.Tensor,\n"
    "        k: torch.Tensor,\n"
    "        v: torch.Tensor,\n"
    "        cu_seqlens_q: torch.Tensor,\n"
    "        cu_seqlens_k: torch.Tensor,\n"
    "        max_seqlen_q: int,\n"
    "        max_seqlen_k: int,\n"
    "    ) -> torch.Tensor:\n"
    "        # [Genesis P15B Issue #15 fix] Clamp max_seqlen_k to actual span.\n"
    "        # On cudagraph-captured runtime, max_seqlen_k may equal\n"
    "        # max_model_len (320K+) even though actual span is smaller —\n"
    "        # FA wrapper's C extension over-allocates ~max_seqlen_k-sized\n"
    "        # workspace. Clamp adds one GPU->CPU sync per call but the call\n"
    "        # is on infrequent continuation-prefill path; sync cost amortizes.\n"
    "        if cu_seqlens_k is not None and cu_seqlens_k.numel() >= 2:\n"
    "            try:\n"
    "                if cu_seqlens_k.shape[0] == 2:\n"
    "                    _genesis_p15b_actual = int(cu_seqlens_k[-1].item())\n"
    "                else:\n"
    "                    _genesis_p15b_actual = int(\n"
    "                        (cu_seqlens_k[1:] - cu_seqlens_k[:-1]).max().item()\n"
    "                    )\n"
    "                if _genesis_p15b_actual > 0:\n"
    "                    max_seqlen_k = min(max_seqlen_k, _genesis_p15b_actual)\n"
    "            except Exception:\n"
    "                pass  # fall through with original value\n"
    "        # fa_utils.get_flash_attn_version() returns None on backends that\n"
)


def _make_patcher() -> TextPatcher | None:
    target = resolve_vllm_file("v1/attention/backends/turboquant_attn.py")
    if target is None:
        return None
    return TextPatcher(
        patch_name=(
            "P15B turboquant_attn.py — _flash_attn_varlen max_seqlen_k clamp "
            "(Issue #15 fix)"
        ),
        target_file=str(target),
        marker=GENESIS_P15B_MARKER,
        sub_patches=[
            TextPatch(
                name="p15b_fa_varlen_clamp",
                anchor=P15B_ANCHOR,
                replacement=P15B_REPLACEMENT,
                required=True,
            ),
        ],
        upstream_drift_markers=[
            "[Genesis P15B",
            "_genesis_p15b_actual",
        ],
    )


def apply() -> tuple[str, str]:
    """Apply P15B — FA varlen max_seqlen_k clamp."""
    from vllm._genesis.dispatcher import log_decision, should_apply

    decision, reason = should_apply("P15B")
    log_decision("P15B", decision, reason)
    if not decision:
        return "skipped", reason

    if vllm_install_root() is None:
        return "skipped", "vllm install root not discoverable"

    patcher = _make_patcher()
    if patcher is None:
        return "skipped", "turboquant_attn.py not resolvable"

    result, failure = patcher.apply()
    return result_to_wiring_status(
        result, failure,
        applied_message=(
            "P15B applied: _flash_attn_varlen now clamps max_seqlen_k to "
            "actual cu_seqlens_k span. Prevents 50 MiB FA wrapper workspace "
            "OOM on long-context continuation-prefill (Issue #15). Adds "
            "one GPU->CPU sync per call on infrequent path."
        ),
        patch_name=patcher.patch_name,
    )


def is_applied() -> bool:
    target = resolve_vllm_file("v1/attention/backends/turboquant_attn.py")
    if target is None:
        return False
    try:
        with open(str(target)) as f:
            return GENESIS_P15B_MARKER in f.read()
    except OSError:
        return False
