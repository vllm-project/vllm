# SPDX-License-Identifier: Apache-2.0
"""Wiring for Patch 65 — TurboQuant cudagraph downgrade under spec-decode.

Genesis-original — addresses MTP × TurboQuant × FULL cudagraph degenerate
output bug per noonghunna #40880 + Genesis test cycle 2026-04-25.

================================================================
ROOT CAUSE (identified by Genesis investigation 2026-04-25):

`vllm/v1/attention/backends/turboquant_attn.py::TurboQuantAttentionImpl
._prefill_attention` has TWO code paths that BOTH pass `cu_seqlens_k =
query_start_loc`:
  - Fast path (line 568-578): only when `max_query_len == max_seq_len`
    (genuine first-chunk prefill, no prior cached KV).
  - Cudagraph capture bypass (line 580-602, applied via
    external_probe/patch_tolist_cudagraph.py): UNCONDITIONAL under
    `torch.cuda.is_current_stream_capturing()`.

For MTP n=3 spec-verify batches: q_len=4, max_seq_len=290+ (continuation
prefill — has prior cached KV in TurboQuant compressed form). At runtime
under FULL cudagraph capture, the bypass fires → flash_attn called with
cu_seqlens_k=cu_seqlens_q → kernel attends ONLY to the 4 query tokens of
current chunk, missing the entire cached history.

Captured kernel replays at runtime with same wrong logic → drafter/verifier
attention without context → predictions collapse.

For PLAIN TEXT (low-bias tokens) drafter randomness → verifier rejects →
falls back to correct main forward → output works (slower).

For TOOL-CALL (high-bias special tokens like `<tool_call>`) drafter+
verifier converge on the same bias-driven token → all accepted → cascade
`<tool_call><tool_call><tool_call>...`.
================================================================

Fix design — least-invasive workaround
--------------------------------------
Downgrade TurboQuant's `_cudagraph_support` from `UNIFORM_BATCH` (allows
spec-verify K+1 capture) to `UNIFORM_SINGLE_TOKEN_DECODE` (only 1-token
decode capture) when speculative_config is active. Spec-verify batches
fall through to eager execution → per-request continuation branch (lines
605+) runs correctly with full cached KV.

Cost: spec-verify batches lose cudagraph speedup (~30-50% throughput drop
on those batches, but main decode batches retain cudagraph). Net throughput
should land between cudagraph=ON broken (~85 TPS) and cudagraph=NONE
correct (~33 TPS).

This is a WORKAROUND not a fix. Proper fix requires reworking
`_prefill_attention` bypass to handle continuation prefill under capture
(needs upstream contribution since TurboQuant cached KV format is opaque
to flash_attn).

Status: opt-in via `GENESIS_ENABLE_P65_TURBOQUANT_SPEC_CG_DOWNGRADE=1`.

Compatibility
-------------
- Affects ONLY when both `kv_cache_dtype` is TurboQuant AND
  `speculative_config` is set
- Idempotent (marker + class-attribute check)
- Auto-no-op if upstream fixes _prefill_attention bypass

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

log = logging.getLogger("genesis.wiring.p65_turboquant_spec_cg_downgrade")

GENESIS_P65_MARKER = "Genesis P65 TurboQuant spec-decode CG downgrade v7.13"


# ─── Sub-patch: downgrade _cudagraph_support in TurboQuantMetadataBuilder ───
# Anchor on the existing class-attribute declaration. Replace UNIFORM_BATCH
# with UNIFORM_SINGLE_TOKEN_DECODE conditionally inside __init__ — but doing
# that via classvar mutation requires care. Simpler: change the ClassVar
# declaration itself to UNIFORM_SINGLE_TOKEN_DECODE always when this patch
# is enabled. Effect: under spec-decode, K+1 batches don't get FULL cudagraph
# capture, fall to eager. Without spec-decode, behavior unchanged (eager
# wouldn't have cudagraph for K+1 batches anyway since K+1 reduces to 1).

TQ_CG_SUPPORT_OLD = (
    "    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH\n"
)

TQ_CG_SUPPORT_NEW = (
    "    # [Genesis P65 v2] Context-aware downgrade. Keep UNIFORM_BATCH as the\n"
    "    # ClassVar default (full caps for non-spec-decode setups), and override\n"
    "    # `get_cudagraph_support` classmethod to downgrade to\n"
    "    # UNIFORM_SINGLE_TOKEN_DECODE only when speculative_config is active.\n"
    "    # Why: under spec-decode, K+1 batches hit _prefill_attention's\n"
    "    # structurally-wrong cudagraph capture bypass (cu_seqlens_k =\n"
    "    # cu_seqlens_q assumes first-chunk prefill, missing prior cached KV in\n"
    "    # TurboQuant compressed format). Captured kernel attends without context\n"
    "    # → drafter outputs collapse to high-bias tokens (e.g. `<tool_call>`\n"
    "    # cascade). The downgrade forces spec-verify K+1 batches to eager (the\n"
    "    # per-request continuation branch decompresses cache correctly).\n"
    "    # NOTE: vLLM compilation.py:1356 globally flips cudagraph_mode to\n"
    "    # PIECEWISE when our backend declares < UNIFORM_BATCH AND uniform_decode\n"
    "    # _query_len > 1. So even 1-token decode loses FULL cudagraph under\n"
    "    # spec-decode. A finer-grained per-batch dispatch would require\n"
    "    # upstream architecture change OR the proper P67 multi-query kernel.\n"
    "    # Reference: noonghunna #40880 + Genesis investigation 2026-04-25.\n"
    "    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH\n"
    "\n"
    "    @classmethod\n"
    "    def get_cudagraph_support(\n"
    "        cls,\n"
    "        vllm_config: \"VllmConfig\",\n"
    "        kv_cache_spec: \"AttentionSpec\",\n"
    "    ) -> AttentionCGSupport:\n"
    "        \"\"\"[Genesis P65 v2] Context-aware downgrade for spec-decode only.\"\"\"\n"
    "        if vllm_config.speculative_config is not None:\n"
    "            return AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE\n"
    "        return cls._cudagraph_support\n"
)


def _make_patcher() -> TextPatcher | None:
    target = resolve_vllm_file("v1/attention/backends/turboquant_attn.py")
    if target is None:
        return None
    return TextPatcher(
        patch_name="P65 turboquant_attn.py — spec-decode CG downgrade",
        target_file=str(target),
        marker=GENESIS_P65_MARKER,
        sub_patches=[
            TextPatch(name="p65_cg_support_downgrade",
                      anchor=TQ_CG_SUPPORT_OLD,
                      replacement=TQ_CG_SUPPORT_NEW,
                      required=True),
        ],
        upstream_drift_markers=[
            "[Genesis P65]",
            "AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE",
        ],
    )


def apply() -> tuple[str, str]:
    """Apply P65 — downgrade TurboQuant cudagraph support."""
    from vllm._genesis.dispatcher import should_apply, log_decision
    decision, reason = should_apply("P65")
    log_decision("P65", decision, reason)
    if not decision:
        return "skipped", reason

    if vllm_install_root() is None:
        return "skipped", "vllm install root not discoverable"

    patcher = _make_patcher()
    if patcher is None:
        return "skipped", "turboquant_attn.py not found"

    # Pre-flight
    if not os.path.isfile(patcher.target_file):
        return "skipped", f"target disappeared: {patcher.target_file}"
    with open(patcher.target_file) as f:
        content = f.read()
    if patcher.marker in content:
        pass  # idempotent
    else:
        for m in patcher.upstream_drift_markers:
            if m in content:
                return (
                    "skipped",
                    f"upstream drift marker {m!r} in {patcher.target_file} — "
                    "TurboQuant CG support already changed upstream.",
                )
        if patcher.sub_patches[0].anchor not in content:
            return (
                "skipped",
                f"required anchor not found — TurboQuant _cudagraph_support "
                "declaration may have changed; P65 cannot apply.",
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
        "P65 applied: TurboQuant CG support downgraded to "
        "UNIFORM_SINGLE_TOKEN_DECODE. Spec-verify K+1 batches now run eager "
        "(correct per-request continuation), 1-token decode batches still "
        "use cudagraph. Workaround for noonghunna #40880 MTP+TurboQuant bug."
    )
