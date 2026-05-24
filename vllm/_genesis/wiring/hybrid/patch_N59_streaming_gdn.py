# SPDX-License-Identifier: Apache-2.0
"""Wiring for PN59 — streaming GDN orchestrator (Variant D Phase 2).

Text-patches `vllm/model_executor/layers/fla/ops/chunk.py` to redirect
`chunk_gated_delta_rule_fwd` through Genesis's window-iterative
`streaming_chunk_gated_delta_rule_fwd` driver when eligible.

Eliminates Cliff 2b multi-turn OOM by replacing the `(B, NT, H, V, K)`
single-allocation peak (805 MiB at T=64K Genesis 27B Lorbus shapes)
with shape-keyed scratch pool of `(B, WINDOW_NT, H, V, K)` (~3-12 MiB).

Independent confirmation (issue #20, 2026-05-05) — noonghunna:
"the limitation is the triton kernel for cliff 2; doesn't appear with
llama.cpp" — exact materialization pattern this fix removes.

Architecture
------------
PN59 is a **single-anchor text patch** on the body of
`chunk_gated_delta_rule_fwd` orchestrator. Replacement wraps the
entire function body in a runtime dispatcher:

  - If GENESIS_ENABLE_PN59_STREAMING_GDN=1 AND eligible (single-seq,
    long-T, NVIDIA CUDA) → call `streaming_chunk_gated_delta_rule_fwd`
    (Genesis-managed window-iterative driver)
  - Otherwise → run vanilla code unchanged (zero-regression contract)

Strict no-regression: any failure in streaming path → fall through
to vanilla path with WARNING log.

Compatibility
-------------
- **PN50** GDN proj fusion — operates BEFORE chunk_gated_delta_rule
  (in gdn_linear_attn.py); orthogonal, no conflict
- **PN54** GDN contiguous dedup — operates on ssm_state read; orthogonal
- **P103** FLA Cliff2 chunked — operates AT outer-orchestrator level;
  PN59 supersedes when both ON (auto-fallthrough handles)
- **PN26b** sparse-V — non-GDN attention path; orthogonal

Default OFF until live A/B prod-validates on 27B Lorbus.

Author: Sandermage 2026-05-05, Variant D Phase 2.
Phase 1 numerical proof: tests/integration/test_streaming_gdn_numerical.py
Cross-engine references: llama.cpp ssm-scan.cu (register-streaming),
  Mamba2 ssd_combined (3-stage chunk split), FLA RFC #485.
"""
from __future__ import annotations

import logging
import os

from vllm._genesis.guards import resolve_vllm_file, vllm_install_root
from vllm._genesis.wiring.text_patch import (
    TextPatch,
    TextPatcher,
    TextPatchResult,
)

log = logging.getLogger("genesis.wiring.pn59_streaming_gdn")

GENESIS_PN59_MARKER = "Genesis PN59 streaming GDN orchestrator (Variant D Phase 2)"


def _is_enabled() -> bool:
    return os.environ.get(
        "GENESIS_ENABLE_PN59_STREAMING_GDN", ""
    ).strip().lower() in ("1", "true", "yes", "on")


# Anchor on the entire `chunk_gated_delta_rule_fwd` function body.
# Pristine upstream + post-Genesis state both have this signature
# unchanged (Genesis P103 modifies a different orchestrator wrap).
ANCHOR_OLD = (
    "def chunk_gated_delta_rule_fwd(\n"
    "    q: torch.Tensor,\n"
    "    k: torch.Tensor,\n"
    "    v: torch.Tensor,\n"
    "    g: torch.Tensor,\n"
    "    beta: torch.Tensor,\n"
    "    scale: float,\n"
    "    initial_state: torch.Tensor,\n"
    "    output_final_state: bool,\n"
    "    cu_seqlens: torch.Tensor | None = None,\n"
    "    chunk_indices: torch.Tensor | None = None,\n"
    "    chunk_offsets: torch.Tensor | None = None,\n"
    "):\n"
)

ANCHOR_NEW = (
    "def chunk_gated_delta_rule_fwd(\n"
    "    q: torch.Tensor,\n"
    "    k: torch.Tensor,\n"
    "    v: torch.Tensor,\n"
    "    g: torch.Tensor,\n"
    "    beta: torch.Tensor,\n"
    "    scale: float,\n"
    "    initial_state: torch.Tensor,\n"
    "    output_final_state: bool,\n"
    "    cu_seqlens: torch.Tensor | None = None,\n"
    "    chunk_indices: torch.Tensor | None = None,\n"
    "    chunk_offsets: torch.Tensor | None = None,\n"
    "):\n"
    "    # [Genesis PN59 Variant D Phase 2] streaming-GDN dispatch.\n"
    "    # When GENESIS_ENABLE_PN59_STREAMING_GDN=1 AND eligible single-seq\n"
    "    # long-T prefill, route through window-iterative driver to avoid\n"
    "    # full (B, NT, H, V, K) materialization (Cliff 2b OOM trigger).\n"
    "    # Strict no-regression: any failure → vanilla fallback below.\n"
    "    try:\n"
    "        from vllm._genesis.kernels.streaming_gdn_driver import (\n"
    "            streaming_chunk_gated_delta_rule_fwd as _genesis_pn59_streaming,\n"
    "        )\n"
    "        return _genesis_pn59_streaming(\n"
    "            q=q, k=k, v=v, g=g, beta=beta, scale=scale,\n"
    "            initial_state=initial_state,\n"
    "            output_final_state=output_final_state,\n"
    "            cu_seqlens=cu_seqlens,\n"
    "            chunk_indices=chunk_indices,\n"
    "            chunk_offsets=chunk_offsets,\n"
    "            chunk_local_cumsum=chunk_local_cumsum,\n"
    "            chunk_scaled_dot_kkt_fwd=chunk_scaled_dot_kkt_fwd,\n"
    "            solve_tril=solve_tril,\n"
    "            recompute_w_u_fwd=recompute_w_u_fwd,\n"
    "            chunk_gated_delta_rule_fwd_h=chunk_gated_delta_rule_fwd_h,\n"
    "            chunk_fwd_o=chunk_fwd_o,\n"
    "            SUPPRESS_LEVEL=SUPPRESS_LEVEL,\n"
    "        )\n"
    "    except Exception:\n"
    "        # Defensive: fall through to vanilla original code below\n"
    "        pass\n"
)


def _make_patcher() -> TextPatcher | None:
    target = resolve_vllm_file("model_executor/layers/fla/ops/chunk.py")
    if target is None:
        return None
    return TextPatcher(
        patch_name="PN59 streaming GDN orchestrator (Variant D Phase 2)",
        target_file=str(target),
        marker=GENESIS_PN59_MARKER,
        sub_patches=[
            TextPatch(
                name="pn59_chunk_orchestrator_dispatch",
                anchor=ANCHOR_OLD,
                replacement=ANCHOR_NEW,
                required=True,
            ),
        ],
        upstream_drift_markers=[
            # If upstream FLA lands #485 (Songlin Yang memory_efficient flag),
            # signature changes → drift detected → SKIP cleanly
            "memory_efficient",
            "streaming_window_chunks",
            # Or if upstream merges any equivalent
            "streaming_chunk_gated_delta_rule_fwd",
        ],
    )


def apply() -> tuple[str, str]:
    from vllm._genesis.dispatcher import log_decision, should_apply

    decision, reason = should_apply("PN59")
    log_decision("PN59", decision, reason)
    if not decision:
        return "skipped", reason
    if vllm_install_root() is None:
        return "skipped", "vllm install root not discoverable"
    patcher = _make_patcher()
    if patcher is None:
        return "skipped", "fla/ops/chunk.py not found"
    result, failure = patcher.apply()
    if result == TextPatchResult.APPLIED:
        return (
            "applied",
            "PN59 applied: streaming-GDN dispatcher inserted; runtime "
            "engages when single-seq long-T (eliminates Cliff 2b OOM)",
        )
    if result == TextPatchResult.IDEMPOTENT:
        return "applied", "already applied (idempotent)"
    if result == TextPatchResult.SKIPPED:
        msg = failure.reason if failure else "anchor not found"
        return ("skipped",
                f"{msg} — likely upstream merged #485-style fix or signature drift")
    return "failed", failure.reason if failure else "unknown failure"
