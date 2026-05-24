# SPDX-License-Identifier: Apache-2.0
"""Wiring for Patch 37 — Shared MoE `intermediate_cache13` / `cache2` pool.

Problem recap
-------------
`_fused_marlin_moe` in `fused_marlin_moe.py` allocates two intermediate
tensors with `torch.empty(...)` on every MoE-layer forward call. For our
Qwen3.6-35B-A3B chunked-prefill at M=4096: ~553 MiB peak per layer ×
30 MoE layers → ~16 GiB of allocator traffic per step. Even with
`expandable_segments=True` the fragmentation competes with weight-load
slabs and can contribute to the 50k-prefill OOM we observed.

Fix
---
Replace the two `if cache is None: cache = torch.empty(...)` blocks
with a call to `GenesisMoEIntermediateCacheManager.acquire_cache{13,2}`.
The manager maintains a module-level shared pool keyed by
`(max_M_times_topk, N, K, num_shards, dtype, device_index)` — all TQ
MoE layers reuse ONE pool set because:

  - They all use the same `(N, K, num_shards)` — model-wide constants.
  - They execute SEQUENTIALLY per step — no cross-layer races.
  - Max M is bounded by scheduler's `max_num_batched_tokens`.

dynamo-safety
-------------
`_fused_marlin_moe` is called from `FusedMoE.forward` which runs inside
`aot_compile_fullgraph`. The manager's `acquire_cache*` functions are
decorated with `@torch._dynamo.allow_in_graph` — dynamo treats them as
opaque graph nodes with tensor outputs, which is the supported path for
side-effectful allocator logic.

Platform compatibility
----------------------
Gated by `GENESIS_ENABLE_P37=1` because:
  1. It's a new-in-v7.1 optimisation and operators should opt-in
     explicitly until we have 48h stability data from integration.
  2. `@torch._dynamo.allow_in_graph` behavior changed between torch
     versions; the env-gate lets us disable quickly if a future torch
     breaks tracing.
  3. Manager itself is always importable and always registers the
     buffer-pool API — only the text-patch is gated.

When gate is ON and platform supports (NVIDIA CUDA SM ≥ 8.0) the pool
engages. Otherwise each `acquire_*` call falls back to `torch.empty`
identical to upstream — ZERO regression on unsupported setups.

Upstream drift detection
------------------------
If `fused_marlin_moe.py` gains an upstream shared-pool implementation
(hypothetical PR #40655-class for MoE), the markers in
`UPSTREAM_DRIFT_MARKERS` self-retire our patch.

Author: Sandermage(Sander)-Barzov Aleksandr, Ukraine, Odessa
"""
from __future__ import annotations

import logging
import os

from vllm._genesis.guards import resolve_vllm_file, vllm_install_root
from vllm._genesis.wiring.text_patch import (
    TextPatch, TextPatcher, TextPatchResult,
)

log = logging.getLogger("genesis.wiring.p37_moe_intermediate_cache")

GENESIS_P37_MARKER = "Genesis P37 MoE intermediate cache pool v7.0"

UPSTREAM_DRIFT_MARKERS = [
    # Hypothetical signatures of an upstream shared-pool refactor.
    "_shared_intermediate_cache13",
    "_moe_intermediate_pool",
    "get_shared_moe_intermediate",
    # If someone adds a "class FusedMoEIntermediateCacheManager" upstream.
    "FusedMoEIntermediateCacheManager",
    # PR #41184 (bnellnm, OPEN 2026-04-29) — MoE Refactor: class rename
    # FusedMoE → RoutedExperts. If this lands our anchor on FusedMoE
    # method definitions will silently miss; watch for:
    "class RoutedExperts",
    "RoutedExperts(",
]


# Anchor: the exact two `if ... is None: ... = torch.empty(...)` blocks.
_OLD = (
    "    if intermediate_cache13 is None:\n"
    "        intermediate_cache13 = torch.empty(\n"
    "            (M * num_topk * max(w13_num_shards * N, K),),\n"
    "            device=hidden_states.device,\n"
    "            dtype=hidden_states.dtype,\n"
    "        )\n"
    "\n"
    "    if intermediate_cache2 is None:\n"
    "        intermediate_cache2 = torch.empty(\n"
    "            (M * num_topk, N),\n"
    "            device=hidden_states.device,\n"
    "            dtype=hidden_states.dtype,\n"
    "        )"
)

_NEW = (
    "    # [Genesis P37] Shared MoE intermediate-cache pool. All MoE layers\n"
    "    # in a step execute sequentially and use identical (N, K, topk,\n"
    "    # shards) config, so one pool is safe. Saves ~553 MiB × N_moe_layers\n"
    "    # allocator churn per step on our Qwen3.6-35B-A3B chunked-prefill\n"
    "    # M=4096 profile. Falls back to upstream per-call torch.empty if\n"
    "    # GENESIS_ENABLE_P37 is unset or platform is unsupported.\n"
    "    if intermediate_cache13 is None:\n"
    "        from vllm._genesis.kernels.moe_intermediate_cache import (\n"
    "            acquire_cache13 as _genesis_acquire_cache13,\n"
    "        )\n"
    "        intermediate_cache13 = _genesis_acquire_cache13(\n"
    "            M=M, num_topk=num_topk,\n"
    "            w13_num_shards=w13_num_shards, N=N, K=K,\n"
    "            dtype=hidden_states.dtype,\n"
    "            device=hidden_states.device,\n"
    "        )\n"
    "\n"
    "    if intermediate_cache2 is None:\n"
    "        from vllm._genesis.kernels.moe_intermediate_cache import (\n"
    "            acquire_cache2 as _genesis_acquire_cache2,\n"
    "        )\n"
    "        intermediate_cache2 = _genesis_acquire_cache2(\n"
    "            M=M, num_topk=num_topk, N=N,\n"
    "            dtype=hidden_states.dtype,\n"
    "            device=hidden_states.device,\n"
    "        )"
)


def _make_patcher() -> TextPatcher | None:
    target = resolve_vllm_file(
        "model_executor/layers/fused_moe/fused_marlin_moe.py"
    )
    if target is None:
        return None
    return TextPatcher(
        patch_name="P37 MoE intermediate cache pool",
        target_file=target,
        marker=GENESIS_P37_MARKER,
        sub_patches=[
            TextPatch(
                name="p37_moe_intermediate_pool",
                anchor=_OLD,
                replacement=_NEW,
                required=True,
            ),
        ],
        upstream_drift_markers=UPSTREAM_DRIFT_MARKERS,
    )


def _is_enabled() -> bool:
    """Env-gate check. Re-reads env every apply() call — OK because
    apply() runs at register time, not in traced region."""
    return os.environ.get("GENESIS_ENABLE_P37", "").strip().lower() in (
        "1", "true", "yes", "on",
    )


def apply() -> tuple[str, str]:
    """Apply P37 wiring. Never raises.

    v7.12: consults `config_detect.should_apply("P37")` first.
    Skipped automatically if `max_num_seqs < 8` (MoE pool benefit
    marginal at low concurrency).
    """
    try:
        from vllm._genesis import config_detect
        ok, reason = config_detect.should_apply("P37")
        if not ok:
            return "skipped", reason
    except Exception:
        pass  # config_detect optional; fall through to legacy logic

    # Always run warm_up so the manager's `should_apply` cache is populated
    # even when the text-patch is disabled — this keeps the manager API
    # usable for operators who pre-acquire buffers manually.
    try:
        from vllm._genesis.kernels.moe_intermediate_cache import warm_up
        warm_up()
    except Exception as e:
        log.info("[Genesis P37] warm_up failed (non-fatal): %s", e)

    if not _is_enabled():
        return (
            "skipped",
            "opt-in only — set GENESIS_ENABLE_P37=1 to engage. "
            "Manager API is registered and usable independently of "
            "this text-patch.",
        )

    # P52 (v7.9): MoE-active dispatch gate. If config-probe says this model
    # has no MoE layers, skip the text-patch entirely. The manager API is
    # still registered (warm_up ran above) so tests / manual callers work.
    try:
        from vllm._genesis.model_detect import is_moe_model, log_skip
        if not is_moe_model():
            log_skip("P37 MoE intermediate cache pool", "dense model (no MoE layers)")
            return "skipped", "P52 dispatch: model has no MoE layers"
    except Exception as e:
        log.debug("[Genesis P37] model_detect probe failed (proceeding): %s", e)

    if vllm_install_root() is None:
        return "skipped", "vllm install root not discoverable"

    patcher = _make_patcher()
    if patcher is None:
        return "skipped", "fused_marlin_moe.py not found"

    result, failure = patcher.apply()
    if result == TextPatchResult.APPLIED:
        return (
            "applied",
            "intermediate_cache13 + cache2 rewired to shared pool "
            "(saves allocator churn across N_moe_layers per step)",
        )
    if result == TextPatchResult.IDEMPOTENT:
        return "applied", "already applied this image layer (idempotent)"
    if result == TextPatchResult.SKIPPED:
        return "skipped", failure.reason if failure else "unknown skip"
    return "failed", failure.reason if failure else "unknown failure"
