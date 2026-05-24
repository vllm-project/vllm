# SPDX-License-Identifier: Apache-2.0
"""Memory metrics + diagnostics for Genesis v7.0 preallocations.

Exposes a single diagnostic function `genesis_memory_summary()` that
collects **actual observed byte counts** from every Genesis buffer pool
running in the current process. Intended for use as a `/diag` endpoint,
post-warmup self-check, or operator `python3 -c` probe.

What this collects and why
--------------------------
We ship multiple profiler-visible buffer pools (P22 TQ K/V dequant,
P26 TQ prefill output + cu_2, P28 GDN core_attn_out, P32 cu_2, P33
synth_seq_lens). Each pool is keyed by shape/dtype/device — a single
`get_registry_info()` per manager already returns per-pool bytes.
This module **aggregates** all pools into one structure with the sum +
per-pool breakdown, so a single call answers "how much static VRAM does
Genesis add on top of upstream?".

Complements upstream
--------------------
vLLM has `torch.cuda.memory_stats()` and the Prometheus metrics at
`/metrics`, but those report the WHOLE process footprint. Neither
attributes bytes back to Genesis patches. This helper does exactly that
attribution so the operator can answer: "did P28 actually allocate
what CRIT-HW-1 expected, or did attach_buffer silently fall through to
None?".

Use at runtime
--------------
1. From inside vLLM container (exec):
    docker exec vllm-integration-v7 python3 -c \\
      "from vllm._genesis.memory_metrics import genesis_memory_summary; \\
       import json; print(json.dumps(genesis_memory_summary(), indent=2, default=str))"

2. As a post-warmup hook (already wired by caller — no auto-invoke
   here to keep this module dependency-free).

Author: Sandermage(Sander)-Barzov Aleksandr, Ukraine, Odessa
"""
from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger("genesis.memory_metrics")


def _humanize_bytes(n: int) -> str:
    for unit, power in [("B", 0), ("KiB", 10), ("MiB", 20), ("GiB", 30)]:
        if n < (1 << (power + 10)):
            if power == 0:
                return f"{n} B"
            return f"{n / (1 << power):.2f} {unit}"
    return f"{n / (1 << 30):.2f} GiB"


def _safe_call(fn: Any) -> dict:
    """Call a getter that might raise — return {} on any failure so the
    aggregate never crashes on one bad pool."""
    try:
        return fn()
    except Exception as e:
        return {"_error": f"{type(e).__name__}: {e}"}


def genesis_memory_summary() -> dict[str, Any]:
    """Collect bytes from every Genesis buffer pool + upstream torch stats.

    Return dict shape:
      {
        "total_genesis_bytes": int,
        "total_genesis_human": str,
        "per_pool": {
            "turboquant_buffer_manager": {...},
            "gdn_core_attn_manager": {...},
            "prealloc_framework": {...},
          },
        "torch_cuda": {
            "allocated": int,
            "reserved": int,
            "active_blocks": int,
            "max_allocated": int,
          }   # or {} if CUDA unavailable
      }
    """
    out: dict[str, Any] = {
        "total_genesis_bytes": 0,
        "total_genesis_human": "0 B",
        "per_pool": {},
        "torch_cuda": {},
    }

    # ── TurboQuantBufferManager (P22/P26/P32/P33) ──────────────────────
    try:
        from vllm._genesis.kernels.dequant_buffer import (
            TurboQuantBufferManager as _TQM,
        )
        info = _safe_call(_TQM.get_registry_info)
        out["per_pool"]["turboquant_buffer_manager"] = info
        out["total_genesis_bytes"] += int(info.get("total_bytes") or 0)
    except Exception as e:
        out["per_pool"]["turboquant_buffer_manager"] = {"_error": str(e)}

    # ── GdnCoreAttnManager (P28) ────────────────────────────────────────
    try:
        from vllm._genesis.kernels.gdn_core_attn_manager import (
            GdnCoreAttnManager as _GDN,
        )
        info = _safe_call(_GDN.get_registry_info)
        out["per_pool"]["gdn_core_attn_manager"] = info
        out["total_genesis_bytes"] += int(info.get("total_bytes") or 0)
    except Exception as e:
        out["per_pool"]["gdn_core_attn_manager"] = {"_error": str(e)}

    # ── P37 Shared MoE intermediate cache pool ──────────────────────────
    try:
        from vllm._genesis.kernels.moe_intermediate_cache import (
            get_registry_info as _moe_info,
        )
        info = _safe_call(_moe_info)
        out["per_pool"]["moe_intermediate_cache"] = info
        # Count P37 bytes only if the pool isn't ALREADY counted via
        # GenesisPreallocBuffer (it's allocated via torch.empty directly
        # in the P37 manager, not GPB — so it IS a distinct contribution).
        out["total_genesis_bytes"] += int(info.get("total_bytes") or 0)
    except Exception as e:
        out["per_pool"]["moe_intermediate_cache"] = {"_error": str(e)}

    # ── P39a FLA KKT persistent A pool ──────────────────────────────────
    try:
        from vllm._genesis.kernels.fla_kkt_buffer import (
            FlaKktBufferManager as _FLA,
        )
        info = _safe_call(_FLA.get_registry_info)
        out["per_pool"]["fla_kkt_buffer"] = info
        # Pool is allocated via GenesisPreallocBuffer → DO NOT add here
        # (already folded into prealloc_framework total below). Keep the
        # entry for per-pool visibility.
    except Exception as e:
        out["per_pool"]["fla_kkt_buffer"] = {"_error": str(e)}

    # ── P46 GDN gating g / beta_output buffers ─────────────────────────
    try:
        from vllm._genesis.kernels.gdn_gating_buffer import (
            GdnGatingBufferManager as _GAT,
        )
        info = _safe_call(_GAT.get_registry_info)
        out["per_pool"]["gdn_gating_buffer"] = info
        # Allocated via raw `torch.empty` (NOT through GPB) — contributes
        # to total_genesis_bytes directly.
        out["total_genesis_bytes"] += int(info.get("total_bytes") or 0)
    except Exception as e:
        out["per_pool"]["gdn_gating_buffer"] = {"_error": str(e)}

    # ── GenesisPreallocBuffer framework ────────────────────────────────
    try:
        from vllm._genesis.prealloc import GenesisPreallocBuffer as _GPB
        info = _safe_call(_GPB.get_registry_info)
        out["per_pool"]["prealloc_framework"] = info
        # Prealloc framework's total already includes TQ+GDN pools that
        # use it as backing. DO NOT double-count — expose for visibility
        # only, do not add to total.
    except Exception as e:
        out["per_pool"]["prealloc_framework"] = {"_error": str(e)}

    # ── torch.cuda stats — upstream reference, whole-process ────────────
    try:
        import torch
        if torch.cuda.is_available():
            out["torch_cuda"] = {
                "allocated": int(torch.cuda.memory_allocated()),
                "reserved": int(torch.cuda.memory_reserved()),
                "max_allocated": int(torch.cuda.max_memory_allocated()),
                "max_reserved": int(torch.cuda.max_memory_reserved()),
                "device_count": torch.cuda.device_count(),
                "current_device": torch.cuda.current_device(),
            }
            # Human forms for quick log scans
            out["torch_cuda"]["allocated_human"] = _humanize_bytes(
                out["torch_cuda"]["allocated"]
            )
            out["torch_cuda"]["reserved_human"] = _humanize_bytes(
                out["torch_cuda"]["reserved"]
            )
    except Exception as e:
        out["torch_cuda"] = {"_error": str(e)}

    out["total_genesis_human"] = _humanize_bytes(out["total_genesis_bytes"])
    return out


def log_genesis_memory(level: int = logging.INFO) -> None:
    """Convenience: log the summary at the given level. Safe to call at
    apply_all register time (outside any torch.compile region).

    Emits per-manager byte counts for ALL currently-registered pools
    (TQ, GDN, P37 MoE intermediate, P39a FLA KKT), plus torch.cuda
    allocator stats for cross-reference with vLLM's memory profiler
    numbers.
    """
    def _pool_bytes(key: str) -> int:
        return int(
            summary["per_pool"].get(key, {}).get("total_bytes") or 0
        )

    try:
        summary = genesis_memory_summary()
        log.log(
            level,
            "[Genesis memory] total=%s | TQ=%s | GDN=%s | MoE(P37)=%s | "
            "FLA-KKT(P39a)=%s | torch_alloc=%s reserved=%s",
            summary["total_genesis_human"],
            _humanize_bytes(_pool_bytes("turboquant_buffer_manager")),
            _humanize_bytes(_pool_bytes("gdn_core_attn_manager")),
            _humanize_bytes(_pool_bytes("moe_intermediate_cache")),
            _humanize_bytes(_pool_bytes("fla_kkt_buffer")),
            summary["torch_cuda"].get("allocated_human", "n/a"),
            summary["torch_cuda"].get("reserved_human", "n/a"),
        )
    except Exception as e:
        log.warning("[Genesis memory] summary failed: %s", e)
