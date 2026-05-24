# SPDX-License-Identifier: Apache-2.0
"""Wiring for Patch 31 — fp32 upcast on MoE router softmax.

Scope of this wiring
--------------------
The Genesis kernel `vllm._genesis.kernels.router_softmax.router_softmax`
implements the bf16-collision-safe fp32-upcast variant. Where it can plug
into vLLM at runtime is limited because most MoE paths in our baseline
image route through:

  vllm._custom_ops.topk_softmax  (CUDA C++ kernel — not Python-rebindable)

via `vllm.model_executor.layers.fused_moe.router.fused_topk_router.fused_topk`.
That path computes softmax INSIDE the CUDA kernel so a Python-level rebind
has no effect on it.

What we CAN rebind cleanly:

  vllm.model_executor.layers.fused_moe.router.grouped_topk_router.grouped_topk
    → uses `scores = torch.softmax(gating_output, dim=-1)` at module level.

Models impacted by this rebind:
  - DeepSeek-V2/V3 family (uses grouped_topk + topk_group)
  - Mixtral 8×22B (some configurations)
  - Other MoE models with `scoring_func=='softmax'` + `num_expert_group>1`

Models NOT impacted (would require a different intervention):
  - Qwen3.6-MoE — uses the fused_topk path with the CUDA kernel.
  - Most "regular" MoE models with no expert grouping.

For Qwen3.6 specifically, the fp32 upcast benefit is achievable only by
a CUDA-kernel-level fix (out of scope for runtime patching). Our Python
wiring is documentation + future-proofing for grouped-MoE families on
the same Ampere targets.

Status: low-impact for our flagship workload (Qwen3.6), kept for
forward-compat on other MoE families.

Author: Sandermage(Sander)-Barzov Aleksandr, Ukraine, Odessa
"""
from __future__ import annotations

import logging
from typing import Any

from vllm._genesis.guards import is_cpu_only

log = logging.getLogger("genesis.wiring.p31_router_softmax")

# Marker attribute attached to our wrapper so we can detect re-apply.
_GENESIS_P31_MARKER_ATTR = "_genesis_p31_wrapped"


def _import_grouped_router() -> Any | None:
    """Try to import the grouped-topk router module."""
    try:
        from vllm.model_executor.layers.fused_moe.router import (
            grouped_topk_router,
        )
        return grouped_topk_router
    except ImportError as e:
        log.info("[Genesis P31] grouped_topk_router not importable: %s", e)
        return None
    except Exception as e:
        log.warning("[Genesis P31] unexpected import error: %s", e)
        return None


def apply() -> tuple[str, str]:
    """Wrap grouped_topk to upcast gating logits to fp32 before softmax.

    Never raises. Returns (status, reason).
    """
    if is_cpu_only():
        return "skipped", "CPU-only platform; fp32 upcast has no benefit here"

    # P52 (v7.9): MoE-active dispatch gate. grouped_topk only fires on MoE
    # models that use grouped routing. Skip the rebind on dense models to
    # keep dispatch logs clean.
    try:
        from vllm._genesis.model_detect import is_moe_model, log_skip
        if not is_moe_model():
            log_skip("P31 router softmax fp32", "dense model (no grouped routing)")
            return "skipped", "P52 dispatch: model has no MoE layers"
    except Exception as e:
        log.debug("[Genesis P31] model_detect probe failed (proceeding): %s", e)

    mod = _import_grouped_router()
    if mod is None:
        return "skipped", "grouped_topk_router module not in this vLLM build"

    # Candidate function names — try in order. Future-proofs against
    # upstream renames (e.g. `grouped_topk` → `grouped_topk_v2`).
    _CANDIDATE_FN_NAMES = (
        "grouped_topk",
        "grouped_topk_v2",
        "fused_grouped_topk",
    )
    target_fn = None
    target_fn_name = None
    for _name in _CANDIDATE_FN_NAMES:
        _fn = getattr(mod, _name, None)
        if _fn is not None:
            target_fn, target_fn_name = _fn, _name
            break
    if target_fn is None:
        return (
            "skipped",
            f"none of {list(_CANDIDATE_FN_NAMES)} present in "
            f"grouped_topk_router (upstream may have renamed)",
        )

    if getattr(target_fn, _GENESIS_P31_MARKER_ATTR, False):
        return "applied", "already wrapped (idempotent — process already patched)"

    import torch

    def _genesis_wrapped_grouped_topk(
        hidden_states: torch.Tensor,
        gating_output: torch.Tensor,
        *args, **kwargs,
    ):
        """Genesis P31 wrapper: upcast gating to fp32 before grouped_topk.

        The original function does `scores = torch.softmax(gating_output,
        dim=-1)` internally — by handing it fp32 logits, the softmax stays
        in fp32 and tail-expert collisions are avoided. After the topk
        decision, downstream weights cast back to original dtype.
        """
        if gating_output.dtype != torch.float32:
            gating_output = gating_output.float()
        return target_fn(hidden_states, gating_output, *args, **kwargs)

    setattr(_genesis_wrapped_grouped_topk, _GENESIS_P31_MARKER_ATTR, True)
    setattr(_genesis_wrapped_grouped_topk, "_genesis_p31_original", target_fn)
    setattr(mod, target_fn_name, _genesis_wrapped_grouped_topk)

    log.info(
        "[Genesis P31] wrapped grouped_topk_router.grouped_topk "
        "(fp32 upcast active for grouped-MoE models)"
    )
    return "applied", "grouped_topk wrapped (effective in this process)"


def is_applied() -> bool:
    mod = _import_grouped_router()
    if mod is None:
        return False
    fn = getattr(mod, "grouped_topk", None)
    if fn is None:
        return False
    return getattr(fn, _GENESIS_P31_MARKER_ATTR, False)


def revert() -> bool:
    mod = _import_grouped_router()
    if mod is None:
        return False
    fn = getattr(mod, "grouped_topk", None)
    if fn is None:
        return False
    if not getattr(fn, _GENESIS_P31_MARKER_ATTR, False):
        return False
    original = getattr(fn, "_genesis_p31_original", None)
    if original is None:
        return False
    setattr(mod, "grouped_topk", original)
    return True
