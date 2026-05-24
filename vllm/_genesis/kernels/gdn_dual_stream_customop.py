# SPDX-License-Identifier: Apache-2.0
"""P7b — GDN dual-stream parallelism via `torch.library.custom_op`.

Problem (continuation of P7)
---------------------------
Original P7 (`kernels/gdn_dual_stream.py::DualStreamDispatcher`)
issues two `self.in_proj_*(hidden_states)` GEMMs on separate CUDA
streams with event sync. The idiom is correct in eager mode and
measured +8% on A5000 Qwen3-Next, but it **breaks `torch.compile(
fullgraph=True)`**: dynamo cannot symbolically trace
`torch.cuda.Stream()` / `torch.cuda.Event()` / `with torch.cuda.stream()`.
On vLLM's default `aot_compile_fullgraph` (mode=3) path, P7 is
force-skipped and the parallelism win is lost.

Fix (P7b, this module)
----------------------
Wrap the dual-GEMM as a single **opaque** `torch.library.custom_op`.
Dynamo treats custom ops as uninterpreted nodes — it emits a call
to our op and DOES NOT trace through the body. Our body is free to
use CUDA streams / events with no compile-graph impact.

The op signature is functional (mutates_args=()) and returns a
`(Tensor, Tensor)` tuple for `(mixed_qkvz, ba)`. Fake impl is a
pure-shape inference using `torch.empty_strided`.

Compat
------
- Requires `torch.library.custom_op` (PyTorch ≥ 2.4, available on
  our dev134 stack).
- Enabled via `GENESIS_ENABLE_P7B=1`. OFF by default so operators
  explicitly benchmark before enabling — P7b changes op graph
  topology and could interact with compile-cache hashing.
- Platform-skip fallback: on non-CUDA hosts, the op is NEVER
  registered → wiring falls through to upstream serial call.

Author: Sandermage(Sander)-Barzov Aleksandr, Ukraine, Odessa
Status: v7.5 implementation (opt-in)
"""
from __future__ import annotations

import logging
import os
from typing import Optional, Tuple

import torch

log = logging.getLogger("genesis.gdn_dual_stream_customop")

_ENV_ENABLE_P7B = "GENESIS_ENABLE_P7B"


def is_p7b_enabled() -> bool:
    return os.environ.get(_ENV_ENABLE_P7B, "").strip().lower() in (
        "1", "true", "yes", "on",
    )


def should_apply() -> bool:
    """Platform gate: NVIDIA CUDA SM≥8.0 + env opt-in."""
    if not is_p7b_enabled():
        return False
    from vllm._genesis.guards import is_nvidia_cuda, is_sm_at_least
    if not is_nvidia_cuda():
        return False
    if not is_sm_at_least(8, 0):
        return False
    return True


# Module-level stream cache — one side stream PER DEVICE. Safe to
# allocate lazily from the custom-op body (body runs outside any
# compile/capture region). Cache is keyed by device index so multi-GPU
# TP deployments don't contend on a single stream.
_SIDE_STREAM: dict[int, "torch.cuda.Stream"] = {}


def _get_side_stream(device: torch.device) -> Optional["torch.cuda.Stream"]:
    """Return a cached CUDA side stream for the given device.

    Returns None on non-CUDA hosts or alloc failure. PyTorch CUDA
    semantics notes recommend `wait_stream`-based cross-stream sync
    over `wait_event` for this pattern — simpler and equivalent.
    """
    if not torch.cuda.is_available():
        return None
    idx = device.index if device.index is not None else 0
    s = _SIDE_STREAM.get(idx)
    if s is not None:
        return s
    try:
        s = torch.cuda.Stream(device=device)
        _SIDE_STREAM[idx] = s
        log.info("[P7b] allocated side CUDA stream for device %s", device)
    except Exception as e:
        log.info(
            "[P7b] side stream alloc on %s failed: %s — falling back",
            device, e,
        )
    return s


# ──────────────────────────────────────────────────────────────────
# Custom op registration (lazy — only when platform matches)
# ──────────────────────────────────────────────────────────────────

_OP_QUALNAME = "genesis::dual_linear_parallel"
_op_registered = False


# ─── Genesis library (vLLM canonical pattern, v7.66 refactor) ─────────
#
# Same Library("genesis", "FRAGMENT") namespace shared with PN25's
# silu_and_mul_pooled. FRAGMENT mode means each worker process can
# create its own Library object pointing at the same global "genesis"
# namespace without conflict. Reference: vllm/utils/torch_utils.py:896.

try:
    from torch.library import Library
    _HAS_TORCH_LIBRARY = True
except ImportError:
    Library = None  # type: ignore
    _HAS_TORCH_LIBRARY = False


_GENESIS_LIB = None


def _make_genesis_lib():
    """Construct or fetch the module-level `genesis` FRAGMENT Library.

    Idempotent across re-imports. Each worker process creates its own
    object pointing at the shared "genesis" namespace.
    """
    global _GENESIS_LIB
    if _GENESIS_LIB is None and _HAS_TORCH_LIBRARY:
        _GENESIS_LIB = Library("genesis", "FRAGMENT")
    return _GENESIS_LIB


def _dual_linear_parallel_impl(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    b1: Optional[torch.Tensor],
    w2: torch.Tensor,
    b2: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Real (eager) implementation — registered as the CUDA dispatch.

    Canonical PyTorch cross-stream sync per `notes/cuda.html`:
    `side.wait_stream(current)` before issue, `current.wait_stream(
    side)` after. Equivalent to explicit events but simpler and
    officially recommended.

    Supports 2-D `(N, K)` and 3-D `(B, N, K)` inputs since
    `nn.Linear` is shape-polymorphic and GDN in_proj is called with
    `hidden_states: (num_tokens, hidden_size)`.
    """
    import torch.nn.functional as F

    side = _get_side_stream(hidden_states.device)
    if side is None:
        # No CUDA or side stream alloc failed — serial fallback.
        return (
            F.linear(hidden_states, w1, b1),
            F.linear(hidden_states, w2, b2),
        )

    current = torch.cuda.current_stream(hidden_states.device)
    # Side waits for whatever produced `hidden_states` on current.
    side.wait_stream(current)
    # GEMM-B on side stream (issues async).
    with torch.cuda.stream(side):
        out2 = F.linear(hidden_states, w2, b2)
    # GEMM-A on current stream (true parallelism — overlaps with B).
    out1 = F.linear(hidden_states, w1, b1)
    # Re-join: current waits for side before any consumer sees out2.
    current.wait_stream(side)
    return out1, out2


def _dual_linear_parallel_fake(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    b1: Optional[torch.Tensor],
    w2: torch.Tensor,
    b2: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Shape inference for dynamo. Runs in `FakeTensorMode`.

    `F.linear(X:(..., K), W:(M, K))` → `(..., M)`. Preserves all
    leading dims so 2-D, 3-D, and any other rank works as upstream.
    """
    lead = list(hidden_states.shape[:-1])
    out1 = hidden_states.new_empty(lead + [w1.shape[0]])
    out2 = hidden_states.new_empty(lead + [w2.shape[0]])
    return out1, out2


def _register_op_once() -> bool:
    """Register `genesis::dual_linear_parallel` via vLLM canonical
    `direct_register_custom_op` pattern.

    Idempotent. Sister to PN25's silu_and_mul_pooled refactor in v7.66
    — same Library + direct_register pattern, same fork-safe pre-check.

    Why this is safer than v7.65's `@custom_op` approach
    ----------------------------------------------------
    v7.65 (commit d92bcb3) wrapped `@torch.library.custom_op(...)` in
    a hasattr() pre-check guard. That works, but @custom_op triggers
    `torch.library.infer_schema()` at decoration time. If decoration
    happens inside a torch.compile region, Dynamo refuses to trace
    `infer_schema` and crashes with:

        torch._dynamo.exc.Unsupported: Attempted to call function
        marked as skipped: infer_schema

    `direct_register_custom_op` calls `infer_schema` at module-import
    time (synchronously, BEFORE any dynamo trace context exists), so
    even if the op is registered in a worker process, Dynamo never
    sees the infer_schema call.

    Bug class reported by noonghunna at issue #16 against PN25's
    silu_and_mul_pooled. Applied same v7.66 refactor here as
    preventive measure — P7b would hit the same crash on any
    single-GPU spawn config (1×3090, 1×4090, etc.) if a future
    workload routes GDN dual-stream through a torch.compile region.
    """
    global _op_registered
    if _op_registered:
        return True

    if not _HAS_TORCH_LIBRARY:
        log.info(
            "[P7b] torch.library.Library not available — "
            "falling back to serial path"
        )
        return False

    # [Genesis #16 sister-fix, retained from v7.65] Pre-check global
    # C++ registry. If op is already registered (parent process or
    # earlier import — survives spawn), sync local flag and return
    # True without calling Library.define() (which would raise on
    # duplicate name).
    try:
        if hasattr(torch.ops, "genesis") and hasattr(
            torch.ops.genesis, "dual_linear_parallel"
        ):
            _op_registered = True
            log.info(
                "[P7b] op %s already globally registered — synced "
                "local flag (worker-spawn path; v7.66 direct_register)",
                _OP_QUALNAME,
            )
            return True
    except (AttributeError, RuntimeError):
        # Namespace doesn't exist yet — fall through to registration.
        pass

    try:
        # vLLM canonical helper — drops infer_schema overhead and
        # keeps schema computation outside dynamo trace context.
        from vllm.utils.torch_utils import direct_register_custom_op
        genesis_lib = _make_genesis_lib()
        if genesis_lib is None:
            log.info("[P7b] Library construction failed")
            return False
        direct_register_custom_op(
            op_name="dual_linear_parallel",
            op_func=_dual_linear_parallel_impl,
            mutates_args=[],
            fake_impl=_dual_linear_parallel_fake,
            target_lib=genesis_lib,
        )
    except ImportError:
        # Older vLLM without direct_register_custom_op — fall back to
        # @custom_op path (still has fork-safe pre-check above).
        log.info(
            "[P7b] vllm.utils.torch_utils.direct_register_custom_op "
            "not available — falling back to @custom_op path"
        )
        try:
            custom_op = getattr(torch.library, "custom_op", None)
            if custom_op is None:
                log.info(
                    "[P7b] torch.library.custom_op also unavailable "
                    "(torch<2.4) — falling back to serial path"
                )
                return False

            @custom_op(_OP_QUALNAME, mutates_args=(), device_types=("cuda",))
            def _dual_linear_parallel_decorated(
                hidden_states: torch.Tensor,
                w1: torch.Tensor,
                b1: Optional[torch.Tensor],
                w2: torch.Tensor,
                b2: Optional[torch.Tensor],
            ) -> Tuple[torch.Tensor, torch.Tensor]:
                return _dual_linear_parallel_impl(
                    hidden_states, w1, b1, w2, b2,
                )

            @_dual_linear_parallel_decorated.register_fake
            def _dual_linear_parallel_decorated_fake(
                hidden_states: torch.Tensor,
                w1: torch.Tensor,
                b1: Optional[torch.Tensor],
                w2: torch.Tensor,
                b2: Optional[torch.Tensor],
            ) -> Tuple[torch.Tensor, torch.Tensor]:
                return _dual_linear_parallel_fake(
                    hidden_states, w1, b1, w2, b2,
                )
        except Exception as e:
            log.info("[P7b] @custom_op fallback registration failed: %s", e)
            return False
    except Exception as e:
        log.info("[P7b] direct_register_custom_op failed: %s", e)
        return False

    _op_registered = True
    log.info(
        "[P7b] registered custom op %s via direct_register_custom_op "
        "(vLLM canonical, v7.66 — fork-safe + Inductor-opaque)",
        _OP_QUALNAME,
    )
    return True


def dual_linear_parallel(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    b1: Optional[torch.Tensor],
    w2: torch.Tensor,
    b2: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Public entry point.

    - If the custom op is registered (P7b enabled + CUDA available):
      dispatches through `torch.ops.genesis.dual_linear_parallel`
      (graph-safe, dynamo-traceable as opaque).
    - Else: plain `F.linear` pair (serial, identical to upstream).
    """
    if should_apply() and _register_op_once():
        return torch.ops.genesis.dual_linear_parallel(
            hidden_states, w1, b1, w2, b2,
        )
    # Fallback — serial
    import torch.nn.functional as F
    return F.linear(hidden_states, w1, b1), F.linear(hidden_states, w2, b2)
