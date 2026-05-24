# SPDX-License-Identifier: Apache-2.0
#
# ================================================================
# v7.67 design candidate (NOT shipped) — `@torch.compiler.disable`
# ================================================================
#
# SGLang ships `@torch.compiler.disable` on FLA's chunk_gated_delta_rule
# (python/sglang/srt/layers/attention/fla/chunk.py) and on its Triton
# attention paths (python/sglang/srt/layers/attention/triton_backend.py).
# This is a SIMPLER mechanism for Inductor opacity than custom_op
# registration:
#
#     @staticmethod
#     @torch.compiler.disable
#     def forward_native(x):
#         # body change: acquire from FFNIntermediateCache pool directly
#         ...
#
# vs the ~300-line registration module here. Same Inductor opacity
# (graph-break instead of opaque-op-node; compute cost equivalent —
# both atomic FX nodes). NO fork-safety surface. NO fake_impl. NO
# schema introspection.
#
# Why we didn't ship v2 in v7.66 (the current refactor):
#   - v7.66 (direct_register_custom_op) is a verified incremental
#     improvement over v7.65 with same external semantics
#   - PN25-v2 changes pool-acquire semantics (pool accessed inside
#     `forward_native` body directly, not through a custom_op → must
#     verify pointer-stable views don't break surrounding compile
#     context)
#   - Requires A/B validation against noonghunna's OpenCode reproducer
#     (5,900-char IDE-agent prompt) which we can't run locally
#
# When to ship v2:
#   - After noonghunna validates current v7.66 PN25 closes Cliff 1
#     mech B on club-3090 long-text / long-vision configs
#   - Then a v7.67 follow-up swaps the registration mechanism while
#     keeping the proven body-change pattern
#
# ================================================================

"""PN25 — `silu_and_mul` as torch.library.custom_op (Inductor-safe pool).

Problem (continuation of PN12)
------------------------------
PN12 text-patches `SiluAndMul.forward_cuda` to acquire its
`[M, intermediate_size]` BF16/FP16 transient from `FFNIntermediateCache`
instead of `torch.empty()`. That works in eager mode.

Reported by noonghunna 2026-04-30 in club-3090#16, with confirmation
from VolandBerlioz on a real OpenCode workload (29K sys+tools prefill,
24 GB single 3090): when `custom_ops=["none"]` is the default (which is
typical under V1 `aot_compile_fullgraph`), `SiluAndMul.__call__`
dispatches to `forward_native`, NOT `forward_cuda`. `forward_native`
is

    @staticmethod
    def forward_native(x):
        d = x.shape[-1] // 2
        return F.silu(x[..., :d]) * x[..., d:]

torch.compile's Inductor traces this body into a fused kernel and
issues its own `empty_strided_cuda((s18, intermediate_size), fp16)`
for the multiplication output. PN12's pool never gets a chance —
the patched `forward_cuda` method is never reached.

Symptom on a 24 GB 3090 + Lorbus 27B (intermediate=17408) at
`max_num_batched_tokens=4128`: 137.6 MiB allocation, 131.75 MiB free,
OOM. Cliff 1 mech B fires on real workloads while our verify-stress
25K synthetic happens to hit shapes that DO reach eager forward_cuda
and pass.

Genesis stack vulnerability
---------------------------
Same architectural flaw exists in our PN12 — we only patch
`forward_cuda`. Our 27B PROD configs avoid the inductor path because
`--cudagraph-mode=PIECEWISE` + offline-quant INT4 short-circuits the
compile pipeline on this kernel. But long-context + chunked prefill
under a future Inductor-default config could hit it.

Fix design (this module)
------------------------
Register `genesis::silu_and_mul_pooled` as `torch.library.custom_op`
with `device_types=("cuda",)`. Inductor treats custom ops as opaque
nodes — emits a call to the op and does NOT trace through the body.
Inside the op body we run the same eager logic as PN12's patched
forward_cuda: acquire output from `FFNIntermediateCache.acquire_silu_out`
when the [M, d] 2-D shape matches, fall back to `torch.empty` otherwise,
then dispatch to the underlying CUDA `silu_and_mul` kernel.

Companion patch PN25 (`patch_N25_silu_inductor_safe_pool.py`) edits
`SiluAndMul.forward_native` to route through this op when available.
PN12 stays as the eager-path patch on `forward_cuda`. Both can run
simultaneously without conflict — `forward_cuda` is called when
`custom_ops=["+silu_and_mul"]`, `forward_native` is called otherwise.

Composition with PN12
---------------------
PN12 patches `forward_cuda` (eager dispatch).
PN25 patches `forward_native` via opaque op (compile dispatch).

Together: both paths acquire from the same `FFNIntermediateCache`
pool. No state collision — pool is keyed by (intermediate_size, dtype,
device), and forward is strictly sequential in vLLM's schedule.

If only PN12 enabled: eager workloads work, compile path leaks.
If only PN25 enabled: compile workloads work, eager path leaks.
If both enabled: full coverage. Recommended for any inductor-heavy
config (35B FP8 + future MoE; club-3090 long-text/long-vision).

Compat
------
- Requires `torch.library.custom_op` (PyTorch ≥ 2.4, available on
  current vLLM nightly).
- Enabled via `GENESIS_ENABLE_PN25_SILU_INDUCTOR_SAFE=1`. OFF by
  default.
- Falls back gracefully if torch < 2.4 OR `torch.ops._C.silu_and_mul`
  is missing (CPU-only build).

Author: Sandermage(Sander) Barzov Aleksandr, Ukraine, Odessa
Cross-engine inspiration:
  - club-3090#16 (noonghunna independent work-in-progress)
  - Genesis P7b `gdn_dual_stream_customop.py` (custom_op template)
"""
from __future__ import annotations

import logging
import os

import torch

log = logging.getLogger("genesis.silu_and_mul_customop")

_ENV_ENABLE_PN25 = "GENESIS_ENABLE_PN25_SILU_INDUCTOR_SAFE"


def is_pn25_enabled() -> bool:
    return os.environ.get(_ENV_ENABLE_PN25, "").strip().lower() in (
        "1", "true", "yes", "on",
    )


def should_apply() -> bool:
    """Platform gate: NVIDIA CUDA + env opt-in."""
    if not is_pn25_enabled():
        return False
    from vllm._genesis.guards import is_nvidia_cuda
    if not is_nvidia_cuda():
        return False
    if not torch.cuda.is_available():
        return False
    return True


_OP_QUALNAME = "genesis::silu_and_mul_pooled"
_op_registered = False


def _silu_and_mul_native_fallback(x: torch.Tensor) -> torch.Tensor:
    """Pure-PyTorch fallback when CUDA op `_C.silu_and_mul` is not present.

    Equivalent to `forward_native`. Allocates fresh — no pool benefit,
    but preserves correctness. Hit only on CPU-only builds or in tests.
    """
    import torch.nn.functional as F
    d = x.shape[-1] // 2
    return F.silu(x[..., :d]) * x[..., d:]


# ─── Genesis library (vLLM canonical pattern, v7.66 refactor) ─────────
#
# Module-level `Library("genesis", "FRAGMENT")` — `FRAGMENT` mode
# permits multiple Library objects with the same name to coexist, which
# is the spawn-safe property we need: each worker process re-imports
# this module and creates a fresh `_GENESIS_LIB` against the same
# global "genesis" namespace without conflict.
#
# Reference: vLLM canonical pattern at vllm/utils/torch_utils.py:896
# (`vllm_lib = Library("vllm", "FRAGMENT")`) + direct_register_custom_op
# at L899. By using the same Library/FRAGMENT idiom Genesis aligns with
# vLLM core convention — easier upstream review if this pattern ever
# gets contributed back, and matches SGLang's approach in
# python/sglang/srt/utils/custom_op.py.
#
# Fork-safety mechanism (replaces v7.65 d92bcb3 `hasattr` pre-check
# pattern that needed to wrap @custom_op decoration):
#   - `Library.define()` raises if op_name already defined in the
#     namespace. In a fresh worker process, the C++ registry STILL
#     has the op from the parent's earlier `define()` call (C++ state
#     survives spawn), so the worker's `define()` would fail.
#   - We pre-check `torch.ops.genesis.silu_and_mul_pooled` existence
#     BEFORE calling `define()` — the same belt-and-suspenders SGLang
#     uses (custom_op.py:155 `if not hasattr(torch.ops.sglang,
#     op_name)`). On hit: skip define, sync local flag.
#   - `direct_register_custom_op` calls `infer_schema` at registration
#     time (synchronously), NOT inside dynamo trace at first call. So
#     even when define() runs in a worker, `infer_schema` runs in
#     normal Python context — no Dynamo "skipped frame" crash.

try:
    from torch.library import Library
    _HAS_TORCH_LIBRARY = True
except ImportError:
    Library = None  # type: ignore
    _HAS_TORCH_LIBRARY = False


# Created lazily inside _register_op_once() so import of this module
# doesn't have side-effects on torch.library state. Once created, the
# module-level reference keeps it alive for the operator lifetime
# (per direct_register_custom_op docstring caveat).
_GENESIS_LIB = None


def _make_genesis_lib():
    """Construct or fetch the module-level `genesis` FRAGMENT Library.

    Idempotent. Each worker process gets its own object pointing at
    the shared "genesis" namespace.
    """
    global _GENESIS_LIB
    if _GENESIS_LIB is None and _HAS_TORCH_LIBRARY:
        _GENESIS_LIB = Library("genesis", "FRAGMENT")
    return _GENESIS_LIB


def _silu_and_mul_pooled_impl(x: torch.Tensor) -> torch.Tensor:
    """Real (eager) implementation — registered as the CUDA dispatch.

    For 2-D `(M, 2*d)` tensors, acquires output from the shared
    `FFNIntermediateCache` pool. For 3-D `(B, S, 2*d)` we fall back
    to `torch.empty` because the pool is keyed on `(num_tokens,
    intermediate_size)` and 3-D shapes only appear in non-prefill
    paths where the alloc is small enough to not matter.

    Module-level (not nested in `_register_op_once`) so it can be
    referenced by name from `direct_register_custom_op` AND so the
    fake_impl can mirror the same shape-only logic.
    """
    has_cuda_op = (
        hasattr(torch.ops, "_C") and
        hasattr(torch.ops._C, "silu_and_mul")
    )
    d = x.shape[-1] // 2

    if has_cuda_op and x.dim() == 2:
        try:
            from vllm._genesis.kernels.ffn_intermediate_cache import (
                FFNIntermediateCache as _Cache,
            )
            if _Cache.is_production_eligible():
                out = _Cache.acquire_silu_out(
                    num_tokens=x.shape[0],
                    intermediate_size=d,
                    dtype=x.dtype, device=x.device,
                )
                torch.ops._C.silu_and_mul(out, x)
                return out
        except Exception as e:
            log.debug("[PN25] pool acquire failed, fallback: %s", e)

    if has_cuda_op:
        output_shape = x.shape[:-1] + (d,)
        out = torch.empty(output_shape, dtype=x.dtype, device=x.device)
        torch.ops._C.silu_and_mul(out, x)
        return out

    return _silu_and_mul_native_fallback(x)


def _silu_and_mul_pooled_fake(x: torch.Tensor) -> torch.Tensor:
    """Shape-inference impl for dynamo tracing.

    Returns an empty tensor of the correct shape; dynamo never
    executes the body so this is never observed at runtime — only
    used for output shape propagation through the compiled graph.
    """
    d = x.shape[-1] // 2
    output_shape = x.shape[:-1] + (d,)
    return torch.empty(output_shape, dtype=x.dtype, device=x.device)


def _register_op_once() -> bool:
    """Register `genesis::silu_and_mul_pooled` via vLLM canonical
    `direct_register_custom_op` pattern.

    Idempotent — and **fork-safe across worker spawn** (issue #16).
    Returns True on success. On any failure (torch too old, op already
    globally registered, dispatch_key resolution fail) returns False
    and PN25 wiring will fall back to upstream `forward_native`.

    Cross-process registration model (v7.66 refactor)
    -------------------------------------------------
    `_op_registered` is a module-level Python flag — it RESETS to False
    in each worker process when vLLM spawns workers via
    `VLLM_WORKER_MULTIPROC_METHOD=spawn` (fresh interpreter, re-imports
    this module). The C++ `torch.ops.genesis.silu_and_mul_pooled`
    state persists across spawn though, so workers detect existing
    global registration BEFORE attempting `Library.define()` — which
    would raise on duplicate name.

    Why this is safer than v7.65's `@custom_op` approach
    ----------------------------------------------------
    The original v7.65 fix (commit d92bcb3) wrapped the
    `@torch.library.custom_op(...)` decorator in the same pre-check
    guard. That works, but `@custom_op` triggers
    `torch.library.infer_schema()` at decoration time, which Dynamo
    refuses to trace if the call happens inside a torch.compile
    region (see noonghunna issue #16 reproducer). The v7.66
    `direct_register_custom_op` path:

      1. Calls `infer_schema` at module import time (BEFORE any
         dynamo trace context exists)
      2. Uses raw `Library.define()` + `Library.impl()` — no decorator
         magic, no per-call schema introspection
      3. Per `direct_register_custom_op` docstring:
         *"`torch.library.custom_op` can have significant overhead
         because it needs to consider complicated dispatching logic.
         This function directly registers a custom op and dispatches
         it to the CUDA backend."*
    """
    global _op_registered
    if _op_registered:
        return True

    if not _HAS_TORCH_LIBRARY:
        log.info(
            "[PN25] torch.library.Library not available — "
            "falling back to vanilla forward_native"
        )
        return False

    # [Genesis #16 fork-safety guard, retained from v7.65]
    # Check global torch.library registry FIRST. Survives worker-fork/spawn
    # even though our in-process flag does not. If the op is already
    # globally registered (parent process or earlier import), we just
    # sync our local flag and return True — no `define()` call, no
    # duplicate-name raise.
    try:
        if hasattr(torch.ops, "genesis") and hasattr(
            torch.ops.genesis, "silu_and_mul_pooled"
        ):
            _op_registered = True
            log.info(
                "[PN25] op %s already globally registered — synced "
                "local flag (worker-spawn path; v7.66 direct_register)",
                _OP_QUALNAME,
            )
            return True
    except (AttributeError, RuntimeError):
        # torch.ops.genesis namespace may not exist yet — fall through.
        pass

    try:
        # Try vLLM's canonical helper first — drops infer_schema
        # overhead and keeps schema computation outside dynamo trace.
        from vllm.utils.torch_utils import direct_register_custom_op
        genesis_lib = _make_genesis_lib()
        if genesis_lib is None:
            log.info("[PN25] Library construction failed")
            return False
        direct_register_custom_op(
            op_name="silu_and_mul_pooled",
            op_func=_silu_and_mul_pooled_impl,
            mutates_args=[],
            fake_impl=_silu_and_mul_pooled_fake,
            target_lib=genesis_lib,
        )
    except ImportError:
        # Older vLLM without direct_register_custom_op — fall back to
        # the v7.65 @custom_op path (still has fork-safe pre-check
        # above).
        log.info(
            "[PN25] vllm.utils.torch_utils.direct_register_custom_op "
            "not available — falling back to @custom_op path"
        )
        try:
            custom_op = getattr(torch.library, "custom_op", None)
            if custom_op is None:
                log.info(
                    "[PN25] torch.library.custom_op also unavailable "
                    "(torch<2.4) — falling back to vanilla forward_native"
                )
                return False

            @custom_op(_OP_QUALNAME, mutates_args=(), device_types=("cuda",))
            def _silu_and_mul_pooled_decorated(x: torch.Tensor) -> torch.Tensor:
                return _silu_and_mul_pooled_impl(x)

            @_silu_and_mul_pooled_decorated.register_fake
            def _silu_and_mul_pooled_decorated_fake(x: torch.Tensor) -> torch.Tensor:
                return _silu_and_mul_pooled_fake(x)
        except Exception as e:
            log.info("[PN25] @custom_op fallback registration failed: %s", e)
            return False
    except Exception as e:
        log.info("[PN25] direct_register_custom_op failed: %s", e)
        return False

    _op_registered = True
    log.info(
        "[PN25] registered torch op %s via direct_register_custom_op "
        "(vLLM canonical, v7.66 — fork-safe + Inductor-opaque)",
        _OP_QUALNAME,
    )
    return True


def get_op_callable():
    """Return the registered op callable, or None if registration failed.

    Used by the PN25 wiring patch to populate the replacement body.
    Caller is responsible for graceful degradation on None.

    v7.68 trace-context guard
    -------------------------
    If this function is called from inside a Dynamo trace context
    (e.g. v7.66 path where forward_native calls us during
    profile_run.aot_compile_fullgraph), short-circuit to None BEFORE
    attempting any `torch.library.Library` construction. v7.66 v3+
    PN25 wiring should call this at activation.py module-import time
    (before any trace context exists), but defense-in-depth in case
    the wiring text-patch fails or anchor drifts and the legacy
    `forward_native -> get_op_callable` path is exercised.

    Returning None here makes the caller fall through to vanilla
    `F.silu(x[..., :d]) * x[..., d:]` math instead of crashing the
    engine with `instantiate_user_defined_class_object` Dynamo error.
    """
    # v7.68 defense: if we're inside a torch.compile / Dynamo trace
    # AND the op isn't already registered, do not attempt registration
    # (would call `Library` constructor which Dynamo can't trace).
    if not _op_registered:
        try:
            compiler = getattr(torch, "compiler", None)
            if compiler is not None and getattr(compiler, "is_compiling", lambda: False)():
                log.debug(
                    "[PN25] get_op_callable called during torch.compile "
                    "before registration; returning None (use import-time "
                    "registration via PN25 v7.68 wiring instead)"
                )
                return None
        except Exception:
            pass
        try:
            dynamo = getattr(torch, "_dynamo", None)
            if dynamo is not None and getattr(dynamo, "is_compiling", lambda: False)():
                log.debug(
                    "[PN25] get_op_callable called during Dynamo tracing "
                    "before registration; returning None (use import-time "
                    "registration via PN25 v7.68 wiring instead)"
                )
                return None
        except Exception:
            pass

    if not _register_op_once():
        return None
    try:
        return torch.ops.genesis.silu_and_mul_pooled
    except (AttributeError, RuntimeError):
        return None
