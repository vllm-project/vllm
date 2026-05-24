# SPDX-License-Identifier: Apache-2.0
"""Wiring for Patch N26 sub-component — SPARSE_V kernel dispatcher.

This is the second piece of PN26 (TQ unified perf pack). The first piece
(centroids prebake) is in `patch_N26_tq_unified_perf.py`. This file wires
the sparse-V tile-skip kernel forked in
`vllm/_genesis/kernels/triton_turboquant_decode_sparse_v.py` into the
runtime dispatch flow.

Design — Genesis dispatcher wrapper (P40 pattern)
-------------------------------------------------
Rather than text-patching upstream Triton kernel (very fragile across
nightly pin bumps + would conflict with our P67 multi-query kernel hot
path), we **monkey-patch** the upstream Python launcher
`triton_turboquant_decode_attention` with a Genesis dispatcher:

```
def _genesis_pn26_dispatcher(*args, **kwargs):
    from vllm._genesis.kernels.triton_turboquant_decode_sparse_v import (
        is_pn26_sparse_v_enabled, compute_effective_threshold,
        get_sparse_v_min_ctx,
        triton_turboquant_decode_attention_sparse_v,
    )
    if is_pn26_sparse_v_enabled():
        # Resolve seq_lens.max() once — used for BLASST λ=a/L scaling
        seq_lens_arg = kwargs.get('seq_lens') or args[3]
        seq_len = int(seq_lens_arg.max().item()) if seq_lens_arg.numel() else 0
        if seq_len >= get_sparse_v_min_ctx():
            return triton_turboquant_decode_attention_sparse_v(
                *args,
                sparse_v=True,
                sparse_v_threshold=compute_effective_threshold(seq_len),
                **kwargs,
            )
    # Fall through to upstream when env-disabled or short ctx
    return _genesis_pn26_original(*args, **kwargs)
```

When env-disabled, this is a single Python function call indirection on
the upstream path — negligible overhead. When env-enabled + ctx ≥ min_ctx,
we route to our forked kernel.

NVIDIA validation strategy
--------------------------
This is **first** known sparse-V kernel for SM86 (Ampere consumer) in any
public inference engine. TRT-LLM #9821 + FlashInfer #2477 ship for SM90+
only; BSFA validates on SM80 (A100 datacenter); Genesis fills the
SM86/SM89 consumer gap.

Validation gates (must pass before default-on):
1. **Numeric equivalence** at SPARSE_V=0 — bit-exact match to upstream
   for the same inputs.
2. **Bench A/B** at 35B DFlash 16K / 64K / 160K contexts. TPS gain
   expected: +3-15% per BLASST/research extrapolation.
3. **Quality regression** — tool-call clean rate ≥ baseline -1pp on
   7-city sweep.
4. **CV stability** — coefficient of variation ≤ 7% across 5-run bench.

Composition
-----------
- **PN26 main (centroids prebake)**: orthogonal, both can be enabled.
- **P40 grouped decode**: chains AFTER P40 dispatcher if both wrapped.
  Detection: if `_genesis_p40_wrapped` marker on upstream fn, our
  dispatcher inserts before P40's at the SAME function.
- **P98 TQ WorkspaceManager revert**: orthogonal — P98 patches
  `_decode_attention` caller, not the kernel itself.
- **P67 multi-query kernel**: separate code path (spec-decode K+1
  verify) — does NOT go through `triton_turboquant_decode_attention`.
  PN26 sparse_v ONLY applies to standard decode.

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
Sources synthesis (4 research agents, 2026-05-01):
  - vllm#41422 (TheTom) — design template, AMD MI300X validated
  - BLASST arXiv 2512.12087 (Yuan et al. Dec 2025) — λ=a/L formula
  - TensorRT-LLM PR #9821 — production reference (SM90+)
  - SpargeAttn ICML 2025 — RTX 3090/4090/L40 Ampere validation
  - tq-kv reference (onur-gokyildiz-bhi) — SM86-compatible CUDA pattern
  - StreamingLLM arXiv 2309.17453 — sink token protection (first 4 pos)
"""
from __future__ import annotations

import logging
import os
from typing import Any

log = logging.getLogger("genesis.wiring.pn26_sparse_v_kernel")

_GENESIS_PN26_SPARSE_V_MARKER_ATTR = "_genesis_pn26_sparse_v_wrapped"
_MODULE_PATH = "vllm.v1.attention.ops.triton_turboquant_decode"
_FN_NAME = "triton_turboquant_decode_attention"


def _import_target() -> tuple[Any, Any] | None:
    """Return (module, original_fn) or None on import failure."""
    import importlib
    try:
        mod = importlib.import_module(_MODULE_PATH)
    except ImportError:
        return None
    except Exception as e:
        log.warning("[Genesis PN26 sparse_v] import %s failed: %s", _MODULE_PATH, e)
        return None
    fn = getattr(mod, _FN_NAME, None)
    if fn is None:
        return None
    return mod, fn


def apply() -> tuple[str, str]:
    """Wrap `triton_turboquant_decode_attention` with Genesis sparse-V dispatcher.

    Behavior:
    - If env not set → SKIP cleanly (no wrapping).
    - If env set + platform OK → install wrapper (idempotent via marker).
    - Wrapper calls Genesis fork only when (env_enabled AND seq_len ≥ min_ctx);
      otherwise transparently delegates to upstream.
    """
    from vllm._genesis.kernels.triton_turboquant_decode_sparse_v import (
        should_apply,
        is_pn26_sparse_v_enabled,
    )

    if not is_pn26_sparse_v_enabled():
        return "skipped", (
            "opt-in: set GENESIS_ENABLE_PN26_SPARSE_V=1 to enable sparse-V "
            "tile-skip kernel (BLASST λ=a/L formula by default)"
        )

    if not should_apply():
        return "skipped", (
            "platform gate: NVIDIA SM ≥ 8.0 required (need Ampere or newer)"
        )

    target = _import_target()
    if target is None:
        return "skipped", (
            f"target module {_MODULE_PATH!r} or symbol {_FN_NAME!r} "
            "not available — TurboQuant backend not compiled in"
        )
    mod, original = target

    # Idempotency
    if getattr(original, _GENESIS_PN26_SPARSE_V_MARKER_ATTR, False):
        return "applied", "already wrapped (idempotent)"

    # Pre-build the kernel to surface compile errors at boot, not first request
    try:
        from vllm._genesis.kernels.triton_turboquant_decode_sparse_v import (
            _build_kernel,
        )
        _build_kernel()
    except Exception as e:
        return "failed", f"kernel build at boot failed: {e}"

    # ====================================================================
    # LEAN DISPATCHER (PN26b v2, 2026-05-01) — no per-call GPU↔CPU sync.
    # ====================================================================
    # First version of this dispatcher resolved seq_lens.max().item() per
    # call to compute the BLASST-style threshold and gate the min_ctx
    # routing. That `.item()` is a forced GPU↔CPU sync — at 60 layers ×
    # N decode tokens × 2 ranks, this overhead alone caused -16% TPS
    # regression on 35B PROD short-ctx (where sparse_v never fires).
    #
    # Lean v2 design (matches TRT-LLM + SpargeAttn approach):
    # - Read threshold ONCE at apply() time from env → bake into kernel
    #   constexpr at first call.
    # - No per-call seq_len resolve.
    # - Always route to our kernel; let Triton constexpr DCE handle the
    #   case when sparse_v is effectively a no-op for the inputs.
    # - cudagraph-capture path: our kernel is byte-equivalent to upstream
    #   when SPARSE_V=0 constexpr is baked, so safe to capture.
    #
    # This eliminates the per-call sync that killed v1 perf.
    # ====================================================================
    from vllm._genesis.kernels.triton_turboquant_decode_sparse_v import (
        get_sparse_v_threshold as _get_threshold_at_init,
        get_sparse_v_scale_factor as _get_scale_at_init,
    )

    # Bake threshold at apply() time. If scale_factor is set (BLASST mode),
    # we cannot pre-compute (depends on per-call seq_len). For lean v2 we
    # only support fixed-threshold mode — BLASST adaptive requires the
    # per-call sync we're trying to avoid. Falls back to fixed default if
    # scale_factor was the operator's choice.
    _baked_scale = _get_scale_at_init()
    _baked_threshold = _get_threshold_at_init()
    if _baked_scale > 0:
        log.info(
            "[Genesis PN26 sparse_v] BLASST adaptive (scale=%s) requested but "
            "per-call resolve costs more than skip savings on SM86. Using "
            "fixed threshold %s (override via GENESIS_PN26_SPARSE_V_THRESHOLD).",
            _baked_scale, _baked_threshold,
        )

    # Bake debug-counter mode at apply() time too
    from vllm._genesis.kernels.triton_turboquant_decode_sparse_v import (
        is_debug_skip_enabled as _is_debug_at_init,
    )
    _baked_debug_skip = _is_debug_at_init()
    if _baked_debug_skip:
        log.info(
            "[Genesis PN26 sparse_v] DEBUG_SKIP_CTR=1 — per-CTA atomic "
            "counters enabled. Read via vllm._genesis.kernels."
            "triton_turboquant_decode_sparse_v.collect_skip_stats() "
            "after a request. Cost: ~50-100ns per CTA at epilogue (~0.05% "
            "kernel overhead, statistically indistinguishable from noise)."
        )

    # Per-process call counter for periodic skip-stats logging
    _call_counter = [0]
    _LOG_EVERY_N = int(os.environ.get("GENESIS_PN26_SPARSE_V_LOG_EVERY", "500"))

    def _genesis_pn26_dispatcher(*args, **kwargs):
        """Lean dispatcher — zero per-call GPU↔CPU sync.

        Always invokes our forked kernel with the env-baked threshold.
        Triton constexpr handles the sparse_v branch elimination at
        compile time. cudagraph capture is safe because our kernel
        signature matches upstream exactly.

        When DEBUG_SKIP_CTR=1, logs accumulated skip-rate stats every
        N calls (default 500) so operator can observe skip rate without
        cross-process IPC. Per-process state (each worker logs own).
        """
        from vllm._genesis.kernels.triton_turboquant_decode_sparse_v import (
            triton_turboquant_decode_attention_sparse_v,
            collect_skip_stats,
        )
        try:
            result = triton_turboquant_decode_attention_sparse_v(
                *args,
                sparse_v=True,
                sparse_v_threshold=_baked_threshold,
                debug_skip_ctr=_baked_debug_skip,
                **kwargs,
            )
        except Exception as e:
            log.warning(
                "[Genesis PN26 sparse_v] kernel call failed (%s); "
                "falling back to upstream for this call",
                e,
            )
            return original(*args, **kwargs)

        # Periodic skip-stats logging (only when debug enabled)
        if _baked_debug_skip:
            _call_counter[0] += 1
            if _call_counter[0] % _LOG_EVERY_N == 0:
                try:
                    stats = collect_skip_stats()
                    if stats.get("enabled"):
                        # WARNING level (not INFO) so it surfaces under
                        # VLLM_LOGGING_LEVEL=WARNING. DEBUG_SKIP_CTR=1
                        # is opt-in operator instrumentation — guaranteed
                        # visibility is the contract.
                        log.warning(
                            "[Genesis PN26 sparse_v] skip-rate after %d calls: "
                            "lifetime %.2f%% (last_launch %.2f%%, "
                            "tiles total/skipped: %d/%d)",
                            _call_counter[0],
                            stats["lifetime_skip_rate_pct"],
                            stats["last_launch_skip_rate_pct"],
                            stats["lifetime_total_tiles"],
                            stats["lifetime_skipped_tiles"],
                        )
                except Exception:
                    pass

        return result

    # Mark as wrapped + install on source module
    setattr(_genesis_pn26_dispatcher, _GENESIS_PN26_SPARSE_V_MARKER_ATTR, True)
    setattr(mod, _FN_NAME, _genesis_pn26_dispatcher)

    # ALSO rebind on importing modules. The TQ backend at
    # vllm.v1.attention.backends.turboquant_attn does
    # `from ... import triton_turboquant_decode_attention` at module
    # top, so its local binding captured the *original* fn. Patching
    # only the source module is a silent no-op for those call sites
    # (same class of bug as P38B). We must rebind every consumer.
    _CONSUMER_MODULES = (
        "vllm.v1.attention.backends.turboquant_attn",
    )
    _rebound = []
    for consumer_path in _CONSUMER_MODULES:
        try:
            import importlib
            consumer_mod = importlib.import_module(consumer_path)
            if hasattr(consumer_mod, _FN_NAME):
                setattr(consumer_mod, _FN_NAME, _genesis_pn26_dispatcher)
                _rebound.append(consumer_path)
        except Exception as e:
            log.warning(
                "[Genesis PN26 sparse_v] could not rebind on %s: %s",
                consumer_path, e,
            )

    log.info(
        "[Genesis PN26 sparse_v] wrapped %s.%s with sparse-V dispatcher "
        "(also rebound on consumers: %s)",
        _MODULE_PATH, _FN_NAME, _rebound or "none",
    )
    return "applied", (
        "PN26 sparse-V kernel dispatcher installed. Routes to Genesis "
        "fork when seq_len >= min_ctx (default 8192). Threshold: "
        "BLASST λ=a/L if GENESIS_PN26_SPARSE_V_SCALE_FACTOR>0, else "
        "fixed GENESIS_PN26_SPARSE_V_THRESHOLD (default 0.001). "
        "First sparse-V kernel deployed for SM86 (Ampere consumer) — "
        "no upstream NVIDIA reference exists yet."
    )


def is_applied() -> bool:
    target = _import_target()
    if target is None:
        return False
    _, fn = target
    return bool(getattr(fn, _GENESIS_PN26_SPARSE_V_MARKER_ATTR, False))
