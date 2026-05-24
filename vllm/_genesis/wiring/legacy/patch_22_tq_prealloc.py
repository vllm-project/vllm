# SPDX-License-Identifier: Apache-2.0
"""Wiring for Patch 22 — TurboQuant K/V dequant buffer prealloc hook.

Background
----------
The Phase 2 kernel `vllm._genesis.kernels.dequant_buffer.TurboQuantBufferManager`
implements shared K/V dequant buffer allocation. The MISSING piece was the
hook into vLLM's `TurboQuantAttentionImpl._ensure_on_device(self, layer,
device)` — Phase 2 only had the helper code, not the live binding.

This module adds the class-method override that makes vLLM call our
prealloc helper from inside `_ensure_on_device`, which fires during the
memory-profiler warmup. That makes the buffer footprint VISIBLE to the
profiler → KV cache sized correctly → no #40420-class OOM at long context.

Process semantics — important
-----------------------------
vLLM workers spawn fresh Python interpreters (`VLLM_WORKER_MULTIPROC_METHOD=
spawn`), so class-attribute changes done in the container entrypoint
process do NOT propagate to TP workers. To make this rebind effective in
each worker:

  1. Install the genesis-vllm-plugin properly (entry_points discovery).
  2. The plugin's `register()` is auto-called in every process:
       - main API server
       - engine core
       - each TP worker rank
     (See `vllm.plugins.load_general_plugins()` callsites in
      `vllm/v1/engine/core.py:103`, `vllm/v1/worker/worker_base.py:239`,
      `vllm/engine/arg_utils.py:685`.)
  3. apply_all.run(apply=True) fires from register() in each process →
     this wiring rebinds the class method per-process.

If the plugin install fails (e.g. read-only volume mount), the rebind
applies only in the entrypoint process and has NO effect on inference.
The Genesis container compose copies /plugin to a writable location
before pip-install precisely to avoid that failure mode.

Compatibility with upstream OOM-fix PRs
---------------------------------------
Three upstream PRs propose alternatives (all OPEN as of 2026-04-26):

- **PR #40798** (Bot1822): WorkspaceManager-based reservation in
  `gpu_model_runner.capture_model()` BEFORE `lock_workspace()`.
  Iterates ALL attention groups (hybrid-safe). Drift marker:
  `_reserve_turboquant_decode_workspace` symbol in `gpu_model_runner.py`.
  REMOVES `buf_holder` kwarg from `triton_turboquant_decode_attention` —
  conflicts with our P67b. P67b will need rebase when this lands.
- **PR #40706** (lesj0610): WorkspaceManager but reserved per-layer
  in `_init_turboquant_buffers`. PRESERVES `buf_holder` fallback.
  Drift marker: `reserve_turboquant_decode_workspace` symbol in
  `turboquant_attn.py`.
- **PR #40655** (bhoomit): class-shared buffers; no profiler visibility.
  CHANGES_REQUESTED by maintainer (LucasWilkinson). Less likely to land.
  Drift marker: removed `_init_turboquant_buffers` from `turboquant_attn.py`.

Our P22 sits closest to the #40706 / Genesis hybrid class. When ANY of
these three land and our drift markers are detected, this wiring
auto-skips. After #40798 merges: P67b also needs rebase to drop the
4 explicit buffer kwargs (`mid_o_buf`, `output_buf`, `lse_buf`,
`buf_holder=layer`) which #40798 removes from the kernel signature.

Author: Sandermage(Sander)-Barzov Aleksandr, Ukraine, Odessa
"""
from __future__ import annotations

import logging
from typing import Any

from vllm._genesis.guards import is_nvidia_cuda, is_sm_at_least

log = logging.getLogger("genesis.wiring.p22_tq_prealloc")


# Marker attribute attached to our wrapper function so we can detect
# already-applied state and skip re-wrap (idempotency guard).
_GENESIS_P22_MARKER_ATTR = "_genesis_p22_wrapped"


def should_apply() -> bool:
    """Match TurboQuantBufferManager.should_apply()."""
    if not is_nvidia_cuda():
        return False
    if not is_sm_at_least(8, 0):
        return False
    return True


# Class-name candidate list — future-proofs against upstream renames.
# If the TurboQuant backend class gets renamed (e.g. TurboQuantAttentionImpl
# → TurboQuantAttentionImplV1 or similar), add the new name HERE without
# touching wiring logic. We try them in order; the FIRST one that imports
# wins.
_CANDIDATE_TQ_IMPL_NAMES = (
    "TurboQuantAttentionImpl",  # current (fe9c3d6c5 baseline)
    # Future candidates — add as upstream evolves. Listed here so the
    # single point of change for a rename is this tuple.
)


def _import_tq_impl() -> Any | None:
    """Try to import the TurboQuant attention impl, trying all known class
    names. Returns None if NONE resolve (indicating TQ backend not
    present in this vLLM build — e.g. CPU-only, ROCm, or a future
    upstream that moved / renamed the backend).
    """
    try:
        import importlib
        mod = importlib.import_module(
            "vllm.v1.attention.backends.turboquant_attn"
        )
    except ImportError as e:
        log.info("[Genesis P22] turboquant_attn module not importable: %s", e)
        return None
    except Exception as e:
        log.warning("[Genesis P22] unexpected error importing TQ module: %s", e)
        return None
    for name in _CANDIDATE_TQ_IMPL_NAMES:
        impl = getattr(mod, name, None)
        if impl is not None:
            return impl
    log.info(
        "[Genesis P22] none of %s found in turboquant_attn — "
        "upstream may have renamed the impl class; add new name to "
        "_CANDIDATE_TQ_IMPL_NAMES to re-enable",
        list(_CANDIDATE_TQ_IMPL_NAMES),
    )
    return None


def _check_upstream_tq_workspace_drift() -> tuple[bool, str]:
    """Detect if any of the 3 upstream TQ-workspace PRs has merged.

    Returns (drifted, reason). When drifted=True, P22 should auto-skip
    because upstream now does equivalent (or stronger) workspace dedup.

    Drift markers (each PR introduces unique symbol):
    - PR #40798: `_reserve_turboquant_decode_workspace` in `gpu_model_runner.py`
    - PR #40706: `reserve_turboquant_decode_workspace` in `turboquant_attn.py`
    - PR #40655: `_init_turboquant_buffers` REMOVED from `turboquant_attn.py`
    """
    try:
        # Check #40798 marker (in gpu_model_runner)
        try:
            import vllm.v1.worker.gpu_model_runner as _gmr
            if hasattr(_gmr, "_reserve_turboquant_decode_workspace"):
                return True, "PR #40798 merged (Bot1822 WorkspaceManager) — auto-skip"
        except Exception:
            # Module path not present in this pin — drift probe continues with next marker
            pass
        # Check #40706 marker (in turboquant_attn)
        try:
            import vllm.v1.attention.backends.turboquant_attn as _tqa
            if hasattr(_tqa, "reserve_turboquant_decode_workspace"):
                return True, "PR #40706 merged (lesj0610 WorkspaceManager) — auto-skip"
            # #40655 marker — REMOVAL of _init_turboquant_buffers method
            impl = _import_tq_impl()
            if impl is not None and not hasattr(impl, "_init_turboquant_buffers"):
                return True, "PR #40655 merged (bhoomit moved init out) — auto-skip"
        except Exception:
            # turboquant_attn backend missing — outer except handles broader fallback
            pass
    except Exception as e:
        log.debug("[P22] drift check probe failed (%s); proceeding to apply", e)
    return False, "no upstream TQ workspace PR detected"


def apply() -> tuple[str, str]:
    """Wire ensure_turboquant_buffers() into TurboQuantAttentionImpl._ensure_on_device.

    Never raises. Returns (status, reason).
    """
    if not should_apply():
        return "skipped", "platform: NVIDIA SM 8.0+ required for TurboQuant"

    # Drift check: auto-skip if any of upstream PRs #40798/#40706/#40655 merged
    drifted, drift_reason = _check_upstream_tq_workspace_drift()
    if drifted:
        log.info("[P22] %s — patch retired in favor of upstream", drift_reason)
        return "skipped", drift_reason

    impl_cls = _import_tq_impl()
    if impl_cls is None:
        return "skipped", "TurboQuant backend (turboquant_attn module) not available"

    if not hasattr(impl_cls, "_ensure_on_device"):
        return "skipped", (
            "TurboQuantAttentionImpl._ensure_on_device not present "
            "(upstream may have refactored this method)"
        )

    # P49 interface contract check (v7.8): our prealloc helper reads
    # `impl.num_kv_heads`, `impl.head_size`, `impl._max_model_len`
    # (via fallback chain) from the instance. Verify the class-level
    # interface at least HAS the method slot; instance-level attrs are
    # set in __init__ so we can't check them here — runtime fallbacks
    # in `ensure_turboquant_buffers` handle those.
    try:
        from vllm._genesis.interface_guard import (
            validate_impl,
        )
        validate_impl(
            impl_cls,
            role="TurboQuantAttentionImpl (P22 _ensure_on_device)",
            required_methods=["_ensure_on_device"],
        )
    except Exception as e:
        if "GenesisInterfaceMismatch" in type(e).__name__:
            return "skipped", f"P49 interface drift: {e}"

    original = impl_cls._ensure_on_device

    if getattr(original, _GENESIS_P22_MARKER_ATTR, False):
        return "applied", "already wrapped (idempotent — process already patched)"

    # Import the helper here so any kernel-import error bubbles up cleanly
    # before we touch the live class.
    try:
        from vllm._genesis.kernels.dequant_buffer import (
            ensure_turboquant_buffers,
        )
    except Exception as e:
        return "failed", f"genesis kernel import failed: {e}"

    def _genesis_wrapped_ensure_on_device(self, layer, device):
        """Genesis P22 wrapper around TurboQuantAttentionImpl._ensure_on_device.

        Calls the original first (so vLLM's internal state is set up),
        then attaches Genesis prealloc buffers to `layer`. Never raises —
        any prealloc failure becomes a logged warning, allowing the
        upstream lazy-allocation path to take over (= same behavior as
        if our patch were never present).
        """
        # Original first — sets _tq_PiT, _tq_Pi, _tq_midpoints, _tq_cached
        original(self, layer, device)
        try:
            ensure_turboquant_buffers(self, layer, device)
        except Exception as e:
            log.warning(
                "[Genesis P22] prealloc helper failed for layer (%s): %s. "
                "Falling back to upstream lazy allocation.",
                getattr(layer, "_layer_name", "<unknown>"),
                e,
            )

    setattr(_genesis_wrapped_ensure_on_device, _GENESIS_P22_MARKER_ATTR, True)
    # Stash original as explicit attribute so revert() doesn't rely on
    # __closure__ ordering (which is alphabetized by free var name in CPython).
    setattr(_genesis_wrapped_ensure_on_device, "_genesis_p22_original", original)
    impl_cls._ensure_on_device = _genesis_wrapped_ensure_on_device

    log.info(
        "[Genesis P22] rebound TurboQuantAttentionImpl._ensure_on_device "
        "(prealloc helper will fire during profile_run warmup)"
    )
    return "applied", "class method wrapped (effective in this process)"


def is_applied() -> bool:
    """Post-apply assertion helper. True if our wrapper is the live binding."""
    impl_cls = _import_tq_impl()
    if impl_cls is None:
        return False
    method = getattr(impl_cls, "_ensure_on_device", None)
    if method is None:
        return False
    return getattr(method, _GENESIS_P22_MARKER_ATTR, False)


def revert() -> bool:
    """Restore the original method. For tests only."""
    impl_cls = _import_tq_impl()
    if impl_cls is None:
        return False
    method = getattr(impl_cls, "_ensure_on_device", None)
    if method is None:
        return False
    if not getattr(method, _GENESIS_P22_MARKER_ATTR, False):
        return False  # not ours
    original = getattr(method, "_genesis_p22_original", None)
    if original is None:
        return False
    impl_cls._ensure_on_device = original
    return True
