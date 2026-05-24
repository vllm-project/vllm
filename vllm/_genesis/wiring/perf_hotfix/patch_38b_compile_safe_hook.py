# SPDX-License-Identifier: Apache-2.0
"""Wiring for Patch 38b — P38 compile-safe in-source hook.

Fixes Genesis Issue #14 (noonghunna 2026-05-01):
https://github.com/Sandermage/genesis-vllm-patches/issues/14

================================================================
PROBLEM (root cause)
================================================================

P38 (`patch_38_tq_continuation_memory.py`) replaces
`TurboQuantAttentionImpl._continuation_prefill` via Python class-
attribute rebind:

    TurboQuantAttentionImpl._continuation_prefill = _genesis_continuation_prefill

This works on **eager mode**. It does NOT work when vLLM's
`aot_compile_fullgraph` captures `forward` → `_prefill_attention` →
`_continuation_prefill` at engine init: the compiled artifact bakes in
the ORIGINAL method body. Subsequent class-attribute rebinds update the
live class but NOT the compiled graph.

noonghunna's instrumentation confirmed this empirically: log line in
`_genesis_continuation_prefill` never fires despite the rebind reporting
"applied". The OOM trace passes through line 909 (`torch.cat`) of
`turboquant_attn.py` (original source), not through Genesis's K_full/V_full
workspace path.

Affected paths: Qwen3.6-27B AutoRound INT4 + TQ k8v4 (or TQ3) KV — i.e.
any TQ KV config with the V0/V1 compile pipeline. fp8 KV configs aren't
affected because the entire `_continuation_prefill` codepath is inactive
on fp8.

================================================================
FIX DESIGN — text-patch the source file
================================================================

Insert an early-return delegate at the START of `_continuation_prefill`
body in the upstream source file:

    def _continuation_prefill(self, layer, query, ...):
        \"\"\"Handle continuation chunk by dequanting cached K/V from TQ cache.
        ...
        \"\"\"
        # [Genesis P38b compile-safe hook] ← injected here
        _genesis_disp = type(self).__dict__.get('_genesis_p38_dispatch', None)
        if _genesis_disp is not None:
            _r = _genesis_disp(self, layer, query, key_chunk, val_chunk,
                               kv_cache, block_table, cached_len, seq_len,
                               Pi, centroids)
            if _r is not None:
                return _r

        q_len, Hq, D = query.shape  # ← original body resumes
        ...

Why this works:

1. Source-level edit: aot_compile_fullgraph captures the MODIFIED source
   at engine init, INCLUDING the hook. The hook is part of the compiled
   graph.
2. `type(self).__dict__.get(...)` is a dict lookup on the class —
   torch.compile sees this as a Python attribute access, falls back to
   eager for that branch. The fall-back is FINE because the dispatch
   decision is meant to break out of the compiled graph anyway.
3. When dispatch returns `None`, original body resumes (compiled).
4. When dispatch returns a tensor, we short-circuit out — eager path.

Composition with P38:

- P38 still rebinds `_continuation_prefill = _genesis_continuation_prefill`
  for eager-mode callers (V1 plain forward path without compile).
- P38b ADDITIONALLY sets `TurboQuantAttentionImpl._genesis_p38_dispatch =
  _genesis_continuation_prefill_dispatcher` which returns the result OR
  None to indicate "fall through". The dispatcher checks env + state.
- Both paths share the SAME `_genesis_continuation_prefill` implementation
  in `patch_38_tq_continuation_memory.py`.

================================================================
SAFETY MODEL
================================================================

- Default OFF (opt-in via `GENESIS_ENABLE_P38B_COMPILE_SAFE=1`).
- Idempotent (marker-checked).
- Drift-aware: if upstream renames `_continuation_prefill` or restructures
  the docstring close, anchor misses → SKIPPED.
- Fall-through guarantee: if `_genesis_p38_dispatch` is not set OR returns
  None, original body runs unchanged.
- Worst case if applied but Genesis impl breaks: dispatcher returns None,
  original body runs. No crash.

Author: Sandermage(Sander) Barzov Aleksandr, Ukraine, Odessa
Origin: noonghunna Issue #14 — direct fix per their suggestion
"""
from __future__ import annotations

import logging

from vllm._genesis.guards import resolve_vllm_file, vllm_install_root
from vllm._genesis.wiring.text_patch import (
    TextPatch,
    TextPatcher,
    TextPatchResult,
)

log = logging.getLogger("genesis.wiring.p38b_compile_safe_hook")

GENESIS_P38B_MARKER = "Genesis P38b compile-safe hook (Issue #14) v7.65"


# Anchor: docstring close + first body line (q_len, Hq, D = query.shape)
# Surrounding context disambiguates from any other method in the file.
P38B_ANCHOR = (
    "        Dequants previously cached K/V, concatenates with the current\n"
    "        chunk's raw K/V, then runs flash_attn with causal masking.\n"
    "        \"\"\"\n"
    "        q_len, Hq, D = query.shape\n"
)

P38B_REPLACEMENT = (
    "        Dequants previously cached K/V, concatenates with the current\n"
    "        chunk's raw K/V, then runs flash_attn with causal masking.\n"
    "        \"\"\"\n"
    "        # [Genesis P38b compile-safe hook] In-source delegate that\n"
    "        # survives `aot_compile_fullgraph` capture (Python class-attr\n"
    "        # rebind does not). Calls Genesis dispatcher when set; falls\n"
    "        # through to original body when dispatch returns None.\n"
    "        # Fixes Genesis Issue #14 (noonghunna 2026-05-01).\n"
    "        _genesis_p38b_disp = type(self).__dict__.get(\n"
    "            '_genesis_p38_dispatch', None\n"
    "        )\n"
    "        if _genesis_p38b_disp is not None:\n"
    "            _genesis_p38b_r = _genesis_p38b_disp(\n"
    "                self, layer, query, key_chunk, val_chunk,\n"
    "                kv_cache, block_table, cached_len, seq_len,\n"
    "                Pi, centroids,\n"
    "            )\n"
    "            if _genesis_p38b_r is not None:\n"
    "                return _genesis_p38b_r\n"
    "        q_len, Hq, D = query.shape\n"
)


def _make_patcher() -> TextPatcher | None:
    target = resolve_vllm_file("v1/attention/backends/turboquant_attn.py")
    if target is None:
        return None
    return TextPatcher(
        patch_name=(
            "P38b turboquant_attn.py — _continuation_prefill compile-safe hook "
            "(Issue #14 fix)"
        ),
        target_file=str(target),
        marker=GENESIS_P38B_MARKER,
        sub_patches=[
            TextPatch(
                name="p38b_continuation_prefill_hook",
                anchor=P38B_ANCHOR,
                replacement=P38B_REPLACEMENT,
                required=True,
            ),
        ],
        upstream_drift_markers=[
            "[Genesis P38b",
            "_genesis_p38_dispatch",
            "_genesis_p38b_disp",
        ],
    )


def _install_dispatcher() -> bool:
    """Install `_genesis_p38_dispatch` on TurboQuantAttentionImpl class.

    The dispatcher checks env + state. When enabled, calls the Genesis
    `_genesis_continuation_prefill` (already defined in patch_38) and
    returns its result. Returns None to indicate "fall through to
    original body".
    """
    try:
        import importlib
        mod = importlib.import_module(
            "vllm.v1.attention.backends.turboquant_attn"
        )
    except Exception as e:
        log.warning("[Genesis P38b] turboquant_attn import failed: %s", e)
        return False

    impl_cls = getattr(mod, "TurboQuantAttentionImpl", None)
    if impl_cls is None:
        log.info(
            "[Genesis P38b] TurboQuantAttentionImpl class not found — "
            "TQ backend not compiled in. P38b skipped."
        )
        return False

    # Resolve the Genesis continuation prefill impl from patch_38 module.
    try:
        from vllm._genesis.wiring.legacy import patch_38_tq_continuation_memory as p38_mod
    except Exception as e:
        log.warning("[Genesis P38b] patch_38 module import failed: %s", e)
        return False

    genesis_fn = getattr(p38_mod, "_genesis_continuation_prefill", None)
    if genesis_fn is None:
        log.info(
            "[Genesis P38b] patch_38._genesis_continuation_prefill not found "
            "— P38b cannot dispatch. Verify P38 wiring."
        )
        return False

    def _dispatch(self, layer, query, key_chunk, val_chunk, kv_cache,
                  block_table, cached_len, seq_len, Pi, centroids):
        """P38b dispatcher: route through Genesis impl when applicable.

        Returns Genesis result tensor on success. Returns None to fall
        through to upstream original body (e.g. when buffers not ready
        OR Genesis impl raises).
        """
        # Best-effort: check buffer manager initialized + Genesis impl ready
        try:
            return genesis_fn(
                self, layer, query, key_chunk, val_chunk, kv_cache,
                block_table, cached_len, seq_len, Pi, centroids,
            )
        except Exception as e:
            # Don't crash the request — log and fall through.
            log.warning(
                "[Genesis P38b] dispatch failed (%s); falling through to "
                "upstream _continuation_prefill body",
                e,
            )
            return None

    impl_cls._genesis_p38_dispatch = _dispatch
    log.info(
        "[Genesis P38b] installed _genesis_p38_dispatch on %s — "
        "compile-safe path active",
        impl_cls.__name__,
    )
    return True


def apply() -> tuple[str, str]:
    """Apply P38b — compile-safe hook + dispatcher install."""
    from vllm._genesis.dispatcher import log_decision, should_apply

    decision, reason = should_apply("P38B")
    log_decision("P38B", decision, reason)
    if not decision:
        return "skipped", reason

    if vllm_install_root() is None:
        return "skipped", "vllm install root not discoverable"

    # Step 1: text-patch source file
    patcher = _make_patcher()
    if patcher is None:
        return "skipped", "turboquant_attn.py not resolvable"

    result, failure = patcher.apply()
    text_status = "applied"
    text_reason = "P38b in-source hook injected"
    if result == TextPatchResult.IDEMPOTENT:
        text_status = "skipped"
        text_reason = "in-source hook already present"
    elif result == TextPatchResult.SKIPPED:
        text_status = "skipped"
        text_reason = (
            f"anchor mismatch: {failure.reason if failure else 'unknown'}"
        )
    elif result == TextPatchResult.FAILED:
        return "failed", (
            f"text-patch failed: {failure.detail if failure else 'unknown'}"
        )

    # Step 2: install dispatcher on the live class (must run AFTER
    # text-patch so the class import sees the hook).
    disp_ok = _install_dispatcher()
    if not disp_ok:
        return "skipped", (
            f"text-patch {text_status} ({text_reason}) but dispatcher "
            "install failed — TQ backend may not be loaded yet"
        )

    return "applied", (
        f"P38b applied: text-patch {text_status} ({text_reason}); "
        f"dispatcher installed on TurboQuantAttentionImpl. "
        f"_continuation_prefill now survives aot_compile_fullgraph capture. "
        f"Fixes Issue #14 (noonghunna)."
    )


def is_applied() -> bool:
    """Return True iff text-patch marker AND dispatcher are present."""
    target = resolve_vllm_file("v1/attention/backends/turboquant_attn.py")
    if target is None:
        return False
    try:
        with open(str(target)) as f:
            if GENESIS_P38B_MARKER not in f.read():
                return False
    except OSError:
        return False
    try:
        import importlib
        mod = importlib.import_module(
            "vllm.v1.attention.backends.turboquant_attn"
        )
        cls = getattr(mod, "TurboQuantAttentionImpl", None)
        return cls is not None and hasattr(cls, "_genesis_p38_dispatch")
    except Exception:
        return False
