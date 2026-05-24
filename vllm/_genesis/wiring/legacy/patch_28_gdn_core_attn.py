# SPDX-License-Identifier: Apache-2.0
"""Wiring for Patch 28 — GDN core_attn_out prealloc (CRIT-HW-1 correct form).

Architecture
------------
Per master-plan CRIT-HW-1 ("P28 MUST pre-allocate at `__init__`, NEVER
lazy in forward"), this module does TWO things:

  1. **Class-method monkey-patch on `GatedDeltaNet.__init__`**. After the
     original __init__ runs, we call `gdn_core_attn_manager.attach_buffer`
     which allocates `self._genesis_gdn_core_attn_buf` (tensor OR None).
     This runs EAGER, once per module, outside any torch.compile trace —
     so device probes, env reads, dict lookups, logging are all safe.

  2. **Text-patch on `forward_cuda`**. The original `torch.zeros(...)`
     line is replaced with a pure-tensor conditional slice:

         core_attn_out = (
             self._genesis_gdn_core_attn_buf[:num_tokens].zero_()
             if self._genesis_gdn_core_attn_buf is not None
             else torch.zeros(
                 (num_tokens, self.num_v_heads // self.tp_size,
                  self.head_v_dim),
                 dtype=hidden_states.dtype, device=hidden_states.device,
             )
         )

     Both branches are pure tensor ops. The `is not None` guard resolves
     at trace time against a constant module attribute — `torch.dynamo`
     compiles only the selected branch and everything stays in-graph.

Platform compatibility
----------------------
  - NVIDIA CUDA SM ≥ 8.0 with the attribute set → pre-allocated slice.
  - All others (attribute is None) → fall-through `torch.zeros`
    identical to upstream behavior.

Upstream drift detection
------------------------
If `_genesis_gdn_core_attn_buf` already appears in the file OR upstream
lands its own buffer-pool fix, we skip.

Author: Sandermage(Sander)-Barzov Aleksandr, Ukraine, Odessa
"""
from __future__ import annotations

import logging

from vllm._genesis.guards import resolve_vllm_file, vllm_install_root
from vllm._genesis.wiring.text_patch import (
    TextPatch, TextPatcher, TextPatchResult,
)

log = logging.getLogger("genesis.wiring.p28_gdn_core_attn")

GENESIS_P28_MARKER = "Genesis P28 GDN core_attn_out prealloc v7.0"

UPSTREAM_DRIFT_MARKERS = [
    "_genesis_gdn_core_attn_buf",
    "gdn_core_attn_out_buffer",
    "gdn_core_attn_prealloc",
]


# Anchor: disambiguates from forward_xpu's identical line via the
# preceding "see discussions in https://github.com/vllm-project/vllm/pull/28182"
# comment (unique to forward_cuda).
_OLD_ALLOC = (
    "        # Note: we should not use torch.empty here like other attention backends,\n"
    "        # see discussions in https://github.com/vllm-project/vllm/pull/28182\n"
    "        core_attn_out = torch.zeros(\n"
    "            (num_tokens, self.num_v_heads // self.tp_size, self.head_v_dim),\n"
    "            dtype=hidden_states.dtype,\n"
    "            device=hidden_states.device,\n"
    "        )"
)

_NEW_ALLOC = (
    "        # Note: we should not use torch.empty here like other attention backends,\n"
    "        # see discussions in https://github.com/vllm-project/vllm/pull/28182\n"
    "        # [Genesis P28] Pre-allocated buffer attached by attach_buffer()\n"
    "        # at module __init__ (see vllm._genesis.kernels.gdn_core_attn_manager).\n"
    "        # Both branches are pure tensor ops — fully torch.dynamo-safe.\n"
    "        core_attn_out = (\n"
    "            self._genesis_gdn_core_attn_buf[:num_tokens].zero_()\n"
    "            if getattr(self, '_genesis_gdn_core_attn_buf', None) is not None\n"
    "            else torch.zeros(\n"
    "                (num_tokens, self.num_v_heads // self.tp_size, self.head_v_dim),\n"
    "                dtype=hidden_states.dtype,\n"
    "                device=hidden_states.device,\n"
    "            )\n"
    "        )"
)


def _make_patcher() -> TextPatcher | None:
    target = resolve_vllm_file("model_executor/layers/mamba/gdn_linear_attn.py")
    if target is None:
        return None
    return TextPatcher(
        patch_name="P28 GDN core_attn_out prealloc",
        target_file=target,
        marker=GENESIS_P28_MARKER,
        sub_patches=[
            TextPatch(
                name="p28_core_attn_out_alloc",
                anchor=_OLD_ALLOC,
                replacement=_NEW_ALLOC,
                required=True,
            ),
        ],
        upstream_drift_markers=UPSTREAM_DRIFT_MARKERS,
    )


# ─── Runtime init wrap ─────────────────────────────────────────────────────
# Wraps `GatedDeltaNet.__init__` so every new instance gets its buffer
# attached after the original init completes. Idempotent.

_INIT_WRAPPED_ATTR = "_genesis_p28_init_wrapped"


# Candidate class names across vLLM versions. Older baselines named the
# class `GatedDeltaNet`; post-2026-04 renamed to `GatedDeltaNetAttention`
# (to reflect the PluggableLayer / MambaBase mixin). We try both and use
# whichever imports cleanly.
_CANDIDATE_CLASS_NAMES = ("GatedDeltaNetAttention", "GatedDeltaNet")


def _resolve_gdn_class():
    """Import the GDN class, trying known names. Returns class or None."""
    try:
        import importlib
        mod = importlib.import_module(
            "vllm.model_executor.layers.mamba.gdn_linear_attn"
        )
    except Exception as e:
        log.info("[Genesis P28] gdn_linear_attn module not importable: %s", e)
        return None
    for name in _CANDIDATE_CLASS_NAMES:
        cls = getattr(mod, name, None)
        if cls is not None:
            return cls
    log.info(
        "[Genesis P28] none of %s found in gdn_linear_attn "
        "(upstream may have renamed the class — update _CANDIDATE_CLASS_NAMES)",
        list(_CANDIDATE_CLASS_NAMES),
    )
    return None


def _wrap_gdn_init() -> bool:
    """Monkey-patch the GDN class's `__init__`. Return True on success."""
    cls = _resolve_gdn_class()
    if cls is None:
        return False

    if getattr(cls.__init__, _INIT_WRAPPED_ATTR, False):
        return True  # already wrapped (idempotent)

    orig_init = cls.__init__

    from vllm._genesis.kernels.gdn_core_attn_manager import attach_buffer

    def _genesis_wrapped_init(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        try:
            attach_buffer(self)
        except Exception as e:
            log.warning(
                "[Genesis P28] attach_buffer in __init__ failed: %s "
                "(module will fall back to eager alloc on first forward)",
                e,
            )
            if not hasattr(self, "_genesis_gdn_core_attn_buf"):
                self._genesis_gdn_core_attn_buf = None

    setattr(_genesis_wrapped_init, _INIT_WRAPPED_ATTR, True)
    setattr(_genesis_wrapped_init, "_genesis_p28_original_init", orig_init)
    cls.__init__ = _genesis_wrapped_init
    log.info(
        "[Genesis P28] wrapped %s.__init__ to attach "
        "_genesis_gdn_core_attn_buf on each instance",
        cls.__name__,
    )
    return True


def is_applied() -> bool:
    """Verify init wrap is live (used by verify_live_rebinds)."""
    cls = _resolve_gdn_class()
    if cls is None:
        return False
    return getattr(cls.__init__, _INIT_WRAPPED_ATTR, False)


def revert() -> bool:
    """Restore original __init__. Returns True on success."""
    cls = _resolve_gdn_class()
    if cls is None:
        return False
    cur = cls.__init__
    if not getattr(cur, _INIT_WRAPPED_ATTR, False):
        return False
    orig = getattr(cur, "_genesis_p28_original_init", None)
    if orig is None:
        return False
    cls.__init__ = orig
    return True


def apply() -> tuple[str, str]:
    """Apply P28 wiring: warm-up caches + text-patch forward + wrap __init__.

    Never raises.
    """
    # Step 0: warm up the module-level caches (should_apply, env budget)
    # so traced forward paths never have to do device probes or env reads.
    try:
        from vllm._genesis.kernels.gdn_core_attn_manager import warm_up
        warm_up()
    except Exception as e:
        log.info("[Genesis P28] warm_up failed (non-fatal): %s", e)

    # P53 (v7.9): Hybrid-active dispatch gate. GDN attention only exists
    # on hybrid models (Qwen3-Next, Mamba2 variants). On pure-attention
    # models the text-patch anchor won't even match, but skipping early
    # keeps dispatch logs clean.
    try:
        from vllm._genesis.model_detect import is_hybrid_model, log_skip
        if not is_hybrid_model():
            log_skip("P28 GDN core-attn forward rewire", "pure-attention model (no GDN)")
            return "skipped", "P53 dispatch: model has no hybrid linear-attention layers"
    except Exception as e:
        log.debug("[Genesis P28] model_detect probe failed (proceeding): %s", e)

    # Step 1: text-patch forward_cuda
    if vllm_install_root() is None:
        return "skipped", "vllm install root not discoverable"
    patcher = _make_patcher()
    if patcher is None:
        return "skipped", "gdn_linear_attn.py not found"
    result, failure = patcher.apply()
    if result == TextPatchResult.FAILED:
        return "failed", failure.reason if failure else "unknown failure"
    if result == TextPatchResult.SKIPPED:
        return "skipped", failure.reason if failure else "unknown skip"
    # APPLIED or IDEMPOTENT — proceed to init wrap.

    # Step 2: wrap __init__ so new GDN instances get the buffer attached.
    init_ok = _wrap_gdn_init()
    if result == TextPatchResult.APPLIED:
        reason = "forward_cuda patched + __init__ wrapped" if init_ok \
            else "forward_cuda patched, __init__ wrap skipped"
    else:
        reason = "already applied (idempotent)" if init_ok \
            else "idempotent; init wrap skipped"
    return "applied", reason
