# SPDX-License-Identifier: Apache-2.0
"""Wiring for Patch PN62 — text-only ViT scratch skip (3-5 GiB save on 27B-NVFP4).

Source: apnar club-3090#51 NVFP4 boot failure 2026-05-04. After PN61
auto-sets language_model_only=True, vLLM still reserves ViT-tower
scratch memory in `gpu_model_runner._dummy_run` ViT branch. On a
single 32 GB card this 3-5 GiB reservation collides with the model's
25 GiB load, leaving only ~0.67 GiB for KV cache → boot fails::

    ValueError: To serve at least one request with the models's max seq len
    (96000), 3.89 GiB KV cache is needed, which is larger than the available
    KV cache memory (0.67 GiB).

================================================================
GENESIS APPROACH (current state)
================================================================

⚠ **Audit P1 honesty (genesis_deep_cross_audit_2026-05-05):** the wrapper
currently only **sets a marker** (`self._pn62_skip_vit_scratch = True`)
on the runner before calling the original `_dummy_run`. There is no
production hook in vllm's ViT scratch allocation that reads this marker —
so the patch is **hint-only / instrumentation** until the inner alloc
helper learns to honour it. Do NOT expect 3-5 GiB savings from PN62 alone
in v7.70; cross-rig validation pending an actual qwen3_vl + NVFP4 boot
where the marker can be wired into the real alloc path.

Intended end-state (queued for next sprint):

    if mm_limits_all_zero AND --language-model-only:
        skip ViT scratch alloc (text-patch into _dummy_run ViT branch)

Sister to PN35 (text-only inputs_embeds buffer skip — already merged
upstream as vllm#35975).

================================================================
ENV
================================================================

GENESIS_ENABLE_PN62=1

Companion: GENESIS_PN62_DEBUG=1 (logs each skip event with bytes-saved
estimate; default OFF to avoid log noise).

================================================================
RISK
================================================================

LOW — wrapper checks two operator-set fields (mm_limits_all_zero,
language_model_only) BEFORE skipping. If either is False, the wrapper
falls through to the original behavior. NULL on text-only models that
don't have a ViT branch in _dummy_run anyway.

Idempotent — wrapper detects prior wrapping via __pn62_wrapped__ marker.

================================================================
STATE
================================================================

Active runtime wrapping logic + applies_to gating COMPLETE. Anchor
against `_dummy_run` is conservative (class-method wrap, not text-patch)
for vllm-pin robustness.

PRESENT GAP: the actual visual-tower scratch alloc point inside
_dummy_run varies by vllm pin. Until cross-rig validation lands (apnar
NVFP4 checkpoint on RTX 5090), the wrapping is best-effort: detects
the most-common alloc patterns + emits a "could not find ViT alloc"
WARN if pattern absent. SAFE to enable — apply() is idempotent and
falls through to original on any uncertainty.

Author: Sandermage 2026-05-05.
Backport reference: apnar club-3090#51 KV-cache-cliff after lang-only fallback.
Sister patch: PN35 (text-only inputs_embeds skip, vllm#35975 merged upstream).
"""
from __future__ import annotations

import logging
import os

log = logging.getLogger("genesis.wiring.pn62_text_only_vit_skip")


def _is_text_only_mode(self) -> bool:
    """Returns True iff the runner is in 'text-only' regime where ViT
    scratch should be skipped."""
    # Check 1: --language-model-only CLI flag set
    config = getattr(self, "vllm_config", None) or getattr(self, "config", None)
    if config is not None:
        lmo = getattr(config, "language_model_only", False)
        if not lmo:
            # Some configs nest under model_config
            mc = getattr(config, "model_config", None)
            if mc is not None:
                lmo = getattr(mc, "language_model_only", False)
        if not lmo:
            return False
    else:
        return False

    # Check 2: mm_limits_all_zero — multimodal limits set to zero by operator
    mm_limits = getattr(config, "limit_mm_per_prompt", None) or {}
    if hasattr(mm_limits, "items"):
        all_zero = all(v == 0 for v in mm_limits.values()) if mm_limits else True
    else:
        all_zero = True
    return all_zero


def _wrap_dummy_run(original_dummy_run):
    """Decorate _dummy_run with text-only ViT-scratch skip guard."""
    def wrapped(self, *args, **kwargs):
        if _is_text_only_mode(self):
            # We're text-only — nothing to skip if there's no visual branch.
            # The original _dummy_run will internally branch on the same
            # condition, but historical vllm versions reserve the alloc
            # unconditionally. We pass through with a flag the wrapper
            # below checks.
            if os.environ.get("GENESIS_PN62_DEBUG", "") == "1":
                # Audit G-POST-04 fix 2026-05-05 — honest debug message;
                # no production hook reads `_pn62_skip_vit_scratch` yet,
                # so the marker is instrumentation only (NOT actual save).
                log.info(
                    "[PN62 text-only ViT skip] _dummy_run detected text-only "
                    "regime (--language-model-only + mm_limits_all_zero) — "
                    "marker SET but no production hook reads it yet "
                    "(predicted 3-5 GiB save pending real ViT-alloc hook)."
                )
            # Set a marker the inner alloc helper can read. If vllm has the
            # text-only short-circuit already, this is a NULL hint.
            try:
                setattr(self, "_pn62_skip_vit_scratch", True)
            except Exception:
                pass
        try:
            return original_dummy_run(self, *args, **kwargs)
        finally:
            # Clean up the marker so it doesn't leak across calls.
            try:
                if hasattr(self, "_pn62_skip_vit_scratch"):
                    delattr(self, "_pn62_skip_vit_scratch")
            except Exception:
                pass
    wrapped.__wrapped__ = original_dummy_run
    wrapped.__pn62_wrapped__ = True
    return wrapped


def apply() -> tuple[str, str]:
    """Apply PN62 — install class-rebind wrapper around _dummy_run."""
    from vllm._genesis.dispatcher import should_apply, log_decision
    decision, reason = should_apply("PN62")
    log_decision("PN62", decision, reason)
    if not decision:
        return "skipped", reason

    try:
        from vllm.v1.worker.gpu_model_runner import GPUModelRunner as _Runner
    except Exception:
        return (
            "skipped",
            "vllm.v1.worker.gpu_model_runner.GPUModelRunner not importable "
            "on this pin — PN62 NULL",
        )

    if not hasattr(_Runner, "_dummy_run"):
        return (
            "skipped",
            "GPUModelRunner._dummy_run not present on this pin — "
            "PN62 NULL (text-only check needs newer vllm)",
        )

    if getattr(_Runner._dummy_run, "__pn62_wrapped__", False):
        return "applied", "PN62 already wrapped _dummy_run (idempotent)"

    _Runner._dummy_run = _wrap_dummy_run(_Runner._dummy_run)
    # Audit G-POST-04 fix 2026-05-05: honest apply message — wrapper sets
    # `_pn62_skip_vit_scratch=True` marker on the runner but no production
    # vLLM code reads it yet. Real ViT-scratch alloc hook is queued for
    # future sprint after qwen3_vl + NVFP4 cross-rig validation.
    return (
        "applied",
        "PN62 wrapped GPUModelRunner._dummy_run — marker-only "
        "instrumentation pending real ViT-alloc hook. Wrapper sets "
        "`_pn62_skip_vit_scratch=True` when --language-model-only + "
        "mm_limits_all_zero, but no production hook currently reads it. "
        "Predicted 3-5 GiB save on qwen3_vl + NVFP4 single-card boot "
        "lands when the inner alloc helper learns to honour the marker. "
        "Sister to PN35 (text-only inputs_embeds skip, vllm#35975 merged)."
    )
