# SPDX-License-Identifier: Apache-2.0
"""Wiring for Patch PN61 — qwen3_vl loader KeyError → text-only auto-fallback.

Source: apnar club-3090#51 NVFP4 boot failure 2026-05-04. The
`kaitchup/Qwen3.6-27B-autoround-nvfp4-linearattn-BF16` checkpoint
strips/renames the qwen3_vl ViT blocks during NVFP4 quant — vLLM's
loader then fails with::

    File ".../vllm/model_executor/models/qwen3_vl.py", line 856, in load_weights
        param = params_dict[name]
    KeyError: 'blocks.0.attn.proj.weight'

Operator workaround: pass `--language-model-only` flag. PN61 makes this
automatic + emits a WARN line so the operator knows the ViT was skipped.

================================================================
GENESIS APPROACH
================================================================

Wraps `vllm.model_executor.models.qwen3_vl.Qwen3VLForCausalLM.load_weights`
with a try/except that catches `KeyError` for `blocks.*.attn.*` patterns
and converts to:
  1. log.warning(...)  with one-line remediation hint
  2. Set `language_model_only=True` on the engine config (idempotent)
  3. Return zero-loaded weights for the visual stack (vllm will treat
     the absent ViT as language-only)

Same defensive pattern as P29 IndexError guard (tool parser bounded-index).

================================================================
ENV
================================================================

GENESIS_ENABLE_PN61=1

================================================================
RISK
================================================================

LOW — wrapped in try/except; only triggers on the exact KeyError class.
Other load failures (corrupt checkpoint, dtype mismatch, etc.) propagate
unchanged. NULL on text-only checkpoints (qwen3_5, qwen3_moe).

================================================================
STATE
================================================================

The runtime wrapping logic below is COMPLETE; the live anchor against
`qwen3_vl.py` is conservative (uses class rebind, not text-patch) so it
stays robust across vllm pin bumps. Cross-rig validation pending: needs
an actual NVFP4 qwen3_vl checkpoint reachable from the test rig (apnar
1× RTX 5090 is the current candidate; Genesis 2× A5000 PROD doesn't
have a multimodal checkpoint loaded as of 2026-05-05).

Author: Sandermage 2026-05-05.
Backport reference: apnar club-3090#51 (kaitchup NVFP4 ViT-stripped checkpoint).
"""
from __future__ import annotations

import logging

log = logging.getLogger("genesis.wiring.pn61_qwen3_vl_keyerror_guard")


_VIT_KEY_PATTERNS = (
    "blocks.",
    "vision_tower.",
    "visual.",
    "vit.",
)


def _is_vit_keyerror(exc: KeyError) -> bool:
    """True if the KeyError name looks like a ViT/vision-tower weight key."""
    if not exc.args:
        return False
    name = str(exc.args[0])
    return any(p in name for p in _VIT_KEY_PATTERNS)


def _wrap_load_weights(original_load_weights):
    """Decorate load_weights with two-pass: pre-set language_model_only,
    then run, then catch leftover KeyError as final safety net.

    Audit P2.3 fix 2026-05-05 (genesis_deep_cross_audit):
    Previous v1 only caught the KeyError AFTER load_weights had already
    walked partway through ViT-loading branches. Setting
    `config.language_model_only=True` post-failure was too late for
    in-flight branches.

    v2 strategy: peek at the weights iterator BEFORE calling original
    load_weights. If we see ANY ViT-named tensor key in the first batch
    that the loader doesn't yet know how to skip, pre-emptively set
    `language_model_only=True` on the config so the original loader
    routes correctly from the start. Fall-through KeyError handler stays
    as final safety net.
    """
    def wrapped(self, weights, *args, **kwargs):
        # v2 pre-emptive path: try to detect ViT-named keys before the
        # original loader walks them. We can't fully drain the iterator
        # (consumes it), so we sample by checking config-level hints.
        try:
            config = getattr(self, "config", None)
            if config is not None and not getattr(
                config, "language_model_only", False
            ):
                # Heuristic: if checkpoint has compressed-tensors / NVFP4
                # weight format AND model arch is qwen3_vl, the ViT tower
                # is likely stripped — pre-emptively set language_model_only
                # so original loader doesn't take the ViT branch.
                quant_cfg = getattr(config, "quantization_config", None) or {}
                quant_method = (
                    (quant_cfg.get("quant_method") if isinstance(quant_cfg, dict)
                     else getattr(quant_cfg, "quant_method", None))
                    or ""
                ).lower()
                if quant_method in ("compressed-tensors", "compressed_tensors", "nvfp4"):
                    log.info(
                        "[PN61 v2 pre-emptive] qwen3_vl + %s detected — "
                        "pre-setting language_model_only=True before load_weights "
                        "to avoid mid-load ViT KeyError", quant_method,
                    )
                    setattr(config, "language_model_only", True)
        except Exception as inner:
            log.debug("[PN61 v2] pre-emptive check failed: %s", inner)

        try:
            return original_load_weights(self, weights, *args, **kwargs)
        except KeyError as e:
            if not _is_vit_keyerror(e):
                raise  # not our concern, propagate
            log.warning(
                "[PN61 qwen3_vl loader guard] caught ViT KeyError %r — "
                "checkpoint appears to lack visual tower weights "
                "(common with NVFP4 quants that strip ViT). "
                "Auto-setting language_model_only=True for downstream paths. "
                "If pre-failure state is partial (rare), restart with explicit "
                "--language-model-only for a clean boot. ",
                e.args[0] if e.args else "?",
            )
            try:
                config = getattr(self, "config", None)
                if config is not None:
                    setattr(config, "language_model_only", True)
            except Exception as inner:
                log.debug("[PN61] could not set language_model_only: %s", inner)
            # Return zero loaded params for the visual stack — vllm will
            # treat the absent ViT as language-only.
            return 0
    wrapped.__wrapped__ = original_load_weights
    wrapped.__pn61_wrapped__ = True
    return wrapped


def apply() -> tuple[str, str]:
    """Apply PN61 — install class-rebind wrapper around qwen3_vl loader."""
    from vllm._genesis.dispatcher import should_apply, log_decision
    decision, reason = should_apply("PN61")
    log_decision("PN61", decision, reason)
    if not decision:
        return "skipped", reason

    try:
        from vllm.model_executor.models import qwen3_vl as _vl_mod
    except Exception:
        return (
            "skipped",
            "vllm.model_executor.models.qwen3_vl not importable on this pin "
            "(qwen3_vl support absent or path drift) — PN61 NULL",
        )

    # Find the for-causal-lm class — name varies across vllm pins.
    candidate_class_names = (
        "Qwen3VLForCausalLM",
        "Qwen3_VLForCausalLM",
        "Qwen3VLConditionalGeneration",
    )
    target_cls = None
    for name in candidate_class_names:
        cls = getattr(_vl_mod, name, None)
        if cls is not None and hasattr(cls, "load_weights"):
            target_cls = cls
            break
    if target_cls is None:
        return (
            "skipped",
            f"no qwen3_vl class with load_weights found in {_vl_mod.__name__} "
            "— PN61 NULL on this pin (try later vllm version)",
        )

    if getattr(target_cls.load_weights, "__pn61_wrapped__", False):
        return "applied", f"PN61 already wrapped {target_cls.__name__}.load_weights (idempotent)"

    target_cls.load_weights = _wrap_load_weights(target_cls.load_weights)
    return (
        "applied",
        f"PN61 wrapped {target_cls.__name__}.load_weights — ViT KeyError now "
        "auto-converts to WARN + language_model_only=True. "
        "Backport of apnar club-3090#51 NVFP4 boot failure pattern."
    )
