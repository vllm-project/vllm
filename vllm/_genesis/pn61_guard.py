"""
[Genesis PN61] qwen3_vl NVFP4 loader guard.
Wraps Qwen3VLForConditionalGeneration.load_weights to auto-detect
missing ViT tower weights (common with NVFP4 quants) and set
language_model_only=True.

Backport reference: apnar club-3090#51 NVFP4 boot failure.
"""
import logging

log = logging.getLogger("genesis.pn61_qwen3_vl_keyerror_guard")

_VIT_KEY_PATTERNS = (
    "vision_tower", "visual", "vit_model", "visual.tower",
    "multi_modal_projector", "image_newline",
)


def _is_vit_keyerror(exc: KeyError) -> bool:
    """True if the KeyError name looks like a ViT/vision-tower weight key."""
    if not exc.args:
        return False
    name = str(exc.args[0])
    return any(p in name for p in _VIT_KEY_PATTERNS)


def _wrap_load_weights(original_load_weights):
    """Decorate load_weights with two-pass: pre-set language_model_only,
    then run, then catch leftover KeyError as final safety net."""
    def wrapped(self, weights, *args, **kwargs):
        # v2 pre-emptive path: check config-level hints
        try:
            config = getattr(self, "config", None)
            if config is not None and not getattr(
                config, "language_model_only", False
            ):
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
                "Auto-setting language_model_only=True for downstream paths.",
                e.args[0] if e.args else "?",
            )
            try:
                config = getattr(self, "config", None)
                if config is not None:
                    setattr(config, "language_model_only", True)
            except Exception as inner:
                log.debug("[PN61] could not set language_model_only: %s", inner)
            return 0
    wrapped.__wrapped__ = original_load_weights
    wrapped.__pn61_wrapped__ = True
    return wrapped


def apply_pn61() -> tuple[str, str]:
    """Apply PN61 — install class-rebind wrapper around qwen3_vl loader."""
    try:
        from vllm.model_executor.models import qwen3_vl as _vl_mod
    except Exception:
        return (
            "skipped",
            "vllm.model_executor.models.qwen3_vl not importable on this pin",
        )

    candidate_class_names = (
        "Qwen3VLForConditionalGeneration",
        "Qwen3VLForCausalLM",
        "Qwen3_VLForCausalLM",
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
            f"no qwen3_vl class with load_weights found in {_vl_mod.__name__}",
        )

    if getattr(target_cls.load_weights, "__pn61_wrapped__", False):
        return "applied", f"PN61 already wrapped {target_cls.__name__}.load_weights"

    target_cls.load_weights = _wrap_load_weights(target_cls.load_weights)
    return (
        "applied",
        f"PN61 wrapped {target_cls.__name__}.load_weights — ViT KeyError now "
        "auto-converts to WARN + language_model_only=True.",
    )

