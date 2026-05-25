"""
[Genesis PN62] Text-only ViT scratch skip.
Wraps GPUModelRunner._dummy_run to detect text-only mode
(--language-model-only + mm_limits_all_zero) and set a marker
that downstream alloc helpers can honour.

Sister patch to PN61 (NVFP4 loader guard).
Predicted 3-5 GiB save on single-card qwen3_vl + NVFP4 boot.
"""
import logging
import os

log = logging.getLogger("genesis.pn62_text_only_vit_skip")


def _is_text_only_mode(self) -> bool:
    """Returns True iff the runner is in 'text-only' regime where ViT
    scratch should be skipped."""
    config = getattr(self, "vllm_config", None) or getattr(self, "config", None)
    if config is not None:
        lmo = getattr(config, "language_model_only", False)
        if not lmo:
            mc = getattr(config, "model_config", None)
            if mc is not None:
                lmo = getattr(mc, "language_model_only", False)
        if not lmo:
            return False
    else:
        return False
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
            if os.environ.get("GENESIS_PN62_DEBUG", "") == "1":
                log.info(
                    "[PN62 text-only ViT skip] _dummy_run detected text-only "
                    "regime (--language-model-only + mm_limits_all_zero) — "
                    "marker SET (predicted 3-5 GiB save pending real ViT-alloc hook)."
                )
            try:
                setattr(self, "_pn62_skip_vit_scratch", True)
            except Exception:
                pass
        try:
            return original_dummy_run(self, *args, **kwargs)
        finally:
            try:
                if hasattr(self, "_pn62_skip_vit_scratch"):
                    delattr(self, "_pn62_skip_vit_scratch")
            except Exception:
                pass
    wrapped.__wrapped__ = original_dummy_run
    wrapped.__pn62_wrapped__ = True
    return wrapped


def apply_pn62() -> tuple[str, str]:
    """Apply PN62 — install class-rebind wrapper around _dummy_run."""
    try:
        from vllm.v1.worker.gpu_model_runner import GPUModelRunner as _Runner
    except Exception:
        return (
            "skipped",
            "GPUModelRunner not importable on this pin",
        )

    if not hasattr(_Runner, "_dummy_run"):
        return (
            "skipped",
            "GPUModelRunner._dummy_run not present on this pin",
        )

    if getattr(_Runner._dummy_run, "__pn62_wrapped__", False):
        return "applied", "PN62 already wrapped _dummy_run"

    _Runner._dummy_run = _wrap_dummy_run(_Runner._dummy_run)
    return (
        "applied",
        "PN62 wrapped GPUModelRunner._dummy_run — sets _pn62_skip_vit_scratch=True "
        "marker when in text-only mode. No production hook reads it yet "
        "(pending ViT-alloc hook integration).",
    )

