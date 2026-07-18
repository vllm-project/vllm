# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Runtime monkeypatch: route the native FlashMLA *sparse* ops to Triton.

On consumer Blackwell (sm_121 / SM12x) the compiled ``vllm._flashmla_C``
extension does not exist, so ``vllm.v1.attention.ops.flashmla`` binds
``flash_mla_sparse_fwd`` and ``flash_mla_with_kvcache`` to a stub that raises
``"vllm._flashmla_C is not available"``.  The V3.2 sparse-MLA backend
(``flashmla_sparse.py``) crashes on the first real request.

Importing this module (e.g. very early in engine startup, or via a vLLM plugin
entrypoint) rebinds those two names to the portable Triton implementations in
``sm12x_sparse_mla_attn.py`` whenever:

    * ``is_triton_sparse_mla_enabled()`` is true (auto on sm12x, or forced via
      ``VLLM_TRITON_MLA_SPARSE=1``), OR
    * ``vllm._flashmla_C`` is unavailable (the stub is currently bound).

It also makes the backend believe sparse is supported by patching
``is_flashmla_sparse_supported`` to return ``True`` on SM12x, and registers the
``VLLM_TRITON_MLA_SPARSE*`` env vars into ``vllm.envs`` if the running image's
``envs.py`` predates them (so the copied ``sparse_mla_env.py`` /
``sparse_mla_kernels.py`` import cleanly without editing ``envs.py``).

Idempotent: calling ``apply()`` more than once is safe.
"""

from __future__ import annotations

import os

from vllm.logger import init_logger

logger = init_logger(__name__)

_APPLIED = False


# ---------------------------------------------------------------------------
# 1. Backfill VLLM_TRITON_MLA_SPARSE* env vars if the image predates them.
# ---------------------------------------------------------------------------
def _ensure_envs() -> None:
    import vllm.envs as envs

    def _opt_bool(name: str):
        raw = os.getenv(name)
        if raw is None:
            return None
        return bool(int(raw))

    def _opt_int(name: str):
        raw = os.getenv(name)
        if raw is None:
            return None
        return int(raw)

    defaults: dict[str, object] = {
        # None => auto-select (sparse_mla_env decides per-device).
        "VLLM_TRITON_MLA_SPARSE": lambda: _opt_bool("VLLM_TRITON_MLA_SPARSE"),
        "VLLM_TRITON_MLA_SPARSE_TOPK_CHUNK_SIZE": lambda: int(
            os.getenv("VLLM_TRITON_MLA_SPARSE_TOPK_CHUNK_SIZE", "512")
        ),
        "VLLM_TRITON_MLA_SPARSE_QUERY_CHUNK_SIZE": lambda: int(
            os.getenv("VLLM_TRITON_MLA_SPARSE_QUERY_CHUNK_SIZE", "256")
        ),
        "VLLM_TRITON_MLA_SPARSE_HEAD_BLOCK_SIZE": lambda: _opt_int(
            "VLLM_TRITON_MLA_SPARSE_HEAD_BLOCK_SIZE"
        ),
        "VLLM_TRITON_MLA_SPARSE_MATMUL_DECODE": lambda: _opt_bool(
            "VLLM_TRITON_MLA_SPARSE_MATMUL_DECODE"
        ),
    }
    registry = getattr(envs, "environment_variables", None)
    if registry is None:
        logger.warning(
            "vllm.envs has no environment_variables registry; cannot backfill "
            "VLLM_TRITON_MLA_SPARSE* env vars."
        )
        return
    for name, fn in defaults.items():
        if name not in registry:
            registry[name] = fn
            logger.info("Registered missing env var %s into vllm.envs", name)


# ---------------------------------------------------------------------------
# 2. Decide whether to enable the Triton sparse path.
# ---------------------------------------------------------------------------
def _should_enable() -> bool:
    from vllm.v1.attention.ops import flashmla as ops_flashmla

    # Stub currently bound (i.e. _flashmla_C unavailable) -> always enable, we
    # have no alternative.
    native_available, _ = ops_flashmla._is_flashmla_available()
    if not native_available:
        return True

    # Native available but user/device wants Triton (e.g. SM12x with a partial
    # _flashmla_C, or forced via env).
    try:
        from vllm.v1.attention.backends.mla.sparse_mla_env import (
            is_triton_sparse_mla_enabled_for_platform,
        )

        return bool(is_triton_sparse_mla_enabled_for_platform())
    except Exception:  # pragma: no cover - defensive
        return False


# ---------------------------------------------------------------------------
# 3. Apply the rebind.
# ---------------------------------------------------------------------------
def apply(force: bool = False) -> bool:
    """Rebind the native sparse ops to Triton. Returns True if applied."""
    global _APPLIED
    if _APPLIED and not force:
        return True

    _ensure_envs()

    if not force and not _should_enable():
        logger.info(
            "Native FlashMLA sparse ops available and Triton path not requested; "
            "leaving flash_mla_sparse_fwd / flash_mla_with_kvcache untouched."
        )
        return False

    from vllm.v1.attention.ops import flashmla as ops_flashmla

    # Import the adapter. When this file lives inside
    # vllm/v1/attention/backends/mla/ the package-relative import works; the
    # fallbacks let it also run as a loose module (e.g. in the validation
    # container) where it is importable by bare name.
    try:
        from .sm12x_sparse_mla_attn import (
            flash_mla_sparse_fwd_triton,
            flash_mla_with_kvcache_triton,
        )
    except ImportError:
        try:
            from vllm.v1.attention.backends.mla.sm12x_sparse_mla_attn import (
                flash_mla_sparse_fwd_triton,
                flash_mla_with_kvcache_triton,
            )
        except ImportError:
            from sm12x_sparse_mla_attn import (
                flash_mla_sparse_fwd_triton,
                flash_mla_with_kvcache_triton,
            )

    ops_flashmla.flash_mla_sparse_fwd = flash_mla_sparse_fwd_triton
    ops_flashmla.flash_mla_with_kvcache = flash_mla_with_kvcache_triton

    # SM12x: get_mla_metadata computes the NATIVE FlashMLA tile-scheduler
    # metadata (needs _flashmla_C). The fp8-mixed-batch decode metadata builder
    # (flashmla_sparse.py _build_fp8_mixed_decode_prefill) calls it, but our
    # Triton kernels self-schedule and ignore tile_scheduler_metadata (it's a
    # None-default arg). Return (None, None) so no native call is made.
    def _sm12x_get_mla_metadata(*args, **kwargs):  # noqa: ANN001, ANN002
        return None, None

    ops_flashmla.get_mla_metadata = _sm12x_get_mla_metadata

    # The V3.2 backend test gate is `flashmla.is_flashmla_sparse_supported()`;
    # report supported on SM12x so the backend is actually selected.
    _orig_supported = ops_flashmla.is_flashmla_sparse_supported

    def _patched_is_flashmla_sparse_supported() -> tuple[bool, str | None]:
        try:
            from vllm.platforms import current_platform

            if current_platform.is_device_capability_family(120):
                return True, None
        except Exception:  # pragma: no cover - defensive
            pass
        return _orig_supported()

    ops_flashmla.is_flashmla_sparse_supported = _patched_is_flashmla_sparse_supported

    _APPLIED = True
    logger.info(
        "Patched vllm.v1.attention.ops.flashmla: flash_mla_sparse_fwd and "
        "flash_mla_with_kvcache now use portable Triton sparse-MLA kernels "
        "(SM12x / no _flashmla_C)."
    )
    return True


# Auto-apply on import for convenience. Callers that want explicit control can
# import ``apply`` and invoke it themselves (it is idempotent).
try:  # pragma: no cover - exercised at runtime on the cluster
    apply()
except Exception as exc:  # pragma: no cover - never hard-fail import
    logger.warning("patch_flashmla_ops auto-apply failed: %s", exc)
