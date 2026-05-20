# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compatibility wrapper for the optional `nunchaku` dependency.

`nunchaku` ships SVDQuant W4A4 / W4A16 CUDA kernels for diffusion
transformers on consumer NVIDIA GPUs (Turing through consumer Blackwell).
This module collects the lazy availability checks and lazy-imported call
wrappers so the rest of vLLM never imports `nunchaku` at module load
time.

Mirrors the structure of `vllm/utils/flashinfer.py` — `has_*` for
capability detection, `_lazy_import_wrapper` for the call boundary.
"""

import functools
import importlib
import importlib.util
from collections.abc import Callable
from typing import Any, NoReturn

from vllm.logger import init_logger

logger = init_logger(__name__)


@functools.cache
def has_nunchaku() -> bool:
    """Return True if the `nunchaku` package is importable."""
    if importlib.util.find_spec("nunchaku") is None:
        logger.debug_once("Nunchaku unavailable: package not installed")
        return False
    return True


def _get_submodule(module_name: str) -> Any | None:
    """Safely import a submodule, or return None if unavailable."""
    try:
        return importlib.import_module(module_name)
    except (ImportError, ModuleNotFoundError):
        return None


@functools.cache
def has_nunchaku_w4a4() -> bool:
    """Return True if Nunchaku's W4A4 GEMM + activation-quantize ops exist.

    Both ops are required for SVDQuant: the activation-side fused
    quantize+LoRA preprocessing and the main W4A4 scaled GEMM.
    """
    if not has_nunchaku():
        return False
    required = [
        ("nunchaku.ops.gemm", "svdq_gemm_w4a4_cuda"),
        ("nunchaku.ops.quantize", "svdq_quantize_w4a4_act_fuse_lora_cuda"),
    ]
    for module_name, attr_name in required:
        mod = _get_submodule(module_name)
        if mod is None or not hasattr(mod, attr_name):
            logger.debug_once(
                "Nunchaku W4A4 unavailable: missing %s.%s", module_name, attr_name
            )
            return False
    return True


@functools.cache
def has_nunchaku_w4a16() -> bool:
    """Return True if Nunchaku's W4A16 AWQ GEMV op exists.

    Used for batch-1 / decode-style paths where activations stay in
    fp16/bf16 and only the weight is 4-bit.
    """
    if not has_nunchaku():
        return False
    mod = _get_submodule("nunchaku.ops.gemv")
    return mod is not None and hasattr(mod, "awq_gemv_w4a16_cuda")


def _missing(*_: Any, **__: Any) -> NoReturn:
    # The PyPI `nunchaku` package is an unrelated Bayesian library; the
    # SVDQuant kernels are published only on the nunchaku-ai GitHub
    # releases page.
    raise RuntimeError(
        "Nunchaku is not installed. SVDQuant requires the nunchaku-ai "
        "wheels from https://github.com/nunchaku-ai/nunchaku/releases "
        "(do NOT `pip install nunchaku` — that pulls an unrelated PyPI "
        "package). Source: https://github.com/nunchaku-ai/nunchaku"
    )


def _lazy_import_wrapper(
    module_name: str, attr_name: str, fallback_fn: Callable[..., Any] = _missing
):
    """Build a lazy wrapper around a single nunchaku function.

    The first call resolves the underlying op via `importlib`; subsequent
    calls hit the cached resolution. The wrapper raises a clear error
    if the op was never resolved.
    """

    @functools.cache
    def _get_impl():
        if not has_nunchaku():
            return None
        mod = _get_submodule(module_name)
        return getattr(mod, attr_name, None) if mod else None

    def wrapper(*args, **kwargs):
        impl = _get_impl()
        if impl is None:
            return fallback_fn(*args, **kwargs)
        return impl(*args, **kwargs)

    wrapper.__name__ = attr_name
    wrapper.__qualname__ = f"nunchaku::{attr_name}"
    return wrapper


# Public lazy-call surface. Each wrapper has the same signature as the
# underlying nunchaku op (we don't re-document the signature here; see
# the upstream nunchaku source).
svdq_gemm_w4a4 = _lazy_import_wrapper("nunchaku.ops.gemm", "svdq_gemm_w4a4_cuda")
svdq_quantize_w4a4_act_fuse_lora = _lazy_import_wrapper(
    "nunchaku.ops.quantize", "svdq_quantize_w4a4_act_fuse_lora_cuda"
)
awq_gemv_w4a16 = _lazy_import_wrapper("nunchaku.ops.gemv", "awq_gemv_w4a16_cuda")


__all__ = [
    "has_nunchaku",
    "has_nunchaku_w4a4",
    "has_nunchaku_w4a16",
    "svdq_gemm_w4a4",
    "svdq_quantize_w4a4_act_fuse_lora",
    "awq_gemv_w4a16",
]
