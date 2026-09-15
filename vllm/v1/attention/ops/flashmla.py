# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# adapted from: https://github.com/deepseek-ai/FlashMLA/blob/main/flash_mla/flash_mla_interface.py

import functools

import torch

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.platforms import current_platform

logger = init_logger(__name__)

if current_platform.is_cuda():
    try:
        import vllm._flashmla_C  # noqa: F401

        _flashmla_C_AVAILABLE = True
    except ImportError:
        _flashmla_C_AVAILABLE = False
else:
    _flashmla_C_AVAILABLE = False

if current_platform.is_cuda():
    try:
        import vllm._flashmla_extension_C  # noqa: F401

        _flashmla_extension_C_AVAILABLE = True
    except ImportError:
        _flashmla_extension_C_AVAILABLE = False
else:
    _flashmla_extension_C_AVAILABLE = False


def _is_flashmla_available() -> tuple[bool, str | None]:
    if not _flashmla_C_AVAILABLE:
        return (
            False,
            "vllm._flashmla_C is not available, likely was not "
            "compiled due to insufficient nvcc version or a supported arch "
            "was not in the list of target arches to compile for.",
        )
    if not _flashmla_extension_C_AVAILABLE:
        return (
            False,
            "vllm._flashmla_extension_C is not available, likely "
            "was not compiled due to a build error.",
        )

    return True, None


def is_flashmla_dense_supported() -> tuple[bool, str | None]:
    """
    Return: is_supported_flag, unsupported_reason (optional).
    """
    is_available, maybe_reason = _is_flashmla_available()
    if not is_available:
        return False, maybe_reason
    if not current_platform.is_device_capability_family(90):
        return False, "FlashMLA Dense is only supported on Hopper devices."
    return True, None


@functools.lru_cache(maxsize=1)
def _use_triton_sparse_mla() -> bool:
    """Whether to bind the portable Triton sparse-MLA kernels instead of the
    native FlashMLA ops. The native sparse kernels (vllm._flashmla_C) only
    build for sm90/sm100; on consumer Blackwell (SM12x: RTX 5090 / GB10) the
    Triton implementations in sm12x_sparse_mla_attn.py are the only working
    sparse-MLA path. Unset VLLM_TRITON_MLA_SPARSE = auto (SM12x only);
    "1"/"0" force it on/off for A/B testing.

    The result is cached (maxsize=1) so the import-time symbol binding below and
    every runtime consumer agree on a single decision. Without caching, a probe
    that raised at import time (binding ``_raise_flashmla_unavailable``) could
    later return True at runtime, enabling the SM12x path against a module that
    still holds the raising stubs -> ``RuntimeError: FlashMLA is not available``
    on the first FP8 decode. Env/capability are fixed for a process lifetime, so
    a stable cached decision is correct."""
    try:
        if not current_platform.is_cuda():
            return False
        configured = envs.VLLM_TRITON_MLA_SPARSE
        if configured is not None:
            return configured
        return current_platform.is_device_capability_family(120)
    except Exception:  # pragma: no cover - platform probing must never break import
        # Log rather than silently swallow: a probe failure here freezes the
        # cached decision to False, so it must be visible if it ever happens.
        logger.warning(
            "Triton sparse-MLA capability probe failed; assuming disabled. "
            "Set VLLM_TRITON_MLA_SPARSE=1 to force it on.",
            exc_info=True,
        )
        return False


def is_flashmla_sparse_supported() -> tuple[bool, str | None]:
    """
    Return: is_supported_flag, unsupported_reason (optional).
    """
    if _use_triton_sparse_mla():
        return True, None
    is_available, maybe_reason = _is_flashmla_available()
    if not is_available:
        return False, maybe_reason
    if not (
        current_platform.is_device_capability_family(90)
        or current_platform.is_device_capability_family(100)
    ):
        return (
            False,
            "FlashMLA Sparse is only supported on Hopper and Blackwell DC devices.",
        )
    return True, None


def _raise_flashmla_unavailable(*_args, **_kwargs):
    _, reason = _is_flashmla_available()
    raise RuntimeError(reason or "FlashMLA is not available")


if _use_triton_sparse_mla():
    # SM12x (or forced): portable Triton sparse-MLA kernels. The dense varlen
    # entry points still require the native extension and keep raising.
    from vllm.v1.attention.backends.mla.sm12x_sparse_mla_attn import (  # noqa: F401
        flash_mla_sparse_fwd_triton as flash_mla_sparse_fwd,
    )
    from vllm.v1.attention.backends.mla.sm12x_sparse_mla_attn import (
        flash_mla_with_kvcache_triton as flash_mla_with_kvcache,
    )

    class FlashMLASchedMeta:  # type: ignore[no-redef]
        pass

    def get_mla_metadata(*_args, **_kwargs):  # type: ignore[misc]
        # Computes NATIVE FlashMLA tile-scheduler metadata (needs _flashmla_C).
        # The Triton kernels self-schedule and ignore tile_scheduler_metadata
        # (a None-default arg), so return (None, None) instead of calling it.
        return None, None

    flash_attn_varlen_func = _raise_flashmla_unavailable  # type: ignore[assignment]
    flash_attn_varlen_kvpacked_func = _raise_flashmla_unavailable  # type: ignore[assignment]
    flash_attn_varlen_qkvpacked_func = _raise_flashmla_unavailable  # type: ignore[assignment]
elif _is_flashmla_available()[0]:
    from vllm.third_party.flashmla.flash_mla_interface import (  # noqa: F401
        FlashMLASchedMeta,
        flash_attn_varlen_func,
        flash_attn_varlen_kvpacked_func,
        flash_attn_varlen_qkvpacked_func,
        flash_mla_sparse_fwd,
        flash_mla_with_kvcache,
        get_mla_metadata,
    )
else:

    class FlashMLASchedMeta:  # type: ignore[no-redef]
        pass

    flash_attn_varlen_func = _raise_flashmla_unavailable  # type: ignore[assignment]
    flash_attn_varlen_kvpacked_func = _raise_flashmla_unavailable  # type: ignore[assignment]
    flash_attn_varlen_qkvpacked_func = _raise_flashmla_unavailable  # type: ignore[assignment]
    flash_mla_sparse_fwd = _raise_flashmla_unavailable  # type: ignore[assignment]
    flash_mla_with_kvcache = _raise_flashmla_unavailable  # type: ignore[assignment]
    get_mla_metadata = _raise_flashmla_unavailable  # type: ignore[assignment]


def get_mla_metadata_dense_fp8(
    cache_seqlens: torch.Tensor,
    num_q_tokens_per_head_k: int,
    num_heads_k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not _is_flashmla_available()[0]:
        _raise_flashmla_unavailable()
    return torch.ops._flashmla_extension_C.get_mla_decoding_metadata_dense_fp8(
        cache_seqlens,
        num_q_tokens_per_head_k,
        num_heads_k,
    )


def flash_mla_with_kvcache_fp8(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    block_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    head_dim_v: int,
    tile_scheduler_metadata: torch.Tensor,
    num_splits: torch.Tensor,
    softmax_scale: float | None = None,
    causal: bool = False,
    descale_q: torch.Tensor | None = None,
    descale_k: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not _is_flashmla_available()[0]:
        _raise_flashmla_unavailable()
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)
    out, softmax_lse = torch.ops._flashmla_extension_C.fwd_kvcache_mla_fp8(
        q,
        k_cache,
        head_dim_v,
        cache_seqlens,
        block_table,
        softmax_scale,
        causal,
        tile_scheduler_metadata,
        num_splits,
        descale_q,
        descale_k,
    )
    return out, softmax_lse
