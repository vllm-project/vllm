# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.logger import init_logger
from vllm.platforms import current_platform

logger = init_logger(__name__)
_ROCM_FLASH_ATTN_AVAILABLE = False
_ROCM_FLASH_ATTN_IMPORT_ERROR: Exception | None = None

if current_platform.is_cuda():
    from vllm import _custom_ops as ops

    reshape_and_cache_flash = ops.reshape_and_cache_flash
    from vllm.vllm_flash_attn import flash_attn_varlen_func, get_scheduler_metadata
elif current_platform.is_xpu():
    from vllm._ipex_ops import ipex_ops as ops

    reshape_and_cache_flash = ops.reshape_and_cache_flash
    flash_attn_varlen_func = ops.flash_attn_varlen_func
    get_scheduler_metadata = ops.get_scheduler_metadata
elif current_platform.is_rocm():
    import torch

    from vllm import _custom_ops as ops

    reshape_and_cache_flash = ops.reshape_and_cache_flash

    try:
        from flash_attn import flash_attn_varlen_func as _flash_attn_varlen_func

        _ROCM_FLASH_ATTN_AVAILABLE = True
    except ImportError as e:
        _ROCM_FLASH_ATTN_IMPORT_ERROR = e
        _flash_attn_varlen_func = None

    def get_scheduler_metadata(*_args, **_kwargs):
        return None

    def flash_attn_varlen_func(  # noqa: PLR0913
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor | None = None,
        max_seqlen_q: int | None = None,
        max_seqlen_k: int | None = None,
        dropout_p: float = 0.0,
        softmax_scale: float | None = None,
        causal: bool = False,
        window_size: tuple[int, int] = (-1, -1),
        softcap: float = 0.0,
        alibi_slopes: torch.Tensor | None = None,
        deterministic: bool = False,
        return_softmax_lse: bool = False,
        return_attn_probs: bool = False,
        block_table: torch.Tensor | None = None,
        out: torch.Tensor | None = None,
        seqused_k: torch.Tensor | None = None,
        scheduler_metadata: torch.Tensor | None = None,
        fa_version: int | None = None,
        num_splits: int = 0,
        q_descale: torch.Tensor | None = None,
        k_descale: torch.Tensor | None = None,
        v_descale: torch.Tensor | None = None,
        s_aux: torch.Tensor | None = None,
        **_kwargs,
    ):
        del scheduler_metadata, fa_version, num_splits, q_descale, k_descale, v_descale
        del s_aux

        if _flash_attn_varlen_func is None:
            raise ImportError(
                "ROCm FlashAttention is not available. "
                "Please install flash-attn built for ROCm."
            ) from _ROCM_FLASH_ATTN_IMPORT_ERROR

        if cu_seqlens_k is None:
            raise ValueError(
                "ROCm FlashAttention requires cu_seqlens_k to be provided "
                "to remain CUDA-graph compatible."
            )

        want_attn_probs = return_attn_probs
        return_attn_probs = return_attn_probs or return_softmax_lse

        result = _flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            softcap=softcap,
            alibi_slopes=alibi_slopes,
            deterministic=deterministic,
            return_attn_probs=return_attn_probs,
            block_table=block_table,
        )

        if return_attn_probs:
            if isinstance(result, tuple):
                out_res, lse, attn_probs = result
            else:
                out_res = result
                lse = None
                attn_probs = None
            if out is not None:
                out.copy_(out_res)
                out_res = out
            if return_softmax_lse and not want_attn_probs:
                return out_res, lse
            return out_res, lse, attn_probs

        if out is not None:
            out.copy_(result)
            return out
        return result


def get_flash_attn_version(requires_alibi: bool = False) -> int | None:
    # import here to avoid circular dependencies
    from vllm.platforms import current_platform

    if current_platform.is_xpu():
        return 2
    if current_platform.is_rocm():
        if not is_flash_attn_varlen_func_available():
            return None
        from vllm.config import get_current_vllm_config

        vllm_config = get_current_vllm_config()
        if vllm_config.attention_config.flash_attn_version not in (None, 2):
            logger.warning_once(
                "ROCm FlashAttention only supports FA version 2; "
                "defaulting to FA version 2."
            )
        return 2
    try:
        from vllm.vllm_flash_attn.flash_attn_interface import (
            fa_version_unsupported_reason,
            is_fa_version_supported,
        )

        device_capability = current_platform.get_device_capability()

        assert device_capability is not None

        # 1. default version depending on platform
        fa_version = (
            3 if (device_capability.major == 9 and is_fa_version_supported(3)) else 2
        )

        # 2. override if passed by environment or config
        from vllm.config import get_current_vllm_config

        vllm_config = get_current_vllm_config()
        if vllm_config.attention_config.flash_attn_version is not None:
            fa_version = vllm_config.attention_config.flash_attn_version

        # 3. fallback for unsupported combinations
        if device_capability.major == 10 and fa_version == 3:
            logger.warning_once(
                "Cannot use FA version 3 on Blackwell platform "
                "defaulting to FA version 2."
            )
            fa_version = 2

        if requires_alibi and fa_version == 3:
            logger.warning_once(
                "Cannot use FA version 3 with ALiBi, defaulting to FA version 2."
            )
            fa_version = 2

        if not is_fa_version_supported(fa_version):
            logger.error(
                "Cannot use FA version %d is not supported due to %s",
                fa_version,
                fa_version_unsupported_reason(fa_version),
            )

        assert is_fa_version_supported(fa_version)
        return fa_version
    except (ImportError, AssertionError):
        return None


def flash_attn_supports_fp8() -> bool:
    return (
        get_flash_attn_version() == 3
        and current_platform.get_device_capability().major == 9
    )


def flash_attn_supports_sinks() -> bool:
    if current_platform.is_xpu():
        return True
    else:
        return get_flash_attn_version() == 3


def flash_attn_supports_mla():
    from vllm.platforms import current_platform

    if current_platform.is_cuda():
        try:
            from vllm.vllm_flash_attn.flash_attn_interface import (
                is_fa_version_supported,
            )

            return (
                is_fa_version_supported(3)
                and current_platform.get_device_capability()[0] == 9
            )
        except (ImportError, AssertionError):
            pass
    return False


def is_flash_attn_varlen_func_available() -> bool:
    if current_platform.is_rocm():
        return _ROCM_FLASH_ATTN_AVAILABLE
    return current_platform.is_cuda() or current_platform.is_xpu()
