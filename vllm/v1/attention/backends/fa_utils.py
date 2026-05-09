# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.platforms import current_platform

logger = init_logger(__name__)

# Track which flash-attn varlen implementation is bound at module load.
# Set during module initialization and never modified afterwards.
# - "upstream": vllm.vllm_flash_attn (CUDA) / flash_attn (ROCm) / xpu_ops (XPU)
# - "aiter":    rocm_aiter_ops.flash_attn_varlen_func (ROCm fallback)
# - "none":     no working impl; flash_attn_varlen_func is a stub that raises
# Some kwargs (e.g. fa_version, block_table, seqused_k, softcap, q/k/v_descale,
# scheduler_metadata, num_splits, s_aux) are upstream-only. The AITER fallback
# is wrapped to raise NotImplementedError if any of them are passed, so callers
# can gate on FA_VARLEN_SOURCE / FA_VARLEN_HAS_VERSION_KW to avoid hitting that
# error path.
FA_VARLEN_SOURCE = "none"

# kwargs accepted by upstream flash_attn_varlen_func but not by AITER's variant.
# Listed here so the wrapper can reject them with a clear error rather than
# silently producing wrong results.
_AITER_UNSUPPORTED_KWARGS = (
    "block_table",
    "seqused_k",
    "softcap",
    "scheduler_metadata",
    "q_descale",
    "k_descale",
    "v_descale",
    "num_splits",
    "s_aux",
    "fa_version",
)


def _build_aiter_varlen_wrapper(aiter_fn: Any) -> Any:
    """Wrap AITER's flash_attn_varlen_func with the upstream calling convention.

    Differences AITER must absorb:
    - upstream allows cu_seqlens_k / max_seqlen_k to be None (they default to
      the q-side values); AITER requires them
    - upstream accepts kwargs (block_table, seqused_k, softcap, fa_version, ...)
      that AITER does not implement; reject those explicitly
    """

    def _wrapper(
        q: Any,
        k: Any,
        v: Any,
        cu_seqlens_q: Any,
        max_seqlen_q: int,
        cu_seqlens_k: Any = None,
        max_seqlen_k: int | None = None,
        dropout_p: float = 0.0,
        softmax_scale: float | None = None,
        causal: bool = False,
        window_size: Any = None,
        alibi_slopes: Any = None,
        return_lse: bool = False,
        out: Any = None,
        **kwargs: Any,
    ) -> Any:
        for unsupported in _AITER_UNSUPPORTED_KWARGS:
            if kwargs.get(unsupported) is not None:
                raise NotImplementedError(
                    f"AITER flash_attn_varlen_func fallback does not support "
                    f"`{unsupported}`. Install upstream flash-attn for ROCm to "
                    f"use this code path."
                )
        return aiter_fn(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k if cu_seqlens_k is not None else cu_seqlens_q,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k if max_seqlen_k is not None else max_seqlen_q,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            alibi_slopes=alibi_slopes,
            return_lse=return_lse,
            out=out,
        )

    return _wrapper


if current_platform.is_cuda():
    from vllm._custom_ops import reshape_and_cache_flash
    from vllm.vllm_flash_attn import (  # type: ignore[attr-defined]
        flash_attn_varlen_func,
        get_scheduler_metadata,
    )

    FA_VARLEN_SOURCE = "upstream"

elif current_platform.is_xpu():
    from vllm import _custom_ops as ops
    from vllm._xpu_ops import xpu_ops

    reshape_and_cache_flash = ops.reshape_and_cache_flash
    flash_attn_varlen_func = xpu_ops.flash_attn_varlen_func  # type: ignore[assignment]
    get_scheduler_metadata = xpu_ops.get_scheduler_metadata  # type: ignore[assignment]
    FA_VARLEN_SOURCE = "upstream"
elif current_platform.is_rocm():
    try:
        from flash_attn import flash_attn_varlen_func  # type: ignore[no-redef]

        FA_VARLEN_SOURCE = "upstream"
    except ImportError:
        # Upstream flash-attn isn't shipped in ROCm prebuilt wheels and a
        # from-source build of Composable Kernel can take 30-60 minutes.
        # Fall back to AITER's flash_attn_varlen_func which installs in
        # seconds and runs natively on gfx9. Without this fallback, callers
        # that import `flash_attn_varlen_func` from this module hit a stub
        # at call time, which previously sent some long-context paths into
        # an O(N^2) torch SDPA path and OOMed at 32K context. AITER is an
        # already-supported optional dep on ROCm (see vllm._aiter_ops), so
        # this preserves the existing dependency surface.
        from vllm._aiter_ops import IS_AITER_FOUND

        if IS_AITER_FOUND:
            from vllm._aiter_ops import rocm_aiter_ops

            flash_attn_varlen_func = _build_aiter_varlen_wrapper(  # type: ignore[no-redef,assignment]
                rocm_aiter_ops.flash_attn_varlen_func
            )
            FA_VARLEN_SOURCE = "aiter"
            logger.info_once(
                "Upstream flash-attn not found on ROCm; using AITER "
                "flash_attn_varlen_func as fallback. Install flash-attn "
                "from source for the upstream path."
            )
        else:

            def flash_attn_varlen_func(  # type: ignore[no-redef,misc]
                *args: Any, **kwargs: Any
            ) -> Any:
                raise ImportError(
                    "ROCm platform requires upstream flash-attn or AITER "
                    "to be installed. Install flash-attn from source, or "
                    "install AITER (`pip install amd-aiter`)."
                )

    # ROCm doesn't use scheduler metadata (FA3 feature), provide stub
    def get_scheduler_metadata(*args: Any, **kwargs: Any) -> None:  # type: ignore[misc]
        return None

    # ROCm uses the C++ custom op for reshape_and_cache
    from vllm import _custom_ops as ops

    reshape_and_cache_flash = ops.reshape_and_cache_flash

# Whether the bound flash_attn_varlen_func accepts the upstream-only
# `fa_version` kwarg. AITER's variant does not. Public so external callers
# (custom backends, etc.) can gate on it before passing fa_version.
FA_VARLEN_HAS_VERSION_KW = FA_VARLEN_SOURCE == "upstream"


def get_flash_attn_version(
    requires_alibi: bool = False,
    head_size: int | None = None,
    head_size_v: int | None = None,
    has_sinks: bool = False,
) -> int | None:
    if current_platform.is_xpu():
        return 2
    if current_platform.is_rocm():
        # ROCm doesn't use vllm_flash_attn; return None to skip fa_version arg
        return None
    try:
        from vllm.vllm_flash_attn.flash_attn_interface import (
            fa_version_unsupported_reason,
            is_fa_version_supported,
        )

        device_capability = current_platform.get_device_capability()

        assert device_capability is not None

        # 1. default version depending on platform
        if device_capability.major == 9 and is_fa_version_supported(3):
            # Hopper (SM90): prefer FA3
            fa_version = 3
        elif device_capability.major == 10 and is_fa_version_supported(4):
            # Blackwell (SM100+, restrict to SM100 for now): prefer FA4
            fa_version = 4
        else:
            # Fallback to FA2
            fa_version = 2

        # 2. override if passed by environment or config
        from vllm.config import get_current_vllm_config_or_none

        vllm_config = get_current_vllm_config_or_none()
        if (
            vllm_config is not None
            and vllm_config.attention_config.flash_attn_version is not None
        ):
            fa_version = vllm_config.attention_config.flash_attn_version

        # 3. fallback for unsupported combinations
        if device_capability.major >= 10 and fa_version == 3:
            logger.warning_once(
                "Cannot use FA version 3 on Blackwell platform, "
                "defaulting to FA version 4 if supported, otherwise FA2."
            )
            fa_version = 4 if is_fa_version_supported(4) else 2

        if requires_alibi and fa_version == 3:
            logger.warning_once(
                "Cannot use FA version 3 with ALiBi, defaulting to FA version 2."
            )
            fa_version = 2

        if requires_alibi and fa_version == 4:
            logger.warning_once(
                "Cannot use FA version 4 with ALiBi, defaulting to FA version 2."
            )
            fa_version = 2

        # Some FA3 unsupported SM90 cases can use FA4 when available.
        if (
            fa_version == 3
            and device_capability.major == 9
            and is_fa_version_supported(4)
        ):
            upgrade_reason = None
            if head_size is not None and head_size > 256:
                upgrade_reason = f"FA3 does not support head_size={head_size} on SM90"
            elif (
                has_sinks
                and head_size is not None
                and head_size_v is not None
                and head_size != head_size_v
            ):
                upgrade_reason = "Diff-KV with sinks"
            if upgrade_reason:
                logger.info_once(
                    "%s: upgrading FlashAttention 3 -> 4",
                    upgrade_reason,
                    scope="local",
                )
                fa_version = 4

        # FA4 currently uses batch-shape-dependent scheduling
        # heuristics on SM100+, which breaks batch invariance.
        if envs.VLLM_BATCH_INVARIANT and fa_version == 4:
            logger.warning_once(
                "Cannot use FA version 4 with batch invariance, "
                "defaulting to FA version 2.",
            )
            fa_version = 2

        # FA4 on SM100 (Blackwell) has TMEM capacity limits that restrict
        # supported head dimensions.
        # See: https://github.com/Dao-AILab/flash-attention/issues/1959
        # Exception: hdim 192 is supported for MLA's diff-headdim case
        # (qk=192, v=128), added upstream in commits 1a15733e/1b36ab19.
        if (
            fa_version == 4
            and device_capability.major >= 10
            and head_size is not None
            and head_size > 128
            and head_size != 192
        ):
            logger.warning_once(
                "FA4 on Blackwell does not support head_size=%d due to TMEM "
                "capacity limits, defaulting to FA version 2.",
                head_size,
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


def is_fa_version_supported(fa_version: int) -> bool:
    try:
        from vllm.vllm_flash_attn.flash_attn_interface import (
            is_fa_version_supported as _is_fa_version_supported,
        )

        return _is_fa_version_supported(fa_version)
    except ImportError:
        return False


def flash_attn_supports_fp8() -> bool:
    if current_platform.is_xpu():
        return True
    return (
        get_flash_attn_version() == 3
        and current_platform.is_device_capability_family(90)
    )


def flash_attn_supports_quant_query_input() -> bool:
    return not current_platform.is_xpu()


def flash_attn_supports_sinks() -> bool:
    if current_platform.is_xpu():
        return True
    return get_flash_attn_version() in (3, 4)


def flash_attn_supports_mla():
    from vllm.platforms import current_platform

    if current_platform.is_cuda():
        try:
            from vllm.vllm_flash_attn.flash_attn_interface import (
                is_fa_version_supported,
            )

            return is_fa_version_supported(
                3
            ) and current_platform.is_device_capability_family(90)

            # NOTE(Lucas): FA4 CuteDSL does NOT currently support MLA's non-standard
            # head dimensions (576 for qk, 512 for v) due to TMEM capacity limits.

        except (ImportError, AssertionError):
            pass
    return False


def is_flash_attn_varlen_func_available() -> bool:
    """Check if flash_attn_varlen_func is available.

    This function determines whether the flash_attn_varlen_func imported at module
    level is a working implementation or a stub.

    Platform-specific sources:
    - CUDA: vllm.vllm_flash_attn.flash_attn_varlen_func
    - XPU: xpu_ops.flash_attn_varlen_func
    - ROCm: upstream flash_attn.flash_attn_varlen_func, or AITER's
      rocm_aiter_ops.flash_attn_varlen_func as a fallback when upstream
      flash-attn is not installed.

    Returns:
        bool: True if a working flash_attn_varlen_func implementation is available.
    """
    if current_platform.is_cuda() or current_platform.is_xpu():
        # CUDA and XPU always have flash_attn_varlen_func available
        return True

    if current_platform.is_rocm():
        # True if either upstream flash-attn or the AITER fallback is bound.
        return FA_VARLEN_SOURCE != "none"

    return False
