# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.platforms import current_platform

logger = init_logger(__name__)

# Track whether upstream flash-attn is available on ROCm.
# Set during module initialization and never modified afterwards.
# This module-level flag avoids repeated import attempts and ensures
# consistent behavior (similar to IS_AITER_FOUND in _aiter_ops.py).
_ROCM_FLASH_ATTN_AVAILABLE = False

if current_platform.is_cuda():
    from vllm._custom_ops import reshape_and_cache_flash
    from vllm.vllm_flash_attn import (  # type: ignore[attr-defined]
        compile_flash_attn_varlen_func_from_specs,
        flash_attn_varlen_func,
        get_scheduler_metadata,
    )

elif current_platform.is_xpu():
    from vllm import _custom_ops as ops
    from vllm._xpu_ops import xpu_ops

    reshape_and_cache_flash = ops.reshape_and_cache_flash
    flash_attn_varlen_func = xpu_ops.flash_attn_varlen_func  # type: ignore[assignment]
    compile_flash_attn_varlen_func_from_specs = None  # type: ignore[assignment]
    get_scheduler_metadata = xpu_ops.get_scheduler_metadata  # type: ignore[assignment]
elif current_platform.is_rocm():
    # On ROCm we use AITER's Triton flash-attention; the upstream flash-attn
    # package is not installed/available. (Same source as aiter_triton_mla.py.)
    # The FA4 compile-from-specs API is CUDA-only, so it is unavailable on ROCm
    # regardless of whether AITER is present.
    from vllm.platforms.rocm import on_gfx1250

    compile_flash_attn_varlen_func_from_specs = None  # type: ignore[assignment]
    try:
        if on_gfx1250():
            from aiter.ops.triton.mha import (  # type: ignore[no-redef]
                flash_attn_varlen_func,
            )
        else:
            from flash_attn import flash_attn_varlen_func  # type: ignore[no-redef]

        _ROCM_FLASH_ATTN_AVAILABLE = True
    except ImportError:

        def flash_attn_varlen_func(*args: Any, **kwargs: Any) -> Any:  # type: ignore[no-redef,misc]
            package = "aiter" if on_gfx1250() else "flash-attn"
            raise ImportError(
                f"ROCm platform requires upstream {package} "
                f"to be installed. Please install {package} first."
            )

    # ROCm doesn't use scheduler metadata (FA3 feature), provide stub
    def get_scheduler_metadata(*args: Any, **kwargs: Any) -> None:  # type: ignore[misc]
        return None

    # ROCm uses the C++ custom op for reshape_and_cache
    from vllm import _custom_ops as ops

    reshape_and_cache_flash = ops.reshape_and_cache_flash


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
            elif (
                vllm_config is not None
                and vllm_config.model_config is not None
                and vllm_config.model_config.is_diffusion
            ):
                upgrade_reason = "Per-sequence causal (dynamic_causal) requires FA4"
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

        if fa_version == 4 and device_capability.major >= 10 and head_size == 256:
            logger.warning_once(
                "FA4 on Blackwell is temporarily disabled for head_size=256, "
                "defaulting to FA version 2."
            )
            fa_version = 2

        # FA4 on SM100 (Blackwell) has TMEM capacity limits that restrict
        # supported head dimensions to ≤128. The 192/128 MLA prefill case is
        # supported; 256 is temporarily disabled until upstream supports the
        # required features. Development of symmetric 192, 384, and 512 support
        # is tracked in https://github.com/Dao-AILab/flash-attention/issues/2456
        if (
            fa_version == 4
            and device_capability.major >= 10
            and head_size is not None
            and head_size > 128
            and not (head_size == 192 and head_size_v == 128)
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


def flash_attn_supports_kv_cache_dtype(
    kv_cache_dtype: str = "fp8_e4m3",
    *,
    requires_alibi: bool = False,
    head_size: int | None = None,
    head_size_v: int | None = None,
    has_sinks: bool = False,
) -> bool:
    if kv_cache_dtype == "fp8_e5m2":
        return False
    if current_platform.is_xpu():
        return True
    fa_version = get_flash_attn_version(
        requires_alibi=requires_alibi,
        head_size=head_size,
        head_size_v=head_size_v,
        has_sinks=has_sinks,
    )
    return (fa_version == 3 and current_platform.is_device_capability_family(90)) or (
        fa_version == 4 and current_platform.is_device_capability_family(100)
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
    - ROCm: aiter.ops.triton.mha.flash_attn_varlen_func (if AITER available) or
    upstream flash_attn.flash_attn_varlen_func

    Note: This is separate from the AITER flash attention backend (rocm_aiter_fa.py)
    which uses rocm_aiter_ops.flash_attn_varlen_func. The condition to use AITER is
    handled separately via _aiter_ops.is_aiter_found_and_supported().

    Returns:
        bool: True if a working flash_attn_varlen_func implementation is available.
    """
    if current_platform.is_cuda() or current_platform.is_xpu():
        # CUDA and XPU always have flash_attn_varlen_func available
        return True

    if current_platform.is_rocm():
        # Use the flag set during module import to check if
        # upstream flash-attn was successfully imported
        return _ROCM_FLASH_ATTN_AVAILABLE

    return False
