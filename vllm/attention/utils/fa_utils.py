# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.logger import init_logger
from vllm.platforms import current_platform

logger = init_logger(__name__)

if current_platform.is_cuda():
    from vllm import _custom_ops as ops

    reshape_and_cache_flash = ops.reshape_and_cache_flash
    from vllm.vllm_flash_attn import flash_attn_varlen_func, get_scheduler_metadata
    from vllm.vllm_flash_attn.flash_attn_interface import (
        FA2_AVAILABLE,
        FA2_UNAVAILABLE_REASON,
        FA3_AVAILABLE,
        FA3_UNAVAILABLE_REASON,
    )
elif current_platform.is_xpu():
    from vllm._ipex_ops import ipex_ops as ops

    reshape_and_cache_flash = ops.reshape_and_cache_flash
    flash_attn_varlen_func = ops.flash_attn_varlen_func
    get_scheduler_metadata = ops.get_scheduler_metadata
elif current_platform.is_rocm():
    try:
        from flash_attn import flash_attn_varlen_func  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "Rocm platform requires upstream flash-attn "
            "to be installed. Please install flash-attn first."
        ) from e


# Functions copied from vllm/vllm_flash_attn/flash_attn_interface.py
# Modified to use current_platform.get_device_capability() instead of
# torch.cuda.get_device_capability(device) because current_platform.get_device_capability()
# does not initialize CUDA.
def _is_fa2_supported(device=None) -> Tuple[bool, Optional[str]]:
    if not FA2_AVAILABLE:
        return False, f"FA2 is unavaible due to: {FA2_UNAVAILABLE_REASON}"
    device_capability = current_platform.get_device_capability()
    if device_capability.major < 8:
        return (
            False,
            "FA2 is only supported on devices with compute capability >= 8",
        )
    return True, None


def _is_fa3_supported(device=None) -> Tuple[bool, Optional[str]]:
    if not FA3_AVAILABLE:
        return False, f"FA3 is unavaible due to: {FA3_UNAVAILABLE_REASON}"
    device_capability = current_platform.get_device_capability()
    if (
        device_capability.major < 8
        or device_capability.major >= 10
        or device_capability == (8, 6)
        or device_capability == (8, 9)
    ):
        return (
            False,
            "FA3 is only supported on devices with compute capability >= 8"
            " excluding 8.6 and 8.9 and Blackwell archs (>=10)",
        )
    return True, None


def is_fa_version_supported(fa_version: int, device=None) -> bool:
    assert fa_version in [2, 3], f"Unsupported FA version: {fa_version}"
    if fa_version == 2:
        return _is_fa2_supported(device)[0]
    elif fa_version == 3:
        return _is_fa3_supported(device)[0]


def fa_version_unsupported_reason(fa_version: int, device=None) -> Optional[str]:
    assert fa_version in [2, 3], f"Unsupported FA version: {fa_version}"
    if fa_version == 2:
        return _is_fa2_supported(device)[1]
    elif fa_version == 3:
        return _is_fa3_supported(device)[1]


def get_flash_attn_version(requires_alibi: bool = False) -> int | None:
    # import here to avoid circular dependencies
    from vllm.platforms import current_platform

    if current_platform.is_xpu():
        return 2
    try:
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
    return current_platform.is_cuda() or current_platform.is_xpu()
