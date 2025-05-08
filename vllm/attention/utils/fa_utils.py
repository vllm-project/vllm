# SPDX-License-Identifier: Apache-2.0
from typing import Optional

from vllm import envs
from vllm.logger import init_logger

logger = init_logger(__name__)


def get_flash_attn_version(requires_alibi: bool = False) -> Optional[int]:
    # import here to avoid circular dependencies
    from vllm.platforms import current_platform
    try:
        from vllm.vllm_flash_attn.flash_attn_interface import (
            fa_version_unsupported_reason, is_fa_version_supported)
        device_capability = current_platform.get_device_capability()

        assert device_capability is not None

        # 1. default version depending on platform
        fa_version = 3 if (device_capability.major == 9
                           and is_fa_version_supported(3)) else 2

        # 2. override if passed by environment
        if envs.VLLM_FLASH_ATTN_VERSION is not None:
            assert envs.VLLM_FLASH_ATTN_VERSION in [2, 3]
            fa_version = envs.VLLM_FLASH_ATTN_VERSION

        # 3. fallback for unsupported combinations
        if device_capability.major == 10 and fa_version == 3:
            logger.warning_once(
                "Cannot use FA version 3 on Blackwell platform "
                "defaulting to FA version 2.")
            fa_version = 2

        if requires_alibi and fa_version == 3:
            logger.warning_once("Cannot use FA version 3 with ALiBi, "
                                "defaulting to FA version 2.")
            fa_version = 2

        if not is_fa_version_supported(fa_version):
            logger.error("Cannot use FA version %d is not supported due to %s",
                         fa_version, fa_version_unsupported_reason(fa_version))

        assert is_fa_version_supported(fa_version)
        return fa_version
    except (ImportError, AssertionError):
        return None


def flash_attn_supports_fp8() -> bool:
    from vllm.platforms import current_platform
    return get_flash_attn_version() == 3 and \
        current_platform.get_device_capability().major == 9
