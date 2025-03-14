from vllm import envs
from vllm.logger import init_logger

logger = init_logger(__name__)

def get_flash_attn_version():
    # import here to avoid circular dependencies
    from vllm.platforms import current_platform
    try:
        from vllm.vllm_flash_attn.flash_attn_interface import (
            fa_version_unsupported_reason, is_fa_version_supported)

        # if hopper default to FA3, otherwise stick to FA2 for now
        # TODO(lucas): profile FA3 on ampere to see if it makes sense to
        #  use FA3 as default for both
        if current_platform.get_device_capability()[0] == 9:
            fa_version = 3 if is_fa_version_supported(3) else 2
        else:
            fa_version = 2

        if envs.VLLM_FLASH_ATTN_VERSION is not None:
            assert envs.VLLM_FLASH_ATTN_VERSION in [2, 3]
            fa_version = envs.VLLM_FLASH_ATTN_VERSION
            if (current_platform.get_device_capability()[0] == 10
                    and envs.VLLM_FLASH_ATTN_VERSION == 3):
                logger.warning("Cannot use FA version 3 on Blackwell platform",
                               "defaulting to FA version 2.")
                fa_version = 2

        if not is_fa_version_supported(fa_version):
            logger.error("Cannot use FA version %d is not supported due to %s",
                         fa_version, fa_version_unsupported_reason(fa_version))

        assert is_fa_version_supported(fa_version)
        return fa_version
    except (ImportError, AssertionError):
        return None
