# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from math import lcm
from typing import TYPE_CHECKING

from vllm.attention.backends.abstract import MultipleOf
from vllm.attention.selector import get_attn_backend
from vllm.logger import init_logger
from vllm.model_executor.models.config import VerifyAndUpdateConfig

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_logger(__name__)


class AttentionConfig(VerifyAndUpdateConfig):
    @classmethod
    def verify_and_update_config(cls, vllm_config: "VllmConfig") -> None:
        """
        Align cache_config.block_size with attention backend's minimum
        supported kernel block size.

        Args:
            vllm_config: vLLM Config
        """
        model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config
        assert cache_config is not None

        backend_cls = get_attn_backend(
            head_size=model_config.get_head_size(),
            dtype=model_config.dtype,
            kv_cache_dtype=cache_config.cache_dtype,
            block_size=cache_config.block_size
            if cache_config.block_size is not None
            else 16,
        )

        supported_sizes = backend_cls.get_supported_kernel_block_size()
        supported_sizes = [
            s.base if isinstance(s, MultipleOf) else s for s in supported_sizes
        ]
        min_size = min(supported_sizes)
        if cache_config.block_size is None:
            new_block_size = min_size
        else:
            new_block_size = lcm(cache_config.block_size, min_size)

        if cache_config.block_size is None or new_block_size != cache_config.block_size:
            cache_config.block_size = new_block_size
            logger.info(
                "Setting attention block size to %d tokens "
                "to align with %s attention backend's supported kernel block sizes.",
                new_block_size,
                backend_cls.get_name(),
            )
