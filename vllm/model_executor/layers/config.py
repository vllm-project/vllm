# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import TYPE_CHECKING

from vllm.logger import init_logger
from vllm.model_executor.models.config import VerifyAndUpdateConfig
from vllm.v1.attention.backend import MultipleOf

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_logger(__name__)


class AttentionConfig(VerifyAndUpdateConfig):
    @classmethod
    def verify_and_update_config(cls, vllm_config: "VllmConfig") -> None:
        """
        Set `cache_config.mamba_alignment_kernel_block_size` to attention
        backend's minimum supported kernel block size, with which
        cache_config.block_size will align.

        Args:
            vllm_config: vLLM Config
        """
        cache_config = vllm_config.cache_config
        model_config = vllm_config.model_config
        attention_config = vllm_config.attention_config
        assert cache_config is not None
        from vllm.v1.attention.selector import (
            AttentionSelectorConfig,
            _cached_get_attn_backend,
        )

        # NOTE: To avoid getting vllm_config, which will raise error outside
        # the `set_current_vllm_config`, we use _cached_get_attn_backend
        # instead of get_attn_backend to get the current attention backend.
        # Notice that this backend may not be exactly same with that of
        # runtime, cause we doesn't get the exact information of sinks,
        # sparse attention, multi-modality prefix, and so on. Thus a clear
        # of cache is required after this.
        attn_selector_config = AttentionSelectorConfig(
            head_size=model_config.get_head_size(),
            dtype=model_config.dtype,
            kv_cache_dtype=cache_config.cache_dtype,
            block_size=cache_config.block_size
            if cache_config.block_size is not None
            else 16,
        )
        backend_cls = _cached_get_attn_backend(
            backend=attention_config.backend,
            attn_selector_config=attn_selector_config,
        )
        ori_supported_kernel_block_sizes = (
            backend_cls.get_supported_kernel_block_sizes()
        )
        # clear the cache of attention backend to avoid influcing the
        # attention backend selection during inference runtime
        _cached_get_attn_backend.cache_clear()
        supported_kernel_block_sizes = [
            kernel_block_size.base
            if isinstance(kernel_block_size, MultipleOf)
            else kernel_block_size
            for kernel_block_size in ori_supported_kernel_block_sizes
        ]

        kernel_block_alignment_size = min(supported_kernel_block_sizes)
        cache_config.mamba_alignment_kernel_block_size = kernel_block_alignment_size
        if cache_config.block_size is None:
            cache_config.block_size = kernel_block_alignment_size
        else:
            is_valid_block_size = any(
                (cache_config.block_size % kernel_block_size.base == 0)
                if isinstance(kernel_block_size, MultipleOf)
                else (cache_config.block_size == kernel_block_size)
                for kernel_block_size in ori_supported_kernel_block_sizes
            )
            if not is_valid_block_size:
                raise ValueError(
                    f"Unexpected block_size {cache_config.block_size} for current"
                    f" attention backend {attention_config.backend},"
                    f" the supported block_sizes of {attention_config.backend}:"
                    f" {supported_kernel_block_sizes}"
                )
