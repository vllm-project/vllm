# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import TYPE_CHECKING

from vllm.logger import init_logger
from vllm.model_executor.models.config import VerifyAndUpdateConfig
from vllm.v1.attention.backend import MultipleOf

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_logger(__name__)


class AttentionVerifyAndUpdateConfig(VerifyAndUpdateConfig):
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
        parallel_config = vllm_config.parallel_config
        assert cache_config is not None
        from vllm.config import set_current_vllm_config
        from vllm.v1.attention.selector import (
            _cached_get_attn_backend,
            get_attn_backend,
        )

        # NOTE: Notice that this backend may not be exactly same with that of
        # runtime, cause we doesn't get the exact information of sinks,
        # multi-modality prefix, and so on. Thus a clear
        # of cache is required after this.
        with set_current_vllm_config(vllm_config):
            num_heads = model_config.get_num_attention_heads(parallel_config)
            backend_cls = get_attn_backend(
                head_size=model_config.get_head_size(),
                dtype=model_config.dtype,
                kv_cache_dtype=cache_config.cache_dtype,
                use_mla=model_config.use_mla,
                use_sparse=hasattr(model_config.hf_config, "index_topk"),
                num_heads=num_heads,
            )
            ori_supported_kernel_block_sizes = (
                backend_cls.get_supported_kernel_block_sizes()
            )
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
        elif cache_config.user_specified_block_size:
            # Only validate block_size when the user explicitly set it via
            # --block-size. Otherwise block_size is still the default and will
            # be overwritten later by Platform.update_block_size_for_backend().
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
