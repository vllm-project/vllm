# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from vllm.config import VllmConfig
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionMetadataBuilder,
    MultipleOf,
)
from vllm.v1.attention.backends.mla.indexer import (
    DeepseekV32IndexerMetadataBuilder as _IndexerMetadataBuilderStub,
)


class DeepseekV4IndexerBackend(AttentionBackend):
    @staticmethod
    def get_name() -> str:
        return "DEEPSEEK_V4_INDEXER"

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [256]

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return [32, 64, 128]

    @staticmethod
    def get_builder_cls() -> type[AttentionMetadataBuilder]:
        return _IndexerMetadataBuilderStub

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        assert num_kv_heads == 1
        return (num_blocks, block_size, head_size)

    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        if include_num_layers_dimension:
            return (0, 1, 2, 3)
        return (0, 1, 2)


def get_max_prefill_buffer_size(vllm_config: VllmConfig) -> int:
    # 40 = (576 * 2 // 132) * 5: indexer prefill (132 B) fits inside the
    # FlashMLA-sparse workspace (576 * 2 B, 5 * max_model_len entries).
    return 40 * vllm_config.model_config.max_model_len
