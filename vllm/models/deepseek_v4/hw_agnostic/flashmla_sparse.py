# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek V4 sparse MLA backend re-export for the hw_agnostic path.

The canonical FlashMLASparseBackend / FlashMLASparseMetadata live in
vllm.v1.attention.backends.mla.flashmla_sparse and already carry the V4
metadata fields (c128a_*). All this module needs to add is the V4 subclass
that overrides head_size / kv-cache layout / kernel block size — mirroring
vllm/models/deepseek_v4/nvidia/flashmla.py. The Impl in the canonical file
calls FlashMLA kernels directly, so on non-CUDA backends instantiating the
attention layer is the point at which we will need a different attention
formulation; the subclass here exists so the registry symbol resolves.
"""

from vllm.v1.attention.backend import MultipleOf
from vllm.v1.attention.backends.mla.flashmla_sparse import (
    FlashMLASparseBackend,
    FlashMLASparseMetadata,
)

__all__ = [
    "DeepseekV4FlashMLASparseBackend",
    "FlashMLASparseBackend",
    "FlashMLASparseMetadata",
]


class DeepseekV4FlashMLASparseBackend(FlashMLASparseBackend):
    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [256]

    @staticmethod
    def get_name() -> str:
        return "FLASHMLA_SPARSE_DSV4_HW_AGNOSTIC"

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return [512]

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        if cache_dtype_str == "fp8_ds_mla":
            return (num_blocks, block_size, 584)
        return (num_blocks, block_size, head_size)
