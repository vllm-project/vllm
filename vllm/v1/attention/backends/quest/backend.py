# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""QuestSparseOffloadBackend: vLLM v1 attention backend (Phase A skeleton).

Phase A registers and routes; it does not change attention semantics.
Forward is delegated to FlashAttentionImpl (see impl.py). Phase B will swap
the forward implementation in place — this file should not need changes
beyond updating supports_* flags.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import torch

from vllm.v1.attention.backend import AttentionBackend, AttentionType

if TYPE_CHECKING:
    from vllm.config.cache import CacheDType
    from vllm.v1.attention.backend import (
        AttentionImpl,
        AttentionMetadataBuilder,
    )


class QuestSparseOffloadBackend(AttentionBackend):
    """Sparse + KV-offload backend driven by Quest block selection.

    Phase A: identical behavior to FlashAttention.
    Phase B+: real sparse path (see implementation plan / spec).
    """

    supported_dtypes: ClassVar[list[torch.dtype]] = [
        torch.float16,
        torch.bfloat16,
    ]
    supported_kv_cache_dtypes: ClassVar[list["CacheDType"]] = [
        "auto",
        "float16",
        "bfloat16",
    ]

    @staticmethod
    def get_name() -> str:
        return "QUEST_SPARSE_OFFLOAD"

    @staticmethod
    def get_impl_cls() -> type["AttentionImpl"]:
        from vllm.v1.attention.backends.quest.impl import QuestSparseOffloadImpl

        return QuestSparseOffloadImpl

    @staticmethod
    def get_builder_cls() -> type["AttentionMetadataBuilder"]:
        from vllm.v1.attention.backends.quest.metadata import QuestMetadataBuilder

        return QuestMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        # Match FlashAttention's layout exactly so that delegation in Phase A
        # is binary-identical and Phase B can pass the same kv_cache tensor
        # to flash_attn_varlen_func with a custom block_table.
        if block_size % 16 != 0:
            raise ValueError("Block size must be a multiple of 16.")
        return (num_blocks, 2, block_size, num_kv_heads, head_size)

    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        from vllm.v1.attention.backends.flash_attn import FlashAttentionBackend

        return FlashAttentionBackend.get_kv_cache_stride_order(
            include_num_layers_dimension=include_num_layers_dimension
        )

    @classmethod
    def is_sparse(cls) -> bool:
        return True

    @classmethod
    def is_mla(cls) -> bool:
        return False

    @classmethod
    def supports_head_size(cls, head_size: int) -> bool:
        from vllm.v1.attention.backends.flash_attn import FlashAttentionBackend

        return FlashAttentionBackend.supports_head_size(head_size)

    @classmethod
    def supports_attn_type(cls, attn_type: str) -> bool:
        return attn_type == AttentionType.DECODER
