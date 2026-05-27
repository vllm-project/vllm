# SPDX-License-Identifier: Apache-2.0
"""Static-method tests for QuestSparseOffloadBackend."""
from __future__ import annotations

import torch


def test_get_name():
    from vllm.v1.attention.backends.quest.backend import (
        QuestSparseOffloadBackend,
    )

    assert QuestSparseOffloadBackend.get_name() == "QUEST_SPARSE_OFFLOAD"


def test_get_impl_cls_returns_quest_impl():
    from vllm.v1.attention.backends.quest.backend import (
        QuestSparseOffloadBackend,
    )
    from vllm.v1.attention.backends.quest.impl import QuestSparseOffloadImpl

    assert QuestSparseOffloadBackend.get_impl_cls() is QuestSparseOffloadImpl


def test_get_builder_cls_returns_quest_builder():
    from vllm.v1.attention.backends.quest.backend import (
        QuestSparseOffloadBackend,
    )
    from vllm.v1.attention.backends.quest.metadata import QuestMetadataBuilder

    assert QuestSparseOffloadBackend.get_builder_cls() is QuestMetadataBuilder


def test_kv_cache_shape_matches_flash_attn():
    from vllm.v1.attention.backends.flash_attn import FlashAttentionBackend
    from vllm.v1.attention.backends.quest.backend import (
        QuestSparseOffloadBackend,
    )

    args = (128, 16, 8, 128)  # num_blocks, block_size, num_kv_heads, head_size
    fa_shape = FlashAttentionBackend.get_kv_cache_shape(*args)
    q_shape = QuestSparseOffloadBackend.get_kv_cache_shape(*args)
    assert fa_shape == q_shape, f"FA={fa_shape} Quest={q_shape}"


def test_is_sparse():
    from vllm.v1.attention.backends.quest.backend import (
        QuestSparseOffloadBackend,
    )

    assert QuestSparseOffloadBackend.is_sparse() is True


def test_is_mla_false():
    from vllm.v1.attention.backends.quest.backend import (
        QuestSparseOffloadBackend,
    )

    assert QuestSparseOffloadBackend.is_mla() is False


def test_supports_dtype_fp16_bf16():
    from vllm.v1.attention.backends.quest.backend import (
        QuestSparseOffloadBackend,
    )

    assert QuestSparseOffloadBackend.supports_dtype(torch.float16)
    assert QuestSparseOffloadBackend.supports_dtype(torch.bfloat16)
    assert not QuestSparseOffloadBackend.supports_dtype(torch.float32)
