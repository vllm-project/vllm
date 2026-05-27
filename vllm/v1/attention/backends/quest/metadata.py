# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Phase A: alias FlashAttention's metadata + builder.

Phase B will replace QuestAttentionMetadata with a subclass that adds
sparse_block_table / residency fields.
"""
from __future__ import annotations

from vllm.v1.attention.backends.flash_attn import (
    FlashAttentionMetadata,
    FlashAttentionMetadataBuilder,
)

# Phase A: identity alias. Phase B will subclass and add fields.
QuestAttentionMetadata = FlashAttentionMetadata
QuestMetadataBuilder = FlashAttentionMetadataBuilder
