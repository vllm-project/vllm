# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass

from vllm.v1.attention.backend import AttentionBackend
from vllm.v1.attention.backends.mamba_attn import (
    BaseMambaAttentionMetadata,
    BaseMambaAttentionMetadataBuilder,
)


class ShortConvAttentionBackend(AttentionBackend):
    @staticmethod
    def get_name() -> str:
        return "SHORT_CONV_ATTN"

    @staticmethod
    def get_builder_cls() -> type["ShortConvAttentionMetadataBuilder"]:
        return ShortConvAttentionMetadataBuilder


@dataclass(slots=True)
class ShortConvAttentionMetadata(BaseMambaAttentionMetadata):
    pass


class ShortConvAttentionMetadataBuilder(
    BaseMambaAttentionMetadataBuilder[ShortConvAttentionMetadata]
):
    metadata_cls = ShortConvAttentionMetadata
