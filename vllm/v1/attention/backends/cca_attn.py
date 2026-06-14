# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass

from vllm.v1.attention.backend import AttentionBackend
from vllm.v1.attention.backends.mamba_attn import (
    BaseMambaAttentionMetadata,
    BaseMambaAttentionMetadataBuilder,
)


class CCAAttentionBackend(AttentionBackend):
    @staticmethod
    def get_name() -> str:
        return "CCA_ATTN"

    @staticmethod
    def get_builder_cls() -> type["CCAAttentionMetadataBuilder"]:
        return CCAAttentionMetadataBuilder


@dataclass
class CCAAttentionMetadata(BaseMambaAttentionMetadata):
    pass


class CCAAttentionMetadataBuilder(
    BaseMambaAttentionMetadataBuilder[CCAAttentionMetadata]
):
    metadata_cls = CCAAttentionMetadata
    supports_update_block_table: bool = False
