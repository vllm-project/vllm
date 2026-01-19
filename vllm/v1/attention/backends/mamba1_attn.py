# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass

from vllm.v1.attention.backend import AttentionBackend
from vllm.v1.attention.backends.mamba_attn import (
    BaseMambaAttentionMetadata,
    BaseMambaAttentionMetadataBuilder,
)


class Mamba1AttentionBackend(AttentionBackend):
    @staticmethod
    def get_builder_cls() -> type["Mamba1AttentionMetadataBuilder"]:
        return Mamba1AttentionMetadataBuilder


@dataclass
class Mamba1AttentionMetadata(BaseMambaAttentionMetadata):
    pass


class Mamba1AttentionMetadataBuilder(
    BaseMambaAttentionMetadataBuilder[Mamba1AttentionMetadata]
):
    metadata_cls = Mamba1AttentionMetadata
    supports_update_block_table: bool = False
