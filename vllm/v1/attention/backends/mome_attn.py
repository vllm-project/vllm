# SPDX-License-Identifier: Apache-2.0


from vllm.v1.attention.backend import AttentionBackend
from vllm.v1.attention.backends.mamba_attn import (
    BaseMambaAttentionMetadata,
    BaseMambaAttentionMetadataBuilder,
)


class MomeAttentionMetadata(BaseMambaAttentionMetadata):
    pass


class MomeAttentionMetadataBuilder(BaseMambaAttentionMetadataBuilder):
    metadata_cls = MomeAttentionMetadata


class MomeAttentionBackend(AttentionBackend):
    @staticmethod
    def get_builder_cls() -> type["MomeAttentionMetadataBuilder"]:
        return MomeAttentionMetadataBuilder
