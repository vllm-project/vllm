# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass, replace

import torch

from vllm.v1.attention.backend import AttentionBackend, CommonAttentionMetadata
from vllm.v1.attention.backends.mamba_attn import (
    BaseMambaAttentionMetadata,
    BaseMambaAttentionMetadataBuilder,
)


class Mamba1AttentionBackend(AttentionBackend):
    @staticmethod
    def get_name() -> str:
        return "MAMBA1_ATTN"

    @staticmethod
    def get_builder_cls() -> type["Mamba1AttentionMetadataBuilder"]:
        return Mamba1AttentionMetadataBuilder


@dataclass
class Mamba1AttentionMetadata(BaseMambaAttentionMetadata):
    # Chunk alignment: offset within the first block where processing starts
    # This is num_computed_tokens % block_size for each prefill request
    chunk_start_offsets_p: torch.Tensor | None = None


class Mamba1AttentionMetadataBuilder(
    BaseMambaAttentionMetadataBuilder[Mamba1AttentionMetadata]
):
    metadata_cls = Mamba1AttentionMetadata
    supports_update_block_table: bool = False

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> Mamba1AttentionMetadata:
        common = self._compute_common_metadata(common_attn_metadata)

        chunk_start_offsets_p = None

        if (
            common.num_prefills > 0
            and self.vllm_config.cache_config.mamba_cache_mode == "all"
            and common.num_computed_tokens_p is not None
        ):
            chunk_start_offsets_p = (
                common.num_computed_tokens_p % self.kv_cache_spec.block_size
            )

        return replace(common, chunk_start_offsets_p=chunk_start_offsets_p)
