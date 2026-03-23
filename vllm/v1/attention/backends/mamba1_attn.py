# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass, replace
from typing import Any

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
    pass


class Mamba1AttentionMetadataBuilder(
    BaseMambaAttentionMetadataBuilder[Mamba1AttentionMetadata]
):
    metadata_cls = Mamba1AttentionMetadata

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
        **kwargs: Any,
    ) -> Mamba1AttentionMetadata:
        common = self._compute_common_metadata(
            common_attn_metadata,
            num_accepted_tokens=kwargs.get("num_accepted_tokens"),
        )

        # selective_scan_fwd (C/CUDA kernel) receives cache_indices as a raw
        # pointer and ignores tensor strides.  With spec decode the block table
        # has 1+num_spec_tokens columns, so [:, 0] in _compute_common_metadata
        # produces a strided (non-contiguous) 1D view that the kernel would
        # misread.  Mamba2 kernels handle strides correctly and don't need this.
        if self.use_spec_decode and common.state_indices_tensor_p is not None:
            common = replace(
                common,
                state_indices_tensor_p=common.state_indices_tensor_p.contiguous(),
            )

        if (
            common.num_prefills > 0
            and self.vllm_config.cache_config.mamba_cache_mode == "all"
        ):
            cu_chunk_seqlen_p, _, last_chunk_indices_p = (
                self._build_chunk_metadata_tensors(
                    self.kv_cache_spec.block_size,
                    common,
                    common_attn_metadata,
                )
            )
            return replace(
                common,
                cu_chunk_seqlen_p=cu_chunk_seqlen_p,
                last_chunk_indices_p=last_chunk_indices_p,
            )

        return common
