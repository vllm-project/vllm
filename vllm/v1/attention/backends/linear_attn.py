# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass

import torch

from vllm.config import VllmConfig
from vllm.utils.math_utils import cdiv
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
)
from vllm.v1.attention.backends.utils import (
    mamba_get_block_table_tensor,
    split_decodes_and_prefills,
)
from vllm.v1.kv_cache_interface import AttentionSpec, MambaSpec


class LinearAttentionBackend(AttentionBackend):
    @staticmethod
    def get_name() -> str:
        return "LINEAR_ATTN"

    @staticmethod
    def get_builder_cls() -> type["LinearAttentionMetadataBuilder"]:
        return LinearAttentionMetadataBuilder

    @classmethod
    def is_ssm(cls) -> bool:
        return True


@dataclass
class LinearAttentionMetadata:
    num_prefills: int
    num_prefill_tokens: int
    num_decodes: int
    num_decode_tokens: int
    query_start_loc: torch.Tensor
    seq_lens: torch.Tensor

    state_indices_tensor: torch.Tensor  # shape: [batch] or [batch, max_blocks]
    num_computed_tokens: torch.Tensor | None = None
    block_idx_last_computed_token: torch.Tensor | None = None
    block_idx_first_scheduled_token: torch.Tensor | None = None
    block_idx_last_scheduled_token: torch.Tensor | None = None


class LinearAttentionMetadataBuilder(AttentionMetadataBuilder[LinearAttentionMetadata]):
    reorder_batch_threshold: int = 1

    _cudagraph_support = AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        assert isinstance(kv_cache_spec, MambaSpec)

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> LinearAttentionMetadata:
        del common_prefix_len, fast_build
        query_start_loc = common_attn_metadata.query_start_loc
        seq_lens = common_attn_metadata.seq_lens
        cache_mode = self.vllm_config.cache_config.mamba_cache_mode

        num_computed_tokens = None
        block_idx_last_computed_token = None
        block_idx_first_scheduled_token = None
        block_idx_last_scheduled_token = None

        if cache_mode == "all":
            num_computed_tokens = common_attn_metadata.compute_num_computed_tokens()
            state_indices_tensor = common_attn_metadata.block_table_tensor
            mamba_block_size = self.kv_cache_spec.block_size
            block_idx_last_computed_token = (
                cdiv(num_computed_tokens, mamba_block_size) - 1
            )
            block_idx_first_scheduled_token = (
                cdiv(num_computed_tokens + 1, mamba_block_size) - 1
            )
            block_idx_last_scheduled_token = (
                cdiv(common_attn_metadata.seq_lens, mamba_block_size) - 1
            )
            block_idx_last_computed_token = torch.clamp(
                block_idx_last_computed_token,
                min=0,
            )
            block_idx_last_scheduled_token = torch.clamp(
                block_idx_last_scheduled_token,
                min=0,
            )
        else:
            state_indices_tensor = mamba_get_block_table_tensor(
                common_attn_metadata.block_table_tensor,
                common_attn_metadata.seq_lens,
                self.kv_cache_spec,
                cache_mode,
            )[:, 0]

        num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (
            split_decodes_and_prefills(
                common_attn_metadata, decode_threshold=self.reorder_batch_threshold
            )
        )

        attn_metadata = LinearAttentionMetadata(
            num_prefills=num_prefills,
            num_prefill_tokens=num_prefill_tokens,
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
            query_start_loc=query_start_loc,
            seq_lens=seq_lens,
            state_indices_tensor=state_indices_tensor,
            num_computed_tokens=num_computed_tokens,
            block_idx_last_computed_token=block_idx_last_computed_token,
            block_idx_first_scheduled_token=block_idx_first_scheduled_token,
            block_idx_last_scheduled_token=block_idx_last_scheduled_token,
        )
        return attn_metadata
