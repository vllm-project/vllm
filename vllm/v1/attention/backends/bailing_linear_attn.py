# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass

import torch

from vllm.config import CompilationConfig, VllmConfig
from vllm.v1.attention.backend import AttentionCGSupport, CommonAttentionMetadata
from vllm.v1.attention.backends.linear_attn import (
    LinearAttentionBackend,
    LinearAttentionMetadata,
    LinearAttentionMetadataBuilder,
)
from vllm.v1.attention.backends.utils import (
    PAD_SLOT_ID,
    mamba_get_block_table_tensor,
    split_decodes_and_prefills,
)
from vllm.v1.kv_cache_interface import AttentionSpec


class BailingLinearAttentionBackend(LinearAttentionBackend):
    @staticmethod
    def get_name() -> str:
        return "BAILING_LINEAR_ATTN"

    @staticmethod
    def get_builder_cls() -> type["BailingLinearAttentionMetadataBuilder"]:
        return BailingLinearAttentionMetadataBuilder


@dataclass
class BailingLinearAttentionMetadata(LinearAttentionMetadata):
    state_indices_tensor_d: torch.Tensor | None = None
    state_indices_tensor_p: torch.Tensor | None = None
    num_accepted_tokens: torch.Tensor | None = None
    query_start_loc_d: torch.Tensor | None = None


class BailingLinearAttentionMetadataBuilder(LinearAttentionMetadataBuilder):
    supports_spec_decode_metadata = True
    supports_update_block_table: bool = False

    @classmethod
    def get_cudagraph_support(
        cls,
        vllm_config: VllmConfig,
        kv_cache_spec: AttentionSpec,
    ) -> AttentionCGSupport:
        return AttentionCGSupport.UNIFORM_BATCH

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self.compilation_config: CompilationConfig = vllm_config.compilation_config
        self.num_spec_tokens: int = vllm_config.num_speculative_tokens
        self.use_spec_decode: bool = self.num_spec_tokens > 0
        self.decode_cudagraph_max_bs: int = vllm_config.scheduler_config.max_num_seqs
        if self.compilation_config.max_cudagraph_capture_size is not None:
            self.decode_cudagraph_max_bs = min(
                self.decode_cudagraph_max_bs,
                self.compilation_config.max_cudagraph_capture_size,
            )
        self.decode_state_indices_tensor: torch.Tensor = torch.empty(
            (self.decode_cudagraph_max_bs, 1 + self.num_spec_tokens),
            dtype=torch.int32,
            device=device,
        )
        self.decode_legacy_state_indices_tensor: torch.Tensor = torch.empty(
            (self.decode_cudagraph_max_bs,),
            dtype=torch.int32,
            device=device,
        )
        self.decode_query_start_loc: torch.Tensor = torch.empty(
            (self.decode_cudagraph_max_bs + 1,),
            dtype=torch.int32,
            device=device,
        )
        self.decode_num_accepted_tokens: torch.Tensor = torch.empty(
            (self.decode_cudagraph_max_bs,),
            dtype=torch.int32,
            device=device,
        )
        self._init_reorder_batch_threshold(1, self.use_spec_decode)

    def build_for_cudagraph_capture(
        self,
        common_attn_metadata: CommonAttentionMetadata,
    ) -> BailingLinearAttentionMetadata:
        num_accepted_tokens = None
        if self.use_spec_decode:
            assert common_attn_metadata.max_query_len <= 1 + self.num_spec_tokens, (
                "Bailing linear attention only supports speculative decoding "
                "with query length <= 1 + number of speculative tokens."
            )
            num_accepted_tokens = torch.diff(common_attn_metadata.query_start_loc)
        return self.build(
            common_prefix_len=0,
            common_attn_metadata=common_attn_metadata,
            num_accepted_tokens=num_accepted_tokens,
        )

    def build(  # type: ignore[override]
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
        *,
        num_accepted_tokens: torch.Tensor | None = None,
        num_decode_draft_tokens_cpu: torch.Tensor | None = None,
    ) -> BailingLinearAttentionMetadata:
        del common_prefix_len, fast_build, num_decode_draft_tokens_cpu
        query_start_loc = common_attn_metadata.query_start_loc
        seq_lens = common_attn_metadata.seq_lens
        num_reqs = common_attn_metadata.num_reqs
        use_spec_decode = self.use_spec_decode and num_accepted_tokens is not None

        state_indices_tensor = mamba_get_block_table_tensor(
            common_attn_metadata.block_table_tensor,
            common_attn_metadata.seq_lens,
            self.kv_cache_spec,
            self.vllm_config.cache_config.mamba_cache_mode,
        )
        if state_indices_tensor.dim() == 1:
            state_indices_tensor = state_indices_tensor.unsqueeze(-1)

        decode_threshold = self.reorder_batch_threshold if use_spec_decode else 1
        num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (
            split_decodes_and_prefills(
                common_attn_metadata,
                decode_threshold=decode_threshold,
            )
        )
        state_indices_tensor_d, state_indices_tensor_p = torch.split(
            state_indices_tensor,
            [num_decodes, num_prefills],
            dim=0,
        )
        state_indices_tensor_p = state_indices_tensor_p[:, 0]

        query_start_loc_d = None
        if use_spec_decode:
            assert num_accepted_tokens is not None
            state_indices_tensor_d = state_indices_tensor_d[
                :, : 1 + self.num_spec_tokens
            ]
            query_start_loc_d = query_start_loc[: num_decodes + 1]
            num_accepted_tokens = num_accepted_tokens[:num_decodes]
        else:
            state_indices_tensor_d = state_indices_tensor_d[:, 0]
            num_accepted_tokens = None

        legacy_state_indices_tensor = state_indices_tensor[:, 0]
        cudagraph_mode = self.compilation_config.cudagraph_mode
        use_full_cudagraph = (
            cudagraph_mode is not None and cudagraph_mode.has_full_cudagraphs()
        )
        if (
            num_prefills == 0
            and num_decodes <= self.decode_cudagraph_max_bs
            and use_full_cudagraph
        ):
            padded_bs = num_reqs
            is_padded_decode = seq_lens[:num_decodes] == 0
            if state_indices_tensor_d.dim() > 1:
                state_indices_tensor_d = torch.where(
                    is_padded_decode.unsqueeze(1),
                    torch.full_like(state_indices_tensor_d, PAD_SLOT_ID),
                    state_indices_tensor_d,
                )
                self.decode_state_indices_tensor[:num_decodes].copy_(
                    state_indices_tensor_d,
                    non_blocking=True,
                )
                state_indices_tensor_d = self.decode_state_indices_tensor[:padded_bs]
                state_indices_tensor_d[num_decodes:] = PAD_SLOT_ID
            else:
                state_indices_tensor_d = torch.where(
                    is_padded_decode,
                    torch.full_like(state_indices_tensor_d, PAD_SLOT_ID),
                    state_indices_tensor_d,
                )
                self.decode_legacy_state_indices_tensor[:num_decodes].copy_(
                    state_indices_tensor_d,
                    non_blocking=True,
                )
                state_indices_tensor_d = self.decode_legacy_state_indices_tensor[
                    :padded_bs
                ]
                state_indices_tensor_d[num_decodes:] = PAD_SLOT_ID

            self.decode_legacy_state_indices_tensor[:num_decodes].copy_(
                torch.where(
                    is_padded_decode,
                    torch.full_like(
                        legacy_state_indices_tensor[:num_decodes],
                        PAD_SLOT_ID,
                    ),
                    legacy_state_indices_tensor[:num_decodes],
                ),
                non_blocking=True,
            )
            legacy_state_indices_tensor = self.decode_legacy_state_indices_tensor[
                :padded_bs
            ]
            legacy_state_indices_tensor[num_decodes:] = PAD_SLOT_ID

            if use_spec_decode and num_accepted_tokens is not None:
                assert query_start_loc_d is not None
                self.decode_query_start_loc[: num_decodes + 1].copy_(
                    query_start_loc_d,
                    non_blocking=True,
                )
                decode_num_query_tokens = query_start_loc_d[-1]
                query_start_loc_d = self.decode_query_start_loc[: padded_bs + 1]
                query_start_loc_d[num_decodes + 1 :] = decode_num_query_tokens

                self.decode_num_accepted_tokens[:num_decodes].copy_(
                    num_accepted_tokens,
                    non_blocking=True,
                )
                num_accepted_tokens = self.decode_num_accepted_tokens[:padded_bs]
                num_accepted_tokens[num_decodes:] = 1

        return BailingLinearAttentionMetadata(
            num_prefills=num_prefills,
            num_prefill_tokens=num_prefill_tokens,
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
            query_start_loc=query_start_loc,
            seq_lens=seq_lens,
            state_indices_tensor=legacy_state_indices_tensor,
            state_indices_tensor_d=state_indices_tensor_d,
            state_indices_tensor_p=state_indices_tensor_p,
            num_accepted_tokens=num_accepted_tokens,
            query_start_loc_d=query_start_loc_d,
        )
