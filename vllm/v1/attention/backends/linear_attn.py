# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass

import torch

from vllm.config import VllmConfig
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


@dataclass
class LinearAttentionMetadata:
    num_prefills: int
    num_prefill_tokens: int
    num_decodes: int
    num_decode_tokens: int
    query_start_loc: torch.Tensor
    seq_lens: torch.Tensor

    state_indices_tensor: torch.Tensor  # shape: [batch,]


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

        # Pre-allocate buffer for CUDA graph capture
        # This ensures consistent tensor addresses during capture and replay
        self.compilation_config = vllm_config.compilation_config
        self.decode_cudagraph_max_bs = min(
            vllm_config.scheduler_config.max_num_seqs,
            self.compilation_config.max_cudagraph_capture_size,
        )
        self._state_indices_tensor = torch.empty(
            (self.decode_cudagraph_max_bs,),
            dtype=torch.int32,
            device=device,
        )

    def build_for_cudagraph_capture(
        self, common_attn_metadata: CommonAttentionMetadata
    ) -> LinearAttentionMetadata:
        """
        Build metadata for full cudagraph capture.
        Only decode is supported for full cudagraphs with linear attention.
        """
        m = common_attn_metadata

        assert m.num_reqs == m.num_actual_tokens, (
            "Linear attention only supports decode-only full CUDAGraph capture. "
            "Make sure all cudagraph capture sizes <= max_num_seq."
        )

        m.max_query_len = 1  # decode-only

        return self.build(0, m)

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> LinearAttentionMetadata:
        query_start_loc = common_attn_metadata.query_start_loc
        seq_lens = common_attn_metadata.seq_lens

        state_indices_tensor = mamba_get_block_table_tensor(
            common_attn_metadata.block_table_tensor,
            common_attn_metadata.seq_lens,
            self.kv_cache_spec,
            self.vllm_config.cache_config.mamba_cache_mode,
        )[:, 0]

        num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (
            split_decodes_and_prefills(
                common_attn_metadata, decode_threshold=self.reorder_batch_threshold
            )
        )

        # For decode-only CUDA graph capture, copy to pre-allocated buffer
        # to ensure consistent tensor addresses during capture and replay.
        # Only do this for decode-only batches (no prefills).
        # Use num_actual_tokens which is the PADDED batch size for CUDA graphs.
        padded_batch_size = common_attn_metadata.num_actual_tokens
        if (
            num_prefills == 0
            and num_decodes <= self.decode_cudagraph_max_bs
            and self.compilation_config.cudagraph_mode.has_full_cudagraphs()
        ):
            self._state_indices_tensor[:num_decodes].copy_(
                state_indices_tensor, non_blocking=True
            )
            # Pad unused slots with 0 (a valid index) to avoid out-of-bounds
            # access during CUDA graph replay. During replay, padded tokens
            # may access state at index 0, but their outputs are masked.
            if num_decodes < self.decode_cudagraph_max_bs:
                self._state_indices_tensor[num_decodes:].fill_(0)
            # Return slice matching the PADDED batch size for CUDA graph
            # shape consistency during capture and replay.
            state_indices_tensor = self._state_indices_tensor[:padded_batch_size]

        attn_metadata = LinearAttentionMetadata(
            num_prefills=num_prefills,
            num_prefill_tokens=num_prefill_tokens,
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
            query_start_loc=query_start_loc,
            seq_lens=seq_lens,
            state_indices_tensor=state_indices_tensor,
        )
        return attn_metadata
