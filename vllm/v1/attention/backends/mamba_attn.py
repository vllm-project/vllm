# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import abc
from typing import ClassVar, TypeVar

import torch

from vllm.config import VllmConfig
from vllm.utils import cdiv
from vllm.v1.attention.backends.utils import (
    AttentionCGSupport,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
)
from vllm.v1.kv_cache_interface import AttentionSpec, MambaSpec

M = TypeVar("M")


class BaseMambaAttentionMetadataBuilder(AttentionMetadataBuilder[M], abc.ABC):
    reorder_batch_threshold: int = 1
    cudagraph_support: ClassVar[AttentionCGSupport] = (
        AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE
    )

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)

        assert isinstance(kv_cache_spec, MambaSpec)
        self.compilation_config = vllm_config.compilation_config
        self.decode_cudagraph_max_bs = min(
            self.vllm_config.scheduler_config.max_num_seqs,
            self.compilation_config.max_capture_size,
        )

        if self.vllm_config.cache_config.enable_prefix_caching:
            self.state_indices_tensor = torch.empty(
                (
                    self.decode_cudagraph_max_bs,
                    cdiv(
                        self.vllm_config.model_config.max_model_len,
                        self.kv_cache_spec.block_size,
                    ),
                ),
                dtype=torch.int32,
                device=device,
            )
            self.block_idx_last_scheduled_token = torch.empty(
                (self.decode_cudagraph_max_bs,),
                dtype=torch.int32,
                device=device,
            )
            self.block_idx_last_computed_token = torch.empty(
                (self.decode_cudagraph_max_bs,),
                dtype=torch.int32,
                device=device,
            )
        else:
            self.state_indices_tensor = torch.empty(
                (self.decode_cudagraph_max_bs,),
                dtype=torch.int32,
                device=device,
            )

    def build_for_cudagraph_capture(
        self, common_attn_metadata: CommonAttentionMetadata
    ) -> M:
        """
        This method builds the metadata for full cudagraph capture.
        Currently, only decode is supported for full cudagraphs with Mamba.
        """
        m = common_attn_metadata

        assert m.num_reqs == m.num_actual_tokens, (
            "Mamba only supports decode-only full CUDAGraph capture. "
            "Make sure all cudagraph capture sizes <= max_num_seq."
        )

        m.max_query_len = 1  # decode-only

        return self.build(0, m)
