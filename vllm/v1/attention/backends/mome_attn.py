# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.config import VllmConfig
from vllm.utils.math_utils import cdiv
from vllm.v1.attention.backend import AttentionBackend, AttentionMetadataBuilder
from vllm.v1.attention.backends.mamba_attn import (
    BaseMambaAttentionMetadata,
    BaseMambaAttentionMetadataBuilder,
)
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    SlidingWindowMomeSpec,
)


class MomeAttentionMetadata(BaseMambaAttentionMetadata):
    pass


class MomeAttentionMetadataBuilder(BaseMambaAttentionMetadataBuilder):
    metadata_cls = MomeAttentionMetadata

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        AttentionMetadataBuilder.__init__(
            self, kv_cache_spec, layer_names, vllm_config, device
        )

        # Enable speculative decoding support
        self.speculative_config = vllm_config.speculative_config
        self.compilation_config = vllm_config.compilation_config
        self.num_spec_tokens: int = vllm_config.num_speculative_tokens
        self.use_spec_decode = self.num_spec_tokens > 0

        # FIXME(runze): this is the only difference from the parent class.
        # Find a better way to share the parent implementation.
        assert isinstance(kv_cache_spec, SlidingWindowMomeSpec)
        scheduler_config = vllm_config.scheduler_config
        self.decode_cudagraph_max_bs: int = scheduler_config.max_num_seqs
        if self.compilation_config.max_cudagraph_capture_size is not None:
            self.decode_cudagraph_max_bs = min(
                self.decode_cudagraph_max_bs,
                self.compilation_config.max_cudagraph_capture_size,
            )

        if self.vllm_config.cache_config.mamba_cache_mode == "all":
            max_num_blocks = cdiv(
                self.vllm_config.model_config.max_model_len,
                kv_cache_spec.block_size,
            )
            # TODO: reduce this size as needed for decode-only cudagraph capture
            self.state_indices_tensor_d: torch.Tensor = torch.empty(
                (
                    self.decode_cudagraph_max_bs,
                    max_num_blocks,
                ),
                dtype=torch.int32,
                device=device,
            )
            self.block_idx_last_scheduled_token: torch.Tensor = torch.empty(
                (self.decode_cudagraph_max_bs,),
                dtype=torch.int32,
                device=device,
            )
            self.block_idx_last_computed_token: torch.Tensor = torch.empty(
                (self.decode_cudagraph_max_bs,),
                dtype=torch.int32,
                device=device,
            )
            if self.use_spec_decode:
                self.block_idx_last_scheduled_token_prev_step: torch.Tensor = (
                    torch.empty(
                        (self.decode_cudagraph_max_bs,),
                        dtype=torch.int32,
                        device=device,
                    )
                )
        else:
            self.state_indices_tensor_d = torch.empty(
                (self.decode_cudagraph_max_bs, 1 + self.num_spec_tokens),
                dtype=torch.int32,
                device=device,
            )

        # For speculative decoding, we need to store the following buffers
        # for CUDA graph capture during decode
        if self.num_spec_tokens > 0:
            self.decode_num_accepted_tokens: torch.Tensor = torch.empty(
                (self.decode_cudagraph_max_bs,),
                dtype=torch.int32,
                device=device,
            )

        self._init_reorder_batch_threshold(1, self.use_spec_decode)
        if self.use_spec_decode:
            self.supports_update_block_table = False


class MomeAttentionBackend(AttentionBackend):
    @staticmethod
    def get_builder_cls() -> type["MomeAttentionMetadataBuilder"]:
        return MomeAttentionMetadataBuilder
