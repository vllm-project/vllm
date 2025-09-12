# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import functools
from copy import copy
from typing import Optional

import numpy as np
import torch

from vllm import envs
from vllm.attention.backends.abstract import (AttentionBackend,
                                              AttentionMetadata, AttentionType)
from vllm.attention.layer import Attention
from vllm.attention.selector import get_attn_backend
from vllm.config import CacheConfig, VllmConfig
from vllm.logger import init_logger
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.utils import cdiv
from vllm.v1.attention.backends.utils import (CommonAttentionMetadata,
                                              subclass_attention_backend)
from vllm.v1.kv_cache_interface import CrossAttentionSpec

logger = init_logger(__name__)


def _get_max_encoder_len(vllm_config: VllmConfig) -> int:
    return MULTIMODAL_REGISTRY.get_encdec_max_encoder_len(
        vllm_config.model_config)


def _get_cross_slot_mapping(encoder_seq_lens: np.ndarray,
                            block_table_tensor: torch.Tensor,
                            kv_cache_spec: CrossAttentionSpec,
                            device: torch.device) -> torch.Tensor:
    """Get cross-attention slot mappings."""

    block_size = kv_cache_spec.block_size
    slot_mappings = []

    # Find indices with non-zero encoder sequence lengths
    # The majority of parallel requests will be running the
    # decoder, so this list should be relatively small.
    active_indices = np.nonzero(encoder_seq_lens)[0]

    for req_index in active_indices:
        encoder_seq_len = encoder_seq_lens[req_index].item()

        # Calculate the number of blocks needed for this request
        num_blocks_needed = cdiv(encoder_seq_len, block_size)

        # Get the block IDs for this request from the tensor
        req_block_ids = block_table_tensor[req_index]

        # Get only the blocks we need (first num_blocks_needed blocks)
        needed_block_ids = req_block_ids[:num_blocks_needed]

        # All needed blocks are allocated
        i_values = torch.arange(encoder_seq_len,
                                dtype=torch.int64,
                                device=device)
        block_indices = i_values // block_size
        block_offsets = i_values % block_size
        block_numbers = needed_block_ids[block_indices]
        slot_mapping = block_numbers * block_size + block_offsets

        slot_mappings.append(slot_mapping)

    if slot_mappings:
        return torch.cat(slot_mappings)
    else:
        return torch.empty(0, dtype=torch.int64, device=device)


@functools.lru_cache
def create_cross_attention_backend(
    underlying_attn_backend: AttentionBackend, ) -> type[AttentionBackend]:
    prefix = "CrossAttention_"
    underlying_builder = underlying_attn_backend.get_builder_cls()

    class CrossAttentionBuilder(underlying_builder):  # type: ignore

        def build(self,
                  common_prefix_len: int,
                  common_attn_metadata: CommonAttentionMetadata,
                  fast_build: bool = False) -> AttentionMetadata:
            new_metadata = copy(common_attn_metadata)
            new_metadata.causal = False
            max_encoder_len = _get_max_encoder_len(self.vllm_config)
            new_metadata.max_seq_len = max_encoder_len

            new_metadata.seq_lens = torch.full(
                (new_metadata.num_reqs, ),
                max_encoder_len,
                dtype=torch.int32,
                device=self.device,
            )
            new_metadata.seq_lens_cpu = torch.full(
                (new_metadata.num_reqs, ),
                max_encoder_len,
                dtype=torch.int32,
                device="cpu",
            )
            new_metadata.slot_mapping = _get_cross_slot_mapping(
                new_metadata.encoder_seq_lens, new_metadata.block_table_tensor,
                self.kv_cache_spec, self.device)
            return super().build(common_prefix_len, new_metadata, fast_build)

    attn_backend = subclass_attention_backend(
        name_prefix=prefix,
        attention_backend_cls=underlying_attn_backend,
        builder_cls=CrossAttentionBuilder)

    return attn_backend


class CrossAttention(Attention):
    """
    Cross-attention for encoder-decoder models.
    Handles attention between decoder queries and encoder keys/values.
    """

    def __init__(self,
                 num_heads: int,
                 head_size: int,
                 scale: float,
                 cache_config: Optional[CacheConfig] = None,
                 attn_type: Optional[str] = None,
                 **kwargs):
        dtype = torch.get_default_dtype()

        if cache_config is not None:
            kv_cache_dtype = cache_config.cache_dtype
            block_size = cache_config.block_size
        else:
            kv_cache_dtype = "auto"
            block_size = 16

        if envs.VLLM_USE_V1:
            underlying_attn_backend = get_attn_backend(head_size, dtype,
                                                       kv_cache_dtype,
                                                       block_size)

            attn_backend = create_cross_attention_backend(
                underlying_attn_backend)
        else:
            # in v0 cross attention is handled inside the backends
            attn_backend = None

        if attn_type is not None:
            assert attn_type == AttentionType.ENCODER_DECODER, (
                "CrossAttention only supports AttentionType.ENCODER_DECODER")

        super().__init__(num_heads=num_heads,
                         head_size=head_size,
                         scale=scale,
                         cache_config=cache_config,
                         attn_backend=attn_backend,
                         attn_type=AttentionType.ENCODER_DECODER,
                         **kwargs)
