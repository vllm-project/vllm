# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import functools
from copy import copy

import numpy as np
import torch

from vllm.config import CacheConfig, VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.attention import Attention
from vllm.utils.math_utils import cdiv
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionMetadata,
    AttentionType,
    CommonAttentionMetadata,
    subclass_attention_backend_with_overrides,
)
from vllm.v1.attention.selector import get_attn_backend
from vllm.v1.kv_cache_interface import CrossAttentionSpec, KVCacheSpec

logger = init_logger(__name__)


def _get_cross_slot_mapping(
    encoder_seq_lens: np.ndarray,
    block_table_tensor: torch.Tensor,
    kv_cache_spec: CrossAttentionSpec,
    device: torch.device,
) -> torch.Tensor:
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
        i_values = torch.arange(encoder_seq_len, dtype=torch.int64, device=device)
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
    underlying_attn_backend: AttentionBackend,
) -> type[AttentionBackend]:
    prefix = "CrossAttention_"
    underlying_builder = underlying_attn_backend.get_builder_cls()
    underlying_impl = underlying_attn_backend.get_impl_cls()

    class CrossAttentionBuilder(underlying_builder):  # type: ignore
        def build(
            self,
            common_prefix_len: int,
            common_attn_metadata: CommonAttentionMetadata,
            fast_build: bool = False,
        ) -> AttentionMetadata:
            new_metadata = copy(common_attn_metadata)
            new_metadata.causal = False
            max_encoder_len = int(new_metadata.encoder_seq_lens_cpu.max())
            new_metadata.max_seq_len = max_encoder_len
            # Any computed tokens indicated decode step>1 (no chunked prefill)
            num_cache_decodes = (
                (common_attn_metadata.num_computed_tokens_cpu > 0).sum().item()
            )
            if num_cache_decodes > 0:
                # CrossAttn KV cache has already been populated on first decoder step,
                # skip slot_mapping calculation for requests that do not need
                # reshape_and_cache.
                num_tokens = common_attn_metadata.num_computed_tokens_cpu.numpy()
                new_metadata.encoder_seq_lens_cpu = np.where(
                    num_tokens > 0, 0, new_metadata.encoder_seq_lens_cpu
                )

            # seq_lens is provided by model runner: initial encoder input length is
            # needed here to know how many tokens to attend to from the cached
            # cross-attention KV cache.
            new_metadata.seq_lens = common_attn_metadata.encoder_seq_lens
            new_metadata._seq_lens_cpu = torch.from_numpy(
                common_attn_metadata.encoder_seq_lens_cpu
            )

            # NOTE (NickLucche) use `new_metadata` instead of `common_*` (initial) here
            slot_mapping = _get_cross_slot_mapping(
                new_metadata.encoder_seq_lens_cpu,
                new_metadata.block_table_tensor,
                self.kv_cache_spec,
                self.device,
            )
            attn_metadata = super().build(common_prefix_len, new_metadata, fast_build)
            attn_metadata.slot_mapping = slot_mapping
            return attn_metadata

    # NOTE(Lucas): we need a custom impl so we can use the slot-mapping computed by
    # `CrossAttentionBuilder` instead of the one computed by `BlockTable`
    # (gpu_model_runner)
    class CrossAttentionImpl(underlying_impl):  # type: ignore[valid-type,misc]
        def forward(
            self,
            layer: torch.nn.Module,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            kv_cache: torch.Tensor,
            attn_metadata: AttentionMetadata,
            output: torch.Tensor | None = None,
            output_scale: torch.Tensor | None = None,
            output_block_scale: torch.Tensor | None = None,
        ) -> torch.Tensor:
            if (
                not underlying_attn_backend.forward_includes_kv_cache_update
                and attn_metadata is not None
            ):
                self.do_kv_cache_update(
                    layer, key, value, kv_cache, attn_metadata.slot_mapping
                )

            return super().forward(
                layer,
                query,
                key,
                value,
                kv_cache,
                attn_metadata,
                output,
                output_scale,
                output_block_scale,
            )

    attn_backend = subclass_attention_backend_with_overrides(
        name_prefix=prefix,
        attention_backend_cls=underlying_attn_backend,
        overrides={
            "get_builder_cls": lambda: CrossAttentionBuilder,
            "get_impl_cls": lambda: CrossAttentionImpl,
            "forward_includes_kv_cache_update": True,
        },
    )

    return attn_backend


class CrossAttention(Attention):
    """
    Cross-attention for encoder-decoder models.
    Handles attention between decoder queries and encoder keys/values.
    """

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        cache_config: CacheConfig | None = None,
        attn_type: str | None = None,
        **kwargs,
    ):
        dtype = torch.get_default_dtype()

        if cache_config is not None:
            kv_cache_dtype = cache_config.cache_dtype
            block_size = cache_config.block_size
        else:
            kv_cache_dtype = "auto"
            block_size = 16

        if attn_type is not None:
            assert attn_type == AttentionType.ENCODER_DECODER, (
                "CrossAttention only supports AttentionType.ENCODER_DECODER"
            )

        underlying_attn_backend = get_attn_backend(
            head_size,
            dtype,
            kv_cache_dtype,
            block_size,
            attn_type=AttentionType.ENCODER_DECODER,
        )
        attn_backend = create_cross_attention_backend(underlying_attn_backend)

        super().__init__(
            num_heads=num_heads,
            head_size=head_size,
            scale=scale,
            cache_config=cache_config,
            attn_backend=attn_backend,
            attn_type=AttentionType.ENCODER_DECODER,
            **kwargs,
        )

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        return CrossAttentionSpec(
            block_size=vllm_config.cache_config.block_size,
            num_kv_heads=self.num_kv_heads,
            head_size=self.head_size,
            dtype=self.kv_cache_torch_dtype,
        )
