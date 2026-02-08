# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Hidden States Extractor Model.

This model extracts and caches hidden states from the target model
without performing actual token generation. It's used with the
extract_hidden_states speculative decoding method.
"""

import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.models.utils import maybe_prefix
from vllm.v1.attention.backend import AttentionType
from vllm.v1.attention.backends.registry import AttentionBackendEnum, register_backend
from vllm.v1.attention.backends.triton_attn import (
    TritonAttentionBackend,
    TritonAttentionImpl,
    TritonAttentionMetadata,
)

logger = init_logger(__name__)


def reshape_hidden_states_for_kv_cache(
    hidden_states: torch.Tensor, num_heads: int, head_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reshape hidden states into key and value tensors for KV cache.

    Args:
        hidden_states: shape [batch_size, 2 * num_heads * head_size ]
            where num_heads = (original_num_heads // 2) * num_hidden_layers
            e.g. hidden_states = torch.cat([h_1, h_2, ..., h_n], dim=1)
        num_heads: Number of attention heads
        head_size: Size of each attention head

    Returns:
        key, value: each of shape [batch_size, num_heads, head_size]
    """

    batch_size = hidden_states.shape[0]
    # Split into two equal parts for key and value
    split_size = hidden_states.shape[1] // 2

    key, value = torch.split(hidden_states, [split_size, split_size], dim=1)
    # key/value shape: [batch_size, hidden_size * num_hidden_states / 2]

    # Reshape to attention head format
    key = key.view(batch_size, num_heads, head_size)
    value = value.view(batch_size, num_heads, head_size)
    return key, value


@register_backend(AttentionBackendEnum.CUSTOM)
class CacheOnlyAttentionBackend(TritonAttentionBackend):
    """Attention backend that only caches KV without computing attention."""

    @staticmethod
    def get_name() -> str:
        return "CUSTOM"

    @classmethod
    def supports_attn_type(cls, attn_type: str) -> bool:
        return attn_type == AttentionType.DECODER

    @staticmethod
    def get_impl_cls() -> type["TritonAttentionImpl"]:
        return CacheOnlyAttentionImpl


class CacheOnlyAttentionImpl(TritonAttentionImpl):
    """Attention implementation that only caches KV states."""

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: TritonAttentionMetadata,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass with Paged Attention impl. in Triton.

        Args:
            query: shape = [num_tokens, num_heads, head_size]
            key: shape = [num_tokens, num_kv_heads, head_size]
            value: shape = [num_tokens, num_kv_heads, head_size]
            kv_cache: shape =
                [num_blocks, 2, block_size, num_kv_heads, head_size]
            attn_metadata: Metadata for attention.
        Returns:
            shape = [num_tokens, num_heads * head_size]
        """
        assert output is not None, "Output tensor must be provided."

        return output.fill_(0)


class ExtractHiddenStatesModel(nn.Module):
    """Model that extracts and caches hidden states without token generation.

    This model is used with the extract_hidden_states speculative decoding method.
    It processes hidden states from the target model and caches them via a
    cache-only attention mechanism, optionally transferring them to external storage.
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        self.config = vllm_config.speculative_config.draft_model_config.hf_config
        self.hidden_size = vllm_config.model_config.get_hidden_size()
        self.target_num_hidden_layers = (
            vllm_config.model_config.get_total_num_hidden_layers()
        )
        self.num_hidden_states = len(
            getattr(self.config, "eagle_aux_hidden_state_layer_ids", [])
        )

        # Attention configuration from draft config
        self.head_size = self.config.head_dim
        cache_config = vllm_config.cache_config

        # todo(fynn): loosen this constraint
        # Currently because we store data in both k and v caches
        # We need self.hidden_size // self.head_size to be even
        assert self.hidden_size % (self.head_size * 2) == 0, (
            "(hidden_size // head_size) must be even"
        )

        self.num_heads = (
            self.hidden_size // (self.head_size * 2)
        ) * self.num_hidden_states

        # Create a single cache-only attention layer
        self.cache_only_layers = nn.ModuleDict(
            {
                str(self.target_num_hidden_layers): Attention(
                    num_heads=self.num_heads,  # no gqa
                    head_size=self.head_size,
                    scale=1.0,
                    cache_config=cache_config,
                    prefix=maybe_prefix(
                        prefix, f"cache_only_layers.{self.target_num_hidden_layers}"
                    ),
                    attn_backend=CacheOnlyAttentionBackend,
                )
            }
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Process and cache hidden states.

        Args:
            hidden_states: Hidden states from target model
                          shape: [num_tokens, hidden_size * num_hidden_states]

        Returns:
            Tuple of (dummy_output, dummy_output) - both unused
        """
        # Cache the hidden states
        cache_layer = self.cache_only_layers[str(self.target_num_hidden_layers)]

        key, value = reshape_hidden_states_for_kv_cache(
            hidden_states, self.num_heads, self.head_size
        )
        _out = cache_layer(
            key,  # query (we just pass key for now, which will be ignored)
            key,
            value,
            # output_shape=None,
        )

        # Return dummy outputs (required by interface but not used)
        dummy_output = hidden_states.new_zeros(
            (hidden_states.shape[0], hidden_states.shape[1] // self.num_hidden_states)
        )
        return dummy_output, dummy_output

    def load_weights(self, weights):
        """No weights to load for this dummy model."""
        return None
