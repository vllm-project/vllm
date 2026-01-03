# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Hidden States Extractor Model.

This model extracts and caches hidden states from the target model
without performing actual token generation. It's used with the
extract_hidden_states speculative decoding method.
"""

from typing import ClassVar

import torch
import torch.nn as nn

from vllm.config import CacheConfig, VllmConfig
from vllm.config.cache import CacheDType
from vllm.logger import init_logger
from vllm.model_executor.layers.attention.attention import (
    get_attention_context,
    set_default_quant_scales,
)
from vllm.model_executor.layers.attention.kv_transfer_utils import (
    maybe_transfer_kv_layer,
)
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.models.utils import maybe_prefix
from vllm.utils.torch_utils import kv_cache_dtype_str_to_dtype
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionImpl,
    AttentionMetadataBuilder,
    AttentionType,
    CommonAttentionMetadata,
    is_quantized_kv_cache,
)
from vllm.v1.attention.backends.registry import AttentionBackendEnum, register_backend
from vllm.v1.kv_cache_interface import AttentionSpec, FullAttentionSpec, KVCacheSpec

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
class CacheOnlyAttentionBackend(AttentionBackend):
    """Attention backend that only caches KV without computing attention."""

    accept_output_buffer: bool = False
    supported_dtypes: ClassVar[list[torch.dtype]] = [
        torch.float16,
        torch.bfloat16,
        torch.float32,
    ]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = ["auto"]

    @staticmethod
    def get_name() -> str:
        return "CUSTOM"

    @classmethod
    def supports_attn_type(cls, attn_type: str) -> bool:
        return attn_type == AttentionType.DECODER

    @classmethod
    def supports_mm_prefix(cls) -> bool:
        return True

    @staticmethod
    def get_impl_cls() -> type["CacheOnlyAttentionImpl"]:
        return CacheOnlyAttentionImpl

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        return (2, num_blocks, block_size, num_kv_heads, head_size)

    @staticmethod
    def get_builder_cls() -> type["CacheOnlyAttentionMetadataBuilder"]:
        return CacheOnlyAttentionMetadataBuilder

    @staticmethod
    def use_cascade_attention(*args, **kwargs) -> bool:
        return False

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return []


class CacheOnlyAttentionMetadata:
    def __init__(self, slot_mapping: torch.Tensor):
        self.slot_mapping = slot_mapping


class CacheOnlyAttentionMetadataBuilder(
    AttentionMetadataBuilder[CacheOnlyAttentionMetadata]
):
    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self.model_config = vllm_config.model_config
        self.parallel_config = vllm_config.parallel_config
        self.cache_config = vllm_config.cache_config
        self.block_size = kv_cache_spec.block_size
        self.kv_cache_spec = kv_cache_spec

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> CacheOnlyAttentionMetadata:
        use_cascade = common_prefix_len > 0
        if use_cascade:
            raise NotImplementedError(
                "Cascade attention not supported by CacheOnlyAttention"
            )
        causal = common_attn_metadata.causal
        if not causal:
            raise NotImplementedError(
                "Non-causal attention not supported by CacheOnlyAttention"
            )

        slot_mapping = common_attn_metadata.slot_mapping
        return CacheOnlyAttentionMetadata(
            slot_mapping=slot_mapping,
        )


class CacheOnlyAttentionImpl(AttentionImpl):
    """Attention implementation that only caches KV states."""

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None = None,
        attn_type: AttentionType = AttentionType.DECODER,
        kv_sharing_target_layer_name: str | None = None,
        **kwargs,
    ) -> None:
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        self.attn_type = attn_type
        self.sliding_window = sliding_window
        self.kv_cache_dtype = kv_cache_dtype
        self.logits_soft_cap = logits_soft_cap
        self.alibi_slopes = None

        if attn_type != AttentionType.DECODER:
            raise NotImplementedError(f"Unsupported attention type: {attn_type}")
        if sliding_window is not None:
            raise NotImplementedError("Sliding window is not supported")
        if alibi_slopes is not None:
            raise NotImplementedError("ALiBi slopes not supported")
        if logits_soft_cap is not None:
            raise NotImplementedError("Logits soft cap not supported")
        if kv_sharing_target_layer_name is not None:
            raise NotImplementedError("KV sharing not supported")
        if is_quantized_kv_cache(kv_cache_dtype):
            raise NotImplementedError("Quantized KV cache not supported")

        self.num_queries_per_kv = 1

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: CacheOnlyAttentionMetadata,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Cache key/value states without computing attention.

        Args:
            layer: Attention layer module
            query: Not used (can be None)
            key: shape = [num_tokens, num_kv_heads, head_size]
            value: shape = [num_tokens, num_kv_heads, head_size]
            kv_cache: shape = [2, num_blocks, block_size, num_kv_heads, head_size]
            attn_metadata: Metadata for caching
            output: Output tensor (filled with zeros)

        Returns:
            output tensor filled with zeros
        """
        assert output is not None, "Output tensor required"
        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError("Fused output quantization not supported")

        if attn_metadata is None:
            # Profiling run
            return output.fill_(0)

        assert self.attn_type == AttentionType.DECODER

        # Cache the key/value states
        key_cache, value_cache = kv_cache.unbind(0)
        torch.ops._C_cache_ops.reshape_and_cache_flash(
            key,
            value,
            key_cache,
            value_cache,
            attn_metadata.slot_mapping,
            self.kv_cache_dtype,
            layer._k_scale,
            layer._v_scale,
        )

        # Return dummy output (not used)
        return output.fill_(0)


class CacheOnlyAttentionLayer(nn.Module, AttentionLayerBase):
    """Attention layer that only caches key/value states without computing attention."""

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        attn_type: str = AttentionType.DECODER,
        num_hidden_states: int = 1,
        **extra_impl_args,
    ):
        super().__init__()

        self.num_heads = num_heads
        self.head_size = head_size
        self.quant_config = quant_config
        self.layer_name = prefix

        # Get vllm config
        from vllm.config import get_current_vllm_config

        vllm_config = get_current_vllm_config()

        # KV cache configuration
        cache_config = cache_config or vllm_config.cache_config
        if cache_config is not None:
            kv_cache_dtype = cache_config.cache_dtype
            self.block_size = cache_config.block_size
        else:
            kv_cache_dtype = "auto"
            self.block_size = 16

        self.kv_cache_torch_dtype = kv_cache_dtype_str_to_dtype(
            kv_cache_dtype, vllm_config.model_config
        )

        # Initialize KV cache quantization attributes
        set_default_quant_scales(self, register_buffer=True)

        # Default dtype
        dtype = torch.get_default_dtype()
        self.dtype = dtype

        # Attention backend
        self.attn_backend = CacheOnlyAttentionBackend
        impl_cls = self.attn_backend.get_impl_cls()
        self.impl = impl_cls(
            num_heads,
            head_size,
            1.0,  # scale
            num_heads,  # num kv heads
            None,  # alibi_slopes
            None,  # sliding_window
            kv_cache_dtype,
            None,  # logits_soft_cap
            attn_type,
            None,  # kv_sharing_target_layer_name
            **extra_impl_args,
        )

        # Placeholder KV cache (replaced by bind_kv_cache)
        self.kv_cache = [
            torch.tensor([])
            for _ in range(vllm_config.parallel_config.pipeline_parallel_size)
        ]

        # Register in compilation context
        compilation_config = vllm_config.compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self

    def forward(self, hidden_states: torch.Tensor):
        """Cache hidden states as KV pairs without computing attention.

        Args:
            hidden_states: shape [num_tokens, hidden_size * num_hidden_states]

        Returns:
            Dummy output tensor (not used)
        """
        output = torch.empty(
            hidden_states.shape[0],  # num_tokens
            self.num_heads,
            self.head_size,
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

        # Reshape hidden states into key/value format
        key, value = reshape_hidden_states_for_kv_cache(
            hidden_states, self.num_heads, self.head_size
        )

        # Cache the KV states (with optional KV transfer)
        cache_only_attention_with_kv_transfer(None, key, value, output, self.layer_name)

        return output

    def get_attn_backend(self) -> type[AttentionBackend]:
        return self.attn_backend

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        return FullAttentionSpec(
            block_size=self.block_size,
            num_kv_heads=self.num_heads,
            head_size=self.head_size,
            dtype=self.kv_cache_torch_dtype,
        )


@maybe_transfer_kv_layer
def cache_only_attention_with_kv_transfer(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
    output_scale: torch.Tensor | None = None,
    output_block_scale: torch.Tensor | None = None,
) -> None:
    """Cache KV states with optional KV transfer support."""
    attn_metadata, self, kv_cache = get_attention_context(layer_name)

    self.impl.forward(
        self,
        query,
        key,
        value,
        kv_cache,
        attn_metadata,
        output=output,
        output_scale=output_scale,
        output_block_scale=output_block_scale,
    )


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
            "hidden_size // head_size must be even"
        )

        self.num_kv_heads = (
            self.hidden_size // (self.head_size * 2)
        ) * self.num_hidden_states

        # Create a single cache-only attention layer
        # Note: We double the heads to match the combined hidden states format
        self.cache_only_layers = nn.ModuleDict(
            {
                str(self.target_num_hidden_layers): CacheOnlyAttentionLayer(
                    num_heads=self.num_kv_heads,
                    head_size=self.head_size,
                    cache_config=cache_config,
                    prefix=maybe_prefix(
                        prefix, f"cache_only_layers.{self.target_num_hidden_layers}"
                    ),
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
        # Output is ignored - we only care about the KV cache side effects
        # TODO(fynn): Confirm this doesn't get optimized away when compiled
        cache_layer(hidden_states)

        # Return dummy outputs (required by interface but not used)
        dummy_output = hidden_states.new_zeros(
            (hidden_states.shape[0], hidden_states.shape[1] // self.num_hidden_states)
        )
        return dummy_output, dummy_output

    def load_weights(self, weights):
        """No weights to load for this dummy model."""
        return None
