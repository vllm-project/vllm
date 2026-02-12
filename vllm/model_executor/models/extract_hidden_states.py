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

from vllm.config import CacheConfig, VllmConfig, get_current_vllm_config
from vllm.config.cache import CacheDType
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.attention.attention import set_default_quant_scales
from vllm.model_executor.layers.attention.kv_transfer_utils import (
    maybe_transfer_kv_layer,
)
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
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
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    KVCacheSpec,
    MLAAttentionSpec,
)

########## Custom Ops ########


def unified_kv_cache_update(
    to_cache: torch.Tensor,
    layer_name: str,
) -> torch.Tensor:
    """
    Returns a dummy that is passed to unified_attention to signal a side effect and
    the data dependency between them to ensure torch.compile preserves ordering.
    """
    forward_context = get_forward_context()
    attn_layer = forward_context.no_compile_layers[layer_name]
    kv_cache = attn_layer.kv_cache[forward_context.virtual_engine]

    slot_mapping = forward_context.slot_mapping
    assert isinstance(slot_mapping, dict), (
        f"Expected slot_mapping to be a dict, got {type(slot_mapping)}. "
    )
    layer_slot_mapping = slot_mapping.get(layer_name)
    if layer_slot_mapping is not None:
        assert hasattr(attn_layer.impl, "do_kv_cache_update"), (
            f"{attn_layer.impl.__class__.__name__} does not support kv cache update"
        )
        attn_layer.impl.do_kv_cache_update(
            attn_layer,
            to_cache,
            kv_cache,
            layer_slot_mapping,
        )

    return torch.empty(0, device=kv_cache.device, dtype=kv_cache.dtype)


@maybe_transfer_kv_layer
def dummy_attention(layer_name):
    # Note: layer_name arg required by @maybe_transfer_kv_layer
    pass


def basic_cache(
    to_cache: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
):
    num_blocks, block_size, num_heads, head_size = kv_cache.shape
    token_kv_cache = kv_cache.view(num_blocks * block_size, num_heads, head_size)
    token_kv_cache[slot_mapping] = to_cache


######### CacheOnlyAttentionBackend ########


@register_backend(AttentionBackendEnum.CUSTOM)
class CacheOnlyAttentionBackend(AttentionBackend):
    """Attention backend that only caches KV without computing attention."""

    accept_output_buffer: bool = False
    supported_dtypes: ClassVar[list[torch.dtype]] = [
        torch.float16,
        torch.bfloat16,
        torch.float32,
    ]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "bfloat16",
    ]
    forward_includes_kv_cache_update: bool = False

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
        # We set `num_kv_heads = num_hidden_layers` and `head_size = hidden_size`
        # We also don't use a k/v (2) dim
        return (num_blocks, block_size, num_kv_heads, head_size)

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

        return CacheOnlyAttentionMetadata(
            slot_mapping=common_attn_metadata.slot_mapping,
        )


class CacheOnlyAttentionImpl(AttentionImpl):
    """Attention implementation that only caches KV states."""

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        kv_cache_dtype: str,
        kv_cache_torch_dtype: torch.dtype,
        attn_type: AttentionType = AttentionType.DECODER,
    ) -> None:
        self.num_heads = num_heads
        self.head_size = head_size
        self.kv_cache_dtype = kv_cache_dtype
        self.kv_cache_torch_dtype = kv_cache_torch_dtype

        if attn_type != AttentionType.DECODER:
            raise NotImplementedError(f"Unsupported attention type: {attn_type}")
        if is_quantized_kv_cache(kv_cache_dtype):
            raise NotImplementedError("Quantized KV cache not supported")

        self.num_queries_per_kv = 1

    def do_kv_cache_update(
        self,
        layer,
        to_cache,
        kv_cache,
        slot_mapping,
    ):
        assert to_cache.dtype == self.kv_cache_torch_dtype, (
            f"Data to cache must be {self.kv_cache_torch_dtype}, got {to_cache.dtype}"
        )
        assert kv_cache.dtype == self.kv_cache_torch_dtype, (
            f"KV cache must be {self.kv_cache_torch_dtype}, got {kv_cache.dtype}"
        )

        # todo(fynn): Implement more performant op (maybe custom triton op?)
        basic_cache(to_cache, kv_cache, slot_mapping)

    def forward(self, *args, **kwargs):
        # Empty implementation of abstract method
        pass


############## CacheOnlyAttentionLayer (replaces Attention) ############


class CacheOnlyAttentionLayer(nn.Module, AttentionLayerBase):
    """Attention layer that only caches key/value states without computing attention."""

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        cache_config: CacheConfig | None = None,
        prefix: str = "",
        attn_type: str = AttentionType.DECODER,
    ):
        super().__init__()

        self.num_heads = num_heads
        self.head_size = head_size
        self.layer_name = prefix

        vllm_config = get_current_vllm_config()

        # KV cache configuration
        cache_config = cache_config or vllm_config.cache_config
        if cache_config is not None:
            kv_cache_dtype = cache_config.cache_dtype
            self.block_size = cache_config.block_size
        else:
            kv_cache_dtype = "auto"
            self.block_size = 16

        assert kv_cache_dtype in ["auto", "bfloat16", "float16"], (
            "CacheOnlyAttentionLayer doesn't currently support quantized kv cache but"
            f"kv cache dtype was set to {kv_cache_dtype}"
        )
        self.kv_cache_torch_dtype = kv_cache_dtype_str_to_dtype(
            kv_cache_dtype, vllm_config.model_config
        )

        # Initialize KV cache quantization attributes
        set_default_quant_scales(self, register_buffer=True)

        # Attention backend
        self.attn_backend = CacheOnlyAttentionBackend
        impl_cls = self.attn_backend.get_impl_cls()
        self.impl = impl_cls(
            num_heads,
            head_size,
            kv_cache_dtype,
            self.kv_cache_torch_dtype,
            attn_type,
        )

        assert not self.attn_backend.forward_includes_kv_cache_update, (
            "KV cache update should be independent of forward"
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

    def forward(self, to_cache: torch.Tensor):
        """Cache hidden states as KV pairs without computing attention.

        Args:
            hidden_states: shape [num_tokens, num_heads, head_size]

        Returns:
            Dummy output tensor (not used)
        """
        # Note: we set num_heads to num_hidden_layers and
        # head_size to hidden_size for hidden states storage
        output = torch.empty(0, device=to_cache.device, dtype=to_cache.dtype)

        # todo(fynn): determine if we need to use a custom op here
        # or if direct call only is sufficient
        _ = unified_kv_cache_update(to_cache, self.layer_name)

        # Triggers kv_connector transfer via decorator
        dummy_attention(self.layer_name)

        return output

    def get_attn_backend(self) -> type[AttentionBackend]:
        return self.attn_backend

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        # Note: we use MLAAttentionSpec here to because it will
        # produce page sizes of (block_size * num_kv_heads * head_size * dtype_size)
        # whereas FullAttentionSpec will add an additional factor of 2
        return MLAAttentionSpec(
            block_size=self.block_size,
            num_kv_heads=self.num_heads,
            head_size=self.head_size,
            dtype=self.kv_cache_torch_dtype,
        )


############ ExtractHiddenStatesModel definition ##########


class ExtractHiddenStatesModel(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        self.vllm_config = vllm_config
        self.hf_config = vllm_config.speculative_config.draft_model_config.hf_config
        self.hidden_size = vllm_config.model_config.get_hidden_size()
        self.target_num_hidden_layers = (
            vllm_config.model_config.get_total_num_hidden_layers()
        )
        self.num_hidden_states = len(
            getattr(self.hf_config, "eagle_aux_hidden_state_layer_ids", [])
        )

        # todo(fynn): Set up a new cache config for the drafter
        # independent of verifier's cache config
        cache_config = vllm_config.cache_config

        # Create a single cache-only attention layer
        # Note: We set num_heads <- self.num_hidden_states
        # and head_size <- hidden_size so that we can insert
        # the hidden states directly into the cache without
        # reshaping
        self.cache_only_layers = nn.ModuleDict(
            {
                str(self.target_num_hidden_layers): CacheOnlyAttentionLayer(
                    num_heads=self.num_hidden_states,
                    head_size=self.hidden_size,
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
                          shape: [num_tokens, num_hidden_states, hidden_size]

        Returns:
            Tuple of (dummy_output, dummy_output) - both unused
        """

        # Call dummy attention layer to cache hidden states
        # Output is ignored - we only care about the KV cache side effects
        _ = self.cache_only_layers[str(self.target_num_hidden_layers)](hidden_states)

        # Return dummy outputs (required by interface but not used)
        # todo(fynn): Remove these
        dummy_output = hidden_states.new_zeros(
            (hidden_states.shape[0], hidden_states.shape[1] // self.num_hidden_states)
        )
        return dummy_output, dummy_output

    def load_weights(self, weights):
        """No weights to load for this dummy model."""
        return None
