# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import functools
from typing import Any

import torch

from vllm.config import CacheConfig, VllmConfig
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionMetadata,
    AttentionMetadataBuilder,
    AttentionType,
    CommonAttentionMetadata,
    subclass_attention_backend_with_overrides,
)
from vllm.v1.attention.selector import get_attn_backend
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheSpec,
    get_kv_quant_mode,
)

_USE_DECODE_SLIDING_WINDOW = "_ref_sliding_window_attention_use_window"


def _full_attention_window(window: Any) -> Any:
    if isinstance(window, tuple):
        return (-1, -1)
    if isinstance(window, int):
        return -1
    return None


def _is_homogeneous_decode_batch(
    common_attn_metadata: CommonAttentionMetadata,
    decode_threshold: int = 1,
) -> bool:
    query_lens_cpu = (
        common_attn_metadata.query_start_loc_cpu[1:]
        - common_attn_metadata.query_start_loc_cpu[:-1]
    )
    if common_attn_metadata.seq_lens_cpu_upper_bound is not None:
        seq_lens_cpu = common_attn_metadata.seq_lens_cpu_upper_bound
    else:
        seq_lens_cpu = common_attn_metadata.seq_lens.cpu()
    context_lens_cpu = seq_lens_cpu - query_lens_cpu

    is_decode = (context_lens_cpu > 0) & (query_lens_cpu <= decode_threshold)
    if common_attn_metadata.is_prefilling is not None:
        is_prefilling = common_attn_metadata.is_prefilling.cpu()
        is_decode &= ~is_prefilling

    has_decode = bool(is_decode.any().item())
    has_prefill = bool((~is_decode).any().item())
    if has_decode and has_prefill:
        raise ValueError(
            "RefSlidingWindowAttention does not support mixed prefill/decode "
            "batches because attention backends accept one sliding-window "
            "setting per forward call."
        )
    return has_decode


@functools.lru_cache
def create_ref_sliding_window_attention_backend(
    underlying_attn_backend: type[AttentionBackend],
) -> type[AttentionBackend]:
    underlying_builder = underlying_attn_backend.get_builder_cls()
    underlying_impl = underlying_attn_backend.get_impl_cls()

    class RefSlidingWindowAttentionBuilder(underlying_builder):  # type: ignore
        @classmethod
        def get_cudagraph_support(
            cls: type["AttentionMetadataBuilder"],
            vllm_config: VllmConfig,
            kv_cache_spec,
        ) -> AttentionCGSupport:
            return AttentionCGSupport.NEVER

        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            if hasattr(self, "aot_schedule"):
                self.aot_schedule = False

        def build(
            self,
            common_prefix_len: int,
            common_attn_metadata: CommonAttentionMetadata,
            fast_build: bool = False,
        ) -> AttentionMetadata:
            use_sliding_window = _is_homogeneous_decode_batch(common_attn_metadata)
            metadata = super().build(
                common_prefix_len, common_attn_metadata, fast_build
            )
            setattr(metadata, _USE_DECODE_SLIDING_WINDOW, use_sliding_window)
            return metadata

    class RefSlidingWindowAttentionImpl(underlying_impl):  # type: ignore
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self.decode_sliding_window = self.sliding_window
            self.prefill_sliding_window = _full_attention_window(self.sliding_window)

        def forward(
            self,
            layer: torch.nn.Module,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            kv_cache: torch.Tensor,
            attn_metadata: AttentionMetadata,
            output: torch.Tensor,
            output_scale: torch.Tensor | None = None,
            output_block_scale: torch.Tensor | None = None,
        ) -> torch.Tensor:
            if attn_metadata is None:
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

            use_sliding_window = getattr(
                attn_metadata, _USE_DECODE_SLIDING_WINDOW, False
            )
            original_sliding_window = self.sliding_window
            self.sliding_window = (
                self.decode_sliding_window
                if use_sliding_window
                else self.prefill_sliding_window
            )
            try:
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
            finally:
                self.sliding_window = original_sliding_window

    return subclass_attention_backend_with_overrides(
        name_prefix="RefSlidingWindowAttention_",
        attention_backend_cls=underlying_attn_backend,
        overrides={
            "get_builder_cls": lambda: RefSlidingWindowAttentionBuilder,
            "get_impl_cls": lambda: RefSlidingWindowAttentionImpl,
        },
    )


class RefSlidingWindowAttention(Attention):
    """Decoder attention with causal prefill and sliding-window decode."""

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        sliding_window: int,
        num_kv_heads: int | None = None,
        alibi_slopes: list[float] | None = None,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        kv_sharing_target_layer_name: str | None = None,
        prefix: str = "",
        attn_type: str | None = None,
        **kwargs,
    ):
        if sliding_window <= 0:
            raise ValueError("sliding_window must be a positive integer.")
        if attn_type is not None:
            assert attn_type == AttentionType.DECODER, (
                "RefSlidingWindowAttention only supports AttentionType.DECODER"
            )

        self.decode_sliding_window = sliding_window
        dtype = torch.get_default_dtype()
        kv_cache_dtype = (
            cache_config.cache_dtype if cache_config is not None else "auto"
        )
        underlying_attn_backend = get_attn_backend(
            head_size,
            dtype,
            kv_cache_dtype,
            attn_type=AttentionType.DECODER,
        )
        attn_backend = create_ref_sliding_window_attention_backend(
            underlying_attn_backend
        )

        super().__init__(
            num_heads=num_heads,
            head_size=head_size,
            scale=scale,
            num_kv_heads=num_kv_heads,
            alibi_slopes=alibi_slopes,
            cache_config=cache_config,
            quant_config=quant_config,
            per_layer_sliding_window=sliding_window,
            prefix=prefix,
            attn_type=AttentionType.DECODER,
            kv_sharing_target_layer_name=kv_sharing_target_layer_name,
            attn_backend=attn_backend,
            **kwargs,
        )

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        assert self.attn_type == AttentionType.DECODER
        return FullAttentionSpec(
            block_size=vllm_config.cache_config.block_size,
            num_kv_heads=self.num_kv_heads,
            head_size=self.head_size,
            head_size_v=self.head_size_v,
            dtype=self.kv_cache_torch_dtype,
            kv_quant_mode=get_kv_quant_mode(self.kv_cache_dtype),
            sliding_window=self.decode_sliding_window,
        )
