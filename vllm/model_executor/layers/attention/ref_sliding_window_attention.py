# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import functools

import torch

from vllm.config import CacheConfig, VllmConfig
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionMetadata,
    AttentionType,
    CommonAttentionMetadata,
    subclass_attention_backend,
)
from vllm.v1.attention.selector import get_attn_backend
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheSpec,
    get_kv_quant_mode,
)


def _seq_lens_cpu(metadata: CommonAttentionMetadata) -> torch.Tensor:
    if metadata.seq_lens_cpu_upper_bound is not None:
        return metadata.seq_lens_cpu_upper_bound
    if metadata._seq_lens_cpu is not None:
        return metadata._seq_lens_cpu
    return metadata.seq_lens.cpu()


def _decode_mask(metadata: CommonAttentionMetadata) -> torch.Tensor:
    query_lens_cpu = (
        metadata.query_start_loc_cpu[1:] - metadata.query_start_loc_cpu[:-1]
    )
    is_decode = (_seq_lens_cpu(metadata) > query_lens_cpu) & (query_lens_cpu == 1)
    if metadata.is_prefilling is not None:
        is_decode &= ~metadata.is_prefilling.cpu()
    return is_decode.to(
        device=metadata.query_start_loc.device,
        dtype=torch.int32,
        non_blocking=True,
    )


@functools.lru_cache
def create_ref_sliding_window_attention_backend(
    underlying_attn_backend: type[AttentionBackend],
    sliding_window: int,
) -> type[AttentionBackend]:
    underlying_builder = underlying_attn_backend.get_builder_cls()

    class RefSlidingWindowAttentionBuilder(underlying_builder):  # type: ignore
        def build(
            self,
            common_prefix_len: int,
            common_attn_metadata: CommonAttentionMetadata,
            fast_build: bool = False,
        ) -> AttentionMetadata:
            metadata = super().build(
                common_prefix_len, common_attn_metadata, fast_build
            )
            metadata.ref_sliding_window = sliding_window
            metadata.ref_sliding_window_decode_mask = _decode_mask(common_attn_metadata)
            return metadata

    return subclass_attention_backend(
        name_prefix=f"RefSlidingWindowAttention_{sliding_window}_",
        attention_backend_cls=underlying_attn_backend,
        builder_cls=RefSlidingWindowAttentionBuilder,
    )


class RefSlidingWindowAttention(Attention):
    """Decoder attention with causal prefill and sliding-window decode.

    This layer relies on FlashAttention v4 mask_mod support to apply the
    sliding-window predicate only to decode rows in mixed batches.
    """

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
        if underlying_attn_backend.get_name() != "FLASH_ATTN":
            raise ValueError(
                "RefSlidingWindowAttention requires the FLASH_ATTN backend "
                "because mixed prefill/decode uses FlashAttention v4 mask_mod."
            )
        attn_backend = create_ref_sliding_window_attention_backend(
            underlying_attn_backend, sliding_window
        )

        super().__init__(
            num_heads=num_heads,
            head_size=head_size,
            scale=scale,
            num_kv_heads=num_kv_heads,
            alibi_slopes=alibi_slopes,
            cache_config=cache_config,
            quant_config=quant_config,
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
