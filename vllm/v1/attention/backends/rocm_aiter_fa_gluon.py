# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ROCM_AITER_FA with single-token decodes on AITER's gluon paged attention."""

import torch

from vllm._aiter_ops import rocm_aiter_ops
from vllm.config import (
    VllmConfig,
    get_current_vllm_config_or_none,
    get_layers_from_vllm_config,
)
from vllm.config.cache import CacheDType
from vllm.model_executor.layers.attention import Attention
from vllm.platforms import current_platform
from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backend import CommonAttentionMetadata, MultipleOf
from vllm.v1.attention.backends.rocm_aiter_fa import (
    AiterFlashAttentionBackend,
    AiterFlashAttentionImpl,
    AiterFlashAttentionMetadata,
    AiterFlashAttentionMetadataBuilder,
)
from vllm.v1.kv_cache_interface import AttentionSpec, is_quantized_kv_cache

_HEAD_SIZE = 256
_QUERY_GROUP_SIZES = (6,)
_BLOCK_SIZE = 16
_CONTEXT_PARTITION_SIZE = 128

if current_platform.is_rocm():
    from aiter.ops.triton.gluon.pa_decode_gluon import (
        get_recommended_splits,
        pa_decode_gluon,
    )


def pa_gluon_supports(
    num_heads_q: int, num_heads_kv: int, head_size: int, kv_cache_dtype: str
) -> bool:
    """Whether this GPU can run the gluon decode for a layer with this head
    config and KV cache."""
    from vllm.platforms.rocm import on_gfx950

    return (
        on_gfx950()
        and rocm_aiter_ops.is_shuffle_kv_cache_enabled()
        and num_heads_kv > 0
        and num_heads_q % num_heads_kv == 0
        and num_heads_q // num_heads_kv in _QUERY_GROUP_SIZES
        and head_size == _HEAD_SIZE
        and not is_quantized_kv_cache(kv_cache_dtype)
    )


class AiterFlashAttentionGluonBackend(AiterFlashAttentionBackend):
    @staticmethod
    def get_supported_kernel_block_sizes(kv_cache_spec=None) -> list[int | MultipleOf]:
        return [_BLOCK_SIZE]

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return [_HEAD_SIZE]

    @classmethod
    def supports_sink(cls) -> bool:
        return False

    @classmethod
    def supports_sliding_window(cls) -> bool:
        return False

    @classmethod
    def supports_combination(
        cls,
        head_size: int,
        dtype: torch.dtype,
        kv_cache_dtype: CacheDType | None,
        block_size: int | None,
        use_mla: bool,
        has_sink: bool,
        use_sparse: bool,
        use_mm_prefix: bool,
        device_capability: DeviceCapability,
    ) -> str | None:
        vllm_config = get_current_vllm_config_or_none()
        if vllm_config is None or vllm_config.model_config is None:
            return "ROCM_AITER_FA_GLUON needs the model's head counts"
        parallel_config = vllm_config.parallel_config
        if parallel_config.world_size != 1 or parallel_config.enable_expert_parallel:
            return (
                "ROCM_AITER_FA_GLUON is validated only for TP x PP x PCP = 1 "
                "without expert parallelism"
            )
        if vllm_config.speculative_config is not None:
            return "ROCM_AITER_FA_GLUON does not support speculative decoding"
        model_config = vllm_config.model_config
        if not pa_gluon_supports(
            model_config.get_num_attention_heads(parallel_config),
            model_config.get_num_kv_heads(parallel_config),
            head_size,
            kv_cache_dtype or "auto",
        ):
            return (
                f"ROCM_AITER_FA_GLUON needs gfx950, the shuffle KV-cache layout, "
                f"head size {_HEAD_SIZE}, a GQA group in {_QUERY_GROUP_SIZES} and "
                "an unquantized KV cache"
            )
        return None

    @staticmethod
    def get_impl_cls() -> type["AiterFlashAttentionGluonImpl"]:
        return AiterFlashAttentionGluonImpl

    @staticmethod
    def get_builder_cls() -> type["AiterFlashAttentionGluonMetadataBuilder"]:
        return AiterFlashAttentionGluonMetadataBuilder

    @staticmethod
    def get_name() -> str:
        return "ROCM_AITER_FA_GLUON"


class AiterFlashAttentionGluonMetadataBuilder(AiterFlashAttentionMetadataBuilder):
    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        layers = get_layers_from_vllm_config(vllm_config, Attention)
        self._gluon_layers = [
            layers[name]
            for name in layer_names
            if name in layers and getattr(layers[name].impl, "use_gluon_decode", False)
        ]

    def _zero_null_block(self) -> None:
        """Zero the null block: the gluon kernel reads masked V rows."""
        blocks = [
            layer.kv_cache[0]
            for layer in self._gluon_layers
            if isinstance(layer.kv_cache, torch.Tensor) and layer.kv_cache.numel() > 0
        ]
        if blocks:
            torch._foreach_zero_(blocks)

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> AiterFlashAttentionMetadata:
        metadata = super().build(common_prefix_len, common_attn_metadata, fast_build)
        if metadata.num_decodes > 0:
            self._zero_null_block()
        return metadata


class AiterFlashAttentionGluonImpl(AiterFlashAttentionImpl):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.use_gluon_decode = (
            pa_gluon_supports(
                self.num_heads, self.num_kv_heads, self.head_size, self.kv_cache_dtype
            )
            and self.alibi_slopes is None
            and not self.logits_soft_cap
        )

    def _decode_shuffle_layout(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        output: torch.Tensor,
        attn_metadata: AiterFlashAttentionMetadata,
        num_decodes: int,
        num_decode_tokens: int,
    ) -> None:
        if not self.use_gluon_decode:
            super()._decode_shuffle_layout(
                layer,
                query,
                key_cache,
                value_cache,
                output,
                attn_metadata,
                num_decodes,
                num_decode_tokens,
            )
            return
        num_blocks, block_size, num_kv_heads, head_size = key_cache.shape
        num_heads = query.shape[1]
        x = 16 // key_cache.element_size()
        split_kwargs = (
            {"head_size": head_size, "kv_cache_dtype": key_cache.dtype}
            if rocm_aiter_ops.pa_gluon_splits_support_kv_row_size()
            else {}
        )
        splits = get_recommended_splits(num_decodes, num_kv_heads, **split_kwargs)
        partial_shape = (num_decodes, num_kv_heads, splits, num_heads // num_kv_heads)
        exp_sums = torch.empty(partial_shape, dtype=torch.float32, device=query.device)
        pa_decode_gluon(
            output=output[:num_decode_tokens],
            query=query[:num_decode_tokens],
            key_cache=key_cache.reshape(
                num_blocks, num_kv_heads, head_size // x, block_size, x
            ),
            value_cache=value_cache.reshape(
                num_blocks, num_kv_heads, block_size // x, head_size, x
            ),
            context_lengths=attn_metadata.seq_lens[:num_decodes].to(torch.int32),
            block_tables=attn_metadata.block_table[:num_decodes].to(torch.int32),
            softmax_scale=self.scale,
            query_length=1,
            max_context_partition_num=splits,
            context_partition_size=_CONTEXT_PARTITION_SIZE,
            compute_type=query.dtype,
            query_scale=None,
            key_scale=None,
            value_scale=None,
            exp_sums=exp_sums,
            max_logits=torch.empty_like(exp_sums),
            temporary_output=torch.empty(
                (*partial_shape, head_size), dtype=output.dtype, device=query.device
            ),
            alibi_slopes=None,
            sinks=None,
            sliding_window=0,
            ps=True,
        )
