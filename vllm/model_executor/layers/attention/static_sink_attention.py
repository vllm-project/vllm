# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import functools
from typing import cast

import torch
from torch import nn

from vllm.config import CacheConfig, VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.custom_op import CustomOp
from vllm.model_executor.layers.attention import Attention, MLAAttention
from vllm.model_executor.layers.linear import ColumnParallelLinear
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.utils.torch_utils import kv_cache_dtype_str_to_dtype
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionType,
    MLAAttentionImpl,
    subclass_attention_backend_with_overrides,
)
from vllm.v1.attention.selector import get_attn_backend
from vllm.v1.kv_cache_interface import (
    KVCacheSpec,
    MLAAttentionSpec,
    SlidingWindowMLASpec,
)

logger = init_logger(__name__)


@functools.lru_cache
def _get_static_sink_attn_impl_override() -> dict[type, type]:
    """Maps underlying attention impl classes to static-sink variants.

    Flash-attn based overrides are CUDA-only and may fail to import when FA2/FA3
    extensions are unavailable. Non-CUDA platforms fall back to the underlying
    impl with only the KV-cache reshape override applied.
    """
    overrides: dict[type, type] = {}
    try:
        from vllm.v1.attention.backends.mla.flashattn_mla import (
            FlashAttnMLAImpl,
            FlashAttnStaticSinkMLAImpl,
        )

        overrides[FlashAttnMLAImpl] = FlashAttnStaticSinkMLAImpl
    except ImportError:
        pass
    try:
        from vllm.v1.attention.backends.mla.flashmla_sparse import (
            FlashMLASparseImpl,
            FlashMLASparseStaticSinkImpl,
        )

        overrides[FlashMLASparseImpl] = FlashMLASparseStaticSinkImpl
    except ImportError:
        pass
    try:
        from vllm.v1.attention.backends.flash_attn_diffkv import (
            FlashAttentionDiffKVImpl,
            FlashAttentionStaticSinkDiffKVImpl,
        )

        overrides[FlashAttentionDiffKVImpl] = FlashAttentionStaticSinkDiffKVImpl
    except ImportError:
        pass
    return overrides


@functools.lru_cache
def create_static_sink_attention_backend(
    underlying_attn_backend: type[AttentionBackend],
) -> type[AttentionBackend]:
    prefix = "StaticSink_"
    underlying_impl = underlying_attn_backend.get_impl_cls()
    sink_impl = _get_static_sink_attn_impl_override().get(underlying_impl)

    overrides: dict[str, object] = {}
    if sink_impl is not None:
        overrides["get_impl_cls"] = lambda: sink_impl

    return subclass_attention_backend_with_overrides(
        name_prefix=prefix,
        attention_backend_cls=underlying_attn_backend,
        overrides=overrides,
    )


@CustomOp.register("static_sink_attention")
class StaticSinkAttention(Attention, CustomOp):
    """
    Attention with static sink tokens
    """

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        sink_len: int,
        attn_backend: type[AttentionBackend] | None = None,
        cache_config: CacheConfig | None = None,
        **kwargs,
    ):
        CustomOp.__init__(self)
        dtype = torch.get_default_dtype()

        if cache_config is not None:
            kv_cache_dtype = cache_config.cache_dtype
        else:
            kv_cache_dtype = "auto"

        if attn_backend is not None:
            underlying_attn_backend = attn_backend
        else:
            underlying_attn_backend = get_attn_backend(head_size, dtype, kv_cache_dtype)
        attn_backend = create_static_sink_attention_backend(
            underlying_attn_backend,  # type: ignore[arg-type]
        )
        Attention.__init__(
            self=self,
            num_heads=num_heads,
            head_size=head_size,
            scale=scale,
            cache_config=cache_config,
            attn_backend=attn_backend,
            **kwargs,
        )
        self.sink_len = sink_len

    def update_sink_kv(self, sink_key, sink_value) -> None:
        self.impl.update_sink_kv(sink_key, sink_value)  # type: ignore[attr-defined]


@CustomOp.register("static_sink_mla_attention")
class StaticSinkMLAAttention(MLAAttention, CustomOp):
    """MLAAttention with static sink tokens."""

    def __init__(
        self,
        num_heads: int,
        scale: float,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        q_lora_rank: int | None,
        kv_lora_rank: int,
        kv_b_proj: ColumnParallelLinear,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        attn_backend: type[AttentionBackend] | None = None,
        use_sparse: bool = False,
        indexer: nn.Module | None = None,
        sink_len: int | None = None,
        sliding_window: int | None = None,
        **extra_impl_args,
    ):
        CustomOp.__init__(self)
        head_size = kv_lora_rank + qk_rope_head_dim
        if cache_config is not None:
            kv_cache_dtype = cache_config.cache_dtype
        else:
            kv_cache_dtype = "auto"
        dtype = torch.get_default_dtype()
        if attn_backend is not None:
            underlying_attn_backend = attn_backend
        else:
            underlying_attn_backend = get_attn_backend(
                head_size,
                dtype,
                kv_cache_dtype,
                use_mla=True,
                use_sparse=use_sparse,
                num_heads=num_heads,
            )
        sink_attn_backend = create_static_sink_attention_backend(
            underlying_attn_backend,
        )
        super().__init__(
            num_heads=num_heads,
            scale=scale,
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            v_head_dim=v_head_dim,
            q_lora_rank=q_lora_rank,
            kv_lora_rank=kv_lora_rank,
            kv_b_proj=kv_b_proj,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=prefix,
            attn_backend=sink_attn_backend,
            use_sparse=use_sparse,
            indexer=indexer,
            **extra_impl_args,
        )
        self.indexer = indexer
        self.use_sparse = use_sparse
        self.sink_len = sink_len
        self.sliding_window = sliding_window
        self.block_size = cache_config.block_size if cache_config is not None else 16
        impl_cls = cast(type[MLAAttentionImpl], self.attn_backend.get_impl_cls())
        self.impl = impl_cls(  # type: ignore[assignment]
            num_heads=self.num_heads,
            head_size=self.head_size,
            scale=self.scale,
            num_kv_heads=1,
            alibi_slopes=None,
            sliding_window=self.sliding_window,
            kv_cache_dtype=self.kv_cache_dtype,
            logits_soft_cap=None,
            attn_type=AttentionType.DECODER,
            kv_sharing_target_layer_name=None,
            q_lora_rank=self.q_lora_rank,
            kv_lora_rank=self.kv_lora_rank,
            qk_nope_head_dim=self.qk_nope_head_dim,
            qk_rope_head_dim=self.qk_rope_head_dim,
            qk_head_dim=self.qk_nope_head_dim + self.qk_rope_head_dim,
            v_head_dim=self.v_head_dim,
            kv_b_proj=kv_b_proj,
            indexer=indexer,
            **extra_impl_args,
        )

    def update_sink_kv(self, sink_k_pe, sink_compressed_kv) -> None:
        self.impl.update_sink_kv(sink_k_pe, sink_compressed_kv)  # type: ignore[attr-defined]

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        kv_cache_dtype = kv_cache_dtype_str_to_dtype(
            self.kv_cache_dtype, vllm_config.model_config
        )
        # Use max_sliding_window for KV management grouping
        max_sliding_window = getattr(
            vllm_config.model_config.hf_config,
            "max_sliding_window",
            self.sliding_window,
        )
        if self.sliding_window is not None:
            return SlidingWindowMLASpec(
                block_size=vllm_config.cache_config.block_size,
                num_kv_heads=1,
                head_size=self.head_size,
                dtype=kv_cache_dtype,
                cache_dtype_str=vllm_config.cache_config.cache_dtype,
                sliding_window=max_sliding_window,  # type: ignore[arg-type]
                compress_ratio=1,
                alignment=576,
            )
        return MLAAttentionSpec(
            block_size=vllm_config.cache_config.block_size,
            num_kv_heads=1,
            head_size=self.head_size,
            dtype=kv_cache_dtype,
            cache_dtype_str=vllm_config.cache_config.cache_dtype,
            compress_ratio=1,
            alignment=576,
        )
