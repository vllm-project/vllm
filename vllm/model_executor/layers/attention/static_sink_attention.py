# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import functools
from typing import cast

import torch
from vllm import _custom_ops as ops
from vllm.config import CacheConfig, VllmConfig
from vllm.forward_context import ForwardContext, get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.custom_op import CustomOp
from vllm.model_executor.layers.attention import Attention, MLAAttention
from vllm.model_executor.layers.linear import ColumnParallelLinear
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.utils.math_utils import cdiv
from vllm.utils.torch_utils import (
    LayerNameType,
    _encode_layer_name,
    _resolve_layer_name,
    direct_register_custom_op,
    kv_cache_dtype_str_to_dtype,
)
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionMetadata,
    AttentionType,
    CommonAttentionMetadata,
    MLAAttentionImpl,
    subclass_attention_backend,
)
from vllm.v1.attention.ops.triton_reshape_and_cache_flash import (
    triton_reshape_and_cache_flash_diffkv,
)
from vllm.v1.attention.selector import get_attn_backend
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    KVCacheSpec,
    SinkFullAttentionSpec,
    get_kv_quant_mode,
    MLAAttentionSpec,
    SinkMLAAttentionSpec,
    SinkMLASlidingWindowSpec
)
from vllm.v1.attention.backends.mla.flashattn_mla import FlashAttnStaticSinkMLAImpl

logger = init_logger(__name__)


@functools.lru_cache
def create_static_sink_attention_backend(
    underlying_attn_backend: type[AttentionBackend],
    sink_len: int = 0,
    sink_kv_block_offset: int = 1
) -> type[AttentionBackend]:
    prefix = "StaticSink_"
    underlying_builder = underlying_attn_backend.get_builder_cls()

    class StaticSinkAttentionBuilder(underlying_builder):  # type: ignore
        def __init__(
            self,
            kv_cache_spec: AttentionSpec,
            layer_names: list[str],
            vllm_config: VllmConfig,
            device: torch.device,
        ):
            super().__init__(kv_cache_spec, layer_names, vllm_config, device)
            model_config = vllm_config.model_config
            scheduler_config = vllm_config.scheduler_config
            self.sink_len = sink_len
            self.block_size = vllm_config.cache_config.block_size
            self.num_sink_blocks = self.sink_len // vllm_config.cache_config.block_size
            self.max_num_blocks = cdiv(
                model_config.max_model_len, vllm_config.cache_config.block_size
            )
            self.block_table_with_sink = torch.zeros(
                (
                    scheduler_config.max_num_seqs,
                    self.max_num_blocks + self.num_sink_blocks,
                ),
                device=device,
                dtype=torch.int32,
            )
            self.block_table_with_sink[:, : self.num_sink_blocks] = torch.arange(
                sink_kv_block_offset,
                self.num_sink_blocks + sink_kv_block_offset,
                device=device,
                dtype=torch.int32,
            )

        def build(
            self,
            common_prefix_len: int,
            common_attn_metadata: CommonAttentionMetadata,
            fast_build: bool = False,
        ) -> AttentionMetadata:
            if common_attn_metadata.block_table_tensor[0, 0] != 1:
                max_num_blocks = cdiv(common_attn_metadata.max_seq_len, self.block_size)
                num_reqs = common_attn_metadata.num_reqs
                self.block_table_with_sink[
                    :num_reqs, self.num_sink_blocks : self.num_sink_blocks + max_num_blocks
                ] = common_attn_metadata.block_table_tensor[:, :max_num_blocks]
                common_attn_metadata.block_table_tensor = self.block_table_with_sink[:num_reqs]
                common_attn_metadata.block_table_tensor.masked_fill_(
                    common_attn_metadata.block_table_tensor == -1 , 0
                )

                zero_mask = common_attn_metadata.seq_lens.eq(0)
                common_attn_metadata.seq_lens.add_(self.sink_len)
                common_attn_metadata.seq_lens.masked_fill_(zero_mask, 0)
                common_attn_metadata.max_seq_len += self.sink_len


            return super().build(common_prefix_len, common_attn_metadata, fast_build)

    attn_backend = subclass_attention_backend(
        name_prefix=prefix,
        attention_backend_cls=underlying_attn_backend,
        builder_cls=StaticSinkAttentionBuilder,
    )

    return attn_backend


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
            sink_len=sink_len,
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
        CustomOp.__init__(self)

        self.sink_len = sink_len
        self.sink_populated = False
        self.sink_key = None
        self.sink_value = None

    def update_sink_kv(self, sink_key, sink_value) -> None:
        self.sink_key = sink_key
        self.sink_value = sink_value

    def forward_native(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        output_shape: torch.Size | None = None,
    ) -> torch.Tensor:
        assert self.sink_key is not None and self.sink_value is not None, (
            "sink_key and sink_value have not been prepared"
        )
        if not self.sink_populated:
            self_kv_cache = self.kv_cache
            torch.ops.vllm.maybe_populate_sink(
                self_kv_cache, _encode_layer_name(self.layer_name)
            )

        return super().forward(query, key, value, output_shape)

    def forward_cuda(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        output_shape: torch.Size | None = None,
    ) -> torch.Tensor:
        return self.forward_native(query, key, value, output_shape)

    def forward(self, *args, **kwargs):
        return self._forward_method(*args, **kwargs)

    def populate_sink_kv(self, self_kv_cache):
        sink_kv_slot_mapping = torch.arange(
            self.block_size,
            self.sink_len + self.block_size,
            device=torch.accelerator.current_device_index(),
            dtype=torch.long,
        )
        triton_reshape_and_cache_flash_diffkv(
            self.sink_key,
            self.sink_value,
            self_kv_cache,
            sink_kv_slot_mapping,
            self.kv_cache_dtype,
            self._k_scale,
            self._v_scale,
        )
        # We only populate the sink_key and sink_value once
        self.sink_populated = True

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        # Block size may get updated after model loading, refresh it
        self.block_size = vllm_config.cache_config.block_size
        # Should not be called for enc-dec or encoder-only attention.
        assert self.attn_type == AttentionType.DECODER

        return SinkFullAttentionSpec(
            block_size=self.block_size,
            num_kv_heads=self.num_kv_heads,
            head_size=self.head_size,
            head_size_v=self.head_size_v,
            sink_len=self.sink_len,
            dtype=self.kv_cache_torch_dtype,
            kv_quant_mode=get_kv_quant_mode(self.kv_cache_dtype),
        )


def maybe_populate_sink(
    self_kv_cache: torch.Tensor,
    layer_name: LayerNameType,
) -> None:
    layer_name = _resolve_layer_name(layer_name)
    forward_context: ForwardContext = get_forward_context()
    self = forward_context.no_compile_layers[layer_name]
    if self.sink_populated or self_kv_cache.numel() == 0:
        return
    self.populate_sink_kv(self_kv_cache)


def maybe_populate_sink_fake(
    self_kv_cache: torch.Tensor,
    layer_name: LayerNameType,
) -> None:
    return


direct_register_custom_op(
    op_name="maybe_populate_sink",
    op_func=maybe_populate_sink,
    mutates_args=["self_kv_cache"],
    fake_impl=maybe_populate_sink_fake,
)


class StaticSinkMLAAttention(MLAAttention):
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
        use_sparse: bool = False,
        indexer: object | None = None,
        sink_len: int | None = None,
        sliding_window: int | None = None,
        is_hybrid_kv: bool = False,
        **extra_impl_args,
    ):
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
            use_sparse=use_sparse,
            indexer=indexer,
            **extra_impl_args,
        )
        self.sink_len = sink_len
        self.sliding_window = sliding_window
        self.sink_kv_block_offset = 1
        self.block_size = cache_config.block_size if cache_config is not None else 16
        self.sink_populated = False
        self.sink_k_pe = None
        self.sink_compressed_kv = None
        if is_hybrid_kv and self.sliding_window is None:
            self.sink_kv_block_offset += self.sink_len // self.block_size
        self.attn_backend = create_static_sink_attention_backend(
            self.attn_backend,
            sink_len=self.sink_len,
            sink_kv_block_offset=self.sink_kv_block_offset
        )
        impl_cls = cast(type[MLAAttentionImpl], FlashAttnStaticSinkMLAImpl)
        self.impl = impl_cls(
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
        self.sink_k_pe = sink_k_pe
        self.sink_compressed_kv = sink_compressed_kv
        self.impl.update_sink_kv(sink_k_pe, sink_compressed_kv)

    def forward(self, *args, **kwargs) -> torch.Tensor:
        assert self.sink_k_pe is not None and self.sink_compressed_kv is not None, (
            "sink_k_pe and sink_compressed_kv have not been prepared"
        )
        if not self.sink_populated:
            forward_context: ForwardContext = get_forward_context()
            self_kv_cache = self.kv_cache[forward_context.virtual_engine]
            torch.ops.vllm.maybe_populate_sink(self_kv_cache, self.layer_name)
        return super().forward(*args, **kwargs)

    def populate_sink_kv(self, self_kv_cache: torch.Tensor):
        sink_kv_slot_mapping = torch.arange(
            self.block_size * self.sink_kv_block_offset,
            self.sink_len + self.block_size * self.sink_kv_block_offset,
            device=torch.accelerator.current_device_index(),
            dtype=torch.long,
        )
        ops.concat_and_cache_mla(
            self.sink_compressed_kv,
            self.sink_k_pe,
            self_kv_cache,
            sink_kv_slot_mapping,
            kv_cache_dtype=self.kv_cache_dtype,
            scale=self._k_scale
        )
        self.sink_populated = True

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        kv_cache_dtype = kv_cache_dtype_str_to_dtype(
            self.kv_cache_dtype, vllm_config.model_config
        )
        page_size_padded = vllm_config.cache_config.mamba_page_size_padded
        # Use max_sliding_window for KV management grouping
        max_sliding_window = getattr(vllm_config.model_config.hf_config, "max_sliding_window", self.sliding_window)
        if self.sink_len > 0 and self.sliding_window is not None:
            return SinkMLASlidingWindowSpec(
                block_size=vllm_config.cache_config.block_size,
                num_kv_heads=1,
                head_size=self.head_size,
                dtype=kv_cache_dtype,
                page_size_padded=page_size_padded,
                cache_dtype_str=vllm_config.cache_config.cache_dtype,
                sink_len=self.sink_len,
                sliding_window=max_sliding_window,
            )
        if self.sink_len > 0:
            return SinkMLAAttentionSpec(
                block_size=vllm_config.cache_config.block_size,
                num_kv_heads=1,
                head_size=self.head_size,
                dtype=kv_cache_dtype,
                page_size_padded=page_size_padded,
                cache_dtype_str=vllm_config.cache_config.cache_dtype,
                sink_len=self.sink_len,
            )
        return MLAAttentionSpec(
                block_size=vllm_config.cache_config.block_size,
                num_kv_heads=1,
                head_size=self.head_size,
                dtype=kv_cache_dtype,
                page_size_padded=page_size_padded,
                cache_dtype_str=vllm_config.cache_config.cache_dtype,
            )
