# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
import torch._inductor.pattern_matcher as pm
from torch import fx
from torch._higher_order_ops import auto_functionalized
from torch._inductor.fx_passes.post_grad import view_to_reshape
from torch._inductor.pattern_matcher import PatternMatcherPass

from vllm.config import VllmConfig, get_layers_from_vllm_config
from vllm.config.utils import Range
from vllm.logger import init_logger
from vllm.model_executor.layers.attention.attention import (
    Attention,
    get_attention_context,
)
from vllm.model_executor.layers.attention.mla_attention import MLAAttention
from vllm.utils.torch_utils import direct_register_custom_op

from ..inductor_pass import enable_fake_mode
from ..vllm_inductor_pass import VllmInductorPass, VllmPatternMatcherPass
from .matcher_utils import (
    MatcherRotaryEmbedding,
)
from .rms_quant_fusion import (
    empty_bf16,
    empty_i64,
)

logger = init_logger(__name__)


def fused_rope_and_unified_kv_cache_update_impl(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    is_neox: bool,
    layer_name: str = "",
) -> torch.Tensor:
    """
    This impl fetches the KV cache and slot mapping from the forward context,
    then calls the layer impl's `AttentionImpl.do_rope_and_kv_cache_update` method.
    It also returns a dummy tensor, similar to `Attention.unified_kv_cache_update`,
    that is passed to unified_attention to signal a side effect and
    the data dependency between them to ensure torch.compile preserves ordering.
    """
    _, attn_layer, kv_cache, layer_slot_mapping = get_attention_context(layer_name)
    if layer_slot_mapping is not None:
        attn_layer.impl.do_rope_and_kv_cache_update(
            attn_layer,
            query,
            key,
            value,
            positions,
            cos_sin_cache,
            is_neox,
            kv_cache,
            layer_slot_mapping,
        )

    return torch.empty(0, device=kv_cache.device, dtype=kv_cache.dtype)


def fused_rope_and_unified_kv_cache_update_fake(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    is_neox: bool,
    layer_name: str = "",
) -> torch.Tensor:
    return torch.empty(0, device=query.device, dtype=query.dtype)


direct_register_custom_op(
    op_name="fused_rope_and_unified_kv_cache_update",
    op_func=fused_rope_and_unified_kv_cache_update_impl,
    mutates_args=["query", "key"],
    fake_impl=fused_rope_and_unified_kv_cache_update_fake,
)


def fused_rope_and_unified_mla_kv_cache_update_impl(
    positions: torch.Tensor,
    q_pe: torch.Tensor,
    k_pe: torch.Tensor,
    kv_c: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    is_neox: bool,
    layer_name: str = "",
    kv_cache_dtype: str = "auto",
    k_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Fused MLA RoPE + KV cache update.

    This applies RoPE to q_pe/k_pe and inserts [kv_c, k_rope] into the MLA KV
    cache using concat_and_cache_mla_rope_fused, then returns a dummy tensor to
    preserve ordering/data dependency in torch.compile.
    """
    _, _, kv_cache, layer_slot_mapping = get_attention_context(layer_name)
    # Keep parity with unified_mla_kv_cache_update: this op should be present
    # and callable even when metadata is not initialized yet. We only skip
    # work when the KV cache itself is empty.
    if kv_cache.numel() == 0:
        return torch.empty(0, device=kv_c.device, dtype=kv_c.dtype)

    if layer_slot_mapping is not None:
        from vllm import _custom_ops as ops

        kv_cache_scale = (
            k_scale
            if k_scale is not None
            else torch.tensor([1.0], device=q_pe.device, dtype=torch.float32)
        )
        cos_sin_cache_for_kernel = (
            cos_sin_cache
            if cos_sin_cache.dtype == q_pe.dtype
            else cos_sin_cache.to(dtype=q_pe.dtype)
        )
        ops.concat_and_cache_mla_rope_fused(
            positions=positions,
            q_pe=q_pe,
            k_pe=k_pe.squeeze(1),
            kv_c=kv_c,
            cos_sin_cache=cos_sin_cache_for_kernel,
            is_neox=is_neox,
            slot_mapping=layer_slot_mapping,
            kv_cache=kv_cache,
            kv_cache_dtype=kv_cache_dtype,
            kv_cache_scale=kv_cache_scale,
        )

    return torch.empty(0, device=kv_c.device, dtype=kv_c.dtype)


def fused_rope_and_unified_mla_kv_cache_update_fake(
    positions: torch.Tensor,
    q_pe: torch.Tensor,
    k_pe: torch.Tensor,
    kv_c: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    is_neox: bool,
    layer_name: str = "",
    kv_cache_dtype: str = "auto",
    k_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    del positions, k_pe, kv_c, cos_sin_cache, is_neox, layer_name
    del kv_cache_dtype, k_scale
    return torch.empty(0, device=q_pe.device, dtype=q_pe.dtype)


direct_register_custom_op(
    op_name="fused_rope_and_unified_mla_kv_cache_update",
    op_func=fused_rope_and_unified_mla_kv_cache_update_impl,
    mutates_args=["q_pe", "k_pe"],
    fake_impl=fused_rope_and_unified_mla_kv_cache_update_fake,
)


class RopeReshapeKVCachePattern:
    """
    This pattern matches the following unfused inplace ops:
      q, k = rotary_embedding(positions, q, k, head_size, cos_sin_cache, is_neox)
      kv_cache_dummy = unified_kv_cache_update(k, v, layer_name)

    and replaces it with the fused inplace op:
      kv_cache_dummy = fused_rope_and_unified_kv_cache_update(
        q, k, v, positions, cos_sin_cache, is_neox, layer_name
      )
    """

    FUSED_OP = torch.ops.vllm.fused_rope_and_unified_kv_cache_update.default

    def __init__(
        self,
        layer: Attention,
        is_neox: bool,
    ) -> None:
        self.layer_name = layer.layer_name
        self.num_heads = layer.num_heads
        self.num_kv_heads = layer.num_kv_heads
        self.head_size = layer.head_size
        self.head_size_v = layer.head_size_v
        self.is_neox = is_neox

        self.q_size = self.num_heads * self.head_size
        self.k_size = self.num_kv_heads * self.head_size
        self.v_size = self.num_kv_heads * self.head_size_v

        self.rope_matcher = MatcherRotaryEmbedding(
            is_neox=self.is_neox,
            head_size=self.head_size,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
        )

    def get_inputs(self) -> list[torch.Tensor]:
        # Sample inputs to help pattern tracing
        T = 5
        L = 4096
        qkv = empty_bf16(T, self.q_size + self.k_size + self.v_size)
        positions = empty_i64(T)
        cos_sin_cache = empty_bf16(L, self.head_size)
        return [
            qkv,
            positions,
            cos_sin_cache,
        ]

    def register(self, pm_pass: PatternMatcherPass) -> None:
        def pattern(
            qkv: torch.Tensor,
            positions: torch.Tensor,
            cos_sin_cache: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            q, k, v = qkv.split([self.q_size, self.k_size, self.v_size], dim=-1)
            q, k = self.rope_matcher(positions, q, k, cos_sin_cache)
            q = q.view(-1, self.num_heads, self.head_size)
            k = k.view(-1, self.num_kv_heads, self.head_size)
            v = v.view(-1, self.num_kv_heads, self.head_size_v)
            dummy = torch.ops.vllm.unified_kv_cache_update(k, v, self.layer_name)
            return dummy, q, k, v

        def replacement(
            qkv: torch.Tensor,
            positions: torch.Tensor,
            cos_sin_cache: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            q, k, v = qkv.split([self.q_size, self.k_size, self.v_size], dim=-1)
            q = q.view(-1, self.num_heads, self.head_size)
            k = k.view(-1, self.num_kv_heads, self.head_size)
            v = v.view(-1, self.num_kv_heads, self.head_size_v)
            results = auto_functionalized(
                self.FUSED_OP,
                query=q,
                key=k,
                value=v,
                positions=positions,
                cos_sin_cache=cos_sin_cache,
                is_neox=self.is_neox,
                layer_name=self.layer_name,
            )
            return results[0], results[1], results[2], v

        # NOTE: use view_to_reshape to unify view/reshape to simplify
        # pattern and increase matching opportunities
        def fwd_and_view_to_reshape(*args, **kwargs) -> fx.GraphModule:
            gm = pm.fwd_only(*args, **kwargs)
            view_to_reshape(gm)
            return gm

        pm.register_replacement(
            pattern, replacement, self.get_inputs(), fwd_and_view_to_reshape, pm_pass
        )


class RopeMLAKVCachePattern:
    """
    This pass fuses the rotary embedding and KV cache update operations for MLA.
    Note: this only works when FlashInfer RoPE is enabled.
    """

    FUSED_OP = torch.ops.vllm.fused_rope_and_unified_mla_kv_cache_update.default

    def __init__(
        self,
        layer: MLAAttention,
        is_neox: bool,
    ) -> None:
        self.layer_name = layer.layer_name
        self.num_heads = layer.num_heads
        self.kv_lora_rank = layer.kv_lora_rank
        self.qk_rope_head_dim = layer.qk_rope_head_dim
        self.kv_cache_dtype = layer.kv_cache_dtype
        self.is_neox = is_neox

        self.rope_matcher = MatcherRotaryEmbedding(
            is_neox=self.is_neox,
            head_size=self.qk_rope_head_dim,
            num_heads=self.num_heads,
            num_kv_heads=1,
            use_flashinfer=True,
        )

    def get_inputs(self) -> list[torch.Tensor]:
        T = 5
        L = 4096
        q_pe = empty_bf16(T, self.num_heads, self.qk_rope_head_dim)
        k_pe_2d = empty_bf16(T, self.qk_rope_head_dim)
        k_c_and_k_pe = empty_bf16(T, self.kv_lora_rank + self.qk_rope_head_dim)
        kv_c = empty_bf16(T, self.kv_lora_rank)
        positions = empty_i64(T)
        cos_sin_cache = empty_bf16(L, self.qk_rope_head_dim)
        k_scale = torch.empty(1, dtype=torch.float32, device=q_pe.device)
        return [q_pe, k_pe_2d, k_c_and_k_pe, kv_c, positions, cos_sin_cache, k_scale]

    def register(self, pm_pass: PatternMatcherPass) -> None:
        def pattern(
            q_pe: torch.Tensor,
            k_pe_2d: torch.Tensor,
            k_c_and_k_pe: torch.Tensor,
            kv_c: torch.Tensor,
            positions: torch.Tensor,
            cos_sin_cache: torch.Tensor,
            k_scale: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            k_pe = k_pe_2d.unsqueeze(1)
            q_pe, k_pe = self.rope_matcher(positions, q_pe, k_pe, cos_sin_cache)
            assert k_pe is not None
            k_pe_2d = k_pe.squeeze(1)
            k_c_and_k_pe = k_c_and_k_pe.slice_scatter(
                k_pe_2d,
                dim=1,
                start=self.kv_lora_rank,
                end=self.kv_lora_rank + self.qk_rope_head_dim,
            )
            _, k_pe_2d = k_c_and_k_pe.split(
                [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
            )
            k_pe = k_pe_2d.unsqueeze(1)
            dummy = torch.ops.vllm.unified_mla_kv_cache_update(
                kv_c, k_pe, self.layer_name, self.kv_cache_dtype, k_scale
            )
            return dummy, q_pe, k_pe

        def replacement(
            q_pe: torch.Tensor,
            k_pe_2d: torch.Tensor,
            k_c_and_k_pe: torch.Tensor,
            kv_c: torch.Tensor,
            positions: torch.Tensor,
            cos_sin_cache: torch.Tensor,
            k_scale: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            del k_c_and_k_pe
            k_pe = k_pe_2d.unsqueeze(1)
            results = auto_functionalized(
                self.FUSED_OP,
                positions=positions,
                q_pe=q_pe,
                k_pe=k_pe,
                kv_c=kv_c,
                cos_sin_cache=cos_sin_cache,
                is_neox=self.is_neox,
                layer_name=self.layer_name,
                kv_cache_dtype=self.kv_cache_dtype,
                k_scale=k_scale,
            )
            return results[0], results[1], results[2]

        def fwd_and_view_to_reshape(*args, **kwargs) -> fx.GraphModule:
            gm = pm.fwd_only(*args, **kwargs)
            view_to_reshape(gm)
            return gm

        pm.register_replacement(
            pattern, replacement, self.get_inputs(), fwd_and_view_to_reshape, pm_pass
        )


class RopeKVCacheFusionPass(VllmPatternMatcherPass):
    """
    This pass fuses the rotary embedding and KV cache update operations
    into a single fused kernel if available.

    It uses the pattern matcher and matches each layer manually, as strings
    cannot be wildcarded. This also lets us check support on attention layers
    upon registration instead of during pattern matching.

    This fusion eliminates the need for separate kernel launches and
    intermediate memory operations between the RoPE and cache update steps.
    """

    @enable_fake_mode
    def __init__(self, config: VllmConfig) -> None:
        super().__init__(config)

        self.patterns: PatternMatcherPass = PatternMatcherPass(
            pass_name="rope_kv_cache_fusion_pass"
        )

        cc = config.compilation_config
        self.max_token_num = cc.pass_config.rope_kvcache_fusion_max_token_num

        attn_layers = get_layers_from_vllm_config(config, Attention)
        for _, layer in attn_layers.items():
            if layer.impl.fused_rope_kvcache_supported():
                for is_neox in [True, False]:
                    RopeReshapeKVCachePattern(
                        layer=layer,
                        is_neox=is_neox,
                    ).register(self.patterns)

        mla_layers = get_layers_from_vllm_config(config, MLAAttention)
        for _, layer in mla_layers.items():
            for is_neox in [True, False]:
                RopeMLAKVCachePattern(
                    layer=layer,
                    is_neox=is_neox,
                ).register(self.patterns)

        self.dump_patterns(config, self.patterns)

    @VllmInductorPass.time_and_log
    def __call__(self, graph: fx.Graph) -> None:
        self.matched_count = self.patterns.apply(graph)
        logger.debug("Replaced %s patterns", self.matched_count)

    def is_applicable_for_range(self, compile_range: Range) -> bool:
        # This pass works best for the small-batch decode setting.
        # For large-batch e.g. prefill, it is better to use two separate kernels
        # since they are compute bound and the fused kernels require further tuning.
        return compile_range.end <= self.max_token_num

    def uuid(self) -> str:
        return VllmInductorPass.hash_source(
            self,
            RopeReshapeKVCachePattern,
            RopeMLAKVCachePattern,
        )
