# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch
from torch import fx
from torch._higher_order_ops import auto_functionalized
from torch._inductor.fx_passes.post_grad import view_to_reshape
from torch._inductor.pattern_matcher import PatternMatcherPass
import torch._inductor.pattern_matcher as pm

from vllm.compilation.passes.fusion.matcher_utils import MatcherCustomOp

import vllm._custom_ops as ops

from vllm.config import VllmConfig, get_layers_from_vllm_config
from vllm.model_executor.layers.attention import MLAAttention, Attention

from vllm.logger import init_logger
from vllm.utils.torch_utils import direct_register_custom_op
from vllm.model_executor.layers.rotary_embedding.common import rotate_gptj, rotate_neox

from .matcher_utils import MatcherRotaryEmbedding, MatcherDeepseekScalingRotaryEmbedding
from .rms_quant_fusion import empty_bf16, empty_i64
from ..vllm_inductor_pass import VllmInductorPass, VllmPatternMatcherPass
from ..inductor_pass import enable_fake_mode

from vllm.model_executor.layers.attention.attention import get_attention_context

logger = init_logger(__name__)

def fused_concat_and_cache_mla_rope_impl(
    positions: torch.Tensor,
    q_pe: torch.Tensor,
    k_pe: torch.Tensor,
    kv_c: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    is_neox: bool,
    kv_cache_dtype: str,
    kv_cache_scale: torch.Tensor) -> None:
        _, attn_layer, kv_cache, layer_slot_mapping = get_attention_context(layer_name)
        ops.concat_and_cache_mla_rope_fused(
            positions, q_pe, k_pe, kv_c, cos_sin_cache, is_neox, layer_slot_mapping,
            kv_cache, kv_cache_dtype, kv_cache_scale)
        return torch.empty(0, device=kv_cache.device, dtype=kv_cache.dtype)

def fused_concat_and_cache_mla_rope_fake(
    positions: torch.Tensor,
    q_pe: torch.Tensor,
    k_pe: torch.Tensor,
    kv_c: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    is_neox: bool,
    kv_cache_dtype: str,
    kv_cache_scale: torch.Tensor) -> None:
        return torch.empty(0, device=kv_c.device, dtype=kv_c.dtype)

direct_register_custom_op(
    op_name="fused_concat_and_cache_mla_rope",
    op_func=fused_concat_and_cache_mla_rope_impl,
    fake_impl=fused_concat_and_cache_mla_rope_fake,
    mutates_args=["q_pe", "k_pe"],
)

class MLAKVCacheMatcher(MatcherCustomOp):
    def __init__(self, layer: Attention) -> None:
        self.layer_name = layer.layer_name
        self.kv_cache_dtype = layer.kv_cache_dtype
        self.kv_lora_rank = layer.kv_lora_rank
        self.qk_rope_head_dim = layer.qk_rope_head_dim
        super().__init__(enabled=True)

    def forward_custom(
            self,
            k_pe: torch.Tensor,
            kv_c_normed: torch.Tensor,
            k_scale: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            #TODO q = q[..., self.qk_nope_head_dim :]
            ### q, k_pe = self.rope_matcher(positions, q, k_pe, cos_sin_cache)
            # print("sizes:", self.q_size, self.k_size, self.v_size, self.head_size, self.num_heads, self.num_kv_heads)
            # print("pattern:", kv_c_normed.shape, k_pe.shape, self.layer_name, self.kv_cache_dtype, k_scale.shape)
            dummy = torch.ops.vllm.unified_mla_kv_cache_update(
                kv_c_normed, k_pe, self.layer_name, self.kv_cache_dtype, k_scale)
            # dummy = auto_functionalized(
            #     torch.ops.vllm.unified_mla_kv_cache_update.default,
            #     kv_c_normed=kv_c_normed,
            #     k_pe=k_pe,
            #     layer_name=self.layer_name,
            #     kv_cache_dtype=self.kv_cache_dtype,
            #     k_scale=k_scale,
            # )
            return dummy

    def forward_native(
            self,
            k_pe: torch.Tensor,
            kv_c_normed: torch.Tensor,
            k_scale: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            return self.forward_custom(k_pe, kv_c_normed, k_scale)

    def inputs(self) -> list[torch.Tensor]:
        kv_c_normed =  empty_bf16(5, self.kv_lora_rank)
        k_pe = self.empty_bf16(5, self.qk_rope_head_dim)
        k_scale = self.empty(0)
        return [kv_c_normed, k_pe, k_scale]

class KVCacheMLARoPEFusionPattern:
    FUSED_OP = torch.ops.vllm.fused_concat_and_cache_mla_rope.default

    def __init__(
        self,
        layer: Attention,
        is_neox: bool,
    ) -> None:
        self.layer_name = layer.layer_name
        self.kv_cache_dtype = layer.kv_cache_dtype
        # self.k_scale = layer._k_scale
        self.num_heads = layer.num_heads
        self.num_kv_heads = layer.num_kv_heads
        self.head_size = layer.head_size
        self.kv_lora_rank = layer.kv_lora_rank
        self.qk_rope_head_dim = layer.qk_rope_head_dim
        # self.head_size_v = layer.head_size_v
        self.is_neox = is_neox

        self.q_size = self.num_heads * self.head_size
        self.k_size = self.num_kv_heads * self.head_size
        # self.v_size = self.num_kv_heads * self.head_size_v
        self.v_size = self.num_kv_heads * self.head_size

        # print("create pattern with", layer.layer_name, self.kv_cache_dtype)

        self.rope_matcher = MatcherDeepseekScalingRotaryEmbedding(
            is_neox=self.is_neox,
            head_size=self.head_size,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
        )
        self.kv_cache_matcher = MLAKVCacheMatcher(layer)

    def get_inputs(self) -> list[torch.Tensor]:
        # # # output: bf16[s72, 576]

        # # # q: bf16[s72, 16, 64]
        # # # k_pe: bf16[s72, 64]
        # # # kv_c: bf16[s72, 512]
        # Sample inputs to help pattern tracing
        T = 5
        L = 163840
        # q = empty_bf16(T, 16, 64)
        k_pe = empty_bf16(T, 1, 64) # ok
        kv_c_normed = empty_bf16(T, 512) # ok
        cos_sin_cache = empty_bf16(L, 64) # ok
        # q = empty_bf16(T, self.kv_lora_rank)
        # k_pe = empty_bf16(T, 1, 64) #self.qk_rope_head_dim)
        # kv_c_normed = empty_bf16(T, 512) #self.kv_lora_rank)
        positions = empty_i64(T) # ok
        # cos_sin_cache = empty_bf16(L, self.head_size)
        k_scale = torch.empty(0, device=k_pe.device, dtype=torch.float32)
        # print("input shapes:", q.shape, k_pe.shape, kv_c_normed.shape, k_scale.shape)
        # print("input dtypes:", q.dtype, k_pe.dtype, kv_c_normed.dtype, k_scale.dtype)
        return [
            # q,
            k_pe, kv_c_normed,
            # positions,
            # cos_sin_cache,
            k_scale,
        ]

    def register(self, pm_pass: PatternMatcherPass) -> None:
        def pattern(
            # q: torch.Tensor,
            k_pe: torch.Tensor,
            kv_c_normed: torch.Tensor,
            # positions: torch.Tensor,
            # cos_sin_cache: torch.Tensor,
            k_scale: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            #TODO q = q[..., self.qk_nope_head_dim :]
            # k_pe = self.rope_matcher(positions, k_pe, cos_sin_cache)
            # print("sizes:", self.q_size, self.k_size, self.v_size, self.head_size, self.num_heads, self.num_kv_heads)
            # print("pattern:", kv_c_normed.shape, k_pe.shape, self.layer_name, self.kv_cache_dtype, k_scale.shape)

            # # # cos_sin = cos_sin_cache[
            # # #     positions
            # # # ]
            # # # cos, sin = cos_sin.chunk(2, dim=-1)
            # if is_neox_style:
            #     # NOTE(woosuk): Here we assume that the positions tensor has the
            #     # shape [batch_size, seq_len].
            #     cos = cos.repeat(1, 1, 2).unsqueeze(-2)
            #     sin = sin.repeat(1, 1, 2).unsqueeze(-2)
            # else:
            # cos = cos.repeat_interleave(2, dim=-1).unsqueeze(-2)
            # sin = sin.repeat_interleave(2, dim=-1).unsqueeze(-2)
            # # cos = cos.unsqueeze(2).expand(
            # #     (cos.shape[0], cos.shape[1], 2)).clone(
            # #         memory_format=torch.contiguous_format).reshape(
            # #             (cos.shape[0], cos.shape[1] * 2)).unsqueeze(-2)
            # # sin = sin.unsqueeze(2).expand(
            # #     (sin.shape[0], sin.shape[1], 2)).clone(
            # #         memory_format=torch.contiguous_format).reshape(
            # #             (sin.shape[0], sin.shape[1] * 2)).unsqueeze(-2)

            # # # # # rotate_fn = rotate_neox if is_neox_style else rotate_gptj

            # # key_chunked_1, key_chunked_2 = k_pe.chunk(2, dim=2)
            # # key_chunked_1_unsqueeze = key_chunked_1.unsqueeze(3)
            # # key_chunked_2_neg = key_chunked_2.neg()
            # # key_chunked_2_neg_unsqueeze = key_chunked_2_neg.unsqueeze(3)
            # # key_cat = torch.cat(
            # # [key_chunked_2_neg_unsqueeze,
            # # key_chunked_1_unsqueeze], -1)
            # # key_rotated = key_cat.reshape(
            # #     (k_pe.shape[0], 1, 64))
            
            # query_rot = query * cos + rotate_fn(query) * sin
            # # # key_rot = k_pe * cos + key_rotated * sin
            
            dummy = torch.ops.vllm.unified_mla_kv_cache_update(
                kv_c_normed, k_pe, self.layer_name, self.kv_cache_dtype, k_scale)
            return dummy, k_pe, kv_c_normed, k_scale

        def replacement(
            # q: torch.Tensor,
            k_pe: torch.Tensor,
            kv_c_normed: torch.Tensor,
            # positions: torch.Tensor,
            # cos_sin_cache: torch.Tensor,
            k_scale: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            print("replacement called")

            # results = auto_functionalized(
            #     self.FUSED_OP,
            #     positions=positions,
            #     q_pe=q,
            #     k_pe=k_pe,
            #     kv_c=kv_c_normed,
            #     cos_sin_cache=cos_sin_cache,
            #     is_neox=self.is_neox,
            #     kv_cache_dtype=self.kv_cache_dtype,
            #     kv_cache_scale=k_scale,
            # )
            # return results[0], results[1], results[2], kv_c_normed
            return torch.empty(0, device=kv_c_normed.device, dtype=kv_c_normed.dtype), k_pe, kv_c_normed, k_scale

        # NOTE: use view_to_reshape to unify view/reshape to simplify
        # pattern and increase matching opportunities
        def fwd_and_view_to_reshape(*args, **kwargs) -> fx.GraphModule:
            gm = pm.fwd_only(*args, **kwargs)
            view_to_reshape(gm)
            return gm

        pm.register_replacement(
            pattern, replacement, self.get_inputs(), fwd_and_view_to_reshape, pm_pass
        )


class KVCacheMLARoPEFusionPass(VllmPatternMatcherPass):
    @enable_fake_mode
    def __init__(self, config: VllmConfig) -> None:
        super().__init__(config)

        self.patterns: PatternMatcherPass = PatternMatcherPass(
            pass_name="rms_mla_kv_update_fusion_pass"
        )
        attn_layers = get_layers_from_vllm_config(config, MLAAttention)

        print("====RUN MLA KV CACHE UPDATE ROPE FUSION=====")
        for _, layer in attn_layers.items():
            # print("layer vars:", vars(layer))
            print(f"registering KVCacheMLARoPEFusionPattern for {layer.layer_name}")
            for is_neox in [False]: #, False]:
                KVCacheMLARoPEFusionPattern(layer, is_neox).register(
                        self.patterns)

        self.dump_patterns(config, self.patterns)

    @VllmInductorPass.time_and_log
    def __call__(self, graph: fx.Graph) -> None:
        self.matched_count = self.patterns.apply(graph)
        logger.debug("Replaced %s patterns", self.matched_count)
        print("matched count:", self.matched_count)

    def uuid(self) -> str:
        return self.hash_source(
            self,
            KVCacheMLARoPEFusionPattern,
        )