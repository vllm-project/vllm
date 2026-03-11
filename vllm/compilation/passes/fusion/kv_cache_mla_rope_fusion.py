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
from vllm.forward_context import ForwardContext, get_forward_context

logger = init_logger(__name__)

def fused_concat_and_cache_mla_rope_impl(
    dummy: torch.Tensor,
    positions: torch.Tensor,
    q_pe: torch.Tensor,
    k_pe: torch.Tensor,
    kv_c: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    is_neox: bool,
    kv_cache_dtype: str,
    kv_cache_scale: torch.Tensor,
    layer_name: str) -> None:
        forward_context = get_forward_context()
        attn_layer = forward_context.no_compile_layers[layer_name]
        kv_cache = attn_layer.kv_cache[forward_context.virtual_engine]
        layer_slot_mapping = forward_context.slot_mapping.get(layer_name)
        ops.concat_and_cache_mla_rope_fused(
            positions, q_pe, k_pe, kv_c, cos_sin_cache, is_neox, layer_slot_mapping,
            kv_cache, kv_cache_dtype, kv_cache_scale, layer_slot_mapping is not None)

def fused_concat_and_cache_mla_rope_fake(
    dummy: torch.Tensor,
    positions: torch.Tensor,
    q_pe: torch.Tensor,
    k_pe: torch.Tensor,
    kv_c: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    is_neox: bool,
    kv_cache_dtype: str,
    kv_cache_scale: torch.Tensor,
    layer_name: str) -> None:
        pass

direct_register_custom_op(
    op_name="fused_concat_and_cache_mla_rope",
    op_func=fused_concat_and_cache_mla_rope_impl,
    fake_impl=fused_concat_and_cache_mla_rope_fake,
    mutates_args=["dummy", "q_pe", "k_pe"],
)

class KVCacheMLARoPEFusionPattern:
    FUSED_OP = torch.ops.vllm.fused_concat_and_cache_mla_rope.default

    def __init__(
        self,
        layer: Attention,
        is_neox: bool,
    ) -> None:
        self.layer_name = layer.layer_name
        self.kv_cache_dtype = layer.kv_cache_dtype
        self.num_heads = layer.num_heads
        self.num_kv_heads = layer.num_kv_heads
        self.head_size = 64 #layer.head_size
        self.kv_lora_rank = layer.kv_lora_rank
        self.qk_rope_head_dim = layer.qk_rope_head_dim
        # self.head_size_v = layer.head_size_v
        self.is_neox = is_neox

        self.q_size = self.num_heads * self.head_size
        self.k_size = self.num_kv_heads * self.head_size
        # self.v_size = self.num_kv_heads * self.head_size_v
        self.v_size = self.num_kv_heads * self.head_size

        self.rope_matcher = MatcherRotaryEmbedding(
            is_neox=self.is_neox,
            head_size=self.head_size,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
        )

    def get_inputs(self) -> list[torch.Tensor]:
        # Sample inputs to help pattern tracing
        T = 5
        L = 163840
        q = empty_bf16(T, 16, 64)
        k_pe = empty_bf16(T, 1, 64)
        kv_c_normed = empty_bf16(T, 512)
        cos_sin_cache = empty_bf16(L, 64)
        mm = empty_bf16(T, 576)
        positions = empty_i64(T)
        k_scale = torch.empty(0, device=k_pe.device, dtype=torch.float32)
        return [
            q,
            k_pe,
            kv_c_normed,
            mm,
            positions,
            cos_sin_cache,
            k_scale,
        ]

    def register(self, pm_pass: PatternMatcherPass) -> None:
        def pattern(
            q: torch.Tensor,
            k_pe: torch.Tensor,
            kv_c_normed: torch.Tensor,
            mm: torch.Tensor,
            positions: torch.Tensor,
            cos_sin_cache: torch.Tensor,
            k_scale: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            query, key = self.rope_matcher(positions, q, k_pe, cos_sin_cache)
            k = key.squeeze(1)
            scatter = torch.ops.aten.slice_scatter.default(mm, k, 1, 512, 576)
            _, k2 = torch.ops.aten.split_with_sizes.default(scatter, [512, 64], -1)
            k3 = k2.unsqueeze(1)
            
            dummy = torch.ops.vllm.unified_mla_kv_cache_update(
                kv_c_normed, k3, self.layer_name, self.kv_cache_dtype, k_scale)
            return dummy, query, k3

        def replacement(
            q: torch.Tensor,
            k_pe: torch.Tensor,
            kv_c_normed: torch.Tensor,
            mm: torch.Tensor,
            positions: torch.Tensor,
            cos_sin_cache: torch.Tensor,
            k_scale: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            dummy = torch.empty(0, device=kv_c_normed.device, dtype=kv_c_normed.dtype)
            k_pe_squeezed = k_pe.squeeze(1)
            self.FUSED_OP(
                dummy=dummy,
                positions=positions,
                q_pe=q,
                k_pe=k_pe_squeezed,
                kv_c=kv_c_normed,
                cos_sin_cache=cos_sin_cache.to(q.dtype),
                is_neox=self.is_neox,
                kv_cache_dtype=self.kv_cache_dtype,
                kv_cache_scale=k_scale,
                layer_name=self.layer_name,
            )
            return dummy, q, k_pe_squeezed.unsqueeze(1)

            # TODO this is a fallback - delete when done
            torch.ops.vllm.flashinfer_rotary_embedding(
                positions=positions,
                query=q,
                key=k_pe,
                head_size=q.shape[2],
                cos_sin_cache=cos_sin_cache,
                is_neox=self.is_neox
            )
            k = k_pe.squeeze(1)
            scatter = torch.ops.aten.slice_scatter.default(mm, k, 1, 512, 576)
            _, k2 = torch.ops.aten.split_with_sizes.default(scatter, [512, 64], -1)
            k3 = k2.unsqueeze(1)
            dummy = torch.ops.vllm.unified_mla_kv_cache_update(
                kv_c_normed, k3, self.layer_name, self.kv_cache_dtype, k_scale)
            return dummy, q, k3


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

        for _, layer in attn_layers.items():
            for is_neox in [False, True]:
                KVCacheMLARoPEFusionPattern(layer, is_neox).register(
                        self.patterns)

        self.dump_patterns(config, self.patterns)

    @VllmInductorPass.time_and_log
    def __call__(self, graph: fx.Graph) -> None:
        self.matched_count = self.patterns.apply(graph)
        logger.debug("Replaced %s patterns", self.matched_count)

    def uuid(self) -> str:
        return self.hash_source(
            self,
            KVCacheMLARoPEFusionPattern,
        )
