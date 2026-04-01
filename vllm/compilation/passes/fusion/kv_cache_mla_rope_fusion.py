# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch
import torch._inductor.pattern_matcher as pm
from torch import fx
from torch._higher_order_ops.auto_functionalize import auto_functionalized
from torch._inductor.fx_passes.post_grad import view_to_reshape
from torch._inductor.pattern_matcher import PatternMatcherPass

import vllm._custom_ops as ops
from vllm.config import VllmConfig, get_layers_from_vllm_config
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.attention import Attention, MLAAttention
from vllm.model_executor.layers.rotary_embedding import RotaryEmbedding
from vllm.utils.torch_utils import direct_register_custom_op

from ..inductor_pass import enable_fake_mode
from ..vllm_inductor_pass import VllmInductorPass, VllmPatternMatcherPass
from .matcher_utils import MatcherDeepseekScalingRotaryEmbedding, MatcherRotaryEmbedding
from .rms_quant_fusion import empty_bf16, empty_i64

logger = init_logger(__name__)

# aten.slice / slice_scatter use this for "to end" on post-grad graphs.
_ATEN_SLICE_TO_END: int = 9223372036854775807


def fused_concat_and_cache_mla_rope_impl(
    positions: torch.Tensor,
    q_pe: torch.Tensor,
    k_pe: torch.Tensor,
    kv_c: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    is_neox: bool,
    kv_cache_dtype: str,
    kv_cache_scale: torch.Tensor,
    layer_name: str,
) -> torch.Tensor:
    forward_context = get_forward_context()
    attn_layer = forward_context.no_compile_layers[layer_name]
    kv_cache = attn_layer.kv_cache[0]
    slot_mapping = forward_context.slot_mapping
    assert isinstance(slot_mapping, dict), (
        f"Expected slot_mapping to be a dict, got {type(slot_mapping)}. "
    )
    layer_slot_mapping = slot_mapping.get(layer_name)
    ops.concat_and_cache_mla_rope_fused(
        positions,
        q_pe,
        k_pe,
        kv_c,
        cos_sin_cache,
        is_neox,
        layer_slot_mapping,
        kv_cache,
        kv_cache_dtype,
        kv_cache_scale,
        layer_slot_mapping is not None,
    )
    return torch.empty(0, device=kv_c.device, dtype=kv_c.dtype)


def fused_concat_and_cache_mla_rope_fake(
    positions: torch.Tensor,
    q_pe: torch.Tensor,
    k_pe: torch.Tensor,
    kv_c: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    is_neox: bool,
    kv_cache_dtype: str,
    kv_cache_scale: torch.Tensor,
    layer_name: str,
) -> torch.Tensor:
    return torch.empty(0, dtype=kv_c.dtype, device=kv_c.device)


direct_register_custom_op(
    op_name="fused_concat_and_cache_mla_rope",
    op_func=fused_concat_and_cache_mla_rope_impl,
    fake_impl=fused_concat_and_cache_mla_rope_fake,
    mutates_args=["q_pe", "k_pe"],
)


class KVCacheMLARoPEFusionPattern:
    FUSED_OP = torch.ops.vllm.fused_concat_and_cache_mla_rope.default

    def __init__(
        self,
        layer: Attention,
        is_neox: bool,
        use_flashinfer: bool = False,
    ) -> None:
        self.layer_name = layer.layer_name
        self.kv_cache_dtype = layer.kv_cache_dtype
        self.num_heads = layer.num_heads
        self.num_kv_heads = layer.num_kv_heads
        self.kv_lora_rank = layer.kv_lora_rank
        self.qk_rope_head_dim = layer.qk_rope_head_dim
        self.is_neox = is_neox
        self.use_flashinfer = use_flashinfer

        self.rope_matcher = MatcherRotaryEmbedding(
            is_neox=self.is_neox,
            head_size=self.qk_rope_head_dim,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            use_flashinfer=self.use_flashinfer,
        )

    def get_inputs(self) -> list[torch.Tensor]:
        # Sample inputs to help pattern tracing
        T = 5
        L = 163840
        q = empty_bf16(T, self.num_heads, self.qk_rope_head_dim)
        k_pe = empty_bf16(T, 1, self.qk_rope_head_dim)
        kv_c_normed = empty_bf16(T, self.kv_lora_rank)
        cos_sin_cache = empty_bf16(L, self.qk_rope_head_dim)
        mm = empty_bf16(T, self.qk_rope_head_dim + self.kv_lora_rank)
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
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            query, key = self.rope_matcher(positions, q, k_pe, cos_sin_cache)
            k = key.squeeze(1)
            scatter = torch.ops.aten.slice_scatter.default(
                mm, k, 1, self.kv_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim
            )
            _, k2 = torch.ops.aten.split_with_sizes.default(
                scatter, [self.kv_lora_rank, self.qk_rope_head_dim], -1
            )
            k3 = k2.unsqueeze(1)

            dummy = torch.ops.vllm.unified_mla_kv_cache_update(
                kv_c_normed, k3, self.layer_name, self.kv_cache_dtype, k_scale
            )
            return dummy, query, k3

        def replacement(
            q: torch.Tensor,
            k_pe: torch.Tensor,
            kv_c_normed: torch.Tensor,
            mm: torch.Tensor,
            positions: torch.Tensor,
            cos_sin_cache: torch.Tensor,
            k_scale: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            res = auto_functionalized(
                self.FUSED_OP,
                positions=positions,
                q_pe=q,
                k_pe=k_pe,
                kv_c=kv_c_normed,
                cos_sin_cache=cos_sin_cache.to(q.dtype),
                is_neox=self.is_neox,
                kv_cache_dtype=self.kv_cache_dtype,
                kv_cache_scale=k_scale,
                layer_name=self.layer_name,
            )
            return res[0], res[1], res[2]

        # NOTE: use view_to_reshape to unify view/reshape to simplify
        # pattern and increase matching opportunities
        def fwd_and_view_to_reshape(*args, **kwargs) -> fx.GraphModule:
            gm = pm.fwd_only(*args, **kwargs)
            view_to_reshape(gm)
            return gm

        pm.register_replacement(
            pattern, replacement, self.get_inputs(), fwd_and_view_to_reshape, pm_pass
        )


class KVCacheMLARoPEDeepseekScalingFusionPattern:
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
        self.kv_lora_rank = layer.kv_lora_rank
        self.qk_rope_head_dim = layer.qk_rope_head_dim
        self.qk_head_dim = layer.qk_head_dim
        self.qk_nope_head_dim = layer.qk_nope_head_dim
        self.is_neox = is_neox

        self.rope_matcher = MatcherDeepseekScalingRotaryEmbedding(
            is_neox=self.is_neox,
            head_size=self.qk_rope_head_dim,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
        )

    def get_inputs(self) -> list[torch.Tensor]:
        T = 5
        L = 163840
        k_pe = empty_bf16(T, self.qk_rope_head_dim)
        kv_c_normed = empty_bf16(T, self.kv_lora_rank)
        mm = empty_bf16(
            T, self.num_heads * (self.qk_nope_head_dim + self.qk_rope_head_dim)
        )
        cos_sin_cache = empty_bf16(L, self.qk_rope_head_dim)
        positions = empty_i64(T)
        k_scale = torch.empty(0, device=k_pe.device, dtype=torch.float32)
        return [
            k_pe,
            kv_c_normed,
            mm,
            positions,
            cos_sin_cache,
            k_scale,
        ]

    def register(self, pm_pass: PatternMatcherPass) -> None:
        def pattern(
            k_pe: torch.Tensor,
            kv_c_normed: torch.Tensor,
            mm: torch.Tensor,
            positions: torch.Tensor,
            cos_sin_cache: torch.Tensor,
            k_scale: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            h = self.qk_nope_head_dim + self.qk_rope_head_dim
            v1 = mm.reshape(-1, self.num_heads, h)
            v2 = mm.reshape(-1, self.num_heads, h)
            s_r = torch.ops.aten.slice.Tensor(
                v2, 2, self.qk_nope_head_dim, _ATEN_SLICE_TO_END
            )
            s_w = torch.ops.aten.slice.Tensor(
                v2, 2, self.qk_nope_head_dim, _ATEN_SLICE_TO_END
            )
            query, key = self.rope_matcher(positions, s_r, k_pe, cos_sin_cache)
            query_rope = query[..., : self.qk_rope_head_dim]
            copy_out = torch.ops.aten.copy.default(s_w, query_rope)
            q_full = torch.ops.aten.slice_scatter.default(
                v1, copy_out, 2, self.qk_nope_head_dim, _ATEN_SLICE_TO_END
            )
            dummy = torch.ops.vllm.unified_mla_kv_cache_update(
                kv_c_normed,
                key,
                self.layer_name,
                self.kv_cache_dtype,
                k_scale,
            )
            return dummy, q_full, key

        def replacement(
            k_pe: torch.Tensor,
            kv_c_normed: torch.Tensor,
            mm: torch.Tensor,
            positions: torch.Tensor,
            cos_sin_cache: torch.Tensor,
            k_scale: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            h = self.qk_nope_head_dim + self.qk_rope_head_dim
            v2 = mm.reshape(-1, self.num_heads, h)
            q_pe_slice = v2[:, :, self.qk_nope_head_dim :]
            res = self.FUSED_OP(
                positions=positions,
                q_pe=q_pe_slice,
                k_pe=k_pe,
                kv_c=kv_c_normed,
                cos_sin_cache=cos_sin_cache.to(mm.dtype),
                is_neox=self.is_neox,
                kv_cache_dtype=self.kv_cache_dtype,
                kv_cache_scale=k_scale,
                layer_name=self.layer_name,
            )
            return res, v2, k_pe

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
                if RotaryEmbedding.enabled():
                    for use_flashinfer in [False, True]:
                        KVCacheMLARoPEFusionPattern(
                            layer,
                            is_neox,
                            use_flashinfer,
                        ).register(self.patterns)
                else:
                    KVCacheMLARoPEFusionPattern(
                        layer,
                        is_neox,
                    ).register(self.patterns)

            KVCacheMLARoPEDeepseekScalingFusionPattern(
                layer,
                is_neox=False,
            ).register(self.patterns)

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
