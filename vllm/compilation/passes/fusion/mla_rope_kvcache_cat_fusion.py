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
from vllm.model_executor.layers.attention import MLAAttention
from vllm.model_executor.layers.attention.attention import get_attention_context
from vllm.model_executor.layers.rotary_embedding import RotaryEmbedding
from vllm.utils.torch_utils import (
    _USE_LAYERNAME,
    LayerNameType,
    _encode_layer_name,
    _resolve_layer_name,
    direct_register_custom_op,
)

from ..inductor_pass import enable_fake_mode
from ..vllm_inductor_pass import VllmInductorPass, VllmPatternMatcherPass
from .matcher_utils import MatcherDeepseekScalingRotaryEmbedding, MatcherRotaryEmbedding
from .rms_quant_fusion import empty_bf16, empty_i64

logger = init_logger(__name__)

def fused_concat_and_cache_mla_rope_impl(
    positions: torch.Tensor,
    q_pe: torch.Tensor,
    k_pe: torch.Tensor,
    kv_c: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    is_neox: bool,
    kv_cache_dtype: str,
    kv_cache_scale: torch.Tensor,
    layer_name: LayerNameType,
) -> torch.Tensor:
    forward_context = get_forward_context()
    layer_name = _resolve_layer_name(layer_name)
    attn_layer = forward_context.no_compile_layers[layer_name]
    kv_cache = attn_layer.kv_cache
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
    layer_name: LayerNameType,
) -> torch.Tensor:
    return torch.empty(0, dtype=kv_c.dtype, device=kv_c.device)


direct_register_custom_op(
    op_name="fused_concat_and_cache_mla_rope",
    op_func=fused_concat_and_cache_mla_rope_impl,
    fake_impl=fused_concat_and_cache_mla_rope_fake,
    mutates_args=["q_pe", "k_pe"],
)


class MLARoPEKVCacheCatPattern:
    FUSED_OP = torch.ops.vllm.fused_concat_and_cache_mla_rope.default

    def __init__(
        self,
        layer: MLAAttention,
        is_neox: bool,
        use_flashinfer: bool = False,
        use_deepseek_scaling: bool = False,
    ) -> None:
        self.layer_name = layer.layer_name
        self.kv_cache_dtype = layer.kv_cache_dtype
        self.num_heads = layer.num_heads
        self.num_kv_heads = layer.num_kv_heads
        self.kv_lora_rank = layer.kv_lora_rank
        self.qk_rope_head_dim = layer.qk_rope_head_dim
        self.is_neox = is_neox
        self.use_flashinfer = use_flashinfer

        if use_deepseek_scaling:
            self.rope_matcher = MatcherDeepseekScalingRotaryEmbedding(
                is_neox=self.is_neox,
                head_size=self.qk_rope_head_dim,
                num_heads=self.num_heads,
                num_kv_heads=self.num_kv_heads,
                mla_mode=True,
            )
        else:
            self.rope_matcher = MatcherRotaryEmbedding(
                is_neox=self.is_neox,
                head_size=self.qk_rope_head_dim,
                num_heads=self.num_heads,
                num_kv_heads=self.num_kv_heads,
                use_flashinfer=self.use_flashinfer,
                mla_mode=True,
            )

    def get_inputs(self) -> list:
        # Sample inputs to help pattern tracing
        T = 5
        L = 4096
        q = empty_bf16(T, self.num_heads, self.qk_rope_head_dim)
        k_pe = empty_bf16(T, self.qk_rope_head_dim)
        kv_c_normed = empty_bf16(T, self.kv_lora_rank)
        cos_sin_cache = empty_bf16(L, self.qk_rope_head_dim)
        positions = empty_i64(T)
        k_scale = torch.empty(0, device=k_pe.device, dtype=torch.float32)
        inputs = [
            q,
            k_pe,
            kv_c_normed,
            positions,
            cos_sin_cache,
            k_scale,
        ]
        if _USE_LAYERNAME:
            inputs.append(_encode_layer_name(self.layer_name))
        return inputs

    def _mk_pattern_with_layer_name_input(self, _ln):
        """Pattern/replacement with layer_name as an explicit input."""

        def pattern(
            q: torch.Tensor,
            k_pe: torch.Tensor,
            kv_c_normed: torch.Tensor,
            positions: torch.Tensor,
            cos_sin_cache: torch.Tensor,
            k_scale: torch.Tensor,
            layer_name: LayerNameType,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            k_pe_unsqueezed = k_pe.unsqueeze(1)
            query, key = self.rope_matcher(
                positions, q, k_pe_unsqueezed, cos_sin_cache
            )
            dummy = torch.ops.vllm.unified_mla_kv_cache_update(
                kv_c_normed, key, layer_name, self.kv_cache_dtype, k_scale
            )
            return dummy, query, key

        def replacement(
            q: torch.Tensor,
            k_pe: torch.Tensor,
            kv_c_normed: torch.Tensor,
            positions: torch.Tensor,
            cos_sin_cache: torch.Tensor,
            k_scale: torch.Tensor,
            layer_name: LayerNameType,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            at = auto_functionalized(
                self.FUSED_OP,
                positions=positions,
                q_pe=q,
                k_pe=k_pe,
                kv_c=kv_c_normed,
                cos_sin_cache=cos_sin_cache,
                is_neox=self.is_neox,
                kv_cache_dtype=self.kv_cache_dtype,
                kv_cache_scale=k_scale,
                layer_name=layer_name,
            )
            dummy, q, k_pe_squeezed = at
            k_pe_unsqueezed_2 = k_pe_squeezed.unsqueeze(1)
            return dummy, q, k_pe_unsqueezed_2

        return pattern, replacement

    def _mk_pattern_with_layer_name_closure(self, _ln):
        """Pattern/replacement with layer_name as a closure constant."""

        def pattern(
            q: torch.Tensor,
            k_pe: torch.Tensor,
            kv_c_normed: torch.Tensor,
            positions: torch.Tensor,
            cos_sin_cache: torch.Tensor,
            k_scale: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            k_pe_unsqueezed = k_pe.unsqueeze(1)
            query, key = self.rope_matcher(
                positions, q, k_pe_unsqueezed, cos_sin_cache
            )
            dummy = torch.ops.vllm.unified_mla_kv_cache_update(
                kv_c_normed, key, _ln, self.kv_cache_dtype, k_scale
            )
            return dummy, query, key

        def replacement(
            q: torch.Tensor,
            k_pe: torch.Tensor,
            kv_c_normed: torch.Tensor,
            positions: torch.Tensor,
            cos_sin_cache: torch.Tensor,
            k_scale: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            at = auto_functionalized(
                self.FUSED_OP,
                positions=positions,
                q_pe=q,
                k_pe=k_pe,
                kv_c=kv_c_normed,
                cos_sin_cache=cos_sin_cache,
                is_neox=self.is_neox,
                kv_cache_dtype=self.kv_cache_dtype,
                kv_cache_scale=k_scale,
                layer_name=_ln,
            )
            dummy, q, k_pe_squeezed = at
            k_pe_unsqueezed_2 = k_pe_squeezed.unsqueeze(1)
            return dummy, q, k_pe_unsqueezed_2

        return pattern, replacement

    def register(self, pm_pass: PatternMatcherPass) -> None:
        _ln = _encode_layer_name(self.layer_name)

        if _USE_LAYERNAME:
            pattern, replacement = self._mk_pattern_with_layer_name_input(_ln)
        else:
            pattern, replacement = self._mk_pattern_with_layer_name_closure(_ln)

        # NOTE: use view_to_reshape to unify view/reshape to simplify
        # pattern and increase matching opportunities
        def fwd_and_view_to_reshape(*args, **kwargs) -> fx.GraphModule:
            gm = pm.fwd_only(*args, **kwargs)
            view_to_reshape(gm)
            return gm

        pm.register_replacement(
            pattern, replacement, self.get_inputs(), fwd_and_view_to_reshape, pm_pass
        )

class MLARoPEKVCacheCatFusionPass(VllmPatternMatcherPass):
    @enable_fake_mode
    def __init__(self, config: VllmConfig) -> None:
        super().__init__(config)

        self.patterns: PatternMatcherPass = PatternMatcherPass(
            pass_name="mla_rope_kv_cache_fusion_pass"
        )
        attn_layers = get_layers_from_vllm_config(config, MLAAttention)

        for _, layer in attn_layers.items():
            for is_neox in [False, True]:
                if RotaryEmbedding.enabled():
                    for use_flashinfer in [False, True]:
                        for use_deepseek_scaling in [False, True]:
                            MLARoPEKVCacheCatPattern(
                                layer,
                                is_neox,
                                use_flashinfer,
                                use_deepseek_scaling,
                            ).register(self.patterns)
                else:
                    MLARoPEKVCacheCatPattern(
                        layer,
                        is_neox,
                    ).register(self.patterns)
                    MLARoPEKVCacheCatPattern(
                        layer,
                        is_neox,
                        use_deepseek_scaling=True,
                    ).register(self.patterns)

            if _USE_LAYERNAME:
                break

        self.dump_patterns(config, self.patterns)

    @VllmInductorPass.time_and_log
    def __call__(self, graph: fx.Graph) -> None:
        self.matched_count = self.patterns.apply(graph)
        logger.debug("Replaced %s patterns", self.matched_count)

    def uuid(self) -> str:
        return self.hash_source(
            self,
            MLARoPEKVCacheCatPattern,
        )
