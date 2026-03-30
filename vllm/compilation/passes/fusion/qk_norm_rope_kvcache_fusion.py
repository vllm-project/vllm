# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
import torch._inductor.pattern_matcher as pm
from torch import fx
from torch._higher_order_ops.auto_functionalize import auto_functionalized
from torch._inductor.fx_passes.post_grad import view_to_reshape
from torch._inductor.pattern_matcher import PatternMatcherPass

from vllm._aiter_ops import rocm_aiter_ops
from vllm.config import VllmConfig, get_layers_from_vllm_config
from vllm.config.utils import Range
from vllm.logger import init_logger
from vllm.model_executor.layers.attention.attention import (
    Attention,
    get_attention_context,
)
from vllm.model_executor.layers.rotary_embedding import RotaryEmbedding
from vllm.utils.torch_utils import direct_register_custom_op

from ..inductor_pass import enable_fake_mode
from ..vllm_inductor_pass import VllmInductorPass, VllmPatternMatcherPass
from .matcher_utils import MatcherRMSNorm, MatcherRotaryEmbedding
from .rms_quant_fusion import empty_bf16, empty_fp32, empty_i64

logger = init_logger(__name__)


# ---------------------------------------------------------------------------
# Custom op: fused QK-norm + RoPE + KV cache update
# ---------------------------------------------------------------------------

def fused_qk_norm_rope_and_unified_kv_cache_update_impl(
    q_out: torch.Tensor,
    k_out: torch.Tensor,
    qkv: torch.Tensor,
    positions: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    rms_norm_eps: float,
    cos_sin_cache: torch.Tensor,
    is_neox: bool,
    layer_name: str = "",
) -> torch.Tensor:
    _, attn_layer, kv_cache, layer_slot_mapping = get_attention_context(
        layer_name)
    if layer_slot_mapping is not None:
        attn_layer.impl.do_qk_norm_rope_kvcache_update(
            attn_layer,
            qkv,
            q_out,
            k_out,
            positions,
            q_weight,
            k_weight,
            rms_norm_eps,
            cos_sin_cache,
            is_neox,
            kv_cache,
            layer_slot_mapping,
        )

    return torch.empty(0, device=qkv.device, dtype=qkv.dtype)


def fused_qk_norm_rope_and_unified_kv_cache_update_fake(
    q_out: torch.Tensor,
    k_out: torch.Tensor,
    qkv: torch.Tensor,
    positions: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    rms_norm_eps: float,
    cos_sin_cache: torch.Tensor,
    is_neox: bool,
    layer_name: str = "",
) -> torch.Tensor:
    return torch.empty(0, device=qkv.device, dtype=qkv.dtype)


direct_register_custom_op(
    op_name="fused_qk_norm_rope_and_unified_kv_cache_update",
    op_func=fused_qk_norm_rope_and_unified_kv_cache_update_impl,
    mutates_args=["q_out", "k_out"],
    fake_impl=fused_qk_norm_rope_and_unified_kv_cache_update_fake,
)


# ---------------------------------------------------------------------------
# Pattern: QK-norm + RoPE + unified_kv_cache_update
# ---------------------------------------------------------------------------

class QkNormRopeKvCachePattern:
    """
    Match the unfused sequence:
      q, k, v = split(qkv, ...)
      q = rms_norm(q.view(heads), q_weight).view(flat)
      k = rms_norm(k.view(heads), k_weight).view(flat)
      q, k = rotary_embedding(positions, q, k, cos_sin_cache, is_neox)
      q = q.view(num_heads, head_dim)
      k = k.view(num_kv_heads, head_dim)
      v = v.view(num_kv_heads, head_dim)
      dummy = unified_kv_cache_update(k, v, layer_name)

    Replace with:
      q_out = empty(...)
      k_out = empty(...)
      dummy = fused_qk_norm_rope_and_unified_kv_cache_update(
          q_out, k_out, qkv, positions, q_weight, k_weight,
          eps, cos_sin_cache, is_neox, layer_name)
      v = split(qkv, ...)[2].view(num_kv_heads, head_dim)
    """

    FUSED_OP = (
        torch.ops.vllm.fused_qk_norm_rope_and_unified_kv_cache_update.default
    )

    def __init__(
        self,
        layer: Attention,
        eps: float,
        is_neox: bool,
        rope_flashinfer: bool = False,
        match_rocm_aiter: bool = False,
    ) -> None:
        self.layer_name = layer.layer_name
        self.num_heads = layer.num_heads
        self.num_kv_heads = layer.num_kv_heads
        self.head_size = layer.head_size
        self.head_size_v = layer.head_size_v
        self.eps = eps
        self.is_neox = is_neox
        self.rope_flashinfer = rope_flashinfer

        self.q_size = self.num_heads * self.head_size
        self.k_size = self.num_kv_heads * self.head_size
        self.v_size = self.num_kv_heads * self.head_size_v

        self.rmsnorm_matcher = MatcherRMSNorm(
            eps, match_rocm_aiter=match_rocm_aiter
        )
        self.rope_matcher = MatcherRotaryEmbedding(
            is_neox=is_neox,
            head_size=self.head_size,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            use_flashinfer=rope_flashinfer,
            match_rocm_aiter=match_rocm_aiter if match_rocm_aiter else None,
        )

    def get_inputs(self) -> list[torch.Tensor]:
        T = 5
        L = 4096
        qkv = empty_bf16(T, self.q_size + self.k_size + self.v_size)
        positions = empty_i64(T)
        q_weight = empty_bf16(1, self.head_size)
        k_weight = empty_bf16(1, self.head_size)
        if self.rope_flashinfer:
            cos_sin_cache = empty_fp32(L, self.head_size)
        else:
            cos_sin_cache = empty_bf16(L, self.head_size)
        return [qkv, positions, q_weight, k_weight, cos_sin_cache]

    def register(self, pm_pass: PatternMatcherPass) -> None:
        num_heads = self.num_heads
        num_kv_heads = self.num_kv_heads
        head_dim = self.head_size
        head_dim_v = self.head_size_v
        q_size = self.q_size
        k_size = self.k_size
        v_size = self.v_size
        layer_name = self.layer_name
        eps = self.eps
        is_neox = self.is_neox

        rmsnorm_matcher = self.rmsnorm_matcher
        rope_matcher = self.rope_matcher

        def pattern(
            qkv: torch.Tensor,
            positions: torch.Tensor,
            q_weight: torch.Tensor,
            k_weight: torch.Tensor,
            cos_sin_cache: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            q, k, v = qkv.split([q_size, k_size, v_size], dim=-1)

            q_by_head = q.view(
                *q.shape[:-1], q.shape[-1] // head_dim, head_dim
            )
            q_normed = rmsnorm_matcher(q_by_head, q_weight)
            q_flat = q_normed.view(q.shape)

            k_by_head = k.view(
                *k.shape[:-1], k.shape[-1] // head_dim, head_dim
            )
            k_normed = rmsnorm_matcher(k_by_head, k_weight)
            k_flat = k_normed.view(k.shape)

            q_rope, k_rope = rope_matcher(
                positions, q_flat, k_flat, cos_sin_cache
            )

            q_rope = q_rope.view(-1, num_heads, head_dim)
            k_rope = k_rope.view(-1, num_kv_heads, head_dim)
            v = v.view(-1, num_kv_heads, head_dim_v)
            dummy = torch.ops.vllm.unified_kv_cache_update(
                k_rope, v, layer_name
            )
            return dummy, q_rope, k_rope, v

        def replacement(
            qkv: torch.Tensor,
            positions: torch.Tensor,
            q_weight: torch.Tensor,
            k_weight: torch.Tensor,
            cos_sin_cache: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            q_out = torch.empty(
                qkv.shape[0], num_heads, head_dim,
                device=qkv.device, dtype=qkv.dtype,
            )
            k_out = torch.empty(
                qkv.shape[0], num_kv_heads, head_dim,
                device=qkv.device, dtype=qkv.dtype,
            )
            _, _, v = qkv.split([q_size, k_size, v_size], dim=-1)
            v = v.view(-1, num_kv_heads, head_dim_v)

            results = auto_functionalized(
                self.FUSED_OP,
                q_out=q_out,
                k_out=k_out,
                qkv=qkv,
                positions=positions,
                q_weight=q_weight,
                k_weight=k_weight,
                rms_norm_eps=eps,
                cos_sin_cache=cos_sin_cache,
                is_neox=is_neox,
                layer_name=layer_name,
            )

            # results[0] = dummy, results[1] = q_out, results[2] = k_out
            return results[0], results[1], results[2], v

        def fwd_and_view_to_reshape(*args, **kwargs) -> fx.GraphModule:
            gm = pm.fwd_only(*args, **kwargs)
            view_to_reshape(gm)
            return gm

        pm.register_replacement(
            pattern,
            replacement,
            self.get_inputs(),
            fwd_and_view_to_reshape,
            pm_pass,
        )


# ---------------------------------------------------------------------------
# Pass class
# ---------------------------------------------------------------------------

class QkNormRopeKvCacheFusionPass(VllmPatternMatcherPass):
    """
    Fuse QK-norm + RoPE + KV cache update into a single AITER HIP kernel.

    Supersedes both QKNormRoPEFusionPass and RopeKVCacheFusionPass for
    attention layers that support the combined operation, eliminating two
    separate kernel launches and the intermediate memory traffic.
    """

    @enable_fake_mode
    def __init__(self, config: VllmConfig) -> None:
        super().__init__(config)

        self.patterns: PatternMatcherPass = PatternMatcherPass(
            pass_name="qk_norm_rope_kvcache_fusion_pass"
        )

        cc = config.compilation_config
        self.max_token_num = cc.pass_config.rope_kvcache_fusion_max_token_num

        dtype = config.model_config.dtype
        if dtype not in (torch.bfloat16, torch.float16):
            logger.warning_once(
                "QK Norm+RoPE+KVCache fusion not enabled: "
                "unsupported dtype %s", dtype
            )
            return

        attn_layers = get_layers_from_vllm_config(config, Attention)

        aiter_enabled = rocm_aiter_ops.is_rmsnorm_enabled()
        aiter_variants = [False]
        if aiter_enabled:
            aiter_variants.append(True)

        for _, layer in attn_layers.items():
            if not layer.impl.fused_qk_norm_rope_kvcache_supported():
                continue
            for match_aiter in aiter_variants:
                for epsilon in [1e-5, 1e-6]:
                    for neox in [True, False]:
                        if RotaryEmbedding.enabled():
                            for rope_flashinfer in [False, True]:
                                QkNormRopeKvCachePattern(
                                    layer=layer,
                                    eps=epsilon,
                                    is_neox=neox,
                                    rope_flashinfer=rope_flashinfer,
                                    match_rocm_aiter=match_aiter,
                                ).register(self.patterns)
                        else:
                            QkNormRopeKvCachePattern(
                                layer=layer,
                                eps=epsilon,
                                is_neox=neox,
                                match_rocm_aiter=match_aiter,
                            ).register(self.patterns)

        self.dump_patterns(config, self.patterns)

    @VllmInductorPass.time_and_log
    def __call__(self, graph: fx.Graph) -> None:
        self.matched_count = self.patterns.apply(graph)
        logger.debug(
            "QK-Norm+RoPE+KVCache fusion: replaced %s pattern(s) "
            "with AITER fused_qk_norm_rope_cache_pts_quant_shuffle",
            self.matched_count,
        )

    def is_applicable_for_range(self, compile_range: Range) -> bool:
        return compile_range.end <= self.max_token_num

    def uuid(self) -> str:
        return VllmInductorPass.hash_source(
            self, QkNormRopeKvCachePattern
        )
