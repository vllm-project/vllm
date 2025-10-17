# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pattern-based fusion pass for Q/K RMSNorm + RoPE -> fused_qk_norm_rope."""

import operator

import torch
import torch._inductor.pattern_matcher as pm
from torch import fx
from torch._higher_order_ops.auto_functionalize import auto_functionalized
from torch._inductor.pattern_matcher import PatternMatcherPass

from vllm.attention import Attention
from vllm.config import VllmConfig, get_layers_from_vllm_config
from vllm.logger import init_logger
from vllm.platforms import current_platform

from .inductor_pass import enable_fake_mode
from .fusion import empty_bf16, empty_i64
from .vllm_inductor_pass import VllmInductorPass, VllmPatternMatcherPass

logger = init_logger(__name__)


# Ops used in the pattern (assume kernels are built and available)
RMS_OP = torch.ops._C.rms_norm.default
CONTIGUOUS_OP = torch.ops.aten.contiguous.default
# Some graphs canonicalize `.contiguous()` into a clone with memory_format
CLONE_OP = torch.ops.aten.clone.default
ROPE_OPS: list[torch._ops.OpOverload] = [
    torch.ops._C.rotary_embedding.default,
    torch.ops.vllm.flashinfer_rotary_embedding.default,
]
FUSED_QK_ROPE_OP = torch.ops._C.fused_qk_norm_rope.default
SPLIT_SIZES_OP = torch.ops.aten.split_with_sizes.default
RESHAPE_OP = torch.ops.aten.reshape.default
EMPTY_LIKE_OP = torch.ops.aten.empty_like.default


class QkNormRopePattern:
    """
    Match the unfused sequence in attention blocks and replace with the fused op.

    Unfused (conceptually):
      q, k, v = split(qkv, [qsz, kvsz, kvsz], -1)
      qh = reshape(q, [-1, num_heads, head_dim])
      kh = reshape(k, [-1, num_kv_heads, head_dim])
      qn = rms_norm(qh, q_weight, eps)
      kn = rms_norm(kh, k_weight, eps)
      qf = reshape(qn, [-1, num_heads * head_dim])
      kf = reshape(kn, [-1, num_kv_heads * head_dim])
      qf, kf = rotary_embedding(positions, qf, kf, head_dim, cos_sin_cache, is_neox)
      return qf, kf, v

    Fused replacement:
      fused_qk_norm_rope(qkv, num_heads, num_kv_heads, num_kv_heads, head_dim,
                         eps, q_weight, k_weight, cos_sin_cache, is_neox,
                         positions.view(-1))
      return split(qkv, [qsz, kvsz, kvsz], -1)
    """

    def __init__(
        self,
        layer: Attention,
        eps: float,
        rope_op: torch._ops.OpOverload,
        is_neox: bool,
    ) -> None:
        self.layer = layer
        self.num_heads = layer.num_heads
        self.num_kv_heads = layer.num_kv_heads
        self.head_dim = layer.head_size
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.eps = eps
        self.rope_op = rope_op
        self.is_neox = is_neox

    def register(self, pm_pass: PatternMatcherPass):
        def pattern(
            qkv: torch.Tensor,
            positions: torch.Tensor,
            q_weight: torch.Tensor,
            k_weight: torch.Tensor,
            cos_sin_cache: torch.Tensor,
        ):
            # split qkv -> q,k,v
            # split_tuple = SPLIT_SIZES_OP(
            #     qkv, [self.q_size, self.kv_size, self.kv_size], -1
            # )
            # q = operator.getitem(split_tuple, 0)
            # k = operator.getitem(split_tuple, 1)
            # v = operator.getitem(split_tuple, 2)
            q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

            # Q path: view -> (optional contiguous) -> RMS -> view back to q.shape
            q_by_head = q.view(*q.shape[:-1], q.shape[-1] // self.head_dim, self.head_dim)
            q_out = torch.empty_like(q_by_head)
            q_by_head_contiguous = q_by_head.contiguous()
            qn = auto_functionalized(
                RMS_OP,
                result=q_out,
                input=q_by_head_contiguous,
                weight=q_weight,
                epsilon=self.eps,
            )
            q_normed_by_head = qn[1]
            # RMS_OP(result=q_out, input=q_by_head_contiguous, weight=q_weight, epsilon=self.eps)
            # q_normed_by_head = q_out
            q_flat = q_normed_by_head.view(q.shape)

            # K path: view -> (optional contiguous) -> RMS -> view back to k.shape
            k_by_head = k.view(*k.shape[:-1], k.shape[-1] // self.head_dim, self.head_dim)
            k_out = torch.empty_like(k_by_head)
            k_by_head_contiguous = k_by_head.contiguous()
            kn = auto_functionalized(
                RMS_OP,
                result=k_out,
                input=k_by_head_contiguous,
                weight=k_weight,
                epsilon=self.eps,
            )
            k_normed_by_head = kn[1]
            # RMS_OP(result=k_out, input=k_by_head_contiguous, weight=k_weight, epsilon=self.eps)
            # k_normed_by_head = k_out
            k_flat = k_normed_by_head.view(k.shape)
    
            # RoPE: apply to flattened q/k
            rope = auto_functionalized(
                self.rope_op,
                positions=positions,
                query=q_flat,
                key=k_flat,
                head_size=self.head_dim,
                cos_sin_cache=cos_sin_cache,
                is_neox=self.is_neox,
            )
            return rope[1], rope[2], v
            # self.rope_op(
            #     positions=positions,
            #     query=q_flat,
            #     key=k_flat,
            #     head_size=self.head_dim,
            #     cos_sin_cache=cos_sin_cache,
            #     is_neox=self.is_neox
            # )
            # return q_flat, k_flat, v


        def replacement(
            qkv: torch.Tensor,
            positions: torch.Tensor,
            q_weight: torch.Tensor,
            k_weight: torch.Tensor,
            cos_sin_cache: torch.Tensor,
        ):
            # Flatten positions to 1D as expected by fused op
            pos_flat = RESHAPE_OP(positions, [-1])

            # Run fused op (mutates qkv)
            auto_functionalized(
                FUSED_QK_ROPE_OP,
                qkv=qkv,
                num_heads_q=self.num_heads,
                num_heads_k=self.num_kv_heads,
                num_heads_v=self.num_kv_heads,
                head_dim=self.head_dim,
                eps=self.eps,
                q_weight=q_weight,
                k_weight=k_weight,
                cos_sin_cache=cos_sin_cache,
                is_neox=self.is_neox,
                position_ids=pos_flat,
            )

            # Split back to q,k,v and return
            split_tuple = SPLIT_SIZES_OP(
                qkv, [self.q_size, self.kv_size, self.kv_size], -1
            )
            return (
                operator.getitem(split_tuple, 0),
                operator.getitem(split_tuple, 1),
                operator.getitem(split_tuple, 2),
            )

        # Sample inputs to help pattern tracing (sizes don't have to be exact at runtime)
        T = 5
        qkv = empty_bf16(T, self.q_size + 2 * self.kv_size)
        positions = empty_i64(T)
        q_weight = empty_bf16(1, self.head_dim)
        k_weight = empty_bf16(1, self.head_dim)
        cos_sin_cache = empty_bf16(4096, self.head_dim)
        inputs = [
            qkv,
            positions,
            q_weight,
            k_weight,
            cos_sin_cache,
        ]

        # # Register variants across rope ops and with/without contiguous()
        # # Ensure view ops are canonicalized to reshape in the traced pattern
        # def wrap_trace_fn(process_fx, trace_fn):
        #     def wrapped(*args, **kwargs):
        #         return process_fx(trace_fn(*args, **kwargs))

        #     return wrapped

        # def fx_view_to_reshape(gm: torch.fx.GraphModule):
        #     from torch._inductor.fx_passes.post_grad import view_to_reshape

        #     view_to_reshape(gm)
        #     return gm

        pm.register_replacement(
            pattern,
            replacement,
            inputs,
            pm.fwd_only,
            pm_pass,
        )


class QKNormRoPEFusionPass(VllmPatternMatcherPass):
    """Fuse Q/K RMSNorm + RoPE into fused_qk_norm_rope when the custom op exists."""

    @enable_fake_mode
    def __init__(self, config: VllmConfig):
        super().__init__(config)
        self.patterns: PatternMatcherPass = PatternMatcherPass(
            pass_name="qk_norm_rope_fusion_pass"
        )

        if not current_platform.is_cuda_alike():
            logger.debug(
                "QK Norm+RoPE fusion not enabled: unsupported platform"
            )
            return

        # Register a pattern per attention layer, as sizes differ by shard
        attn_layers = get_layers_from_vllm_config(config, Attention)
        if len(attn_layers) == 0:
            logger.warning(
                "QK Norm+RoPE fusion enabled, but no Attention layers were discovered."
            )
            return

        layer_name, layer = next(iter(attn_layers.items()))

        # Derive parameters from the layer to avoid combinatorial loops
        eps = getattr(getattr(layer, "q_norm", None), "variance_epsilon", None)
        if not isinstance(eps, float):
            eps = 1e-6  # fallback default

        rope_mod = getattr(layer, "rotary_emb", None)
        use_flashinfer = getattr(rope_mod, "use_flashinfer", False)
        rope_op = (
            torch.ops.vllm.flashinfer_rotary_embedding.default
            if use_flashinfer
            else torch.ops._C.rotary_embedding.default
        )
        is_neox = getattr(rope_mod, "is_neox_style", True)

        try:
            QkNormRopePattern(
                layer,
                eps=eps,
                rope_op=rope_op,
                is_neox=is_neox,
            ).register(self.patterns)
        except Exception as e:
            logger.debug(
                "Skipping pattern registration for layer %s: %s",
                layer_name,
                e,
            )

        # Dump patterns for debugging if enabled
        self.dump_patterns(config, self.patterns)

    @VllmInductorPass.time_and_log
    def __call__(self, graph: fx.Graph) -> None:
        if not current_platform.is_cuda_alike():
            return
        self.matched_count = self.patterns.apply(graph)
        logger.debug("Fused QK Norm+RoPE on %s sites", self.matched_count)

    def uuid(self):
        return VllmInductorPass.hash_source(self, QkNormRopePattern)
