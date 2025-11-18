# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable

import torch
import torch._inductor.pattern_matcher as pm
from torch import fx
from torch._higher_order_ops.auto_functionalize import auto_functionalized
from torch._inductor.pattern_matcher import PatternMatcherPass

from vllm.attention import Attention
from vllm.config import VllmConfig, get_layers_from_vllm_config
from vllm.logger import init_logger
from vllm.model_executor.layers.rotary_embedding import RotaryEmbedding

from .fusion import empty_bf16, empty_fp32, empty_i64
from .inductor_pass import enable_fake_mode
from .matcher_utils import MatcherRMSNorm, MatcherRotaryEmbedding
from .vllm_inductor_pass import VllmInductorPass, VllmPatternMatcherPass

logger = init_logger(__name__)

FUSED_QK_ROPE_OP = torch.ops._C.fused_qk_norm_rope.default


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
        head_dim: int,
        num_heads: int,
        num_kv_heads: int,
        eps: float,
        is_neox: bool,
        rope_flashinfer: bool = False,
    ) -> None:
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.eps = eps
        self.rmsnorm_matcher = MatcherRMSNorm(eps)
        self.is_neox = is_neox
        self.rope_flashinfer = rope_flashinfer
        self.rope_matcher = MatcherRotaryEmbedding(
            is_neox=is_neox,
            head_size=self.head_dim,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            use_flashinfer=self.rope_flashinfer,
        )

    def get_inputs(self):
        # Sample inputs to help pattern tracing
        T = 5
        qkv = empty_bf16(T, self.q_size + 2 * self.kv_size)
        positions = empty_i64(T)
        q_weight = empty_bf16(1, self.head_dim)
        k_weight = empty_bf16(1, self.head_dim)
        if self.rope_flashinfer:
            cos_sin_cache = empty_fp32(4096, self.head_dim)
        else:
            cos_sin_cache = empty_bf16(4096, self.head_dim)
        return [
            qkv,
            positions,
            q_weight,
            k_weight,
            cos_sin_cache,
        ]

    @staticmethod
    def wrap_trace_fn(trace_fn, *process_fx_fns: Callable[[fx.GraphModule], None]):
        def wrapped(*args, **kwargs):
            gm = trace_fn(*args, **kwargs)
            for process_fx in process_fx_fns:
                process_fx(gm)

            return gm

        return wrapped

    @staticmethod
    def fx_view_to_reshape(gm: torch.fx.GraphModule):
        from torch._inductor.fx_passes.post_grad import view_to_reshape

        view_to_reshape(gm)

    def register(self, pm_pass: PatternMatcherPass):
        def pattern(
            qkv: torch.Tensor,
            positions: torch.Tensor,
            q_weight: torch.Tensor,
            k_weight: torch.Tensor,
            cos_sin_cache: torch.Tensor,
        ):
            # split qkv -> q,k,v
            q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

            # Q path: view -> RMS -> view back to q.shape
            q_by_head = q.view(
                *q.shape[:-1], q.shape[-1] // self.head_dim, self.head_dim
            )
            q_normed_by_head = self.rmsnorm_matcher(q_by_head, q_weight)
            q_flat = q_normed_by_head.view(q.shape)

            # K path: view -> RMS -> view back to k.shape
            k_by_head = k.view(
                *k.shape[:-1], k.shape[-1] // self.head_dim, self.head_dim
            )
            k_normed_by_head = self.rmsnorm_matcher(k_by_head, k_weight)
            k_flat = k_normed_by_head.view(k.shape)

            # RoPE: apply to flattened q/k
            q_rope, k_rope = self.rope_matcher(positions, q_flat, k_flat, cos_sin_cache)
            return q_rope, k_rope, v

        def replacement(
            qkv: torch.Tensor,
            positions: torch.Tensor,
            q_weight: torch.Tensor,
            k_weight: torch.Tensor,
            cos_sin_cache: torch.Tensor,
        ):
            # Run fused qk_norm_rope op
            result = auto_functionalized(
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
                position_ids=positions.view(-1),
            )
            result_qkv = result[1]

            # Split back to q,k,v and return
            return result_qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        # NOTE: use fx_view_to_reshape to unify view/reshape to simplify
        # pattern and increase matching opportunities
        pm.register_replacement(
            pattern,
            replacement,
            self.get_inputs(),
            QkNormRopePattern.wrap_trace_fn(
                pm.fwd_only,
                QkNormRopePattern.fx_view_to_reshape,
            ),
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

        dtype = config.model_config.dtype
        if dtype not in (torch.bfloat16, torch.float16):
            logger.warning_once(
                "QK Norm+RoPE fusion not enabled: unsupported dtype %s", dtype
            )
            return

        # use one attn layer to get meta (such as head_dim) for QkNormRopePattern
        attn_layers: dict[str, Attention] = get_layers_from_vllm_config(
            config, Attention
        )
        if len(attn_layers) == 0:
            logger.warning_once(
                "QK Norm+RoPE fusion enabled, but no Attention layers were discovered."
            )
            return
        layer = next(iter(attn_layers.values()))

        for epsilon in [1e-5, 1e-6]:
            for neox in [True, False]:
                if RotaryEmbedding.enabled():
                    for rope_flashinfer in [False, True]:
                        QkNormRopePattern(
                            head_dim=layer.head_size,
                            num_heads=layer.num_heads,
                            num_kv_heads=layer.num_kv_heads,
                            eps=epsilon,
                            is_neox=neox,
                            rope_flashinfer=rope_flashinfer,
                        ).register(self.patterns)
                else:
                    QkNormRopePattern(
                        head_dim=layer.head_size,
                        num_heads=layer.num_heads,
                        num_kv_heads=layer.num_kv_heads,
                        eps=epsilon,
                        is_neox=neox,
                    ).register(self.patterns)

        self.dump_patterns(config, self.patterns)

    @VllmInductorPass.time_and_log
    def __call__(self, graph: fx.Graph) -> None:
        self.matched_count = self.patterns.apply(graph)
        logger.debug("Fused QK Norm+RoPE on %s sites", self.matched_count)

    def uuid(self):
        return VllmInductorPass.hash_source(self, QkNormRopePattern)
