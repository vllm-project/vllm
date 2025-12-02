# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable

import torch
import torch._inductor.pattern_matcher as pm
from torch import fx
from torch._inductor.pattern_matcher import PatternMatcherPass

from vllm._aiter_ops import rocm_aiter_ops
from vllm.attention import Attention
from vllm.config import VllmConfig, get_layers_from_vllm_config
from vllm.logger import init_logger
from vllm.platforms import current_platform

from .fusion import empty_bf16, empty_i64
from .inductor_pass import enable_fake_mode
from .matcher_utils import MatcherMRotaryEmbedding, MatcherRMSNorm
from .vllm_inductor_pass import VllmInductorPass, VllmPatternMatcherPass

logger = init_logger(__name__)

FUSED_QK_MROPE_OP = None
if current_platform.is_rocm() and rocm_aiter_ops.is_enabled():
    FUSED_QK_MROPE_OP = rocm_aiter_ops.qknorm_mrope


class QkNormMRopePattern:
    """
    Match the unfused sequence in attention blocks and replace with the fused op.

    """

    def __init__(
        self,
        head_dim: int,
        num_heads: int,
        num_kv_heads: int,
        eps: float,
        is_neox: bool,
        mrope_section: list[int],
        mrope_interleaved: bool,
    ) -> None:
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.eps = eps
        self.rmsnorm_matcher = MatcherRMSNorm(eps)
        self.is_neox = is_neox
        self.mrope_section = mrope_section
        self.mrope_interleaved = mrope_interleaved
        self.rope_matcher = MatcherMRotaryEmbedding(
            is_neox=is_neox,
            head_size=self.head_dim,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            mrope_section=self.mrope_section,
            mrope_interleaved=self.mrope_interleaved,
        )

    def get_inputs(self):
        # Sample inputs to help pattern tracing
        T = 5
        qkv = empty_bf16(T, self.q_size + 2 * self.kv_size)
        positions = empty_i64(3, T)
        q_weight = empty_bf16(1, self.head_dim)
        k_weight = empty_bf16(1, self.head_dim)
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
            assert FUSED_QK_MROPE_OP is not None, (
                "Fused QK Norm+mRoPE op is not available."
            )
            return FUSED_QK_MROPE_OP(
                qkv,
                q_weight,
                k_weight,
                cos_sin_cache,
                positions,
                positions.size(-1),
                self.num_heads,
                self.num_kv_heads,
                self.num_kv_heads,
                self.head_dim,
                self.is_neox,
                self.mrope_section,
                self.mrope_interleaved,
                self.eps,
            )

        # NOTE: use fx_view_to_reshape to unify view/reshape to simplify
        # pattern and increase matching opportunities
        pm.register_replacement(
            pattern,
            replacement,
            self.get_inputs(),
            QkNormMRopePattern.wrap_trace_fn(
                pm.fwd_only,
                QkNormMRopePattern.fx_view_to_reshape,
            ),
            pm_pass,
        )


class QKNormMRoPEFusionPass(VllmPatternMatcherPass):
    """Fuse Q/K RMSNorm + RoPE into fused_qk_norm_rope when the custom op exists."""

    @enable_fake_mode
    def __init__(self, config: VllmConfig):
        rope_params = config.model_config.hf_text_config.rope_parameters
        assert rope_params is not None and "mrope_section" in rope_params, (
            "QK Norm+mRoPE fusion pass requires mRoPE "
            "related fields available in the model config."
        )
        self.mrope_section = rope_params["mrope_section"]
        self.mrope_interleaved = rope_params.get("mrope_interleaved", False)
        super().__init__(config)
        self.patterns: PatternMatcherPass = PatternMatcherPass(
            pass_name="qk_norm_mrope_fusion_pass"
        )

        dtype = config.model_config.dtype
        if dtype not in (torch.bfloat16, torch.float16):
            logger.warning_once(
                "QK Norm+mRoPE fusion not enabled: unsupported dtype %s", dtype
            )
            return

        # use one attn layer to get meta (such as head_dim) for QkNormRopePattern
        attn_layers: dict[str, Attention] = get_layers_from_vllm_config(
            config, Attention
        )

        if len(attn_layers) == 0:
            logger.warning_once(
                "QK Norm+mRoPE fusion enabled, but no Attention layers were discovered."
            )
            return
        layer = next(iter(attn_layers.values()))

        for epsilon in [1e-5, 1e-6]:
            for neox in [True, False]:
                QkNormMRopePattern(
                    head_dim=layer.head_size,
                    num_heads=layer.num_heads,
                    num_kv_heads=layer.num_kv_heads,
                    eps=epsilon,
                    is_neox=neox,
                    mrope_section=self.mrope_section,
                    mrope_interleaved=self.mrope_interleaved,
                ).register(self.patterns)

        self.dump_patterns(config, self.patterns)

    @VllmInductorPass.time_and_log
    def __call__(self, graph: fx.Graph) -> None:
        self.matched_count = self.patterns.apply(graph)
        logger.debug("Fused QK Norm+mRoPE on %s sites", self.matched_count)

    def uuid(self):
        return VllmInductorPass.hash_source(self, QkNormMRopePattern)
