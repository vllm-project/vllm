# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi-K3 mono MoE on ROCm CDNA4 (``VLLM_ROCM_MONO_DECODE=1``).

A small decode step's routed MoE -- the biased sigmoid top-k, the expert sort,
the a4w4 routed experts -- and its shared expert run as one persistent FlyDSL
launch (``mono``, this directory) on the layer's loaded weights.
``MonoLatentMoE`` is bound to one ``ROCmLatentMoERunner``: it decides once per
layer whether the layer matches the launch, and per call whether the step does
(at most ``mono.runner.M_MAX`` tokens). Anything else runs the runner's
multi-kernel path; both paths take and return the same tensors.
"""

from functools import cached_property
from typing import TYPE_CHECKING

import torch

import vllm.envs as envs
from vllm._aiter_ops import rocm_aiter_ops
from vllm.logger import init_logger
from vllm.model_executor.layers.activation import SituAndMul
from vllm.model_executor.layers.fused_moe.router.fused_topk_bias_router import (
    FusedTopKBiasRouter,
)
from vllm.model_executor.layers.fused_moe.router.grouped_topk_router import (
    GroupedTopKRouter,
)
from vllm.platforms import current_platform

if TYPE_CHECKING:
    from vllm.models.kimi_k3.amd.latent_moe_runner import ROCmLatentMoERunner

logger = init_logger(__name__)


class MonoLatentMoE:
    """The mono MoE launch for one ``ROCmLatentMoERunner``'s layer."""

    def __init__(self, moe_runner: "ROCmLatentMoERunner"):
        self.r = moe_runner

    @cached_property
    def layer_ok(self) -> bool:
        """Whether this layer matches the launch (VLLM_ROCM_MONO_DECODE=1,
        gfx950).

        The launch is biased sigmoid top-k, renormalised and unscaled, then the
        a4w4 SiTUv2 MoE on [gate; up] a16w4-shuffled weights, with no EP, bias
        or padding, and vLLM's bf16 KimiMLP shared expert. Read on the first
        forward, after weights are processed.
        """
        if not (envs.VLLM_ROCM_MONO_DECODE and current_platform.is_rocm()):
            return False
        from vllm.platforms.rocm import on_gfx950

        if not (
            on_gfx950()
            and rocm_aiter_ops.is_fused_moe_enabled()
            and rocm_aiter_ops.get_fused_moe_situv2_activation() == "a4w4"
        ):
            return False
        try:
            from vllm.models.kimi_k3.amd.mono import runner  # noqa: F401
        except ImportError:
            logger.warning_once(
                "Kimi-K3 mono MoE needs FlyDSL and AITER's FlyDSL kernels; "
                "running the multi-kernel MoE."
            )
            return False
        r = self.r
        quant_method = r._quant_method
        router = r.router
        cfg = r.moe_config
        parallel = cfg.moe_parallel_config
        quant_config = quant_method.moe_quant_config
        ok = (
            r.gate is None
            and self.shared_mlp_weights is not None
            and getattr(quant_method, "is_k3_situ_aiter", False)
            and not quant_method.is_monolithic
            and not quant_method.mk_can_overlap_shared_experts
            and quant_config is not None
            and quant_config.w1_bias is None
            and quant_config.w2_bias is None
            and (
                isinstance(router, FusedTopKBiasRouter)
                or (
                    isinstance(router, GroupedTopKRouter)
                    and router.num_expert_group == 1
                    and router.topk_group == 1
                )
            )
            and router.e_score_correction_bias is not None
            and router.scoring_func == "sigmoid"
            and router.renormalize
            and router.routed_scaling_factor == 1.0
            and router.num_fused_shared_experts == 0
            and getattr(router, "bias_vl", None) is None
            and getattr(router, "_hash_indices_table", None) is None
            and router.eplb_state is None
            and r.routed_experts.expert_map is None
            and not parallel.use_ep
            and not parallel.enable_eplb
            and parallel.dp_size == 1
            and not cfg.is_sequence_parallel
            and cfg.swiglu_limit is None
            and cfg.hidden_dim == cfg.hidden_dim_unpadded
            and cfg.intermediate_size_per_partition
            == cfg.intermediate_size_per_partition_unpadded
            and not cfg.intermediate_pad
        )
        if ok:
            logger.info_once(
                "Kimi-K3 MoE: mono MoE launch (top-k, routed and shared experts "
                "in one launch) for small batches.",
                scope="global",
            )
        return ok

    @cached_property
    def shared_mlp_weights(self) -> tuple | None:
        """(gate_up weight, down weight, beta, linear_beta) of a KimiMLP shared
        expert."""
        from vllm.model_executor.layers.linear import UnquantizedLinearMethod

        mlp = getattr(self.r._shared_experts, "_layer", None)
        gate_up = getattr(mlp, "gate_up_proj", None)
        down = getattr(mlp, "down_proj", None)
        act = getattr(mlp, "act_fn", None)
        if not (
            isinstance(act, SituAndMul)
            and gate_up is not None
            and down is not None
            and isinstance(gate_up.quant_method, UnquantizedLinearMethod)
            and isinstance(down.quant_method, UnquantizedLinearMethod)
            and gate_up.bias is None
            and down.bias is None
            and not down.reduce_results
            and gate_up.weight.dtype == down.weight.dtype == torch.bfloat16
        ):
            return None
        return gate_up.weight, down.weight, act.beta, act.linear_beta

    def eligible(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        shared_experts_input: torch.Tensor | None,
    ) -> bool:
        r = self.r
        if (
            shared_experts_input is None
            or not self.layer_ok
            or r.router.capture_fn is not None
            or hidden_states.dtype != torch.bfloat16
            or router_logits.dtype != torch.float32
        ):
            return False
        from vllm.models.kimi_k3.amd.mono.runner import supported

        w13 = r.routed_experts.w13_weight
        w_gu, w_dn, _, _ = self.shared_mlp_weights
        return supported(
            hidden_states,
            w13.shape[0],
            r.router.top_k,
            w13.shape[1] // 2,
            shared_experts_input,
            w_gu,
            w_dn,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        shared_experts_input: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """(routed, shared) outputs of one mono MoE launch."""
        from vllm.models.kimi_k3.amd.mono.runner import mono_moe

        r = self.r
        quant_config = r._quant_method.moe_quant_config
        w_gu, w_dn, beta, linear_beta = self.shared_mlp_weights
        return mono_moe(
            router_logits.contiguous(),
            r.router.e_score_correction_bias.data,
            hidden_states.contiguous(),
            r.routed_experts.w13_weight,
            r.routed_experts.w2_weight,
            quant_config.w1_scale,
            quant_config.w2_scale,
            shared_experts_input,
            w_gu,
            w_dn,
            topk=r.router.top_k,
            situ_beta=r.moe_config.activation_situ_beta,
            situ_linear_beta=r.moe_config.activation_situ_linear_beta,
            shared_beta=beta,
            shared_linear_beta=linear_beta,
        )
