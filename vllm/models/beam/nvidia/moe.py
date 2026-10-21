# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TYPE_CHECKING

import torch
from torch import nn

from vllm.config import VllmConfig
from vllm.distributed import tensor_model_parallel_all_gather
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.fused_moe import FusedMoEFactory, GateLinear
from vllm.model_executor.layers.fused_moe.moe_output import UnfinalizedMoEOutput
from vllm.model_executor.layers.fused_moe.runner.moe_runner import (
    _layer_name_type,
    _resolve_layer_name,
    get_layer_from_name,
)
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.models.utils import sequence_parallel_chunk
from vllm.utils.torch_utils import direct_register_custom_op

from .activation import beam_silu_mul
from .layernorm import RMSNorm

if TYPE_CHECKING:
    from .allreduce import BeamAllReduceWorkspace


def _beam_moe_forward_boundary(
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    post_weight: torch.Tensor,
    post_eps: float,
    next_weight: torch.Tensor | None,
    next_eps: float | None,
    x_scale: float,
    r_scale: float,
    layer_name: _layer_name_type,
) -> tuple[torch.Tensor, torch.Tensor]:
    experts = get_layer_from_name(_resolve_layer_name(layer_name))
    layer = get_forward_context().no_compile_layers[f"{experts.layer_name}.finalize"]
    assert isinstance(layer, BeamSparseMoeBlock)
    hidden, normalized = layer._forward_boundary(
        hidden_states,
        residual,
        post_weight,
        post_eps,
        next_weight,
        next_eps,
        (x_scale, r_scale),
    )
    return hidden, normalized if normalized is not None else residual.new_empty(0)


def _beam_moe_forward_boundary_fake(
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    post_weight: torch.Tensor,
    post_eps: float,
    next_weight: torch.Tensor | None,
    next_eps: float | None,
    x_scale: float,
    r_scale: float,
    layer_name: _layer_name_type,
) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.empty_like(residual),
        torch.empty_like(residual) if next_eps is not None else residual.new_empty(0),
    )


direct_register_custom_op(
    op_name="beam_moe_forward_boundary",
    op_func=_beam_moe_forward_boundary,
    mutates_args=["hidden_states"],
    fake_impl=_beam_moe_forward_boundary_fake,
    tags=(torch.Tag.needs_fixed_stride_order,),
)


class BeamMLP(nn.Module):
    """SwiGLU MLP for the ``dense`` entries of ``mlp_layer_types``."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        reduce_results: bool = True,
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.down_proj",
            reduce_results=reduce_results,
        )
        if hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {hidden_act}. Only silu is supported."
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(x)
        # Beam rounds once after the fp32 SiLU and gate/up multiplication.
        out = beam_silu_mul(gate_up)
        out, _ = self.down_proj(out)
        return out


class BeamSparseMoeBlock(nn.Module):
    """Grouped sigmoid ``noaux_tc`` MoE block."""

    def __init__(
        self, vllm_config: VllmConfig, prefix: str = "", reduce_results: bool = True
    ) -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        parallel_config = vllm_config.parallel_config
        quant_config = vllm_config.quant_config
        self.is_sequence_parallel = parallel_config.use_sequence_parallel_moe
        self.gather_workspace: BeamAllReduceWorkspace | None = None
        self.output_boundary_fused = False
        self._compilation_config = vllm_config.compilation_config

        self.gate = GateLinear(
            config.hidden_size,
            config.num_experts,
            bias=False,
            params_dtype=torch.float32,
            out_dtype=torch.float32,
            prefix=f"{prefix}.gate",
        )
        e_score_correction_bias = nn.Parameter(
            torch.zeros(config.num_experts, dtype=torch.float32)
        )
        self.gate.register_parameter("e_score_correction_bias", e_score_correction_bias)

        self.experts = FusedMoEFactory(
            reduce_results=reduce_results or self.is_sequence_parallel,
            is_sequence_parallel=self.is_sequence_parallel,
            gate=self.gate,
            num_experts=config.num_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            renormalize=config.norm_topk_prob,
            quant_config=quant_config,
            use_grouped_topk=(
                config.n_group is not None and config.topk_group is not None
            ),
            num_expert_group=config.n_group,
            topk_group=config.topk_group,
            scoring_func="sigmoid",
            e_score_correction_bias=e_score_correction_bias,
            routed_scaling_factor=config.routed_scaling_factor,
            apply_routed_scale_to_output=True,
            router_logits_dtype=self.gate.out_dtype,
            enable_eplb=parallel_config.enable_eplb,
            num_redundant_experts=parallel_config.eplb_config.num_redundant_experts,
            prefix=f"{prefix}.experts",
        )
        self._boundary_layer_name = f"{prefix}.experts.finalize"

    def defer_finalize(self, max_tokens: int, *, allow_ep: bool = False) -> None:
        from vllm.model_executor.layers.fused_moe.experts.trtllm_nvfp4_moe import (
            TrtLlmNvFp4ExpertsModular,
            TrtLlmNvFp4ExpertsMonolithic,
        )

        experts_cls = getattr(
            self.experts.routed_experts.quant_method, "experts_cls", None
        )
        if experts_cls in (TrtLlmNvFp4ExpertsModular, TrtLlmNvFp4ExpertsMonolithic):
            self.experts.moe_config.defer_moe_finalize(max_tokens, allow_ep=allow_ep)

    def fuse_output_post_norm(
        self,
        post_norm: RMSNorm,
        eps: float,
        *,
        workspace: "BeamAllReduceWorkspace | None" = None,
    ) -> bool:
        from vllm.config.kernel import FLASHINFER_MOE_EP_CUTEDSL
        from vllm.model_executor.layers.fused_moe.experts.flashinfer_moe_ep import (
            FlashInferMoeEpExperts,
        )
        from vllm.model_executor.layers.fused_moe.flashinfer_moe_ep import (
            supports_flashinfer_moe_ep_deferred_finalize,
        )

        moe = self.experts.moe_config
        experts = self.experts.routed_experts
        if (
            (not self.is_sequence_parallel and moe.dp_size == 1)
            or moe.moe_backend != FLASHINFER_MOE_EP_CUTEDSL
            or getattr(experts.quant_method, "experts_cls", None)
            is not FlashInferMoeEpExperts
            or moe.in_dtype != torch.bfloat16
            or moe.hidden_dim != moe.hidden_dim_unpadded
            or moe.hidden_dim % 256
            or not 256 <= moe.hidden_dim <= 8192
            or not 1 <= moe.experts_per_token <= 32
            or experts.apply_router_weight_on_input
            or self.experts.shared_experts is not None
            or self.experts.routed_input_transform is not None
            or self.experts.routed_output_transform is not None
            or moe.pcp_size != 1
            or moe.tp_size != 1
            or moe.ep_size <= 1
            or not supports_flashinfer_moe_ep_deferred_finalize()
            or not hasattr(torch.ops._C, "beam_moe_finalize_norm")
        ):
            return False
        fuse_gather = (
            workspace is not None
            and moe.sp_size == workspace.world_size == moe.ep_size
            and moe.dp_size == 1
        )
        if not self.is_sequence_parallel or fuse_gather:
            self.gather_workspace = workspace if fuse_gather else None
            self.output_boundary_fused = True
            moe.defer_moe_finalize(allow_ep=True)
            self._compilation_config.static_forward_context[
                self._boundary_layer_name
            ] = self
            return True
        return False

    def forward_boundary(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        post_weight: torch.Tensor,
        post_eps: float,
        next_weight: torch.Tensor | None,
        next_eps: float | None,
        scales: tuple[float, float],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        hidden, normalized = torch.ops.vllm.beam_moe_forward_boundary(
            hidden_states,
            residual,
            post_weight,
            post_eps,
            next_weight,
            next_eps,
            *scales,
            self.experts._encode_layer_name(),
        )
        return hidden, normalized if next_eps is not None else None

    def _forward_boundary(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        post_weight: torch.Tensor,
        post_eps: float,
        next_weight: torch.Tensor | None,
        next_eps: float | None,
        scales: tuple[float, float],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        from .moe_finalize import (
            beam_moe_finalize_gather_norm_residual_norm,
            beam_moe_finalize_norm_residual_norm,
        )

        local_states = (
            sequence_parallel_chunk(hidden_states)
            if self.is_sequence_parallel
            else hidden_states
        )
        output = self.experts._forward_impl(local_states, local_states, None)
        assert isinstance(output, UnfinalizedMoEOutput)
        # Consume the borrowed expert workspace inside the same opaque op.
        if self.gather_workspace is None:
            return beam_moe_finalize_norm_residual_norm(
                output,
                residual,
                post_weight,
                post_eps,
                next_weight,
                next_eps,
                scales,
                routed_scale=self.experts.routed_scaling_factor,
            )
        return beam_moe_finalize_gather_norm_residual_norm(
            output,
            residual,
            post_weight,
            post_eps,
            next_weight,
            next_eps,
            scales,
            self.gather_workspace,
            routed_scale=self.experts.routed_scaling_factor,
        )

    def forward(
        self, hidden_states: torch.Tensor
    ) -> torch.Tensor | UnfinalizedMoEOutput:
        num_tokens = hidden_states.shape[0]
        if self.is_sequence_parallel:
            hidden_states = sequence_parallel_chunk(hidden_states)
        elif self.experts.moe_config.should_defer_moe_finalize(num_tokens):
            output = self.experts._forward_impl(hidden_states, hidden_states, None)
            assert isinstance(output, UnfinalizedMoEOutput)
            return output

        output = self.experts(hidden_states=hidden_states, router_logits=hidden_states)
        if self.is_sequence_parallel:
            output = tensor_model_parallel_all_gather(output, 0)[:num_tokens]
        return output
