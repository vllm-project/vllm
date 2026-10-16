# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from torch import nn

from vllm.config import VllmConfig
from vllm.distributed import tensor_model_parallel_all_gather
from vllm.model_executor.layers.fused_moe import FusedMoEFactory, GateLinear
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.models.utils import sequence_parallel_chunk
from vllm.utils.torch_utils import STR_DTYPE_TO_TORCH_DTYPE

from .activation import beam_silu_mul


class BeamMLP(nn.Module):
    """Dense SwiGLU MLP (used by ``mlp_only_layers``)."""

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
        out = beam_silu_mul(gate_up)
        out, _ = self.down_proj(out)
        return out


class BeamSparseMoeBlock(nn.Module):
    """Grouped sigmoid ``noaux_tc`` MoE with normal finalized output."""

    def __init__(
        self, vllm_config: VllmConfig, prefix: str = "", reduce_results: bool = True
    ) -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        parallel_config = vllm_config.parallel_config
        quant_config = vllm_config.quant_config
        self.is_sequence_parallel = parallel_config.use_sequence_parallel_moe

        router_dtype = STR_DTYPE_TO_TORCH_DTYPE[config.moe_router_dtype or "float32"]
        self.gate = GateLinear(
            config.hidden_size,
            config.num_experts,
            bias=False,
            params_dtype=router_dtype,
            out_dtype=router_dtype,
            prefix=f"{prefix}.gate",
        )
        e_score_correction_bias = None
        if config.topk_method == "noaux_tc":
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
            scoring_func=config.scoring_func,
            e_score_correction_bias=e_score_correction_bias,
            routed_scaling_factor=(
                config.routed_scaling_factor
                if config.routed_scaling_factor is not None
                else 1.0
            ),
            apply_routed_scale_to_output=True,
            router_logits_dtype=self.gate.out_dtype,
            enable_eplb=parallel_config.enable_eplb,
            num_redundant_experts=parallel_config.eplb_config.num_redundant_experts,
            prefix=f"{prefix}.experts",
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        num_tokens = hidden_states.shape[0]
        if self.is_sequence_parallel:
            hidden_states = sequence_parallel_chunk(hidden_states)
        output = self.experts(hidden_states=hidden_states, router_logits=hidden_states)
        if self.is_sequence_parallel:
            output = tensor_model_parallel_all_gather(output, 0)[:num_tokens]
        return output
