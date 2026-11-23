# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Nemotron Shortcut's latent MoE.

The router reads the full hidden state; the routed experts run in a smaller
latent space between `fc1_latent_proj` and `fc2_norm` + `fc2_latent_proj`;
the shared experts read the full hidden state, which in a shortcut block is
a different tensor from the routed input.
"""

from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING

import torch
from torch import nn

import vllm.envs as envs
from vllm import ir
from vllm.config.parallel import ParallelConfig
from vllm.model_executor.layers.fused_moe import FusedMoEFactory, GateLinear
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.models.utils import AutoWeightsLoader
from vllm.models.nemotron_shortcut.nvidia.mlp import NemotronShortcutMLP
from vllm.models.nemotron_shortcut.nvidia.moe.config import NemotronShortcutMoEConfig
from vllm.models.nemotron_shortcut.nvidia.moe.experts import (
    NemotronShortcutRoutedExperts,
)
from vllm.models.nemotron_shortcut.nvidia.moe.runner import (
    NemotronShortcutMoERunner,
    StagedDispatch,
)

if TYPE_CHECKING:
    from transformers import PreTrainedConfig


class NemotronShortcutMoEOutput(nn.Module):
    """The routed output transform: RMSNorm, then the latent up-projection."""

    def __init__(self, norm: nn.Module, projection: ReplicatedLinear) -> None:
        super().__init__()
        self.norm = norm
        self.projection = projection
        # Not the projection's weight dtype: a quantized (e.g. MXFP8)
        # projection's weight dtype is not an activation dtype.
        self.activation_dtype = torch.get_default_dtype()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # The runner reduces the FP32 latent partials before this nonlinear
        # transform; round them once, here.
        hidden_states = hidden_states.to(self.activation_dtype)
        # Inside the opaque staged-MoE op inductor can't fuse the native norm.
        with ir.ops.rms_norm.set_priority(["vllm_c", "native"]):
            hidden_states = self.norm(hidden_states)
        return self.projection(hidden_states)[0]


class NemotronShortcutMoE(nn.Module):
    """Latent MoE with shared experts, run in caller-placed stages.

    `begin_staged` routes and dispatches, `run_staged_experts` runs the routed
    experts once the caller's `current_path_output` exists, and
    `finish_staged` adds the shared experts' output. `forward` runs the three
    in a row. A caller that runs a mixer between the first two calls
    `receive_staged` before the mixer's attention or GDP op.
    """

    def __init__(
        self,
        config: "PreTrainedConfig",
        quant_config: QuantizationConfig | None = None,
        parallel_config: ParallelConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.routed_scaling_factor = config.routed_scaling_factor
        self.top_k = config.num_experts_per_tok
        self.n_routed_experts = config.n_routed_experts
        self.gate = GateLinear(
            config.hidden_size,
            config.n_routed_experts,
            out_dtype=torch.float32,
            force_fp32_compute=True,
            prefix=f"{prefix}.gate",
        )
        self.gate.e_score_correction_bias = nn.Parameter(
            torch.empty(config.n_routed_experts, dtype=torch.float32)
        )
        self.fc1_latent_proj = ReplicatedLinear(
            config.hidden_size,
            config.moe_latent_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.fc1_latent_proj",
        )
        self.fc2_norm = (
            RMSNorm(config.moe_latent_size, eps=config.rms_norm_eps)
            if config.moe_use_norm_before_up_proj
            else nn.Identity()
        )
        self.fc2_latent_proj = ReplicatedLinear(
            config.moe_latent_size,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.fc2_latent_proj",
        )
        self.shared_experts = (
            NemotronShortcutMLP(
                config,
                config.hidden_size,
                config.moe_shared_expert_intermediate_size,
                quant_config=quant_config,
                # The runner reduces it with the routed output.
                reduce_results=False,
                prefix=f"{prefix}.shared_experts",
            )
            if config.moe_shared_expert_intermediate_size is not None
            else None
        )
        runner = FusedMoEFactory(
            num_experts=config.n_routed_experts,
            top_k=self.top_k,
            hidden_size=config.moe_latent_size,
            intermediate_size=config.moe_intermediate_size,
            quant_config=quant_config,
            prefix=f"{prefix}.experts",
            renormalize=self.top_k > 1,
            activation="relu2_no_mul",
            # Logit-space selection replaces the fused sigmoid top-k router
            # and applies the bias and routed_scaling_factor itself.
            custom_routing_function=(
                self.route_logit_space
                if getattr(config, "moe_router_bias_space", "score") == "logit"
                else None
            ),
            scoring_func="sigmoid",
            e_score_correction_bias=self.gate.e_score_correction_bias,
            routed_scaling_factor=self.routed_scaling_factor,
            routed_input_transform=self.fc1_latent_proj,
            routed_output_transform=NemotronShortcutMoEOutput(
                self.fc2_norm, self.fc2_latent_proj
            ),
            router_logits_dtype=torch.float32,
            ckpt_names=("up_proj", "down_proj", None),
            moe_config_cls=NemotronShortcutMoEConfig,
            routed_experts_cls=NemotronShortcutRoutedExperts,
            routed_experts_args={
                "activation_clamp_value": config.activation_clamp_value
            },
            runner_cls=NemotronShortcutMoERunner,
        )
        assert isinstance(runner, NemotronShortcutMoERunner)
        self.experts = runner

    def route_logit_space(
        self,
        hidden_states: torch.Tensor,
        gating_output: torch.Tensor,
        topk: int,
        renormalize: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Select `topk(logits + bias)`; weight by the unbiased sigmoid scores."""
        logits = gating_output.float()
        topk_ids = torch.topk(
            logits + self.gate.e_score_correction_bias,
            k=topk,
            dim=-1,
            sorted=envs.VLLM_BATCH_INVARIANT,
        )[1]
        topk_weights = logits.gather(1, topk_ids).sigmoid()
        if renormalize:
            topk_weights = topk_weights / (
                topk_weights.sum(dim=-1, keepdim=True) + 1e-20
            )
        return topk_weights * self.routed_scaling_factor, topk_ids

    def begin_staged(self, routed_hidden_states: torch.Tensor) -> StagedDispatch:
        logits, _ = self.gate(routed_hidden_states)
        return self.experts.begin_staged(routed_hidden_states, logits)

    def receive_staged(
        self, dispatch: StagedDispatch, projections: Sequence[torch.Tensor]
    ) -> StagedDispatch:
        return self.experts.receive_staged(dispatch, projections)

    def run_staged_experts(
        self, dispatch: StagedDispatch, current_path_output: torch.Tensor
    ) -> torch.Tensor:
        return self.experts.run_staged_experts(dispatch, current_path_output)

    def finish_staged(
        self, combine: torch.Tensor, shared_hidden_states: torch.Tensor
    ) -> torch.Tensor:
        shared = (
            self.shared_experts(shared_hidden_states)
            if self.shared_experts is not None
            else None
        )
        return self.experts.finish_staged(combine, shared_hidden_states, shared)

    def forward(
        self,
        routed_hidden_states: torch.Tensor,
        shared_hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        dispatch = self.begin_staged(routed_hidden_states)
        combine = self.run_staged_experts(dispatch, shared_hidden_states)
        return self.finish_staged(combine, shared_hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        experts = self.experts.routed_experts
        # Checkpoints store each projection fused over the experts:
        # "experts.{up,down}_proj" plus, when quantized, its scales.
        fused_names = {
            f"experts.{projection}{suffix}": (parameter + target, shard)
            for projection, parameter, shard in (
                ("up_proj", "w13", "w1"),
                ("down_proj", "w2", "w2"),
            )
            for suffix, target in (
                ("", "_weight"),
                ("_scale", "_weight_scale"),
                ("_scale_2", "_weight_scale_2"),
                ("_input_scale", "_input_scale"),
            )
        }
        loaded = set()
        other = []
        for name, weight in weights:
            if name not in fused_names:
                other.append((name, weight))
                continue
            param_name, shard_id = fused_names[name]
            param = getattr(experts, param_name)
            for expert_id, expert_weight in enumerate(weight):
                experts.weight_loader(
                    param,
                    expert_weight,
                    param_name,
                    shard_id,
                    expert_id,
                    return_success=False,
                )
            loaded.add(f"experts.routed_experts.{param_name}")
        return loaded | AutoWeightsLoader(self).load_weights(other)
