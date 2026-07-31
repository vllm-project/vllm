# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Intel XPU Kimi-K3 text model interfaces."""

from collections.abc import Iterable

import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.layers.activation import SiluAndMul, SituAndMul
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.layers.fused_moe.router.gate_linear import GateLinear
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
	MergedColumnParallelLinear,
	ReplicatedLinear,
	RowParallelLinear,
)
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.configs.kimi_linear import KimiLinearConfig


class KimiMLP(nn.Module):
	def __init__(
		self,
		hidden_size: int,
		intermediate_size: int,
		hidden_act: str,
		quant_config: QuantizationConfig | None = None,
		reduce_results: bool = True,
		prefix: str = "",
		activation_situ_beta: float | None = None,
		activation_situ_linear_beta: float | None = None,
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
			reduce_results=reduce_results,
			prefix=f"{prefix}.down_proj",
		)
		if hidden_act == "silu":
			self.act_fn = SiluAndMul()
		elif hidden_act == "situ":
			self.act_fn = SituAndMul(
				beta=activation_situ_beta or 1.0,
				linear_beta=activation_situ_linear_beta,
			)
		else:
			raise ValueError(
				f"Unsupported activation: {hidden_act}. "
				"Only silu and situ are supported."
			)

	def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
		gate_up, _ = self.gate_up_proj(hidden_states)
		hidden_states = self.act_fn(gate_up)
		hidden_states, _ = self.down_proj(hidden_states)
		return hidden_states


class KimiRoutedOutputTransform(nn.Module):
	def __init__(
		self,
		norm: RMSNorm | None,
		up_proj: ReplicatedLinear,
	) -> None:
		super().__init__()
		self.norm = norm
		self.up_proj = up_proj

	def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
		if self.norm is not None:
			hidden_states = self.norm(hidden_states)
		hidden_states, _ = self.up_proj(hidden_states)
		return hidden_states


class KimiMoE(nn.Module):
	"""Native XPU Kimi-K3 routed and shared expert layer."""

	def __init__(
		self,
		config: KimiLinearConfig,
		quant_config: QuantizationConfig | None = None,
		prefix: str = "",
		layer_idx: int = 0,
	) -> None:
		super().__init__()
		hidden_size = config.hidden_size
		moe_intermediate_size = config.moe_intermediate_size
		num_experts = config.num_experts
		num_experts_per_token = config.num_experts_per_token
		assert moe_intermediate_size is not None
		assert num_experts is not None
		assert num_experts_per_token is not None

		routed_expert_hidden_size = config.routed_expert_hidden_size
		self.use_latent_moe = routed_expert_hidden_size is not None
		self.latent_moe_use_norm = config.latent_moe_use_norm
		self.moe_hidden_size = (
			routed_expert_hidden_size
			if routed_expert_hidden_size is not None
			else hidden_size
		)
		self.tp_size = get_tensor_model_parallel_world_size()
		self.routed_scaling_factor = config.routed_scaling_factor
		self.num_shared_experts = config.num_shared_experts
		self.layer_idx = layer_idx
		self.padded_moe_intermediate_size = moe_intermediate_size
		min_moe_intermediate_per_partition = getattr(
			config, "min_moe_intermediate_per_partition", 256
		)
		if self.tp_size > 1:
			moe_intermediate_per_partition = moe_intermediate_size // self.tp_size
			if moe_intermediate_per_partition < min_moe_intermediate_per_partition:
				self.padded_moe_intermediate_size = (
					min_moe_intermediate_per_partition * self.tp_size
				)

		activation_situ_beta = (
			config.activation_situ_beta if config.hidden_act == "situ" else None
		)
		activation_situ_linear_beta = (
			config.activation_situ_linear_beta if config.hidden_act == "situ" else None
		)
		self.gate = GateLinear(
			input_size=hidden_size,
			output_size=num_experts,
			bias=False,
			out_dtype=torch.float32,
			prefix=f"{prefix}.gate",
		)
		self.gate.e_score_correction_bias = nn.Parameter(
			torch.empty(num_experts, dtype=torch.float32)
		)

		if self.num_shared_experts is not None:
			shared_intermediate_size = moe_intermediate_size * self.num_shared_experts
			self.shared_experts = KimiMLP(
				hidden_size=hidden_size,
				intermediate_size=shared_intermediate_size,
				hidden_act=config.hidden_act,
				quant_config=quant_config,
				reduce_results=False,
				prefix=f"{prefix}.shared_experts",
				activation_situ_beta=activation_situ_beta,
				activation_situ_linear_beta=activation_situ_linear_beta,
			)
		else:
			self.shared_experts = None

		self.routed_expert_down_proj: ReplicatedLinear | None
		self.routed_expert_norm: RMSNorm | None
		self.routed_expert_up_proj: ReplicatedLinear | None
		self.routed_output_transform: KimiRoutedOutputTransform | None
		if self.use_latent_moe:
			self.routed_expert_down_proj = ReplicatedLinear(
				hidden_size,
				self.moe_hidden_size,
				bias=False,
				quant_config=None,
				prefix=f"{prefix}.routed_expert_down_proj",
			)
			self.routed_expert_norm = (
				RMSNorm(self.moe_hidden_size, eps=config.rms_norm_eps)
				if self.latent_moe_use_norm
				else None
			)
			self.routed_expert_up_proj = ReplicatedLinear(
				self.moe_hidden_size,
				hidden_size,
				bias=False,
				quant_config=None,
				prefix=f"{prefix}.routed_expert_up_proj",
			)
			self.routed_output_transform = KimiRoutedOutputTransform(
				self.routed_expert_norm,
				self.routed_expert_up_proj,
			)
		else:
			self.routed_expert_down_proj = None
			self.routed_expert_norm = None
			self.routed_expert_up_proj = None
			self.routed_output_transform = None

		self.experts = FusedMoE(
			shared_experts=self.shared_experts,
			num_experts=num_experts,
			top_k=num_experts_per_token,
			hidden_size=self.moe_hidden_size,
			intermediate_size=self.padded_moe_intermediate_size,
			activation=config.hidden_act,
			activation_situ_beta=activation_situ_beta,
			activation_situ_linear_beta=activation_situ_linear_beta,
			renormalize=config.moe_renormalize,
			quant_config=quant_config,
			use_grouped_topk=config.use_grouped_topk,
			num_expert_group=config.num_expert_group,
			topk_group=config.topk_group,
			prefix=f"{prefix}.experts",
			scoring_func=config.moe_router_activation_func,
			e_score_correction_bias=self.gate.e_score_correction_bias,
			routed_scaling_factor=self.routed_scaling_factor,
			routed_input_transform=self.routed_expert_down_proj,
			routed_output_transform=self.routed_output_transform,
		)
		if self.padded_moe_intermediate_size != moe_intermediate_size:
			w13_weight = getattr(self.experts, "w13_weight", None)
			if w13_weight is None:
				w13_weight = self.experts.w13_weight_packed
			w2_weight = getattr(self.experts, "w2_weight", None)
			if w2_weight is None:
				w2_weight = self.experts.w2_weight_packed
			w13_weight.data.zero_()
			w2_weight.data.zero_()
			self.experts.moe_config.intermediate_size_per_partition_unpadded = (
				moe_intermediate_size // self.tp_size
			)

	def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
		num_tokens, hidden_size = hidden_states.shape
		hidden_states = hidden_states.view(-1, hidden_size)
		router_logits, _ = self.gate(hidden_states)
		final_hidden_states = self.experts(
			hidden_states=hidden_states,
			router_logits=router_logits,
		)
		return final_hidden_states.view(num_tokens, hidden_size)


class KimiLinearForCausalLM(nn.Module):
	"""Placeholder for the native Intel XPU Kimi-K3 text model."""

	def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
		super().__init__()
		del vllm_config, prefix
		raise NotImplementedError("Native XPU Kimi-K3 text model is not implemented.")

	def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
		raise NotImplementedError

	def make_empty_intermediate_tensors(
		self,
		batch_size: int,
		dtype: torch.dtype,
		device: torch.device,
	) -> IntermediateTensors:
		raise NotImplementedError

	def forward(
		self,
		input_ids: torch.Tensor | None,
		positions: torch.Tensor,
		intermediate_tensors: IntermediateTensors | None = None,
		inputs_embeds: torch.Tensor | None = None,
		**kwargs: object,
	) -> torch.Tensor | IntermediateTensors:
		raise NotImplementedError

	def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
		raise NotImplementedError

	def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
		raise NotImplementedError

__all__ = ["KimiLinearForCausalLM", "KimiMoE"]