# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Intel XPU Kimi-K3 text model interfaces."""

from collections.abc import Iterable

import torch
import torch.nn as nn

from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import (
	get_tensor_model_parallel_world_size,
)
from vllm.model_executor.layers.activation import SiluAndMul, SituAndMul
from vllm.model_executor.layers.fused_moe import FusedMoEFactory
from vllm.model_executor.layers.fused_moe.router.gate_linear import GateLinear
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
	ColumnParallelLinear,
	MergedColumnParallelLinear,
	ReplicatedLinear,
	RowParallelLinear,
)
from vllm.model_executor.layers.mla import (
	MLAModules,
	MultiHeadLatentAttentionWrapper,
)
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.configs.kimi_linear import KimiLinearConfig
from vllm.utils.math_utils import cdiv

from .ops.attn_res import attn_res


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
		enable_eplb: bool = False,
		num_redundant_experts: int = 0,
	) -> None:
		super().__init__()
		if enable_eplb or num_redundant_experts:
			raise NotImplementedError(
				"XPU KimiMoE does not yet support EPLB or redundant experts"
			)
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

		self.experts = FusedMoEFactory(
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
			routed_experts = self.experts.routed_experts
			w13_weight = getattr(routed_experts, "w13_weight", None)
			if w13_weight is None:
				w13_weight = getattr(routed_experts, "w13_weight_packed", None)
			w2_weight = getattr(routed_experts, "w2_weight", None)
			if w2_weight is None:
				w2_weight = getattr(routed_experts, "w2_weight_packed", None)
			if w13_weight is not None:
				w13_weight.data.zero_()
			if w2_weight is not None:
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


class KimiMLAAttention(nn.Module):
	"""Kimi-K3 NoPE MLA using the common XPU Triton MLA backend."""

	def __init__(
		self,
		config: KimiLinearConfig,
		hidden_size: int,
		num_heads: int,
		qk_nope_head_dim: int,
		qk_rope_head_dim: int,
		v_head_dim: int,
		q_lora_rank: int | None,
		kv_lora_rank: int,
		use_nope: bool = False,
		cache_config: CacheConfig | None = None,
		quant_config: QuantizationConfig | None = None,
		prefix: str = "",
		**kwargs: object,
	) -> None:
		super().__init__()
		del kwargs
		if not use_nope:
			raise NotImplementedError("XPU Kimi-K3 MLA currently requires NoPE")

		self.hidden_size = hidden_size
		self.qk_nope_head_dim = qk_nope_head_dim
		self.qk_rope_head_dim = qk_rope_head_dim
		self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
		self.v_head_dim = v_head_dim
		self.q_lora_rank = q_lora_rank
		self.kv_lora_rank = kv_lora_rank
		self.num_heads = num_heads
		tp_size = get_tensor_model_parallel_world_size()
		if num_heads % tp_size != 0:
			raise ValueError(
				f"num_heads ({num_heads}) must be divisible by TP size ({tp_size})"
			)
		self.num_local_heads = num_heads // tp_size
		self.scaling = self.qk_head_dim**-0.5

		if q_lora_rank is not None:
			self.fused_qkv_a_proj = MergedColumnParallelLinear(
				hidden_size,
				[q_lora_rank, kv_lora_rank + qk_rope_head_dim],
				bias=False,
				quant_config=quant_config,
				prefix=f"{prefix}.fused_qkv_a_proj",
				disable_tp=True,
			)
			self.q_a_layernorm = RMSNorm(q_lora_rank, eps=config.rms_norm_eps)
			self.q_b_proj = ColumnParallelLinear(
				q_lora_rank,
				num_heads * self.qk_head_dim,
				bias=False,
				quant_config=quant_config,
				prefix=f"{prefix}.q_b_proj",
			)
			self.kv_a_proj_with_mqa = None
			self.q_proj = None
		else:
			self.fused_qkv_a_proj = None
			self.q_a_layernorm = None
			self.q_b_proj = None
			self.kv_a_proj_with_mqa = ReplicatedLinear(
				hidden_size,
				kv_lora_rank + qk_rope_head_dim,
				bias=False,
				quant_config=quant_config,
				prefix=f"{prefix}.kv_a_proj_with_mqa",
			)
			self.q_proj = ColumnParallelLinear(
				hidden_size,
				num_heads * self.qk_head_dim,
				bias=False,
				quant_config=quant_config,
				prefix=f"{prefix}.q_proj",
			)

		self.kv_a_layernorm = RMSNorm(kv_lora_rank, eps=config.rms_norm_eps)
		self.kv_b_proj = ColumnParallelLinear(
			kv_lora_rank,
			num_heads * (qk_nope_head_dim + v_head_dim),
			bias=False,
			quant_config=quant_config,
			prefix=f"{prefix}.kv_b_proj",
		)
		self.o_proj = RowParallelLinear(
			num_heads * v_head_dim,
			hidden_size,
			bias=False,
			quant_config=quant_config,
			prefix=f"{prefix}.o_proj",
		)
		self.g_proj = (
			ColumnParallelLinear(
				hidden_size,
				num_heads * v_head_dim,
				bias=False,
				quant_config=quant_config,
				prefix=f"{prefix}.g_proj",
			)
			if bool(getattr(config, "mla_use_output_gate", False))
			else None
		)

		mla_modules = MLAModules(
			kv_a_layernorm=self.kv_a_layernorm,
			kv_b_proj=self.kv_b_proj,
			rotary_emb=None,
			o_proj=self.o_proj,
			fused_qkv_a_proj=self.fused_qkv_a_proj,
			kv_a_proj_with_mqa=self.kv_a_proj_with_mqa,
			q_a_layernorm=self.q_a_layernorm,
			q_b_proj=self.q_b_proj,
			q_proj=self.q_proj,
			indexer=None,
			is_sparse=False,
			topk_indices_buffer=None,
			g_proj=self.g_proj,
		)
		self.mla_attn = MultiHeadLatentAttentionWrapper(
			hidden_size,
			self.num_local_heads,
			self.scaling,
			qk_nope_head_dim,
			qk_rope_head_dim,
			v_head_dim,
			q_lora_rank,
			kv_lora_rank,
			mla_modules,
			cache_config,
			quant_config,
			prefix,
		)

	def forward(
		self,
		positions: torch.Tensor,
		hidden_states: torch.Tensor,
		output: torch.Tensor,
	) -> None:
		output.copy_(self.mla_attn(positions, hidden_states))


class KimiDecoderLayer(nn.Module):
	"""Standalone XPU Kimi-K3 decoder layer with MLA and MoE support."""

	def __init__(
		self,
		config: KimiLinearConfig,
		vllm_config: VllmConfig,
		prefix: str = "",
	) -> None:
		super().__init__()
		self.hidden_size = config.hidden_size
		self.layer_idx = int(prefix.rsplit(".", 1)[1])
		layer_idx = self.layer_idx
		quant_config = vllm_config.quant_config

		if config.is_kda_layer(layer_idx):
			raise NotImplementedError(
				"XPU KimiDecoderLayer does not yet support full-rank KDA layers"
			)

		qk_nope_head_dim = config.qk_nope_head_dim
		qk_rope_head_dim = config.qk_rope_head_dim
		v_head_dim = config.v_head_dim
		kv_lora_rank = config.kv_lora_rank
		if any(
			value is None
			for value in (
				qk_nope_head_dim,
				qk_rope_head_dim,
				v_head_dim,
				kv_lora_rank,
			)
		):
			raise ValueError("Kimi-K3 MLA dimensions must be configured")
		if not config.mla_use_nope:
			raise NotImplementedError("XPU Kimi-K3 MLA currently requires NoPE")
		assert qk_nope_head_dim is not None
		assert qk_rope_head_dim is not None
		assert v_head_dim is not None
		assert kv_lora_rank is not None
		self.self_attn = KimiMLAAttention(
			config=config,
			hidden_size=self.hidden_size,
			num_heads=config.num_attention_heads,
			qk_nope_head_dim=qk_nope_head_dim,
			qk_rope_head_dim=qk_rope_head_dim,
			v_head_dim=v_head_dim,
			q_lora_rank=config.q_lora_rank,
			kv_lora_rank=kv_lora_rank,
			use_nope=True,
			cache_config=vllm_config.cache_config,
			quant_config=quant_config,
			prefix=f"{prefix}.self_attn",
		)

		self.is_moe_layer = (
			config.is_moe
			and config.num_experts is not None
			and layer_idx >= config.first_k_dense_replace
			and layer_idx % config.moe_layer_freq == 0
		)
		if self.is_moe_layer:
			self.block_sparse_moe = KimiMoE(
				config=config,
				quant_config=quant_config,
				prefix=f"{prefix}.block_sparse_moe",
				layer_idx=layer_idx,
			)
			self.mlp = self.block_sparse_moe
		else:
			self.mlp = KimiMLP(
				hidden_size=self.hidden_size,
				intermediate_size=config.intermediate_size,
				hidden_act=config.hidden_act,
				quant_config=quant_config,
				prefix=f"{prefix}.mlp",
				activation_situ_beta=config.activation_situ_beta,
				activation_situ_linear_beta=config.activation_situ_linear_beta,
			)

		self.input_layernorm = RMSNorm(
			config.hidden_size, eps=config.rms_norm_eps
		)
		self.post_attention_layernorm = RMSNorm(
			config.hidden_size, eps=config.rms_norm_eps
		)
		self.use_attn_res = config.attn_res_block_size is not None
		if self.use_attn_res:
			assert config.attn_res_block_size is not None
			self.attn_res_block_size = config.attn_res_block_size
			self.is_block_write_layer = layer_idx % self.attn_res_block_size == 0
			self.block_write_idx = layer_idx // self.attn_res_block_size
			self.prev_valid_blocks = cdiv(layer_idx, self.attn_res_block_size)
			self.self_attention_res_norm = RMSNorm(
				config.hidden_size, eps=config.rms_norm_eps
			)
			self.mlp_res_norm = RMSNorm(
				config.hidden_size, eps=config.rms_norm_eps
			)
			self.self_attention_res_proj = ReplicatedLinear(
				config.hidden_size,
				1,
				bias=False,
				quant_config=None,
				prefix=f"{prefix}.self_attention_res_proj",
			)
			self.mlp_res_proj = ReplicatedLinear(
				config.hidden_size,
				1,
				bias=False,
				quant_config=None,
				prefix=f"{prefix}.mlp_res_proj",
			)

	def _run_self_attn(
		self,
		positions: torch.Tensor,
		hidden_states: torch.Tensor,
	) -> torch.Tensor:
		output = torch.empty_like(hidden_states)
		self.self_attn(
			positions=positions,
			hidden_states=hidden_states,
			output=output,
		)
		return output

	def _pre_attn_norm(
		self,
		hidden_states: torch.Tensor | None,
		residual: torch.Tensor | None,
		prefix_sum: torch.Tensor | None,
	) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
		if not self.use_attn_res:
			assert hidden_states is not None
			if residual is None:
				residual = hidden_states
				hidden_states = self.input_layernorm(hidden_states)
			else:
				hidden_states, residual = self.input_layernorm(
					hidden_states, residual
				)
			return hidden_states, prefix_sum, residual

		assert prefix_sum is not None
		assert residual is not None
		hidden_states = attn_res(
			prefix_sum,
			hidden_states,
			residual,
			self.self_attention_res_norm.weight,
			self.self_attention_res_proj.weight.squeeze(0),
			self.input_layernorm.weight,
			num_blocks=self.prev_valid_blocks,
			block_write_idx=(
				self.block_write_idx if self.is_block_write_layer else -1
			),
			eps=self.self_attention_res_norm.variance_epsilon,
			output_norm_eps=self.input_layernorm.variance_epsilon,
		)
		return hidden_states, prefix_sum, residual

	def _post_attn_norm(
		self,
		hidden_states: torch.Tensor,
		residual: torch.Tensor,
		prefix_sum: torch.Tensor | None,
	) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
		if not self.use_attn_res:
			hidden_states, residual = self.post_attention_layernorm(
				hidden_states, residual
			)
			return hidden_states, prefix_sum, residual

		assert prefix_sum is not None
		if self.is_block_write_layer:
			prefix_sum = hidden_states
			prefix_delta = None
		else:
			prefix_delta = hidden_states
		mlp_valid_blocks = self.prev_valid_blocks + self.is_block_write_layer
		hidden_states = attn_res(
			prefix_sum,
			prefix_delta,
			residual,
			self.mlp_res_norm.weight,
			self.mlp_res_proj.weight.squeeze(0),
			self.post_attention_layernorm.weight,
			num_blocks=mlp_valid_blocks,
			block_write_idx=-1,
			eps=self.mlp_res_norm.variance_epsilon,
			output_norm_eps=self.post_attention_layernorm.variance_epsilon,
		)
		return hidden_states, prefix_sum, residual

	def forward(
		self,
		positions: torch.Tensor,
		hidden_states: torch.Tensor | None,
		residual: torch.Tensor | None,
		prefix_sum: torch.Tensor | None = None,
		**kwargs: object,
	) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
		del kwargs
		hidden_states, prefix_sum, residual = self._pre_attn_norm(
			hidden_states, residual, prefix_sum
		)
		hidden_states = self._run_self_attn(positions, hidden_states)
		hidden_states, prefix_sum, residual = self._post_attn_norm(
			hidden_states, residual, prefix_sum
		)
		hidden_states = self.mlp(hidden_states)
		return hidden_states, prefix_sum, residual


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

__all__ = [
	"KimiDecoderLayer",
	"KimiLinearForCausalLM",
	"KimiMLAAttention",
	"KimiMoE",
]
