# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Intel XPU Kimi-K3 text model interfaces."""

from collections.abc import Iterable
from typing import Any

import torch
import torch.nn as nn

from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import (
	get_pp_group,
	get_tensor_model_parallel_world_size,
)
from vllm.model_executor.layers.activation import SiluAndMul, SituAndMul
from vllm.model_executor.layers.fused_moe import (
	FusedMoEFactory,
	fused_moe_make_expert_params_mapping,
)
from vllm.model_executor.layers.fused_moe.router.gate_linear import GateLinear
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
	ColumnParallelLinear,
	MergedColumnParallelLinear,
	ReplicatedLinear,
	RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.mamba.mamba_utils import (
	MambaStateCopyFunc,
	MambaStateCopyFuncCalculator,
	MambaStateDtypeCalculator,
	MambaStateShapeCalculator,
)
from vllm.model_executor.layers.mla import (
	MLAModules,
	MultiHeadLatentAttentionWrapper,
)
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import (
	ParallelLMHead,
	VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import (
	default_weight_loader,
	maybe_remap_kv_scale_name,
)
from vllm.model_executor.models.interfaces import (
	EagleModelMixin,
	HasInnerState,
	IsHybrid,
	MixtureOfExperts,
	SupportsPP,
)
from vllm.model_executor.models.utils import (
	AutoWeightsLoader,
	PPMissingLayer,
	WeightsMapper,
	get_spec_layer_idx_from_weight_name,
	is_pp_missing_parameter,
	make_layers,
	maybe_prefix,
)
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.configs.kimi_linear import KimiLinearConfig
from vllm.utils.math_utils import cdiv

from .kda import KimiK3DeltaAttention, KimiLinearDeltaAttention
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
			kda_config = config.linear_attn_config
			assert kda_config is not None
			if kda_config.get("use_full_rank_gate", False):
				self.self_attn = KimiK3DeltaAttention(
					config,
					vllm_config,
					prefix=f"{prefix}.self_attn",
				)
			else:
				self.self_attn = KimiLinearDeltaAttention(
					config,
					vllm_config,
					prefix=f"{prefix}.self_attn",
				)
		else:
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


class KimiLinearModel(nn.Module, EagleModelMixin):
	packed_modules_mapping = {
		"gate_up_proj": ["gate_proj", "up_proj"],
		"in_proj_qkvgfab": [
			"q_proj",
			"k_proj",
			"v_proj",
			"b_proj",
			"f_a_proj",
		],
		"conv1d": ["q_conv1d", "k_conv1d", "v_conv1d"],
		"fused_qkv_a_proj": ["q_a_proj", "kv_a_proj_with_mqa"],
	}

	def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
		super().__init__()
		config = vllm_config.model_config.hf_text_config
		self.config = config
		self.attn_res_block_size: int | None = config.attn_res_block_size
		self.use_attn_res = self.attn_res_block_size is not None
		self.vocab_size = config.vocab_size

		if get_pp_group().is_first_rank:
			self.embed_tokens = VocabParallelEmbedding(
				config.vocab_size,
				config.hidden_size,
				prefix=f"{prefix}.embed_tokens",
			)
		else:
			self.embed_tokens = PPMissingLayer()

		def get_layer(layer_prefix: str) -> KimiDecoderLayer:
			return KimiDecoderLayer(config, vllm_config, layer_prefix)

		self.start_layer, self.end_layer, self.layers = make_layers(
			config.num_hidden_layers,
			get_layer,
			prefix=f"{prefix}.layers",
		)
		self.num_attn_res_blocks = (
			cdiv(self.end_layer, self.attn_res_block_size)
			if self.attn_res_block_size is not None
			else 0
		)

		if get_pp_group().is_last_rank:
			self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
			if config.attn_res_block_size is not None:
				self.output_attn_res_norm = RMSNorm(
					config.hidden_size, eps=config.rms_norm_eps
				)
				self.output_attn_res_proj = ReplicatedLinear(
					config.hidden_size,
					1,
					bias=False,
					quant_config=None,
					prefix=f"{prefix}.output_attn_res_proj",
				)
		else:
			self.norm = PPMissingLayer()
			if config.attn_res_block_size is not None:
				self.output_attn_res_norm = PPMissingLayer()
				self.output_attn_res_proj = PPMissingLayer()

		world_size = get_tensor_model_parallel_world_size()
		if config.num_attention_heads % world_size != 0:
			raise ValueError(
				"num_attention_heads must be divisible by tensor parallel size"
			)

	def make_empty_intermediate_tensors(
		self,
		batch_size: int,
		dtype: torch.dtype,
		device: torch.device,
	) -> IntermediateTensors:
		hidden_shape = (batch_size, self.config.hidden_size)
		if not self.use_attn_res:
			return IntermediateTensors(
				{
					"hidden_states": torch.zeros(
						hidden_shape, dtype=dtype, device=device
					),
					"residual": torch.zeros(
						hidden_shape, dtype=dtype, device=device
					),
				}
			)

		assert self.attn_res_block_size is not None
		residual_shape = (
			batch_size,
			cdiv(self.start_layer, self.attn_res_block_size),
			self.config.hidden_size,
		)
		return IntermediateTensors(
			{
				"hidden_states": torch.zeros(
					hidden_shape, dtype=dtype, device=device
				),
				"residual": torch.zeros(
					residual_shape, dtype=dtype, device=device
				),
			}
		)

	def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
		return self.embed_tokens(input_ids)

	def _maybe_add_hidden_state(
		self,
		aux_hidden_states: list[torch.Tensor],
		layer_idx: int,
		hidden_states: torch.Tensor,
		residual: torch.Tensor | None,
	) -> list[torch.Tensor]:
		if self.config.attn_res_block_size is not None:
			residual = None
		return super()._maybe_add_hidden_state(
			aux_hidden_states, layer_idx, hidden_states, residual
		)

	def forward(
		self,
		input_ids: torch.Tensor | None,
		positions: torch.Tensor,
		intermediate_tensors: IntermediateTensors | None,
		inputs_embeds: torch.Tensor | None = None,
		**kwargs: object,
	) -> torch.Tensor | IntermediateTensors | tuple[torch.Tensor, list[torch.Tensor]]:
		del kwargs
		if get_pp_group().is_first_rank:
			if inputs_embeds is not None:
				hidden_states = inputs_embeds
			else:
				if input_ids is None:
					raise ValueError("input_ids or inputs_embeds must be provided")
				hidden_states = self.embed_input_ids(input_ids)
			residual = None
			prefix_sum = None
		else:
			if intermediate_tensors is None:
				raise ValueError("intermediate_tensors must be provided on PP ranks")
			hidden_states = intermediate_tensors["hidden_states"]
			residual = intermediate_tensors["residual"]
			prefix_sum = None

		initial_hidden_states = hidden_states
		if prefix_sum is not None:
			initial_hidden_states = prefix_sum + hidden_states
		aux_hidden_states = self._maybe_add_hidden_state(
			[], self.start_layer, initial_hidden_states, residual
		)
		if not self.use_attn_res:
			for layer_idx, layer in enumerate(
				self.layers[self.start_layer : self.end_layer],
				start=self.start_layer,
			):
				hidden_states, prefix_sum, residual = layer(
					positions=positions,
					hidden_states=hidden_states,
					residual=residual,
					prefix_sum=prefix_sum,
				)
				self._maybe_add_hidden_state(
					aux_hidden_states, layer_idx + 1, hidden_states, residual
				)

			if not get_pp_group().is_last_rank:
				return IntermediateTensors(
					{"hidden_states": hidden_states, "residual": residual}
				)
			if residual is not None:
				hidden_states = hidden_states + residual
			if aux_hidden_states:
				return hidden_states, aux_hidden_states
			return hidden_states

		attn_res_block_size = self.attn_res_block_size
		assert attn_res_block_size is not None
		block_residual = hidden_states.new_empty(
			hidden_states.size(0), self.num_attn_res_blocks, hidden_states.size(1)
		)
		if residual is not None:
			block_residual[:, : residual.size(1), :].copy_(residual)
		residual = block_residual
		if prefix_sum is None:
			prefix_sum = hidden_states
			hidden_states = None

		for layer_idx, layer in enumerate(
			self.layers[self.start_layer : self.end_layer],
			start=self.start_layer,
		):
			hidden_states, prefix_sum, residual = layer(
				positions=positions,
				hidden_states=hidden_states,
				residual=residual,
				prefix_sum=prefix_sum,
			)
			if (layer_idx + 1) in self.aux_hidden_state_layers:
				assert prefix_sum is not None
				aux_state = prefix_sum + hidden_states
				self._maybe_add_hidden_state(
					aux_hidden_states, layer_idx + 1, aux_state, residual
				)

		assert prefix_sum is not None
		if not get_pp_group().is_last_rank:
			assert hidden_states is not None
			return IntermediateTensors(
				{
					"hidden_states": prefix_sum + hidden_states,
					"residual": residual,
				}
			)

		hidden_states = attn_res(
			prefix_sum,
			hidden_states,
			residual,
			self.output_attn_res_norm.weight,
			self.output_attn_res_proj.weight.squeeze(0),
			None,
			num_blocks=self.num_attn_res_blocks,
			block_write_idx=-1,
			eps=self.output_attn_res_norm.variance_epsilon,
			output_norm_eps=0.0,
		)
		if aux_hidden_states:
			return hidden_states, aux_hidden_states
		return hidden_states

	def load_weights(
		self,
		weights: Iterable[
			tuple[str, torch.Tensor] | tuple[str, torch.Tensor, dict[str, Any]]
		],
	) -> set[str]:
		kda_config = self.config.linear_attn_config
		use_full_rank_gate = bool(
			kda_config and kda_config.get("use_full_rank_gate", False)
		)
		beta_shard_id = 5 if use_full_rank_gate else 3
		stacked_params_mapping = [
			(".in_proj_qkvgfab", ".q_proj", 0),
			(".in_proj_qkvgfab", ".k_proj", 1),
			(".in_proj_qkvgfab", ".v_proj", 2),
			(".in_proj_qkvgfab", ".b_proj", beta_shard_id),
			(".in_proj_qkvgfab", ".f_a_proj", 4),
			(".conv1d", ".q_conv1d", 0),
			(".conv1d", ".k_conv1d", 1),
			(".conv1d", ".v_conv1d", 2),
			(".gate_up_proj", ".gate_proj", 0),
			(".gate_up_proj", ".up_proj", 1),
		]
		if use_full_rank_gate:
			stacked_params_mapping.append(
				(".in_proj_qkvgfab", ".g_proj", 3)
			)
		if self.config.q_lora_rank is not None:
			stacked_params_mapping.extend(
				[
					(".fused_qkv_a_proj", ".q_a_proj", 0),
					(".fused_qkv_a_proj", ".kv_a_proj_with_mqa", 1),
				]
			)
		expert_params_mapping = (
			fused_moe_make_expert_params_mapping(
				self,
				ckpt_gate_proj_name="w1",
				ckpt_down_proj_name="w2",
				ckpt_up_proj_name="w3",
				num_experts=self.config.num_experts,
			)
			if self.config.is_moe
			else []
		)
		params_dict = dict(self.named_parameters())
		experts_unpacked = not any(
			name.endswith("w13_weight_packed") for name in params_dict
		)
		loaded_params: set[str] = set()
		for weight in weights:
			name, loaded_weight = weight[0], weight[1]
			loader_kwargs: dict[str, Any] = weight[2] if len(weight) > 2 else {}
			if "rotary_emb.inv_freq" in name:
				continue
			if experts_unpacked and name.endswith(".weight_packed"):
				name = name.replace(".weight_packed", ".weight")
			if get_spec_layer_idx_from_weight_name(self.config, name) is not None:
				continue
			if "rotary_emb.cos_cached" in name or "rotary_emb.sin_cached" in name:
				continue

			for param_name, weight_name, shard_id in stacked_params_mapping:
				if weight_name not in name:
					continue
				if "mlp.experts." in name and name not in params_dict:
					continue
				mapped_name = name.replace(weight_name, param_name)
				if mapped_name not in params_dict:
					continue
				name = mapped_name
				if is_pp_missing_parameter(name, self):
					break
				param = params_dict[name]
				param.weight_loader(param, loaded_weight, shard_id)
				loaded_params.add(name)
				break
			else:
				for (
					expert_param_name,
					expert_weight_name,
					expert_id,
					expert_shard_id,
				) in expert_params_mapping:
					if expert_weight_name not in name:
						continue
					name = name.replace(expert_weight_name, expert_param_name)
					if is_pp_missing_parameter(name, self):
						break
					param = params_dict[name]
					param.weight_loader(
						param,
						loaded_weight,
						name,
						expert_id=expert_id,
						shard_id=expert_shard_id,
					)
					loaded_params.add(name)
					break
				else:
					if name.endswith(".bias") and name not in params_dict:
						continue
					mapped_name = maybe_remap_kv_scale_name(name, params_dict)
					if mapped_name is None:
						continue
					name = mapped_name
					if is_pp_missing_parameter(name, self):
						continue
					param = params_dict[name]
					weight_loader = getattr(
						param, "weight_loader", default_weight_loader
					)
					weight_loader(param, loaded_weight, **loader_kwargs)
					loaded_params.add(name)
		return loaded_params


class KimiLinearForCausalLM(
	nn.Module, HasInnerState, SupportsPP, MixtureOfExperts, IsHybrid
):
	hf_to_vllm_mapper = WeightsMapper(
		orig_to_new_prefix={
			"language_model.model.": "model.",
			"language_model.layers.": "model.layers.",
			"language_model.lm_head.": "lm_head.",
			"vision_tower.": None,
			"mm_projector.": None,
		}
	)

	def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
		super().__init__()
		self.model_config = vllm_config.model_config
		self.vllm_config = vllm_config
		self.config = self.model_config.hf_config
		self.quant_config = vllm_config.quant_config
		self.model = KimiLinearModel(
			vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
		)
		if get_pp_group().is_last_rank:
			self.lm_head = ParallelLMHead(
				self.config.vocab_size,
				self.config.hidden_size,
				quant_config=self.quant_config,
				prefix=maybe_prefix(prefix, "lm_head"),
			)
		else:
			self.lm_head = PPMissingLayer()
		logit_scale = getattr(self.config, "logit_scale", 1.0)
		self.logits_processor = LogitsProcessor(
			self.config.vocab_size, scale=logit_scale
		)

	def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
		return self.model.embed_input_ids(input_ids)

	def make_empty_intermediate_tensors(
		self,
		batch_size: int,
		dtype: torch.dtype,
		device: torch.device,
	) -> IntermediateTensors:
		return self.model.make_empty_intermediate_tensors(batch_size, dtype, device)

	def forward(
		self,
		input_ids: torch.Tensor | None,
		positions: torch.Tensor,
		intermediate_tensors: IntermediateTensors | None = None,
		inputs_embeds: torch.Tensor | None = None,
		**kwargs: object,
	) -> torch.Tensor | IntermediateTensors | tuple[torch.Tensor, list[torch.Tensor]]:
		return self.model(
			input_ids, positions, intermediate_tensors, inputs_embeds, **kwargs
		)

	@classmethod
	def get_mamba_state_dtype_from_config(
		cls,
		vllm_config: VllmConfig,
	) -> tuple[torch.dtype, torch.dtype]:
		return MambaStateDtypeCalculator.kda_state_dtype(
			vllm_config.model_config.dtype,
			vllm_config.cache_config.mamba_cache_dtype,
		)

	@classmethod
	def get_mamba_state_shape_from_config(
		cls,
		vllm_config: VllmConfig,
	) -> tuple[tuple[int, int], tuple[int, int, int]]:
		parallel_config = vllm_config.parallel_config
		hf_config = vllm_config.model_config.hf_config
		num_spec = (
			vllm_config.speculative_config.num_speculative_tokens
			if vllm_config.speculative_config
			else 0
		)
		return MambaStateShapeCalculator.kda_state_shape(
			parallel_config.tensor_parallel_size,
			hf_config.linear_attn_config["num_heads"],
			hf_config.linear_attn_config["head_dim"],
			conv_kernel_size=hf_config.linear_attn_config[
				"short_conv_kernel_size"
			],
			num_spec=num_spec,
		)

	@classmethod
	def get_mamba_state_copy_func(
		cls,
	) -> tuple[MambaStateCopyFunc, MambaStateCopyFunc]:
		return MambaStateCopyFuncCalculator.kda_state_copy_func()

	def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
		hidden_states = self.model.norm(hidden_states, None)
		return self.logits_processor(self.lm_head, hidden_states)

	def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
		loader = AutoWeightsLoader(
			self,
			skip_prefixes=(
				["lm_head."] if self.config.tie_word_embeddings else None
			),
		)
		return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)

__all__ = [
	"KimiDecoderLayer",
	"KimiLinearForCausalLM",
	"KimiLinearModel",
	"KimiMLAAttention",
	"KimiMoE",
]
