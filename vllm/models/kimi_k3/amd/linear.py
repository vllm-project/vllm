# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Iterable
from typing import Any

import torch
from torch import nn

from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import (
    get_pp_group,
    get_tensor_model_parallel_world_size,
)
from vllm.logger import init_logger
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
from vllm.model_executor.layers.mamba.gdn.kimi_gdn_linear_attn import (
    KimiGatedDeltaNetAttention as KimiLinearGatedDeltaNetAttention,
)
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateCopyFunc,
    MambaStateCopyFuncCalculator,
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
)
from vllm.model_executor.layers.mla import MLAModules, MultiHeadLatentAttentionWrapper
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
    get_spec_layer_idx_from_weight_name,
    is_pp_missing_parameter,
    make_layers,
    maybe_prefix,
)
from vllm.models.kimi_k3.amd.kda import KimiK3DeltaAttention
from vllm.models.kimi_k3.amd.ops.attn_res import attn_res
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.configs.kimi_linear import KimiLinearConfig
from vllm.utils.math_utils import cdiv

logger = init_logger(__name__)


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

    def forward(self, x):
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


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


def _apply_attn_res(
    prefix_sum: torch.Tensor,
    block_residual: torch.Tensor,
    proj: ReplicatedLinear,
    norm: RMSNorm,
    num_valid_blocks: int,
    *,
    delta: torch.Tensor | None = None,
    output_norm: RMSNorm | None = None,
    block_write_idx: int = -1,
) -> torch.Tensor:
    return attn_res(
        prefix_sum,
        delta,
        block_residual,
        norm.weight,
        proj.weight.squeeze(0),
        None if output_norm is None else output_norm.weight,
        num_valid_blocks,
        block_write_idx,
        norm.variance_epsilon,
        0.0 if output_norm is None else output_norm.variance_epsilon,
    )


class KimiMoE(nn.Module):
    def __init__(
        self,
        config: KimiLinearConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        layer_idx: int = 0,
    ):
        super().__init__()
        hidden_size = config.hidden_size
        moe_intermediate_size = config.moe_intermediate_size
        num_experts = config.num_experts
        num_experts_per_token = config.num_experts_per_token
        assert moe_intermediate_size is not None
        assert num_experts is not None
        assert num_experts_per_token is not None
        moe_renormalize = config.moe_renormalize
        routed_expert_hidden_size = config.routed_expert_hidden_size
        self.use_latent_moe = routed_expert_hidden_size is not None
        self.moe_hidden_size = (
            routed_expert_hidden_size
            if routed_expert_hidden_size is not None
            else hidden_size
        )
        self.latent_moe_use_norm = config.latent_moe_use_norm
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

        # Route with fp32 logits for numerically stable expert selection.
        self.gate = GateLinear(
            input_size=hidden_size,
            output_size=num_experts,
            bias=False,
            out_dtype=torch.float32,
            prefix=f"{prefix}.gate",
        )

        # Preserve FP32 checkpoint values and match FP32 router logits.
        self.gate.e_score_correction_bias = nn.Parameter(
            torch.empty(num_experts, dtype=torch.float32)
        )

        if self.num_shared_experts is not None:
            shared_intermediate_size = moe_intermediate_size * self.num_shared_experts
            self.shared_experts = KimiMLP(
                hidden_size=config.hidden_size,
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
                self.routed_expert_norm, self.routed_expert_up_proj
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
            renormalize=moe_renormalize,
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
            hidden_states=hidden_states, router_logits=router_logits
        )
        return final_hidden_states.view(num_tokens, hidden_size)


class KimiMLAAttention(nn.Module):
    """
    Main reference: DeepseekV2 vllm Implementation
    """

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
        **kwargs,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.num_heads = num_heads
        tp_size = get_tensor_model_parallel_world_size()
        self.num_local_heads = num_heads // tp_size
        self.scaling = self.qk_head_dim**-0.5
        self.use_nope = use_nope
        assert self.use_nope is True
        assert num_heads % tp_size == 0
        if self.q_lora_rank is not None:
            self.fused_qkv_a_proj = MergedColumnParallelLinear(
                self.hidden_size,
                [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim],
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.fused_qkv_a_proj",
                disable_tp=True,
            )
        else:
            self.kv_a_proj_with_mqa = ReplicatedLinear(
                self.hidden_size,
                self.kv_lora_rank + self.qk_rope_head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.kv_a_proj_with_mqa",
            )
        if self.q_lora_rank is not None:
            self.q_a_layernorm = RMSNorm(
                self.q_lora_rank,
                eps=config.rms_norm_eps,
            )
            self.q_b_proj = ColumnParallelLinear(
                self.q_lora_rank,
                self.num_heads * self.qk_head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.q_b_proj",
            )
        else:
            self.q_proj = ColumnParallelLinear(
                self.hidden_size,
                self.num_heads * self.qk_head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.q_proj",
            )
        self.kv_a_layernorm = RMSNorm(
            self.kv_lora_rank,
            eps=config.rms_norm_eps,
        )
        self.kv_b_proj = ColumnParallelLinear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.kv_b_proj",
        )
        self.o_proj = RowParallelLinear(
            self.num_heads * self.v_head_dim,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        self.use_output_gate = config.mla_use_output_gate
        if self.use_output_gate:
            projection_size = self.num_heads * self.v_head_dim
            self.g_proj = ColumnParallelLinear(
                self.hidden_size,
                projection_size,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.g_proj",
            )

        # TODO: Remove this mypy workaround once the K3 PR is fully merged.
        mla_modules = MLAModules(  # type: ignore[call-arg]
            kv_a_layernorm=self.kv_a_layernorm,
            kv_b_proj=self.kv_b_proj,
            rotary_emb=None,
            o_proj=self.o_proj,
            fused_qkv_a_proj=self.fused_qkv_a_proj
            if self.q_lora_rank is not None
            else None,
            kv_a_proj_with_mqa=self.kv_a_proj_with_mqa
            if self.q_lora_rank is None
            else None,
            q_a_layernorm=self.q_a_layernorm if self.q_lora_rank is not None else None,
            q_b_proj=self.q_b_proj if self.q_lora_rank is not None else None,
            q_proj=self.q_proj if self.q_lora_rank is None else None,
            indexer=None,
            is_sparse=False,
            topk_indices_buffer=None,
            g_proj=getattr(self, "g_proj", None),
        )
        self.mla_attn = MultiHeadLatentAttentionWrapper(
            self.hidden_size,
            self.num_local_heads,
            self.scaling,
            self.qk_nope_head_dim,
            self.qk_rope_head_dim,
            self.v_head_dim,
            self.q_lora_rank,
            self.kv_lora_rank,
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
        output[:] = self.mla_attn(positions, hidden_states)


class KimiDecoderLayer(nn.Module):
    def __init__(
        self,
        config: KimiLinearConfig,
        vllm_config: VllmConfig,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = int(prefix.rsplit(".", 1)[1])

        self.is_moe = config.is_moe
        layer_idx = self.layer_idx
        model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config

        if config.is_kda_layer(layer_idx):
            # Kimi-K3 sets use_full_rank_gate and uses the ROCm-specific K3 KDA
            # layer; Kimi-Linear keeps the shared low-rank-gate implementation.
            kda_config = config.linear_attn_config
            assert kda_config is not None
            if kda_config.get("use_full_rank_gate", False):
                self.self_attn = KimiK3DeltaAttention(
                    config,
                    vllm_config,
                    prefix=f"{prefix}.self_attn",
                )
            else:
                self.self_attn = KimiLinearGatedDeltaNetAttention(
                    config,
                    vllm_config,
                    prefix=f"{prefix}.self_attn",
                )
        else:
            qk_nope_head_dim = config.qk_nope_head_dim
            qk_rope_head_dim = config.qk_rope_head_dim
            v_head_dim = config.v_head_dim
            kv_lora_rank = config.kv_lora_rank
            mla_use_nope = config.mla_use_nope
            assert qk_nope_head_dim is not None
            assert qk_rope_head_dim is not None
            assert v_head_dim is not None
            assert kv_lora_rank is not None
            assert mla_use_nope is not None
            self.self_attn = KimiMLAAttention(
                layer_idx=layer_idx,
                hidden_size=self.hidden_size,
                num_heads=config.num_attention_heads,
                quant_config=quant_config,
                cache_config=cache_config,
                model_config=model_config,
                prefix=f"{prefix}.self_attn",
                config=config,
                qk_nope_head_dim=qk_nope_head_dim,
                qk_rope_head_dim=qk_rope_head_dim,
                v_head_dim=v_head_dim,
                q_lora_rank=config.q_lora_rank,
                kv_lora_rank=kv_lora_rank,
                use_nope=mla_use_nope,
            )

        if (
            self.is_moe
            and config.num_experts is not None
            and layer_idx >= config.first_k_dense_replace
            and layer_idx % config.moe_layer_freq == 0
        ):
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
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        attn_res_block_size = config.attn_res_block_size
        self.use_attn_residuals = attn_res_block_size is not None
        if attn_res_block_size is not None:
            self.attn_res_block_size = attn_res_block_size
            self.is_block_write_layer = layer_idx % self.attn_res_block_size == 0
            self.block_write_idx = layer_idx // self.attn_res_block_size
            self.prev_valid_blocks = cdiv(layer_idx, self.attn_res_block_size)
            self.self_attention_res_norm = RMSNorm(
                config.hidden_size, eps=config.rms_norm_eps
            )
            self.mlp_res_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
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
        attn_output = torch.empty_like(hidden_states)
        self.self_attn(
            hidden_states=hidden_states,
            positions=positions,
            output=attn_output,
        )
        return attn_output

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.use_attn_residuals:
            assert residual is not None
            return self.forward_attn_residual(positions, hidden_states, residual)

        # Self Attention
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        hidden_states = self._run_self_attn(positions, hidden_states)

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual

    def forward_attn_residual(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        block_residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        prefix_sum = hidden_states
        hidden_states = _apply_attn_res(
            prefix_sum,
            block_residual,
            self.self_attention_res_proj,
            self.self_attention_res_norm,
            self.prev_valid_blocks,
            output_norm=self.input_layernorm,
            block_write_idx=(self.block_write_idx if self.is_block_write_layer else -1),
        )

        if self.is_block_write_layer:
            prefix_sum = None

        hidden_states = self._run_self_attn(positions, hidden_states)

        if prefix_sum is None:
            prefix_sum = hidden_states
            prefix_delta = None
        else:
            prefix_delta = hidden_states

        mlp_valid_blocks = self.prev_valid_blocks + (
            1 if self.is_block_write_layer else 0
        )
        hidden_states = _apply_attn_res(
            prefix_sum,
            block_residual,
            self.mlp_res_proj,
            self.mlp_res_norm,
            mlp_valid_blocks,
            delta=prefix_delta,
            output_norm=self.post_attention_layernorm,
        )

        hidden_states = self.mlp(hidden_states)
        prefix_sum = prefix_sum + hidden_states
        return prefix_sum, block_residual


class KimiLinearModel(nn.Module, EagleModelMixin):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = vllm_config.model_config.hf_text_config
        self.config = config

        self.vocab_size = config.vocab_size

        if get_pp_group().is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                prefix=f"{prefix}.embed_tokens",
            )
        else:
            self.embed_tokens = PPMissingLayer()

        def get_layer(prefix: str):
            return KimiDecoderLayer(
                config,
                vllm_config,
                prefix,
            )

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            get_layer,
            prefix=f"{prefix}.layers",
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
        assert config.num_attention_heads % world_size == 0, (
            "num_attention_heads must be divisible by world_size"
        )

    def make_empty_intermediate_tensors(
        self,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> IntermediateTensors:
        residual_shape: tuple[int, ...] = (batch_size, self.config.hidden_size)
        if self.config.attn_res_block_size is not None:
            residual_shape = (
                batch_size,
                cdiv(self.start_layer, self.config.attn_res_block_size),
                self.config.hidden_size,
            )
        return IntermediateTensors(
            {
                "hidden_states": torch.zeros(
                    (batch_size, self.config.hidden_size), dtype=dtype, device=device
                ),
                "residual": torch.zeros(residual_shape, dtype=dtype, device=device),
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
            # attn-res `residual` is a block-state bank, not an additive
            # residual; None makes the mixin capture the prefix sum directly.
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
        **kwargs,
    ) -> torch.Tensor | IntermediateTensors | tuple[torch.Tensor, list[torch.Tensor]]:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_input_ids(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        aux_hidden_states = self._maybe_add_hidden_state(
            [], self.start_layer, hidden_states, residual
        )

        if self.config.attn_res_block_size is None:
            for layer_idx, layer in enumerate(
                self.layers[self.start_layer : self.end_layer],
                start=self.start_layer,
            ):
                hidden_states, residual = layer(
                    positions=positions,
                    hidden_states=hidden_states,
                    residual=residual,
                )
                self._maybe_add_hidden_state(
                    aux_hidden_states, layer_idx + 1, hidden_states, residual
                )

            if not get_pp_group().is_last_rank:
                return IntermediateTensors(
                    {"hidden_states": hidden_states, "residual": residual}
                )

            # NOTE: the final norm is applied in compute_logits instead of here,
            # so the MTP draft model receives the pre-norm hidden states.
            if residual is not None:
                hidden_states = hidden_states + residual
            if aux_hidden_states:
                return hidden_states, aux_hidden_states
            return hidden_states

        attn_res_block_num = cdiv(self.end_layer, self.config.attn_res_block_size)
        block_residual = hidden_states.new_empty(
            hidden_states.size(0), attn_res_block_num, hidden_states.size(1)
        )
        if residual is not None:
            block_residual[:, : residual.size(1), :].copy_(residual)
        residual = block_residual

        for layer_idx, layer in enumerate(
            self.layers[self.start_layer : self.end_layer],
            start=self.start_layer,
        ):
            hidden_states, residual = layer(
                positions=positions,
                hidden_states=hidden_states,
                residual=residual,
            )
            if (layer_idx + 1) in self.aux_hidden_state_layers:
                # AMD attn-res layer already returns prefix_sum + MLP delta as
                # hidden_states; the override drops the block bank in residual.
                self._maybe_add_hidden_state(
                    aux_hidden_states, layer_idx + 1, hidden_states, residual
                )

        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )

        hidden_states = _apply_attn_res(
            hidden_states,
            residual,
            self.output_attn_res_proj,
            self.output_attn_res_norm,
            attn_res_block_num,
        )
        # NOTE: the final norm is applied in compute_logits instead of here, so
        # the MTP draft model receives the pre-norm hidden states.
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
            # (param_name, shard_name, shard_id)
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
            stacked_params_mapping.append((".in_proj_qkvgfab", ".g_proj", 3))
        if getattr(self.config, "q_lora_rank", None) is not None:
            stacked_params_mapping += [
                (".fused_qkv_a_proj", ".q_a_proj", 0),
                (".fused_qkv_a_proj", ".kv_a_proj_with_mqa", 1),
            ]
        if self.config.is_moe:
            # Params for weights, fp8 weight scales, fp8 activation scales
            # (param_name, weight_name, expert_id, shard_id)
            expert_params_mapping = fused_moe_make_expert_params_mapping(
                self,
                ckpt_gate_proj_name="w1",
                ckpt_down_proj_name="w2",
                ckpt_up_proj_name="w3",
                num_experts=self.config.num_experts,
            )
        else:
            expert_params_mapping = []
        params_dict = dict(self.named_parameters())
        # Under the MXFP4 quant interface the routed experts register unpacked
        # params (``w13_weight``), while the compressed-tensors checkpoint names
        # them ``.weight_packed``. Rebind so the expert mapping resolves; scales
        # already share the ``.weight_scale`` suffix.
        experts_unpacked = not any(n.endswith("w13_weight_packed") for n in params_dict)
        loaded_params: set[str] = set()
        for args in weights:
            name, loaded_weight = args[0], args[1]
            kwargs: dict[str, Any] = args[2] if len(args) > 2 else {}
            if "rotary_emb.inv_freq" in name:
                continue
            if experts_unpacked and name.endswith(".weight_packed"):
                name = name.replace(".weight_packed", ".weight")

            spec_layer = get_spec_layer_idx_from_weight_name(self.config, name)
            if spec_layer is not None:
                continue  # skip spec decode layers for main model
            if "rotary_emb.cos_cached" in name or "rotary_emb.sin_cached" in name:
                # Models trained using ColossalAI may include these tensors in
                # the checkpoint. Skip them.
                continue
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                # We have mlp.experts[0].gate_proj in the checkpoint.
                # Since we handle the experts below in expert_params_mapping,
                # we need to skip here BEFORE we update the name, otherwise
                # name will be updated to mlp.experts[0].gate_up_proj, which
                # will then be updated below in expert_params_mapping
                # for mlp.experts[0].gate_gate_up_proj, which breaks load.
                if ("mlp.experts." in name) and name not in params_dict:
                    continue
                name_mapped = name.replace(weight_name, param_name)
                # Packed projections are only present on compatible layers.
                if name_mapped not in params_dict:
                    continue
                name = name_mapped
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if is_pp_missing_parameter(name, self):
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
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
                        continue
                    param = params_dict[name]
                    weight_loader = param.weight_loader
                    weight_loader(
                        param,
                        loaded_weight,
                        name,
                        expert_id=expert_id,
                        shard_id=expert_shard_id,
                    )
                    break
                else:
                    # Skip loading extra bias for GPTQ models.
                    if (
                        name.endswith(".bias")
                        and name not in params_dict
                        and not self.config.is_linear_attn
                    ):  # noqa: E501
                        continue
                    # Remapping the name of FP8 kv-scale.
                    remapped_name = maybe_remap_kv_scale_name(name, params_dict)
                    if remapped_name is None:
                        continue
                    name = remapped_name
                    if is_pp_missing_parameter(name, self):
                        continue

                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight, **kwargs)
            loaded_params.add(name)
        return loaded_params


class KimiLinearForCausalLM(
    nn.Module, HasInnerState, SupportsPP, MixtureOfExperts, IsHybrid
):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.model_config = vllm_config.model_config
        self.vllm_config = vllm_config
        self.config = self.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.quant_config = quant_config
        self.model = KimiLinearModel(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(
                self.config.vocab_size,
                self.config.hidden_size,
                quant_config=quant_config,
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
        **kwargs,
    ) -> torch.Tensor | IntermediateTensors:
        hidden_states = self.model(
            input_ids, positions, intermediate_tensors, inputs_embeds, **kwargs
        )
        return hidden_states

    @classmethod
    def get_mamba_state_dtype_from_config(
        cls,
        vllm_config: "VllmConfig",
    ) -> tuple[torch.dtype, torch.dtype]:
        return MambaStateDtypeCalculator.kda_state_dtype(
            vllm_config.model_config.dtype, vllm_config.cache_config.mamba_cache_dtype
        )

    @classmethod
    def get_mamba_state_shape_from_config(
        cls, vllm_config: "VllmConfig"
    ) -> tuple[tuple[int, int], tuple[int, int, int]]:
        parallel_config = vllm_config.parallel_config
        hf_config = vllm_config.model_config.hf_config
        tp_size = parallel_config.tensor_parallel_size
        num_spec = (
            vllm_config.speculative_config.num_speculative_tokens
            if vllm_config.speculative_config
            else 0
        )
        return MambaStateShapeCalculator.kda_state_shape(
            tp_size,
            hf_config.linear_attn_config["num_heads"],
            hf_config.linear_attn_config["head_dim"],
            conv_kernel_size=hf_config.linear_attn_config["short_conv_kernel_size"],
            num_spec=num_spec,
        )

    @classmethod
    def get_mamba_state_copy_func(
        cls,
    ) -> tuple[MambaStateCopyFunc, MambaStateCopyFunc]:
        return MambaStateCopyFuncCalculator.kda_state_copy_func()

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        # The model's final norm is applied here (not at the end of forward) so
        # that the pre-norm hidden states can be fed to the MTP draft model.
        hidden_states = self.model.norm(hidden_states, None)
        return self.logits_processor(self.lm_head, hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=(["lm_head."] if self.config.tie_word_embeddings else None),
        )
        return loader.load_weights(weights)
