# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
from collections.abc import Iterable
from copy import copy
from dataclasses import replace
from itertools import islice

import torch
import torch.nn.functional as F
from torch import nn

import vllm.envs as envs
from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig, get_layers_from_vllm_config
from vllm.distributed import (
    get_pp_group,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
)
from vllm.model_executor.layers.activation import SiluAndMul, SiluAndMulWithClamp
from vllm.model_executor.layers.attention import MLAAttention
from vllm.model_executor.layers.attention.attention import get_attention_context
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.fused_moe import (
    FusedMoEFactory,
    GateLinear,
    fused_moe_make_expert_params_mapping,
)
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
    QwenGatedDeltaNetAttention,
)
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateCopyFunc,
    MambaStateCopyFuncCalculator,
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.quantization.fp8 import Fp8Config
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    per_token_group_quant_fp8,
)
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.rotary_embedding.common import yarn_get_mscale
from vllm.model_executor.layers.sparse_attn_indexer import fused_indexer_q_rope_quant
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.models.deepseek_v32.attention import DeepseekV32Indexer
from vllm.models.kimi_k3.common.dsa_cache import KimiK3IndexerCache
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.configs.gigachat4 import GigaChat4Config
from vllm.utils.math_utils import cdiv, round_up
from vllm.utils.torch_utils import (
    LayerNameType,
    _encode_layer_name,
    _resolve_layer_name,
    direct_register_custom_op,
)
from vllm.v1.attention.backends.mla.index_group import (
    SparseMLAIndexGroupBuilder,
    get_sparse_mla_index_group_max_rows,
)
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    HiddenStateCacheSpec,
    KVCacheLayerConfig,
    MambaSpec,
    MLAAttentionSpec,
    SparseCacheRole,
)

from .deepseek_v2 import DeepSeekV2FusedQkvAProjLinear
from .interfaces import (
    EagleModelMixin,
    HasInnerState,
    IsHybrid,
    MixtureOfExperts,
    SupportsEagle3,
    SupportsPP,
)
from .utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    extract_layer_index,
    is_pp_missing_parameter,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
    sequence_parallel_chunk,
    spec_decode_needs_target_embed,
)


def _prepare_gigachat4_fp8_config(
    quant_config: QuantizationConfig | None,
) -> QuantizationConfig | None:
    if not isinstance(quant_config, Fp8Config):
        return quant_config
    ignored_layers = []
    for name in quant_config.ignored_layers:
        name = name.rstrip(".").removesuffix(".weight")
        if not name:
            raise ValueError("GigaChat4 FP8 exclusions must name a module")
        if "experts" in name.split(".")[:-1]:
            raise ValueError(
                "GigaChat4 FP8 exclusions must cover the whole experts module, "
                "not individual experts or their projections"
            )
        ignored_layers.append(name)
    quant_config = copy(quant_config)
    quant_config.ignored_layers = ignored_layers
    quant_config.ignored_layers_match_mode = "module"
    return quant_config


class GigaChat4GatedNorm(nn.Module):
    def __init__(
        self, config: GigaChat4Config, hidden_size: int, scale: float = 1.0
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = config.rms_norm_eps
        self.gating_weight = config.layernorm_gating_weight
        self.scale = scale
        self.gate_up_projection = nn.Linear(hidden_size, 16, bias=False)
        self.gate_down_projection = nn.Linear(16, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalized = x.float()
        variance = normalized.square().mean(-1, keepdim=True)
        inv_rms = torch.rsqrt(variance + self.variance_epsilon)
        normalized = normalized * inv_rms
        normalized = normalized.to(x.dtype)
        normalized = self.weight * normalized
        gate = F.linear(normalized.float(), self.gate_up_projection.weight.float())
        gate = F.silu(gate)
        gate = F.linear(gate, self.gate_down_projection.weight.float())
        output = normalized.float() * self.gating_weight
        gate = torch.sigmoid(gate)
        output = output * gate
        output = output.to(x.dtype)
        # MLA scales the completed norm, after its output dtype conversion.
        return output * self.scale


class GigaChat4GatedDeltaNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        normalized = x.float()
        variance = normalized.square().mean(-1, keepdim=True)
        inv_rms = torch.rsqrt(variance + self.variance_epsilon)
        normalized = normalized * inv_rms
        output = normalized * self.weight.float()
        gate = gate.float()
        gate = torch.sigmoid(gate)
        output = output * gate
        output = output.to(x.dtype)
        return output


class GigaChat4GatedDeltaNetAttention(QwenGatedDeltaNetAttention):
    def __init__(self, vllm_config: VllmConfig, prefix: str) -> None:
        config = vllm_config.model_config.hf_text_config
        super().__init__(
            config=config,
            vllm_config=vllm_config,
            prefix=prefix,
            gqa_interleaved_layout=True,
        )
        norm_eps = (
            getattr(config, "linear_attn_o_norm_eps", None) or config.rms_norm_eps
        )
        self.norm = GigaChat4GatedDeltaNorm(self.head_v_dim, norm_eps)
        self.gate_scale = config.linear_sigmoid_gate_scale
        self.output_mult = config.linear_attn_output_mult

    def _output_projection(
        self, core_attn_out: torch.Tensor, z: torch.Tensor
    ) -> torch.Tensor:
        core_attn_out = core_attn_out.reshape_as(z)
        output = self.norm(core_attn_out, z)
        output = output * self.gate_scale
        output = output.flatten(-2)
        output, _ = self.out_proj(output)
        if self.output_mult != 1.0:
            output = output * self.output_mult
        return output


def gigachat4_record_topk_ready(
    topk_indices_buffer: torch.Tensor, layer_name: LayerNameType
) -> None:
    layer_name = _resolve_layer_name(layer_name)
    _, layer, _, _ = get_attention_context(layer_name)
    layer.impl.record_logical_topk_ready()  # type: ignore[attr-defined]


def gigachat4_record_topk_ready_fake(
    topk_indices_buffer: torch.Tensor, layer_name: LayerNameType
) -> None:
    pass


# Order the event after the indexer's buffer write without tracing the stream.
direct_register_custom_op(
    op_name="gigachat4_record_topk_ready",
    op_func=gigachat4_record_topk_ready,
    mutates_args=["topk_indices_buffer"],
    fake_impl=gigachat4_record_topk_ready_fake,
)


class GigaChat4Indexer(DeepseekV32Indexer):
    indexer_cache_cls = KimiK3IndexerCache

    def __init__(self, vllm_config: VllmConfig, topk_indices_buffer, prefix: str):
        config = vllm_config.model_config.hf_text_config
        super().__init__(
            vllm_config=vllm_config,
            config=config,
            hidden_size=config.hidden_size,
            q_lora_rank=config.q_lora_rank,
            quant_config=vllm_config.quant_config,
            cache_config=vllm_config.cache_config,
            topk_indices_buffer=topk_indices_buffer,
            prefix=prefix,
        )
        self.k_norm = RMSNorm(
            config.index_head_dim, eps=getattr(config, "indexer_norm_eps", None) or 1e-6
        )
        self.rotary_emb = get_rope(
            config.qk_rope_head_dim,
            max_position=config.max_position_embeddings,
            rope_parameters=config.rope_parameters,
            is_neox_style=True,
        )
        self.n_head_scale = self.n_head**-0.5
        self.use_fused_indexer_q = (
            current_platform.is_cuda()
            and self.quant_block_size == self.head_dim == 128
            and self.rope_dim == 64
            and self.scale_fmt is not None
        )

    def forward(
        self, hidden_states: torch.Tensor, qr: torch.Tensor, positions: torch.Tensor
    ) -> torch.Tensor:
        q, _ = self.wq_b(qr)
        q = q.view(-1, self.n_head, self.head_dim)
        kw, _ = self.wk_weights_proj(hidden_states)
        k = kw[:, : self.head_dim]
        k = self.k_norm(k)
        weights = kw[:, self.head_dim :]
        k_pe, k_nope = k.split([self.rope_dim, self.head_dim - self.rope_dim], dim=-1)
        if self.use_fused_indexer_q and q.dtype == torch.bfloat16:
            q_fp8, weights = fused_indexer_q_rope_quant(
                positions,
                q,
                self.rotary_emb.cos_sin_cache,
                weights,
                self.softmax_scale,
                self.n_head_scale,
                self.rotary_emb.is_neox_style,
            )
            k_pe = k_pe.unsqueeze(1)
            _, k_pe = self.rotary_emb(positions, torch.empty_like(k_pe), k_pe)
        else:
            q_pe, q_nope = q.split(
                [self.rope_dim, self.head_dim - self.rope_dim], dim=-1
            )
            k_pe = k_pe.unsqueeze(1)
            q_pe, k_pe = self.rotary_emb(positions, q_pe, k_pe)
            q_pe = q_pe.reshape(-1, self.n_head, self.rope_dim)
            q = torch.cat([q_pe, q_nope], dim=-1)
            q = q.view(-1, self.head_dim)
            q_fp8, q_scale = per_token_group_quant_fp8(
                q,
                self.quant_block_size,
                column_major_scales=False,
                use_ue8m0=self.scale_fmt is not None,
            )
            q_fp8 = q_fp8.view(-1, self.n_head, self.head_dim)
            q_scale = q_scale.view(-1, self.n_head)
            weights = weights * q_scale * self.softmax_scale * self.n_head_scale
        k_pe = k_pe.reshape(-1, self.rope_dim)
        k = torch.cat([k_pe, k_nope], dim=-1)
        return self.indexer_op(hidden_states, q_fp8, k, weights)


class GigaChat4Attention(nn.Module):
    def __init__(
        self,
        vllm_config: VllmConfig,
        topk_indices_buffer: torch.Tensor | None,
        index_group_builder: SparseMLAIndexGroupBuilder | None,
        prefix: str,
        *,
        non_causal_multi_token_decode: bool = False,
    ) -> None:
        super().__init__()
        config = vllm_config.model_config.hf_text_config
        quant_config = vllm_config.quant_config
        use_dsa = bool(config.index_topk)
        num_heads = config.num_attention_heads
        num_local_heads = num_heads // get_tensor_model_parallel_world_size()
        qk_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim
        self.num_local_heads = num_local_heads
        self.qk_head_dim = qk_head_dim
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.v_head_dim = config.v_head_dim
        self.q_lora_rank = config.q_lora_rank
        self.kv_lora_rank = config.kv_lora_rank
        alpha_q = alpha_kv = 1.0
        if config.use_mla_scaling_factor:
            alpha_q = math.sqrt(config.hidden_size / config.q_lora_rank)
            alpha_kv = math.sqrt(config.hidden_size / config.kv_lora_rank)
        self.fused_qkv_a_proj = DeepSeekV2FusedQkvAProjLinear(
            config.hidden_size,
            [config.q_lora_rank, config.kv_lora_rank + config.qk_rope_head_dim],
            quant_config=quant_config,
            prefix=f"{prefix}.fused_qkv_a_proj",
        )
        self.q_a_layernorm = GigaChat4GatedNorm(config, config.q_lora_rank, alpha_q)
        self.kv_a_layernorm = GigaChat4GatedNorm(config, config.kv_lora_rank, alpha_kv)
        self.q_b_proj = ColumnParallelLinear(
            config.q_lora_rank,
            num_heads * qk_head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.q_b_proj",
        )
        self.kv_b_proj = ColumnParallelLinear(
            config.kv_lora_rank,
            num_heads * (config.qk_nope_head_dim + config.v_head_dim),
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.kv_b_proj",
        )
        self.o_proj = RowParallelLinear(
            num_heads * config.v_head_dim,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )
        self.gate_proj = (
            ColumnParallelLinear(
                config.hidden_size,
                num_heads * config.v_head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.gate_proj",
            )
            if config.gated_attention
            else None
        )
        self.rotary_emb = get_rope(
            config.qk_rope_head_dim,
            max_position=config.max_position_embeddings,
            rope_parameters=config.rope_parameters,
            is_neox_style=not config.rope_interleave,
        )
        scaling = qk_head_dim**-0.5
        rope = config.rope_parameters
        if rope["rope_type"] == "yarn" and rope.get("mscale_all_dim", 0):
            scaling *= yarn_get_mscale(rope["factor"], rope["mscale_all_dim"]) ** 2
        self.indexer = None
        if use_dsa:
            self.indexer = GigaChat4Indexer(
                vllm_config, topk_indices_buffer, f"{prefix}.indexer"
            )
        self.mla_attn = MLAAttention(
            num_heads=num_local_heads,
            scale=scaling,
            qk_nope_head_dim=config.qk_nope_head_dim,
            qk_rope_head_dim=config.qk_rope_head_dim,
            v_head_dim=config.v_head_dim,
            q_lora_rank=config.q_lora_rank,
            kv_lora_rank=config.kv_lora_rank,
            kv_b_proj=self.kv_b_proj,
            cache_config=vllm_config.cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
            indexer=self.indexer,
            use_sparse=use_dsa,
            topk_indices_buffer=topk_indices_buffer,
            index_group_builder=index_group_builder,
            non_causal_multi_token_decode=non_causal_multi_token_decode,
        )

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor):
        qkv, _ = self.fused_qkv_a_proj(hidden_states)
        q_c, kv_c, k_pe = qkv.split(
            [self.q_lora_rank, self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
        )
        q_c = self.q_a_layernorm(q_c)
        kv_c = self.kv_a_layernorm(kv_c)
        q, _ = self.q_b_proj(q_c)
        q = q.view(-1, self.num_local_heads, self.qk_head_dim)
        k_pe = k_pe.unsqueeze(1)
        q[..., self.qk_nope_head_dim :], k_pe = self.rotary_emb(
            positions, q[..., self.qk_nope_head_dim :], k_pe
        )
        if self.indexer is not None:
            self.indexer(hidden_states, q_c, positions)
            torch.ops.vllm.gigachat4_record_topk_ready(
                self.indexer.topk_indices_buffer,
                _encode_layer_name(self.mla_attn.layer_name),
            )
        output = self.mla_attn(
            q,
            kv_c,
            k_pe,
            output_shape=(
                hidden_states.shape[0],
                self.num_local_heads * self.v_head_dim,
            ),
        )
        if self.gate_proj is not None:
            gate, _ = self.gate_proj(hidden_states)
            output_dtype = output.dtype
            output = output.float()
            gate = gate.float()
            gate = torch.sigmoid(gate)
            output = output * gate
            output = output.to(output_dtype)
        output, _ = self.o_proj(output)
        return output


class GigaChat4MLP(nn.Module):
    def __init__(
        self,
        config: GigaChat4Config,
        intermediate_size: int,
        quant_config: QuantizationConfig | None,
        prefix: str,
        reduce_results: bool = True,
        is_sequence_parallel: bool = False,
    ) -> None:
        super().__init__()
        swiglu_limit = config.swiglu_limit or 0.0
        self.gate_up_proj = MergedColumnParallelLinear(
            config.hidden_size,
            [intermediate_size, intermediate_size],
            bias=False,
            quant_config=quant_config,
            disable_tp=is_sequence_parallel,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=reduce_results,
            disable_tp=is_sequence_parallel,
            prefix=f"{prefix}.down_proj",
        )
        self.act_fn = (
            SiluAndMulWithClamp(swiglu_limit) if swiglu_limit > 0 else SiluAndMul()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class GigaChat4MoE(nn.Module):
    def __init__(self, vllm_config: VllmConfig, prefix: str) -> None:
        super().__init__()
        config = vllm_config.model_config.hf_text_config
        swiglu_limit = config.swiglu_limit or 0.0
        parallel_config = vllm_config.parallel_config
        self.is_sequence_parallel = parallel_config.use_sequence_parallel_moe
        self.gate = GateLinear(
            config.hidden_size,
            config.n_routed_experts,
            out_dtype=torch.float32,
            force_fp32_compute=True,
            prefix=f"{prefix}.gate",
        )
        self.gate.e_score_correction_bias = nn.Parameter(
            torch.zeros(config.n_routed_experts, dtype=torch.float32)
        )
        self.shared_experts = (
            GigaChat4MLP(
                config,
                config.moe_intermediate_size * config.n_shared_experts,
                vllm_config.quant_config,
                f"{prefix}.shared_experts",
                reduce_results=False,
                is_sequence_parallel=self.is_sequence_parallel,
            )
            if config.n_shared_experts
            else None
        )
        self.experts = FusedMoEFactory(
            shared_experts=self.shared_experts,
            gate=self.gate,
            num_experts=config.n_routed_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            renormalize=config.norm_topk_prob,
            quant_config=vllm_config.quant_config,
            use_grouped_topk=True,
            num_expert_group=config.n_group,
            topk_group=config.topk_group,
            scoring_func=config.scoring_func,
            routed_scaling_factor=config.routed_scaling_factor,
            swiglu_limit=swiglu_limit if swiglu_limit > 0 else None,
            e_score_correction_bias=self.gate.e_score_correction_bias,
            is_sequence_parallel=self.is_sequence_parallel,
            router_logits_dtype=torch.float32,
            prefix=f"{prefix}.experts",
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        num_tokens = hidden_states.shape[0]
        if self.is_sequence_parallel:
            hidden_states = sequence_parallel_chunk(hidden_states)
        output = self.experts(hidden_states=hidden_states, router_logits=hidden_states)
        if self.is_sequence_parallel:
            output = tensor_model_parallel_all_gather(output, 0)
            output = output[:num_tokens]
        return output


class GigaChat4DecoderLayer(nn.Module):
    def __init__(
        self,
        vllm_config: VllmConfig,
        topk_indices_buffer: torch.Tensor | None,
        index_group_builder: SparseMLAIndexGroupBuilder | None,
        prefix: str,
        *,
        is_nextn: bool = False,
    ) -> None:
        super().__init__()
        config = vllm_config.model_config.hf_text_config
        layer_idx = extract_layer_index(prefix)
        self.branch_multiplier = config.branch_multiplier
        self.layer_type = (
            "full_attention" if is_nextn else config.layer_types[layer_idx]
        )
        if self.layer_type == "linear_attention":
            self.self_attn = GigaChat4GatedDeltaNetAttention(
                vllm_config, f"{prefix}.self_attn"
            )
        else:
            self.self_attn = GigaChat4Attention(
                vllm_config,
                topk_indices_buffer,
                index_group_builder,
                f"{prefix}.self_attn",
            )
        self.mlp = (
            GigaChat4MLP(
                config,
                config.intermediate_size,
                vllm_config.quant_config,
                f"{prefix}.mlp",
            )
            if is_nextn or layer_idx < config.first_k_dense_replace
            else GigaChat4MoE(vllm_config, f"{prefix}.mlp")
        )
        self.input_layernorm = GigaChat4GatedNorm(config, config.hidden_size)
        self.post_attention_layernorm = GigaChat4GatedNorm(config, config.hidden_size)
        self.post_self_attn_layernorm = GigaChat4GatedNorm(config, config.hidden_size)
        self.post_feedforward_layernorm = GigaChat4GatedNorm(config, config.hidden_size)

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        if self.layer_type == "linear_attention":
            hidden_states = self.self_attn(hidden_states)
        else:
            hidden_states = self.self_attn(positions, hidden_states)
        hidden_states = self.post_self_attn_layernorm(hidden_states)
        if self.branch_multiplier != 1.0:
            hidden_states = hidden_states * self.branch_multiplier
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        if self.branch_multiplier != 1.0:
            hidden_states = hidden_states * self.branch_multiplier
        hidden_states = residual + hidden_states
        return hidden_states


@support_torch_compile
class GigaChat4Model(nn.Module, EagleModelMixin):
    supports_aux_hidden_states_over_pp = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        self.config = config = vllm_config.model_config.hf_text_config
        self.num_redundant_experts = (
            vllm_config.parallel_config.eplb_config.num_redundant_experts
        )
        topk_indices_buffer = None
        index_group_builder = None
        if config.use_dsa:
            topk_indices_buffer = torch.empty(
                vllm_config.scheduler_config.max_num_batched_tokens,
                config.index_topk,
                dtype=torch.int32,
                device=current_platform.device_type,
            )
            index_group_builder = SparseMLAIndexGroupBuilder(
                topk_indices_buffer, get_sparse_mla_index_group_max_rows(vllm_config)
            )
        self.embed_tokens = (
            VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                quant_config=vllm_config.quant_config,
                prefix=f"{prefix}.embed_tokens",
            )
            if get_pp_group().is_first_rank
            or (config.tie_word_embeddings and get_pp_group().is_last_rank)
            or spec_decode_needs_target_embed(vllm_config)
            else PPMissingLayer()
        )
        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: GigaChat4DecoderLayer(
                vllm_config, topk_indices_buffer, index_group_builder, prefix
            ),
            prefix=f"{prefix}.layers",
        )
        self.norm = (
            GigaChat4GatedNorm(config, config.hidden_size)
            if get_pp_group().is_last_rank
            else PPMissingLayer()
        )
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states"], config.hidden_size
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors | tuple[torch.Tensor, list[torch.Tensor]]:
        if get_pp_group().is_first_rank:
            hidden_states = (
                inputs_embeds
                if inputs_embeds is not None
                else self.embed_input_ids(input_ids)
            )
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
        aux_hidden_states: list[torch.Tensor] = []
        if get_pp_group().is_first_rank:
            self._maybe_add_hidden_state(
                aux_hidden_states, self.start_layer, hidden_states, None
            )
        for layer_idx, layer in enumerate(
            islice(self.layers, self.start_layer, self.end_layer),
            start=self.start_layer,
        ):
            hidden_states = layer(positions, hidden_states)
            self._maybe_add_hidden_state(
                aux_hidden_states, layer_idx + 1, hidden_states, None
            )
        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {
                    "hidden_states": hidden_states,
                    **self.pack_local_aux_hidden_states(aux_hidden_states),
                }
            )
        hidden_states = self.norm(hidden_states)
        if self.aux_hidden_state_layers:
            return hidden_states, aux_hidden_states
        return hidden_states

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        return fused_moe_make_expert_params_mapping(
            self,
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.n_routed_experts,
            num_redundant_experts=self.num_redundant_experts,
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_mapping = (
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
            ("fused_qkv_a_proj", "q_a_proj", 0),
            ("fused_qkv_a_proj", "kv_a_proj_with_mqa", 1),
            ("wk_weights_proj", "wk", 0),
            ("wk_weights_proj", "weights_proj", 1),
        )
        expert_mapping = self.get_expert_mapping()
        params = dict(self.named_parameters())
        loaded = set()
        for name, weight in weights:
            if is_pp_missing_parameter(name, self):
                continue
            if ".experts." in name:
                for param_name, weight_name, expert_id, shard_id in expert_mapping:
                    if weight_name not in name:
                        continue
                    name = name.replace(weight_name, param_name)
                    param = params[name]
                    param.weight_loader(
                        param, weight, name, shard_id=shard_id, expert_id=expert_id
                    )
                    break
                else:
                    raise ValueError(f"Unexpected GigaChat4 expert weight: {name}")
            else:
                for param_name, weight_name, stacked_shard_id in stacked_mapping:
                    if f".{weight_name}." not in name:
                        continue
                    mapped_name = name.replace(f".{weight_name}.", f".{param_name}.")
                    if mapped_name not in params:
                        continue
                    name = mapped_name
                    param = params[name]
                    param.weight_loader(param, weight, stacked_shard_id)
                    break
                else:
                    param = params[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, weight)
            loaded.add(name)
        return loaded


def _packed_hybrid_block_size(
    vllm_config: VllmConfig,
    layer_configs: dict[str, KVCacheLayerConfig],
    mla_bytes_per_token: int,
    state_bytes: int,
    alignment: int,
) -> int | None:
    cache_config = vllm_config.cache_config
    if cache_config.mamba_page_size_padded is not None:
        return None

    from vllm.v1.core.kv_cache_planning import (
        DefaultKVCacheConfigBuilder,
        _get_packed_kv_cache_groups,
        _max_memory_usage_bytes_from_groups,
    )

    specs = [layer_config.spec for layer_config in layer_configs.values()]
    attention = [spec for spec in specs if isinstance(spec, MLAAttentionSpec)]
    num_states = sum(isinstance(spec, MambaSpec) for spec in specs)
    num_attention = sum(
        spec.cache_role != SparseCacheRole.INDEXER for spec in attention
    )
    if not num_states or not num_attention:
        return None
    row_bytes_per_token = sum(
        spec.page_size_bytes // spec.block_size for spec in attention
    )
    states_per_row = cdiv(num_states, cdiv(num_states, num_attention))
    balanced = round_up(
        max(
            cache_config.block_size,
            cdiv(states_per_row * state_bytes, row_bytes_per_token),
        ),
        alignment,
    )

    single_page_alignment = max(128, cache_config.block_size)
    if cache_config.enable_prefix_caching:
        single_page_alignment = math.lcm(single_page_alignment, alignment)
    generic = single_page_alignment * cdiv(
        state_bytes, single_page_alignment * mla_bytes_per_token
    )
    generic = max(cache_config.block_size, round_up(generic, alignment))

    config = copy(vllm_config)
    config.cache_config = copy(cache_config)
    config.cache_config.kv_cache_layout = "BLHNC"
    builder = DefaultKVCacheConfigBuilder()

    def memory_usage(block_size: int, padded: bool) -> int | None:
        config.cache_config.block_size = block_size
        mamba_block_size = (
            block_size
            if cache_config.enable_prefix_caching
            else cache_config.mamba_block_size or config.model_config.max_model_len
        )
        config.cache_config.mamba_block_size = mamba_block_size
        candidate_configs = {}
        for name, layer_config in layer_configs.items():
            spec = layer_config.spec
            if isinstance(spec, MambaSpec):
                spec = replace(
                    spec,
                    block_size=mamba_block_size,
                    page_size_padded=(
                        block_size * mla_bytes_per_token if padded else None
                    ),
                )
            else:
                spec = replace(spec, block_size=block_size)
            candidate_configs[name] = replace(layer_config, spec=spec)
        groups = _get_packed_kv_cache_groups(config, candidate_configs)
        if groups is None:
            return None
        return _max_memory_usage_bytes_from_groups(builder, config, groups)

    # Physical page alignment can make the smaller balanced block cost more.
    baseline_bytes = memory_usage(generic, padded=True)
    balanced_bytes = memory_usage(balanced, padded=False)
    if (
        baseline_bytes is not None
        and balanced_bytes is not None
        and balanced_bytes < baseline_bytes
    ):
        return balanced
    return None


class GigaChat4ForCausalLM(
    nn.Module, HasInnerState, IsHybrid, SupportsPP, SupportsEagle3, MixtureOfExperts
):
    packed_modules_mapping = {
        "gate_up_proj": ["gate_proj", "up_proj"],
        "fused_qkv_a_proj": ["q_a_proj", "kv_a_proj_with_mqa"],
        "wk_weights_proj": ["wk", "weights_proj"],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        quant_config = _prepare_gigachat4_fp8_config(vllm_config.quant_config)
        if quant_config is not vllm_config.quant_config:
            vllm_config = copy(vllm_config)
            vllm_config.quant_config = quant_config
        self.config = config = vllm_config.model_config.hf_text_config
        if (
            vllm_config.parallel_config.enable_eplb
            or vllm_config.parallel_config.eplb_config.num_redundant_experts
        ):
            raise NotImplementedError("GigaChat4 does not support EPLB")
        if config.use_dsa:
            if vllm_config.attention_config.resolve_indexer_kv_dtype("fp8") != "fp8":
                raise ValueError("GigaChat4 requires an FP8 indexer KV cache")
            if vllm_config.parallel_config.prefill_context_parallel_size > 1:
                raise NotImplementedError("GigaChat4 DSA does not support PCP")
        self.model = GigaChat4Model(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        self.lm_head = (
            ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=vllm_config.quant_config,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
            if get_pp_group().is_last_rank
            else PPMissingLayer()
        )
        if config.tie_word_embeddings and get_pp_group().is_last_rank:
            self.lm_head = self.lm_head.tie_weights(self.model.embed_tokens)
        self.logits_processor = LogitsProcessor(
            config.vocab_size, soft_cap=config.final_logit_softcapping
        )
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )
        self.moe_layers = [
            layer.mlp.experts
            for layer in self.model.layers
            if isinstance(layer, GigaChat4DecoderLayer)
            and isinstance(layer.mlp, GigaChat4MoE)
        ]
        self.expert_weights = []
        self.num_moe_layers = len(self.moe_layers)
        self.num_expert_groups = config.n_group
        self.num_shared_experts = config.n_shared_experts
        self.num_logical_experts = self.num_routed_experts = config.n_routed_experts
        self.num_redundant_experts = self.model.num_redundant_experts
        self.num_physical_experts = (
            self.num_logical_experts + self.num_redundant_experts
        )
        self.num_local_physical_experts = (
            self.moe_layers[0].moe_config.num_local_experts if self.moe_layers else 0
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors | tuple[torch.Tensor, list[torch.Tensor]]:
        return self.model(input_ids, positions, intermediate_tensors, inputs_embeds)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)

    @classmethod
    def get_packed_hybrid_block_size(
        cls,
        vllm_config: VllmConfig,
        mla_bytes_per_token: int,
        state_bytes: int,
        alignment: int,
    ) -> int | None:
        from vllm.v1.worker.utils import customize_layer_config

        parallel = vllm_config.parallel_config
        if (
            parallel.pipeline_parallel_size > 1
            or parallel.decode_context_parallel_size > 1
            or parallel.prefill_context_parallel_size > 1
            or vllm_config.attention_config.hisparse_config is not None
            or vllm_config.scheduler_config.disable_hybrid_kv_cache_manager
            or vllm_config.cache_config.kv_cache_dtype_skip_layers
            or (
                vllm_config.speculative_config is not None
                and vllm_config.speculative_config.method == "extract_hidden_states"
            )
        ):
            return None
        cache_config = vllm_config.cache_config
        layout = cache_config.kv_cache_layout or envs.VLLM_KV_CACHE_LAYOUT
        layers = get_layers_from_vllm_config(
            vllm_config,
            AttentionLayerBase,  # type: ignore[type-abstract]
        )
        indexers = [
            layer for layer in layers.values() if isinstance(layer, KimiK3IndexerCache)
        ]
        if layout not in (None, "BLHNC") or (layout is None and not indexers):
            return None
        if any(layer.cache_config is not cache_config for layer in indexers):
            return None

        config = copy(vllm_config)
        config.cache_config = copy(cache_config)
        config.cache_config.mamba_block_size = (
            cache_config.mamba_block_size or config.model_config.max_model_len
        )
        layer_configs: dict[str, KVCacheLayerConfig] = {}
        needs_128_alignment = False
        for name, layer in layers.items():
            if layer.kv_sharing_target_layer_name:
                continue
            layer_config = layer.get_kv_cache_config(config)
            if layer_config is None:
                continue
            spec = layer_config.spec
            if not isinstance(spec, (MLAAttentionSpec, MambaSpec)) or isinstance(
                spec, HiddenStateCacheSpec
            ):
                return None
            if isinstance(spec, AttentionSpec):
                backend = layer.get_attn_backend()
                layer_config = customize_layer_config(backend, layer_config, name)
                if backend.get_name() in ("FLASHINFER_MLA", "FLASHINFER_MLA_SPARSE"):
                    needs_128_alignment = True
            layer_configs[name] = replace(layer_config, is_draft=layer.is_draft_layer)
        block_size = _packed_hybrid_block_size(
            vllm_config, layer_configs, mla_bytes_per_token, state_bytes, alignment
        )
        if block_size is not None and needs_128_alignment and block_size % 128:
            return None
        return block_size

    @classmethod
    def get_mamba_state_dtype_from_config(cls, vllm_config: VllmConfig):
        return MambaStateDtypeCalculator.gated_delta_net_state_dtype(
            vllm_config.model_config.dtype,
            vllm_config.cache_config.mamba_cache_dtype,
            vllm_config.cache_config.mamba_ssm_cache_dtype,
        )

    @classmethod
    def get_mamba_state_shape_from_config(cls, vllm_config: VllmConfig):
        config = vllm_config.model_config.hf_text_config
        num_spec = (
            vllm_config.speculative_config.num_speculative_tokens
            if vllm_config.speculative_config
            else 0
        )
        return MambaStateShapeCalculator.gated_delta_net_state_shape(
            vllm_config.parallel_config.tensor_parallel_size,
            config.linear_num_key_heads,
            config.linear_num_value_heads,
            config.linear_key_head_dim,
            config.linear_value_head_dim,
            config.linear_conv_kernel_dim,
            num_spec,
        )

    @classmethod
    def get_mamba_state_copy_func(cls) -> tuple[MambaStateCopyFunc, MambaStateCopyFunc]:
        return MambaStateCopyFuncCalculator.gated_delta_net_state_copy_func()

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        return self.model.get_expert_mapping()

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        nextn_prefixes = tuple(
            f"model.layers.{idx}."
            for idx in range(
                self.config.num_hidden_layers,
                self.config.num_hidden_layers + self.config.num_nextn_predict_layers,
            )
        )
        loader = AutoWeightsLoader(self)
        weights = (
            (name, weight)
            for name, weight in weights
            if not name.startswith(nextn_prefixes)
        )
        return loader.load_weights(weights)
