# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-only Raon-1-mini model.

Raon-1-mini interleaves Raon Delta Attention (RDA) linear-attention layers with
NoPE gated MLA layers. Every layer reads from and writes to a multi-stream
residual through Raon mHC, and uses SiTU-GLU dense or sigmoid-routed MoE MLPs.
"""

from collections.abc import Callable, Iterable

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn
from transformers.configuration_utils import PreTrainedConfig

from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, ModelConfig, VllmConfig, get_current_vllm_config
from vllm.distributed import get_pp_group, get_tensor_model_parallel_world_size
from vllm.forward_context import ForwardContext, get_forward_context
from vllm.model_executor.custom_op import PluggableLayer
from vllm.model_executor.layers.activation import SituAndMul
from vllm.model_executor.layers.fused_moe import (
    FusedMoEFactory,
    GateLinear,
    fused_moe_make_expert_params_mapping,
)
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.mamba.abstract import MambaBase
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateCopyFuncCalculator,
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
    is_conv_state_dim_first,
)
from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
    causal_conv1d_fn,
    causal_conv1d_update,
)
from vllm.model_executor.layers.mla import MLAModules, MultiHeadLatentAttentionWrapper
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
    sharded_weight_loader,
)
from vllm.model_executor.utils import set_weight_attrs
from vllm.sequence import IntermediateTensors
from vllm.third_party.flash_linear_attention.ops.kda import (
    FusedRMSNormGated,
    chunk_kda,
    fused_kda_gate,
    fused_recurrent_kda,
    fused_recurrent_kda_fwd,
)
from vllm.utils.torch_utils import direct_register_custom_op
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum

from .interfaces import EagleModelMixin, HasInnerState, IsHybrid, SupportsEagle3
from .utils import WeightsMapper, make_layers, maybe_prefix


def raon_rda_attention(
    q_proj_states: torch.Tensor,
    k_proj_states: torch.Tensor,
    v_proj_states: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    value_scale: torch.Tensor,
    core_attn_out: torch.Tensor,
    layer_name: str,
) -> None:
    forward_context: ForwardContext = get_forward_context()
    layer = forward_context.no_compile_layers[layer_name]
    layer._forward(
        q_proj_states=q_proj_states,
        k_proj_states=k_proj_states,
        v_proj_states=v_proj_states,
        g=g,
        beta=beta,
        value_scale=value_scale,
        core_attn_out=core_attn_out,
    )


direct_register_custom_op(
    op_name="raon_rda_attention",
    op_func=raon_rda_attention,
    mutates_args=["core_attn_out"],
)


def _is_rda_layer(config: PreTrainedConfig, layer_idx: int) -> bool:
    # rda_layers holds 1-based layer indices.
    return layer_idx + 1 in config.rda_config["rda_layers"]


def _get_rda_state_shape(
    vllm_config: VllmConfig,
) -> tuple[tuple[int, int], tuple[int, int, int]]:
    rda_config = vllm_config.model_config.hf_text_config.rda_config
    spec_config = vllm_config.speculative_config
    return MambaStateShapeCalculator.kda_state_shape(
        vllm_config.parallel_config.tensor_parallel_size,
        rda_config["num_heads"],
        rda_config["head_dim"],
        conv_kernel_size=rda_config["conv_kernel_size"],
        num_spec=spec_config.num_speculative_tokens if spec_config else 0,
    )


class Raon1MiniMLP(nn.Module):
    def __init__(
        self,
        config: PreTrainedConfig,
        intermediate_size: int,
        quant_config: QuantizationConfig | None = None,
        reduce_results: bool = True,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            config.hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=reduce_results,
            prefix=f"{prefix}.down_proj",
        )
        self.act_fn = SituAndMul(
            config.activation_situ_beta, config.activation_situ_linear_beta
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self.gate_up_proj(x)
        x = self.act_fn(x)
        x, _ = self.down_proj(x)
        return x


class Raon1MiniMoE(nn.Module):
    def __init__(
        self,
        config: PreTrainedConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        if config.moe_router_activation_func != "sigmoid":
            raise ValueError("Raon-1-mini expects a sigmoid router.")
        self.gate = GateLinear(
            config.hidden_size,
            config.num_experts,
            out_dtype=torch.float32,
            params_dtype=torch.float32,
            prefix=f"{prefix}.gate",
        )
        self.gate.e_score_correction_bias = nn.Parameter(
            torch.empty(config.num_experts, dtype=torch.float32)
        )
        self.shared_experts = Raon1MiniMLP(
            config,
            config.moe_intermediate_size * config.num_shared_experts,
            quant_config=quant_config,
            reduce_results=False,
            prefix=f"{prefix}.shared_experts",
        )
        self.experts = FusedMoEFactory(
            shared_experts=self.shared_experts,
            num_experts=config.num_experts,
            top_k=config.num_experts_per_token,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            renormalize=config.moe_renormalize,
            quant_config=quant_config,
            activation="situ",
            activation_situ_beta=config.activation_situ_beta,
            activation_situ_linear_beta=config.activation_situ_linear_beta,
            scoring_func="sigmoid",
            e_score_correction_bias=self.gate.e_score_correction_bias,
            routed_scaling_factor=config.routed_scaling_factor,
            router_logits_dtype=torch.float32,
            prefix=f"{prefix}.experts",
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        router_logits, _ = self.gate(hidden_states)
        return self.experts(hidden_states=hidden_states, router_logits=router_logits)


class Raon1MiniMLAAttention(nn.Module):
    """NoPE MLA with a sigmoid output gate.

    The rope channels are never rotated: k_pe is shared across heads as extra
    key channels.
    """

    def __init__(
        self,
        config: PreTrainedConfig,
        quant_config: QuantizationConfig | None = None,
        cache_config: CacheConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.v_head_dim = config.v_head_dim
        self.q_lora_rank = config.q_lora_rank
        self.kv_lora_rank = config.kv_lora_rank
        tp_size = get_tensor_model_parallel_world_size()
        assert self.num_heads % tp_size == 0
        self.num_local_heads = self.num_heads // tp_size

        self.fused_qkv_a_proj = MergedColumnParallelLinear(
            self.hidden_size,
            [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim],
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.fused_qkv_a_proj",
            disable_tp=True,
        )
        self.q_a_layernorm = RMSNorm(self.q_lora_rank, eps=config.mla_latent_norm_eps)
        self.q_b_proj = ColumnParallelLinear(
            self.q_lora_rank,
            self.num_heads * self.qk_head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.q_b_proj",
        )
        self.kv_a_layernorm = RMSNorm(self.kv_lora_rank, eps=config.mla_latent_norm_eps)
        self.kv_b_proj = ColumnParallelLinear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.kv_b_proj",
        )
        self.g_proj = (
            ColumnParallelLinear(
                self.hidden_size,
                self.num_heads * self.v_head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.g_proj",
            )
            if config.mla_use_output_gate
            else None
        )
        self.o_proj = RowParallelLinear(
            self.num_heads * self.v_head_dim,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )
        mla_modules = MLAModules(
            kv_a_layernorm=self.kv_a_layernorm,
            kv_b_proj=self.kv_b_proj,
            rotary_emb=None,
            o_proj=self.o_proj,
            fused_qkv_a_proj=self.fused_qkv_a_proj,
            kv_a_proj_with_mqa=None,
            q_a_layernorm=self.q_a_layernorm,
            q_b_proj=self.q_b_proj,
            q_proj=None,
            indexer=None,
            is_sparse=False,
            topk_indices_buffer=None,
            g_proj=self.g_proj,
        )
        self.mla_attn = MultiHeadLatentAttentionWrapper(
            self.hidden_size,
            self.num_local_heads,
            self.qk_head_dim**-0.5,
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
        self, hidden_states: torch.Tensor, positions: torch.Tensor
    ) -> torch.Tensor:
        return self.mla_attn(positions, hidden_states)


# --8<-- [start:raon_delta_attention]
@PluggableLayer.register("raon_delta_attention")
class Raon1MiniDeltaAttention(PluggableLayer, MambaBase):
    """Raon Delta Attention: KDA with separate write and erase gates.

    The write gate sigmoid(b + w) and erase gate sigmoid(b) of the delta rule
    are realized as the KDA beta sigmoid(b) plus a per-head scale on the
    convolved values.
    """

    # --8<-- [end:raon_delta_attention]

    @property
    def mamba_type(self) -> MambaAttentionBackendEnum:
        return MambaAttentionBackendEnum.GDN_ATTN

    def get_state_dtype(self) -> tuple[torch.dtype, torch.dtype]:
        if self.model_config is None or self.cache_config is None:
            raise ValueError("model_config and cache_config must be set")
        return MambaStateDtypeCalculator.kda_state_dtype(
            self.model_config.dtype, self.cache_config.mamba_cache_dtype
        )

    def get_state_shape(self) -> tuple[tuple[int, int], tuple[int, int, int]]:
        return MambaStateShapeCalculator.kda_state_shape(
            self.tp_size,
            self.num_heads,
            self.head_dim,
            conv_kernel_size=self.conv_size,
            num_spec=self.num_speculative_tokens,
        )

    def __init__(
        self,
        config: PreTrainedConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        model_config: ModelConfig | None = None,
        cache_config: CacheConfig | None = None,
        num_speculative_tokens: int = 0,
    ) -> None:
        super().__init__()
        rda_config = config.rda_config
        self.tp_size = get_tensor_model_parallel_world_size()
        self.hidden_size = config.hidden_size
        self.num_heads = rda_config["num_heads"]
        self.head_dim = rda_config["head_dim"]
        self.conv_size = rda_config["conv_kernel_size"]
        self.gate_lower_bound = rda_config.get("gate_lower_bound")
        assert self.num_heads % self.tp_size == 0
        self.local_num_heads = self.num_heads // self.tp_size
        self.prefix = prefix
        self.model_config = model_config
        self.cache_config = cache_config
        self.num_speculative_tokens = num_speculative_tokens

        projection_size = self.num_heads * self.head_dim
        self.local_projection_size = projection_size // self.tp_size
        self.qkvg_proj = MergedColumnParallelLinear(
            self.hidden_size,
            [projection_size] * 4,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.qkvg_proj",
        )
        self.b_proj = ColumnParallelLinear(
            self.hidden_size,
            self.num_heads,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.b_proj",
        )
        self.rda_w_proj = ColumnParallelLinear(
            self.hidden_size,
            self.num_heads,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.rda_w_proj",
        )
        self.f_a_proj = ReplicatedLinear(
            self.hidden_size,
            config.rda_decay_rank,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.f_a_proj",
        )
        self.f_b_proj = ColumnParallelLinear(
            config.rda_decay_rank,
            projection_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.f_b_proj",
        )
        self.A_log = nn.Parameter(
            torch.empty(self.local_num_heads, dtype=torch.float32)
        )
        self.dt_bias = nn.Parameter(
            torch.empty(self.local_projection_size, dtype=torch.float32)
        )
        set_weight_attrs(self.A_log, {"weight_loader": sharded_weight_loader(0)})
        set_weight_attrs(self.dt_bias, {"weight_loader": sharded_weight_loader(0)})
        for name in ("q_conv1d", "k_conv1d", "v_conv1d"):
            conv = nn.Module()
            conv.weight = nn.Parameter(
                torch.empty(
                    self.local_projection_size, self.conv_size, dtype=torch.float32
                )
            )
            set_weight_attrs(conv.weight, {"weight_loader": sharded_weight_loader(0)})
            self.add_module(name, conv)
        self.o_norm = FusedRMSNormGated(
            self.head_dim, eps=config.rms_norm_eps, activation="sigmoid"
        )
        self.o_proj = RowParallelLinear(
            projection_size,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        compilation_config = get_current_vllm_config().compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self
        splitting_op = "vllm::raon_rda_attention"
        if (
            compilation_config.splitting_ops is not None
            and splitting_op not in compilation_config.splitting_ops
        ):
            compilation_config.splitting_ops.append(splitting_op)

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        del positions
        num_tokens = hidden_states.size(0)
        q, k, v, gate = self.qkvg_proj(hidden_states)[0].split(
            self.local_projection_size, dim=-1
        )
        beta_logits = self.b_proj(hidden_states)[0].float()
        write_logits = beta_logits + self.rda_w_proj(hidden_states)[0].float()
        beta = beta_logits.sigmoid().to(hidden_states.dtype).unsqueeze(0)
        value_scale = (F.logsigmoid(write_logits) - F.logsigmoid(beta_logits)).exp()
        g = fused_kda_gate(
            self.f_b_proj(self.f_a_proj(hidden_states)[0])[0],
            self.A_log,
            self.head_dim,
            g_bias=self.dt_bias,
            lower_bound=self.gate_lower_bound,
        ).unsqueeze(0)

        core_attn_out = torch.zeros(
            (1, num_tokens, self.local_num_heads, self.head_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        torch.ops.vllm.raon_rda_attention(
            q, k, v, g, beta, value_scale, core_attn_out, self.prefix
        )
        gate = rearrange(gate, "... (h d) -> ... h d", d=self.head_dim)
        core_attn_out = self.o_norm(core_attn_out, gate)
        core_attn_out = rearrange(core_attn_out, "1 n h d -> n (h d)")
        output[:] = self.o_proj(core_attn_out)[0]

    def _scale_values(self, v: torch.Tensor, value_scale: torch.Tensor) -> torch.Tensor:
        v = v.view(-1, self.local_num_heads, self.head_dim)
        return (v.float() * value_scale.unsqueeze(-1)).to(v.dtype)

    def _forward(
        self,
        q_proj_states: torch.Tensor,
        k_proj_states: torch.Tensor,
        v_proj_states: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        value_scale: torch.Tensor,
        core_attn_out: torch.Tensor,
    ) -> None:
        attn_metadata_map = get_forward_context().attn_metadata
        if attn_metadata_map is None:
            return
        assert isinstance(attn_metadata_map, dict)
        attn_metadata = attn_metadata_map.get(self.prefix)
        if attn_metadata is None:
            # Profile/warmup dummy runs skip mamba-family metadata.
            return
        assert isinstance(attn_metadata, GDNAttentionMetadata)
        has_initial_state = attn_metadata.has_initial_state
        spec_query_start_loc = attn_metadata.spec_query_start_loc
        query_start_loc = attn_metadata.non_spec_query_start_loc
        spec_state_indices = attn_metadata.spec_state_indices_tensor
        state_indices = attn_metadata.non_spec_state_indices_tensor
        spec_sequence_masks = attn_metadata.spec_sequence_masks
        spec_token_indx = attn_metadata.spec_token_indx
        non_spec_token_indx = attn_metadata.non_spec_token_indx
        num_accepted_tokens = attn_metadata.num_accepted_tokens
        num_actual_tokens = attn_metadata.num_actual_tokens
        conv_state, recurrent_state = self.kv_cache
        recurrent_state_active = recurrent_state[..., : self.head_dim]
        if not is_conv_state_dim_first():
            conv_state = conv_state.transpose(-1, -2)
        conv_states = conv_state.chunk(3, dim=-2)
        conv_weights = (
            self.q_conv1d.weight,
            self.k_conv1d.weight,
            self.v_conv1d.weight,
        )

        qkv = (
            q_proj_states[:num_actual_tokens],
            k_proj_states[:num_actual_tokens],
            v_proj_states[:num_actual_tokens],
        )
        g = g[:, :num_actual_tokens]
        beta = beta[:, :num_actual_tokens]
        value_scale = value_scale[:num_actual_tokens]

        spec_only = (
            spec_sequence_masks is not None
            and attn_metadata.num_prefills == 0
            and attn_metadata.num_decodes == 0
        )
        qkv_spec: tuple[torch.Tensor, ...] | None
        qkv_non_spec: tuple[torch.Tensor, ...] | None
        if spec_sequence_masks is None:
            qkv_spec = g_spec = beta_spec = scale_spec = None
            qkv_non_spec, g_non_spec, beta_non_spec = qkv, g, beta
            scale_non_spec = value_scale
        elif spec_only:
            qkv_spec, g_spec, beta_spec, scale_spec = qkv, g, beta, value_scale
            qkv_non_spec = g_non_spec = beta_non_spec = scale_non_spec = None
        else:
            assert spec_token_indx is not None and non_spec_token_indx is not None
            qkv_spec = tuple(x.index_select(0, spec_token_indx) for x in qkv)
            g_spec = g.index_select(1, spec_token_indx)
            beta_spec = beta.index_select(1, spec_token_indx)
            scale_spec = value_scale.index_select(0, spec_token_indx)
            qkv_non_spec = tuple(x.index_select(0, non_spec_token_indx) for x in qkv)
            g_non_spec = g.index_select(1, non_spec_token_indx)
            beta_non_spec = beta.index_select(1, non_spec_token_indx)
            scale_non_spec = value_scale.index_select(0, non_spec_token_indx)

        def to_heads(x: torch.Tensor) -> torch.Tensor:
            return rearrange(x, "n (h d) -> 1 n h d", d=self.head_dim)

        out_spec = None
        if spec_sequence_masks is not None:
            assert qkv_spec is not None and scale_spec is not None
            assert g_spec is not None and beta_spec is not None
            assert spec_query_start_loc is not None
            assert spec_state_indices is not None
            conv_state_indices = spec_state_indices[:, 0][
                : attn_metadata.num_spec_decodes
            ]
            q_spec, k_spec, v_spec = (
                causal_conv1d_update(
                    x,
                    state,
                    weight,
                    None,
                    activation="silu",
                    conv_state_indices=conv_state_indices,
                    num_accepted_tokens=num_accepted_tokens,
                    query_start_loc=spec_query_start_loc,
                    max_query_len=spec_state_indices.size(-1),
                    validate_data=False,
                )
                for x, state, weight in zip(qkv_spec, conv_states, conv_weights)
            )
            v_spec = self._scale_values(v_spec, scale_spec)
            out_spec, _ = fused_recurrent_kda_fwd(
                q=to_heads(q_spec).contiguous(),
                k=to_heads(k_spec).contiguous(),
                v=to_heads(v_spec.flatten(1)).contiguous(),
                g=g_spec.contiguous(),
                beta=beta_spec.contiguous(),
                scale=self.head_dim**-0.5,
                initial_state=recurrent_state_active,
                inplace_final_state=True,
                cu_seqlens=spec_query_start_loc[: attn_metadata.num_spec_decodes + 1],
                ssm_state_indices=spec_state_indices,
                num_accepted_tokens=num_accepted_tokens,
                use_qk_l2norm_in_kernel=True,
            )

        out = None
        if attn_metadata.num_prefills > 0:
            assert qkv_non_spec is not None and scale_non_spec is not None
            assert state_indices is not None and has_initial_state is not None
            q, k, v = (
                causal_conv1d_fn(
                    x.transpose(0, 1),
                    weight,
                    None,
                    activation="silu",
                    conv_states=state,
                    has_initial_state=has_initial_state,
                    cache_indices=state_indices,
                    query_start_loc=query_start_loc,
                    metadata=attn_metadata,
                ).transpose(0, 1)
                for x, state, weight in zip(qkv_non_spec, conv_states, conv_weights)
            )
            v = self._scale_values(v, scale_non_spec)
            initial_state = recurrent_state_active[state_indices].contiguous()
            initial_state.masked_fill_(~has_initial_state[:, None, None, None], 0)
            out, last_state = chunk_kda(
                q=to_heads(q),
                k=to_heads(k),
                v=to_heads(v.flatten(1)),
                g=g_non_spec,
                beta=beta_non_spec,
                initial_state=initial_state,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
                cu_seqlens=query_start_loc,
            )
            recurrent_state_active[state_indices] = last_state
        elif attn_metadata.num_decodes > 0:
            assert qkv_non_spec is not None and scale_non_spec is not None
            assert state_indices is not None and query_start_loc is not None
            decode_indices = state_indices[:num_actual_tokens]
            q, k, v = (
                causal_conv1d_update(
                    x,
                    state,
                    weight,
                    None,
                    activation="silu",
                    conv_state_indices=decode_indices,
                    validate_data=True,
                )
                for x, state, weight in zip(qkv_non_spec, conv_states, conv_weights)
            )
            v = self._scale_values(v, scale_non_spec)
            out, _ = fused_recurrent_kda(
                q=to_heads(q),
                k=to_heads(k),
                v=to_heads(v.flatten(1)),
                g=g_non_spec,
                beta=beta_non_spec,
                initial_state=recurrent_state_active,
                use_qk_l2norm_in_kernel=True,
                cu_seqlens=query_start_loc[: attn_metadata.num_decodes + 1],
                ssm_state_indices=state_indices,
            )

        if out_spec is not None and out is not None:
            assert spec_token_indx is not None and non_spec_token_indx is not None
            merged_out = torch.empty(
                (1, num_actual_tokens, *out_spec.shape[2:]),
                dtype=out_spec.dtype,
                device=out_spec.device,
            )
            merged_out.index_copy_(1, spec_token_indx, out_spec)
            merged_out.index_copy_(1, non_spec_token_indx, out)
            core_attn_out[0, :num_actual_tokens] = merged_out[0]
        elif out_spec is not None:
            core_attn_out[0, :num_actual_tokens] = out_spec[0, :num_actual_tokens]
        elif out is not None:
            core_attn_out[0, :num_actual_tokens] = out[0, :num_actual_tokens]


class Raon1MiniMHCStreamMix(nn.Module):
    def __init__(self, num_streams: int, hidden_size: int) -> None:
        super().__init__()
        width = (num_streams - 1) ** 2
        self.phi_z = nn.Parameter(torch.empty(width, num_streams * hidden_size))
        self.alpha_z = nn.Parameter(torch.empty(()))
        self.b_z = nn.Parameter(torch.empty(width))
        self.eta = nn.Parameter(torch.empty((), dtype=torch.float32))


class Raon1MiniMHC(nn.Module):
    """One Raon mHC site: reads the branch input from the residual streams and
    mixes the branch output back into them."""

    def __init__(self, config: PreTrainedConfig) -> None:
        super().__init__()
        n, hidden_size = config.raon_mhc_streams, config.hidden_size
        self.num_streams = n
        self.eps = config.raon_mhc_selector_eps
        self.post_factor = config.raon_mhc_post_factor
        self.phi_pre = nn.Parameter(torch.empty(n, n * hidden_size))
        self.phi_post = nn.Parameter(torch.empty(n, n * hidden_size))
        self.alpha_pre = nn.Parameter(torch.empty(()))
        self.alpha_post = nn.Parameter(torch.empty(()))
        self.b_pre = nn.Parameter(torch.empty(n))
        self.b_post = nn.Parameter(torch.empty(n))
        self.res = Raon1MiniMHCStreamMix(n, hidden_size)

        # Orthonormal Helmert basis Q (n x n-1) with Q^T 1 = 0.
        helmert = torch.zeros(n, n - 1, dtype=torch.float32)
        for k in range(1, n):
            helmert[:k, k - 1] = (k * (k + 1)) ** -0.5
            helmert[k, k - 1] = -k * (k * (k + 1)) ** -0.5
        eye = torch.eye(n, dtype=torch.float32)
        self.register_buffer("helmert", helmert, persistent=False)
        self.register_buffer("eye", eye, persistent=False)
        self.register_buffer("chart", 0.5 * (eye + 1 / n), persistent=False)

    def route(
        self, stream: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        n = self.num_streams
        flat = stream.flatten(1)
        weight = torch.cat((self.phi_pre, self.phi_post, self.res.phi_z))
        inv_rms = torch.rsqrt(flat.float().square().mean(-1, keepdim=True) + self.eps)
        z = F.linear(flat, weight).float() * inv_rms
        h_pre = torch.sigmoid(self.alpha_pre.float() * z[:, :n] + self.b_pre.float())
        h_post = self.post_factor * torch.sigmoid(
            self.alpha_post.float() * z[:, n : 2 * n] + self.b_post.float()
        )
        t_z = self.res.alpha_z.float() * z[:, 2 * n :] + self.res.b_z.float()
        e = self.helmert @ t_z.view(-1, n - 1, n - 1) @ self.helmert.T
        rho = (-e / self.chart).amax(dim=(-2, -1)).clamp_min(0)
        phi = self.chart + e / (1 + rho)[:, None, None]
        gamma = torch.sigmoid(self.res.eta.float())
        h_res = (1 - gamma) * self.eye + gamma * phi
        return h_pre, h_post, h_res

    def forward(
        self,
        stream: torch.Tensor,
        branch: Callable[[torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        h_pre, h_post, h_res = self.route(stream)
        streams = stream.float()
        branch_in = torch.einsum("tn,tnc->tc", h_pre, streams).to(stream.dtype)
        branch_out = branch(branch_in).float()
        mixed = torch.bmm(h_res, streams) + h_post.unsqueeze(-1) * branch_out.unsqueeze(
            1
        )
        return mixed.to(stream.dtype)


class Raon1MiniDecoderLayer(nn.Module):
    def __init__(self, vllm_config: VllmConfig, prefix: str) -> None:
        super().__init__()
        config = vllm_config.model_config.hf_text_config
        quant_config = vllm_config.quant_config
        cache_config = vllm_config.cache_config
        layer_idx = int(prefix.split(".")[-1])
        self.is_rda_layer = _is_rda_layer(config, layer_idx)
        if self.is_rda_layer:
            spec_config = vllm_config.speculative_config
            self.self_attn = Raon1MiniDeltaAttention(
                config,
                quant_config=quant_config,
                prefix=f"{prefix}.self_attn",
                model_config=vllm_config.model_config,
                cache_config=cache_config,
                num_speculative_tokens=(
                    spec_config.num_speculative_tokens if spec_config else 0
                ),
            )
        else:
            self.self_attn = Raon1MiniMLAAttention(
                config,
                quant_config=quant_config,
                cache_config=cache_config,
                prefix=f"{prefix}.self_attn",
            )

        self.is_moe_layer = bool(
            config.num_experts
            and layer_idx >= config.first_k_dense_replace
            and layer_idx % config.moe_layer_freq == 0
        )
        # Attribute names follow the checkpoint: "moe" or the dense "mlp".
        if self.is_moe_layer:
            self.moe = Raon1MiniMoE(config, quant_config, prefix=f"{prefix}.moe")
        else:
            self.mlp = Raon1MiniMLP(
                config,
                config.intermediate_size,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
            )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.self_attn_raon_mhc = Raon1MiniMHC(config)
        self.mlp_raon_mhc = Raon1MiniMHC(config)

    def _attention(self, hidden_states: torch.Tensor, positions: torch.Tensor):
        hidden_states = self.input_layernorm(hidden_states)
        if self.is_rda_layer:
            output = torch.empty_like(hidden_states)
            self.self_attn(hidden_states, positions, output)
            return output
        return self.self_attn(hidden_states, positions)

    def forward(self, positions: torch.Tensor, stream: torch.Tensor) -> torch.Tensor:
        stream = self.self_attn_raon_mhc(
            stream, lambda x: self._attention(x, positions)
        )
        mlp = self.moe if self.is_moe_layer else self.mlp
        return self.mlp_raon_mhc(
            stream, lambda x: mlp(self.post_attention_layernorm(x))
        )


@support_torch_compile
class Raon1MiniModel(nn.Module, EagleModelMixin):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_text_config
        self.config = config
        self.num_streams = config.raon_mhc_streams
        if get_pp_group().world_size > 1:
            raise NotImplementedError("Raon-1-mini does not support pipeline parallel.")
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            prefix=f"{prefix}.embed_tokens",
        )
        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: Raon1MiniDecoderLayer(vllm_config, prefix),
            prefix=f"{prefix}.layers",
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        if inputs_embeds is None:
            inputs_embeds = self.embed_input_ids(input_ids)
        stream = inputs_embeds.unsqueeze(1).expand(-1, self.num_streams, -1)
        stream = stream.contiguous()
        # Auxiliary hidden states carry all residual streams.
        aux_hidden_states = []
        for layer_idx, layer in enumerate(self.layers):
            if layer_idx in self.aux_hidden_state_layers:
                aux_hidden_states.append(stream.flatten(1))
            stream = layer(positions, stream)
        if len(self.layers) in self.aux_hidden_state_layers:
            aux_hidden_states.append(stream.flatten(1))
        hidden_states = self.norm(stream.float().mean(1).to(stream.dtype))
        if aux_hidden_states:
            return hidden_states, aux_hidden_states
        return hidden_states


class Raon1MiniForCausalLM(nn.Module, HasInnerState, IsHybrid, SupportsEagle3):
    # MXFP8 checkpoints store E8M0 expert scales as weight_scale_inv.
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_suffix={".weight_scale_inv": ".weight_scale"}
    )
    packed_modules_mapping = {
        "qkvg_proj": ["q_proj", "k_proj", "v_proj", "g_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
        "fused_qkv_a_proj": ["q_a_proj", "kv_a_proj_with_mqa"],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_text_config
        self.config = config
        self.model = Raon1MiniModel(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=vllm_config.quant_config,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        self.logits_processor = LogitsProcessor(config.vocab_size)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        return self.model(input_ids, positions, intermediate_tensors, inputs_embeds)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.logits_processor(self.lm_head, hidden_states)

    @classmethod
    def get_mamba_state_shape_from_config(
        cls, vllm_config: VllmConfig
    ) -> tuple[tuple[int, int], tuple[int, int, int]]:
        return _get_rda_state_shape(vllm_config)

    @classmethod
    def get_mamba_state_dtype_from_config(
        cls, vllm_config: VllmConfig
    ) -> tuple[torch.dtype, torch.dtype]:
        return MambaStateDtypeCalculator.kda_state_dtype(
            vllm_config.model_config.dtype,
            vllm_config.cache_config.mamba_cache_dtype,
        )

    @classmethod
    def get_mamba_state_copy_func(cls) -> tuple:
        return MambaStateCopyFuncCalculator.kda_state_copy_func()

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        return fused_moe_make_expert_params_mapping(
            self,
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.num_experts or 0,
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        weights = self.hf_to_vllm_mapper.apply(weights)
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        loaded_params: set[str] = set()
        stacked_mappings = [
            (".qkvg_proj", ".q_proj", 0),
            (".qkvg_proj", ".k_proj", 1),
            (".qkvg_proj", ".v_proj", 2),
            (".qkvg_proj", ".g_proj", 3),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
            (".fused_qkv_a_proj", ".q_a_proj", 0),
            (".fused_qkv_a_proj", ".kv_a_proj_with_mqa", 1),
        ]
        expert_mappings = self.get_expert_mapping()

        def load_param(name: str, tensor: torch.Tensor, shard_id=None) -> bool:
            if name not in params_dict:
                return False
            param = params_dict[name]
            weight_loader: Callable[..., None] = getattr(
                param, "weight_loader", default_weight_loader
            )
            if shard_id is None:
                weight_loader(param, tensor)
            elif isinstance(shard_id, int):
                weight_loader(param, tensor, shard_id)
            else:
                weight_loader(
                    param, tensor, name, expert_id=shard_id[0], shard_id=shard_id[1]
                )
            loaded_params.add(name)
            return True

        for name, weight in weights:
            if ".moe.experts." in name:
                for param_name, weight_name, expert_id, shard_id in expert_mappings:
                    if weight_name in name:
                        mapped = name.replace(weight_name, param_name)
                        load_param(mapped, weight, (expert_id, shard_id))
                        break
                continue
            # MLA layers keep a separate g_proj, so fall through to the
            # unstacked name when the stacked parameter does not exist.
            for param_suffix, weight_suffix, shard_idx in stacked_mappings:
                if weight_suffix in name and load_param(
                    name.replace(weight_suffix, param_suffix), weight, shard_idx
                ):
                    break
            else:
                load_param(name, weight)
        return loaded_params
