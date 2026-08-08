# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""vLLM implementation for BailingMoeV3ForCausalLM.

The HuggingFace reference model mixes MLA full-attention layers with Kimi
Delta Attention linear layers and Bailing MoE blocks.  This file keeps the V3
module/weight names aligned with the reference implementation while reusing
vLLM's parallel linear layers, MLA kernel, KDA kernel and fused MoE loader.
"""

import copy
from collections.abc import Callable, Iterable
from typing import TypeGuard

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from transformers.configuration_utils import PretrainedConfig

from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, ModelConfig, VllmConfig, get_current_vllm_config
from vllm.distributed import (
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.forward_context import ForwardContext, get_forward_context
from vllm.model_executor.custom_op import PluggableLayer
from vllm.model_executor.layers.activation import SiluAndMul, SwigluStepAndMul
from vllm.model_executor.layers.fused_moe import (
    FusedMoEFactory,
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
from vllm.model_executor.layers.mla import (
    MLAModules,
    MultiHeadLatentAttentionWrapper,
)
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.layers.quantization.fp8 import Fp8Config
from vllm.model_executor.layers.quantization.utils.quant_utils import is_layer_skipped
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
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

from .interfaces import HasInnerState, IsHybrid, SupportsPP
from .utils import (
    PPMissingLayer,
    WeightsMapper,
    is_pp_missing_parameter,
    make_layers,
    maybe_prefix,
)


def bailing_v3_kda_attention(
    q_proj_states: torch.Tensor,
    k_proj_states: torch.Tensor,
    v_proj_states: torch.Tensor,
    g1: torch.Tensor,
    beta: torch.Tensor,
    core_attn_out: torch.Tensor,
    layer_name: str,
) -> None:
    """Run Bailing V3's KDA state update outside the compiled graph."""
    forward_context: ForwardContext = get_forward_context()
    layer = forward_context.no_compile_layers[layer_name]
    layer._forward(
        q_proj_states=q_proj_states,
        k_proj_states=k_proj_states,
        v_proj_states=v_proj_states,
        g1=g1,
        beta=beta,
        core_attn_out=core_attn_out,
    )


def bailing_v3_kda_attention_fake(
    q_proj_states: torch.Tensor,
    k_proj_states: torch.Tensor,
    v_proj_states: torch.Tensor,
    g1: torch.Tensor,
    beta: torch.Tensor,
    core_attn_out: torch.Tensor,
    layer_name: str,
) -> None:
    return


direct_register_custom_op(
    op_name="bailing_v3_kda_attention",
    op_func=bailing_v3_kda_attention,
    mutates_args=["core_attn_out"],
    fake_impl=bailing_v3_kda_attention_fake,
)


def _is_kda_layer(
    layer_idx: int, layer_group_size: int, num_hidden_layers: int
) -> bool:
    return not (
        (layer_idx + 1) % layer_group_size == 0
        or layer_idx >= num_hidden_layers // layer_group_size * layer_group_size
    )


def _get_kda_state_shape_for_config(
    vllm_config: VllmConfig,
) -> tuple[tuple[int, int], tuple[int, int, int]]:
    config = vllm_config.model_config.hf_config
    num_spec = (
        vllm_config.speculative_config.num_speculative_tokens
        if vllm_config.speculative_config
        else 0
    )
    shapes = MambaStateShapeCalculator.kda_state_shape(
        vllm_config.parallel_config.tensor_parallel_size,
        config.num_attention_heads,
        config.head_dim,
        conv_kernel_size=config.short_conv_kernel_size,
        num_spec=num_spec,
    )

    return shapes


def _build_rope_parameters(config: PretrainedConfig) -> dict | None:
    rope_parameters = copy.deepcopy(getattr(config, "rope_parameters", None)) or {}
    if "rope_theta" not in rope_parameters and hasattr(config, "rope_theta"):
        rope_parameters["rope_theta"] = config.rope_theta
    # BailingMoeV3 stores the MLA rotary width explicitly in qk_rope_head_dim.
    # Some checkpoints also carry partial_rotary_factor=0.5, but applying it
    # here would shrink the rotary cache to half of the qk_rope_head_dim.
    rope_parameters.pop("partial_rotary_factor", None)

    rope_scaling = getattr(config, "rope_scaling", None)
    if isinstance(rope_scaling, dict):
        rope_scaling = copy.deepcopy(rope_scaling)
        if "type" in rope_scaling and "rope_type" not in rope_scaling:
            rope_scaling["rope_type"] = rope_scaling.pop("type")
        rope_scaling.pop("partial_rotary_factor", None)
        rope_parameters.update(rope_scaling)

    return rope_parameters or None


def _get_layer_swiglu_limit(limit_list: list | None, layer_idx: int) -> float | None:
    if limit_list is None or layer_idx >= len(limit_list):
        return None
    limit = limit_list[layer_idx]
    if limit in (None, 0):
        return None
    return float(limit)


def _load_a_log(param: torch.nn.Parameter, loaded_weight: torch.Tensor) -> None:
    tp_rank = get_tensor_model_parallel_rank()
    tp_size = get_tensor_model_parallel_world_size()
    if loaded_weight.ndim == 1:
        shard_size = loaded_weight.shape[0] // tp_size
        shard = loaded_weight.narrow(0, tp_rank * shard_size, shard_size)
        param.data.copy_(shard.view_as(param.data))
    else:
        sharded_weight_loader(2)(param, loaded_weight)


def _is_block_fp8_config(
    quant_config: QuantizationConfig | None,
) -> TypeGuard[Fp8Config]:
    return (
        isinstance(quant_config, Fp8Config)
        and quant_config.weight_block_size is not None
    )


def _configure_ling_fp8_quant_config(
    quant_config: QuantizationConfig | None,
) -> None:
    if _is_block_fp8_config(quant_config):
        quant_config.ignored_layers_match_mode = "suffix"


def _is_fp8_module_excluded(
    quant_config: QuantizationConfig | None,
    prefix: str,
) -> bool:
    """Match Ling's abbreviated FP8 exclusions against mapped vLLM prefixes."""
    if not _is_block_fp8_config(quant_config):
        return False

    return is_layer_skipped(
        prefix=prefix,
        ignored_layers=quant_config.ignored_layers,
        fused_mapping=quant_config.packed_modules_mapping,
        match_mode=quant_config.ignored_layers_match_mode,
    )


class _IdentityProjection(nn.Module):
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, None]:
        return x, None


class BailingMoeV3MLP(nn.Module):
    def __init__(
        self,
        intermediate_size: int,
        config: PretrainedConfig,
        quant_config: QuantizationConfig | None = None,
        reduce_results: bool = True,
        prefix: str = "",
        swiglu_limit: float | None = None,
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            config.hidden_size,
            [intermediate_size] * 2,
            bias=config.use_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            config.hidden_size,
            bias=config.use_bias,
            quant_config=quant_config,
            reduce_results=reduce_results,
            prefix=f"{prefix}.down_proj",
        )
        self.act_fn = (
            SwigluStepAndMul(limit=swiglu_limit) if swiglu_limit else SiluAndMul()
        )

    def forward(self, x):
        x, _ = self.gate_up_proj(x)
        x = self.act_fn(x)
        x, _ = self.down_proj(x)
        return x


class BailingMoeV3MLAAttention(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: QuantizationConfig | None = None,
        layer_id: int = 0,
        prefix: str = "attention",
        cache_config: CacheConfig | None = None,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.layer_id = layer_id

        self.qk_nope_head_dim = getattr(config, "qk_nope_head_dim", 128)
        self.qk_rope_head_dim = getattr(config, "qk_rope_head_dim", 64)
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.v_head_dim = getattr(config, "v_head_dim", 128)
        self.q_lora_rank = getattr(config, "q_lora_rank", None)
        self.kv_lora_rank = getattr(config, "kv_lora_rank", 512)

        tp_size = get_tensor_model_parallel_world_size()
        assert self.num_heads % tp_size == 0
        self.num_local_heads = self.num_heads // tp_size
        self.scaling = self.qk_head_dim**-0.5

        self.q_proj: ColumnParallelLinear | None
        self.kv_a_proj_with_mqa: ReplicatedLinear | None
        if self.q_lora_rank is None:
            self.q_proj = ColumnParallelLinear(
                self.hidden_size,
                self.num_heads * self.qk_head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.q_proj",
            )
            self.fused_qkv_a_proj = None
            self.q_a_layernorm = None
            self.q_b_proj = None
            self.kv_a_proj_with_mqa = ReplicatedLinear(
                self.hidden_size,
                self.kv_lora_rank + self.qk_rope_head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.kv_a_proj_with_mqa",
            )
        else:
            self.fused_qkv_a_proj = MergedColumnParallelLinear(
                self.hidden_size,
                [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim],
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.fused_qkv_a_proj",
                disable_tp=True,
            )
            self.q_a_layernorm = RMSNorm(self.q_lora_rank, eps=config.rms_norm_eps)
            self.q_b_proj = ColumnParallelLinear(
                self.q_lora_rank,
                self.num_heads * self.qk_head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.q_b_proj",
            )
            self.q_proj = None
            self.kv_a_proj_with_mqa = None

        self.kv_a_layernorm = RMSNorm(self.kv_lora_rank, eps=config.rms_norm_eps)
        self.kv_b_proj = ColumnParallelLinear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.kv_b_proj",
        )
        self.gated_attention_proj_granularity_type = getattr(
            config, "gated_attention_proj_granularity_type", None
        )
        if self.gated_attention_proj_granularity_type == "head_wise":
            g_out = self.num_heads
        elif self.gated_attention_proj_granularity_type == "element_wise":
            g_out = self.num_heads * self.v_head_dim
        else:
            g_out = 0
        self.g_proj = (
            ColumnParallelLinear(
                self.hidden_size,
                g_out,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.g_proj",
            )
            if g_out > 0
            else None
        )
        self.dense = RowParallelLinear(
            self.num_heads * self.v_head_dim,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.dense",
        )

        self.rotary_emb = get_rope(
            head_size=self.qk_rope_head_dim,
            max_position=getattr(config, "max_position_embeddings", 8192),
            is_neox_style=False,
            rope_parameters=_build_rope_parameters(config),
        )
        mla_modules = MLAModules(
            kv_a_layernorm=self.kv_a_layernorm,
            kv_b_proj=self.kv_b_proj,
            rotary_emb=self.rotary_emb,
            o_proj=_IdentityProjection(),
            fused_qkv_a_proj=self.fused_qkv_a_proj,
            kv_a_proj_with_mqa=self.kv_a_proj_with_mqa,
            q_a_layernorm=self.q_a_layernorm,
            q_b_proj=self.q_b_proj,
            q_proj=self.q_proj,
            indexer=None,
            is_sparse=False,
            topk_indices_buffer=None,
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

    def forward(self, hidden_states: torch.Tensor, positions: torch.Tensor):
        attn_out = self.mla_attn(positions, hidden_states)
        if self.g_proj is not None:
            gate = torch.sigmoid(self.g_proj(hidden_states)[0].float()).to(
                hidden_states.dtype
            )
            if self.gated_attention_proj_granularity_type == "head_wise":
                attn_out = attn_out.view(-1, self.num_local_heads, self.v_head_dim)
                attn_out = attn_out * gate.unsqueeze(-1)
                attn_out = attn_out.reshape(hidden_states.shape[0], -1)
            else:
                attn_out = attn_out * gate

        return self.dense(attn_out)[0]


# --8<-- [start:bailing_moe_v3_kimi_delta_attention]
@PluggableLayer.register("bailing_moe_v3_kimi_delta_attention")
class BailingMoeV3KimiDeltaAttention(PluggableLayer, MambaBase):
    # --8<-- [end:bailing_moe_v3_kimi_delta_attention]

    @property
    def mamba_type(self) -> MambaAttentionBackendEnum:
        return MambaAttentionBackendEnum.GDN_ATTN

    def get_state_dtype(
        self,
    ) -> tuple[torch.dtype, torch.dtype]:
        if self.model_config is None or self.cache_config is None:
            raise ValueError("model_config and cache_config must be set")
        return MambaStateDtypeCalculator.kda_state_dtype(
            self.model_config.dtype, self.cache_config.mamba_cache_dtype
        )

    def get_state_shape(
        self,
    ) -> tuple[tuple[int, int], tuple[int, int, int]]:
        return MambaStateShapeCalculator.kda_state_shape(
            self.tp_size,
            self.num_heads,
            self.head_dim,
            conv_kernel_size=self.conv_size,
            num_spec=self.num_speculative_tokens,
        )

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: QuantizationConfig | None = None,
        layer_id: int = 0,
        prefix: str = "attention",
        model_config: ModelConfig | None = None,
        cache_config: CacheConfig | None = None,
        num_speculative_tokens: int = 0,
    ) -> None:
        super().__init__()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.hidden_size = config.hidden_size
        self.head_dim = config.head_dim
        self.num_heads = config.num_attention_heads
        self.local_num_heads = self.num_heads // self.tp_size
        self.conv_size = config.short_conv_kernel_size
        self.layer_id = layer_id
        self.prefix = prefix
        self.model_config = model_config
        self.cache_config = cache_config
        self.num_speculative_tokens = num_speculative_tokens
        self.safe_gate = getattr(config, "kda_safe_gate", True)
        self.lower_bound = getattr(config, "kda_lower_bound", -5.0)
        if not getattr(config, "no_kda_lora", False):
            raise ValueError("BailingMoeV3 KDA currently expects no_kda_lora=True")

        projection_size = self.head_dim * self.num_heads
        self.projection_size_per_partition = projection_size // self.tp_size
        self.separate_b_proj = _is_fp8_module_excluded(quant_config, f"{prefix}.b_proj")
        self.qkv_proj: MergedColumnParallelLinear | None
        self.b_proj: ColumnParallelLinear | None
        self.qkvb_proj: MergedColumnParallelLinear | None
        if self.separate_b_proj:
            self.qkv_proj = MergedColumnParallelLinear(
                self.hidden_size,
                [projection_size, projection_size, projection_size],
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.qkv_proj",
            )
            self.b_proj = ColumnParallelLinear(
                self.hidden_size,
                self.num_heads,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.b_proj",
            )
            self.qkvb_proj = None
        else:
            self.qkvb_proj = MergedColumnParallelLinear(
                self.hidden_size,
                [projection_size, projection_size, projection_size, self.num_heads],
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.qkvb_proj",
            )
            self.qkv_proj = None
            self.b_proj = None
        self.f_proj = ColumnParallelLinear(
            self.hidden_size,
            projection_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.f_proj",
        )
        self.dt_bias = nn.Parameter(
            torch.empty(projection_size // self.tp_size, dtype=torch.float32)
        )
        set_weight_attrs(self.dt_bias, {"weight_loader": sharded_weight_loader(0)})

        self.q_conv1d = ColumnParallelLinear(
            input_size=self.conv_size,
            output_size=projection_size,
            bias=False,
            params_dtype=torch.float32,
            prefix=f"{prefix}.q_conv1d",
        )
        self.k_conv1d = ColumnParallelLinear(
            input_size=self.conv_size,
            output_size=projection_size,
            bias=False,
            params_dtype=torch.float32,
            prefix=f"{prefix}.k_conv1d",
        )
        self.v_conv1d = ColumnParallelLinear(
            input_size=self.conv_size,
            output_size=projection_size,
            bias=False,
            params_dtype=torch.float32,
            prefix=f"{prefix}.v_conv1d",
        )
        self.q_conv1d.weight.data = self.q_conv1d.weight.data.unsqueeze(1)
        self.k_conv1d.weight.data = self.k_conv1d.weight.data.unsqueeze(1)
        self.v_conv1d.weight.data = self.v_conv1d.weight.data.unsqueeze(1)

        self.A_log = nn.Parameter(
            torch.empty(1, 1, self.local_num_heads, 1, dtype=torch.float32)
        )
        set_weight_attrs(self.A_log, {"weight_loader": _load_a_log})

        self.g_proj = ColumnParallelLinear(
            self.hidden_size,
            projection_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.g_proj",
        )
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
        splitting_op = "vllm::bailing_v3_kda_attention"
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
        if self.separate_b_proj:
            assert self.qkv_proj is not None and self.b_proj is not None
            qkv = self.qkv_proj(hidden_states)[0]
            q, k, v = qkv.split(self.projection_size_per_partition, dim=-1)
            beta_logits = self.b_proj(hidden_states)[0]
        else:
            assert self.qkvb_proj is not None
            qkvb = self.qkvb_proj(hidden_states)[0]
            q, k, v, beta_logits = qkvb.split(
                [
                    self.projection_size_per_partition,
                    self.projection_size_per_partition,
                    self.projection_size_per_partition,
                    self.local_num_heads,
                ],
                dim=-1,
            )
        beta = beta_logits.float().sigmoid().unsqueeze(0)
        g1 = self.f_proj(hidden_states)[0]
        g1 = fused_kda_gate(
            g1,
            self.A_log,
            self.head_dim,
            g_bias=self.dt_bias,
            lower_bound=self.lower_bound if self.safe_gate else None,
        )
        g1 = g1.unsqueeze(0)
        g2 = rearrange(
            self.g_proj(hidden_states)[0], "... (h d) -> ... h d", d=self.head_dim
        )

        core_attn_out = torch.zeros(
            (1, num_tokens, self.local_num_heads, self.head_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        torch.ops.vllm.bailing_v3_kda_attention(
            q, k, v, g1, beta, core_attn_out, self.prefix
        )
        core_attn_out = self.o_norm(core_attn_out, g2)
        core_attn_out = rearrange(core_attn_out, "1 n h d -> n (h d)")
        output[:] = self.o_proj(core_attn_out)[0]

    def _forward(
        self,
        q_proj_states: torch.Tensor,
        k_proj_states: torch.Tensor,
        v_proj_states: torch.Tensor,
        g1: torch.Tensor,
        beta: torch.Tensor,
        core_attn_out: torch.Tensor,
    ) -> None:
        forward_context = get_forward_context()
        attn_metadata_map = forward_context.attn_metadata
        if attn_metadata_map is None:
            return

        assert isinstance(attn_metadata_map, dict)
        attn_metadata = attn_metadata_map[self.prefix]
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
        conv_state_q, conv_state_k, conv_state_v = conv_state.chunk(3, dim=-2)

        q_proj_states = q_proj_states[:num_actual_tokens]
        k_proj_states = k_proj_states[:num_actual_tokens]
        v_proj_states = v_proj_states[:num_actual_tokens]
        g1 = g1[:, :num_actual_tokens]
        beta = beta[:, :num_actual_tokens]

        spec_only = (
            spec_sequence_masks is not None
            and attn_metadata.num_prefills == 0
            and attn_metadata.num_decodes == 0
        )
        if spec_sequence_masks is not None:
            assert spec_token_indx is not None
            assert non_spec_token_indx is not None
            if spec_only:
                q_proj_states_spec = q_proj_states
                k_proj_states_spec = k_proj_states
                v_proj_states_spec = v_proj_states
                g1_spec = g1
                beta_spec = beta
                q_proj_states_non_spec = None
                k_proj_states_non_spec = None
                v_proj_states_non_spec = None
                g1_non_spec = None
                beta_non_spec = None
            else:
                q_proj_states_spec = q_proj_states.index_select(0, spec_token_indx)
                k_proj_states_spec = k_proj_states.index_select(0, spec_token_indx)
                v_proj_states_spec = v_proj_states.index_select(0, spec_token_indx)
                g1_spec = g1.index_select(1, spec_token_indx)
                beta_spec = beta.index_select(1, spec_token_indx)
                q_proj_states_non_spec = q_proj_states.index_select(
                    0, non_spec_token_indx
                )
                k_proj_states_non_spec = k_proj_states.index_select(
                    0, non_spec_token_indx
                )
                v_proj_states_non_spec = v_proj_states.index_select(
                    0, non_spec_token_indx
                )
                g1_non_spec = g1.index_select(1, non_spec_token_indx)
                beta_non_spec = beta.index_select(1, non_spec_token_indx)
        else:
            q_proj_states_spec = None
            k_proj_states_spec = None
            v_proj_states_spec = None
            g1_spec = None
            beta_spec = None
            q_proj_states_non_spec = q_proj_states
            k_proj_states_non_spec = k_proj_states
            v_proj_states_non_spec = v_proj_states
            g1_non_spec = g1
            beta_non_spec = beta

        q_conv_weights = self.q_conv1d.weight.view(
            self.q_conv1d.weight.size(0), self.q_conv1d.weight.size(2)
        )
        k_conv_weights = self.k_conv1d.weight.view(
            self.k_conv1d.weight.size(0), self.k_conv1d.weight.size(2)
        )
        v_conv_weights = self.v_conv1d.weight.view(
            self.v_conv1d.weight.size(0), self.v_conv1d.weight.size(2)
        )

        if spec_sequence_masks is not None:
            assert q_proj_states_spec is not None
            assert k_proj_states_spec is not None
            assert v_proj_states_spec is not None
            assert spec_state_indices is not None
            q_spec = causal_conv1d_update(
                q_proj_states_spec,
                conv_state_q,
                q_conv_weights,
                self.q_conv1d.bias,
                activation="silu",
                conv_state_indices=spec_state_indices[:, 0][
                    : attn_metadata.num_spec_decodes
                ],
                num_accepted_tokens=num_accepted_tokens,
                query_start_loc=spec_query_start_loc,
                max_query_len=spec_state_indices.size(-1),
                validate_data=False,
            )
            k_spec = causal_conv1d_update(
                k_proj_states_spec,
                conv_state_k,
                k_conv_weights,
                self.k_conv1d.bias,
                activation="silu",
                conv_state_indices=spec_state_indices[:, 0][
                    : attn_metadata.num_spec_decodes
                ],
                num_accepted_tokens=num_accepted_tokens,
                query_start_loc=spec_query_start_loc,
                max_query_len=spec_state_indices.size(-1),
                validate_data=False,
            )
            v_spec = causal_conv1d_update(
                v_proj_states_spec,
                conv_state_v,
                v_conv_weights,
                self.v_conv1d.bias,
                activation="silu",
                conv_state_indices=spec_state_indices[:, 0][
                    : attn_metadata.num_spec_decodes
                ],
                num_accepted_tokens=num_accepted_tokens,
                query_start_loc=spec_query_start_loc,
                max_query_len=spec_state_indices.size(-1),
                validate_data=False,
            )
        else:
            q_spec = k_spec = v_spec = None

        if attn_metadata.num_prefills > 0:
            assert q_proj_states_non_spec is not None
            assert k_proj_states_non_spec is not None
            assert v_proj_states_non_spec is not None
            q = causal_conv1d_fn(
                q_proj_states_non_spec.transpose(0, 1),
                q_conv_weights,
                self.q_conv1d.bias,
                activation="silu",
                conv_states=conv_state_q,
                has_initial_state=has_initial_state,
                cache_indices=state_indices,
                query_start_loc=query_start_loc,
                metadata=attn_metadata,
            ).transpose(0, 1)
            k = causal_conv1d_fn(
                k_proj_states_non_spec.transpose(0, 1),
                k_conv_weights,
                self.k_conv1d.bias,
                activation="silu",
                conv_states=conv_state_k,
                has_initial_state=has_initial_state,
                cache_indices=state_indices,
                query_start_loc=query_start_loc,
                metadata=attn_metadata,
            ).transpose(0, 1)
            v = causal_conv1d_fn(
                v_proj_states_non_spec.transpose(0, 1),
                v_conv_weights,
                self.v_conv1d.bias,
                activation="silu",
                conv_states=conv_state_v,
                has_initial_state=has_initial_state,
                cache_indices=state_indices,
                query_start_loc=query_start_loc,
                metadata=attn_metadata,
            ).transpose(0, 1)
        elif attn_metadata.num_decodes > 0:
            assert q_proj_states_non_spec is not None
            assert k_proj_states_non_spec is not None
            assert v_proj_states_non_spec is not None
            assert state_indices is not None
            decode_indices = state_indices[:num_actual_tokens]
            q = causal_conv1d_update(
                q_proj_states_non_spec,
                conv_state_q,
                q_conv_weights,
                self.q_conv1d.bias,
                activation="silu",
                conv_state_indices=decode_indices,
                validate_data=True,
            )
            k = causal_conv1d_update(
                k_proj_states_non_spec,
                conv_state_k,
                k_conv_weights,
                self.k_conv1d.bias,
                activation="silu",
                conv_state_indices=decode_indices,
                validate_data=True,
            )
            v = causal_conv1d_update(
                v_proj_states_non_spec,
                conv_state_v,
                v_conv_weights,
                self.v_conv1d.bias,
                activation="silu",
                conv_state_indices=decode_indices,
                validate_data=True,
            )
        else:
            q = k = v = None

        if q_spec is not None:
            q_spec, k_spec, v_spec = map(
                lambda x: rearrange(x, "n (h d) -> 1 n h d", d=self.head_dim),
                (q_spec, k_spec, v_spec),
            )
        if q is not None:
            q, k, v = map(
                lambda x: rearrange(x, "n (h d) -> 1 n h d", d=self.head_dim),
                (q, k, v),
            )
        if spec_sequence_masks is not None:
            assert g1_spec is not None
            assert beta_spec is not None
            assert spec_query_start_loc is not None
            assert spec_state_indices is not None
            out_spec, _ = fused_recurrent_kda_fwd(
                q=q_spec.contiguous(),
                k=k_spec.contiguous(),
                v=v_spec.contiguous(),
                g=g1_spec.contiguous(),
                beta=beta_spec.contiguous(),
                scale=self.head_dim**-0.5,
                initial_state=recurrent_state_active,
                inplace_final_state=True,
                cu_seqlens=spec_query_start_loc[: attn_metadata.num_spec_decodes + 1],
                ssm_state_indices=spec_state_indices,
                num_accepted_tokens=num_accepted_tokens,
                use_qk_l2norm_in_kernel=True,
            )
        else:
            out_spec = None

        if attn_metadata.num_prefills > 0:
            assert q is not None and k is not None and v is not None
            assert g1_non_spec is not None and beta_non_spec is not None
            assert state_indices is not None
            assert has_initial_state is not None
            zero_idx = state_indices[~has_initial_state]
            recurrent_state[zero_idx] = 0
            initial_state = recurrent_state_active[state_indices].contiguous()
            out, last_state = chunk_kda(
                q=q,
                k=k,
                v=v,
                g=g1_non_spec,
                beta=beta_non_spec,
                initial_state=initial_state,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
                safe_gate=self.safe_gate,
                lower_bound=self.lower_bound,
                cu_seqlens=query_start_loc,
            )
            recurrent_state_active[state_indices] = last_state
        elif attn_metadata.num_decodes > 0:
            assert q is not None and k is not None and v is not None
            assert g1_non_spec is not None and beta_non_spec is not None
            assert state_indices is not None
            assert query_start_loc is not None
            out, _ = fused_recurrent_kda(
                q=q,
                k=k,
                v=v,
                g=g1_non_spec,
                beta=beta_non_spec,
                initial_state=recurrent_state_active,
                use_qk_l2norm_in_kernel=True,
                safe_gate=self.safe_gate,
                lower_bound=self.lower_bound,
                cu_seqlens=query_start_loc[: attn_metadata.num_decodes + 1],
                ssm_state_indices=state_indices,
            )
        else:
            out = None

        if spec_sequence_masks is not None and out is not None:
            assert spec_token_indx is not None and non_spec_token_indx is not None
            merged_out = torch.empty(
                (1, num_actual_tokens, *out_spec.shape[2:]),
                dtype=out_spec.dtype,
                device=out_spec.device,
            )
            merged_out.index_copy_(1, spec_token_indx, out_spec)
            merged_out.index_copy_(1, non_spec_token_indx, out)
            core_attn_out[0, :num_actual_tokens] = merged_out[0, :num_actual_tokens]
        elif spec_sequence_masks is not None:
            core_attn_out[0, :num_actual_tokens] = out_spec[0, :num_actual_tokens]
        else:
            core_attn_out[0, :num_actual_tokens] = out[0, :num_actual_tokens]


class BailingMoeV3Gate(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        params_dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if params_dtype is None:
            params_dtype = torch.float32
        self.weight = nn.Parameter(
            torch.empty((config.num_experts, config.hidden_size), dtype=params_dtype)
        )
        self.expert_bias = nn.Parameter(
            torch.empty((config.num_experts,), dtype=torch.float32)
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return F.linear(hidden_states.to(self.weight.dtype), self.weight).to(
            hidden_states.dtype
        )


class BailingMoeV3MoE(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: QuantizationConfig | None = None,
        layer_id: int = 0,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.hidden_size = config.hidden_size
        self.routed_scaling_factor = getattr(config, "routed_scaling_factor", 1.0)
        self.gate = BailingMoeV3Gate(config)
        self.expert_swiglu_limit = _get_layer_swiglu_limit(
            getattr(config, "expert_swiglu_limit_list", None), layer_id
        )
        self.share_expert_swiglu_limit = _get_layer_swiglu_limit(
            getattr(config, "share_expert_swiglu_limit_list", None), layer_id
        )
        shared_intermediate = (
            config.moe_shared_expert_intermediate_size * config.num_shared_experts
        )
        self.shared_experts = BailingMoeV3MLP(
            intermediate_size=shared_intermediate,
            config=config,
            quant_config=quant_config,
            reduce_results=False,
            prefix=f"{prefix}.shared_experts",
            swiglu_limit=self.share_expert_swiglu_limit,
        )
        self.experts = FusedMoEFactory(
            shared_experts=self.shared_experts,
            num_experts=self.num_experts,
            top_k=self.top_k,
            hidden_size=self.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            renormalize=True,
            quant_config=quant_config,
            activation="silu",
            swiglu_limit=self.expert_swiglu_limit,
            prefix=f"{prefix}.experts",
            scoring_func="sigmoid",
            routed_scaling_factor=self.routed_scaling_factor,
            e_score_correction_bias=self.gate.expert_bias,
            num_expert_group=config.n_group,
            topk_group=config.topk_group,
            use_grouped_topk=True,
            router_logits_dtype=torch.float32,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        num_tokens, hidden_size = hidden_states.shape
        hidden_states = hidden_states.contiguous().view(-1, hidden_size)
        router_logits = self.gate(hidden_states.to(torch.float32))
        hidden_states = self.experts(
            hidden_states=hidden_states, router_logits=router_logits
        )
        return hidden_states.view(num_tokens, hidden_size)


class BailingMoeV3DecoderLayer(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: QuantizationConfig | None = None,
        layer_id: int = 0,
        prefix: str = "layer",
        model_config: ModelConfig | None = None,
        cache_config: CacheConfig | None = None,
        num_speculative_tokens: int = 0,
    ) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.hidden_size = config.hidden_size
        self.is_kda_layer = _is_kda_layer(
            layer_id, config.layer_group_size, config.num_hidden_layers
        )
        if self.is_kda_layer:
            self.self_attn = BailingMoeV3KimiDeltaAttention(
                config,
                quant_config=quant_config,
                layer_id=layer_id,
                prefix=f"{prefix}.self_attn",
                model_config=model_config,
                cache_config=cache_config,
                num_speculative_tokens=num_speculative_tokens,
            )
        else:
            self.self_attn = BailingMoeV3MLAAttention(
                config,
                quant_config=quant_config,
                layer_id=layer_id,
                prefix=f"{prefix}.self_attn",
                cache_config=cache_config,
            )

        if config.num_experts is not None and layer_id >= config.first_k_dense_replace:
            self.mlp = BailingMoeV3MoE(
                config,
                quant_config=quant_config,
                layer_id=layer_id,
                prefix=f"{prefix}.mlp",
            )
        else:
            self.mlp = BailingMoeV3MLP(
                intermediate_size=config.intermediate_size,
                config=config,
                quant_config=quant_config,
                reduce_results=True,
                prefix=f"{prefix}.mlp",
            )

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        if self.is_kda_layer:
            attn_output = torch.zeros_like(hidden_states)
            self.self_attn(hidden_states, positions, attn_output)
        else:
            attn_output = self.self_attn(hidden_states, positions)

        hidden_states, residual = self.post_attention_layernorm(attn_output, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        "positions": -1,
        "intermediate_tensors": 0,
        "inputs_embeds": 0,
    }
)
class BailingMoeV3Model(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        model_config = vllm_config.model_config
        quant_config = vllm_config.quant_config
        cache_config = vllm_config.cache_config
        num_speculative_tokens = (
            vllm_config.speculative_config.num_speculative_tokens
            if vllm_config.speculative_config
            else 0
        )

        self.config = config
        self.vocab_size = config.vocab_size
        self.embed_dim = config.hidden_size
        self.num_layers = config.num_hidden_layers

        if get_pp_group().is_first_rank:
            self.word_embeddings = VocabParallelEmbedding(
                self.vocab_size,
                self.embed_dim,
                org_num_embeddings=self.vocab_size,
            )
        else:
            self.word_embeddings = PPMissingLayer()

        def layer_fn(prefix: str):
            layer_idx = int(prefix.split(".")[-1])
            return BailingMoeV3DecoderLayer(
                config=config,
                quant_config=quant_config,
                layer_id=layer_idx,
                prefix=prefix,
                model_config=model_config,
                cache_config=cache_config,
                num_speculative_tokens=num_speculative_tokens,
            )

        self.start_layer, self.end_layer, self.layers = make_layers(
            self.num_layers, layer_fn, prefix=f"{prefix}.layers"
        )

        if get_pp_group().is_last_rank:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()

    @property
    def embed_tokens(self) -> nn.Module:
        return self.word_embeddings

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.word_embeddings(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        if get_pp_group().is_first_rank:
            hidden_states = (
                self.word_embeddings(input_ids)
                if inputs_embeds is None
                else inputs_embeds
            )
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        for layer in self.layers[self.start_layer : self.end_layer]:
            hidden_states, residual = layer(hidden_states, positions, residual)

        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )
        if residual is not None:
            hidden_states, _ = self.norm(hidden_states, residual)
        else:
            hidden_states = self.norm(hidden_states)
        return hidden_states

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        return fused_moe_make_expert_params_mapping(
            self,
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.num_experts,
        )


class BailingMoeV3ForCausalLM(nn.Module, HasInnerState, IsHybrid, SupportsPP):
    hf_to_vllm_mapper = WeightsMapper(orig_to_new_substr={".attention.": ".self_attn."})

    packed_modules_mapping = {
        "gate_up_proj": ["gate_proj", "up_proj"],
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "qkvb_proj": ["q_proj", "k_proj", "v_proj", "b_proj"],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        _configure_ling_fp8_quant_config(quant_config)
        self.config = config
        self.quant_config = quant_config
        self.model = BailingMoeV3Model(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
            self.logits_processor = LogitsProcessor(config.vocab_size)
        else:
            self.lm_head = PPMissingLayer()

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        return self.model(input_ids, positions, intermediate_tensors, inputs_embeds)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.logits_processor(self.lm_head, hidden_states)

    def make_empty_intermediate_tensors(
        self, batch_size: int, dtype: torch.dtype, device: torch.device
    ) -> IntermediateTensors:
        return IntermediateTensors(
            {
                "hidden_states": torch.zeros(
                    (batch_size, self.config.hidden_size), dtype=dtype, device=device
                ),
                "residual": torch.zeros(
                    (batch_size, self.config.hidden_size), dtype=dtype, device=device
                ),
            }
        )

    @classmethod
    def get_mamba_state_shape_from_config(
        cls, vllm_config: VllmConfig
    ) -> tuple[tuple[int, int], tuple[int, int, int]]:
        return _get_kda_state_shape_for_config(vllm_config)

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
        return self.model.get_expert_mapping()

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        weights = self.hf_to_vllm_mapper.apply(weights)
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        loaded_params: set[str] = set()
        stacked_mappings = [
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
            (".qkv_proj", ".q_proj", 0),
            (".qkv_proj", ".k_proj", 1),
            (".qkv_proj", ".v_proj", 2),
            (".qkvb_proj", ".q_proj", 0),
            (".qkvb_proj", ".k_proj", 1),
            (".qkvb_proj", ".v_proj", 2),
            (".qkvb_proj", ".b_proj", 3),
            (".fused_qkv_a_proj", ".q_a_proj", 0),
            (".fused_qkv_a_proj", ".kv_a_proj_with_mqa", 1),
        ]
        expert_mappings = list(self.get_expert_mapping())

        def load_param(name: str, tensor: torch.Tensor, shard_id=None) -> bool:
            if name not in params_dict or is_pp_missing_parameter(name, self):
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

        def normalize_name(name: str) -> str | None:
            if name.startswith("model.layers."):
                layer_idx = int(name.split("model.layers.")[1].split(".")[0])
                if layer_idx >= self.config.num_hidden_layers:
                    return None
            return maybe_remap_kv_scale_name(name, params_dict)

        for orig_name, weight in weights:
            name = normalize_name(orig_name)
            if name is None:
                continue
            loaded = False
            for param_suf, weight_suf, stacked_shard_id in stacked_mappings:
                if weight_suf in name:
                    mapped = name.replace(weight_suf, param_suf)
                    if load_param(mapped, weight, stacked_shard_id):
                        loaded = True
                        break
            if loaded:
                continue

            if ".mlp.experts." in name:
                for param_name, weight_name, expert_id, shard_id in expert_mappings:
                    if weight_name in name:
                        mapped = name.replace(weight_name, param_name)
                        load_param(mapped, weight, (expert_id, shard_id))
                        break
                continue

            load_param(name, weight)
        return loaded_params
