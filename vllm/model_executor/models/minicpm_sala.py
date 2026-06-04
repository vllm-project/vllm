# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-only MiniCPM-SALA model.

MVP scope:
* `minicpm4` layers use dense vLLM Attention as a correctness fallback for
  short contexts.
* `lightning-attn` layers use SALALightningAttention (Simple GLA recurrent
  state via LinearAttentionMetadata).
"""

from __future__ import annotations

import math
from collections.abc import Iterable

import torch
import torch.nn.functional as F
from torch import nn

from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, ModelConfig, VllmConfig
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.distributed.parallel_state import get_pp_group
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateCopyFunc,
    MambaStateCopyFuncCalculator,
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
)
from vllm.model_executor.layers.mamba.sala_simple_gla import SALALightningAttention
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.configs.minicpm_sala import MiniCPMSALAConfig

from .interfaces import (
    HasInnerState,
    IsHybrid,
    SupportsMambaPrefixCaching,
    SupportsPP,
    SupportsQuant,
)
from .utils import (
    default_weight_loader,
    is_pp_missing_parameter,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)


class MiniCPMSALAMLP(nn.Module):
    def __init__(
        self,
        config: MiniCPMSALAConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        if config.hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {config.hidden_act}. "
                "MiniCPM-SALA MVP supports only silu."
            )
        self.gate_up_proj = MergedColumnParallelLinear(
            config.hidden_size,
            [config.intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            config.intermediate_size,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.down_proj",
        )
        self.act_fn = SiluAndMul()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(hidden_states)
        hidden_states = self.act_fn(gate_up)
        hidden_states, _ = self.down_proj(hidden_states)
        return hidden_states


class SALASparseAttention(nn.Module):
    """Dense attention fallback for MiniCPM-SALA `minicpm4` layers.

    This is intentionally not the full InfLLM-V2 sparse path. It matches the
    public checkpoint contract for short-context MVP correctness: GQA enabled,
    optional output gate, and `attn_use_rope=False` for openbmb/MiniCPM-SALA.
    """

    def __init__(
        self,
        config: MiniCPMSALAConfig,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.total_num_heads = config.num_attention_heads
        self.total_num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.scaling = self.head_dim**-0.5
        self.use_rope = config.attn_use_rope
        self.use_output_gate = config.attn_use_output_gate

        tp_size = get_tensor_model_parallel_world_size()
        assert self.total_num_heads % tp_size == 0
        if self.total_num_kv_heads >= tp_size:
            assert self.total_num_kv_heads % tp_size == 0
        else:
            assert tp_size % self.total_num_kv_heads == 0
        self.num_heads = self.total_num_heads // tp_size
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim

        self.qkv_proj = QKVParallelLinear(
            config.hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=config.attention_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )
        if self.use_output_gate:
            self.o_gate = ColumnParallelLinear(
                config.hidden_size,
                self.total_num_heads * self.head_dim,
                bias=config.attention_bias,
                quant_config=quant_config,
                prefix=f"{prefix}.o_gate",
            )

        if self.use_rope:
            rope_params = {"rope_theta": config.rope_theta}
            if config.rope_scaling:
                rope_params.update(config.rope_scaling)
            self.rotary_emb = get_rope(
                self.head_dim,
                max_position=config.max_position_embeddings,
                rope_parameters=rope_params,
            )

        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        # Public MiniCPM-SALA checkpoint has attn_use_rope=False for `minicpm4`
        # layers. Keep this branch conditional; applying RoPE unconditionally is
        # a silent correctness bug for the MVP target model.
        if self.use_rope:
            q, k = self.rotary_emb(positions, q, k)

        attn_output = self.attn(q, k, v)
        if self.use_output_gate:
            gate, _ = self.o_gate(hidden_states)
            attn_output = attn_output * F.sigmoid(gate)
        output, _ = self.o_proj(attn_output)
        return output


class MiniCPMSALADecoderLayer(nn.Module):
    def __init__(
        self,
        config: MiniCPMSALAConfig,
        layer_idx: int,
        model_config: ModelConfig | None = None,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.scale = config.scale_depth / math.sqrt(config.num_hidden_layers)
        self.mixer_type = config.mixer_types[layer_idx]

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        if config.is_sparse_layer(layer_idx):
            self.self_attn = SALASparseAttention(
                config=config,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=f"{prefix}.self_attn",
            )
            self.is_lightning = False
        elif config.is_lightning_layer(layer_idx):
            self.self_attn = SALALightningAttention(
                config=config,
                layer_idx=layer_idx,
                model_config=model_config,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=f"{prefix}.self_attn",
            )
            self.is_lightning = True
        else:
            raise ValueError(
                f"Unsupported MiniCPM-SALA mixer type at layer {layer_idx}: "
                f"{self.mixer_type}"
            )

        self.mlp = MiniCPMSALAMLP(
            config=config,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        # Self attention / Simple GLA with layer-local scaled residual.
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        if self.is_lightning:
            # SALALightningAttention writes only `num_actual_tokens` into this
            # buffer; scheduler metadata excludes any padded/profile tokens.
            attn_output = torch.empty_like(hidden_states)
            self.self_attn(
                hidden_states=hidden_states,
                output=attn_output,
                positions=positions,
            )
        else:
            attn_output = self.self_attn(
                positions=positions,
                hidden_states=hidden_states,
            )
        hidden_states = residual + attn_output * self.scale

        # MLP with the same layer-local scaled residual.
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states * self.scale
        return hidden_states


@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        "positions": 0,
        "inputs_embeds": 0,
        "intermediate_tensors": 0,
    }
)
class MiniCPMSALAModel(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config: MiniCPMSALAConfig = vllm_config.model_config.hf_config
        model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config

        self.config = config
        self.vocab_size = config.vocab_size
        self.embed_tokens = VocabParallelEmbedding(
            self.vocab_size,
            config.hidden_size,
        )

        def get_layer(prefix: str) -> MiniCPMSALADecoderLayer:
            layer_idx = int(prefix.rsplit(".", 1)[1])
            return MiniCPMSALADecoderLayer(
                config=config,
                layer_idx=layer_idx,
                model_config=model_config,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=prefix,
            )

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            get_layer,
            prefix=f"{prefix}.layers",
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states"], config.hidden_size
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids) * self.config.scale_emb

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_input_ids(input_ids)
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]

        for layer in self.layers:
            hidden_states = layer(
                positions=positions,
                hidden_states=hidden_states,
            )

        if not get_pp_group().is_last_rank:
            return IntermediateTensors({"hidden_states": hidden_states})

        return self.norm(hidden_states)


class MiniCPMSALAForCausalLM(
    nn.Module,
    HasInnerState,
    SupportsPP,
    IsHybrid,
    SupportsQuant,
    SupportsMambaPrefixCaching,
):
    @classmethod
    def get_mamba_state_dtype_from_config(
        cls,
        vllm_config: VllmConfig,
    ) -> tuple[torch.dtype, ...]:
        return MambaStateDtypeCalculator.simple_gla_state_dtype(
            vllm_config.model_config.dtype,
            vllm_config.cache_config.mamba_cache_dtype,
        )

    @classmethod
    def get_mamba_state_shape_from_config(
        cls,
        vllm_config: VllmConfig,
    ) -> tuple[tuple[int, int, int], ...]:
        parallel_config = vllm_config.parallel_config
        config: MiniCPMSALAConfig = vllm_config.model_config.hf_config
        return MambaStateShapeCalculator.simple_gla_state_shape(
            tp_world_size=parallel_config.tensor_parallel_size,
            num_heads=config.lightning_nh,
            head_dim=config.lightning_head_dim,
        )

    @classmethod
    def get_mamba_state_copy_func(cls) -> tuple[MambaStateCopyFunc, ...]:
        return MambaStateCopyFuncCalculator.simple_gla_state_copy_func()

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config: MiniCPMSALAConfig = vllm_config.model_config.hf_config
        self.config = config
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.quant_config = vllm_config.quant_config

        self.model = MiniCPMSALAModel(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "model"),
        )
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=self.quant_config,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        if config.tie_word_embeddings:
            self.lm_head = self.lm_head.tie_weights(self.model.embed_tokens)

        self.scale_width = config.hidden_size / config.dim_model_base
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor | IntermediateTensors:
        model_output = self.model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )
        return model_output

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        # HF MiniCPM-SALA applies width scaling immediately before lm_head:
        # `self.lm_head(hidden_states / (hidden_size / dim_model_base))`.
        hidden_states = hidden_states / self.scale_width
        return self.logits_processor(self.lm_head, hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()

        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            if "rotary_emb.cos_cached" in name or "rotary_emb.sin_cached" in name:
                continue
            if "compress" in name.lower():
                continue
            if self.config.tie_word_embeddings and name.startswith("lm_head."):
                continue

        for param_name, weight_name, shard_id in stacked_params_mapping:
            if weight_name not in name:
                continue
            mapped_name = name.replace(weight_name, param_name)
            if mapped_name not in params_dict:
                continue
            if is_pp_missing_parameter(mapped_name, self):
                continue
            param = params_dict[mapped_name]
            weight_loader = param.weight_loader
            weight_loader(param, loaded_weight, shard_id)
            loaded_params.add(mapped_name)
            break
        else:
            if name not in params_dict:
                continue
            if is_pp_missing_parameter(name, self):
                continue
            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)
            loaded_params.add(name)

    return loaded_params
