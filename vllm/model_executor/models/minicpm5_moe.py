# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MiniCPM5 MoE adapted from the existing checkpoint implementation.

Reuses upstream vLLM attention, fused MoE, and quantized Linear kernels.
The checkpoint uses GQA with head-interleaved query/output gates, no QK
normalization, full RoPE, and FP32 sigmoid/noaux_tc routing,
matching the team SGLang implementation.
"""

import copy
from collections.abc import Iterable
from itertools import islice

import torch
import torch.nn.functional as F
from torch import nn

from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.distributed import get_pp_group, get_tensor_model_parallel_world_size
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.fused_moe import fused_moe_make_expert_params_mapping
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import QKVParallelLinear, RowParallelLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors

from .deepseek_v2 import DeepseekV2MLP, DeepseekV2MoE
from .interfaces import (
    EagleModelMixin,
    LocalArgmaxMixin,
    SupportsEagle,
    SupportsEagle3,
    SupportsLoRA,
    SupportsPP,
    SupportsQuant,
)
from .utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    WeightsMapper,
    extract_layer_index,
    is_pp_missing_parameter,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
    spec_decode_needs_target_embed,
)


class MiniCPM5Attention(nn.Module):
    def __init__(self, vllm_config: VllmConfig, prefix: str):
        super().__init__()
        c = vllm_config.model_config.hf_config
        tp = get_tensor_model_parallel_world_size()
        if c.num_attention_heads % tp:
            raise ValueError("num_attention_heads must be divisible by TP size")
        if c.num_key_value_heads >= tp:
            if c.num_key_value_heads % tp:
                raise ValueError("num_key_value_heads must be divisible by TP size")
        elif tp % c.num_key_value_heads:
            raise ValueError("TP size must be divisible by num_key_value_heads")
        self.num_heads = c.num_attention_heads // tp
        self.num_kv_heads = max(1, c.num_key_value_heads // tp)
        self.head_dim = c.head_dim
        self.gated = c.use_gated_attention
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.qkv_proj = QKVParallelLinear(
            c.hidden_size,
            self.head_dim,
            c.num_attention_heads * (2 if self.gated else 1),
            c.num_key_value_heads,
            bias=c.attention_bias,
            quant_config=vllm_config.quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            c.num_attention_heads * self.head_dim,
            c.hidden_size,
            bias=c.attention_bias,
            quant_config=vllm_config.quant_config,
            prefix=f"{prefix}.o_proj",
        )
        rope_parameters = getattr(c, "rope_parameters", None) or {
            "rope_type": "default",
            "rope_theta": c.rope_theta,
        }
        self.rotary_emb = get_rope(
            self.head_dim,
            max_position=c.max_position_embeddings,
            rope_parameters=rope_parameters,
        )
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.head_dim**-0.5,
            num_kv_heads=self.num_kv_heads,
            cache_config=vllm_config.cache_config,
            quant_config=vllm_config.quant_config,
            prefix=f"{prefix}.attn",
        )

    def forward(self, positions, hidden_states):
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split(
            [self.q_size * (2 if self.gated else 1), self.kv_size, self.kv_size],
            dim=-1,
        )
        gate = None
        if self.gated:
            # HF reference: [token, head, 2 * head_dim], then split last axis.
            q, gate = q.reshape(-1, self.num_heads, 2 * self.head_dim).chunk(2, -1)
            q = q.reshape(-1, self.q_size)
            gate = gate.reshape(-1, self.q_size)
        q, k = self.rotary_emb(positions, q, k)
        out = self.attn(q, k, v)
        if gate is not None:
            out = out * gate.sigmoid()
        return self.o_proj(out)[0]


class MiniCPM5Gate(nn.Module):
    def __init__(self, weight: nn.Parameter, correction_bias: nn.Parameter):
        super().__init__()
        self.weight = weight
        self.e_score_correction_bias = correction_bias

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, None]:
        return F.linear(hidden_states.float(), self.weight.float()), None


class MiniCPM5DecoderLayer(nn.Module):
    def __init__(self, vllm_config: VllmConfig, prefix: str):
        super().__init__()
        c = vllm_config.model_config.hf_config
        self.self_attn = MiniCPM5Attention(vllm_config, f"{prefix}.self_attn")
        layer = extract_layer_index(prefix)
        if layer < c.first_k_dense_replace:
            self.mlp = DeepseekV2MLP(
                c.hidden_size,
                c.intermediate_size,
                c.hidden_act,
                quant_config=vllm_config.quant_config,
                prefix=f"{prefix}.mlp",
            )
        else:
            # HF MiniCPM5 router hard-codes these semantics, while config's
            # inherited topk_method default is "greedy". Do not mutate config.
            moe_config = copy.copy(c)
            moe_config.topk_method = "noaux_tc"
            moe_config.scoring_func = "sigmoid"
            moe_config.moe_router_dtype = "float32"
            self.mlp = DeepseekV2MoE(
                moe_config,
                vllm_config.parallel_config,
                quant_config=vllm_config.quant_config,
                prefix=f"{prefix}.mlp",
            )
            if self.mlp.is_fused_shared_expert_enabled:
                raise NotImplementedError(
                    "MiniCPM5-15B-A2.5B does not support ROCm AITER shared "
                    "expert fusion yet. Set "
                    "VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS=0 to disable it."
                )
            gate = self.mlp.gate
            self.mlp.gate = MiniCPM5Gate(gate.weight, gate.e_score_correction_bias)
            self.mlp.experts.gate = self.mlp.gate
        self.input_layernorm = RMSNorm(c.hidden_size, eps=c.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(c.hidden_size, eps=c.rms_norm_eps)

    def forward(self, positions, hidden_states, residual):
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions, hidden_states)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        return self.mlp(hidden_states), residual


@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": {0: "b"},
        "positions": {0: "b"},
        "intermediate_tensors": {0: "b"},
        "inputs_embeds": {0: "b"},
    },
)
class MiniCPM5Model(nn.Module, EagleModelMixin):
    supports_aux_hidden_states_over_pp = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.config = config
        self.quant_config = vllm_config.quant_config
        self.vocab_size = config.vocab_size

        pp_group = get_pp_group()
        if (
            pp_group.is_first_rank
            or (config.tie_word_embeddings and pp_group.is_last_rank)
            or spec_decode_needs_target_embed(vllm_config)
        ):
            self.embed_tokens = VocabParallelEmbedding(
                self.vocab_size, config.hidden_size, quant_config=self.quant_config
            )
        else:
            self.embed_tokens = PPMissingLayer()

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: MiniCPM5DecoderLayer(vllm_config=vllm_config, prefix=prefix),
            prefix=f"{prefix}.layers",
        )
        if pp_group.is_last_rank:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], config.hidden_size
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
        pp_group = get_pp_group()
        aux_hidden_states: list[torch.Tensor] = []
        if pp_group.is_first_rank:
            hidden_states = (
                inputs_embeds
                if inputs_embeds is not None
                else self.embed_input_ids(input_ids)
            )
            residual = None
            self._maybe_add_hidden_state(
                aux_hidden_states, self.start_layer, hidden_states, residual
            )
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        for layer_idx, layer in enumerate(
            islice(self.layers, self.start_layer, self.end_layer),
            start=self.start_layer,
        ):
            hidden_states, residual = layer(positions, hidden_states, residual)
            self._maybe_add_hidden_state(
                aux_hidden_states, layer_idx + 1, hidden_states, residual
            )

        if not pp_group.is_last_rank:
            return IntermediateTensors(
                {
                    "hidden_states": hidden_states,
                    "residual": residual,
                    **self.pack_local_aux_hidden_states(aux_hidden_states),
                }
            )

        hidden_states, _ = self.norm(hidden_states, residual)
        if aux_hidden_states:
            return hidden_states, aux_hidden_states
        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked: list[tuple[str, str, str | int]] = [
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        expert_mapping = fused_moe_make_expert_params_mapping(
            self,
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.n_routed_experts,
            num_redundant_experts=0,
        )
        params = dict(self.named_parameters())
        loaded = set()
        for name, weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            if ".experts." in name:
                for param_name, weight_name, expert_id, shard_id in expert_mapping:
                    if weight_name not in name:
                        continue
                    mapped = name.replace(weight_name, param_name)
                    if is_pp_missing_parameter(mapped, self):
                        break
                    param = params[mapped]
                    param.weight_loader(
                        param, weight, mapped, shard_id=shard_id, expert_id=expert_id
                    )
                    loaded.add(mapped)
                    break
                else:
                    raise KeyError(f"Unmapped expert tensor: {name}")
                continue
            for param_name, weight_name, dense_shard_id in stacked:
                if weight_name not in name:
                    continue
                mapped = name.replace(weight_name, param_name)
                if is_pp_missing_parameter(mapped, self):
                    break
                param = params[mapped]
                param.weight_loader(param, weight, dense_shard_id)
                loaded.add(mapped)
                break
            else:
                if is_pp_missing_parameter(name, self):
                    continue
                if name.endswith(".bias") and name not in params:
                    continue
                param = params[name]
                getattr(param, "weight_loader", default_weight_loader)(param, weight)
                loaded.add(name)
        return loaded


class MiniCPM5MoEForCausalLM(
    LocalArgmaxMixin,
    nn.Module,
    SupportsLoRA,
    SupportsPP,
    SupportsEagle,
    SupportsEagle3,
    SupportsQuant,
):
    # The custom loader maps expert projections separately from dense projections.
    hf_to_vllm_mapper = WeightsMapper()
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }
    embedding_modules = {
        "embed_tokens": "input_embeddings",
        "lm_head": "output_embeddings",
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        if vllm_config.parallel_config.enable_eplb:
            raise NotImplementedError(
                "MiniCPM5-15B-A2.5B does not support EPLB. "
                "Omit --enable-eplb to use expert parallelism."
            )
        config = vllm_config.model_config.hf_config
        self.config = config
        self.model = MiniCPM5Model(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )

        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=vllm_config.quant_config,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
            if config.tie_word_embeddings:
                self.lm_head = self.lm_head.tie_weights(self.model.embed_tokens)
            self.logits_processor = LogitsProcessor(
                config.vocab_size, scale=getattr(config, "logit_scale", 1.0)
            )
        else:
            self.lm_head = PPMissingLayer()
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )
        if current_platform.is_xpu() and (
            vllm_config.kernel_config.monokernel or vllm_config.additional_config
        ):
            from vllm.models.minicpm5_moe.xpu.model import configure_xpu_model

            configure_xpu_model(self, vllm_config)

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

    def compute_logits_local(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.logits_processor(self.lm_head, hidden_states, skip_gather=True)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        return AutoWeightsLoader(self).load_weights(weights)

    def get_expert_mapping(self):
        return fused_moe_make_expert_params_mapping(
            self,
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.n_routed_experts,
            num_redundant_experts=0,
        )
