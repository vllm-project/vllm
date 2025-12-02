# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Adapted from
# https://github.com/inclusionAI/Ling/blob/master/models/modeling_bailing_moe.py
# Copyright 2023 The vLLM team.
# Copyright 2023 Antgroup and The HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Inference-only BailingMoE model compatible with HuggingFace weights."""

from collections.abc import Iterable
from itertools import islice

import torch
import torch.nn.functional as F
from torch import nn
from transformers.configuration_utils import PretrainedConfig

from vllm.attention.layer import Attention
from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import (
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.fused_moe import SharedFusedMoE
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.sequence import IntermediateTensors

from .interfaces import SupportsLoRA, SupportsPP
from .utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    is_pp_missing_parameter,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)


class BailingAttention(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        reduce_results: bool = True,
        prefix: str = "",
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.total_num_heads = config.num_attention_heads
        self.total_kv_heads = config.num_key_value_heads
        tp_size = get_tensor_model_parallel_world_size()

        assert self.total_num_heads % tp_size == 0
        assert self.total_num_heads >= self.total_kv_heads

        self.num_heads = self.total_num_heads // tp_size
        self.head_dim = config.head_dim or (self.hidden_size // self.total_num_heads)
        self.q_size_per_rank = self.head_dim * self.num_heads
        self.num_kv_heads = max(1, self.total_kv_heads // tp_size)
        self.kv_size_per_rank = self.num_kv_heads * self.head_dim
        self.scale = self.head_dim**-0.5
        self.use_qk_norm = getattr(config, "use_qk_norm", False)
        self.use_rmsnorm = getattr(config, "use_rmsnorm", False)

        self.query_key_value = QKVParallelLinear(
            self.hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_kv_heads,
            bias=(config.use_bias or config.use_qkv_bias),
            quant_config=quant_config,
            prefix=f"{prefix}.query_key_value",
        )

        if self.use_qk_norm:
            self.query_layernorm = (
                RMSNorm(self.head_dim, eps=config.rms_norm_eps)
                if self.use_rmsnorm
                else nn.LayerNorm(self.head_dim, eps=1e-6)
            )
            self.key_layernorm = (
                RMSNorm(self.head_dim, eps=config.rms_norm_eps)
                if self.use_rmsnorm
                else nn.LayerNorm(self.head_dim, eps=1e-6)
            )

        self.dense = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            self.hidden_size,
            bias=config.use_bias,
            quant_config=quant_config,
            reduce_results=reduce_results,
            prefix=f"{prefix}.dense",
        )

        self.partial_rotary_factor = getattr(config, "partial_rotary_factor", 1.0)

        self.rotary_dim = getattr(config, "rotary_dim", self.head_dim)

        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.rotary_dim,
            max_position=config.max_position_embeddings,
            rope_parameters=config.rope_parameters,
            is_neox_style=True,
            partial_rotary_factor=self.partial_rotary_factor,
        )

        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scale,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            prefix=f"{prefix}.attn",
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        qkv, _ = self.query_key_value(hidden_states)
        q, k, v = qkv.split(
            [self.q_size_per_rank, self.kv_size_per_rank, self.kv_size_per_rank], dim=-1
        )

        if self.use_qk_norm:
            q = q.view(-1, self.num_heads, self.head_dim)
            k = k.view(-1, self.num_kv_heads, self.head_dim)
            q = self.query_layernorm(q)
            k = self.key_layernorm(k)
            q = q.view(-1, self.q_size_per_rank)
            k = k.view(-1, self.kv_size_per_rank)

        q, k = self.rotary_emb(position_ids, q, k)

        context_layer = self.attn(q, k, v)

        attn_output, _ = self.dense(context_layer)
        return attn_output


class BailingMLP(nn.Module):
    def __init__(
        self,
        intermediate_size: int,
        config: PretrainedConfig,
        quant_config: QuantizationConfig | None = None,
        reduce_results: bool | None = True,
        prefix: str = "",
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
        self.act_fn = SiluAndMul()

    def forward(self, x):
        x, _ = self.gate_up_proj(x)
        x = self.act_fn(x)
        x, _ = self.down_proj(x)
        return x


class BailingMoE(nn.Module):
    def __init__(
        self,
        intermediate_size: int,
        config: PretrainedConfig,
        quant_config: QuantizationConfig | None = None,
        reduce_results: bool | None = True,
        prefix: str = "",
    ):
        super().__init__()

        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.norm_expert_prob = config.norm_topk_prob
        self.hidden_size = config.hidden_size
        self.quant_config = quant_config
        self.num_shared_experts = config.num_shared_experts
        self.score_function = getattr(config, "score_function", None)
        self.n_group = getattr(config, "n_group", None)
        self.topk_group = getattr(config, "topk_group", None)
        self.use_grouped_topk = self.n_group is not None and self.topk_group is not None
        self.routed_scaling_factor = getattr(config, "routed_scaling_factor", 1.0)

        router_dtype = getattr(config, "router_dtype", None)
        if router_dtype is None:
            self.router_dtype = None
        elif router_dtype == "fp32":
            self.router_dtype = torch.float32
        else:
            self.router_dtype = torch.bfloat16

        self.gate = nn.Linear(
            self.hidden_size,
            self.num_experts,
            bias=False,
            dtype=self.router_dtype,
        )

        if getattr(config, "moe_router_enable_expert_bias", False):
            self.gate.expert_bias = nn.Parameter(
                torch.empty((config.num_experts,), dtype=torch.float32)
            )
        else:
            self.gate.expert_bias = None

        self.correction_bias = (
            self.gate.expert_bias.data if self.gate.expert_bias is not None else None
        )

        if self.score_function is not None:
            assert (
                self.score_function == "softmax" and self.correction_bias is None
            ) or (
                self.score_function == "sigmoid" and self.correction_bias is not None
            ), (
                "score_function and correction_bias should be in 2 combination (softmax, None) or (sigmoid, not None)"  # noqa: E501
            )
        else:
            # default value for scoring_func
            self.score_function = "softmax"

        if self.num_shared_experts > 0:
            if hasattr(config, "moe_shared_expert_intermediate_size"):
                intermediate_size = config.moe_shared_expert_intermediate_size
            else:
                intermediate_size = config.moe_intermediate_size
            intermediate_size *= config.num_shared_experts
            self.shared_experts = BailingMLP(
                intermediate_size=intermediate_size,
                config=config,
                quant_config=quant_config,
                reduce_results=False,
                prefix=f"{prefix}.shared_experts",
            )
        else:
            self.shared_experts = None

        self.experts = SharedFusedMoE(
            shared_experts=self.shared_experts,
            num_experts=self.num_experts,
            top_k=self.top_k,
            hidden_size=self.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            reduce_results=False,
            renormalize=self.norm_expert_prob,
            quant_config=quant_config,
            prefix=f"{prefix}.experts",
            scoring_func=self.score_function,
            e_score_correction_bias=self.gate.expert_bias,
            num_expert_group=self.n_group,
            topk_group=self.topk_group,
            use_grouped_topk=self.use_grouped_topk,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        num_tokens, hidden_size = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_size)

        # router_logits: (num_tokens, n_experts)
        router_logits = self.gate(hidden_states.to(self.router_dtype))
        router_logits = router_logits.to(hidden_states.dtype)

        final_hidden_states = self.experts(
            hidden_states=hidden_states, router_logits=router_logits
        )

        if self.shared_experts is not None:
            shared_output, final_hidden_states = final_hidden_states
        else:
            shared_output = None

        final_hidden_states *= self.routed_scaling_factor

        if shared_output is not None:
            final_hidden_states = final_hidden_states + shared_output

        if self.tp_size > 1:
            final_hidden_states = self.experts.maybe_all_reduce_tensor_model_parallel(
                final_hidden_states
            )
        return final_hidden_states.view(num_tokens, hidden_size)


class BailingMoeBlock(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        layer_idx = int(prefix.split(".")[-1])
        self.config = config
        hidden_size = config.hidden_size
        intermediate_size = config.intermediate_size

        self.input_layernorm = RMSNorm(hidden_size, eps=config.rms_norm_eps)
        self.attention = BailingAttention(
            config, cache_config, quant_config, prefix=f"{prefix}.attention"
        )

        self.post_attention_layernorm = RMSNorm(hidden_size, eps=config.rms_norm_eps)

        # Choose MLP class based on the number of experts and layer index
        if layer_idx < config.first_k_dense_replace:
            mlp_class = BailingMLP
        else:
            mlp_class = BailingMoE
        self.mlp = mlp_class(
            intermediate_size, config, quant_config, True, prefix=f"{prefix}.mlp"
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> torch.Tensor:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        hidden_states = self.attention(
            hidden_states=hidden_states,
            position_ids=position_ids,
        )

        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


@support_torch_compile
class BailingMoeModel(nn.Module):
    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
    ):
        super().__init__()
        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config

        self.config = config
        self.vocab_size = config.vocab_size
        self.embed_dim = config.hidden_size
        self.tie_word_embeddings = getattr(config, "tie_word_embeddings", False)

        if get_pp_group().is_first_rank or (
            self.tie_word_embeddings and get_pp_group().is_last_rank
        ):
            self.word_embeddings = VocabParallelEmbedding(
                self.vocab_size,
                self.embed_dim,
                quant_config=quant_config,
                prefix=f"{prefix}.word_embeddings",
            )
        else:
            self.word_embeddings = PPMissingLayer()

        self.embedding_dropout = torch.nn.Dropout(config.embedding_dropout)

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: BailingMoeBlock(
                config=config,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=prefix,
            ),
            prefix=f"{prefix}.layers",
        )

        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], config.hidden_size
        )

        if get_pp_group().is_last_rank:
            self.norm = RMSNorm(self.embed_dim, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.word_embeddings(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
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

        for layer in islice(self.layers, self.start_layer, self.end_layer):
            hidden_states, residual = layer(
                hidden_states,
                position_ids,
                residual,
            )

        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )
        else:
            if residual is None:
                hidden_states = self.norm(hidden_states)
            else:
                hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        return SharedFusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.num_experts,
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        params_dict = dict(self.named_parameters(remove_duplicate=False))
        loaded_params: set[str] = set()
        expert_params_mapping = self.get_expert_mapping()
        for name, loaded_weight in weights:
            if (
                hasattr(self.config, "norm_head")
                and self.config.norm_head
                and "lm_head.weight" in name
            ):
                loaded_weight = F.normalize(loaded_weight, dim=0, p=2, eps=1e-7)

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                if "mlp.experts" in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if name not in params_dict:
                    continue

                if is_pp_missing_parameter(name, self):
                    continue

                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                for mapping in expert_params_mapping:
                    param_name, weight_name, expert_id, shard_id = mapping
                    if weight_name not in name:
                        continue
                    name = name.replace(weight_name, param_name)

                    if is_pp_missing_parameter(name, self):
                        continue
                    if name not in params_dict:
                        continue
                    param = params_dict[name]
                    weight_loader = param.weight_loader
                    weight_loader(
                        param,
                        loaded_weight,
                        name,
                        shard_id=shard_id,
                        expert_id=expert_id,
                    )
                    break
                else:
                    if name.endswith(".bias") and name not in params_dict:
                        continue
                    if name not in params_dict:
                        continue

                    if is_pp_missing_parameter(name, self):
                        continue

                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params


class BailingMoeForCausalLM(nn.Module, SupportsPP, SupportsLoRA):
    packed_modules_mapping = {
        "query_key_value": ["query_key_value"],
        "gate_up_proj": [
            "gate_proj",
            "up_proj",
        ],
    }

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
    ) -> None:
        super().__init__()

        config = vllm_config.model_config.hf_config.get_text_config()
        vllm_config.model_config.hf_config = config
        quant_config = vllm_config.quant_config

        self.config = config
        self.quant_config = quant_config
        self.max_position_embeddings = config.max_position_embeddings
        self.model = BailingMoeModel(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        self.tie_word_embeddings = getattr(config, "tie_word_embeddings", False)

        if get_pp_group().is_last_rank:
            if self.tie_word_embeddings:
                self.lm_head = self.model.word_embeddings
            else:
                self.lm_head = ParallelLMHead(
                    config.vocab_size,
                    config.hidden_size,
                    quant_config=quant_config,
                    prefix=maybe_prefix(prefix, "lm_head"),
                )
            self.logits_processor = LogitsProcessor(config.vocab_size)
        else:
            self.lm_head = PPMissingLayer()

        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        model_output = self.model(
            input_ids, positions, intermediate_tensors, inputs_embeds
        )
        return model_output

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        logits = self.logits_processor(self.lm_head, hidden_states)
        return logits

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=(["lm_head."] if self.tie_word_embeddings else None),
        )
        return loader.load_weights(weights)

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        return self.model.get_expert_mapping()


class BailingMoeV2ForCausalLM(BailingMoeForCausalLM):
    pass
