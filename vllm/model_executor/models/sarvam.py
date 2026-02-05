# coding=utf-8
# Copyright 2026 Sarvam AI team. All rights reserved.
#
# This code is based on Llama, Deepseek, and Bailing MoE implementations
# in this library. It has been modified from its original forms to
# accommodate Sarvam's MoE architectures.
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

from __future__ import annotations

from collections.abc import Iterable, Iterator
from itertools import islice

import torch
from torch import nn

from vllm.attention.layer import Attention
from vllm.config import CacheConfig, ParallelConfig, VllmConfig
from vllm.distributed import (
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.fused_moe import SharedFusedMoE
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    ReplicatedLinear,
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
from vllm.model_executor.layers.mla import MLAModules, MultiHeadLatentAttentionWrapper
from vllm.sequence import IntermediateTensors

from .bailing_moe import BailingMoeForCausalLM
from .interfaces import MixtureOfExperts, SupportsLoRA, SupportsPP
from .utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    is_pp_missing_parameter,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)


def yarn_get_mscale(scale: float = 1, mscale: float = 1) -> float:
    import math

    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


def _is_gate_expert_bias_name(name: str) -> bool:
    return name.endswith(".mlp.gate.e_score_correction_bias") or name.endswith(".gate.e_score_correction_bias")


def _zero_mean_tensor(t: torch.Tensor) -> torch.Tensor:
    if t.numel() == 0:
        return t
    return t - t.mean()


def _normalized_weights(
    weights: Iterable[tuple[str, torch.Tensor]],
) -> Iterator[tuple[str, torch.Tensor]]:
    for name, w in weights:
        if _is_gate_expert_bias_name(name):
            yield name, _zero_mean_tensor(w)
        else:
            yield name, w


class SarvamMLAAttention(nn.Module):
    def __init__(
        self,
        vllm_config: VllmConfig,
        config,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()

        self.config = config
        self.hidden_size = config.hidden_size
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.v_head_dim = config.v_head_dim

        self.q_lora_rank = getattr(config, "q_lora_rank", None)
        self.kv_lora_rank = config.kv_lora_rank

        self.total_num_heads = config.num_attention_heads
        tp_size = get_tensor_model_parallel_world_size()
        assert self.total_num_heads % tp_size == 0
        self.num_local_heads = self.total_num_heads // tp_size

        self.scaling = self.qk_head_dim**-0.5
        self.max_position_embeddings = config.max_position_embeddings

        if self.q_lora_rank is not None:
            self.q_a_proj = ReplicatedLinear(
                self.hidden_size,
                self.q_lora_rank,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.q_a_proj",
            )
            self.q_a_layernorm = RMSNorm(self.q_lora_rank, eps=config.rms_norm_eps)
            self.q_b_proj = ColumnParallelLinear(
                self.q_lora_rank,
                self.total_num_heads * self.qk_head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.q_b_proj",
            )
            self.q_proj = None  # type: ignore
        else:
            self.q_proj = ColumnParallelLinear(
                self.hidden_size,
                self.total_num_heads * self.qk_head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.q_proj",
            )
            self.q_a_proj = None  # type: ignore
            self.q_a_layernorm = None  # type: ignore
            self.q_b_proj = None  # type: ignore

        # KV latent (MQA-style) A-proj
        self.kv_a_proj_with_mqa = ReplicatedLinear(
            self.hidden_size,
            self.kv_lora_rank + self.qk_rope_head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.kv_a_proj_with_mqa",
        )
        self.kv_a_layernorm = RMSNorm(self.kv_lora_rank, eps=config.rms_norm_eps)

        # KV B-proj produces per-head K_nope and V
        self.kv_b_proj = ColumnParallelLinear(
            self.kv_lora_rank,
            self.total_num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.kv_b_proj",
        )

        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.v_head_dim,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        self.rotary_emb = get_rope(
            self.qk_rope_head_dim,
            # rotary_dim=self.qk_rope_head_dim,
            max_position=config.max_position_embeddings,
            rope_parameters=config.rope_parameters,
            is_neox_style=False,
        )

        if config.rope_parameters.get("rope_type", None) == "deepseek_yarn":
            mscale_all_dim = config.rope_parameters.get("mscale_all_dim", False)
            scaling_factor = config.rope_parameters["factor"]
            mscale = yarn_get_mscale(scaling_factor, float(mscale_all_dim))
            self.scaling = self.scaling * mscale * mscale

        mla_modules = MLAModules(
            kv_a_layernorm=self.kv_a_layernorm,
            kv_b_proj=self.kv_b_proj,
            rotary_emb=self.rotary_emb,
            o_proj=self.o_proj,
            fused_qkv_a_proj=None,
            kv_a_proj_with_mqa=self.kv_a_proj_with_mqa,
            q_a_layernorm=self.q_a_layernorm if self.q_lora_rank is not None else None,
            q_b_proj=self.q_b_proj if self.q_lora_rank is not None else None,
            q_proj=self.q_proj if self.q_lora_rank is None else None,
            indexer=None,
            indexer_rotary_emb=None,
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
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=prefix,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        return self.mla_attn(positions, hidden_states, llama_4_scaling=None)


class SarvamMLAMLP(nn.Module):
    def __init__(
        self,
        intermediate_size: int,
        config,
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
        self.act_fn = SiluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class SarvamMLAMoE(nn.Module):
    def __init__(
        self,
        config,
        parallel_config: ParallelConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()

        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()
        self.hidden_size = config.hidden_size

        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.routed_scaling_factor = getattr(config, "routed_scaling_factor", 1.0)

        self.n_group = getattr(config, "n_group", self.num_experts // 8)
        self.topk_group = getattr(config, "topk_group", 2)
        self.use_grouped_topk = self.n_group is not None and self.topk_group is not None

        self.norm_expert_prob = getattr(config, "norm_topk_prob", True)

        router_dtype_cfg = getattr(config, "router_dtype", None)
        if router_dtype_cfg is None:
            self.router_dtype = None
        elif router_dtype_cfg == "fp32":
            self.router_dtype = torch.float32
        else:
            self.router_dtype = torch.bfloat16

        self.gate = nn.Linear(
            self.hidden_size,
            self.num_experts,
            bias=False,
            dtype=self.router_dtype,
        )

        if getattr(config, "moe_router_enable_expert_bias", True):
            self.gate.e_score_correction_bias = nn.Parameter(
                torch.empty(
                    (self.num_experts,),
                    dtype=torch.float32,
                )
            )
        else:
            self.gate.e_score_correction_bias = None

        self.score_function = getattr(config, "score_function", "sigmoid")
        self.num_shared_experts = getattr(config, "num_shared_experts", 0)
        if self.num_shared_experts > 0:
            if hasattr(config, "moe_shared_expert_intermediate_size"):
                shared_int = config.moe_shared_expert_intermediate_size
            else:
                shared_int = config.moe_intermediate_size
            shared_int *= self.num_shared_experts
            self.shared_experts = SarvamMLAMLP(
                intermediate_size=shared_int,
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
            e_score_correction_bias=self.gate.e_score_correction_bias,
            num_expert_group=self.n_group,
            topk_group=self.topk_group,
            use_grouped_topk=self.use_grouped_topk,
            routed_scaling_factor=1.0,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        router_logits = self.gate(
            hidden_states.to(self.router_dtype) if self.router_dtype is not None else hidden_states
        )
        router_logits = router_logits.to(hidden_states.dtype)
        final_hidden = self.experts(
            hidden_states=hidden_states,
            router_logits=router_logits,
        )

        if self.shared_experts is not None:
            shared_output, expert_output = final_hidden
        else:
            shared_output, expert_output = None, final_hidden

        expert_output *= self.routed_scaling_factor

        if shared_output is not None:
            expert_output = expert_output + shared_output

        if self.tp_size > 1:
            expert_output = self.experts.maybe_all_reduce_tensor_model_parallel(expert_output)

        return expert_output.view(num_tokens, hidden_dim)


class SarvamMLABlock(nn.Module):
    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str = "",
    ) -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        parallel_config = vllm_config.parallel_config
        layer_idx = int(prefix.split(".")[-1])
        hidden_size = config.hidden_size
        dense_intermediate = getattr(
            config,
            "intermediate_size",
            config.moe_intermediate_size * 2,
        )

        self.input_layernorm = RMSNorm(hidden_size, eps=config.rms_norm_eps)
        self.self_attn = SarvamMLAAttention(
            vllm_config=vllm_config,
            config=config,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
        )
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=config.rms_norm_eps)
        use_moe = hasattr(config, "num_experts") and config.num_experts is not None
        first_k_dense = getattr(config, "first_k_dense_replace", 0)
        moe_layer_freq = getattr(config, "moe_layer_freq", 1)
        if use_moe:
            is_moe_layer = layer_idx >= first_k_dense and ((layer_idx - first_k_dense) % moe_layer_freq == 0)
        else:
            is_moe_layer = False

        if is_moe_layer:
            self.mlp = SarvamMLAMoE(
                config=config,
                parallel_config=parallel_config,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
            )
        else:
            self.mlp = SarvamMLAMLP(
                intermediate_size=dense_intermediate,
                config=config,
                quant_config=quant_config,
                reduce_results=True,
                prefix=f"{prefix}.mlp",
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
        )
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class SarvamMLAModel(nn.Module):
    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
    ) -> None:
        super().__init__()

        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config

        self.config = config
        self.vocab_size = config.vocab_size
        self.embed_dim = config.hidden_size
        self.tie_word_embeddings = getattr(config, "tie_word_embeddings", False)
        if get_pp_group().is_first_rank or (self.tie_word_embeddings and get_pp_group().is_last_rank):
            self.embed_tokens = VocabParallelEmbedding(
                self.vocab_size,
                self.embed_dim,
                quant_config=quant_config,
                prefix=f"{prefix}.embed_tokens",
            )
        else:
            self.embed_tokens = PPMissingLayer()

        self.embedding_dropout = torch.nn.Dropout(getattr(config, "embedding_dropout", 0.0))
        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: SarvamMLABlock(
                vllm_config=vllm_config,
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
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_input_ids(input_ids)
            hidden_states = self.embedding_dropout(hidden_states)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        for layer in islice(self.layers, self.start_layer, self.end_layer):
            hidden_states, residual = layer(
                hidden_states,
                positions,
                residual,
            )
        if not get_pp_group().is_last_rank:
            return IntermediateTensors({"hidden_states": hidden_states, "residual": residual})
        if residual is None:
            hidden_states = self.norm(hidden_states)
        else:
            hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        return SharedFusedMoE.make_expert_params_mapping(
            self,
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.num_experts,
        )

    def load_weights(
        self,
        weights: Iterable[tuple[str, torch.Tensor]],
    ) -> set[str]:
        """Load weights with stacked gate+up and MoE expert remapping."""
        weights = _normalized_weights(weights)
        stacked_params_mapping = [
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        params_dict = dict(self.named_parameters(remove_duplicate=False))
        loaded_params: set[str] = set()
        expert_params_mapping = self.get_expert_mapping()

        for name, loaded_weight in weights:
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                if "mlp.experts" in name:
                    continue
                new_name = name.replace(weight_name, param_name)
                if new_name.endswith(".bias") and new_name not in params_dict:
                    continue
                if new_name not in params_dict:
                    continue
                if is_pp_missing_parameter(new_name, self):
                    continue

                param = params_dict[new_name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight, shard_id)
                loaded_params.add(new_name)
                break
            else:
                mapped = False
                for (
                    param_name,
                    weight_name,
                    expert_id,
                    shard_id,
                ) in expert_params_mapping:
                    if weight_name not in name:
                        continue

                    new_name = name.replace(weight_name, param_name)
                    if is_pp_missing_parameter(new_name, self):
                        continue
                    if new_name not in params_dict:
                        continue

                    param = params_dict[new_name]
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(
                        param,
                        loaded_weight,
                        name,
                        shard_id=shard_id,
                        expert_id=expert_id,
                    )
                    loaded_params.add(new_name)
                    mapped = True
                    break

                if mapped:
                    continue

                if name.endswith(".bias") and name not in params_dict:
                    continue
                if name not in params_dict:
                    continue
                if is_pp_missing_parameter(name, self):
                    continue

                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
                loaded_params.add(name)

        return loaded_params


class SarvamMLAForCausalLM(nn.Module, SupportsPP, SupportsLoRA, MixtureOfExperts):
    packed_modules_mapping = {
        "q_proj": ["q_proj"],
        "q_a_proj": ["q_a_proj"],
        "q_b_proj": ["q_b_proj"],
        "kv_a_proj_with_mqa": ["kv_a_proj_with_mqa"],
        "kv_b_proj": ["kv_b_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
    ) -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.config = config
        quant_config = vllm_config.quant_config
        self.quant_config = quant_config
        self.max_position_embeddings = config.max_position_embeddings
        self.model = SarvamMLAModel(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "model"),
        )
        self.tie_word_embeddings = getattr(config, "tie_word_embeddings", False)

        if get_pp_group().is_last_rank:
            if self.tie_word_embeddings:
                # Tie to embed_tokens like HF
                self.lm_head = self.model.embed_tokens
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
            self.logits_processor = None  # type: ignore

        self.make_empty_intermediate_tensors = self.model.make_empty_intermediate_tensors

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        return self.model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        if not get_pp_group().is_last_rank:
            return None
        logits = self.logits_processor(self.lm_head, hidden_states)
        return logits

    def load_weights(
        self,
        weights: Iterable[tuple[str, torch.Tensor]],
    ) -> set[str]:
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=(["lm_head."] if self.tie_word_embeddings else None),
        )
        return loader.load_weights(weights)

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        return self.model.get_expert_mapping()


class SarvamMoEForCausalLM(BailingMoeForCausalLM):
    """Same as BailingMoeForCausalLM, but normalizes gate expert_bias pre-load."""

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        return super().load_weights(_normalized_weights(weights))
