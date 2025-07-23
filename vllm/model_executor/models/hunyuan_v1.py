# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# coding=utf-8
# Copyright 2024 The HunYuan team.
# Copyright 2023 The vLLM team.
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
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
"""Inference-only HunYuan model compatible with HuggingFace weights."""
from collections.abc import Iterable
from typing import Any, Optional, Union

import regex as re
import torch
from torch import nn
from transformers import PretrainedConfig

from vllm.attention import Attention, AttentionType
from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import (get_pp_group,
                              get_tensor_model_parallel_world_size,
                              tensor_model_parallel_all_reduce)
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               MergedColumnParallelLinear,
                                               QKVParallelLinear,
                                               ReplicatedLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    DEFAULT_VOCAB_PADDING_SIZE, ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader, maybe_remap_kv_scale_name)
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import IntermediateTensors

from .interfaces import SupportsLoRA
from .utils import (AutoWeightsLoader, PPMissingLayer, is_pp_missing_parameter,
                    make_layers)


def _is_moe(config: PretrainedConfig) -> bool:
    num_experts = getattr(config, "num_experts", None)
    if isinstance(num_experts, int):
        return num_experts > 1
    if isinstance(num_experts, list) and num_experts:
        # Ensure all elements are integers before calling max.
        if all(isinstance(e, int) for e in num_experts):
            return max(num_experts) > 1
        else:
            return False
    return False


def _get_cla_factor(config: PretrainedConfig) -> int:
    if not getattr(config, "use_cla", False):
        return 1
    return getattr(config, "cla_share_factor", 1)


class HunYuanMLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: Optional[QuantizationConfig] = None,
        bias: bool = False,
        prefix: str = "",
        reduce_results: bool = True,
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            input_size=hidden_size,
            output_sizes=[intermediate_size] * 2,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            input_size=intermediate_size,
            output_size=hidden_size,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.down_proj",
            reduce_results=reduce_results,
        )
        if hidden_act != "silu":
            raise ValueError(f"Unsupported activation: {hidden_act}. "
                             "Only silu is supported for now.")
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class HunYuanAttention(nn.Module):

    def __init__(
        self,
        config: PretrainedConfig,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        rope_theta: float = 10000,
        rope_scaling: Optional[dict[str, Any]] = None,
        max_position_embeddings: int = 8192,
        quant_config: Optional[QuantizationConfig] = None,
        bias: bool = False,
        cache_config: Optional[CacheConfig] = None,
        prefix: str = "",
        layer_id: int = -1,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)

        if hasattr(config, "head_dim") and config.head_dim:
            self.head_dim = config.head_dim
        elif hasattr(config, "attention_head_dim"):
            self.head_dim = config.attention_head_dim
        else:
            self.head_dim = self.hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings
        self.use_qk_norm = getattr(config, "use_qk_norm", False)
        self.layer_id = layer_id

        self.qkv_proj = QKVParallelLinear(
            hidden_size=hidden_size,
            head_size=self.head_dim,
            total_num_heads=self.total_num_heads,
            total_num_kv_heads=self.total_num_kv_heads,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )

        self.o_proj = RowParallelLinear(
            input_size=self.total_num_heads * self.head_dim,
            output_size=hidden_size,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
            rope_scaling=rope_scaling,
            is_neox_style=True,
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

        if self.use_qk_norm:
            self.query_layernorm = RMSNorm(self.head_dim,
                                           eps=config.rms_norm_eps)
            self.key_layernorm = RMSNorm(self.head_dim,
                                         eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_states: Optional[tuple[torch.Tensor]] = None,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        ori_k = k
        if self.use_qk_norm:
            q = self.query_layernorm(
                q.view(-1, self.num_heads, self.head_dim).contiguous())
            k = self.key_layernorm(
                k.view(-1, self.num_kv_heads, self.head_dim).contiguous())

        attn_output = self.attn(q, k, v)
        # For o_proj
        attn_output = attn_output.view(q.shape[0], -1)
        output, _ = self.o_proj(attn_output)
        return output, (ori_k, v)


class HunYuanCrossAttention(nn.Module):

    def __init__(
        self,
        config: PretrainedConfig,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        rope_theta: float = 10000,
        rope_scaling: Optional[dict[str, Any]] = None,
        max_position_embeddings: int = 8192,
        quant_config: Optional[QuantizationConfig] = None,
        bias: bool = False,
        cache_config: Optional[CacheConfig] = None,
        prefix: str = "",
        layer_id: int = -1,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        # MistralConfig has an optional head_dim introduced by Mistral-Nemo
        if hasattr(config, "head_dim"):
            self.head_dim = config.head_dim
        elif hasattr(config, "attention_head_dim"):
            self.head_dim = config.attention_head_dim
        else:
            self.head_dim = self.hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings
        self.use_qk_norm = getattr(config, "use_qk_norm", False)
        self.layer_id = layer_id

        self.q_proj = ColumnParallelLinear(
            hidden_size,
            hidden_size,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.q_proj",
        )

        self.o_proj = RowParallelLinear(
            input_size=self.total_num_heads * self.head_dim,
            output_size=hidden_size,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
            rope_scaling=rope_scaling,
            is_neox_style=True,
        )
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
            attn_type=AttentionType.ENCODER_DECODER,
        )

        if self.use_qk_norm:
            self.query_layernorm = RMSNorm(self.head_dim,
                                           eps=config.rms_norm_eps)
            self.key_layernorm = RMSNorm(self.head_dim,
                                         eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_states: Optional[tuple[torch.Tensor]] = None,
    ) -> torch.Tensor:
        assert kv_states is not None
        ori_k, v = kv_states  # use last layer kv,
        k = ori_k
        q, _ = self.q_proj(hidden_states)
        k_tmp = torch.empty_like(k)  # Todo: reduant rotary embedding
        q, _ = self.rotary_emb(positions, q, k_tmp)
        if self.use_qk_norm:
            q = self.query_layernorm(
                q.view(-1, self.num_heads, self.head_dim).contiguous())
            k = self.key_layernorm(
                k.view(-1, self.num_kv_heads, self.head_dim).contiguous())

        attn_output = self.attn(q, k, v)
        # For o_proj
        attn_output = attn_output.view(q.shape[0], -1)
        output, _ = self.o_proj(attn_output)
        return output, (ori_k, v)


class HunYuanSparseMoeBlock(nn.Module):

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        layer_id: int = -1,
        prefix: str = "",
    ):
        super().__init__()
        self.tp_size = get_tensor_model_parallel_world_size()

        if self.tp_size > config.num_experts:
            raise ValueError(
                f"Tensor parallel size {self.tp_size} is greater than "
                f"the number of experts {config.num_experts}.")

        # Get layer_id topk if config.moe_topk is a list
        if isinstance(config.moe_topk, list):
            assert layer_id >= 0
            assert len(config.moe_topk) > layer_id
            top_k = config.moe_topk[layer_id]
        else:
            top_k = config.moe_topk

        # If it is moe, moe_intermediate_size is preferred
        intermediate_size = config.intermediate_size
        if config.moe_intermediate_size is not None:
            intermediate_size = (config.moe_intermediate_size if isinstance(
                config.moe_intermediate_size, int) else
                                 config.moe_intermediate_size[layer_id])

        self.experts = FusedMoE(
            num_experts=config.num_experts,
            top_k=top_k,
            hidden_size=config.hidden_size,
            intermediate_size=intermediate_size,
            reduce_results=False,
            renormalize=top_k > 1,
            quant_config=quant_config,
            prefix=f"{prefix}.experts",
        )

        self.gate = ReplicatedLinear(config.hidden_size,
                                     config.num_experts,
                                     bias=False,
                                     quant_config=None,
                                     prefix=f"{prefix}.gate")
        if config.use_mixed_mlp_moe > 0:
            # Get layer_id num_shared_expert if config.num_shared_expert is
            # a list.
            if isinstance(config.num_shared_expert, list):
                assert layer_id >= 0
                assert len(config.num_shared_expert) > layer_id
                num_shared_expert = config.num_shared_expert[layer_id]
            else:
                num_shared_expert = config.num_shared_expert

            self.shared_mlp = HunYuanMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size * num_shared_expert,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                reduce_results=False,
            )
        else:
            self.shared_mlp = None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # NOTE: hidden_states can have either 1D or 2D shape.
        orig_shape = hidden_states.shape
        hidden_dim = hidden_states.shape[-1]
        hidden_states = hidden_states.view(-1, hidden_dim)
        shared_output = None
        if self.shared_mlp is not None:
            shared_output = self.shared_mlp(hidden_states)

        # router_logits: (num_tokens, n_experts)
        router_logits, _ = self.gate(hidden_states)
        final_hidden_states = self.experts(hidden_states=hidden_states,
                                           router_logits=router_logits)
        if shared_output is not None:
            final_hidden_states = final_hidden_states + shared_output
        if self.tp_size > 1:
            final_hidden_states = tensor_model_parallel_all_reduce(
                final_hidden_states)

        return final_hidden_states.view(orig_shape)


class HunYuanDecoderLayer(nn.Module):

    def __init__(
        self,
        config: PretrainedConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        layer_id: int = -1,
    ) -> None:
        super().__init__()
        assert layer_id >= 0
        self.layer_id = layer_id
        self.hidden_size = config.hidden_size
        self.intermediate_size = (config.intermediate_size if isinstance(
            config.intermediate_size, int) else
                                  config.intermediate_size[layer_id])
        rope_theta = getattr(config, "rope_theta", 10000)
        rope_scaling = getattr(config, "rope_scaling", None)
        if rope_scaling is not None and getattr(
                config, "original_max_position_embeddings", None):
            rope_scaling["original_max_position_embeddings"] = (
                config.original_max_position_embeddings)
        max_position_embeddings = getattr(config, "max_position_embeddings",
                                          8192)
        attention_bias = getattr(config, "attention_bias", False) or getattr(
            config, "bias", False)
        cla_factor = _get_cla_factor(config)
        attention_type = (AttentionType.ENCODER_DECODER
                          if layer_id >= 0 and layer_id % cla_factor != 0 else
                          AttentionType.DECODER)
        if attention_type == AttentionType.DECODER:
            self.self_attn = HunYuanAttention(
                config=config,
                hidden_size=self.hidden_size,
                num_heads=config.num_attention_heads,
                num_kv_heads=getattr(config, "num_key_value_heads",
                                     config.num_attention_heads),
                rope_theta=rope_theta,
                rope_scaling=rope_scaling,
                max_position_embeddings=max_position_embeddings,
                quant_config=quant_config,
                bias=attention_bias,
                cache_config=cache_config,
                prefix=f"{prefix}.self_attn",
                layer_id=layer_id,
            )
        elif attention_type == AttentionType.ENCODER_DECODER:
            self.self_attn = HunYuanCrossAttention(
                config=config,
                hidden_size=self.hidden_size,
                num_heads=config.num_attention_heads,
                num_kv_heads=getattr(config, "num_key_value_heads",
                                     config.num_attention_heads),
                rope_theta=rope_theta,
                rope_scaling=rope_scaling,
                max_position_embeddings=max_position_embeddings,
                quant_config=quant_config,
                bias=attention_bias,
                cache_config=cache_config,
                prefix=f"{prefix}.self_attn",
                layer_id=layer_id,
            )
        else:
            raise RuntimeError(f"Unsupported attention type: {attention_type}")

        if _is_moe(config):
            self.mlp = HunYuanSparseMoeBlock(
                config=config,
                quant_config=quant_config,
                layer_id=layer_id,
                prefix=f"{prefix}.mlp",
            )
        else:
            self.mlp = HunYuanMLP(
                hidden_size=self.hidden_size,
                intermediate_size=self.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                bias=getattr(config, "mlp_bias", False),
                prefix=f"{prefix}.mlp",
            )

        self.input_layernorm = RMSNorm(config.hidden_size,
                                       eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size,
                                                eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
        kv_states: Optional[tuple[torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Self Attention
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(
                hidden_states, residual)
        hidden_states, ori_kv_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            kv_states=kv_states,
        )

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual, ori_kv_states


@support_torch_compile
class HunYuanModel(nn.Module):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        lora_config = vllm_config.lora_config

        self.config = config
        self.quant_config = quant_config
        self.padding_idx = config.pad_token_id
        lora_vocab = ((lora_config.lora_extra_vocab_size *
                       (lora_config.max_loras or 1)) if lora_config else 0)
        self.vocab_size = config.vocab_size + lora_vocab
        self.org_vocab_size = config.vocab_size
        if get_pp_group().is_first_rank or (config.tie_word_embeddings
                                            and get_pp_group().is_last_rank):
            self.embed_tokens = VocabParallelEmbedding(
                self.vocab_size,
                config.hidden_size,
                org_num_embeddings=config.vocab_size,
                quant_config=quant_config,
            )
        else:
            self.embed_tokens = PPMissingLayer()
        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: HunYuanDecoderLayer(
                config=config,
                layer_id=int(prefix.split(".")[-1]),
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=prefix,
            ),
            prefix=f"{prefix}.layers",
        )
        if get_pp_group().is_last_rank:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: Optional[torch.Tensor],
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors],
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.get_input_embeddings(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        cla_factor = _get_cla_factor(self.config)
        prev_kv_states = None
        for i in range(self.start_layer, self.end_layer):
            layer = self.layers[i]
            hidden_states, residual, kv_states = layer(
                positions,
                hidden_states,
                residual,
                prev_kv_states,
            )

            if (getattr(self.config, "use_cla", False)
                    and (i - self.start_layer) % cla_factor == 0):
                prev_kv_states = kv_states
            else:
                prev_kv_states = None

        if not get_pp_group().is_last_rank:
            return IntermediateTensors({
                "hidden_states": hidden_states,
                "residual": residual
            })

        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states

    def _split_qkv_weight(self, qkv: torch.Tensor):
        num_attention_heads = self.config.num_attention_heads
        num_kv_heads = getattr(self.config, "num_key_value_heads",
                               self.config.num_attention_heads)
        num_key_value_groups = num_attention_heads // num_kv_heads
        hidden_size = self.config.hidden_size

        if hasattr(self.config, "head_dim"):
            attention_head_dim = self.config.head_dim
        elif hasattr(self.config, "attention_head_dim"):
            attention_head_dim = self.config.attention_head_dim
        else:
            attention_head_dim = self.config.hidden_size // num_attention_heads

        qkv = qkv.reshape(num_kv_heads, num_key_value_groups + 2,
                          attention_head_dim, hidden_size)
        q, k, v = torch.split(qkv, (num_key_value_groups, 1, 1), dim=1)
        q = q.reshape(-1, hidden_size)
        k = k.reshape(-1, hidden_size)
        v = v.reshape(-1, hidden_size)
        return torch.concat((q, k, v))

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        if _is_moe(self.config):
            # Params for weights, fp8 weight scales, fp8 activation scales
            # (param_name, weight_name, expert_id, shard_id)
            return FusedMoE.make_expert_params_mapping(
                ckpt_gate_proj_name="gate_proj",
                ckpt_down_proj_name="down_proj",
                ckpt_up_proj_name="up_proj",
                num_experts=self.config.num_experts,
            )
        else:
            return []

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        cla_factor = _get_cla_factor(self.config)
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]

        num_attention_heads = self.config.num_attention_heads
        num_kv_heads = getattr(self.config, "num_key_value_heads",
                               self.config.num_attention_heads)
        split_params_mapping = [
            (".gate_up_proj", ".gate_and_up_proj", 2, [(1, 1), (0, 1)], None),
            (
                ".qkv_proj",
                ".qkv_proj",
                num_attention_heads + num_kv_heads * 2,
                [("q", num_attention_heads), ("k", num_kv_heads),
                 ("v", num_kv_heads)],
                self._split_qkv_weight,
            ),
        ]

        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        expert_params_mapping = self.get_expert_mapping()
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            if "gate_proj_bias" in name:
                name = name.replace("gate_proj_bias", "gate_proj.bias")
            if "up_proj_bias" in name:
                name = name.replace("up_proj_bias", "up_proj.bias")
            if ("rotary_emb.cos_cached" in name
                    or "rotary_emb.sin_cached" in name):
                # Models trained using ColossalAI may include these tensors in
                # the checkpoint. Skip them.
                continue
            # With tie_word_embeddings, we can skip lm_head.weight
            # The weight might appear unnecessarily in the files if the model is
            # processed with quantization, LoRA, fine-tuning, etc.
            if self.config.tie_word_embeddings and "lm_head.weight" in name:
                continue
            if self.quant_config is not None and (
                    scale_name := self.quant_config.get_cache_scale(name)):
                # Loading kv cache scales for compressed-tensors quantization
                param = params_dict[scale_name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                loaded_weight = loaded_weight[0]
                weight_loader(param, loaded_weight)
                continue

            is_found = False
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                if "mlp.experts" in name:
                    continue
                # cross layer only have q_proj, skip qkv pack
                if weight_name == ".q_proj":
                    match = re.search(r"layers\.\d+", name)
                    if match:
                        layer_id = int(match.group(0).split(".")[-1])
                        if cla_factor > 1 and layer_id % cla_factor != 0:
                            continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue

                if is_pp_missing_parameter(name, self):
                    continue

                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                loaded_params.add(name)
                is_found = True
                break
            if is_found:
                continue

            for (
                    param_name,
                    weight_name,
                    den,
                    split_param,
                    func,
            ) in split_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue

                if is_pp_missing_parameter(name, self):
                    continue

                assert loaded_weight.shape[0] % den == 0
                units = loaded_weight.shape[0] // den

                param = params_dict[name]
                weight_loader = param.weight_loader
                offset = 0
                for shard_id, num in split_param:
                    new_offset = offset + num * units
                    if func:
                        weight_loader(param,
                                      func(loaded_weight)[offset:new_offset],
                                      shard_id)
                    else:
                        weight_loader(param, loaded_weight[offset:new_offset],
                                      shard_id)
                    offset = new_offset

                break
            else:
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                for mapping in expert_params_mapping:
                    param_name, weight_name, expert_id, shard_id = mapping
                    if weight_name not in name:
                        continue
                    name = name.replace(weight_name, param_name)
                    # Skip layers on other devices.
                    if is_pp_missing_parameter(name, self):
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
                    # Remapping the name of FP8 kv-scale.
                    name = maybe_remap_kv_scale_name(name, params_dict)
                    if name is None:
                        continue

                    if is_pp_missing_parameter(name, self):
                        continue

                    if "mlp.gate.wg." in name:
                        name = name.replace("wg.", "")

                    param = params_dict[name]
                    weight_loader = getattr(param, "weight_loader",
                                            default_weight_loader)
                    weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params


class HunYuanV1Base(nn.Module, SupportsLoRA):
    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj": [
            "gate_proj",
            "up_proj",
        ],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.quant_config = quant_config

        self.model = HunYuanModel(vllm_config=vllm_config, prefix="model")
        if get_pp_group().is_last_rank:
            self.unpadded_vocab_size = config.vocab_size
            self.lm_head = ParallelLMHead(
                self.unpadded_vocab_size,
                config.hidden_size,
                org_num_embeddings=config.vocab_size,
                padding_size=DEFAULT_VOCAB_PADDING_SIZE,
                quant_config=quant_config,
            )
            if config.tie_word_embeddings:
                self.lm_head.weight = self.model.embed_tokens.weight

            logit_scale = getattr(config, "logit_scale", 1.0)
            self.logits_processor = LogitsProcessor(self.unpadded_vocab_size,
                                                    config.vocab_size,
                                                    logit_scale)
        else:
            self.lm_head = PPMissingLayer()

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        model_output = self.model(input_ids, positions, intermediate_tensors,
                                  inputs_embeds)
        return model_output

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        logits = self.logits_processor(self.lm_head, hidden_states,
                                       sampling_metadata)
        return logits

    def make_empty_intermediate_tensors(
            self, batch_size: int, dtype: torch.dtype,
            device: torch.device) -> IntermediateTensors:
        return IntermediateTensors({
            "hidden_states":
            torch.zeros((batch_size, self.config.hidden_size),
                        dtype=dtype,
                        device=device),
            "residual":
            torch.zeros((batch_size, self.config.hidden_size),
                        dtype=dtype,
                        device=device),
        })

    def load_weights(self, weights: Iterable[tuple[str,
                                                   torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=(["lm_head."]
                           if self.config.tie_word_embeddings else None),
        )
        return loader.load_weights(weights)

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        return self.model.get_expert_mapping()


class HunYuanDenseV1ForCausalLM(HunYuanV1Base):
    pass


class HunYuanMoEV1ForCausalLM(HunYuanV1Base):
    pass
