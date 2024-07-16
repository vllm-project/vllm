# coding=utf-8
# Adapted from
# https://github.com/huggingface/transformers/blob/v4.28.0/src/transformers/models/llama/modeling_llama.py
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
"""Inference-only Mixtral model."""
from typing import Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from transformers import MixtralConfig

from vllm.attention import Attention, AttentionMetadata
from vllm.config import CacheConfig
from vllm.distributed import (get_tensor_model_parallel_rank,
                              get_tensor_model_parallel_world_size,
                              tensor_model_parallel_all_reduce)
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (QKVParallelLinear,
                                               ReplicatedLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.sampler import Sampler
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import SamplerOutput
import intel_extension_for_pytorch as ipex

class MixtralMLP(nn.Module):

    def __init__(
        self,
        num_experts: int,
        hidden_size: int,
        intermediate_size: int,
        quant_config: Optional[QuantizationConfig] = None,
    ) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.ffn_dim = intermediate_size
        self.hidden_dim = hidden_size

        self.w1 = ReplicatedLinear(self.hidden_dim,
                                   self.ffn_dim,
                                   bias=False,
                                   quant_config=quant_config)
        self.w2 = ReplicatedLinear(self.ffn_dim,
                                   self.hidden_dim,
                                   bias=False,
                                   quant_config=quant_config)
        self.w3 = ReplicatedLinear(self.hidden_dim,
                                   self.ffn_dim,
                                   bias=False,
                                   quant_config=quant_config)

        # TODO: Use vllm's SiluAndMul
        self.act_fn = nn.SiLU()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        w1_out, _ = self.w1(hidden_states)
        w1_out = self.act_fn(w1_out)
        w3_out, _ = self.w3(hidden_states)
        current_hidden_states = w1_out * w3_out
        current_hidden_states, _ = self.w2(current_hidden_states)
        return current_hidden_states


class MixtralMoE(nn.Module):

    def __init__(
        self,
        config: MixtralConfig,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()
        self.config = config
        self.rank = get_tensor_model_parallel_rank()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.num_total_experts = config.num_local_experts
        self.top_k = config.num_experts_per_tok
        if self.tp_size > self.num_total_experts:
            raise ValueError(
                f"Tensor parallel size {self.tp_size} is greater than "
                f"the number of experts {self.num_total_experts}.")
        # Split experts equally between ranks
        self.expert_indicies = np.array_split(range(
            self.num_total_experts), self.tp_size)[self.rank].tolist()
        if not self.expert_indicies:
            raise ValueError(
                f"Rank {self.rank} has no experts assigned to it.")
        self.experts = nn.ModuleList([
            MixtralMLP(self.num_total_experts,
                       config.hidden_size,
                       config.intermediate_size,
                       quant_config=quant_config)
            if idx in self.expert_indicies else None
            for idx in range(self.num_total_experts)
        ])
        self.gate = ReplicatedLinear(config.hidden_size,
                                     self.num_total_experts,
                                     bias=False,
                                     quant_config=None)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        # router_logits: (num_tokens, n_experts)
        router_logits, _ = self.gate(hidden_states)

        if hasattr(self, "ipex_moe"):
            final_hidden_states = self.ipex_moe(hidden_states, router_logits, self.top_k)
        else:
            routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
            routing_weights, selected_experts = torch.topk(routing_weights,
                                                        self.top_k,
                                                        dim=-1)
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)

            final_hidden_states = None
            for expert_idx in self.expert_indicies:
                expert_layer = self.experts[expert_idx]
                expert_mask = (selected_experts == expert_idx)
                expert_weights = (routing_weights * expert_mask).sum(dim=-1,
                                                                    keepdim=True)

                current_hidden_states = expert_layer(hidden_states).mul_(
                    expert_weights)
                if final_hidden_states is None:
                    final_hidden_states = current_hidden_states
                else:
                    final_hidden_states.add_(current_hidden_states)

        if not hasattr(self, "ipex_moe"):#and hasattr(self.experts[0].w1, "ipex_qlinear"):
            if self.tp_size > 1 :
                self.ipex_moe = ipex.llm.modules.LinearMOETP(experts_module=self.experts, expert_indicies=self.expert_indicies)
            else:
                self.ipex_moe = ipex.llm.modules.LinearMOE(experts_module=self.experts)

        return tensor_model_parallel_all_reduce(final_hidden_states).view(
            num_tokens, hidden_dim)


class MixtralAttention(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position: int = 4096 * 32,
        rope_theta: float = 10000,
        quant_config: Optional[QuantizationConfig] = None,
        cache_config: Optional[CacheConfig] = None,
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
        self.head_dim = hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.rope_theta = rope_theta

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=False,
            quant_config=quant_config,
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            quant_config=quant_config,
        )
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position,
            base=int(self.rope_theta),
            is_neox_style=True,
        )
        self.attn = Attention(self.num_heads,
                              self.head_dim,
                              self.scaling,
                              num_kv_heads=self.num_kv_heads,
                              cache_config=cache_config,
                              quant_config=quant_config)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        is_prompt,
        block_tables,
        num_prefills,
        num_prefill_tokens,
        num_decode_tokens,
        slot_mapping,
        seq_lens,
        seq_lens_tensor=None,
        max_decode_seq_len=None,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v, kv_cache, is_prompt, block_tables,num_prefills,num_prefill_tokens,num_decode_tokens,slot_mapping,seq_lens,seq_lens_tensor,max_decode_seq_len)
        # move self.o_proj to MixtralDecoderLayer to enable linear+add fusion when tp_size <=1
        # output, _ = self.o_proj(attn_output)
        return attn_output

class MixtralDecoderLayer(nn.Module):

    def __init__(
        self,
        config: MixtralConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        # Requires transformers > 4.32.0
        rope_theta = getattr(config, "rope_theta", 10000)
        self.self_attn = MixtralAttention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            max_position=config.max_position_embeddings,
            num_kv_heads=config.num_key_value_heads,
            rope_theta=rope_theta,
            cache_config=cache_config,
            quant_config=quant_config)
        self.block_sparse_moe = MixtralMoE(config=config,
                                           quant_config=quant_config)
        self.input_layernorm = RMSNorm(config.hidden_size,
                                       eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size,
                                                eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        residual: Optional[torch.Tensor],
        is_prompt,
        block_tables,
        num_prefills,
        num_prefill_tokens,
        num_decode_tokens,
        slot_mapping,
        seq_lens,
        seq_lens_tensor=None,
        max_decode_seq_len=None,
    ) -> torch.Tensor:
        # Self Attention
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(
                hidden_states, residual)
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            kv_cache=kv_cache,
            is_prompt=is_prompt,
            block_tables=block_tables,
            num_prefills=num_prefills,
            num_prefill_tokens=num_prefill_tokens,
            num_decode_tokens=num_decode_tokens,
            slot_mapping=slot_mapping,
            seq_lens=seq_lens,
            seq_lens_tensor=seq_lens_tensor,
            max_decode_seq_len=max_decode_seq_len,
        )

        if self.self_attn.o_proj.tp_size <=1 and not hasattr(self, "ipex_fusion") and hasattr(self.self_attn.o_proj, "ipex_qlinear"):
                self.ipex_fusion = ipex.llm.modules.LinearAdd(self.self_attn.o_proj.ipex_qlinear)
        if hasattr(self, "ipex_fusion"):
            hidden_states = self.ipex_fusion(hidden_states, residual)
            if not self.self_attn.o_proj.skip_bias_add and self.self_attn.o_proj.bias is not None:
                hidden_states = hidden_states + self.self_attn.o_proj.bias
            residual = hidden_states
            hidden_states = self.post_attention_layernorm(
                hidden_states)
        else:
            hidden_states, _ = self.self_attn.o_proj(hidden_states)
            hidden_states, residual = self.post_attention_layernorm(
                hidden_states, residual)
        hidden_states = self.block_sparse_moe(hidden_states)
        return hidden_states, residual


class MixtralModel(nn.Module):

    def __init__(
        self,
        config: MixtralConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
    ) -> None:
        super().__init__()
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
        )
        self.layers = nn.ModuleList([
            MixtralDecoderLayer(config,
                                cache_config,
                                quant_config=quant_config)
            for _ in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        is_prompt,
        block_tables,
        num_prefills,
        num_prefill_tokens,
        num_decode_tokens,
        slot_mapping,
        seq_lens,
        seq_lens_tensor=None,
        max_decode_seq_len=None,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        for i in range(len(self.layers)):
            layer = self.layers[i]
            hidden_states, residual = layer(positions, 
                                            hidden_states,
                                            kv_caches[i], 
                                            residual,
                                            is_prompt,
                                            block_tables,
                                            num_prefills,
                                            num_prefill_tokens,
                                            num_decode_tokens,
                                            slot_mapping,
                                            seq_lens,
                                            seq_lens_tensor,
                                            max_decode_seq_len,
                                            )
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class MixtralForCausalLM(nn.Module):
    fall_back_to_pt_during_load = False

    def __init__(
        self,
        config: MixtralConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.quant_config = quant_config
        self.model = MixtralModel(config, cache_config, quant_config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.sampler = Sampler()
        self.trace_first = None
        self.trace_next = None
    @torch.no_grad
    def enable_jit(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        is_prompt,
        block_tables,
        num_prefills,
        num_prefill_tokens,
        num_decode_tokens,
        slot_mapping,
        seq_lens,
        seq_lens_tensor=None,
        max_decode_seq_len=None,
    ) -> torch.Tensor:

        if is_prompt:
                self.model(input_ids, positions, kv_caches, is_prompt, block_tables,num_prefills,num_prefill_tokens,num_decode_tokens,slot_mapping,seq_lens,seq_lens_tensor,max_decode_seq_len)
                example_input = (
                    input_ids,
                    positions,
                    kv_caches,
                    is_prompt, block_tables,num_prefills,num_prefill_tokens,num_decode_tokens,slot_mapping,seq_lens
                )
                self.trace_first = torch.jit.trace(self.model, example_input, check_trace=False, strict=False)
                self.trace_first = torch.jit.freeze(self.trace_first)
                self.trace_first(*example_input)
                self.trace_first(*example_input)
        else:
                self.model(input_ids, positions, kv_caches, is_prompt, block_tables,num_prefills,num_prefill_tokens,num_decode_tokens,slot_mapping,seq_lens,seq_lens_tensor,max_decode_seq_len)
                example_input = (
                    input_ids,
                    positions,
                    kv_caches,
                    is_prompt, block_tables,num_prefills,num_prefill_tokens,num_decode_tokens,slot_mapping,seq_lens,seq_lens_tensor,max_decode_seq_len
                )
                self.trace_next = torch.jit.trace(
                    self.model, example_input, check_trace=False, strict=False
                )
                self.trace_next = torch.jit.freeze(self.trace_next)
                self.trace_next(*example_input)
                self.trace_next(*example_input)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        is_prompt=torch.tensor(attn_metadata.is_prompt)
        block_tables=attn_metadata.block_tables
        num_prefills=torch.tensor(attn_metadata.num_prefills)
        num_prefill_tokens=torch.tensor(attn_metadata.num_prefill_tokens)
        num_decode_tokens=torch.tensor(attn_metadata.num_decode_tokens)
        slot_mapping = attn_metadata.slot_mapping
        seq_lens=torch.tensor(attn_metadata.seq_lens)
        seq_lens_tensor=attn_metadata.seq_lens_tensor if attn_metadata.seq_lens_tensor is not None else None
        max_decode_seq_len=torch.tensor(attn_metadata.max_decode_seq_len) if attn_metadata.max_decode_seq_len is not None else None
        attn_bias = attn_metadata.attn_bias
        if kv_caches[0] is not None:
            if attn_metadata.is_prompt:
                if self.trace_first is None:
                    self.enable_jit(input_ids, positions, kv_caches, is_prompt, block_tables,num_prefills,num_prefill_tokens,num_decode_tokens,slot_mapping,seq_lens)
                    print("RUNTIME INFO: Optimization is enabled for first token!")
                hidden_states = self.trace_first(
                    input_ids,
                    positions,
                    kv_caches,
                    is_prompt, block_tables,num_prefills,num_prefill_tokens,num_decode_tokens,slot_mapping,seq_lens
                )
            else:
                if self.trace_next is None:
                    self.enable_jit(input_ids, positions, kv_caches, is_prompt, block_tables,num_prefills,num_prefill_tokens,num_decode_tokens,slot_mapping,seq_lens,seq_lens_tensor,max_decode_seq_len)
                    print("RUNTIME INFO: Optimization is enabled for next token!")
                hidden_states = self.trace_next(
                    input_ids,
                    positions,
                    kv_caches,
                    is_prompt, block_tables,num_prefills,num_prefill_tokens,num_decode_tokens,slot_mapping,seq_lens,seq_lens_tensor,max_decode_seq_len
                )
        else:
            hidden_states = self.model(input_ids, positions, kv_caches, is_prompt, block_tables,num_prefills,num_prefill_tokens,num_decode_tokens,slot_mapping,seq_lens,seq_lens_tensor,max_decode_seq_len)
        return hidden_states

    def compute_logits(self, hidden_states: torch.Tensor,
                       sampling_metadata: SamplingMetadata) -> torch.Tensor:
        logits = self.logits_processor(self.lm_head.weight, hidden_states,
                                       sampling_metadata)
        return logits

    def sample(
        self,
        logits: Optional[torch.Tensor],
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        next_tokens = self.sampler(logits, sampling_metadata)
        return next_tokens

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
        ]

        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            for (param_name, weight_name, shard_id) in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                # Skip experts that are not assigned to this worker.
                if ("block_sparse_moe.experts." in name
                        and name not in params_dict):
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)
