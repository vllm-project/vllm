# coding=utf-8
# Adapted from
# https://github.com/huggingface/transformers/blob/v4.28.0/src/transformers/models/gptj/modeling_gptj.py
# Copyright 2023 The vLLM team.
# Copyright 2021 The EleutherAI and HuggingFace Teams. All rights reserved.
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
"""Inference-only GPT-J model compatible with HuggingFace weights."""
from typing import Iterable, List, Optional, Tuple

import torch
from torch import nn
import intel_extension_for_pytorch as ipex
from transformers import GPTJConfig

from vllm.attention import Attention, AttentionMetadata
from vllm.config import CacheConfig
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.layers.activation import get_act_fn
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               QKVParallelLinear,
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


class GPTJAttention(nn.Module):

    def __init__(
        self,
        config: GPTJConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()
        self.total_num_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_size = self.hidden_size // self.total_num_heads

        self.qkv_proj = QKVParallelLinear(
            config.hidden_size,
            self.head_size,
            self.total_num_heads,
            bias=False,
            quant_config=quant_config,
        )
        self.out_proj = RowParallelLinear(
            config.hidden_size,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
        )

        tp_world_size = get_tensor_model_parallel_world_size()
        assert self.total_num_heads % tp_world_size == 0
        self.num_heads = self.total_num_heads // tp_world_size

        scaling = self.head_size**-0.5
        assert getattr(config, "rotary", True)
        assert config.rotary_dim % 2 == 0
        rope_theta = getattr(config, "rope_theta", 10000)
        max_position_embeddings = getattr(config, "max_position_embeddings",
                                          8192)
        self.rotary_emb = get_rope(
            self.head_size,
            rotary_dim=config.rotary_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
            is_neox_style=False,
        )
        self.attn = Attention(self.num_heads,
                              self.head_size,
                              scaling,
                              cache_config=cache_config,
                              quant_config=quant_config)

    def forward(
        self,
        position_ids: torch.Tensor,
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
        q, k, v = qkv.chunk(chunks=3, dim=-1)
        q, k = self.rotary_emb(position_ids, q, k)
        attn_output = self.attn(q, k, v, kv_cache, is_prompt, block_tables,num_prefills,num_prefill_tokens,num_decode_tokens,slot_mapping,seq_lens,seq_lens_tensor,max_decode_seq_len)
        attn_output, _ = self.out_proj(attn_output)
        return attn_output


class GPTJMLP(nn.Module):

    def __init__(
        self,
        intermediate_size: int,
        config: GPTJConfig,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()
        hidden_size = config.n_embd
        self.fc_in = ColumnParallelLinear(
            hidden_size,
            intermediate_size,
            quant_config=quant_config,
        )
        self.fc_out = RowParallelLinear(
            intermediate_size,
            hidden_size,
            quant_config=quant_config,
        )
        self.act = get_act_fn(config.activation_function, quant_config,
                              intermediate_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if not hasattr(self, "ipex_fusion"):
            if hasattr(self.fc_in, "ipex_linear"):
                self.ipex_fusion = ipex.llm.modules.LinearNewGelu(self.fc_in.ipex_linear)
            elif hasattr(self.fc_in, "ipex_qlinear"):
                self.ipex_fusion = ipex.llm.modules.LinearNewGelu(self.fc_in.ipex_qlinear)
        if hasattr(self, "ipex_fusion"):
            hidden_states = self.ipex_fusion(hidden_states)
            if hasattr(self.act, "scales"): #AWQ scales
                hidden_states = hidden_states / self.act.scales
        else:
            hidden_states, _ = self.fc_in(hidden_states)
            hidden_states = self.act(hidden_states)
        # move self.fc_out to GPTJBlock to enable linear+add+add fusion when tp_size <=1c
        # hidden_states, _ = self.fc_out(hidden_states)
        return hidden_states


class GPTJBlock(nn.Module):

    def __init__(
        self,
        config: GPTJConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()
        inner_dim = (4 * config.n_embd
                     if config.n_inner is None else config.n_inner)
        self.ln_1 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.attn = GPTJAttention(config, cache_config, quant_config)
        self.mlp = GPTJMLP(inner_dim, config, quant_config)

    def forward(
        self,
        position_ids: torch.Tensor,
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
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_output = self.attn(
            position_ids=position_ids,
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
        mlp_output = self.mlp(hidden_states)
        if self.mlp.fc_out.tp_size <=1 and not hasattr(self, "ipex_fusion"):
            if hasattr(self.mlp.fc_out, "ipex_linear"):
                self.ipex_fusion = ipex.llm.modules.LinearAddAdd(self.mlp.fc_out.ipex_linear)
            elif hasattr(self.mlp.fc_out, "ipex_qlinear"):
                self.ipex_fusion = ipex.llm.modules.LinearAddAdd(self.mlp.fc_out.ipex_qlinear)
        if hasattr(self, "ipex_fusion"):
            hidden_states = self.ipex_fusion(
                mlp_output, attn_output,  residual
            )
            if not self.mlp.fc_out.skip_bias_add and self.mlp.fc_out.bias is not None:
                hidden_states = hidden_states + self.mlp.fc_out.bias
        else:
            mlp_output, _ = self.mlp.fc_out(mlp_output)
            hidden_states = attn_output + mlp_output + residual
        return hidden_states


class GPTJModel(nn.Module):

    def __init__(
        self,
        config: GPTJConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()
        self.config = config
        self.embed_dim = config.n_embd
        self.wte = VocabParallelEmbedding(
            config.vocab_size,
            self.embed_dim,
        )
        self.h = nn.ModuleList([
            GPTJBlock(config, cache_config, quant_config)
            for _ in range(config.n_layer)
        ])
        self.ln_f = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
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
        hidden_states = self.wte(input_ids)
        for i in range(len(self.h)):
            layer = self.h[i]
            hidden_states = layer(
                position_ids,
                hidden_states,
                kv_caches[i],
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
        hidden_states = self.ln_f(hidden_states)
        return hidden_states


class GPTJForCausalLM(nn.Module):

    def __init__(
        self,
        config: GPTJConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()
        self.config = config
        self.quant_config = quant_config
        assert not config.tie_word_embeddings
        self.transformer = GPTJModel(config, cache_config, quant_config)
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.n_embd,
            bias=True,
        )
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.sampler = Sampler()
        self.trace_first=None
        self.trace_next=None

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
                self.transformer(input_ids, positions, kv_caches, is_prompt, block_tables,num_prefills,num_prefill_tokens,num_decode_tokens,slot_mapping,seq_lens,seq_lens_tensor,max_decode_seq_len)
                example_input = (
                    input_ids,
                    positions,
                    kv_caches,
                    is_prompt, block_tables,num_prefills,num_prefill_tokens,num_decode_tokens,slot_mapping,seq_lens
                )
                self.trace_first = torch.jit.trace(self.transformer, example_input, check_trace=False, strict=False)
                self.trace_first = torch.jit.freeze(self.trace_first)
                self.trace_first(*example_input)
                self.trace_first(*example_input)
        else:
                example_input = (
                    input_ids,
                    positions,
                    kv_caches,
                    is_prompt, block_tables,num_prefills,num_prefill_tokens,num_decode_tokens,slot_mapping,seq_lens,seq_lens_tensor,max_decode_seq_len
                )
                self.trace_next = torch.jit.trace(
                    self.transformer, example_input, check_trace=False, strict=False
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
                hidden_states = self.trace_first(
                    input_ids,
                    positions,
                    kv_caches,
                    is_prompt, block_tables,num_prefills,num_prefill_tokens,num_decode_tokens,slot_mapping,seq_lens
                )
            else:
                if self.trace_next is None:
                    self.enable_jit(input_ids, positions, kv_caches, is_prompt, block_tables,num_prefills,num_prefill_tokens,num_decode_tokens,slot_mapping,seq_lens,seq_lens_tensor,max_decode_seq_len)
                hidden_states = self.trace_next(
                    input_ids,
                    positions,
                    kv_caches,
                    is_prompt, block_tables,num_prefills,num_prefill_tokens,num_decode_tokens,slot_mapping,seq_lens,seq_lens_tensor,max_decode_seq_len
                )
        else:
            # TorchSDPAMetadata(seq_lens_tensor=None, max_decode_seq_len=None, block_tables=tensor([]), num_prefills=1, num_prefill_tokens=5, num_decode_tokens=0, slot_mapping=tensor([9344, 9345, 9346, 9347, 9348]), is_prompt=True, seq_lens=[5])
            # TorchSDPAMetadata(seq_lens_tensor=tensor([6], dtype=torch.int32), max_decode_seq_len=6, block_tables=tensor([[584]], dtype=torch.int32), num_prefills=0, num_prefill_tokens=0, num_decode_tokens=1, slot_mapping=tensor([9349]), is_prompt=False, seq_lens=[6])
            hidden_states = self.transformer(input_ids, positions, kv_caches, is_prompt, block_tables,num_prefills,num_prefill_tokens,num_decode_tokens,slot_mapping,seq_lens,seq_lens_tensor,max_decode_seq_len)

        return hidden_states

    def compute_logits(self, hidden_states: torch.Tensor,
                       sampling_metadata: SamplingMetadata) -> torch.Tensor:
        logits = self.logits_processor(self.lm_head.weight, hidden_states,
                                       sampling_metadata, self.lm_head.bias)
        return logits

    def sample(
        self,
        logits: torch.Tensor,
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
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            if "attn.bias" in name or "attn.masked_bias" in name:
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
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)
