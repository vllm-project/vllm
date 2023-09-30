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
"""Inference-only LLaMA model compatible with HuggingFace weights.

The input of the model is flattened to a 1D tensor of tokens. The model uses
InputMetadata to extract the original 2D shape of the input.
"""
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import nn
from transformers import LlamaConfig

from vllm.model_executor.input_metadata import InputMetadata
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.attention import PagedAttentionWithRoPE
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.quantized_linear import ParallelLinear
from vllm.model_executor.layers.sampler import Sampler
from vllm.model_executor.parallel_utils.parallel_state import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.model_executor.parallel_utils.tensor_parallel import VocabParallelEmbedding
from vllm.model_executor.quantization_utils import QuantizationConfig
from vllm.model_executor.weight_utils import (
    convert_pyslice_to_tensor,
    hf_model_weights_iterator,
    load_padded_tensor_parallel_vocab,
    load_tensor_parallel_weights,
)
from vllm.sequence import SamplerOutput

KVCache = Tuple[torch.Tensor, torch.Tensor]


class LlamaMLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: Optional[QuantizationConfig] = None,
    ) -> None:
        super().__init__()
        self.gate_up_proj = ParallelLinear.column(
            hidden_size,
            2 * intermediate_size,
            bias=False,
            gather_output=False,
            perform_initialization=False,
            quant_config=quant_config,
        )
        self.down_proj = ParallelLinear.row(
            intermediate_size,
            hidden_size,
            bias=False,
            input_is_parallel=True,
            perform_initialization=False,
            quant_config=quant_config,
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


class LlamaAttention(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        rope_theta: float = 10000,
        rope_scaling: Optional[Dict[str, Any]] = None,
        max_position_embeddings: int = 8192,
        quant_config: Optional[QuantizationConfig] = None,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        assert self.total_num_kv_heads % tp_size == 0
        self.num_kv_heads = self.total_num_kv_heads // tp_size
        self.head_dim = hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings

        self.qkv_proj = ParallelLinear.column(
            hidden_size,
            (self.total_num_heads + 2 * self.total_num_kv_heads) *
            self.head_dim,
            bias=False,
            gather_output=False,
            perform_initialization=False,
            quant_config=quant_config,
        )
        self.o_proj = ParallelLinear.row(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            input_is_parallel=True,
            perform_initialization=False,
            quant_config=quant_config,
        )
        self.attn = PagedAttentionWithRoPE(
            self.num_heads,
            self.head_dim,
            self.scaling,
            base=self.rope_theta,
            max_position=self.max_position_embeddings,
            rotary_dim=self.head_dim,
            num_kv_heads=self.num_kv_heads,
            rope_scaling=rope_scaling,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: KVCache,
        input_metadata: InputMetadata,
        cache_event: Optional[torch.cuda.Event],
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        k_cache, v_cache = kv_cache
        attn_output = self.attn(positions, q, k, v, k_cache, v_cache,
                                input_metadata, cache_event)
        output, _ = self.o_proj(attn_output)
        return output


class LlamaDecoderLayer(nn.Module):

    def __init__(
        self,
        config: LlamaConfig,
        quant_config: Optional[QuantizationConfig] = None,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        # Requires transformers > 4.32.0
        rope_theta = getattr(config, "rope_theta", 10000)
        rope_scaling = getattr(config, "rope_scaling", None)
        max_position_embeddings = getattr(config, "max_position_embeddings",
                                          8192)
        self.self_attn = LlamaAttention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            max_position_embeddings=max_position_embeddings,
            quant_config=quant_config,
        )
        self.mlp = LlamaMLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
        )
        self.input_layernorm = RMSNorm(config.hidden_size,
                                       eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size,
                                                eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: KVCache,
        input_metadata: InputMetadata,
        cache_event: Optional[torch.cuda.Event],
    ) -> torch.Tensor:
        # Self Attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            kv_cache=kv_cache,
            input_metadata=input_metadata,
            cache_event=cache_event,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class LlamaModel(nn.Module):

    def __init__(
        self,
        config: LlamaConfig,
        quant_config: Optional[QuantizationConfig] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        vocab_size = ((config.vocab_size + 63) // 64) * 64
        self.embed_tokens = VocabParallelEmbedding(
            vocab_size, config.hidden_size, perform_initialization=False)
        self.layers = nn.ModuleList([
            LlamaDecoderLayer(config, quant_config)
            for _ in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[KVCache],
        input_metadata: InputMetadata,
        cache_events: Optional[List[torch.cuda.Event]],
    ) -> torch.Tensor:
        # print("in llama forward")
        # print(f"{input_ids=}")
        # print(f"{positions=}")
        # print(
        #     f"{[(kvcache[0].data_ptr(), kvcache[1].data_ptr()) if kvcache[0] is not None else () for kvcache in kv_caches]=}"
        # )
        # print(f"{input_metadata=}")
        hidden_states = self.embed_tokens(input_ids)
        for i in range(len(self.layers)):
            if cache_events is None:
                cache_event = None
            else:
                cache_event = cache_events[i]
            layer = self.layers[i]
            hidden_states = layer(
                positions,
                hidden_states,
                kv_caches[i],
                input_metadata,
                cache_event,
            )
        hidden_states = self.norm(hidden_states)
        return hidden_states


class LlamaForCausalLM(nn.Module):

    def __init__(
        self,
        config: LlamaConfig,
        quant_config: Optional[QuantizationConfig] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.quant_config = quant_config
        self.model = LlamaModel(config, quant_config)
        vocab_size = ((config.vocab_size + 63) // 64) * 64
        # NOTE: The LM head is not quantized.
        self.lm_head = ParallelLinear.column(
            config.hidden_size,
            vocab_size,
            bias=False,
            gather_output=False,
            perform_initialization=False,
            quant_config=None,
        )
        self.sampler = Sampler(config.vocab_size)

        self.compiled_model = None
        self._cuda_graph = None
        self._compiled_tensors = None
        self._compiled_logits = None
        self._compiled_input_metadata = None

    def _compile_model(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[KVCache],
        input_metadata: InputMetadata,
        cache_events: Optional[List[torch.cuda.Event]],
    ):
        print(f"{input_ids=}")
        print(f"{positions=}")
        print(f"{input_metadata=}")
        assert (self._cuda_graph is None and self._compiled_tensors is None and
                self._compiled_logits is None), "Already compiled the model"

        print("recording CUDA graph")

        self._cuda_graph: Dict[int, torch.cuda.CUDAGraph] = {}
        self._compiled_tensors: Dict[int, Tuple[torch.Tensor,
                                                torch.Tensor, ], ] = {}
        self._compiled_logits: Dict[int, torch.Tensor] = {}

        # warm up. but why?
        # s = torch.cuda.Stream()
        # print(f"{s.device=}")
        # print(f"{torch.cuda.current_stream().device=}")
        # s.wait_stream(torch.cuda.current_stream())
        # with torch.cuda.stream(s):
        #     _ = self.model.forward(*self._compiled_inputs)
        # torch.cuda.current_stream().wait_stream(s)

        batch_size = input_ids.shape[0]
        for i in range(batch_size, 0, -8):
            print("recording for batch size ", i)
            pool = (None if i == batch_size else
                    self._cuda_graph[batch_size].pool())  # reusing memory pool

            self._cuda_graph[i] = torch.cuda.CUDAGraph()

            # Need the following tensors from input_metadata:
            # input_metadata.block_tables,
            # input_metadata.context_lens,
            # input_metadata.max_context_len,

            self._compiled_input_metadata = InputMetadata(
                input_metadata.seq_groups,
                input_metadata.seq_data,
                input_metadata.prompt_lens,
                input_metadata.slot_mapping.clone(),
                input_metadata.context_lens.clone(),
                input_metadata.max_context_len,
                input_metadata.block_tables.clone(),
                input_metadata.use_cuda_graph,
            )

            self._compiled_input_metadata.seq_data = None
            self._compiled_input_metadata.seq_groups = None
            self._compiled_input_metadata.prompt_lens = None
            self._compiled_input_metadata.max_context_len = -1
            self.num_prompts = -1
            self.num_prompt_tokens = -1
            self.num_generation_tokens = -1
            self.num_valid_tokens = -1
            self.max_num_blocks_per_seq = -1

            # Set during the execution of the first attention op.
            self.attn_bias = None

            self._compiled_tensors[i] = tuple([
                input_ids[:i].clone(),
                positions[:i].clone(),
            ])

            print("warm up before recording")
            # make one test prediction before recording
            self._compiled_logits[i] = self.model.forward(
                *self._compiled_tensors[i],
                kv_caches=kv_caches,
                input_metadata=self._compiled_input_metadata,
                cache_events=None,
            )

            print("actually recording")
            with torch.cuda.graph(self._cuda_graph[i], pool=pool):
                self._compiled_logits[i] = self.model.forward(
                    *self._compiled_tensors[i],
                    kv_caches=kv_caches,
                    input_metadata=self._compiled_input_metadata,
                    cache_events=None,
                )

        print("record fininshed")

        def replay(
            batch_size: int,
            input_ids: torch.Tensor,
            positions: torch.Tensor,
            kv_caches: List[KVCache],
            input_metadata: InputMetadata,
            cache_events: Optional[List[torch.cuda.Event]],
        ):
            # print("replaying with batch size ", batch_size)
            self._compiled_tensors[batch_size][0].copy_(input_ids)
            self._compiled_tensors[batch_size][1].copy_(positions)
            self._compiled_input_metadata.block_tables.copy_(
                input_metadata.block_tables)
            self._compiled_input_metadata.context_lens.copy_(
                input_metadata.context_lens)
            self._compiled_input_metadata.slot_mapping.copy_(
                input_metadata.slot_mapping)

            self._cuda_graph[batch_size].replay()

            return self._compiled_logits[batch_size]

        return replay

    def compile_and_call_model(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[KVCache],
        input_metadata: InputMetadata,
        cache_events: Optional[List[torch.cuda.Event]],
    ) -> torch.Tensor:
        if self.compiled_model is None:
            self.compiled_model = self._compile_model(input_ids, positions,
                                                      kv_caches,
                                                      input_metadata,
                                                      cache_events)

        return self.compiled_model(
            input_ids.shape[0],
            input_ids,
            positions,
            kv_caches,
            input_metadata,
            cache_events,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[KVCache],
        input_metadata: InputMetadata,
        cache_events: Optional[List[torch.cuda.Event]],
    ) -> SamplerOutput:
        assert cache_events is None, "cache_events not supported yet"

        if input_metadata.num_prompt_tokens > 0 or not input_metadata.use_cuda_graph:
            hidden_states = self.model(input_ids, positions, kv_caches,
                                       input_metadata, cache_events)
        else:
            hidden_states = self.compile_and_call_model(
                input_ids, positions, kv_caches, input_metadata, cache_events)
        next_tokens = self.sampler(self.lm_head.weight, hidden_states,
                                   input_metadata)
        return next_tokens

    _column_parallel_layers = []
    _row_parallel_layers = ["o_proj", "down_proj"]

    def load_weights(
        self,
        model_name_or_path: str,
        cache_dir: Optional[str] = None,
        load_format: str = "auto",
        revision: Optional[str] = None,
    ):
        if self.quant_config is None:
            weight_suffixes = ["weight"]
        else:
            weight_suffixes = self.quant_config.get_tp_tensor_names()

        column_parallel_weights: List[str] = []
        for layer in self._column_parallel_layers:
            for suffix in weight_suffixes:
                column_parallel_weights.append(f"{layer}.{suffix}")
        row_parallel_weights: List[str] = []
        for layer in self._row_parallel_layers:
            for suffix in weight_suffixes:
                row_parallel_weights.append(f"{layer}.{suffix}")

        tp_size = get_tensor_model_parallel_world_size()
        tensor_model_parallel_rank = get_tensor_model_parallel_rank()
        q_proj_shard_size = self.config.hidden_size // tp_size
        kv_proj_shard_size = (self.config.hidden_size //
                              self.config.num_attention_heads *
                              self.config.num_key_value_heads // tp_size)
        attention_weight_specs = [
            # (weight_name, shard_size, offset)
            ("q_proj", q_proj_shard_size, 0),
            ("k_proj", kv_proj_shard_size, q_proj_shard_size),
            ("v_proj", kv_proj_shard_size,
             q_proj_shard_size + kv_proj_shard_size),
        ]
        state_dict = self.state_dict()

        for name, loaded_weight in hf_model_weights_iterator(
                model_name_or_path, cache_dir, load_format, revision):
            if "rotary_emb.inv_freq" in name:
                continue

            is_packed = False
            is_transposed = False
            if self.quant_config is not None:
                is_packed = self.quant_config.is_packed(name)
                is_transposed = self.quant_config.is_transposed(name)
            if is_transposed:
                loaded_weight = convert_pyslice_to_tensor(loaded_weight)
                loaded_weight = loaded_weight.T

            is_attention_weight = False
            for weight_name, shard_size, offset in attention_weight_specs:
                if weight_name not in name:
                    continue
                param = state_dict[name.replace(weight_name, "qkv_proj")]
                if is_transposed:
                    param = param.T

                if is_packed:
                    shard_size //= self.quant_config.pack_factor
                    offset //= self.quant_config.pack_factor

                loaded_weight = loaded_weight[
                    shard_size * tensor_model_parallel_rank:shard_size *
                    (tensor_model_parallel_rank + 1)]
                param_slice = param.data[offset:offset + shard_size]
                assert param_slice.shape == loaded_weight.shape

                param_slice.copy_(loaded_weight)
                is_attention_weight = True
                break
            if is_attention_weight:
                continue

            is_gate_up_weight = False
            for stride_id, weight_name in enumerate(["gate_proj", "up_proj"]):
                if weight_name not in name:
                    continue
                param = state_dict[name.replace(weight_name, "gate_up_proj")]
                if is_transposed:
                    param = param.T

                shard_size = param.shape[0] // 2
                loaded_weight = loaded_weight[
                    shard_size * tensor_model_parallel_rank:shard_size *
                    (tensor_model_parallel_rank + 1)]
                param_slice = param.data[shard_size * stride_id:shard_size *
                                         (stride_id + 1)]
                assert param_slice.shape == loaded_weight.shape
                param_slice.copy_(loaded_weight)
                is_gate_up_weight = True
                break
            if is_gate_up_weight:
                continue

            param = state_dict[name]
            if is_transposed:
                param = param.T

            if "embed_tokens" in name or "lm_head" in name:
                load_padded_tensor_parallel_vocab(param, loaded_weight,
                                                  tensor_model_parallel_rank)
                continue

            load_tensor_parallel_weights(
                param,
                loaded_weight,
                name,
                column_parallel_weights,
                row_parallel_weights,
                tensor_model_parallel_rank,
            )
