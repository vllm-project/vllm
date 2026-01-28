# coding=utf-8
# Adapted from vLLM's LLaMA implementation
# Copyright 2025 MiniMax and The vLLM team.
# Copyright 2023 The vLLM team.
# Copyright 2022 EleutherAI and the HuggingFace Inc. team.
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
"""Inference-only MiniMax-M2 model compatible with HuggingFace weights.

MiniMax-M2 is a Mixture-of-Experts (MoE) model with 230B total parameters
and 10B active parameters, designed for coding and agentic tasks.
"""
from typing import Dict, List, Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F

from vllm.model_executor.input_metadata import InputMetadata
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.attention import PagedAttentionWithRoPE
from vllm.model_executor.layers.sampler import Sampler
from vllm.model_executor.weight_utils import (hf_model_weights_iterator,
                                              load_tensor_parallel_weights)
from vllm.model_executor.parallel_utils.parallel_state import (
    get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size)
from vllm.model_executor.parallel_utils.tensor_parallel import (
    VocabParallelEmbedding, ColumnParallelLinear, RowParallelLinear)
from vllm.sequence import SequenceOutputs
from vllm.transformers_utils.configs.minimax import MiniMaxConfig

KVCache = Tuple[torch.Tensor, torch.Tensor]


class MiniMaxMoE(nn.Module):
    """Mixture of Experts layer for MiniMax model.

    This implements a standard MoE layer with top-k expert routing.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        num_experts_per_tok: int,
        hidden_act: str = "silu",
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok

        # Gate for routing tokens to experts
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)

        # Create expert networks
        # Each expert has gate_proj, up_proj, and down_proj
        self.experts_gate_proj = nn.ModuleList([
            nn.Linear(hidden_size, intermediate_size, bias=False)
            for _ in range(num_experts)
        ])
        self.experts_up_proj = nn.ModuleList([
            nn.Linear(hidden_size, intermediate_size, bias=False)
            for _ in range(num_experts)
        ])
        self.experts_down_proj = nn.ModuleList([
            nn.Linear(intermediate_size, hidden_size, bias=False)
            for _ in range(num_experts)
        ])

        if hidden_act != "silu":
            raise ValueError(f"Unsupported activation: {hidden_act}. "
                             "Only silu is supported for now.")

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, hidden_dim = hidden_states.shape
        hidden_states_flat = hidden_states.view(-1, hidden_dim)

        # Compute routing weights
        router_logits = self.gate(hidden_states_flat)
        routing_weights = F.softmax(router_logits, dim=-1)

        # Select top-k experts
        topk_weights, topk_indices = torch.topk(
            routing_weights, self.num_experts_per_tok, dim=-1
        )
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

        # Initialize output
        final_output = torch.zeros_like(hidden_states_flat)

        # Process each expert
        for expert_idx in range(self.num_experts):
            # Find tokens routed to this expert
            expert_mask = (topk_indices == expert_idx).any(dim=-1)
            if not expert_mask.any():
                continue

            # Get the tokens for this expert
            expert_tokens = hidden_states_flat[expert_mask]

            # Get the weight for this expert for each token
            expert_weights = torch.zeros(expert_mask.sum(), device=hidden_states.device)
            for k in range(self.num_experts_per_tok):
                mask_k = topk_indices[expert_mask, k] == expert_idx
                expert_weights[mask_k] = topk_weights[expert_mask, k][mask_k]

            # Apply expert
            gate_out = self.experts_gate_proj[expert_idx](expert_tokens)
            up_out = self.experts_up_proj[expert_idx](expert_tokens)
            expert_out = F.silu(gate_out) * up_out
            expert_out = self.experts_down_proj[expert_idx](expert_out)

            # Weight and accumulate
            final_output[expert_mask] += expert_weights.unsqueeze(-1) * expert_out

        return final_output.view(batch_size, seq_len, hidden_dim)


class MiniMaxMLP(nn.Module):
    """Standard MLP layer (used for dense layers in MiniMax)."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
    ):
        super().__init__()
        self.gate_up_proj = ColumnParallelLinear(hidden_size,
                                                 2 * intermediate_size,
                                                 bias=False,
                                                 gather_output=False,
                                                 perform_initialization=False)
        self.down_proj = RowParallelLinear(intermediate_size,
                                           hidden_size,
                                           bias=False,
                                           input_is_parallel=True,
                                           perform_initialization=False)
        if hidden_act != "silu":
            raise ValueError(f"Unsupported activation: {hidden_act}. "
                             "Only silu is supported for now.")
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class MiniMaxAttention(nn.Module):
    """Multi-head attention with RoPE for MiniMax model."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        max_position_embeddings: int,
        rope_theta: float,
        rotary_dim: int,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            assert self.total_num_kv_heads % tp_size == 0
            self.num_kv_heads = self.total_num_kv_heads // tp_size
        else:
            # Handle case where num_kv_heads < tp_size
            assert tp_size % self.total_num_kv_heads == 0
            self.num_kv_heads = 1

        self.head_dim = head_dim
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5

        self.qkv_proj = ColumnParallelLinear(
            hidden_size,
            (self.total_num_heads + 2 * self.total_num_kv_heads) * self.head_dim,
            bias=False,
            gather_output=False,
            perform_initialization=False,
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            input_is_parallel=True,
            perform_initialization=False,
        )
        self.attn = PagedAttentionWithRoPE(
            self.num_heads,
            self.head_dim,
            self.scaling,
            rotary_dim=rotary_dim,
            max_position=max_position_embeddings,
            base=int(rope_theta),
            num_kv_heads=self.num_kv_heads
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


class MiniMaxDecoderLayer(nn.Module):
    """Decoder layer for MiniMax model with optional MoE."""

    def __init__(self, config: MiniMaxConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx

        # Self attention
        self.self_attn = MiniMaxAttention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            rope_theta=config.rope_theta,
            rotary_dim=config.rotary_dim,
        )

        # Determine if this layer uses MoE or dense MLP
        use_moe = (
            layer_idx >= config.first_k_dense_replace and
            (layer_idx - config.first_k_dense_replace) % config.moe_layer_freq == 0
        )

        if use_moe and config.num_experts > 1:
            self.mlp = MiniMaxMoE(
                hidden_size=self.hidden_size,
                intermediate_size=config.expert_intermediate_size,
                num_experts=config.num_experts,
                num_experts_per_tok=config.num_experts_per_tok,
                hidden_act=config.hidden_act,
            )
        else:
            self.mlp = MiniMaxMLP(
                hidden_size=self.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
            )

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
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

        # MLP/MoE
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class MiniMaxModel(nn.Module):
    """MiniMax transformer model."""

    def __init__(self, config: MiniMaxConfig):
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id if hasattr(config, 'pad_token_id') else None
        self.vocab_size = config.vocab_size

        vocab_size = ((config.vocab_size + 63) // 64) * 64
        self.embed_tokens = VocabParallelEmbedding(
            vocab_size, config.hidden_size, perform_initialization=False)
        self.layers = nn.ModuleList([
            MiniMaxDecoderLayer(config, layer_idx)
            for layer_idx in range(config.num_hidden_layers)
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


class MiniMaxForCausalLM(nn.Module):
    """MiniMax model for causal language modeling."""

    def __init__(self, config: MiniMaxConfig):
        super().__init__()
        self.config = config
        self.model = MiniMaxModel(config)
        vocab_size = ((config.vocab_size + 63) // 64) * 64
        self.lm_head = ColumnParallelLinear(config.hidden_size,
                                            vocab_size,
                                            bias=False,
                                            gather_output=False,
                                            perform_initialization=False)
        self.sampler = Sampler(config.vocab_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[KVCache],
        input_metadata: InputMetadata,
        cache_events: Optional[List[torch.cuda.Event]],
    ) -> Dict[int, SequenceOutputs]:
        hidden_states = self.model(input_ids, positions, kv_caches,
                                   input_metadata, cache_events)
        next_tokens = self.sampler(self.lm_head.weight, hidden_states,
                                   input_metadata)
        return next_tokens

    _column_parallel_weights = [
        "embed_tokens.weight", "lm_head.weight", "qkv_proj.weight",
        "gate_proj.weight", "up_proj.weight", "gate_up_proj.weight",
    ]
    _row_parallel_weights = ["o_proj.weight", "down_proj.weight"]

    def load_weights(self,
                     model_name_or_path: str,
                     cache_dir: Optional[str] = None,
                     use_np_cache: bool = False):
        tp_size = get_tensor_model_parallel_world_size()
        tensor_model_parallel_rank = get_tensor_model_parallel_rank()

        # Calculate shard sizes for attention
        head_dim = self.config.head_dim
        num_heads = self.config.num_attention_heads
        num_kv_heads = self.config.num_key_value_heads

        q_proj_shard_size = (num_heads * head_dim) // tp_size
        kv_proj_shard_size = (num_kv_heads * head_dim) // tp_size

        attention_weight_specs = [
            ("q_proj", q_proj_shard_size, 0),
            ("k_proj", kv_proj_shard_size, q_proj_shard_size),
            ("v_proj", kv_proj_shard_size, q_proj_shard_size + kv_proj_shard_size),
        ]

        state_dict = self.state_dict()

        for name, loaded_weight in hf_model_weights_iterator(
                model_name_or_path, cache_dir, use_np_cache):
            if "rotary_emb.inv_freq" in name:
                continue

            # Handle embedding and lm_head
            if "embed_tokens" in name or "lm_head" in name:
                param = state_dict[name]
                padded_vocab_size = (param.shape[0] * tp_size)
                num_extra_rows = padded_vocab_size - self.config.vocab_size
                extra_rows = torch.empty(num_extra_rows, loaded_weight.shape[1])
                extra_rows = extra_rows.to(loaded_weight)
                loaded_weight = torch.cat([loaded_weight, extra_rows], dim=0)

            # Handle MoE expert weights
            if "experts" in name:
                # MoE weights are not sharded in this basic implementation
                # They are loaded directly
                if name in state_dict:
                    param = state_dict[name]
                    param.data.copy_(loaded_weight)
                continue

            # Handle attention QKV projection weights
            is_attention_weight = False
            for weight_name, shard_size, offset in attention_weight_specs:
                if weight_name not in name or "experts" in name:
                    continue
                param = state_dict[name.replace(weight_name, "qkv_proj")]

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

            # Handle gate_up_proj (fused gate and up projection)
            is_gate_up_weight = False
            for stride_id, weight_name in enumerate(["gate_proj", "up_proj"]):
                if weight_name not in name or "experts" in name:
                    continue
                param = state_dict[name.replace(weight_name, "gate_up_proj")]
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

            # Handle gate (router) weights
            if "gate.weight" in name:
                if name in state_dict:
                    param = state_dict[name]
                    param.data.copy_(loaded_weight)
                continue

            # Load other weights
            if name in state_dict:
                param = state_dict[name]
                load_tensor_parallel_weights(param, loaded_weight, name,
                                             self._column_parallel_weights,
                                             self._row_parallel_weights,
                                             tensor_model_parallel_rank)
