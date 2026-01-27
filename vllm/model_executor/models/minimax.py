# coding=utf-8
# Adapted from
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/minimax/modeling_minimax.py
# Copyright 2025 The vLLM team.
# Copyright 2025 MiniMaxAI and HuggingFace Inc. team. All rights reserved.
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
"""Inference-only MiniMax model compatible with HuggingFace weights.

MiniMax uses a hybrid attention architecture that combines Lightning Attention
(linear attention) with standard Softmax Attention, along with Mixture of Experts (MoE).

Key features:
- Hybrid attention: 7 Lightning Attention layers + 1 Softmax Attention layer (repeating)
- MoE with top-k routing
- RoPE applied to half of the attention head dimension
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

KVCache = Tuple[torch.Tensor, torch.Tensor]


class MiniMaxMLP(nn.Module):
    """Standard MLP for non-MoE layers or as expert in MoE."""

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


class MiniMaxExpertMLP(nn.Module):
    """Single expert MLP for MoE layer."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
    ):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class MiniMaxSparseMoE(nn.Module):
    """Mixture of Experts layer with top-k routing."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        num_experts_per_tok: int,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.hidden_size = hidden_size

        # Router
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)

        # Experts
        self.experts = nn.ModuleList([
            MiniMaxExpertMLP(hidden_size, intermediate_size)
            for _ in range(num_experts)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, hidden_size = x.shape

        # Compute router logits and get top-k experts
        router_logits = self.gate(x)  # [batch_size, num_experts]
        routing_weights, selected_experts = torch.topk(
            router_logits, self.num_experts_per_tok, dim=-1
        )
        routing_weights = F.softmax(routing_weights, dim=-1)

        # Initialize output tensor
        final_hidden_states = torch.zeros_like(x)

        # Route tokens to experts
        expert_mask = F.one_hot(selected_experts, num_classes=self.num_experts)
        expert_mask = expert_mask.permute(2, 1, 0)  # [num_experts, top_k, batch_size]

        for expert_idx in range(self.num_experts):
            expert = self.experts[expert_idx]

            # Find tokens routed to this expert
            for top_k_idx in range(self.num_experts_per_tok):
                mask = expert_mask[expert_idx, top_k_idx]  # [batch_size]
                if mask.any():
                    token_indices = mask.nonzero(as_tuple=True)[0]
                    expert_input = x[token_indices]
                    expert_output = expert(expert_input)

                    # Weight by routing score
                    weight = routing_weights[token_indices, top_k_idx].unsqueeze(-1)
                    final_hidden_states[token_indices] += weight * expert_output

        return final_hidden_states


class MiniMaxPartialRotaryEmbedding(nn.Module):
    """Rotary Position Embedding applied to half of the head dimension."""

    def __init__(
        self,
        head_dim: int,
        max_position: int = 131072,
        base: float = 10000000.0,
    ):
        super().__init__()
        self.head_dim = head_dim
        self.rotary_dim = head_dim // 2  # RoPE applied to half of head dimension
        self.max_position = max_position
        self.base = base

        # Create the cos and sin cache
        inv_freq = 1.0 / (base ** (torch.arange(0, self.rotary_dim, 2).float() / self.rotary_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build cos/sin cache
        t = torch.arange(max_position).float()
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        cache = torch.cat((cos, sin), dim=-1)
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    def forward(self, positions: torch.Tensor, query: torch.Tensor, key: torch.Tensor):
        """Apply partial rotary embedding to query and key."""
        # Split into rotary and non-rotary parts
        q_rot = query[..., :self.rotary_dim]
        q_pass = query[..., self.rotary_dim:]
        k_rot = key[..., :self.rotary_dim]
        k_pass = key[..., self.rotary_dim:]

        # Get cos and sin for positions
        cos_sin = self.cos_sin_cache[positions]
        cos = cos_sin[..., :self.rotary_dim // 2]
        sin = cos_sin[..., self.rotary_dim // 2:]

        # Apply rotary embedding
        q_rot = self._apply_rotary(q_rot, cos, sin)
        k_rot = self._apply_rotary(k_rot, cos, sin)

        # Concatenate back
        query = torch.cat([q_rot, q_pass], dim=-1)
        key = torch.cat([k_rot, k_pass], dim=-1)

        return query, key

    def _apply_rotary(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        """Apply rotary embedding to tensor."""
        x1 = x[..., :x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2:]

        # Reshape cos/sin for broadcasting
        while cos.dim() < x1.dim():
            cos = cos.unsqueeze(1)
            sin = sin.unsqueeze(1)

        return torch.cat([
            x1 * cos - x2 * sin,
            x2 * cos + x1 * sin,
        ], dim=-1)


class MiniMaxSoftmaxAttention(nn.Module):
    """Standard Softmax Attention with partial RoPE for MiniMax."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        max_position: int = 131072,
        rope_theta: float = 10000000.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.head_dim = head_dim
        tp_size = get_tensor_model_parallel_world_size()

        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size

        self.total_num_kv_heads = num_kv_heads
        assert self.total_num_kv_heads % tp_size == 0
        self.num_kv_heads = self.total_num_kv_heads // tp_size

        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim ** -0.5

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

        # Use PagedAttention with RoPE
        # Note: MiniMax applies RoPE to half of head dimension, but we use full head for cache
        self.attn = PagedAttentionWithRoPE(
            self.num_heads,
            self.head_dim,
            self.scaling,
            rotary_dim=self.head_dim // 2,  # Partial RoPE
            max_position=max_position,
            base=int(rope_theta),
            num_kv_heads=self.num_kv_heads,
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


class MiniMaxLightningAttention(nn.Module):
    """Lightning Attention (Linear Attention) for MiniMax.

    Lightning Attention uses a linear attention mechanism that achieves O(n) complexity
    instead of O(n²) for standard softmax attention.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        max_position: int = 131072,
        rope_theta: float = 10000000.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.head_dim = head_dim
        tp_size = get_tensor_model_parallel_world_size()

        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size

        self.total_num_kv_heads = num_kv_heads
        assert self.total_num_kv_heads % tp_size == 0
        self.num_kv_heads = self.total_num_kv_heads // tp_size

        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim ** -0.5

        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

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

        # Partial rotary embedding
        self.rotary_emb = MiniMaxPartialRotaryEmbedding(
            head_dim=head_dim,
            max_position=max_position,
            base=rope_theta,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: KVCache,
        input_metadata: InputMetadata,
        cache_event: Optional[torch.cuda.Event],
    ) -> torch.Tensor:
        batch_size = hidden_states.shape[0]

        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        # Reshape for attention
        q = q.view(batch_size, self.num_heads, self.head_dim)
        k = k.view(batch_size, self.num_kv_heads, self.head_dim)
        v = v.view(batch_size, self.num_kv_heads, self.head_dim)

        # Apply partial rotary embedding
        q, k = self.rotary_emb(positions, q, k)

        # Expand KV heads if using GQA
        if self.num_kv_heads != self.num_heads:
            k = torch.repeat_interleave(k, self.num_queries_per_kv, dim=1)
            v = torch.repeat_interleave(v, self.num_queries_per_kv, dim=1)

        # Linear attention: use feature map (ELU + 1) for kernel approximation
        # φ(x) = elu(x) + 1
        q = F.elu(q) + 1
        k = F.elu(k) + 1

        # Linear attention: O = (Q @ K.T @ V) / (Q @ K.T @ 1)
        # Rewrite as: O = Q @ (K.T @ V) for O(n) complexity
        kv = torch.einsum('bnh,bnv->bhv', k, v)  # [batch, head_dim, head_dim]
        qkv = torch.einsum('bnh,bhv->bnv', q, kv)  # [batch, num_heads, head_dim]

        # Normalization
        k_sum = k.sum(dim=0, keepdim=True)  # [1, num_heads, head_dim]
        normalizer = torch.einsum('bnh,onh->bn', q, k_sum) + 1e-6  # [batch, num_heads]

        attn_output = qkv / normalizer.unsqueeze(-1)

        # Reshape output
        attn_output = attn_output.view(batch_size, -1)
        output, _ = self.o_proj(attn_output)
        return output


class MiniMaxDecoderLayer(nn.Module):
    """MiniMax decoder layer with hybrid attention (Lightning or Softmax) and MoE."""

    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx

        # Determine layer type based on config or pattern
        # Default pattern: 7 Lightning + 1 Softmax
        layer_types = getattr(config, 'layer_types', None)
        if layer_types is not None and layer_idx < len(layer_types):
            self.is_lightning_layer = layer_types[layer_idx] == 'lightning_attention'
        else:
            # Default: every 8th layer (0-indexed: 7, 15, 23, ...) is softmax attention
            self.is_lightning_layer = (layer_idx + 1) % 8 != 0

        # Choose attention type
        head_dim = getattr(config, 'head_dim', config.hidden_size // config.num_attention_heads)
        max_position = getattr(config, 'max_position_embeddings', 131072)
        rope_theta = getattr(config, 'rope_theta', 10000000.0)
        num_kv_heads = getattr(config, 'num_key_value_heads', config.num_attention_heads)

        if self.is_lightning_layer:
            self.self_attn = MiniMaxLightningAttention(
                hidden_size=self.hidden_size,
                num_heads=config.num_attention_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                max_position=max_position,
                rope_theta=rope_theta,
            )
        else:
            self.self_attn = MiniMaxSoftmaxAttention(
                hidden_size=self.hidden_size,
                num_heads=config.num_attention_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                max_position=max_position,
                rope_theta=rope_theta,
            )

        # Check if this layer uses MoE
        num_experts = getattr(config, 'num_local_experts', 1)
        if num_experts > 1:
            self.mlp = MiniMaxSparseMoE(
                hidden_size=self.hidden_size,
                intermediate_size=config.intermediate_size,
                num_experts=num_experts,
                num_experts_per_tok=getattr(config, 'num_experts_per_tok', 2),
            )
        else:
            self.mlp = MiniMaxMLP(
                hidden_size=self.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
            )

        self.input_layernorm = RMSNorm(config.hidden_size,
                                       eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size,
                                                eps=config.rms_norm_eps)

        # Alpha/beta factors for residual connections
        self.attn_alpha = getattr(config, 'linear_attn_alpha_factor' if self.is_lightning_layer
                                  else 'full_attn_alpha_factor', 1.0)
        self.attn_beta = getattr(config, 'linear_attn_beta_factor' if self.is_lightning_layer
                                 else 'full_attn_beta_factor', 1.0)
        self.mlp_alpha = getattr(config, 'mlp_alpha_factor', 1.0)
        self.mlp_beta = getattr(config, 'mlp_beta_factor', 1.0)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: KVCache,
        input_metadata: InputMetadata,
        cache_event: Optional[torch.cuda.Event],
    ) -> torch.Tensor:
        # Self Attention with alpha/beta residual
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            kv_cache=kv_cache,
            input_metadata=input_metadata,
            cache_event=cache_event,
        )
        hidden_states = self.attn_alpha * residual + self.attn_beta * hidden_states

        # MLP with alpha/beta residual
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.mlp_alpha * residual + self.mlp_beta * hidden_states

        return hidden_states


class MiniMaxModel(nn.Module):
    """MiniMax transformer model."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        vocab_size = ((config.vocab_size + 63) // 64) * 64
        self.embed_tokens = VocabParallelEmbedding(
            vocab_size, config.hidden_size, perform_initialization=False)
        self.layers = nn.ModuleList([
            MiniMaxDecoderLayer(config, layer_idx=i)
            for i in range(config.num_hidden_layers)
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
            cache_event = cache_events[i] if cache_events is not None else None
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

    def __init__(self, config):
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
        "gate_proj.weight", "up_proj.weight"
    ]
    _row_parallel_weights = ["o_proj.weight", "down_proj.weight"]

    def load_weights(self,
                     model_name_or_path: str,
                     cache_dir: Optional[str] = None,
                     use_np_cache: bool = False):
        tp_size = get_tensor_model_parallel_world_size()
        tensor_model_parallel_rank = get_tensor_model_parallel_rank()

        head_dim = getattr(self.config, 'head_dim',
                          self.config.hidden_size // self.config.num_attention_heads)
        num_kv_heads = getattr(self.config, 'num_key_value_heads',
                              self.config.num_attention_heads)

        q_proj_shard_size = (self.config.num_attention_heads * head_dim // tp_size)
        kv_proj_shard_size = (num_kv_heads * head_dim // tp_size)

        attention_weight_specs = [
            # (weight_name, shard_size, offset)
            ("q_proj", q_proj_shard_size, 0),
            ("k_proj", kv_proj_shard_size, q_proj_shard_size),
            ("v_proj", kv_proj_shard_size, q_proj_shard_size + kv_proj_shard_size),
        ]
        state_dict = self.state_dict()

        for name, loaded_weight in hf_model_weights_iterator(
                model_name_or_path, cache_dir, use_np_cache):
            if "rotary_emb.inv_freq" in name:
                continue

            # Handle embedding and lm_head weights
            if "embed_tokens" in name or "lm_head" in name:
                param = state_dict[name]
                # Consider padding in the vocab size.
                padded_vocab_size = (param.shape[0] * tp_size)
                num_extra_rows = padded_vocab_size - self.config.vocab_size
                extra_rows = torch.empty(num_extra_rows, loaded_weight.shape[1])
                extra_rows = extra_rows.to(loaded_weight)
                loaded_weight = torch.cat([loaded_weight, extra_rows], dim=0)

            # Handle QKV projection weights
            is_attention_weight = False
            for weight_name, shard_size, offset in attention_weight_specs:
                if weight_name not in name or "qkv_proj" in name:
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

            # Handle gate_up_proj weights (fused gate and up projections)
            is_gate_up_weight = False
            for stride_id, weight_name in enumerate(["gate_proj", "up_proj"]):
                if weight_name not in name or "gate_up_proj" in name:
                    continue
                # Skip MoE expert weights for now - they're handled separately
                if "experts" in name:
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

            # Handle MoE expert weights
            if "experts" in name:
                # MoE expert weights are not tensor parallel
                if name in state_dict:
                    param = state_dict[name]
                    param.copy_(loaded_weight)
                continue

            # Handle router/gate weights for MoE
            if "gate" in name and "gate_proj" not in name and "gate_up_proj" not in name:
                if name in state_dict:
                    param = state_dict[name]
                    param.copy_(loaded_weight)
                continue

            # Default loading with tensor parallel handling
            if name in state_dict:
                param = state_dict[name]
                load_tensor_parallel_weights(param, loaded_weight, name,
                                             self._column_parallel_weights,
                                             self._row_parallel_weights,
                                             tensor_model_parallel_rank)
