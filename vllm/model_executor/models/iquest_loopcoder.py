# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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
"""Inference-only LoopCoder model compatible with HuggingFace weights."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import replace
from typing import Any

import torch
from torch import nn
from transformers import PretrainedConfig

from vllm.attention.layer import Attention
from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
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
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)
from vllm.model_executor.models.llama import LlamaMLP
from vllm.sequence import IntermediateTensors
from vllm.v1.attention.backend import AttentionType

from .utils import (
    AutoWeightsLoader,
    extract_layer_index,
    make_layers,
    maybe_prefix,
)


class LoopCoderAttention(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position: int = 4096 * 32,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        attn_type: str = AttentionType.DECODER,
        dual_chunk_attention_config: dict[str, Any] | None = None,
        layer_idx: int = 0,
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx
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
        self.dual_chunk_attention_config = dual_chunk_attention_config

        # Get loop_num from config, default to 2 if not specified
        self.loop_num = getattr(config, "loop_num", 2)

        self.loop_window_size = getattr(config, "loop_window_size", 64)

        # Use total number of hidden layers instead of hardcoded 24
        total_layers = config.num_hidden_layers

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        self.rotary_emb = get_rope(
            self.head_dim,
            max_position=max_position,
            rope_parameters=config.rope_parameters,
            dual_chunk_attention_config=dual_chunk_attention_config,
        )
        self.attn = nn.ModuleList()

        base_cache_config = cache_config

        for loop_idx in range(self.loop_num):
            base_layer_idx = extract_layer_index(prefix)
            unique_layer_idx = loop_idx * total_layers + base_layer_idx

            unique_prefix = prefix.replace(
                f"layers.{base_layer_idx}", f"layers.{unique_layer_idx}"
            )

            if loop_idx == 0:
                loop_cache_config = cache_config
            else:
                if base_cache_config is not None:
                    loop_cache_config = replace(
                        base_cache_config,
                        sliding_window=self.loop_window_size,
                    )
                else:
                    loop_cache_config = CacheConfig(
                        sliding_window=self.loop_window_size,
                        cache_dtype="auto",
                    )

            self.attn.append(
                Attention(
                    self.num_heads,
                    self.head_dim,
                    self.scaling,
                    num_kv_heads=self.num_kv_heads,
                    cache_config=loop_cache_config,
                    quant_config=quant_config,
                    attn_type=attn_type,
                    prefix=f"{unique_prefix}.attn",
                    **{
                        "layer_idx": unique_layer_idx,
                        "dual_chunk_attention_config": dual_chunk_attention_config,
                    }
                    if dual_chunk_attention_config and loop_idx == 0
                    else {},
                )
            )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        loop_idx: int,
        gate_proj: LoopGateProjection | None = None,
    ) -> torch.Tensor:
        if loop_idx == 0:
            attn = self.attn[0]
            qkv, _ = self.qkv_proj(hidden_states)
            q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
            q, k = self.rotary_emb(positions, q, k)
            attn_output = attn(q, k, v)
            output, _ = self.o_proj(attn_output)
            return output
        else:
            global_attn = self.attn[0]
            local_attn = self.attn[loop_idx]
            qkv, _ = self.qkv_proj(hidden_states)
            q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
            q, k = self.rotary_emb(positions, q, k)
            num_tokens, _ = q.shape
            num_heads = self.num_heads
            head_dim = self.head_dim

            q_reshaped = q.view(num_tokens, num_heads, head_dim).transpose(0, 1)

            global_attn_output = global_attn(q, None, None)
            local_attn_output = local_attn(q, k, v)
            assert gate_proj is not None, "gate_proj must be provided for loop_idx > 0"
            gate = gate_proj(q_reshaped)
            output = global_attn_output * gate + local_attn_output * (1 - gate)
            output, _ = self.o_proj(output)
            return output


class LoopCoderDecoderLayer(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        layer_idx: int = 0,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        dual_chunk_attention_config = getattr(
            config, "dual_chunk_attention_config", None
        )
        self.layer_idx = layer_idx
        if getattr(config, "is_causal", True):
            attn_type = AttentionType.DECODER
        else:
            attn_type = AttentionType.ENCODER_ONLY

        self.self_attn = LoopCoderAttention(
            config=config,
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            max_position=config.max_position_embeddings,
            num_kv_heads=config.num_key_value_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
            attn_type=attn_type,
            dual_chunk_attention_config=dual_chunk_attention_config,
            layer_idx=self.layer_idx,
        )
        self.mlp = LlamaMLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        loop_idx: int,
        gate_proj: LoopGateProjection | None = None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            loop_idx=loop_idx,
            gate_proj=gate_proj,
        )
        hidden_states = hidden_states + residual
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = hidden_states + residual

        return hidden_states


class LoopGateProjection(nn.Module):
    """Gate projection for mixed attention in Loop 2+.

    Computes: g = sigmoid(linear(Q)) for each head independently.
    This gate determines how much to use Loop1's KV (global) vs current
    loop's KV (local).

    Supports tensor parallelism: each GPU handles a subset of heads.
    The weight matrix has shape [num_heads, head_dim] and is split along
    the head dimension.
    """

    def __init__(
        self,
        total_num_heads: int,
        head_dim: int,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.total_num_heads = total_num_heads
        self.head_dim = head_dim
        tp_size = get_tensor_model_parallel_world_size()
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size

        self.gate_proj = ColumnParallelLinear(
            head_dim,
            self.total_num_heads,
            bias=True,
            gather_output=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_proj",
        )

    def forward(self, query: torch.Tensor) -> torch.Tensor:
        """Compute gate values from query tensor.

        Args:
            query: [num_heads, num_tokens, head_dim] (vLLM flattened format)
                where num_heads is the number of heads on this TP rank
                and num_tokens = batch * seq_len

        Returns:
            gate: [num_tokens, num_heads * head_dim] (flattened format matching q shape)
        """
        num_heads, num_tokens, head_dim = query.shape

        assert num_heads == self.num_heads, (
            f"Expected {self.num_heads} heads, got {num_heads}"
        )

        query_flat = query.reshape(-1, head_dim)

        gate_logits_flat, _ = self.gate_proj(query_flat)

        gate_logits = gate_logits_flat.reshape(
            num_heads, num_tokens, self.num_heads
        )  # [num_heads, num_tokens, num_heads]

        # Extract diagonal: each head h's query should use output column h
        # gate_logits[h, :, h] gives the output for head h at each token
        gate_logits = torch.diagonal(
            gate_logits, dim1=0, dim2=2
        )  # [num_tokens, num_heads]
        gate_logits = gate_logits.transpose(0, 1)  # [num_heads, num_tokens]
        gate_logits = gate_logits.unsqueeze(-1)  # [num_heads, num_tokens, 1]

        # Apply sigmoid
        gate = torch.sigmoid(gate_logits)  # [num_heads, num_tokens, 1]

        # Expand and reshape to match q shape: [num_tokens, num_heads * head_dim]
        gate = gate.transpose(0, 1)  # [num_tokens, num_heads, 1]
        gate = gate.expand(-1, -1, head_dim)  # [num_tokens, num_heads, head_dim]
        gate = gate.reshape(
            num_tokens, num_heads * head_dim
        )  # [num_tokens, num_heads * head_dim]

        return gate


@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        "positions": -1,
        "intermediate_tensors": 0,
        "inputs_embeds": 0,
    }
)
class IQuestLoopCoderModel(nn.Module):
    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
        decoder_layer_type: type[nn.Module] = LoopCoderDecoderLayer,
    ):
        super().__init__()

        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config

        # TODO (@robertgshaw2): see if this can be moved out
        if cache_config.sliding_window is not None and hasattr(
            config, "max_window_layers"
        ):
            assert config.max_window_layers == config.num_hidden_layers, (
                "Sliding window for some but all layers is not supported. "
                "This model uses sliding window but `max_window_layers` = {} "
                "is less than `num_hidden_layers` = {}. Please open an issue "
                "to discuss this feature.".format(
                    config.max_window_layers,
                    config.num_hidden_layers,
                )
            )

        self.config = config
        self.quant_config = quant_config
        self.vocab_size = config.vocab_size

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=f"{prefix}.embed_tokens",
        )

        self.loop_num = getattr(self.config, "loop_num", 2)
        self.window_size = getattr(self.config, "loop_window_size", 64)

        # Gate projections for Loop 2+ (one per layer)
        head_dim = config.hidden_size // config.num_attention_heads
        _, _, self.gate_projections = make_layers(
            config.num_hidden_layers,
            lambda prefix: LoopGateProjection(
                total_num_heads=config.num_attention_heads,
                head_dim=head_dim,
                quant_config=quant_config,
                prefix=prefix,
            ),
            prefix=f"{prefix}.gate_projections",
        )

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: LoopCoderDecoderLayer(
                config=config,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=prefix,
                layer_idx=extract_layer_index(prefix),
            ),
            prefix=f"{prefix}.layers",
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            hidden_states = self.embed_input_ids(input_ids)

        for loop_idx in range(self.loop_num):
            for layer_idx, layer in enumerate(
                self.layers[self.start_layer : self.end_layer]
            ):
                # Get the actual layer index (accounting for pipeline parallelism)
                actual_layer_idx = self.start_layer + layer_idx
                # Get gate_proj for this layer (only for loop_idx > 0)
                gate_proj = (
                    self.gate_projections[actual_layer_idx] if loop_idx > 0 else None
                )
                hidden_states = layer(positions, hidden_states, loop_idx, gate_proj)
        hidden_states = self.norm(hidden_states)
        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        loaded_params: set[str] = set()
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            if self.quant_config is not None and (
                scale_name := self.quant_config.get_cache_scale(name)
            ):
                # Loading kv cache quantization scales
                param = params_dict[scale_name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                loaded_weight = (
                    loaded_weight if loaded_weight.dim() == 0 else loaded_weight[0]
                )
                weight_loader(param, loaded_weight)
                loaded_params.add(scale_name)
                continue
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if "gate_projections" in name:
                    continue
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if name.endswith("scale"):
                    # Remapping the name of FP8 kv-scale.
                    name = maybe_remap_kv_scale_name(name, params_dict)
                    if name is None:
                        continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                if weight_loader == default_weight_loader:
                    weight_loader(param, loaded_weight)
                else:
                    weight_loader(param, loaded_weight, shard_id)
                break
            else:
                if name.startswith("gate_projections."):
                    if name.endswith(".weight"):
                        vllm_name = name.replace(".weight", ".gate_proj.weight")
                    elif name.endswith(".bias"):
                        vllm_name = name.replace(".bias", ".gate_proj.bias")
                    else:
                        continue

                    if vllm_name in params_dict:
                        param = params_dict[vllm_name]
                        weight_loader = getattr(
                            param, "weight_loader", default_weight_loader
                        )
                        weight_loader(param, loaded_weight)
                        loaded_params.add(vllm_name)
                        continue
                    continue
                if name.endswith(".bias") and name not in params_dict:
                    continue
                # Remapping the name of FP8 kv-scale.
                name = maybe_remap_kv_scale_name(name, params_dict)
                if name is None:
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params


class IQuestLoopCoderForCausalLM(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config

        self.config = config

        self.quant_config = quant_config
        self.model = IQuestLoopCoderModel(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )

        if config.tie_word_embeddings:
            self.lm_head = self.model.embed_tokens
        else:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "lm_head"),
            )

        self.logits_processor = LogitsProcessor(config.vocab_size)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        hidden_states = self.model(
            input_ids, positions, intermediate_tensors, inputs_embeds
        )
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        logits = self.logits_processor(self.lm_head, hidden_states)
        return logits

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=(["lm_head."] if self.config.tie_word_embeddings else None),
        )
        return loader.load_weights(weights)
