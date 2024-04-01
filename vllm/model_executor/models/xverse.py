# coding=utf-8
# Adapted from
<<<<<<<< HEAD:vllm/model_executor/models/orion.py
# https://huggingface.co/OrionStarAI/Orion-14B-Base/blob/main/modeling_orion.py
# Copyright (c) OrionStar Inc.
# LICENSE: https://huggingface.co/OrionStarAI/Orion-14B-Base/blob/main/LICENSE
"""Inference-only Orion-14B model compatible with HuggingFace weights."""
========
# https://huggingface.co/xverse/XVERSE-7B/blob/main/modeling_xverse.py
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
"""Inference-only Xverse model compatible with HuggingFace weights."""
>>>>>>>> upstream-sync-2024-04-01:vllm/model_executor/models/xverse.py
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import nn
from transformers import PretrainedConfig

from vllm.attention import Attention, AttentionMetadata
from vllm.config import LoRAConfig
from vllm.model_executor.layers.activation import SiluAndMul
<<<<<<<< HEAD:vllm/model_executor/models/orion.py
from vllm.model_executor.layers.attention import Attention
========
from vllm.model_executor.layers.layernorm import RMSNorm
>>>>>>>> upstream-sync-2024-04-01:vllm/model_executor/models/xverse.py
from vllm.model_executor.layers.linear import (LinearMethodBase,
                                               MergedColumnParallelLinear,
                                               QKVParallelLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.sampler import Sampler
from vllm.model_executor.layers.vocab_parallel_embedding import (
<<<<<<<< HEAD:vllm/model_executor/models/orion.py
    VocabParallelEmbedding, ParallelLMHead)
========
    ParallelLMHead, VocabParallelEmbedding)
>>>>>>>> upstream-sync-2024-04-01:vllm/model_executor/models/xverse.py
from vllm.model_executor.parallel_utils.parallel_state import (
    get_tensor_model_parallel_world_size)
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.model_executor.weight_utils import (default_weight_loader,
                                              hf_model_weights_iterator)
from vllm.sequence import SamplerOutput
<<<<<<<< HEAD:vllm/model_executor/models/orion.py

KVCache = Tuple[torch.Tensor, torch.Tensor]


class OrionMLP(nn.Module):
========


class XverseMLP(nn.Module):
>>>>>>>> upstream-sync-2024-04-01:vllm/model_executor/models/xverse.py

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        linear_method: Optional[LinearMethodBase] = None,
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size, [intermediate_size] * 2,
            bias=False,
            linear_method=linear_method)
        self.down_proj = RowParallelLinear(intermediate_size,
                                           hidden_size,
                                           bias=False,
                                           linear_method=linear_method)
        if hidden_act != "silu":
            raise ValueError(f"Unsupported activation: {hidden_act}. "
                             "Only silu is supported for now.")
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate, _ = self.gate_up_proj(x)
        x = self.act_fn(gate)
        x, _ = self.down_proj(x)
        return x


<<<<<<<< HEAD:vllm/model_executor/models/orion.py
class OrionAttention(nn.Module):
========
class XverseAttention(nn.Module):
>>>>>>>> upstream-sync-2024-04-01:vllm/model_executor/models/xverse.py

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        rope_theta: float = 10000,
        rope_scaling: Optional[Dict[str, Any]] = None,
        max_position_embeddings: int = 8192,
        linear_method: Optional[LinearMethodBase] = None,
<<<<<<<< HEAD:vllm/model_executor/models/orion.py
========
        bias: bool = False,
        sliding_window: Optional[int] = None,
>>>>>>>> upstream-sync-2024-04-01:vllm/model_executor/models/xverse.py
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        # partition the KV heads across multiple tensor parallel GPUs.
        assert self.total_num_kv_heads % tp_size == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=bias,
            linear_method=linear_method,
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=bias,
            linear_method=linear_method,
        )

        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
            rope_scaling=rope_scaling,
        )
        self.attn = Attention(self.num_heads,
                              self.head_dim,
                              self.scaling,
<<<<<<<< HEAD:vllm/model_executor/models/orion.py
                              num_kv_heads=self.num_kv_heads)
========
                              num_kv_heads=self.num_kv_heads,
                              sliding_window=sliding_window)
>>>>>>>> upstream-sync-2024-04-01:vllm/model_executor/models/xverse.py

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v, kv_cache, attn_metadata)
        output, _ = self.o_proj(attn_output)
        return output


<<<<<<<< HEAD:vllm/model_executor/models/orion.py
class OrionDecoderLayer(nn.Module):
========
class XverseDecoderLayer(nn.Module):
>>>>>>>> upstream-sync-2024-04-01:vllm/model_executor/models/xverse.py

    def __init__(
        self,
        config: PretrainedConfig,
        linear_method: Optional[LinearMethodBase] = None,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        rope_theta = getattr(config, "rope_theta", 10000)
        rope_scaling = getattr(config, "rope_scaling", None)
        max_position_embeddings = getattr(config, "max_position_embeddings",
                                          8192)
<<<<<<<< HEAD:vllm/model_executor/models/orion.py
        self.self_attn = OrionAttention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
========
        sliding_window = getattr(config, "sliding_window", None)
        self.self_attn = XverseAttention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=getattr(config, "num_key_value_heads",
                                 config.num_attention_heads),
>>>>>>>> upstream-sync-2024-04-01:vllm/model_executor/models/xverse.py
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            max_position_embeddings=max_position_embeddings,
            linear_method=linear_method,
<<<<<<<< HEAD:vllm/model_executor/models/orion.py
        )
        self.mlp = OrionMLP(
========
            bias=getattr(config, "bias", False),
            sliding_window=sliding_window,
        )
        self.mlp = XverseMLP(
>>>>>>>> upstream-sync-2024-04-01:vllm/model_executor/models/xverse.py
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            linear_method=linear_method,
        )

        self.input_layernorm = nn.LayerNorm(config.hidden_size,
                                            eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.LayerNorm(config.hidden_size,
                                                     eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
        residual: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Self Attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            kv_cache=kv_cache,
            attn_metadata=attn_metadata,
        )

        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states, None


<<<<<<<< HEAD:vllm/model_executor/models/orion.py
class OrionModel(nn.Module):
========
class XverseModel(nn.Module):
>>>>>>>> upstream-sync-2024-04-01:vllm/model_executor/models/xverse.py

    def __init__(
        self,
        config: PretrainedConfig,
        linear_method: Optional[LinearMethodBase] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
<<<<<<<< HEAD:vllm/model_executor/models/orion.py
        self.vocab_size = config.vocab_size
========
        lora_vocab = (lora_config.lora_extra_vocab_size *
                      (lora_config.max_loras or 1)) if lora_config else 0
        self.vocab_size = config.vocab_size + lora_vocab
        self.org_vocab_size = config.vocab_size
>>>>>>>> upstream-sync-2024-04-01:vllm/model_executor/models/xverse.py
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
        )
        self.layers = nn.ModuleList([
<<<<<<<< HEAD:vllm/model_executor/models/orion.py
            OrionDecoderLayer(config, linear_method)
========
            XverseDecoderLayer(config, linear_method)
>>>>>>>> upstream-sync-2024-04-01:vllm/model_executor/models/xverse.py
            for _ in range(config.num_hidden_layers)
        ])
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        for i in range(len(self.layers)):
            layer = self.layers[i]
            hidden_states, residual = layer(
                positions,
                hidden_states,
                kv_caches[i],
                attn_metadata,
                residual,
            )
        hidden_states = self.norm(hidden_states)
        return hidden_states


<<<<<<<< HEAD:vllm/model_executor/models/orion.py
class OrionForCausalLM(nn.Module):
========
class XverseForCausalLM(nn.Module):
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

    # LoRA specific attributes
    supported_lora_modules = [
        "qkv_proj",
        "o_proj",
        "gate_up_proj",
        "down_proj",
        "embed_tokens",
        "lm_head",
    ]
    embedding_modules = {
        "embed_tokens": "input_embeddings",
        "lm_head": "output_embeddings",
    }
    embedding_padding_modules = ["lm_head"]
>>>>>>>> upstream-sync-2024-04-01:vllm/model_executor/models/xverse.py

    def __init__(
        self,
        config: PretrainedConfig,
        linear_method: Optional[LinearMethodBase] = None,
<<<<<<<< HEAD:vllm/model_executor/models/orion.py
========
        lora_config=None,
>>>>>>>> upstream-sync-2024-04-01:vllm/model_executor/models/xverse.py
    ) -> None:
        super().__init__()
        self.config = config
        self.linear_method = linear_method
<<<<<<<< HEAD:vllm/model_executor/models/orion.py
        self.model = OrionModel(config, linear_method)
========
        self.model = XverseModel(config, linear_method)
>>>>>>>> upstream-sync-2024-04-01:vllm/model_executor/models/xverse.py
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.sampler = Sampler()

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        hidden_states = self.model(input_ids, positions, kv_caches,
                                   attn_metadata)
        return hidden_states

    def compute_logits(self, hidden_states: torch.Tensor,
                       sampling_metadata: SamplingMetadata) -> torch.Tensor:
        logits = self.logits_processor(self.lm_head.weight, hidden_states,
                                       sampling_metadata)
        return logits

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        next_tokens = self.sampler(logits, sampling_metadata)
        return next_tokens

    def load_weights(self,
                     model_name_or_path: str,
                     cache_dir: Optional[str] = None,
                     load_format: str = "auto",
                     revision: Optional[str] = None):
        stacked_params_mapping = [
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        params_dict = dict(self.named_parameters())
        for name, loaded_weight in hf_model_weights_iterator(
                model_name_or_path, cache_dir, load_format, revision):
            if ("rotary_emb.inv_freq" in name
                    or "rotary_emb.cos_cached" in name
                    or "rotary_emb.sin_cached" in name):
                continue
            if ("rotary_emb.cos_cached" in name
                    or "rotary_emb.sin_cached" in name):
                # Models trained using ColossalAI may include these tensors in
                # the checkpoint. Skip them.
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
