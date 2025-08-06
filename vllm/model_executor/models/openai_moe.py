# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Iterable
from typing import Optional

import torch
import torch.distributed as dist
from torch import nn

from vllm.attention import Attention, AttentionType
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (QKVParallelLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.configs import OpenAIMoeConfig

from .utils import extract_layer_index, maybe_prefix


class OAIAttention(nn.Module):

    def __init__(
        self,
        config: OpenAIMoeConfig,
        quant_config: Optional[QuantizationConfig] = None,
        cache_config: Optional[CacheConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.layer_idx = extract_layer_index(prefix)
        self.head_dim = config.head_dim
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.hidden_size = config.hidden_size

        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=config.initial_context_length *
            config.rope_scaling_factor,
            base=config.rope_theta,
            dtype=torch.float32,
            rope_scaling={
                "rope_type": "yarn",
                "factor": config.rope_scaling_factor,
                "original_max_position_embeddings":
                config.initial_context_length,
                "beta_fast": config.rope_ntk_beta,
                "beta_slow": config.rope_ntk_alpha,
            },
            is_neox_style=True,
        )

        tp_size = get_tensor_model_parallel_world_size()

        self.sinks = torch.nn.Parameter(
            torch.empty(config.num_attention_heads // tp_size,
                        dtype=torch.bfloat16))

        self.norm = RMSNorm(config.hidden_size)

        self.q_size = self.num_attention_heads * self.head_dim // tp_size
        self.kv_size = self.num_key_value_heads * self.head_dim // tp_size
        self.scaling = self.head_dim**-0.5
        self.rope_theta = config.rope_theta

        self.qkv = QKVParallelLinear(
            hidden_size=self.hidden_size,
            head_size=self.head_dim,
            total_num_heads=self.num_attention_heads,
            total_num_kv_heads=self.num_key_value_heads,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )

        self.out = RowParallelLinear(
            input_size=self.num_attention_heads * self.head_dim,
            output_size=self.hidden_size,
            quant_config=quant_config,
            prefix=f"{prefix}.out",
        )

        self.num_local_attention_heads = config.num_attention_heads // tp_size
        self.num_local_key_value_heads = config.num_key_value_heads // tp_size

        # Only apply sliding window to every other layer
        sliding_window = (config.sliding_window if self.layer_idx %
                          2 == 0 else None)
        self.attn = Attention(
            self.num_local_attention_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_local_key_value_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            per_layer_sliding_window=sliding_window,
            attn_type=AttentionType.DECODER,
            prefix=f"{prefix}.attn",
            sinks=self.sinks,
        )

    def forward(self, hidden_states: torch.Tensor,
                positions: torch.Tensor) -> torch.Tensor:
        t = self.norm(hidden_states)

        qkv, _ = self.qkv(t)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        v = v.contiguous()
        attn_output = self.attn(q, k, v)
        output, _ = self.out(attn_output)

        return output + hidden_states


class MLPBlock(torch.nn.Module):

    def __init__(
        self,
        config: OpenAIMoeConfig,
        layer_idx: int,
        quant_config: QuantizationConfig,
        prefix: str = "",
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.num_experts = config.num_experts
        self.experts_per_token = config.experts_per_token
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.norm = RMSNorm(config.hidden_size)
        self.gate = torch.nn.Linear(config.hidden_size,
                                    config.num_experts,
                                    dtype=torch.bfloat16)
        assert config.intermediate_size % self.world_size == 0
        self.experts = FusedMoE(num_experts=config.num_experts,
                                top_k=config.num_experts_per_token,
                                hidden_size=config.hidden_size,
                                intermediate_size=config.intermediate_size,
                                reduce_results=True,
                                renormalize=True,
                                quant_config=quant_config,
                                prefix=f"{prefix}.experts",
                                use_triton_kernels=True,
                                apply_router_weight_on_input=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = self.norm(x)
        g = self.gate(t)
        t = self.experts(hidden_states=t, router_logits=g)
        return x + t


class TransformerBlock(torch.nn.Module):

    def __init__(
        self,
        config: OpenAIMoeConfig,
        quant_config: QuantizationConfig,
        prefix: str = "",
    ):
        super().__init__()
        self.layer_idx = extract_layer_index(prefix)
        self.attn = OAIAttention(config, prefix=f"{prefix}.attn")
        self.mlp = MLPBlock(config,
                            self.layer_idx,
                            quant_config=quant_config,
                            prefix=f"{prefix}.mlp")

    def forward(self, hidden_states: torch.Tensor,
                positions: torch.Tensor) -> torch.Tensor:
        attn_output = self.attn(hidden_states, positions)
        output = self.mlp(attn_output)
        return output


class OpenAIMoeForCausalLM(nn.Module):

    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str = "",
    ):
        super().__init__()
        self.config = vllm_config.model_config.hf_config
        self.quant_config = vllm_config.quant_config
        self.embedding = VocabParallelEmbedding(
            self.config.vocab_size,
            self.config.hidden_size,
        )
        self.block = torch.nn.ModuleList([
            TransformerBlock(
                self.config,
                quant_config=self.quant_config,
                prefix=maybe_prefix(prefix, f"block.{layer_idx}"),
            ) for layer_idx in range(self.config.num_hidden_layers)
        ])
        self.norm = RMSNorm(self.config.hidden_size)
        self.lm_head = ParallelLMHead(
            self.config.vocab_size,
            self.config.hidden_size,
        )
        self.logits_processor = LogitsProcessor(self.config.vocab_size)

    def forward(self,
                input_ids: torch.Tensor,
                positions: torch.Tensor,
                intermediate_tensors: Optional[IntermediateTensors] = None,
                inputs_embeds: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.embedding(input_ids)
        for block in self.block:
            x = block(x, positions)
        x = self.norm(x)
        return x

    def compute_logits(self, hidden_states: torch.Tensor,
                       sampling_metadata: SamplingMetadata) -> torch.Tensor:
        logits = self.logits_processor(self.lm_head, hidden_states,
                                       sampling_metadata)
        return logits

    def load_weights(self, weights: Iterable[tuple[str,
                                                   torch.Tensor]]) -> set[str]:
        rename_mapping = {
            "norm.scale": "norm.weight",
            "unembedding.weight": "lm_head.weight",
        }

        def maybe_rename(name: str) -> str:
            for remap_name, new_name in rename_mapping.items():
                if remap_name in name:
                    return name.replace(remap_name, new_name)
            return name

        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()

        my_rank = dist.get_rank() if dist.is_initialized() else 0
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        intermediate_size = self.config.intermediate_size
        per_rank_intermediate_size = intermediate_size // world_size

        # Calculate common slicing bounds for current rank
        rank_start = my_rank * per_rank_intermediate_size
        rank_end = (my_rank + 1) * per_rank_intermediate_size

        # Attention heads per rank
        heads_per_rank = self.config.num_attention_heads // world_size
        head_start = my_rank * heads_per_rank

        for name, weight in weights:
            # MoE kernels needs these on GPU for shuffle, quant, and swizzling
            weight = weight.cuda()

            if "mlp1_weight" in name:
                # Handle MLP gate and up projection weights
                new_name = name.replace("mlp1_weight", "experts.w13_weight")

                # Extract gate and up projection parts
                gate_part = weight[:, rank_start:rank_end, ...]
                up_part = weight[:, rank_start + intermediate_size:rank_end +
                                 intermediate_size, ...]
                narrow_weight = torch.cat((gate_part, up_part), dim=1)

                param = params_dict[new_name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param,
                              narrow_weight,
                              weight_name=new_name,
                              shard_id=None,
                              expert_id=None)
                loaded_params.add(new_name)

            elif "mlp2_weight" in name:
                # Handle MLP down projection weights
                new_name = name.replace("mlp2_weight", "experts.w2_weight")
                narrow_weight = weight[..., rank_start:rank_end]

                param = params_dict[new_name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param,
                              narrow_weight,
                              weight_name=new_name,
                              shard_id=None,
                              expert_id=None)
                loaded_params.add(new_name)

            elif "mlp1_bias" in name:
                # Handle MLP gate and up projection biases
                new_name = name.replace("mlp1_bias", "experts.w13_bias")

                # Extract gate and up projection bias parts
                gate_bias = weight[:, rank_start:rank_end]
                up_bias = weight[:, rank_start + intermediate_size:rank_end +
                                 intermediate_size]
                narrow_weight = torch.cat((gate_bias, up_bias), dim=1)

                param = params_dict[new_name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param,
                              narrow_weight,
                              weight_name=new_name,
                              shard_id=None,
                              expert_id=None)
                loaded_params.add(new_name)

            elif "mlp2_bias" in name:
                # Handle MLP down projection bias
                # (only load on rank 0 to avoid duplication)
                if dist.get_rank() != 0:
                    weight.zero_()

                new_name = name.replace("mlp2_bias", "experts.w2_bias")
                param = params_dict[new_name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param,
                              weight,
                              weight_name=new_name,
                              shard_id=None,
                              expert_id=None)
                loaded_params.add(new_name)
            elif "sinks" in name:
                # Handle attention sinks (distributed across ranks)
                param = params_dict[name]
                narrow_weight = weight.narrow(0, head_start, heads_per_rank)
                param.data.copy_(narrow_weight)
                loaded_params.add(name)
            else:
                # Handle all other weights with potential renaming
                renamed_name = maybe_rename(name)
                param = params_dict[renamed_name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, weight)
                loaded_params.add(renamed_name)

        return loaded_params
