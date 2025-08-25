# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Iterable
from typing import Optional

import torch
import torch.distributed as dist
from torch import nn
from transformers import GptOssConfig

from vllm.attention import Attention, AttentionType
from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import (get_ep_group, get_tensor_model_parallel_rank,
                              get_tensor_model_parallel_world_size)
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
from vllm.utils import cdiv

from .utils import (AutoWeightsLoader, WeightsMapper, extract_layer_index,
                    maybe_prefix)


class OAIAttention(nn.Module):

    def __init__(
        self,
        config: GptOssConfig,
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
            max_position=config.max_position_embeddings,
            base=config.rope_theta,
            dtype=torch.float32,
            rope_scaling={
                "rope_type":
                "yarn",
                "factor":
                config.rope_scaling["factor"],
                "original_max_position_embeddings":
                config.rope_scaling["original_max_position_embeddings"],
                "beta_fast":
                config.rope_scaling["beta_fast"],
                "beta_slow":
                config.rope_scaling["beta_slow"],
            },
            is_neox_style=True,
        )

        tp_size = get_tensor_model_parallel_world_size()

        self.sinks = torch.nn.Parameter(
            torch.empty(config.num_attention_heads // tp_size,
                        dtype=torch.bfloat16,
                        requires_grad=False))

        self.norm = RMSNorm(config.hidden_size, eps=1e-5)

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

        self.o_proj = RowParallelLinear(
            input_size=self.num_attention_heads * self.head_dim,
            output_size=self.hidden_size,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
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
        output, _ = self.o_proj(attn_output)

        return output + hidden_states


class MLPBlock(torch.nn.Module):

    def __init__(
        self,
        config: GptOssConfig,
        layer_idx: int,
        quant_config: QuantizationConfig,
        prefix: str = "",
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.num_experts = config.num_local_experts
        self.experts_per_token = config.num_experts_per_tok
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.norm = RMSNorm(config.hidden_size, eps=1e-5)
        self.router = torch.nn.Linear(config.hidden_size,
                                      config.num_local_experts,
                                      dtype=torch.bfloat16)
        assert config.intermediate_size % self.world_size == 0
        self.experts = FusedMoE(num_experts=config.num_local_experts,
                                top_k=config.num_experts_per_tok,
                                hidden_size=config.hidden_size,
                                intermediate_size=config.intermediate_size,
                                reduce_results=True,
                                renormalize=True,
                                quant_config=quant_config,
                                prefix=f"{prefix}.experts",
                                apply_router_weight_on_input=False,
                                has_bias=True,
                                activation="swigluoai")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = self.norm(x)
        g = self.router(t)
        t = self.experts(hidden_states=t, router_logits=g)
        return x + t


class TransformerBlock(torch.nn.Module):

    def __init__(
        self,
        config: GptOssConfig,
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


@support_torch_compile
class GptOssModel(nn.Module):

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
    ):
        super().__init__()
        self.config = vllm_config.model_config.hf_config
        self.quant_config = vllm_config.quant_config
        self.parallel_config = vllm_config.parallel_config
        self.config.hidden_size = self.config.hidden_size
        self.embedding = VocabParallelEmbedding(
            self.config.vocab_size,
            self.config.hidden_size,
        )
        self.layers = torch.nn.ModuleList([
            TransformerBlock(
                self.config,
                quant_config=self.quant_config,
                prefix=maybe_prefix(prefix, f"block.{layer_idx}"),
            ) for layer_idx in range(self.config.num_hidden_layers)
        ])
        self.norm = RMSNorm(self.config.hidden_size, eps=1e-5)

    def forward(self, input_ids: torch.Tensor,
                positions: torch.Tensor) -> torch.Tensor:
        x = self.embedding(input_ids)
        for layer in self.layers:
            x = layer(x, positions)
        x = self.norm(x)
        return x

    def _load_weights_mxfp4(
        self,
        ep_rank_end: int,
        ep_rank_start: int,
        heads_per_rank: int,
        head_start: int,
        weights: Iterable[tuple[str, torch.Tensor]],
        stacked_params_mapping: list[tuple[str, ...]],
    ) -> set[str]:
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()

        mxfp4_block = 32
        use_ep = self.parallel_config.enable_expert_parallel
        num_experts = self.config.num_local_experts

        tp_rank = get_tensor_model_parallel_rank()
        tp_size = get_tensor_model_parallel_world_size()

        intermediate_size = self.config.intermediate_size
        intermediate_size_block = intermediate_size // mxfp4_block
        per_rank_intermediate_size_block = cdiv(intermediate_size_block,
                                                tp_size)
        per_rank_intermediate_size = (per_rank_intermediate_size_block *
                                      mxfp4_block)

        # Calculate common slicing bounds for current rank
        tp_rank_start = tp_rank * per_rank_intermediate_size
        tp_rank_end = min((tp_rank + 1) * per_rank_intermediate_size,
                          intermediate_size)

        for name, weight in weights:
            # FIXME(woosuk): Remove this after testing.
            weight = weight.cuda()

            if ".w13_weight_scale" in name:
                # Handle MLP gate and up projection weights scale
                if use_ep:
                    narrow_weight = weight[ep_rank_start:ep_rank_end, ...]
                else:
                    narrow_weight = weight[:,
                                           2 * tp_rank_start:2 * tp_rank_end,
                                           ...]

                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param,
                              narrow_weight,
                              weight_name=name,
                              shard_id=None,
                              expert_id=None)
                loaded_params.add(name)
                continue
            elif ".w2_weight_scale" in name:
                # Handle MLP down projection weights
                if use_ep:
                    narrow_weight = weight[ep_rank_start:ep_rank_end, ...]
                else:
                    narrow_weight = weight[..., tp_rank_start //
                                           mxfp4_block:tp_rank_end //
                                           mxfp4_block]

                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param,
                              narrow_weight,
                              weight_name=name,
                              shard_id=None,
                              expert_id=None)
                loaded_params.add(name)
                continue
            elif ".w13_weight" in name:
                # Handle MLP gate and up projection weights
                # flat weight from (E, 2 * N, block_size, entry_per_block)
                # to (E, 2 * N, -1), shouldn't trigger copy for contiguous
                weight = weight.view(num_experts, 2 * intermediate_size,
                                     -1).contiguous()

                # Extract gate and up projection parts
                # since the weight is shuffled, we can slice directly
                if use_ep:
                    narrow_weight = weight[ep_rank_start:ep_rank_end, ...]
                else:
                    narrow_weight = weight[:,
                                           2 * tp_rank_start:2 * tp_rank_end,
                                           ...]

                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param,
                              narrow_weight,
                              weight_name=name,
                              shard_id=None,
                              expert_id=None)
                loaded_params.add(name)
                continue
            elif ".w2_weight" in name:
                # Handle MLP down projection weights
                # same flatten here, but since 2 mx4 value are packed in 1
                # uint8, divide by 2
                weight = weight.view(num_experts, -1,
                                     intermediate_size // 2).contiguous()
                if use_ep:
                    narrow_weight = weight[ep_rank_start:ep_rank_end, ...]
                else:
                    narrow_weight = weight[...,
                                           tp_rank_start // 2:tp_rank_end // 2]

                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param,
                              narrow_weight,
                              weight_name=name,
                              shard_id=None,
                              expert_id=None)
                loaded_params.add(name)
                continue
            elif ".w13_bias" in name:
                # Handle MLP gate and up projection biases
                # Extract gate and up projection bias parts
                if use_ep:
                    narrow_weight = weight[ep_rank_start:ep_rank_end, ...]
                else:
                    narrow_weight = weight[:,
                                           2 * tp_rank_start:2 * tp_rank_end]

                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param,
                              narrow_weight,
                              weight_name=name,
                              shard_id=None,
                              expert_id=None)
                loaded_params.add(name)
                continue
            elif ".w2_bias" in name:
                # Handle MLP down projection bias
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                if use_ep:
                    weight = weight[ep_rank_start:ep_rank_end, ...]
                else:
                    # (only load on rank 0 to avoid duplication)
                    if tp_rank != 0:
                        weight.zero_()
                weight_loader(param,
                              weight,
                              weight_name=name,
                              shard_id=None,
                              expert_id=None)
                loaded_params.add(name)
                continue
            elif "sinks" in name:
                # Handle attention sinks (distributed across ranks)
                param = params_dict[name]
                narrow_weight = weight.narrow(0, head_start, heads_per_rank)
                param.data.copy_(narrow_weight)
                loaded_params.add(name)
                continue
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                if weight_loader == default_weight_loader:
                    weight_loader(param, weight)
                else:
                    weight_loader(param, weight, shard_id)
                break
            else:
                # Handle all other weights with potential renaming
                if name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, weight)
            loaded_params.add(name)
        return loaded_params

    def _load_weights_other(
        self,
        ep_rank_start: int,
        ep_rank_end: int,
        heads_per_rank: int,
        head_start: int,
        weights: Iterable[tuple[str, torch.Tensor]],
        stacked_params_mapping: list[tuple[str, ...]],
    ) -> set[str]:
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()

        use_ep = self.parallel_config.enable_expert_parallel

        tp_rank = get_tensor_model_parallel_rank()
        tp_size = get_tensor_model_parallel_world_size()

        intermediate_size = self.config.intermediate_size
        per_rank_intermediate_size = cdiv(intermediate_size, tp_size)
        # Calculate common slicing bounds for current rank
        tp_rank_start = tp_rank * per_rank_intermediate_size
        tp_rank_end = min((tp_rank + 1) * per_rank_intermediate_size,
                          intermediate_size)

        for name, weight in weights:
            if ".w13_weight" in name:
                # Handle MLP gate and up projection weights
                # Extract gate and up projection parts
                if use_ep:
                    narrow_weight = weight[ep_rank_start:ep_rank_end, ...]
                else:
                    narrow_weight = weight[:, :,
                                           2 * tp_rank_start:2 * tp_rank_end]

                narrow_weight = narrow_weight.permute(0, 2, 1).contiguous()
                param = params_dict[name]

                param.copy_(narrow_weight)
                loaded_params.add(name)
                continue
            elif ".w2_weight" in name:
                # Handle MLP down projection weights
                if use_ep:
                    narrow_weight = weight[ep_rank_start:ep_rank_end, ...]
                else:
                    narrow_weight = weight[:, tp_rank_start:tp_rank_end, :]
                narrow_weight = narrow_weight.permute(0, 2, 1).contiguous()
                param = params_dict[name]

                param.copy_(narrow_weight)
                loaded_params.add(name)
                continue
            elif ".w13_bias" in name:
                # Handle MLP gate and up projection biases
                # Extract gate and up projection bias parts
                if use_ep:
                    narrow_weight = weight[ep_rank_start:ep_rank_end, ...]
                else:
                    narrow_weight = weight[:,
                                           2 * tp_rank_start:2 * tp_rank_end]

                param = params_dict[name]
                param.copy_(narrow_weight)
                loaded_params.add(name)
                continue
            elif ".w2_bias" in name:
                # Handle MLP down projection bias
                if use_ep:
                    weight = weight[ep_rank_start:ep_rank_end, ...]
                else:
                    # (only load on rank 0 to avoid duplication)
                    if tp_rank != 0:
                        weight.zero_()
                param = params_dict[name]
                param.copy_(weight)
                loaded_params.add(name)
                continue
            elif "sinks" in name:
                # Handle attention sinks (distributed across ranks)
                param = params_dict[name]
                narrow_weight = weight.narrow(0, head_start, heads_per_rank)
                param.data.copy_(narrow_weight)
                loaded_params.add(name)
                continue
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                if weight_loader == default_weight_loader:
                    weight_loader(param, weight)
                else:
                    weight_loader(param, weight, shard_id)
                break
            else:
                # Handle all other weights with potential renaming
                if name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, weight)
            loaded_params.add(name)
        return loaded_params

    def load_weights(self, weights: Iterable[tuple[str,
                                                   torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".qkv", ".q_proj", "q"),
            (".qkv", ".k_proj", "k"),
            (".qkv", ".v_proj", "v"),
        ]

        tp_rank = get_tensor_model_parallel_rank()
        tp_size = get_tensor_model_parallel_world_size()

        # Attention heads per rank
        heads_per_rank = self.config.num_attention_heads // tp_size
        head_start = tp_rank * heads_per_rank

        ep_size = get_ep_group().world_size
        ep_rank = get_ep_group().rank
        num_experts = self.config.num_local_experts
        experts_per_rank = num_experts // ep_size
        ep_rank_start = ep_rank * experts_per_rank
        ep_rank_end = (ep_rank + 1) * experts_per_rank

        quant_method = (self.config.quantization_config['quant_method'] if
                        hasattr(self.config, "quantization_config") else None)
        if quant_method == "mxfp4":
            return self._load_weights_mxfp4(ep_rank_end, ep_rank_start,
                                            heads_per_rank, head_start,
                                            weights, stacked_params_mapping)
        else:
            return self._load_weights_other(ep_rank_end, ep_rank_start,
                                            heads_per_rank, head_start,
                                            weights, stacked_params_mapping)


class GptOssForCausalLM(nn.Module):
    packed_modules_mapping = {"qkv": ["q_proj", "k_proj", "v_proj"]}

    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_substr={
            ".self_attn.": ".attn.",
            ".post_attention_layernorm.": ".mlp.norm.",
        },
        orig_to_new_suffix={
            ".embed_tokens.weight": ".embedding.weight",
            ".input_layernorm.weight": ".attn.norm.weight",
            ".post_attention_layernorm.weight": ".mlp.norm.weight",

            # MoE MXFP4 weights
            ".gate_up_proj_blocks": ".w13_weight",
            ".down_proj_blocks": ".w2_weight",
            ".gate_up_proj_scales": ".w13_weight_scale",
            ".down_proj_scales": ".w2_weight_scale",

            # MoE other weights
            ".gate_up_proj": ".w13_weight",
            ".down_proj": ".w2_weight",

            # MoE Bias
            ".gate_up_proj_bias": ".w13_bias",
            ".down_proj_bias": ".w2_bias",
        },
    )

    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str = "",
    ):
        super().__init__()
        self.vllm_config = vllm_config
        self.config = vllm_config.model_config.hf_config

        self.model = GptOssModel(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "model"),
        )
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
        assert intermediate_tensors is None
        assert inputs_embeds is None
        return self.model(input_ids, positions)

    def compute_logits(self, hidden_states: torch.Tensor,
                       sampling_metadata: SamplingMetadata) -> torch.Tensor:
        logits = self.logits_processor(self.lm_head, hidden_states,
                                       sampling_metadata)
        return logits

    def load_weights(self, weights: Iterable[tuple[str,
                                                   torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=(["lm_head."]
                           if self.config.tie_word_embeddings else None),
        )
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)
