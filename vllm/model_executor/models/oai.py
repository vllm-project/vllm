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
# from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (QKVParallelLinear,
                                               ReplicatedLinear,
                                               MergedColumnParallelLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
# from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.configs import OAIModelConfig

from .utils import extract_layer_index, maybe_prefix
import math


class RMSNorm(torch.nn.Module):
    """Using the reference implementation until we can verify accuracy."""

    def __init__(self, num_features: int, eps: float = 1e-05):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.scale = torch.nn.Parameter(
            torch.ones(num_features, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[-1] == self.num_features
        t, dtype = x.float(), x.dtype
        t = t * torch.rsqrt(torch.mean(t**2, dim=-1, keepdim=True) + self.eps)
        return (t * self.scale).to(dtype)


def _apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    cos = cos.unsqueeze(-2).to(x.dtype)
    sin = sin.unsqueeze(-2).to(x.dtype)
    x1, x2 = torch.chunk(x, 2, dim=-1)
    o1 = x1 * cos - x2 * sin
    o2 = x2 * cos + x1 * sin
    return torch.cat((o1, o2), dim=-1)


class RotaryEmbedding(torch.nn.Module):
    """Using the reference implementation until we can verify accuracy."""

    def __init__(
        self,
        head_dim: int,
        base: int,
        dtype: torch.dtype,
        initial_context_length: int = 4096,
        scaling_factor: float = 1.0,
        ntk_alpha: float = 1.0,
        ntk_beta: float = 32.0,
    ) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.base = base
        self.dtype = dtype
        self.initial_context_length = initial_context_length
        self.scaling_factor = scaling_factor
        self.ntk_alpha = ntk_alpha
        self.ntk_beta = ntk_beta
        self.device = torch.device("cuda")

        cache = self._compute_cos_sin_cache()
        cache = cache.to(dtype)
        self.cos_sin_cache: torch.Tensor
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    def _compute_cos_sin(self, num_tokens: int):
        concentration, inv_freq = self._compute_concentration_and_inv_freq()
        t = torch.arange(num_tokens, dtype=torch.float32, device=self.device)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        cos = freqs.cos() * concentration
        sin = freqs.sin() * concentration
        return cos, sin

    def _compute_cos_sin_cache(self):
        cos, sin = self._compute_cos_sin(self.initial_context_length)
        cache = torch.cat((cos, sin), dim=-1)
        return cache

    def _compute_concentration_and_inv_freq(self) -> torch.Tensor:
        """See YaRN paper: https://arxiv.org/abs/2309.00071"""
        freq = self.base**(torch.arange(
            0, self.head_dim, 2, dtype=torch.float, device=self.device) /
                           self.head_dim)
        if self.scaling_factor > 1.0:
            concentration = (0.1 * math.log(self.scaling_factor) + 1.0
                             )  # YaRN concentration

            d_half = self.head_dim / 2
            # NTK by parts
            low = (d_half * math.log(self.initial_context_length /
                                     (self.ntk_beta * 2 * math.pi)) /
                   math.log(self.base))
            high = (d_half * math.log(self.initial_context_length /
                                      (self.ntk_alpha * 2 * math.pi)) /
                    math.log(self.base))
            assert 0 < low < high < d_half - 1

            interpolation = 1.0 / (self.scaling_factor * freq)
            extrapolation = 1.0 / freq

            ramp = (
                torch.arange(d_half, dtype=torch.float32, device=freq.device) -
                low) / (high - low)
            mask = 1 - ramp.clamp(0, 1)

            inv_freq = interpolation * (1 - mask) + extrapolation * mask
        else:
            concentration = 1.0
            inv_freq = 1.0 / freq

        return concentration, inv_freq

    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_tokens = query.shape[0]

        # The only change from official impl is that we use the cache here.
        # cos, sin = self._compute_cos_sin(num_tokens)
        positions = positions.flatten()
        cos_sin = self.cos_sin_cache.index_select(0, positions)
        cos, sin = cos_sin.chunk(2, dim=-1)

        query_shape = query.shape
        query = query.view(num_tokens, -1, self.head_dim)
        query = _apply_rotary_emb(query, cos, sin)
        query = query.reshape(query_shape)

        key_shape = key.shape
        key = key.view(num_tokens, -1, self.head_dim)
        key = _apply_rotary_emb(key, cos, sin)
        key = key.reshape(key_shape)
        return query, key


class OAIAttention(nn.Module):

    def __init__(
        self,
        config: OAIModelConfig,
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

        # Note that the reference rope didn't set this right.
        max_position_pretrain = 4096  # 131 072 / 32
        # self.rotary_emb = get_rope(
        #     self.head_dim,
        #     rotary_dim=self.head_dim,
        #     max_position=max_position_pretrain,
        #     base=config.rope_theta,
        #     dtype=torch.float32,
        #     rope_scaling={
        #         "rope_type": "yarn",
        #         "factor": config.rope_scaling_factor,
        #         "original_max_position_embeddings": max_position_pretrain,
        #         "beta_fast": config.rope_ntk_beta,
        #         "beta_slow": config.rope_ntk_alpha,
        #     },
        #     is_neox_style=True,
        # )

        self.rotary_emb = RotaryEmbedding(
            head_dim=self.head_dim,
            base=config.rope_theta,
            dtype=torch.float32,
            initial_context_length=config.initial_context_length,
            scaling_factor=config.rope_scaling_factor,
            ntk_alpha=config.rope_ntk_alpha,
            ntk_beta=config.rope_ntk_beta,
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
        sliding_window = config.sliding_window if self.layer_idx % 2 == 0 else None
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

        # TODO: remove this for final release.
        # N = hidden_states.shape[0]
        # if N == 4 and self.layer_idx == 0 and dist.is_initialized(
        # ) and dist.get_rank() == 0:
        #     with open("vllm_attn_output.pkl", "wb") as f:
        #         torch.save(
        #             {
        #                 "hidden_states": hidden_states,
        #                 "t": t,
        #                 "q": q,
        #                 "k": k,
        #                 "v": v,
        #                 "S": self.sinks,
        #                 "attn_output": attn_output,
        #                 "output": output,
        #                 "final_output": hidden_states + output,
        #             }, f)

        return output + hidden_states


def swiglu(x, alpha: float = 1.702):
    # Note we add an extra bias of 1 to the linear layer
    x_glu, x_linear = torch.chunk(x, 2, dim=-1)
    out_glu = x_glu * torch.sigmoid(alpha * x_glu)
    return out_glu * (x_linear + 1)


class MLPBlock(torch.nn.Module):

    def __init__(
        self,
        config: OAIModelConfig,
        layer_idx: int,
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
        self.mlp1_weight = torch.nn.Parameter(
            torch.empty(
                (
                    config.num_experts,
                    config.intermediate_size * 2 // self.world_size,
                    config.hidden_size,
                ),
                dtype=torch.bfloat16,
            ))
        self.mlp1_bias = torch.nn.Parameter(
            torch.empty(
                (config.num_experts,
                 config.intermediate_size * 2 // self.world_size),
                dtype=torch.bfloat16,
            ))
        self.mlp2_weight = torch.nn.Parameter(
            torch.empty(
                (
                    config.num_experts,
                    config.hidden_size,
                    config.intermediate_size // self.world_size,
                ),
                dtype=torch.bfloat16,
            ))
        self.mlp2_bias = torch.nn.Parameter(
            torch.empty(
                (config.num_experts, config.hidden_size),
                dtype=torch.bfloat16,
            ))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = self.norm(x)
        post_norm_t = t.clone()
        g = self.gate(t)
        experts = torch.topk(g, k=self.experts_per_token, dim=-1, sorted=True)
        expert_weights = torch.nn.functional.softmax(experts.values, dim=1)
        expert_indices = experts.indices

        # if dist.is_initialized() and dist.get_rank() == 0:
        #     print(
        #         f"layer {self.layer_idx} expert_indices: {expert_indices}, expert_weights: {expert_weights}"
        #     )

        # MLP #1
        mlp1_weight = self.mlp1_weight[expert_indices, ...]
        mlp1_bias = self.mlp1_bias[expert_indices, ...]
        t = torch.einsum("beck,bk->bec", mlp1_weight, t) + mlp1_bias
        t = swiglu(t)

        # MLP #2
        mlp2_weight = self.mlp2_weight[expert_indices, ...]
        mlp2_bias = self.mlp2_bias[expert_indices, ...]
        t = torch.einsum("beck,bek->bec", mlp2_weight, t)
        if self.world_size > 1:
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
        t += mlp2_bias

        # Weighted sum of experts
        t = torch.einsum("bec,be->bc", t, expert_weights)

        # if dist.is_initialized() and dist.get_rank(
        # ) == 0 and self.layer_idx == 0:
        #     with open("vllm_mlp_pre_dispatch.pkl", "wb") as f:
        #         torch.save(
        #             {
        #                 "x_flat": x,
        #                 "x_norm": post_norm_t,
        #                 "router_logits": g,
        #                 "expert_indices": expert_indices,
        #                 "expert_weights": expert_weights,
        #                 "out_flat": t,
        #             }, f)

        return x + t


class TransformerBlock(torch.nn.Module):

    def __init__(
        self,
        config: OAIModelConfig,
        prefix: str = "",
    ):
        super().__init__()
        self.layer_idx = extract_layer_index(prefix)
        self.attn = OAIAttention(config, prefix=f"{prefix}.attn")
        self.mlp = MLPBlock(config, self.layer_idx, prefix=f"{prefix}.mlp")

    def forward(self, hidden_states: torch.Tensor,
                positions: torch.Tensor) -> torch.Tensor:
        attn_output = self.attn(hidden_states, positions)
        output = self.mlp(attn_output)
        # if dist.is_initialized() and dist.get_rank() == 0:
        #     print(
        #         f"Layer {self.layer_idx} attention sum: {attn_output.sum()}, norm: {attn_output.norm()}"
        #         f"Layer {self.layer_idx} mlp sum: {output.sum()}, norm: {output.norm()}"
        #     )
        return output


class OAIForCausalLM(nn.Module):

    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str = "",
    ):
        super().__init__()
        self.config = vllm_config.model_config.hf_config
        self.embedding = VocabParallelEmbedding(
            self.config.vocab_size,
            self.config.hidden_size,
        )
        self.block = torch.nn.ModuleList([
            TransformerBlock(
                self.config,
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
        # if dist.is_initialized() and dist.get_rank() == 0:
        #     print(f"Input tokens: {input_ids}")
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
            # This is needed if we are using vLLM's RMSNorm
            # "norm.scale": "norm.weight",
            "unembedding.weight": "lm_head.weight",
        }

        def maybe_rename(name: str) -> str:
            for remap_name, new_name in rename_mapping.items():
                if remap_name in name:
                    return name.replace(remap_name, new_name)
            return name

        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()

        using_oai_mlp = True
        my_rank = dist.get_rank() if dist.is_initialized() else 0
        per_rank_intermediate_size = self.config.intermediate_size // dist.get_world_size(
        )

        for name, weight in weights:

            # Useful for debugging
            # if name.startswith("block."):
            #     layer_idx = extract_layer_index(name)
            #     if layer_idx >= self.config.num_hidden_layers:
            #         continue

            weight = weight.cuda()  # make the narrowing to TP faster

            if "mlp1_weight" in name:
                for i in range(self.config.num_experts):
                    if using_oai_mlp:
                        narrow_weight = torch.cat((
                            weight[:, my_rank *
                                   per_rank_intermediate_size:(my_rank + 1) *
                                   per_rank_intermediate_size, ...],
                            weight[:, my_rank * per_rank_intermediate_size +
                                   self.config.intermediate_size:
                                   (my_rank + 1) * per_rank_intermediate_size +
                                   self.config.intermediate_size, ...],
                        ),
                                                  dim=1)
                        param = params_dict[name]
                        param.data.copy_(narrow_weight)
                        loaded_params.add(name)
                    else:
                        new_name = name.replace("mlp1_weight",
                                                f"experts.{i}.mlp1.weight")
                        param = params_dict[new_name]
                        param.weight_loader(param, weight[i])
                        loaded_params.add(new_name)
            elif "mlp2_weight" in name:
                if using_oai_mlp:
                    narrow_weight = weight[
                        ...,
                        my_rank * per_rank_intermediate_size:(my_rank + 1) *
                        per_rank_intermediate_size]
                    param = params_dict[name]
                    param.data.copy_(narrow_weight)
                    loaded_params.add(name)
                else:
                    for i in range(self.config.num_experts):
                        new_name = name.replace("mlp2_weight",
                                                f"experts.{i}.mlp2.weight")
                        param = params_dict[new_name]
                        param.weight_loader(param, weight[i])
                        loaded_params.add(new_name)
            elif "mlp1_bias" in name:
                if using_oai_mlp:
                    narrow_weight = torch.cat((
                        weight[:, my_rank *
                               per_rank_intermediate_size:(my_rank + 1) *
                               per_rank_intermediate_size],
                        weight[:, my_rank * per_rank_intermediate_size +
                               self.config.intermediate_size:(my_rank + 1) *
                               per_rank_intermediate_size +
                               self.config.intermediate_size],
                    ),
                                              dim=1)
                    param = params_dict[name]
                    param.data.copy_(narrow_weight)
                    loaded_params.add(name)
                else:
                    for i in range(self.config.num_experts):
                        new_name = name.replace("mlp1_bias",
                                                f"experts.{i}.mlp1.bias")
                        param = params_dict[new_name]
                        param.weight_loader(param, weight[i])
                        loaded_params.add(new_name)
            elif "mlp2_bias" in name:
                if using_oai_mlp:
                    param = params_dict[name]
                    param.data.copy_(weight)
                    loaded_params.add(name)
                else:
                    for i in range(self.config.num_experts):
                        new_name = name.replace("mlp2_bias",
                                                f"experts.{i}.mlp2.bias")
                    param = params_dict[new_name]
                    param.weight_loader(param, weight[i])
                    loaded_params.add(new_name)
            elif "sinks" in name:
                param = params_dict[name]
                param.data.copy_(
                    weight.narrow(
                        0, my_rank * self.config.num_attention_heads //
                        dist.get_world_size(),
                        self.config.num_attention_heads //
                        dist.get_world_size()))
                loaded_params.add(name)
            else:
                name = maybe_rename(name)
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, weight)
            loaded_params.add(name)

            # TODO: remove this for final release.
            weight.cpu()
        torch.cuda.empty_cache()
        return loaded_params
