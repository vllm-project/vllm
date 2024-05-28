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
"""Inference-only Skywork MoE model."""
from typing import Iterable, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn, Tensor

from transformers.configuration_utils import PretrainedConfig
from vllm import _custom_ops as ops
from vllm.attention import Attention, AttentionMetadata
from vllm.config import CacheConfig, LoRAConfig
from vllm.distributed import (get_tensor_model_parallel_rank,
                              get_tensor_model_parallel_world_size,
                              tensor_model_parallel_all_reduce)
from vllm.model_executor.layers.fused_moe import fused_moe, fused_experts
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (QKVParallelLinear,
                                               ReplicatedLinear,
                                               RowParallelLinear,
                                               LinearMethodBase)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)
from vllm.model_executor.layers.quantization.fp8 import Fp8Config
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.sampler import Sampler
from vllm.model_executor.layers.vocab_parallel_embedding import (
    DEFAULT_VOCAB_PADDING_SIZE, ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.model_executor.utils import set_weight_attrs
from vllm.sequence import SamplerOutput
from vllm.utils import print_warning_once

TOPK = 2

class SwiGLU(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, input: Tensor) -> Tensor:
        c1, c2 = torch.chunk(input, 2, dim=-1)
        return F.silu(c1) * c2


class IdentityMoE(nn.Module):
    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        self.moe = module

    def forward(self, input: Tensor) -> Tensor:
        return self.moe(input)


class SkyworkMoeConfig(PretrainedConfig):
    model_type: str = 'skywork'
    keys_to_ignore_at_inference: List[str] = ['past_key_values']

    def __init__(
            self,
            vocab_size: int = 32000,
            hidden_size: int = 4096,
            intermediate_size: int = 11008,
            num_hidden_layers: int = 32,
            num_attention_heads: int = 32,
            num_key_value_heads: Optional[int] = None,
            hidden_act: str = 'silu',
            max_position_embeddings: int = 2048,
            initializer_range: float = 0.02,
            rms_norm_eps: float = 1e-6,
            use_cache: bool = True,
            pad_token_id: Optional[int] = None,
            bos_token_id: int = 1,
            eos_token_id: int = 2,
            pretraining_tp: int = 1,
            tie_word_embeddings: bool = False,
            rope_theta: Union[int, float] = 10000.0,
            rope_scaling: Optional[float] = None,
            num_experts: List[int] = [32],
            moe_expert_interval: int = 1,
            moe_use_mixtral_gating: bool = False,
            moe_2layer_gate: bool = True,
            moe_use_logits_norm: bool = False,
            moe_gate_norm_std: float = 1.0,
            moe_feature_no_mul_topk: bool = False,

            **kwargs,
    ) -> None:
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads

        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.pretraining_tp = pretraining_tp
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self._rope_scaling_validation()
        self.num_experts = num_experts
        self.moe_expert_interval = moe_expert_interval
        self.moe_use_mixtral_gating = moe_use_mixtral_gating
        self.moe_2layer_gate = moe_2layer_gate
        self.moe_use_logits_norm = moe_use_logits_norm
        self.moe_gate_norm_std = moe_gate_norm_std
        self.moe_feature_no_mul_topk = moe_feature_no_mul_topk

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    def _rope_scaling_validation(self):
        """
        Validate the `rope_scaling` configuration.
        """
        if self.rope_scaling is None:
            return

        if not isinstance(self.rope_scaling, dict) or len(self.rope_scaling) != 2:
            raise ValueError(
                "`rope_scaling` must be a dictionary with with two fields, `type` and `factor`, "
                f"got {self.rope_scaling}"
            )
        rope_scaling_type = self.rope_scaling.get("type", None)
        rope_scaling_factor = self.rope_scaling.get("factor", None)
        if rope_scaling_type is None or rope_scaling_type not in ["linear", "dynamic", "ntk"]:
            raise ValueError(
                f"`rope_scaling`'s type field must be one of ['linear', 'dynamic'], got {rope_scaling_type}"
            )
        if rope_scaling_factor is None or not isinstance(rope_scaling_factor, float) or rope_scaling_factor <= 1.0:
            raise ValueError(f"`rope_scaling`'s factor field must be an float > 1, got {rope_scaling_factor}")

class SkyworkMLP(nn.Module):

    def __init__(
            self,
            num_experts: int,
            hidden_size: int,
            intermediate_size: int,
            linear_method: Optional[LinearMethodBase] = None,
    ) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.ffn_dim = intermediate_size
        self.hidden_dim = hidden_size

        self.gate_proj = ReplicatedLinear(self.hidden_dim,
                                          self.ffn_dim,
                                          bias=False,
                                          linear_method=linear_method, )
        self.up_proj = ReplicatedLinear(self.hidden_dim,
                                        self.ffn_dim,
                                        bias=False,
                                        linear_method=linear_method, )
        self.down_proj = ReplicatedLinear(self.ffn_dim,
                                          self.hidden_dim,
                                          bias=False,
                                          linear_method=linear_method, )

        # TODO: Use vllm's SiluAndMul
        self.act_fn = nn.SiLU()

    def forward(self, hidden_states: Tensor) -> Tensor:
        gate_out, _ = self.gate_proj(hidden_states)
        gate_out = self.act_fn(gate_out)

        up_out, _ = self.up_proj(hidden_states)
        current_hidden_states = gate_out * up_out
        current_hidden_states, _ = self.down_proj(current_hidden_states)
        return current_hidden_states


class SkyworkRouter(nn.Module):
    def __init__(self,
                 hidden_size: int,
                 num_experts: int,
                 moe_2layer_gate: bool = True,
                 linear_method: Optional[LinearMethodBase] = None,
                 params_dtype: Optional[torch.dtype] = None,
                 quant_config: Optional[QuantizationConfig] = None,
                 ) -> None:
        super().__init__()
        self.hidden_dim = hidden_size
        self.num_experts = num_experts
        if moe_2layer_gate:
            middle_dim = self.num_experts * 8
            self.wg = nn.ModuleList(
                [ReplicatedLinear(self.hidden_dim,
                                  middle_dim,
                                  bias=False,
                                  linear_method=linear_method, 
                                  params_dtype=params_dtype,
                                  quant_config=quant_config),
                 nn.Tanh(),
                 ReplicatedLinear(middle_dim,
                                  self.num_experts,
                                  bias=False,
                                  linear_method=linear_method, 
                                  params_dtype=params_dtype,
                                  quant_config=quant_config)])
        else:
            self.wg = ReplicatedLinear(self.hidden_dim,
                                       self.num_experts,
                                       bias=False,
                                       linear_method=linear_method, 
                                       params_dtype=params_dtype,
                                       quant_config=quant_config)

    def forward(self, hidden_states: Tensor) -> Tensor:
        if isinstance(self.wg, nn.ModuleList):
            for m in self.wg:
                if isinstance(m, ReplicatedLinear):
                    hidden_states, _ = m(hidden_states)
                else:
                    hidden_states = m(hidden_states)
        elif isinstance(self.wg, ReplicatedLinear):
            hidden_states, _ = self.wg(hidden_states)

        return hidden_states

class SkyworkMoE(nn.Module):
    """A tensor-parallel MoE implementation for Mixtral that shards each expert
    across all ranks.

    Each expert's weights are sharded across all ranks and a fused MoE
    kernel is used for the forward pass, and finally we reduce the outputs
    across ranks.
    """

    def __init__(
        self,
        num_experts: int,
        top_k: int,
        moe_2layer_gate: bool,
        hidden_size: int,
        intermediate_size: int,
        params_dtype: Optional[torch.dtype] = None,
        tp_size: Optional[int] = None,
        linear_method: Optional[LinearMethodBase] = None,
        config: Optional[SkyworkMoeConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()
        self.tp_size = tp_size or get_tensor_model_parallel_world_size()
        self.num_total_experts = num_experts
        self.top_k = top_k
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size // self.tp_size
        self.config = config
        self.quant_config = quant_config

        # FIXME(pcmoritz): Make this more general to support different
        # quantization schemes
        self.use_fp8 = isinstance(quant_config, Fp8Config)

        if params_dtype is None:
            params_dtype = torch.get_default_dtype()
        self.params_dtype = params_dtype

        # Gate always runs at half / full precision for now.
        self.gate = SkyworkRouter(
            config.hidden_size,
            self.num_total_experts,
            moe_2layer_gate=moe_2layer_gate,
            linear_method=linear_method,
            params_dtype=self.params_dtype,
            quant_config=None
        )

        if self.use_fp8 and self.quant_config.is_checkpoint_fp8_serialized:
            params_dtype = torch.float8_e4m3fn

        self.w_up_proj_weight = nn.Parameter(
            torch.empty(self.num_total_experts,
                        2 * self.intermediate_size,
                        self.hidden_size,
                        dtype=params_dtype))
        self.w_down_proj_weight = nn.Parameter(
            torch.empty(self.num_total_experts,
                        self.hidden_size,
                        self.intermediate_size,
                        dtype=params_dtype))

        set_weight_attrs(self.w_up_proj_weight, {
            "weight_loader": self.weight_loader,
        })
        set_weight_attrs(self.w_down_proj_weight, {
            "weight_loader": self.weight_loader,
        })

        # Used for fp8.
        self.w_up_proj_scale = None
        self.w_down_proj_scale = None
        self.a_up_proj_scale = None
        self.a_down_proj_scale = None

        if self.use_fp8:
            # WEIGHT_SCALE (for fp8)
            self.w_up_proj_scale = nn.Parameter(torch.ones(self.num_total_experts,
                                                     dtype=torch.float32),
                                          requires_grad=False)
            self.w_down_proj_scale = nn.Parameter(torch.ones(self.num_total_experts,
                                                    dtype=torch.float32),
                                         requires_grad=False)

            # If loading fp8 checkpoint, pass the weight loaders.
            # If loading an fp16 checkpoint, do not (we will quantize in
            #   process_weights_after_loading()
            if quant_config.is_checkpoint_fp8_serialized:
                set_weight_attrs(self.w_up_proj_scale, {
                    "weight_loader": self.weight_loader,
                })
                set_weight_attrs(self.w_down_proj_scale, {
                    "weight_loader": self.weight_loader,
                })

            # ACT_SCALE (for fp8)
            if quant_config.activation_scheme == "static":
                if not quant_config.is_checkpoint_fp8_serialized:
                    raise ValueError(
                        "Found static activation scheme for checkpoint that "
                        "was not serialized fp8.")
                self.a_up_proj_scale = nn.Parameter(torch.zeros(
                    self.num_total_experts, dtype=torch.float32),
                                              requires_grad=False)
                self.a_down_proj_scale = nn.Parameter(torch.zeros(
                    self.num_total_experts, dtype=torch.float32),
                                             requires_grad=False)

                set_weight_attrs(self.a_up_proj_scale, {
                    "weight_loader": self.weight_loader,
                })
                set_weight_attrs(self.a_down_proj_scale, {
                    "weight_loader": self.weight_loader,
                })

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor,
                      weight_name: str, expert_id: int):
        tp_rank = get_tensor_model_parallel_rank()
        param_data = param.data
        shard_size = self.intermediate_size
        shard = slice(tp_rank * shard_size, (tp_rank + 1) * shard_size)
        if weight_name.endswith("w_up_proj.weight"):
            param_data[expert_id, 0:shard_size, :] = loaded_weight[shard, :]
        if weight_name.endswith("w_down_proj.weight"):
            param_data[expert_id, :, :] = loaded_weight[:, shard]
        if "act_scale" in weight_name or "weight_scale" in weight_name:
            param_data[expert_id] = loaded_weight

    def process_weights_after_loading(self):
        # Fp8 is the only case where we need to process after loading.
        if not self.use_fp8:
            return

        # If checkpoint is fp16, quantize here.
        if not self.quant_config.is_checkpoint_fp8_serialized:
            w_up_proj_weight = torch.empty_like(self.w_up_proj_weight.data,
                                          dtype=torch.float8_e4m3fn)
            w_down_proj_weight = torch.empty_like(self.w_down_proj_weight.data,
                                         dtype=torch.float8_e4m3fn)
            for expert in range(self.num_total_experts):
                w_up_proj_weight[expert, :, :], self.w_up_proj_scale[
                    expert] = ops.scaled_fp8_quant(
                        self.w_up_proj_weight.data[expert, :, :])
                w_down_proj_weight[expert, :, :], self.w_down_proj_scale[
                    expert] = ops.scaled_fp8_quant(
                        self.w_down_proj_weight.data[expert, :, :])
            self.w_up_proj_weight = nn.Parameter(w_up_proj_weight, requires_grad=False)
            self.w_down_proj_weight = nn.Parameter(w_down_proj_weight, requires_grad=False)

        # If checkpoint is fp8 + static, cleanup act_scales.
        #   Since state_dict has an act_scale per expert but our kernels
        #   are passed one act_scale shared across all experts.
        elif self.quant_config.activation_scheme == "static":
            if self.a_up_proj_scale is None or self.a_down_proj_scale is None:
                raise ValueError(
                    "QuantConfig has static quantization, but found "
                    "activation scales are None.")

            if (not all_close_1d(self.a_up_proj_scale)
                    or not all_close_1d(self.a_down_proj_scale)):
                print_warning_once(
                    "Found act_scales that are not equal for fp8 MoE layer. "
                    "Using the maximum across experts for each layer. ")

            self.a_up_proj_scale = nn.Parameter(self.a_up_proj_scale.max(),
                                          requires_grad=False)
            self.a_down_proj_scale = nn.Parameter(self.a_down_proj_scale.max(),
                                         requires_grad=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        num_tokens, hidden_size = hidden_states.shape
        hidden_states = hidden_states.view(-1, self.hidden_size)
        # router_logits: (num_tokens, n_experts)
        router_logits, _ = self.gate(hidden_states)

        if not (self.config.moe_use_mixtral_gating or self.config.moe_feature_no_mul_topk):
            hidden_states *= 2

        target_std = self.config.moe_gate_norm_std
        if self.config.moe_use_mixtral_gating:
            if self.config.moe_use_logits_norm:
                router_logits_std = router_logits.std(dim=1, keepdim=True)
                router_logits = router_logits_std / (router_logits_std / target_std)
        else:
            if self.config.moe_use_logits_norm:
                router_logits_std = router_logits.std(dim=1, keepdim=True)
                router_logits = router_logits_std / (router_logits_std / target_std)

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, routing_ids = torch.topk(routing_weights,
                                                       self.top_k,
                                                       dim=-1)
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)

        final_hidden_states = fused_experts(hidden_states,
                                        self.w_up_proj_weight,
                                        self.w_down_proj_weight,
                                        routing_weights,
                                        routing_ids,
                                        inplace=True,
                                        use_fp8=self.use_fp8,
                                        w1_scale=self.w_up_proj_scale,
                                        w2_scale=self.w_down_proj_scale,
                                        a1_scale=self.a_up_proj_scale,
                                        a_down_proj_scale=self.a_down_proj_scale)

        if self.tp_size > 1:
            final_hidden_states = tensor_model_parallel_all_reduce(
                final_hidden_states)

        return final_hidden_states.view(num_tokens, hidden_size)


class SkyworkAttention(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position: int = 4096 * 32,
        rope_theta: float = 10000,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
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

        if isinstance(
                quant_config,
                Fp8Config) and not quant_config.is_checkpoint_fp8_serialized:
            print_warning_once(
                "For Mixtral FP8 quantization, we currently do not quantize "
                "the attention layers until their FP8 performance is improved."
            )
            quant_config = None

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
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v, kv_cache, attn_metadata)
        output, _ = self.o_proj(attn_output)
        return output


class SkyworkDecoderLayer(nn.Module):

    def __init__(
        self,
        config: SkyworkMoeConfig,
        layer_id: int,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        linear_method: Optional[LinearMethodBase] = None,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        # Requires transformers > 4.32.0
        rope_theta = getattr(config, "rope_theta", 10000)
        self.self_attn = SkyworkAttention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            max_position=config.max_position_embeddings,
            num_kv_heads=config.num_key_value_heads,
            rope_theta=rope_theta,
            cache_config=cache_config,
            quant_config=quant_config)
        moe = SkyworkMoE(
            num_experts=config.num_experts[0],
            top_k=TOPK,
            moe_2layer_gate=config.moe_2layer_gate,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            linear_method=linear_method,
            quant_config=quant_config,
            config=config)
        if config.moe_expert_interval == 1:
            self.mlp = IdentityMoE(moe)

        else:
            if (layer_id + 1) % config.moe_expert_interval == 0:
                self.mlp = IdentityMoE(moe)
            else:
                self.mlp = SkyworkMLP(
                    config.num_experts[0],
                    self.hidden_size,
                    config.intermediate_size,
                    linear_method=linear_method,
                )

        self.input_layernorm = RMSNorm(config.hidden_size,
                                       eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size,
                                                eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
        residual: Optional[torch.Tensor],
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
            attn_metadata=attn_metadata,
        )

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual)
        hidden_states = self.block_sparse_moe(hidden_states)
        return hidden_states, residual


class MixtralModel(nn.Module):

    def __init__(
        self,
        config: SkyworkMoeConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        lora_config: Optional[LoRAConfig] = None,
    ) -> None:
        super().__init__()
        self.padding_idx = config.pad_token_id
        lora_vocab = (lora_config.lora_extra_vocab_size *
                      (lora_config.max_loras or 1)) if lora_config else 0
        self.vocab_size = config.vocab_size + lora_vocab
        self.org_vocab_size = config.vocab_size

        self.embed_tokens = VocabParallelEmbedding(
            self.vocab_size,
            config.hidden_size,
            org_num_embeddings=config.vocab_size,
        )
        self.layers = nn.ModuleList([
            SkyworkDecoderLayer(config,
                                layer_id,
                                cache_config,
                                quant_config=quant_config)
            for layer_id in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

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
            hidden_states, residual = layer(positions, hidden_states,
                                            kv_caches[i], attn_metadata,
                                            residual)
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class SkyworkMoeForCausalLM(nn.Module):
    fall_back_to_pt_during_load = False

    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
    }

    # LoRA specific attributes
    supported_lora_modules = [
        "qkv_proj",
        "o_proj",
        "embed_tokens",
        "lm_head",
    ]
    embedding_modules = {
        "embed_tokens": "input_embeddings",
        "lm_head": "output_embeddings",
    }
    embedding_padding_modules = ["lm_head"]

    def __init__(
        self,
        config: SkyworkMoeConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        lora_config: Optional[LoRAConfig] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.model = MixtralModel(config,
                                  cache_config,
                                  quant_config,
                                  lora_config=lora_config)
        self.unpadded_vocab_size = config.vocab_size
        if lora_config:
            self.unpadded_vocab_size += lora_config.lora_extra_vocab_size
        self.lm_head = ParallelLMHead(
            self.unpadded_vocab_size,
            config.hidden_size,
            org_num_embeddings=config.vocab_size,
            padding_size=DEFAULT_VOCAB_PADDING_SIZE
            # We need bigger padding if using lora for kernel
            # compatibility
            if not lora_config else lora_config.lora_vocab_padding_size,
        )
        self.logits_processor = LogitsProcessor(self.unpadded_vocab_size,
                                                config.vocab_size)
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

        expert_params_mapping = [
            # These are the weight scales for the experts
            # (param_name, weight_name, expert_id)
            ("w_up_proj_scale" if weight_name in ["w1", "w3"] else "w_down_proj_scale",
             f"experts.{expert_id}.{weight_name}.weight_scale", expert_id)
            for expert_id in range(self.config.num_experts[0])
            for weight_name in ["w1", "w_down_proj", "w3"]
        ] + [
            # These are the weights for the experts
            # (param_name, weight_name, expert_id)
            ("w_up_proj_weight" if weight_name in ["w1", "w3"] else "w_down_proj_weight",
             f"experts.{expert_id}.{weight_name}.weight", expert_id)
            for expert_id in range(self.config.num_experts[0])
            for weight_name in ["w1", "w_down_proj", "w3"]
        ] + [
            # These are the activation scales for the experts
            # (param_name, weight_name, expert_id)
            ("a_up_proj_scale" if weight_name in ["w1", "w3"] else "a_down_proj_scale",
             f"experts.{expert_id}.{weight_name}.act_scale", expert_id)
            for expert_id in range(self.config.num_experts[0])
            for weight_name in ["w1", "w_down_proj", "w3"]
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
                for param_name, weight_name, expert_id in expert_params_mapping:
                    if weight_name not in name:
                        continue
                    name = name.replace(weight_name, param_name)
                    param = params_dict[name]
                    weight_loader = param.weight_loader
                    weight_loader(param,
                                  loaded_weight,
                                  weight_name,
                                  expert_id=expert_id)
                    break
                else:
                    # Skip loading extra bias for GPTQ models.
                    if name.endswith(".bias") and name not in params_dict:
                        continue
                    # Remapping the name of FP8 kv-scale.
                    if name.endswith("kv_scale"):
                        remapped_kv_scale_name = name.replace(
                            ".kv_scale", ".attn.kv_scale")
                        if remapped_kv_scale_name not in params_dict:
                            print_warning_once(
                                "Found kv scale in the checkpoint "
                                f"(e.g. {name}), but not found the expected "
                                f"name in the model "
                                f"(e.g. {remapped_kv_scale_name}). "
                                "kv-scale is not loaded.")
                            continue
                        else:
                            name = remapped_kv_scale_name
                    param = params_dict[name]
                    weight_loader = getattr(param, "weight_loader",
                                            default_weight_loader)
                    weight_loader(param, loaded_weight)


def all_close_1d(x: torch.Tensor) -> bool:
    assert len(x.shape) == 1
    return all(torch.allclose(x[0], x[i]) for i in range(x.shape[0]))
