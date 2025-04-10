# pyright: reportArgumentType=false
# pyright: reportAssignmentType=false
# pyright: reportOptionalMemberAccess=false
# pyright: reportReturnType=false
# pyright: reportOptionalOperand=false
# pyright: reportOperatorIssue=false
# pyright: reportPrivateImportUsage=false
# SPDX-License-Identifier: Apache-2.0

# Adapted from
# #FIXME: add nGPT transformers (HF) link.
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
"""Inference-only nGPT model compatible with HuggingFace weights."""
from typing import Any, Dict, Iterable, Optional, Set, Tuple, Union

import torch
from torch import nn
from transformers import NGPTConfig

from vllm.attention import Attention
from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import (get_pp_group,
                              get_tensor_model_parallel_world_size,
                              tensor_model_parallel_all_gather,
                              tensor_model_parallel_gather)
from vllm.model_executor.layers.activation import get_act_fn
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               QKVParallelLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.sampler import SamplerOutput, get_sampler
from vllm.model_executor.layers.vocab_parallel_embedding import (
    DEFAULT_VOCAB_PADDING_SIZE, ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader, maybe_remap_kv_scale_name)
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors

from .interfaces import SupportsLoRA, SupportsPP
from .utils import (PPMissingLayer, is_pp_missing_parameter,
                    make_empty_intermediate_tensors_factory, make_layers,
                    maybe_prefix)


def normalize_vector(x, eps=0.0):
    return x / (x.norm(p=2, dim=-1, keepdim=True) + eps)


class NGPTMLP(nn.Module):

    def __init__(
        self,
        config: NGPTConfig,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: Optional[QuantizationConfig] = None,
        bias: bool = False,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size

        self.up_proj = ColumnParallelLinear(
            input_size=hidden_size,
            output_size=intermediate_size,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.up_proj",
        )
        self.down_proj = RowParallelLinear(
            input_size=intermediate_size,
            output_size=hidden_size,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.down_proj",
        )
        self.act_fn = get_act_fn(hidden_act)

        # nGPT: mlp scaling parameters
        self.suv_init_value = getattr(config, "suv_init_value", 1.0)
        self.suv_init_scaling = getattr(config, "suv_init_scaling", 1.0)
        self.suv = ColumnParallelLinear(
            input_size=1,
            output_size=intermediate_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.suv",
        )
        torch.nn.init.constant_(self.suv.weight, self.suv_init_scaling)

    def forward(self, x):
        up, _ = self.up_proj(x)

        # nGPT: mlp scaling
        suv, _ = self.suv(torch.ones((1, 1), device=up.device, dtype=up.dtype))
        suv = suv * ((self.suv_init_value / self.suv_init_scaling) *
                     (self.hidden_size**0.5))

        x = self.act_fn(suv * up)
        x, _ = self.down_proj(x)
        return x


class NGPTAttention(nn.Module):

    def __init__(
        self,
        config: NGPTConfig,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        rope_theta: float = 1000000,
        rope_scaling: Optional[Dict[str, Any]] = None,
        max_position_embeddings: int = 8192,
        quant_config: Optional[QuantizationConfig] = None,
        bias: bool = False,
        cache_config: Optional[CacheConfig] = None,
        prefix: str = "",
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
        # MistralConfig has an optional head_dim introduced by Mistral-Nemo
        self.head_dim = getattr(config, "head_dim",
                                self.hidden_size // self.total_num_heads)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**0.5  # nGPT
        self.rope_theta = rope_theta
        self.partial_rotary_factor = config.partial_rotary_factor
        self.max_position_embeddings = max_position_embeddings

        self.qkv_proj = QKVParallelLinear(
            hidden_size=hidden_size,
            head_size=self.head_dim,
            total_num_heads=self.total_num_heads,
            total_num_kv_heads=self.total_num_kv_heads,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            input_size=self.total_num_heads * self.head_dim,
            output_size=hidden_size,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        # nGPT: attention scaling parameters
        self.sqk_init_value = getattr(config, "sqk_init_value", 1.0)
        self.sqk_init_scaling = config.initializer_range
        self.sqk = ColumnParallelLinear(
            input_size=1,
            output_size=self.q_size * get_tensor_model_parallel_world_size(),
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.sqk",
        )
        torch.nn.init.constant_(self.sqk.weight, self.sqk_init_scaling)

        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
            rope_scaling=rope_scaling,
            partial_rotary_factor=self.partial_rotary_factor,
        )
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
        )

    def forward(self, positions: torch.Tensor,
                hidden_states: torch.Tensor) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)

        # nGPT: attention scaling
        sqk, _ = self.sqk(torch.ones((1, 1), device=q.device, dtype=q.dtype))
        sqk = sqk * (self.sqk_init_value / self.sqk_init_scaling)

        q = normalize_vector(q.view(-1, self.num_heads, self.head_dim)).view(
            -1, self.q_size) * (sqk**2)
        k = normalize_vector(k.view(-1, self.num_kv_heads,
                                    self.head_dim)).view(-1, self.kv_size)

        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output


class NGPTDecoderLayer(nn.Module):

    def __init__(
        self,
        config: NGPTConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        rope_theta = getattr(config, "rope_theta", 1000000)
        rope_scaling = getattr(config, "rope_scaling", None)
        if rope_scaling is not None and getattr(
                config, "original_max_position_embeddings", None):
            rope_scaling["original_max_position_embeddings"] = (
                config.original_max_position_embeddings)
        max_position_embeddings = getattr(config, "max_position_embeddings",
                                          8192)

        # Support abacusai/Smaug-72B-v0.1 with attention_bias
        # Support internlm/internlm-7b with bias
        attention_bias = getattr(config, "attention_bias", False) or getattr(
            config, "bias", False)
        self.self_attn = NGPTAttention(
            config=config,
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=getattr(config, "num_key_value_heads",
                                 config.num_attention_heads),
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            max_position_embeddings=max_position_embeddings,
            quant_config=quant_config,
            bias=attention_bias,
            cache_config=cache_config,
            prefix=f"{prefix}.self_attn",
        )
        # nGPT: attention interpolation parameters
        self.attn_alpha_init_value = getattr(config, "attn_alpha_init_value",
                                             0.05)
        self.attn_alpha_init_scaling = config.initializer_range
        self.attn_alpha = torch.nn.Parameter(
            self.attn_alpha_init_scaling *
            torch.ones(config.hidden_size, dtype=torch.float32))

        self.mlp = NGPTMLP(
            config=config,
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            bias=getattr(config, "mlp_bias", False),
            prefix=f"{prefix}.mlp",
        )
        # nGPT: mlp interpolation parameters
        self.mlp_alpha_init_value = getattr(config, "mlp_alpha_init_value",
                                            0.05)
        self.mlp_alpha_init_scaling = config.initializer_range
        self.mlp_alpha = torch.nn.Parameter(
            self.mlp_alpha_init_scaling *
            torch.ones(config.hidden_size, dtype=torch.float32))

    def residual_update(
        self,
        hidden_states,
        residual,
        alpha,
        alpha_init_value,
        alpha_init_scaling,
    ):
        lr = torch.abs(alpha * (alpha_init_value / alpha_init_scaling)).to(
            hidden_states.dtype)
        A_norm = normalize_vector(residual)
        B_norm = normalize_vector(hidden_states)
        return normalize_vector(A_norm + lr * (B_norm - A_norm))

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        residual = hidden_states

        # Self Attention
        hidden_states = self.self_attn(positions=positions,
                                       hidden_states=hidden_states)
        # nGPT: attention interpolation
        hidden_states = self.residual_update(
            hidden_states,
            residual,
            self.attn_alpha,
            self.attn_alpha_init_value,
            self.attn_alpha_init_scaling,
        )

        # Fully Connected
        residual = hidden_states
        hidden_states = self.mlp(hidden_states)
        # nGPT: mlp interpolation
        hidden_states = self.residual_update(
            hidden_states,
            residual,
            self.mlp_alpha,
            self.mlp_alpha_init_value,
            self.mlp_alpha_init_scaling,
        )
        return hidden_states


@support_torch_compile
class NGPTModel(nn.Module):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        lora_config = vllm_config.lora_config

        self.config = config
        self.quant_config = quant_config
        lora_vocab = ((lora_config.lora_extra_vocab_size *
                       (lora_config.max_loras or 1)) if lora_config else 0)
        self.vocab_size = config.vocab_size + lora_vocab
        self.org_vocab_size = config.vocab_size
        if get_pp_group().is_first_rank or (config.tie_word_embeddings
                                            and get_pp_group().is_last_rank):
            self.embed_tokens = VocabParallelEmbedding(
                self.vocab_size,
                config.hidden_size,
                org_num_embeddings=config.vocab_size,
                quant_config=quant_config,
            )
        else:
            self.embed_tokens = PPMissingLayer()

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: NGPTDecoderLayer(
                config=config,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=prefix,
            ),
            prefix=f"{prefix}.layers",
        )

        self.make_empty_intermediate_tensors = (
            make_empty_intermediate_tensors_factory(
                ["hidden_states", "residual"], config.hidden_size))

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: Optional[torch.Tensor],
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors],
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.get_input_embeddings(input_ids)
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]

        for layer in self.layers[self.start_layer:self.end_layer]:
            hidden_states = layer(positions, hidden_states)

        if not get_pp_group().is_last_rank:
            return IntermediateTensors({"hidden_states": hidden_states})

        return hidden_states


class NGPTForCausalLM(nn.Module, SupportsLoRA, SupportsPP):
    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
    }

    # LoRA specific attributes
    embedding_modules = {
        "embed_tokens": "input_embeddings",
        "lm_head": "output_embeddings",
    }
    embedding_padding_modules = ["lm_head"]

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        lora_config = vllm_config.lora_config

        self.config = config
        self.lora_config = lora_config
        self.quant_config = quant_config

        self.model = NGPTModel(vllm_config=vllm_config,
                               prefix=maybe_prefix(prefix, "model"))

        if get_pp_group().is_last_rank:
            self.unpadded_vocab_size = config.vocab_size
            if lora_config:
                self.unpadded_vocab_size += lora_config.lora_extra_vocab_size
            self.lm_head = ParallelLMHead(
                self.unpadded_vocab_size,
                config.hidden_size,
                org_num_embeddings=config.vocab_size,
                padding_size=
                (  # We need bigger padding if using lora for kernel compatibility
                    DEFAULT_VOCAB_PADDING_SIZE if not lora_config else
                    lora_config.lora_vocab_padding_size),
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
            if config.tie_word_embeddings:
                self.lm_head = self.lm_head.tie_weights(
                    self.model.embed_tokens)

            logit_scale = getattr(config, "logit_scale", 1.0)
            self.logits_processor = LogitsProcessor(self.unpadded_vocab_size,
                                                    config.vocab_size,
                                                    logit_scale)
        else:
            self.lm_head = PPMissingLayer()

        self.sampler = get_sampler()
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors)

        # nGPT: output scaling parameters
        self.sz_init_value = getattr(config, "sz_init_value", 1.0)
        self.sz_init_scaling = config.initializer_range
        self.sz = ColumnParallelLinear(
            input_size=1,
            output_size=config.vocab_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.sz",
        )
        torch.nn.init.constant_(self.sz.weight, self.sz_init_scaling)
        # Whether to use gather or all-gather to gather the sz
        self.use_all_gather = current_platform.use_all_gather()

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.get_input_embeddings(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        hidden_states = self.model(input_ids, positions, intermediate_tensors,
                                   inputs_embeds)
        return hidden_states

    def _gather_sz(self, sz: torch.Tensor) -> torch.Tensor:
        """gather/all-gather the sz tensor across model parallel group."""
        if self.use_all_gather:
            # Gather is not supported for some devices such as TPUs.
            # Use all-gather instead.
            # NOTE(woosuk): Here, the outputs of every device should not be None
            # because XLA requires strict SPMD among all devices. Every device
            # should execute the same operations after gathering the logits.
            sz = tensor_model_parallel_all_gather(sz)
        else:
            # None may be returned for rank > 0
            sz = tensor_model_parallel_gather(sz)
        return sz

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        logits = self.logits_processor(self.lm_head, hidden_states,
                                       sampling_metadata)

        # When PP>1, this function is called only on the last rank.
        # logits_processor, gathers the logits from all TP ranks.
        # But when both PP>1 and TP>1, some TP ranks in the last PP rank have logits as None.
        # The if-block below handles this edge case.
        if logits is not None:
            # nGPT: output scaling
            sz, _ = self.sz(
                torch.ones((1, 1), device=logits.device, dtype=logits.dtype))
            sz = sz * (self.sz_init_value / self.sz_init_scaling)
            sz = self._gather_sz(sz)
            logits = logits * sz

        return logits

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        next_tokens = self.sampler(logits, sampling_metadata)
        return next_tokens

    def load_weights(self, weights: Iterable[Tuple[str,
                                                   torch.Tensor]]) -> Set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
        ]
        params_dict = dict(self.named_parameters())
        loaded_params: Set[str] = set()
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            if ("rotary_emb.cos_cached" in name
                    or "rotary_emb.sin_cached" in name):
                # Models trained using ColossalAI may include these tensors in
                # the checkpoint. Skip them.
                continue
            if self.quant_config is not None and (
                    scale_name := self.quant_config.get_cache_scale(name)):
                # Loading kv cache quantization scales
                param = params_dict[scale_name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                loaded_weight = (loaded_weight if loaded_weight.dim() == 0 else
                                 loaded_weight[0])
                weight_loader(param, loaded_weight)
                loaded_params.add(scale_name)
                continue
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue

                if is_pp_missing_parameter(name, self):
                    continue

                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)

                break
            else:
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                # Remapping the name of FP8 kv-scale.
                name = maybe_remap_kv_scale_name(name, params_dict)
                if name is None:
                    continue

                if is_pp_missing_parameter(name, self):
                    continue

                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params
