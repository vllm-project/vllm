# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Copyright 2025 The Baidu team.
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
"""Inference-only Erine VL model compatible with HuggingFace weights."""

from collections.abc import Iterable
from itertools import islice
from typing import Any

import torch
from torch import nn
from transformers import PretrainedConfig

from vllm.attention.layer import Attention

# from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import get_pp_group, get_tensor_model_parallel_world_size
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import SharedFusedMoE
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding.ernie45_vl_rope import (
    Ernie4_5_VLRotaryEmbedding,
)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.config import set_default_rope_theta

from .ernie45_moe import Ernie4_5_MoeMLP
from .interfaces import SupportsPP
from .utils import (
    PPMissingLayer,
    extract_layer_index,
    is_pp_missing_parameter,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)

logger = init_logger(__name__)


class Ernie4_5_VLMoeMLP(Ernie4_5_MoeMLP):
    def __init__(self, shared_experts: torch.nn.Module | None = None, **kwargs):
        super().__init__(**kwargs)
        self.shared_experts = shared_experts

    def forward(self, x):
        if self.shared_experts is not None:
            return self.shared_experts(x) + super().forward(x)
        else:
            return super().forward(x)


class Ernie4_5_VLMoeAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        rope_parameters: dict[str, Any],
        head_dim: int | None = None,
        freq_allocation: int = 20,
        max_position_embeddings: int = 131072,
        rms_norm_eps: float = 1e-05,
        qkv_bias: bool = False,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        layer_idx = extract_layer_index(prefix) if len(prefix) > 0 else 0
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
        self.head_dim = head_dim or (hidden_size // self.total_num_heads)

        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.max_position_embeddings = max_position_embeddings

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=qkv_bias,
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

        t_rope = freq_allocation
        h_rope = (self.head_dim // 2 - freq_allocation) // 2
        w_rope = (self.head_dim // 2 - freq_allocation) // 2

        self.rotary_emb = Ernie4_5_VLRotaryEmbedding(
            head_size=self.head_dim,
            rotary_dim=self.head_dim,
            max_position_embeddings=max_position_embeddings,
            base=rope_parameters["rope_theta"],
            is_neox_style=False,
            dtype=torch.get_default_dtype(),
            mrope_section=[h_rope, w_rope, t_rope],
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

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)

        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)

        # Attention
        attn_output = self.attn(q, k, v)
        # Output projection
        output, _ = self.o_proj(attn_output)
        return output


class Ernie4_5_VLMoeMoE(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()

        layer_idx = extract_layer_index(prefix)
        self.layer_idx = layer_idx
        self.tp_size = get_tensor_model_parallel_world_size()
        self.has_shared_experts = getattr(config, "moe_num_shared_experts", 0) > 0
        self.hidden_size = config.hidden_size

        moe_num_experts = config.moe_num_experts
        max_moe_num_experts = max(moe_num_experts)

        if self.tp_size > max_moe_num_experts:
            raise ValueError(
                f"Tensor parallel size {self.tp_size} is greater than "
                f"the number of experts {moe_num_experts}."
            )

        moe_layer_start_index = config.moe_layer_start_index
        text_moe_layer_start_index = moe_layer_start_index[0]
        vision_moe_layer_start_index = moe_layer_start_index[1]
        moe_layer_end_index = config.moe_layer_end_index
        moe_layer_end_index = getattr(
            config,
            "moe_layer_end_index",
            [config.num_hidden_layers - 1, config.num_hidden_layers - 1],
        )
        text_moe_layer_end_index = moe_layer_end_index[0]
        vision_moe_layer_end_index = moe_layer_end_index[1]

        assert config.moe_num_experts[0] == config.moe_num_experts[1]
        self.e_score_correction_bias = nn.Parameter(
            torch.empty(2, config.moe_num_experts[0], dtype=torch.float32)
        )

        assert text_moe_layer_start_index <= text_moe_layer_end_index

        if self.has_shared_experts:
            intermediate_size = (
                config.moe_intermediate_size[0] * config.moe_num_shared_experts
            )
            self.shared_experts = Ernie4_5_VLMoeMLP(
                hidden_size=config.hidden_size,
                intermediate_size=intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=f"{prefix}.shared_experts",
                reduce_results=False,
            )
        else:
            self.shared_experts = None

        if (
            layer_idx >= text_moe_layer_start_index
            and layer_idx <= text_moe_layer_end_index
        ):
            self.text_experts_gate = ReplicatedLinear(
                config.hidden_size,
                config.moe_num_experts[0],
                bias=False,
                params_dtype=torch.float32,
                quant_config=quant_config,
                prefix=f"{prefix}.text_experts_gate",
            )

            self.text_experts = SharedFusedMoE(
                shared_experts=self.shared_experts,
                num_experts=config.moe_num_experts[0],
                top_k=config.moe_k,
                hidden_size=config.hidden_size,
                intermediate_size=config.moe_intermediate_size[0],
                reduce_results=False,
                renormalize=True,
                quant_config=quant_config,
                e_score_correction_bias=self.e_score_correction_bias[0],
                prefix=f"{prefix}.text_experts",
            )
        else:
            self.text_experts = Ernie4_5_VLMoeMLP(
                shared_experts=self.shared_experts,
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                use_bias=getattr(config, "use_bias", False),
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
            )

        assert vision_moe_layer_start_index <= vision_moe_layer_end_index
        if (
            layer_idx >= vision_moe_layer_start_index
            and layer_idx <= vision_moe_layer_end_index
        ):
            self.vision_experts_gate = ReplicatedLinear(
                config.hidden_size,
                config.moe_num_experts[1],
                bias=False,
                params_dtype=torch.float32,
                quant_config=quant_config,
                prefix=f"{prefix}.vision_experts_gate",
            )

            self.vision_experts = SharedFusedMoE(
                shared_experts=self.shared_experts,
                num_experts=config.moe_num_experts[1],
                top_k=config.moe_k,
                hidden_size=config.hidden_size,
                intermediate_size=config.moe_intermediate_size[1],
                reduce_results=False,
                renormalize=True,
                quant_config=quant_config,
                e_score_correction_bias=self.e_score_correction_bias[1],
                prefix=f"{prefix}.vision_experts",
            )
        else:
            self.vision_experts = Ernie4_5_VLMoeMLP(
                shared_experts=self.shared_experts,
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                use_bias=getattr(config, "use_bias", False),
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        visual_token_mask: torch.Tensor,
        **kwargs: object,
    ) -> torch.Tensor:
        orig_shape = hidden_states.shape
        hidden_dim = hidden_states.shape[-1]
        hidden_states = hidden_states.view(-1, hidden_dim)

        if visual_token_mask is not None and visual_token_mask.all():
            # only vision modal input
            router_logits, _ = self.vision_experts_gate(
                hidden_states.to(dtype=torch.float32)
            )
            final_hidden_states = self.vision_experts(
                hidden_states=hidden_states, router_logits=router_logits
            )
        elif visual_token_mask is not None and visual_token_mask.any():
            # text and vision modals input
            visual_token_mask = visual_token_mask.repeat(1, self.hidden_size).bool()
            text_token_mask = ~visual_token_mask
            final_experts_hidden_states = torch.zeros_like(hidden_states)
            final_shared_ouput = (
                torch.zeros_like(hidden_states) if self.has_shared_experts else None
            )

            text_hidden_states = hidden_states[text_token_mask].reshape(
                -1, self.hidden_size
            )
            vision_hidden_states = hidden_states[visual_token_mask].reshape(
                -1, self.hidden_size
            )

            text_router_logits, _ = self.text_experts_gate(
                text_hidden_states.to(dtype=torch.float32)
            )
            text_shared_ouput, text_experts_output = self.text_experts(
                hidden_states=text_hidden_states, router_logits=text_router_logits
            )
            final_experts_hidden_states[text_token_mask] = text_experts_output.flatten()
            if self.has_shared_experts:
                final_shared_ouput[text_token_mask] = text_shared_ouput.flatten()

            vision_router_logits, _ = self.vision_experts_gate(
                vision_hidden_states.to(dtype=torch.float32)
            )
            vision_shared_ouput, vision_experts_output = self.vision_experts(
                hidden_states=vision_hidden_states, router_logits=vision_router_logits
            )
            final_experts_hidden_states[visual_token_mask] = (
                vision_experts_output.flatten()
            )
            if self.has_shared_experts:
                final_shared_ouput[visual_token_mask] = vision_shared_ouput.flatten()

            final_hidden_states = (final_shared_ouput, final_experts_hidden_states)
        else:
            # only text modal input
            text_router_logits, _ = self.text_experts_gate(
                hidden_states.to(dtype=torch.float32)
            )

            final_hidden_states = self.text_experts(
                hidden_states=hidden_states, router_logits=text_router_logits
            )

        if self.has_shared_experts:
            # for shared_experts model
            final_hidden_states = final_hidden_states[0] + final_hidden_states[1]
        else:
            # for not shared_experts model
            final_hidden_states = final_hidden_states[1]

        if self.tp_size > 1:
            final_hidden_states = (
                self.text_experts.maybe_all_reduce_tensor_model_parallel(
                    final_hidden_states
                )
            )

        return final_hidden_states.view(orig_shape)


class Ernie4_5_VLMoeDecoderLayer(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        set_default_rope_theta(config, default_theta=500000)
        freq_allocation = getattr(config, "freq_allocation", 20)
        max_position_embeddings = getattr(config, "max_position_embeddings", 131072)

        self.self_attn = Ernie4_5_VLMoeAttention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            head_dim=getattr(config, "head_dim", None),
            rope_parameters=config.rope_parameters,
            freq_allocation=freq_allocation,
            max_position_embeddings=max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=getattr(config, "use_bias", False),
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
        )

        layer_idx = extract_layer_index(prefix)
        self.layer_idx = layer_idx

        # MoE
        moe_layer_start_index = config.moe_layer_start_index
        min_moe_layer_start_index = min(moe_layer_start_index)
        moe_layer_end_index = getattr(
            config,
            "moe_layer_end_index",
            [config.num_hidden_layers - 1, config.num_hidden_layers - 1],
        )
        max_moe_layer_end_index = max(moe_layer_end_index)
        assert min_moe_layer_start_index <= max_moe_layer_end_index
        moe_num_experts = config.moe_num_experts
        max_moe_num_experts = max(moe_num_experts)
        moe_layer_interval = getattr(config, "moe_layer_interval", 1)
        use_moe = getattr(config, "use_moe", max_moe_num_experts > 0)

        if (
            use_moe
            and ((layer_idx + 1) % moe_layer_interval == 0)
            and layer_idx >= min_moe_layer_start_index
            and layer_idx <= max_moe_layer_end_index
        ):
            self.mlp = Ernie4_5_VLMoeMoE(
                config=config, quant_config=quant_config, prefix=f"{prefix}.mlp"
            )
        else:
            self.mlp = Ernie4_5_VLMoeMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                use_bias=getattr(config, "use_bias", False),
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
        residual: torch.Tensor | None,
        visual_token_mask: torch.Tensor | None,
        **kwargs: object,
    ) -> torch.Tensor:
        # Self Attention
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
        )

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)

        if isinstance(self.mlp, Ernie4_5_VLMoeMoE):
            hidden_states = self.mlp(hidden_states, visual_token_mask, **kwargs)
        else:
            hidden_states = self.mlp(hidden_states)

        return hidden_states, residual


# Since Ernie VL distinguishes between text experts and vision experts,
# enabling torch.compile will cause errors.
# @support_torch_compile(
#     dynamic_arg_dims={
#         "input_ids": 0,
#         "positions": -1,
#         "intermediate_tensors": 0,
#         "inputs_embeds": 0,
#         "visual_token_mask": 0,
#     })
class Ernie4_5_VLMoeModel(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config

        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.config = config

        self.im_patch_id = config.im_patch_id

        if get_pp_group().is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=f"{prefix}.embed_tokens",
            )
        else:
            self.embed_tokens = PPMissingLayer()

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: Ernie4_5_VLMoeDecoderLayer(
                config=config,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=prefix,
            ),
            prefix=f"{prefix}.layers",
        )

        if get_pp_group().is_last_rank:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()

        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], config.hidden_size
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        visual_token_mask: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_input_ids(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        for layer in islice(self.layers, self.start_layer, self.end_layer):
            hidden_states, residual = layer(
                positions, hidden_states, residual, visual_token_mask, **kwargs
            )

        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )

        hidden_states, _ = self.norm(hidden_states, residual)

        return hidden_states


# only used as text backbone for ernie4.5-vl
class Ernie4_5_VLMoeForCausalLM(nn.Module, SupportsPP):
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

    fall_back_to_pt_during_load = False

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.quant_config = quant_config
        self.model = Ernie4_5_VLMoeModel(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )

        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
        else:
            self.lm_head = PPMissingLayer()

        if self.config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors:
        hidden_states = self.model(
            input_ids, positions, intermediate_tensors, inputs_embeds, **kwargs
        )
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        logits = self.logits_processor(self.lm_head, hidden_states)
        return logits

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        # Params for weights, fp8 weight scales, fp8 activation scales
        # (param_name, weight_name, expert_id, shard_id)
        expert_params_mapping = SharedFusedMoE.make_expert_params_mapping(
            self,
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=max(self.config.moe_num_experts),
        )

        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        for name, loaded_weight in weights:
            if self.config.tie_word_embeddings and name.endswith("lm_head.weight"):
                loaded_params.add("lm_head.weight")
                continue
            # MTP will be supported soon.
            if "mtp" in name or "vision_model" in name or "resampler_model" in name:
                continue

            for param_name, weight_name, shard_id in stacked_params_mapping:
                # Skip non-stacked layers and experts (experts handled below).
                if weight_name not in name:
                    continue

                if ("mlp.experts." in name) and name not in params_dict:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if (
                    name.endswith(".bias") or name.endswith("_bias")
                ) and name not in params_dict:
                    continue
                # Skip layers on other devices.
                if is_pp_missing_parameter(name, self):
                    continue

                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Distinguish between vision experts and text experts
                if "mlp.experts" in name:
                    moe_offset = int(name.split(".")[-3])
                    vision_expert_start_idx = self.config.moe_num_experts[0]
                    is_text_expert = moe_offset <= vision_expert_start_idx - 1
                    if is_text_expert:
                        name = name.replace(".experts.", ".text_experts.")
                    else:
                        name = name.replace(
                            f".experts.{moe_offset}",
                            f".vision_experts.{moe_offset - vision_expert_start_idx}",
                        )

                for mapping in expert_params_mapping:
                    param_name, weight_name, expert_id, shard_id = mapping

                    if weight_name not in name:
                        continue

                    # Distinguish between vision experts and text experts
                    moe_offset = int(name.split(".")[-3])
                    is_text_expert = moe_offset <= self.config.moe_num_experts[0] - 1

                    name = name.replace(weight_name, param_name)
                    if is_text_expert:
                        name = name.replace(".experts.", ".text_experts.")
                    else:
                        name = name.replace(".experts.", ".vision_experts.")

                    # Skip layers on other devices.
                    if is_pp_missing_parameter(name, self):
                        continue

                    # Skip loading extra bias for GPTQ models.
                    if (
                        name.endswith(".bias") or name.endswith("_bias")
                    ) and name not in params_dict:
                        continue
                    param = params_dict[name]

                    weight_loader = param.weight_loader
                    weight_loader(
                        param,
                        loaded_weight,
                        name,
                        shard_id=shard_id,
                        expert_id=expert_id,
                    )
                    break
                else:
                    # Distinguish between vision expert gate
                    # and text expert gate
                    if name.endswith("mlp.gate.weight"):
                        name = name.replace("gate.weight", "text_experts_gate.weight")
                        loaded_weight = loaded_weight.T
                    elif name.endswith("mlp.gate.weight_1"):
                        name = name.replace(
                            "gate.weight_1", "vision_experts_gate.weight"
                        )
                        loaded_weight = loaded_weight.T

                    if "e_score_correction_bias" in name:
                        name = name.replace(".moe_statics.", ".")

                    # Skip loading extra bias for GPTQ models.
                    if (
                        name.endswith(".bias") or name.endswith("_bias")
                    ) and name not in params_dict:
                        continue
                    # Skip layers on other devices.
                    if is_pp_missing_parameter(name, self):
                        continue
                    # Remapping the name of FP8 kv-scale.
                    name = maybe_remap_kv_scale_name(name, params_dict)
                    if name is None:
                        continue

                    param = params_dict[name]

                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params
