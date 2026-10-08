# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

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
"""Inference-only LLaMA model compatible with HuggingFace weights."""

from collections.abc import Iterable
from itertools import islice
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist
from torch import nn
from transformers import LlamaConfig

from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import get_pp_group, get_tensor_model_parallel_world_size
from vllm.distributed.parallel_state import get_tp_group
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.attention import (
    Attention,
    EncoderOnlyAttention,
)
from vllm.model_executor.layers.fusion.fused_act_quant import maybe_fused_act_quant
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
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
from vllm.sequence import IntermediateTensors
from vllm.v1.attention.backend import AttentionType
from vllm.v1.worker.tpsp_profile import (
    SPProfile,
    TPSPBackend,
    TPSPProfileSession,
    TPSPShape,
    select_sp_config,
)

from .adapters import as_embedding_model, as_seq_cls_model
from .interfaces import (
    EagleModelMixin,
    LocalArgmaxMixin,
    SupportsEagle,
    SupportsEagle3,
    SupportsLoRA,
    SupportsPP,
    SupportsQuant,
)
from .utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    WeightsMapper,
    extract_layer_index,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
    spec_decode_needs_target_embed,
)


class LlamaMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: QuantizationConfig | None = None,
        bias: bool = False,
        prefix: str = "",
        reduce_results: bool = True,
        disable_tp: bool = False,
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            input_size=hidden_size,
            output_sizes=[intermediate_size] * 2,
            bias=bias,
            quant_config=quant_config,
            disable_tp=disable_tp,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            input_size=intermediate_size,
            output_size=hidden_size,
            bias=bias,
            quant_config=quant_config,
            reduce_results=reduce_results,
            disable_tp=disable_tp,
            prefix=f"{prefix}.down_proj",
        )
        if hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {hidden_act}. Only silu is supported for now."
            )
        self.act_fn = SiluAndMul()

    def forward(self, x):
        x, _ = self.gate_up_proj(x)
        x = maybe_fused_act_quant(self.act_fn, x, self.down_proj)
        x, _ = self.down_proj(x)
        return x


class LlamaAttention(nn.Module):
    def __init__(
        self,
        config: LlamaConfig,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position_embeddings: int = 8192,
        quant_config: QuantizationConfig | None = None,
        bias: bool = False,
        bias_o_proj: bool = False,
        cache_config: CacheConfig | None = None,
        prefix: str = "",
        attn_type: str = AttentionType.DECODER,
    ) -> None:
        super().__init__()
        layer_idx = extract_layer_index(prefix)
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

        head_dim = getattr(config, "head_dim", None)
        self.head_dim = head_dim or self.hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
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
            bias=bias_o_proj,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        self._init_rotary_emb(config, quant_config=quant_config)

        sliding_window = None
        if layer_types := getattr(config, "layer_types", None):
            # Fix for Eagle3 compatibility:
            # for draft models, subtract target layer count
            # to get draft-relative layer index starting from 0
            if hasattr(config, "target_layer_count"):
                # This is a draft model,
                # adjust layer_idx to be relative to draft layers
                effective_layer_idx = layer_idx - config.target_layer_count
            else:
                # This is a target model, use layer_idx directly
                effective_layer_idx = layer_idx
            assert effective_layer_idx < len(layer_types), (
                f"effective_layer_idx: {effective_layer_idx} "
                f"is out of bounds for layer_types: {layer_types}"
            )

            is_sliding = layer_types[effective_layer_idx] == "sliding_attention"
            if is_sliding:
                sliding_window = config.sliding_window

        attn_cls = (
            EncoderOnlyAttention
            if attn_type == AttentionType.ENCODER_ONLY
            else Attention
        )

        self.attn = attn_cls(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            per_layer_sliding_window=sliding_window,
            attn_type=attn_type,
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
        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output

    def _init_rotary_emb(
        self,
        config: LlamaConfig,
        quant_config: QuantizationConfig | None,
    ) -> None:
        is_neox_style = True

        self.rotary_emb = get_rope(
            self.head_dim,
            max_position=self.max_position_embeddings,
            rope_parameters=getattr(config, "rope_parameters", None),
            is_neox_style=is_neox_style,
        )


class LlamaDecoderLayer(nn.Module):
    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str = "",
        config: LlamaConfig | None = None,
        attn_layer_type: type[nn.Module] = LlamaAttention,
    ) -> None:
        super().__init__()

        config = config or vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = self.get_quant_config(vllm_config)

        self.hidden_size = config.hidden_size
        max_position_embeddings = getattr(config, "max_position_embeddings", 8192)
        # Support abacusai/Smaug-72B-v0.1 with attention_bias
        attention_bias = getattr(config, "attention_bias", False) or getattr(
            config, "bias", False
        )
        bias_o_proj = attention_bias
        # support internlm/internlm3-8b with qkv_bias
        if hasattr(config, "qkv_bias"):
            attention_bias = config.qkv_bias

        # By default, Llama uses causal attention as it is a decoder-only model.
        # You can override the HF config with `is_causal=False` to enable
        # bidirectional attention, which is used in some embedding models
        # (e.g. nvidia/llama-nemotron-embed-1b-v2)
        if getattr(config, "is_causal", True):
            attn_type = AttentionType.DECODER
        else:
            attn_type = AttentionType.ENCODER_ONLY

        self.self_attn = attn_layer_type(
            config=config,
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=getattr(
                config, "num_key_value_heads", config.num_attention_heads
            ),
            max_position_embeddings=max_position_embeddings,
            quant_config=quant_config,
            bias=attention_bias,
            bias_o_proj=bias_o_proj,
            cache_config=cache_config,
            prefix=f"{prefix}.self_attn",
            attn_type=attn_type,
        )
        self.mlp = LlamaMLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            bias=getattr(config, "mlp_bias", False),
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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Self Attention
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions=positions, hidden_states=hidden_states)

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual

    def get_quant_config(self, vllm_config: VllmConfig) -> QuantizationConfig | None:
        """Get quantization config for this layer. Override in subclasses."""
        return vllm_config.quant_config


@support_torch_compile(
    # TODO[#32068]: Investigate recompilation
    # mark_unbacked_dims={"input_ids": 0},
    dynamic_arg_dims={
        "input_ids": {0: "b"},
        "positions": {0: "b"},
        "intermediate_tensors": {0: "b"},
        "inputs_embeds": {0: "b"},
    },
)
class LlamaModel(nn.Module, EagleModelMixin):
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_stacked={
            # weight_name: (param_name, shard_id)
            ".q_proj": (".qkv_proj", "q"),
            ".k_proj": (".qkv_proj", "k"),
            ".v_proj": (".qkv_proj", "v"),
            ".gate_proj": (".gate_up_proj", 0),
            ".up_proj": (".gate_up_proj", 1),
        }
    )
    supports_aux_hidden_states_over_pp = True

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
        layer_type: type[nn.Module] = LlamaDecoderLayer,
    ):
        super().__init__()

        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config

        self.config = config
        self.quant_config = quant_config

        self.vocab_size = config.vocab_size

        if (
            get_pp_group().is_first_rank
            or (config.tie_word_embeddings and get_pp_group().is_last_rank)
            or spec_decode_needs_target_embed(vllm_config)
        ):
            self.embed_tokens = VocabParallelEmbedding(
                self.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
            )
        else:
            self.embed_tokens = PPMissingLayer()
        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: layer_type(vllm_config=vllm_config, prefix=prefix),
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
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None,
        inputs_embeds: torch.Tensor | None = None,
        **extra_layer_kwargs,
    ) -> torch.Tensor | IntermediateTensors | tuple[torch.Tensor, list[torch.Tensor]]:
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

        remote_aux = self.collect_remote_aux_hidden_states(intermediate_tensors)

        aux_hidden_states: list[torch.Tensor] = []
        if get_pp_group().is_first_rank:
            self._maybe_add_hidden_state(
                aux_hidden_states, self.start_layer, hidden_states, residual
            )
        for idx, layer in enumerate(
            islice(self.layers, self.start_layer, self.end_layer),
            start=self.start_layer,
        ):
            hidden_states, residual = layer(
                positions, hidden_states, residual, **extra_layer_kwargs
            )
            self._maybe_add_hidden_state(
                aux_hidden_states, idx + 1, hidden_states, residual
            )

        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {
                    "hidden_states": hidden_states,
                    "residual": residual,
                    **self.pack_local_aux_hidden_states(aux_hidden_states),
                }
            )

        hidden_states, _ = self.norm(hidden_states, residual)

        aux_hidden_states = remote_aux + aux_hidden_states
        if len(aux_hidden_states) > 0:
            return hidden_states, aux_hidden_states
        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)


class LlamaForCausalLM(
    LocalArgmaxMixin,
    nn.Module,
    SupportsLoRA,
    SupportsPP,
    SupportsEagle,
    SupportsEagle3,
    SupportsQuant,
):
    hf_to_vllm_mapper = LlamaModel.hf_to_vllm_mapper
    # LoRA specific attributes
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }
    embedding_modules = {
        "embed_tokens": "input_embeddings",
        "lm_head": "output_embeddings",
    }

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
        layer_type: type[nn.Module] = LlamaDecoderLayer,
    ):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config

        self.model = self._init_model(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "model"),
            layer_type=layer_type,
        )

        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
            if config.tie_word_embeddings:
                self.lm_head = self.lm_head.tie_weights(self.model.embed_tokens)

            logit_scale = getattr(config, "logit_scale", 1.0)
            self.logits_processor = LogitsProcessor(
                config.vocab_size, scale=logit_scale
            )
        else:
            self.lm_head = PPMissingLayer()

        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

    def _init_model(
        self,
        vllm_config: VllmConfig,
        prefix: str = "",
        layer_type: type[nn.Module] = LlamaDecoderLayer,
    ):
        return LlamaModel(vllm_config=vllm_config, prefix=prefix, layer_type=layer_type)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        model_output = self.model(
            input_ids, positions, intermediate_tensors, inputs_embeds
        )
        return model_output

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        logits = self.logits_processor(self.lm_head, hidden_states)
        return logits

    def compute_logits_local(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        return self.logits_processor(self.lm_head, hidden_states, skip_gather=True)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)


if TYPE_CHECKING:
    _LlamaBidirectionalForSequenceClassificationBase = LlamaForCausalLM
    _LlamaBidirectionalModelBase = LlamaForCausalLM
else:
    _LlamaBidirectionalForSequenceClassificationBase = as_seq_cls_model(
        LlamaForCausalLM
    )
    _LlamaBidirectionalModelBase = as_embedding_model(LlamaForCausalLM)


class LlamaBidirectionalForSequenceClassification(
    _LlamaBidirectionalForSequenceClassificationBase
):
    # This class sets the correct attention type and pooling type
    # through LlamaBidirectionalConfig.
    pass


class LlamaBidirectionalModel(_LlamaBidirectionalModelBase):
    # This class sets the correct attention type and pooling type
    # through LlamaBidirectionalConfig.
    pass


class TPSPLlamaDecoderLayer(LlamaDecoderLayer):
    def _project_and_normalize(
        self,
        x,
        projection,
        residual,
        norm,
        profile: SPProfile,
        backend: TPSPBackend | None,
        residual_is_sharded: bool,
    ):
        group = get_tp_group()
        tp_size = group.world_size
        if profile.tp_size != tp_size or profile.hidden_size != self.hidden_size:
            raise RuntimeError("TP/SP profile does not match the Llama projection")
        if (
            profile.input_width is not None
            and projection.input_size_per_partition != profile.input_width
        ):
            raise RuntimeError("TP/SP projection input width was not profiled")
        rows = (x.size(0) + tp_size - 1) // tp_size
        if not residual_is_sharded:
            if residual.shape != (x.size(0), self.hidden_size):
                raise RuntimeError("TP/SP full residual has an unexpected shape")
            start = group.rank_in_group * rows
            local_residual = torch.zeros(
                (rows, residual.size(-1)), device=residual.device, dtype=residual.dtype
            )
            count = min(rows, max(0, x.size(0) - start))
            if count:
                local_residual[:count] = residual[start : start + count]
        else:
            if residual.shape != (rows, self.hidden_size):
                raise RuntimeError("TP/SP residual shard has an unexpected shape")
            local_residual = residual

        if (
            projection.bias is None
            or (backend is not None and backend.device.type == "cuda")
        ) and select_sp_config(profile, x.size(0)):
            if backend is None or profile.config is None:
                raise RuntimeError("TP/SP Llama profile has no enabled configuration")
            if x.dtype != torch.bfloat16 or projection.weight.dtype != torch.bfloat16:
                raise RuntimeError("TP/SP Llama requires BF16 activations and weights")
            weight = projection.weight
            key = (weight.data_ptr(), weight._version)
            cached = getattr(projection, "_tpsp_transposed_weight", None)
            if cached is None or cached[0] != key:
                cached = (key, weight.T.contiguous())
                projection._tpsp_transposed_weight = cached
            reduced, _, gathered = backend.fused(
                x.contiguous(),
                cached[1],
                norm.weight,
                local_residual,
                norm.variance_epsilon,
                profile.config,
                projection_bias=projection.bias,
            )
            return gathered, reduced

        full, _ = projection(x)
        if not residual_is_sharded:
            full_residual = residual.clone()
        else:
            padded = torch.empty(
                (tp_size * rows, self.hidden_size), device=x.device, dtype=x.dtype
            )
            dist.all_gather_into_tensor(
                padded, local_residual.contiguous(), group=group.device_group
            )
            full_residual = padded[: x.size(0)].contiguous()
        gathered, new_residual = norm(full, full_residual)
        reduced = torch.zeros((rows, self.hidden_size), device=x.device, dtype=x.dtype)
        start = group.rank_in_group * rows
        count = min(rows, max(0, x.size(0) - start))
        if count:
            reduced[:count] = new_residual[start : start + count]
        return gathered, reduced

    def forward_sp(
        self,
        positions,
        hidden_states,
        residual,
        next_norm,
        o_profile: SPProfile,
        down_profile: SPProfile,
        backend: TPSPBackend | None,
        residual_is_sharded: bool,
    ):
        attention = self.self_attn
        qkv, _ = attention.qkv_proj(hidden_states)
        q, k, v = qkv.split(
            [attention.q_size, attention.kv_size, attention.kv_size], dim=-1
        )
        q, k = attention.rotary_emb(positions, q, k)
        attn_output = attention.attn(q, k, v)
        hidden_states, residual = self._project_and_normalize(
            attn_output,
            attention.o_proj,
            residual,
            self.post_attention_layernorm,
            o_profile,
            backend,
            residual_is_sharded,
        )
        mlp = self.mlp
        hidden_states, _ = mlp.gate_up_proj(hidden_states)
        hidden_states = mlp.act_fn(hidden_states)
        return self._project_and_normalize(
            hidden_states,
            mlp.down_proj,
            residual,
            next_norm,
            down_profile,
            backend,
            True,
        )


class TPSPLlamaModel(LlamaModel):
    def __init__(self, *, vllm_config, prefix="", layer_type=TPSPLlamaDecoderLayer):
        super().__init__(vllm_config=vllm_config, prefix=prefix, layer_type=layer_type)
        if vllm_config.device_config.device.type == "cuda":
            # NCCL collectives in the CUDA microchunk path cannot be partitioned
            # by Dynamo's full-graph compilation.
            self.do_not_compile = True
        self.tpsp_profile: TPSPProfileSession | None = None

    def forward(
        self,
        input_ids,
        positions,
        intermediate_tensors,
        inputs_embeds=None,
        **extra_layer_kwargs,
    ):
        if get_pp_group().world_size != 1 or intermediate_tensors is not None:
            raise RuntimeError("TP/SP Llama does not support pipeline parallelism")
        if extra_layer_kwargs:
            raise RuntimeError("TP/SP Llama does not support extra layer arguments")
        if self.tpsp_profile is None or self.tpsp_profile.profiles is None:
            raise RuntimeError("TP/SP Llama requires worker startup profiling")
        profiles = self.tpsp_profile.profiles
        if not (profiles["o"].enabled and profiles["down"].enabled):
            return super().forward(
                input_ids, positions, intermediate_tensors, inputs_embeds=inputs_embeds
            )
        hidden_states = (
            inputs_embeds
            if inputs_embeds is not None
            else self.embed_input_ids(input_ids)
        )
        residual = hidden_states
        aux_hidden_states: list[torch.Tensor] = []
        self._maybe_add_hidden_state(
            aux_hidden_states, self.start_layer, hidden_states, None
        )
        hidden_states = self.layers[0].input_layernorm(hidden_states)
        for idx, layer in enumerate(self.layers):
            next_norm = (
                self.layers[idx + 1].input_layernorm
                if idx + 1 < len(self.layers)
                else self.norm
            )
            hidden_states, residual = layer.forward_sp(
                positions,
                hidden_states,
                residual,
                next_norm,
                profiles["o"],
                profiles["down"],
                self.tpsp_profile.backend,
                idx != 0,
            )
            if idx + 1 in self.aux_hidden_state_layers:
                group = get_tp_group()
                padded = torch.empty(
                    (group.world_size * residual.size(0), self.config.hidden_size),
                    device=residual.device,
                    dtype=residual.dtype,
                )
                dist.all_gather_into_tensor(
                    padded, residual.contiguous(), group=group.device_group
                )
                self._maybe_add_hidden_state(
                    aux_hidden_states, idx + 1, padded[: hidden_states.size(0)], None
                )
        if aux_hidden_states:
            return hidden_states, aux_hidden_states
        return hidden_states


class TPSPLlamaForCausalLM(LlamaForCausalLM):
    def __init__(self, *, vllm_config, prefix="", layer_type=TPSPLlamaDecoderLayer):
        if vllm_config.parallel_config.pipeline_parallel_size > 1:
            raise ValueError(
                "--enable-tpsp is incompatible with --pipeline-parallel-size > 1"
            )
        if vllm_config.quant_config is not None:
            raise RuntimeError("TP/SP Llama does not support quantized weights")
        if vllm_config.lora_config is not None:
            raise RuntimeError("TP/SP Llama does not support LoRA")
        super().__init__(vllm_config=vllm_config, prefix=prefix, layer_type=layer_type)
        group = get_tp_group()
        first_layer = self.model.layers[0]
        hidden_size = self.config.hidden_size
        o_width = first_layer.self_attn.o_proj.input_size_per_partition
        o_eps = first_layer.post_attention_layernorm.variance_epsilon
        down_width = first_layer.mlp.down_proj.input_size_per_partition
        down_eps = self.model.layers[-1].input_layernorm.variance_epsilon
        shapes = {
            "o": TPSPShape(o_width, hidden_size, o_eps, True),
            "down": TPSPShape(down_width, hidden_size, down_eps, True),
        }
        self.tpsp_profile = TPSPProfileSession(
            shapes, group.world_size, group.device_group.group_name
        )
        self.model.tpsp_profile = self.tpsp_profile

    def _init_model(self, vllm_config, prefix="", layer_type=TPSPLlamaDecoderLayer):
        return TPSPLlamaModel(
            vllm_config=vllm_config, prefix=prefix, layer_type=layer_type
        )

    def close_tpsp(self) -> None:
        self.tpsp_profile.close()
        for layer in self.model.layers:
            for projection in (layer.self_attn.o_proj, layer.mlp.down_proj):
                if hasattr(projection, "_tpsp_transposed_weight"):
                    del projection._tpsp_transposed_weight
