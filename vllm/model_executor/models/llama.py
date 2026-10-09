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
from vllm.logger import init_logger
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
    TPSPProjection,
    close_tpsp_projections,
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

logger = init_logger(__name__)


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

    def compute_down_proj_input(self, x):
        x, _ = self.gate_up_proj(x)
        return maybe_fused_act_quant(self.act_fn, x, self.down_proj)

    def forward(self, x):
        x = self.compute_down_proj_input(x)
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
        attn_output = self.compute_attention(positions, hidden_states)
        output, _ = self.o_proj(attn_output)
        return output

    def compute_attention(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        return self.attn(q, k, v)

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
        self.tpsp_projections: dict[str, TPSPProjection] | None = None

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        *,
        next_norm: RMSNorm | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        plans = self.tpsp_projections
        tpsp_enabled = plans is not None and any(
            projection.enabled for projection in plans.values()
        )
        tpsp_o = plans.get("o") if plans is not None else None
        tpsp_down = plans.get("down") if plans is not None else None
        if tpsp_enabled and (
            tpsp_o is None
            or tpsp_down is None
            or tpsp_o.profile is None
            or tpsp_down.profile is None
            or next_norm is None
        ):
            raise RuntimeError(
                "Active TP/SP Llama requires profiled projections and norm"
            )
        residual_is_sharded = residual is not None
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        elif not tpsp_enabled:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        if not tpsp_enabled:
            hidden_states = self.self_attn(
                positions=positions, hidden_states=hidden_states
            )
            hidden_states, residual = self.post_attention_layernorm(
                hidden_states, residual
            )
        else:
            assert tpsp_o is not None
            hidden_states = self.self_attn.compute_attention(positions, hidden_states)
            hidden_states, residual = self._project_and_normalize(
                hidden_states,
                self.self_attn.o_proj,
                residual,
                self.post_attention_layernorm,
                tpsp_o,
                residual_is_sharded,
            )
        if not tpsp_enabled:
            hidden_states = self.mlp(hidden_states)
        else:
            assert tpsp_down is not None
            hidden_states = self.mlp.compute_down_proj_input(hidden_states)
            hidden_states, residual = self._project_and_normalize(
                hidden_states,
                self.mlp.down_proj,
                residual,
                next_norm,
                tpsp_down,
                True,
            )
        return hidden_states, residual

    def _project_and_normalize(
        self,
        x,
        projection,
        residual,
        norm,
        tpsp: TPSPProjection,
        residual_is_sharded: bool,
    ):
        """Choose fused or regular projection using this projection's threshold."""
        profile = tpsp.profile
        if profile is None:
            raise RuntimeError("TP/SP projection requires startup profiling")
        backend = tpsp.backend
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

        use_fused_projection = select_sp_config(profile, x.size(0)) and (
            projection.bias is None
            or (backend is not None and backend.supports_projection_bias)
        )
        if use_fused_projection:
            return self._run_tpsp_fused(x, projection, norm, local_residual, tpsp)

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

    def _run_tpsp_fused(self, x, projection, norm, residual, tpsp: TPSPProjection):
        backend = tpsp.backend
        profile = tpsp.profile
        if backend is None or profile is None or profile.config is None:
            raise RuntimeError("TP/SP Llama profile has no enabled configuration")
        if x.dtype != torch.bfloat16 or projection.weight.dtype != torch.bfloat16:
            raise RuntimeError("TP/SP Llama requires BF16 activations and weights")
        weight = projection.weight
        key = (weight.data_ptr(), weight._version)
        cached = getattr(projection, "_tpsp_transposed_weight", None)
        if cached is None or cached[0] != key:
            cached = (key, weight.T.contiguous())
            projection._tpsp_transposed_weight = cached
        reduced, _, gathered = backend.fused_gemm_rs_norm_ag(
            x.contiguous(),
            cached[1],
            norm.weight,
            residual,
            norm.variance_epsilon,
            profile.config,
            projection_bias=projection.bias,
            context=tpsp.context,
        )
        return gathered, reduced

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
        self.tpsp_projections: nn.ModuleDict | None = None

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
        projections = self.tpsp_projections
        if projections is not None and any(
            isinstance(projection, TPSPProjection) and projection.profile is None
            for projection in projections.values()
        ):
            raise RuntimeError("TP/SP Llama requires worker startup profiling")
        tpsp_enabled = projections is not None and any(
            isinstance(projection, TPSPProjection) and projection.enabled
            for projection in projections.values()
        )
        if tpsp_enabled and (
            get_pp_group().world_size != 1
            or intermediate_tensors is not None
            or extra_layer_kwargs
        ):
            raise RuntimeError(
                "TP/SP Llama does not support pipeline or extra layer inputs"
            )

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
            if tpsp_enabled:
                assert projections is not None
                next_norm = (
                    self.layers[idx + 1].input_layernorm
                    if idx + 1 < self.end_layer
                    else self.norm
                )
                hidden_states, residual = layer(
                    positions,
                    hidden_states,
                    residual,
                    next_norm=next_norm,
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
                        aux_hidden_states,
                        idx + 1,
                        padded[: hidden_states.size(0)],
                        None,
                    )
            else:
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

        if not tpsp_enabled:
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
    tpsp_capable = True
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
        if vllm_config.model_config.enable_tpsp:
            self._init_tpsp(vllm_config)

    def _init_tpsp(self, vllm_config: VllmConfig) -> None:
        if not type(self).__dict__.get("tpsp_capable", False):
            logger.warning(
                "TPSP is not supported by %s; using regular forward", type(self)
            )
            return
        if (
            vllm_config.parallel_config.pipeline_parallel_size > 1
            or vllm_config.quant_config is not None
            or vllm_config.lora_config is not None
        ):
            logger.warning(
                "TPSP does not support PP, quantization or LoRA; using regular forward"
            )
            return
        group = get_tp_group()
        first_layer = self.model.layers[0]
        hidden_size = self.config.hidden_size
        self.model.tpsp_projections = nn.ModuleDict(
            {
                "o": TPSPProjection(
                    first_layer.self_attn.o_proj.input_size_per_partition,
                    hidden_size,
                    first_layer.post_attention_layernorm.variance_epsilon,
                    group.world_size,
                    group.device_group.group_name,
                ),
                "down": TPSPProjection(
                    first_layer.mlp.down_proj.input_size_per_partition,
                    hidden_size,
                    self.model.layers[-1].input_layernorm.variance_epsilon,
                    group.world_size,
                    group.device_group.group_name,
                ),
            }
        )
        # The model owns the modules; layers only keep references to them.
        plans = dict(self.model.tpsp_projections.items())
        for layer in self.model.layers:
            layer.tpsp_projections = plans
        if vllm_config.device_config.device_type == "cuda":
            self.model.do_not_compile = True

    def close_tpsp(self) -> None:
        if self.model.tpsp_projections is None:
            return
        close_tpsp_projections(self)
        for layer in self.model.layers:
            for projection in (layer.self_attn.o_proj, layer.mlp.down_proj):
                if hasattr(projection, "_tpsp_transposed_weight"):
                    del projection._tpsp_transposed_weight

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
