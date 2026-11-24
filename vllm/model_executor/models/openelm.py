# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Adapted from
# https://huggingface.co/apple/OpenELM-270M/blob/main/modeling_openelm.py
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
# LICENSE: https://huggingface.co/apple/OpenELM-270M/blob/main/LICENSE
"""Inference-only OpenELM model compatible with HuggingFace weights."""

from collections.abc import Iterable
from itertools import islice
from typing import cast

import torch
from torch import nn

from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import get_pp_group, get_tensor_model_parallel_world_size
from vllm.model_executor.layers.activation import SiluAndMul, get_act_fn
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
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
from vllm.transformers_utils.configs.openelm import OpenELMConfig

from .interfaces import SupportsPP
from .utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    extract_layer_index,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)

# OpenELMRMSNorm uses a fixed epsilon.
_RMS_NORM_EPS = 1e-6


class OpenELMFeedForwardNetwork(nn.Module):
    def __init__(
        self,
        config: OpenELMConfig,
        layer_idx: int,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        model_dim = config.model_dim
        intermediate_size = config.get_intermediate_size(layer_idx)
        self.ffn_with_glu = config.ffn_with_glu
        self.proj_1: MergedColumnParallelLinear | ColumnParallelLinear
        self.act_fn: nn.Module

        if self.ffn_with_glu:
            # proj_1 holds [gate, up] fused on disk.
            self.proj_1 = MergedColumnParallelLinear(
                model_dim,
                [intermediate_size] * 2,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.proj_1",
            )
            if config.activation_fn_name not in ("swish", "silu"):
                raise ValueError(
                    f"Unsupported activation for GLU FFN: "
                    f"{config.activation_fn_name}. Only swish/silu is supported."
                )
            self.act_fn = SiluAndMul()
        else:
            self.proj_1 = ColumnParallelLinear(
                model_dim,
                intermediate_size,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.proj_1",
            )
            act_name = config.activation_fn_name
            self.act_fn = get_act_fn("silu" if act_name == "swish" else act_name)

        self.proj_2 = RowParallelLinear(
            intermediate_size,
            model_dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.proj_2",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self.proj_1(x)
        x = self.act_fn(x)
        x, _ = self.proj_2(x)
        return x


class OpenELMMultiHeadCausalAttention(nn.Module):
    def __init__(
        self,
        config: OpenELMConfig,
        layer_idx: int,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        tp_size = get_tensor_model_parallel_world_size()
        self.head_dim = config.head_dim

        self.total_num_heads = config.num_query_heads[layer_idx]
        if self.total_num_heads % tp_size != 0:
            raise ValueError(
                f"Layer {layer_idx}: number of query heads ({self.total_num_heads}) "
                f"must be divisible by tensor parallel size ({tp_size})."
            )
        self.num_heads = self.total_num_heads // tp_size

        self.total_num_kv_heads = config.num_kv_heads[layer_idx]
        if self.total_num_kv_heads >= tp_size:
            # Partition the KV heads across the tensor parallel GPUs.
            kv_ok = self.total_num_kv_heads % tp_size == 0
        else:
            # Replicate the KV heads across the tensor parallel GPUs.
            kv_ok = tp_size % self.total_num_kv_heads == 0
        if not kv_ok:
            raise ValueError(
                f"Layer {layer_idx}: number of key/value heads "
                f"({self.total_num_kv_heads}) is incompatible with tensor "
                f"parallel size ({tp_size})."
            )
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)

        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5

        # The on-disk qkv_proj is laid out as [q_heads, k_heads, v_heads],
        # which matches QKVParallelLinear's fused layout.
        self.qkv_proj = QKVParallelLinear(
            config.model_dim,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.out_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            config.model_dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.out_proj",
        )

        self.rotary_emb = get_rope(
            self.head_dim,
            max_position=max(config.rope_max_length, config.max_context_length),
            is_neox_style=True,
            rope_parameters={
                "rope_type": "default",
                "rope_theta": config.rope_freq_constant,
            },
        )

        self.q_norm: RMSNorm | None = None
        self.k_norm: RMSNorm | None = None
        if config.normalize_qk_projections:
            self.q_norm = RMSNorm(self.head_dim, eps=_RMS_NORM_EPS)
            self.k_norm = RMSNorm(self.head_dim, eps=_RMS_NORM_EPS)

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
        if self.q_norm is not None and self.k_norm is not None:
            q_by_head = q.view(*q.shape[:-1], self.num_heads, self.head_dim)
            q = self.q_norm(q_by_head).view(q.shape)
            k_by_head = k.view(*k.shape[:-1], self.num_kv_heads, self.head_dim)
            k = self.k_norm(k_by_head).view(k.shape)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v)
        output, _ = self.out_proj(attn_output)
        return output


class OpenELMDecoderLayer(nn.Module):
    def __init__(
        self,
        config: OpenELMConfig,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        layer_idx = extract_layer_index(prefix)
        self.attn = OpenELMMultiHeadCausalAttention(
            config,
            layer_idx,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
        )
        self.ffn = OpenELMFeedForwardNetwork(
            config,
            layer_idx,
            quant_config=quant_config,
            prefix=f"{prefix}.ffn",
        )
        self.attn_norm = RMSNorm(config.model_dim, eps=_RMS_NORM_EPS)
        self.ffn_norm = RMSNorm(config.model_dim, eps=_RMS_NORM_EPS)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.attn_norm(hidden_states)
        else:
            hidden_states, residual = self.attn_norm(hidden_states, residual)
        hidden_states = self.attn(positions=positions, hidden_states=hidden_states)

        hidden_states, residual = self.ffn_norm(hidden_states, residual)
        hidden_states = self.ffn(hidden_states)
        return hidden_states, residual


@support_torch_compile
class OpenELMModel(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = cast(OpenELMConfig, vllm_config.model_config.hf_config)
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config

        self.config = config
        self.vocab_size = config.vocab_size

        self.token_embeddings: VocabParallelEmbedding | PPMissingLayer
        self.norm: RMSNorm | PPMissingLayer
        if get_pp_group().is_first_rank or (
            config.share_input_output_layers and get_pp_group().is_last_rank
        ):
            self.token_embeddings = VocabParallelEmbedding(
                config.vocab_size,
                config.model_dim,
                quant_config=quant_config,
                prefix=f"{prefix}.token_embeddings",
            )
        else:
            self.token_embeddings = PPMissingLayer()

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_transformer_layers,
            lambda prefix: OpenELMDecoderLayer(
                config, cache_config, quant_config, prefix=prefix
            ),
            prefix=f"{prefix}.layers",
        )

        if get_pp_group().is_last_rank:
            self.norm = RMSNorm(config.model_dim, eps=_RMS_NORM_EPS)
        else:
            self.norm = PPMissingLayer()

        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], config.model_dim
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.token_embeddings(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                assert input_ids is not None
                hidden_states = self.embed_input_ids(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        for layer in islice(self.layers, self.start_layer, self.end_layer):
            hidden_states, residual = layer(positions, hidden_states, residual)

        if not get_pp_group().is_last_rank:
            assert residual is not None
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )

        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class OpenELMForCausalLM(nn.Module, SupportsPP):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = cast(OpenELMConfig, vllm_config.model_config.hf_config)
        quant_config = vllm_config.quant_config
        self.config = config

        self.transformer = OpenELMModel(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "transformer")
        )

        self.lm_head: ParallelLMHead | PPMissingLayer
        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.model_dim,
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
            if config.share_input_output_layers:
                # OpenELMModel keeps the embeddings on the last PP rank for this.
                embeddings = self.transformer.token_embeddings
                assert isinstance(embeddings, VocabParallelEmbedding)
                self.lm_head = self.lm_head.tie_weights(embeddings)
            self.logits_processor = LogitsProcessor(config.vocab_size)
        else:
            self.lm_head = PPMissingLayer()

        self.make_empty_intermediate_tensors = (
            self.transformer.make_empty_intermediate_tensors
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.transformer.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        return self.transformer(
            input_ids, positions, intermediate_tensors, inputs_embeds
        )

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)
