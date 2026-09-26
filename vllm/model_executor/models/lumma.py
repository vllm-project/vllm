# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-only Lumma model compatible with HuggingFace weights.

Lumma is a Llama-style decoder with three differences:

- Factorized embeddings: tokens are embedded at `embedding_rank` and projected
  up to `hidden_size` (`model.embedding_proj`); the final hidden state is
  projected back down (`lm_head_proj`) before the tied LM head.
- Shared KV: there is no `v_proj`. The value is the key projection *before*
  RoPE (and before `k_norm`), so attention packs only Q and K.
- Optional per-head `q_norm` (or `q_norm` + `k_norm` via `qk_norm`).
"""

from collections.abc import Iterable
from itertools import islice

import torch
from torch import nn
from transformers import PreTrainedConfig

from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import get_pp_group, get_tensor_model_parallel_world_size
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    ReplicatedLinear,
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

from .interfaces import SupportsLoRA, SupportsPP
from .llama import LlamaMLP
from .utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    WeightsMapper,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)


class LummaAttention(nn.Module):
    def __init__(
        self,
        config: PreTrainedConfig,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        tp_size = get_tensor_model_parallel_world_size()
        self.head_dim = config.head_dim
        self.total_num_heads = config.num_attention_heads
        self.total_num_kv_heads = config.num_key_value_heads
        if self.total_num_kv_heads % tp_size != 0:
            raise ValueError(
                f"Lumma requires num_key_value_heads ({self.total_num_kv_heads}) "
                f"to be divisible by the tensor parallel size ({tp_size})."
            )
        self.num_heads = self.total_num_heads // tp_size
        self.num_kv_heads = self.total_num_kv_heads // tp_size
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim

        # No v_proj: the value reuses the key projection.
        self.qk_proj = MergedColumnParallelLinear(
            config.hidden_size,
            [
                self.total_num_heads * self.head_dim,
                self.total_num_kv_heads * self.head_dim,
            ],
            bias=config.attention_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.qk_proj",
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )
        self.rotary_emb = get_rope(
            self.head_dim,
            max_position=config.max_position_embeddings,
            rope_parameters=config.rope_parameters,
        )
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.head_dim**-0.5,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
        )

        qk_norm = getattr(config, "qk_norm", False)
        use_q_norm = qk_norm or getattr(config, "q_norm", False)
        self.q_norm = (
            RMSNorm(self.head_dim, eps=config.rms_norm_eps) if use_q_norm else None
        )
        self.k_norm = (
            RMSNorm(self.head_dim, eps=config.rms_norm_eps) if qk_norm else None
        )

    def _norm_heads(self, x: torch.Tensor, norm: RMSNorm) -> torch.Tensor:
        by_head = x.view(*x.shape[:-1], x.shape[-1] // self.head_dim, self.head_dim)
        return norm(by_head).view(x.shape)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qk, _ = self.qk_proj(hidden_states)
        q, v = qk.split([self.q_size, self.kv_size], dim=-1)
        k = v
        if self.q_norm is not None:
            q = self._norm_heads(q, self.q_norm)
        if self.k_norm is not None:
            k = self._norm_heads(k, self.k_norm)
        # RoPE rotates q and k in place, so rotate a copy of k to keep v intact.
        q, k = self.rotary_emb(positions, q, k.clone())
        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output


class LummaDecoderLayer(nn.Module):
    def __init__(self, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.self_attn = LummaAttention(
            config,
            cache_config=vllm_config.cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
        )
        self.mlp = LlamaMLP(
            hidden_size=config.hidden_size,
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
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions=positions, hidden_states=hidden_states)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


def _check_supported(config: PreTrainedConfig) -> None:
    if not getattr(config, "shared_kv", False):
        raise NotImplementedError("Lumma with shared_kv=False is not supported yet.")
    if getattr(config, "layer_sharing_repeats", 1) > 1:
        raise NotImplementedError(
            "Lumma with layer_sharing_repeats > 1 is not supported yet."
        )


@support_torch_compile
class LummaModel(nn.Module):
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_stacked={
            # weight_name: (param_name, shard_id)
            ".q_proj": (".qk_proj", 0),
            ".k_proj": (".qk_proj", 1),
            ".gate_proj": (".gate_up_proj", 0),
            ".up_proj": (".gate_up_proj", 1),
        }
    )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        _check_supported(config)
        self.config = config
        factorized = getattr(config, "factorized_embedding", False)
        embedding_dim = config.embedding_rank if factorized else config.hidden_size

        pp_group = get_pp_group()
        if pp_group.is_first_rank or (
            config.tie_word_embeddings and pp_group.is_last_rank
        ):
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                embedding_dim,
                quant_config=vllm_config.quant_config,
                prefix=f"{prefix}.embed_tokens",
            )
        else:
            self.embed_tokens = PPMissingLayer()
        self.embedding_proj: ReplicatedLinear | None = None
        if factorized and pp_group.is_first_rank:
            self.embedding_proj = ReplicatedLinear(
                config.embedding_rank,
                config.hidden_size,
                bias=False,
                quant_config=vllm_config.quant_config,
                prefix=f"{prefix}.embedding_proj",
            )

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: LummaDecoderLayer(vllm_config=vllm_config, prefix=prefix),
            prefix=f"{prefix}.layers",
        )
        if pp_group.is_last_rank:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()

        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], config.hidden_size
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        embeds = self.embed_tokens(input_ids)
        if self.embedding_proj is not None:
            embeds, _ = self.embedding_proj(embeds)
        return embeds

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
                hidden_states = self.embed_input_ids(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        for layer in islice(self.layers, self.start_layer, self.end_layer):
            hidden_states, residual = layer(positions, hidden_states, residual)

        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )

        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)


class LummaForCausalLM(nn.Module, SupportsLoRA, SupportsPP):
    hf_to_vllm_mapper = LummaModel.hf_to_vllm_mapper
    packed_modules_mapping = {
        "qk_proj": ["q_proj", "k_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }
    embedding_modules = {
        "embed_tokens": "input_embeddings",
        "lm_head": "output_embeddings",
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.model = LummaModel(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )

        factorized = getattr(config, "factorized_embedding", False)
        self.lm_head_proj: ReplicatedLinear | None = None
        if get_pp_group().is_last_rank:
            if factorized:
                self.lm_head_proj = ReplicatedLinear(
                    config.hidden_size,
                    config.embedding_rank,
                    bias=False,
                    quant_config=quant_config,
                    prefix=maybe_prefix(prefix, "lm_head_proj"),
                )
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.embedding_rank if factorized else config.hidden_size,
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
            if config.tie_word_embeddings:
                self.lm_head = self.lm_head.tie_weights(self.model.embed_tokens)
            self.logits_processor = LogitsProcessor(config.vocab_size)
        else:
            self.lm_head = PPMissingLayer()

        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        return self.model(input_ids, positions, intermediate_tensors, inputs_embeds)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        if self.lm_head_proj is not None:
            hidden_states, _ = self.lm_head_proj(hidden_states)
        return self.logits_processor(self.lm_head, hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)
