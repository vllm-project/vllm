# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-only PLaMo3 model."""

from collections.abc import Iterable
from itertools import islice
from typing import Any

import torch
from torch import nn
from transformers import PretrainedConfig

from vllm.attention.layer import Attention
from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.distributed.parallel_state import get_pp_group
from vllm.model_executor.layers.activation import SiluAndMul
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
    DEFAULT_VOCAB_PADDING_SIZE,
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import (
    LoaderFunction,
    composed_weight_loader,
    default_weight_loader,
)
from vllm.model_executor.models.interfaces import SupportsLoRA, SupportsPP
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    extract_layer_index,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)
from vllm.model_executor.utils import set_weight_attrs
from vllm.sequence import IntermediateTensors


# Only used for type hinting.
class Plamo3Config(PretrainedConfig):  # type: ignore
    model_type: str = "plamo3"

    hidden_size: int
    num_hidden_layers: int
    rms_norm_eps: float
    # Attention
    num_attention_heads: int
    head_dim: int
    num_key_value_heads: int
    # vllm rename `sliding_window` attr to `interleaved_sliding_window`
    # if `sliding_window` is list
    interleaved_sliding_window: list[int | None]
    sliding_window_pattern: int
    rope_parameters: dict[str, Any]
    rope_local_theta: int
    # MLP
    intermediate_size: int
    # Tokenizer
    vocab_size: int


def rms_norm_weight_loader(offset: float) -> LoaderFunction:
    return composed_weight_loader(
        default_weight_loader,
        lambda x: x + offset,
    )


class DenseMLP(nn.Module):
    def __init__(
        self,
        config: Plamo3Config,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_up_proj = MergedColumnParallelLinear(
            self.hidden_size,
            [self.intermediate_size] * 2,
            bias=False,
            prefix=f"{prefix}.gate_up_proj",
            quant_config=quant_config,
            return_bias=False,
        )
        self.act = SiluAndMul()
        self.down_proj = RowParallelLinear(
            self.intermediate_size,
            self.hidden_size,
            bias=False,
            prefix=f"{prefix}.down_proj",
            quant_config=quant_config,
            return_bias=False,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        h = self.gate_up_proj(hidden_states)
        h = self.act(h)
        return self.down_proj(h)


class Plamo3AttentionMixer(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "", **kwargs) -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config

        self.hidden_size = config.hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = config.num_attention_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = config.num_key_value_heads
        if self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = config.head_dim
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5

        self.qkv_proj = QKVParallelLinear(
            config.hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        layer_idx = extract_layer_index(prefix)
        layer_type = config.layer_types[layer_idx]
        is_sliding = layer_type == "sliding_attention"

        # Initialize the rotary embedding.
        if layer_type in config.rope_parameters:
            # Transformers v5 rope config.
            rope_parameters = config.rope_parameters[layer_type]
        else:
            # Transformers v4 rope config.
            # Global attention. Use the values in config.json.
            rope_parameters = config.rope_parameters
            # Local attention. Override the values in config.json.
            if is_sliding:
                rope_parameters = dict(
                    rope_type="default", rope_theta=config.rope_local_theta
                )
        max_position = config.max_position_embeddings
        if hasattr(vllm_config.model_config, "max_model_len") and isinstance(
            vllm_config.model_config.max_model_len, int
        ):
            max_position = min(max_position, vllm_config.model_config.max_model_len)

        self.rotary_emb = get_rope(
            self.head_dim,
            max_position=max_position,
            rope_parameters=rope_parameters,
        )
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        set_weight_attrs(
            self.q_norm.weight, {"weight_loader": rms_norm_weight_loader(offset=1.0)}
        )
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        set_weight_attrs(
            self.k_norm.weight, {"weight_loader": rms_norm_weight_loader(offset=1.0)}
        )
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=vllm_config.cache_config,
            per_layer_sliding_window=config.interleaved_sliding_window[layer_idx],
            prefix=f"{prefix}.attn",
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        **kwargs: Any,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        q_shape = q.shape
        q = q.reshape(q_shape[:-1] + (q_shape[-1] // self.head_dim, self.head_dim))
        q = self.q_norm.forward_native(q).reshape(q_shape)
        k_shape = k.shape
        k = k.reshape(k_shape[:-1] + (k_shape[-1] // self.head_dim, self.head_dim))
        k = self.k_norm.forward_native(k).reshape(k_shape)

        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output


class Plamo3DecoderLayer(nn.Module):
    def __init__(
        self, vllm_config: VllmConfig, prefix: str = "", **kwargs: Any
    ) -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config

        self.mixer = Plamo3AttentionMixer(
            vllm_config=vllm_config,
            prefix=f"{prefix}.mixer",
        )

        self.mlp = DenseMLP(
            config=config, quant_config=quant_config, prefix=f"{prefix}.mlp"
        )
        self.pre_mixer_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        set_weight_attrs(
            self.pre_mixer_norm.weight,
            {"weight_loader": rms_norm_weight_loader(offset=1.0)},
        )
        self.post_mixer_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        set_weight_attrs(
            self.post_mixer_norm.weight,
            {"weight_loader": rms_norm_weight_loader(offset=1.0 / 5)},
        )
        self.pre_mlp_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        set_weight_attrs(
            self.pre_mlp_norm.weight,
            {"weight_loader": rms_norm_weight_loader(offset=1.0)},
        )
        self.post_mlp_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        set_weight_attrs(
            self.post_mlp_norm.weight,
            {"weight_loader": rms_norm_weight_loader(offset=1.0 / (5**1.5))},
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.pre_mixer_norm(hidden_states)
        else:
            hidden_states, residual = self.pre_mixer_norm(hidden_states, residual)

        hidden_states = self.mixer(
            positions=positions, hidden_states=hidden_states, residual=residual
        )
        hidden_states = self.post_mixer_norm(hidden_states)
        # Fully Connected
        hidden_states, residual = self.pre_mlp_norm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_mlp_norm(hidden_states)
        return hidden_states, residual


class Plamo3Decoder(torch.nn.Module):
    def __init__(self, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        num_hidden_layers = vllm_config.model_config.hf_config.num_hidden_layers

        self.start_layer, self.end_layer, self.layers = make_layers(
            num_hidden_layers,
            lambda prefix: Plamo3DecoderLayer(vllm_config, prefix=prefix),
            prefix=f"{prefix}.layers",
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        for layer in islice(self.layers, self.start_layer, self.end_layer):
            hidden_states, residual = layer(
                positions=positions,
                hidden_states=hidden_states,
                residual=residual,
            )
        return hidden_states, residual


@support_torch_compile
class Plamo3Model(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config

        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.org_vocab_size = config.vocab_size

        self.embed_tokens = VocabParallelEmbedding(
            self.vocab_size,
            config.hidden_size,
            org_num_embeddings=config.vocab_size,
            prefix=f"{prefix}.embed_tokens",
        )
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], config.hidden_size
        )
        self.layers = Plamo3Decoder(vllm_config, prefix=f"{prefix}.layers")
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        set_weight_attrs(
            self.norm.weight,
            {"weight_loader": rms_norm_weight_loader(offset=1.0)},
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
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

        hidden_states, residual = self.layers(
            positions=positions, hidden_states=hidden_states, residual=residual
        )
        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class Plamo3ForCausalLM(nn.Module, SupportsLoRA, SupportsPP):
    packed_modules_mapping = {
        "qkv_proj": ["qkv_proj"],
        "gate_up_proj": ["gate_up_proj"],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        self.config = vllm_config.model_config.hf_config
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.scheduler_config = vllm_config.scheduler_config

        self.model = Plamo3Model(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )

        self.vocab_size = self.config.vocab_size
        self.unpadded_vocab_size = self.config.vocab_size

        num_embeddings = ((self.vocab_size + 15) // 16) * 16
        self.lm_head = ParallelLMHead(
            num_embeddings,
            self.config.hidden_size,
            org_num_embeddings=self.config.vocab_size,
            padding_size=DEFAULT_VOCAB_PADDING_SIZE,
            prefix=f"{prefix}.lm_head",
        )
        if self.config.tie_word_embeddings:
            self.lm_head = self.lm_head.tie_weights(self.model.embed_tokens)

        self.logits_processor = LogitsProcessor(
            self.unpadded_vocab_size, self.config.vocab_size
        )
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
    ) -> torch.Tensor:
        hidden_states = self.model(
            input_ids, positions, intermediate_tensors, inputs_embeds
        )
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        logits = self.logits_processor(self.lm_head, hidden_states)
        return logits

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=(["lm_head."] if self.config.tie_word_embeddings else None),
        )
        return loader.load_weights(weights)
