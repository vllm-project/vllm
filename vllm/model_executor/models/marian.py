# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MarianMT (Helsinki-NLP/opus-mt-*) encoder-decoder model for vLLM V1.

The source sentence is fed to the encoder as a single "text" modality via a
``EncDecMultiModalProcessor`` (the same pattern Whisper uses for audio), while
the decoder runs as a normal V1 encoder-decoder.

Architecture notes:
  * Positional embeddings are sinusoidal-static (``static_position_embeddings``)
    and stored in the checkpoint as ``embed_positions.weight``, looked up by
    absolute position (no padding offset).
  * No ``layernorm_embedding`` (``normalize_embedding=False``) and no top-level
    ``layer_norm``.
  * Activation is ``swish`` (SiLU), from ``config.activation_function``.
  * Encoder and decoder share ``model.shared`` embeddings and the LM head is
    tied to them (``share_encoder_decoder_embeddings`` / ``tie_word_embeddings``).
  * HF checkpoint keys already carry the ``model.`` prefix, so the weights
    mapper is an identity map.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import torch
from torch import nn
from transformers import MarianConfig
from transformers.utils import logging

from vllm.config import CacheConfig, VllmConfig
from vllm.config.lora import LoRAConfig
from vllm.config.multimodal import BaseDummyOptions
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.inputs import MultiModalDataDict
from vllm.model_executor.layers.activation import get_act_fn
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.attention.cross_attention import CrossAttention
from vllm.model_executor.layers.attention.mm_encoder_attention import (
    MMEncoderAttention,
)
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (
    MultiModalFieldConfig,
    MultiModalKwargsItems,
)
from vllm.multimodal.parse import (
    ModalityData,
    ModalityDataItems,
    ModalityDataParser,
    MultiModalDataItems,
    MultiModalDataParser,
    ProcessorBatchItems,
)
from vllm.multimodal.processing import (
    BaseDummyInputsBuilder,
    BaseProcessingInfo,
    EncDecMultiModalProcessor,
    PromptUpdate,
)
from vllm.sequence import IntermediateTensors
from vllm.utils.collection_utils import is_list_of
from vllm.v1.attention.backend import AttentionType

from .interfaces import (
    MultiModalEmbeddings,
    SupportsMultiModal,
    SupportsQuant,
)
from .utils import (
    AutoWeightsLoader,
    WeightsMapper,
    cast_overflow_tensors,
    maybe_prefix,
)

logger = logging.get_logger(__name__)


class MarianSinusoidalPositionalEmbedding(VocabParallelEmbedding):
    """Static sinusoidal positional embeddings.

    Marian stores precomputed sinusoids in ``embed_positions.weight`` and looks
    them up by absolute position (no padding offset).
    """

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        return super().forward(positions)


class MarianScaledWordEmbedding(VocabParallelEmbedding):
    """Word embedding scaled by ``sqrt(d_model)`` (``scale_embedding=True``)."""

    def __init__(
        self, num_embeddings: int, embedding_dim: int, embed_scale: float = 1.0
    ):
        super().__init__(num_embeddings, embedding_dim)
        self.embed_scale = embed_scale

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return super().forward(input_ids) * self.embed_scale


class MarianEncoderAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        bias: bool = True,
        config: MarianConfig | None = None,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.d_model = config.d_model
        self.embed_dim = embed_dim
        self.total_num_heads = num_heads
        self.total_num_kv_heads = self.total_num_heads
        self.head_dim = embed_dim // num_heads
        self.config = config

        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads "
                f"(got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        self.scaling = self.head_dim**-0.5

        self.qkv_proj = QKVParallelLinear(
            self.d_model,
            self.d_model // self.total_num_heads,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=bias,
            quant_config=quant_config,
        )

        self.out_proj = RowParallelLinear(
            embed_dim,
            embed_dim,
            bias=bias,
            quant_config=quant_config,
        )

        tp_world_size = get_tensor_model_parallel_world_size()
        assert self.total_num_heads % tp_world_size == 0
        self.num_heads = self.total_num_heads // tp_world_size

        if self.total_num_kv_heads >= tp_world_size:
            assert self.total_num_kv_heads % tp_world_size == 0
        else:
            assert tp_world_size % self.total_num_kv_heads == 0
        self.num_kv_heads = self.num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim

        self.attn = MMEncoderAttention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        is_2d = q.dim() == 2
        if is_2d:
            q = q.unsqueeze(0)
            k = k.unsqueeze(0)
            v = v.unsqueeze(0)

        attn_output = self.attn(q, k, v)

        output, _ = self.out_proj(attn_output)
        if is_2d:
            output = output.squeeze(0)
        return output


class MarianDecoderSelfAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        bias: bool = True,
        config: MarianConfig | None = None,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.d_model = config.d_model
        self.embed_dim = embed_dim
        self.total_num_heads = num_heads
        self.total_num_kv_heads = self.total_num_heads
        self.head_dim = embed_dim // num_heads
        self.config = config

        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads "
                f"(got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        self.scaling = self.head_dim**-0.5

        self.qkv_proj = QKVParallelLinear(
            self.d_model,
            self.d_model // self.total_num_heads,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=bias,
            quant_config=quant_config,
        )

        self.out_proj = RowParallelLinear(
            embed_dim,
            embed_dim,
            bias=bias,
            quant_config=quant_config,
        )

        tp_world_size = get_tensor_model_parallel_world_size()
        assert self.total_num_heads % tp_world_size == 0
        self.num_heads = self.total_num_heads // tp_world_size

        if self.total_num_kv_heads >= tp_world_size:
            assert self.total_num_kv_heads % tp_world_size == 0
        else:
            assert tp_world_size % self.total_num_kv_heads == 0
        self.num_kv_heads = self.num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim

        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
            attn_type=AttentionType.DECODER,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        attn_output = self.attn(q, k, v)

        output, _ = self.out_proj(attn_output)
        return output


class MarianCrossAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        bias: bool = True,
        config: MarianConfig | None = None,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.prefix = prefix
        self.d_model = config.d_model
        self.embed_dim = embed_dim
        self.total_num_heads = num_heads
        self.total_num_kv_heads = self.total_num_heads
        self.head_dim = embed_dim // num_heads
        self.config = config

        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads "
                f"(got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        self.scaling = self.head_dim**-0.5
        self.kv_size = self.total_num_kv_heads * self.head_dim

        # Q projects decoder hidden states.
        self.q_proj = ColumnParallelLinear(
            input_size=embed_dim,
            output_size=embed_dim,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.q_proj",
        )

        # KV projects encoder hidden states; total_num_heads=0 avoids an
        # unused Q shard inside the fused QKV layer.
        self.kv_proj = QKVParallelLinear(
            hidden_size=embed_dim,
            head_size=self.head_dim,
            total_num_heads=0,
            total_num_kv_heads=self.total_num_kv_heads,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.kv_proj",
        )

        self.out_proj = RowParallelLinear(
            embed_dim,
            embed_dim,
            bias=bias,
            quant_config=quant_config,
        )

        tp_world_size = get_tensor_model_parallel_world_size()
        assert self.total_num_heads % tp_world_size == 0
        self.num_heads = self.total_num_heads // tp_world_size

        if self.total_num_kv_heads >= tp_world_size:
            assert self.total_num_kv_heads % tp_world_size == 0
        else:
            assert tp_world_size % self.total_num_kv_heads == 0
        self.num_kv_heads = self.num_heads  # No GQA in Marian
        self.attn = CrossAttention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
            attn_type=AttentionType.ENCODER_DECODER,
        )

    def forward(
        self,
        decoder_hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
    ) -> torch.Tensor:
        q, _ = self.q_proj(decoder_hidden_states)

        # Encoder K/V are computed once during prefill; afterwards they live in
        # the cross-attention KV cache.
        if encoder_hidden_states is not None:
            kv, _ = self.kv_proj(encoder_hidden_states)
            k, v = kv.split([self.kv_size, self.kv_size], dim=-1)
        else:
            k = v = None

        attn_output = self.attn(q, k, v)
        output, _ = self.out_proj(attn_output)
        return output


class MarianEncoderLayer(nn.Module):
    def __init__(
        self,
        config: MarianConfig,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.embed_dim = config.d_model

        self.self_attn = MarianEncoderAttention(
            embed_dim=self.embed_dim,
            num_heads=config.encoder_attention_heads,
            config=config,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
        )
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.activation_fn = get_act_fn(config.activation_function)

        ffn_has_bias = True
        self.fc1 = ColumnParallelLinear(
            self.embed_dim,
            config.encoder_ffn_dim,
            bias=ffn_has_bias,
            quant_config=quant_config,
        )
        self.fc2 = RowParallelLinear(
            config.encoder_ffn_dim,
            self.embed_dim,
            bias=ffn_has_bias,
            quant_config=quant_config,
        )

        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.self_attn(hidden_states=hidden_states)

        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        residual = hidden_states
        fc1_out, _ = self.fc1(hidden_states)
        hidden_states = self.activation_fn(fc1_out)

        hidden_states, _ = self.fc2(hidden_states)

        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        if hidden_states.dtype == torch.float16:
            hidden_states = cast_overflow_tensors(hidden_states)

        return hidden_states


class MarianDecoderLayer(nn.Module):
    def __init__(
        self,
        config: MarianConfig,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.embed_dim = config.d_model

        self.self_attn = MarianDecoderSelfAttention(
            embed_dim=self.embed_dim,
            num_heads=config.decoder_attention_heads,
            config=config,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
        )
        self.activation_fn = get_act_fn(config.activation_function)

        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        # Named "encoder_attn" to match the pretrained weights (this is the
        # decoder's cross-attention over encoder outputs).
        self.encoder_attn = MarianCrossAttention(
            self.embed_dim,
            config.decoder_attention_heads,
            config=config,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.encoder_attn",
        )
        self.encoder_attn_layer_norm = nn.LayerNorm(self.embed_dim)

        ffn_has_bias = True
        self.fc1 = ColumnParallelLinear(
            self.embed_dim,
            config.decoder_ffn_dim,
            bias=ffn_has_bias,
            quant_config=quant_config,
        )
        self.fc2 = RowParallelLinear(
            config.decoder_ffn_dim,
            self.embed_dim,
            bias=ffn_has_bias,
            quant_config=quant_config,
        )

        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(
        self,
        decoder_hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
    ) -> torch.Tensor:
        residual = decoder_hidden_states

        # Self attention
        hidden_states = self.self_attn(hidden_states=decoder_hidden_states)

        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        # Cross attention
        residual = hidden_states
        hidden_states = self.encoder_attn(
            decoder_hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
        )
        hidden_states = residual + hidden_states
        hidden_states = self.encoder_attn_layer_norm(hidden_states)

        # Feed-forward
        residual = hidden_states
        fc1_out, _ = self.fc1(hidden_states)
        hidden_states = self.activation_fn(fc1_out)

        hidden_states, _ = self.fc2(hidden_states)

        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        return hidden_states


class MarianEncoder(nn.Module):
    """Transformer encoder of ``config.encoder_layers`` [`MarianEncoderLayer`]."""

    def __init__(
        self,
        config: MarianConfig,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        lora_config: LoRAConfig | None = None,
        embed_tokens: nn.Embedding | None = None,
        prefix: str = "",
    ):
        super().__init__()

        self.cache_config = cache_config
        self.quant_config = quant_config
        self.lora_config = lora_config
        embed_dim = config.d_model
        self.max_source_positions = config.max_position_embeddings
        embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0

        self.embed_tokens = MarianScaledWordEmbedding(
            config.vocab_size, embed_dim, embed_scale=embed_scale
        )
        if embed_tokens is not None:
            self.embed_tokens.weight = embed_tokens.weight

        # Static sinusoidal positions, looked up by absolute position.
        self.embed_positions = MarianSinusoidalPositionalEmbedding(
            config.max_position_embeddings,
            embed_dim,
        )
        self.layers = nn.ModuleList(
            [
                MarianEncoderLayer(
                    config,
                    cache_config,
                    quant_config,
                    prefix=f"{prefix}.layers.{layer_idx}",
                )
                for layer_idx in range(config.encoder_layers)
            ]
        )
        # Marian has no layernorm_embedding (normalize_embedding=False).

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        embed_pos = self.embed_positions(positions)
        embed_pos = embed_pos.to(inputs_embeds.device)

        hidden_states = inputs_embeds + embed_pos

        for encoder_layer in self.layers:
            hidden_states = encoder_layer(hidden_states=hidden_states)
        return hidden_states


class MarianDecoder(nn.Module):
    """Transformer decoder of ``config.decoder_layers`` [`MarianDecoderLayer`]."""

    def __init__(
        self,
        config: MarianConfig,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        lora_config: LoRAConfig | None = None,
        embed_tokens: nn.Embedding | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.cache_config = cache_config
        self.quant_config = quant_config
        self.lora_config = lora_config
        self.max_target_positions = config.max_position_embeddings
        embed_scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0

        self.embed_tokens = MarianScaledWordEmbedding(
            config.vocab_size, config.d_model, embed_scale=embed_scale
        )
        if embed_tokens is not None:
            self.embed_tokens.weight = embed_tokens.weight

        self.embed_positions = MarianSinusoidalPositionalEmbedding(
            config.max_position_embeddings,
            config.d_model,
        )
        self.layers = nn.ModuleList(
            [
                MarianDecoderLayer(
                    config,
                    cache_config,
                    quant_config,
                    prefix=f"{prefix}.layers.{layer_idx}",
                )
                for layer_idx in range(config.decoder_layers)
            ]
        )
        # Marian has no layernorm_embedding (normalize_embedding=False).

    def forward(
        self,
        decoder_input_ids: torch.Tensor,
        decoder_positions: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if inputs_embeds is None:
            assert decoder_input_ids is not None
            inputs_embeds = self.embed_input_ids(decoder_input_ids)

        embed_pos = self.embed_positions(decoder_positions)
        embed_pos = embed_pos.to(inputs_embeds.device)

        hidden_states = inputs_embeds + embed_pos

        for decoder_layer in self.layers:
            hidden_states = decoder_layer(
                decoder_hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
            )
        return hidden_states

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: MultiModalEmbeddings | None = None,
        is_multimodal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.embed_tokens(input_ids)


class MarianModel(nn.Module, SupportsQuant):
    _tied_weights_keys = [
        "encoder.embed_tokens.weight",
        "decoder.embed_tokens.weight",
    ]

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        lora_config = vllm_config.lora_config

        self.config = config

        lora_vocab = (
            (lora_config.lora_extra_vocab_size * (lora_config.max_loras or 1))
            if lora_config
            else 0
        )
        self.vocab_size = config.vocab_size + lora_vocab
        self.org_vocab_size = config.vocab_size

        self.encoder = MarianEncoder(
            config, cache_config, quant_config=quant_config, prefix=f"{prefix}.encoder"
        )
        self.decoder = MarianDecoder(
            config, cache_config, quant_config=quant_config, prefix=f"{prefix}.decoder"
        )

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None,
        encoder_outputs: torch.Tensor | None,
    ) -> torch.Tensor:
        decoder_outputs = self.decoder(
            decoder_input_ids=input_ids,
            decoder_positions=positions,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_outputs,
        )
        return decoder_outputs

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
        ]
        # Cross-attention fuses only K/V, keeping Q separate.
        cross_attn_stacked_params_mapping = [
            ("kv_proj", "k_proj", "k"),
            ("kv_proj", "v_proj", "v"),
        ]

        other_weights = []
        loaded_stacked_params = []
        model_params_dict = dict(self.named_parameters())

        for name, loaded_weight in weights:
            for param_name, weight_name, shard_id in cross_attn_stacked_params_mapping:
                if weight_name not in name or "encoder_attn" not in name:
                    continue
                name = name.replace(weight_name, param_name)
                if name not in model_params_dict:
                    continue
                param = model_params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                loaded_stacked_params.append(name)
                break
            else:
                for param_name, weight_name, shard_id in stacked_params_mapping:
                    if weight_name not in name or "encoder_attn" in name:
                        # Skip cross-attn q_proj here; it loads normally.
                        continue
                    name = name.replace(weight_name, param_name)
                    if name not in model_params_dict:
                        continue
                    param = model_params_dict[name]
                    weight_loader = param.weight_loader
                    weight_loader(param, loaded_weight, shard_id)
                    loaded_stacked_params.append(name)
                    break
                else:
                    if name in model_params_dict:
                        other_weights.append((name, loaded_weight))

        loader = AutoWeightsLoader(self)
        loaded_params = loader.load_weights(other_weights)
        loaded_params.update(loaded_stacked_params)
        return loaded_params


class MarianProcessingInfo(BaseProcessingInfo):
    """Processing information for MarianMT encoder-decoder models."""

    def get_hf_config(self) -> MarianConfig:
        return self.ctx.get_hf_config(MarianConfig)

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        # Marian's encoder input (the source sentence) is treated as a single
        # "text" modality; the decoder side is plain text.
        return {"text": 1}

    def get_mm_max_tokens_per_item(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> Mapping[str, int] | None:
        # The encoder handles up to max_position_embeddings source tokens;
        # return it directly to avoid complex profiling.
        config = self.get_hf_config()
        return {"text": config.max_position_embeddings}

    def get_data_parser(self) -> MultiModalDataParser:
        return TextDataParser()

    def get_default_tok_params(self):
        # Decoder-prompt tokenization: add_special_tokens=False keeps an empty
        # decoder prompt empty, so the runtime prepends only
        # decoder_start_token_id (= pad). The encoder source is tokenized
        # separately (add_special_tokens=True) and does not use these params.
        return super().get_default_tok_params().with_kwargs(add_special_tokens=False)


class MarianDummyInputsBuilder(BaseDummyInputsBuilder[MarianProcessingInfo]):
    """Builds dummy inputs for profiling MarianMT models."""

    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        # The decoder prompt is separate from the encoder; minimal dummy text.
        return ""

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions] | None = None,
    ) -> MultiModalDataDict:
        num_texts = mm_counts.get("text", 0)
        if num_texts == 0:
            return {}
        # Simple repeated words of the profiled length for the encoder side.
        # Leave room for the trailing eos the tokenizer appends
        # (add_special_tokens=True), so the total token count -- and hence the
        # max encoder position -- stays within max_position_embeddings.
        num_words = max(seq_len - 1, 1)
        dummy_text = " ".join(["word"] * num_words)
        return {"text": dummy_text}


class TextProcessorItems(ProcessorBatchItems[str]):
    """Data items for the text modality (Marian encoder input is source text)."""

    def __init__(self, data) -> None:
        if data is None:
            data = [""]
        elif isinstance(data, str):
            data = [data]
        super().__init__(data, "text")


class TextDataParser(MultiModalDataParser):
    def __init__(self):
        super().__init__()

    def _parse_text_data(
        self,
        data: ModalityData[str],
    ) -> ModalityDataItems[Any, Any] | None:
        if data is None or not len(data):
            return TextProcessorItems(None)

        if isinstance(data, str) or is_list_of(data, str):
            return TextProcessorItems(data)
        raise TypeError(
            f"Text data must be a string or list of strings, got {type(data)}"
        )

    def _get_subparsers(self) -> Mapping[str, ModalityDataParser]:
        return {
            "text": self._parse_text_data,
        }


class MarianMultiModalProcessor(EncDecMultiModalProcessor[MarianProcessingInfo]):
    """Multimodal processor for MarianMT (source text as a "text" modality)."""

    def create_encoder_prompt(
        self,
        prompt: str | list[int],
        mm_items: MultiModalDataItems,
    ) -> str | list[int]:
        # Single placeholder token; expanded to the real source length by
        # _get_prompt_updates. The actual token ids come from the mm kwargs.
        return [0]

    def create_decoder_prompt(
        self,
        prompt: str | list[int],
        mm_items: MultiModalDataItems,
    ) -> str | list[int]:
        # MarianMT is bilingual: the target language is implied by the model, so
        # a decoder-prompt string (e.g. the target_language forwarded by
        # /v1/translations) carries no decoder tokens -- return an empty prompt
        # and let the runtime prepend only decoder_start_token_id. An explicit
        # token-id list (teacher forcing) is a real sequence, passed unchanged.
        if isinstance(prompt, str):
            return []
        return prompt

    def _apply_hf_processor_main(
        self,
        mm_items: MultiModalDataItems,
        hf_kwargs: Mapping[str, object],
    ):
        """Tokenize the source text directly.

        Marian has no HF Processor (only a tokenizer), so the base path -- which
        requires a real ``ProcessorMixin`` -- cannot be used. The decoder prompt
        is built separately via ``create_decoder_prompt``, so this emits only
        ``encoder_input_ids``.

        Uses ``add_special_tokens=True`` so the trailing </s> (eos) Marian's
        encoder requires is appended, matching HuggingFace and consistent with
        ``_get_prompt_updates``.
        """
        from transformers.feature_extraction_utils import BatchFeature

        num_text_items = mm_items.get_count("text", strict=False)
        if num_text_items == 0:
            return BatchFeature({})

        tokenizer = self.info.get_tokenizer()
        text_items = mm_items.get_items("text", TextProcessorItems)
        encoder_text = text_items.get(0)

        encoder_tokenized = tokenizer(
            encoder_text,
            return_tensors="pt",
            add_special_tokens=True,
        )
        return BatchFeature({"encoder_input_ids": encoder_tokenized["input_ids"]})

    def _get_mm_fields_config(
        self,
        hf_inputs,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return dict(encoder_input_ids=MultiModalFieldConfig.batched("text"))

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        """Expand the single ``[0]`` encoder placeholder to N placeholders,
        where N is the tokenized source length (``add_special_tokens=True``,
        matching ``_apply_hf_processor_main`` -- includes the trailing eos)."""
        from vllm.multimodal.processing import PromptReplacement

        num_text_items = mm_items.get_count("text", strict=False)
        if num_text_items == 0:
            return []

        text_items = mm_items.get_items("text", TextProcessorItems)
        tokenizer = self.info.get_tokenizer()
        text = text_items.get(0)
        num_tokens = len(tokenizer.encode(text, add_special_tokens=True))

        return [
            PromptReplacement(
                modality="text",
                target=[0],
                replacement=[0] * num_tokens,
            )
        ]

    def build_data_parser(self) -> MultiModalDataParser:
        return TextDataParser()


@MULTIMODAL_REGISTRY.register_processor(
    MarianMultiModalProcessor,
    info=MarianProcessingInfo,
    dummy_inputs=MarianDummyInputsBuilder,
)
class MarianMTModel(nn.Module, SupportsQuant, SupportsMultiModal):
    """MarianMT for conditional generation (encoder-decoder + tied LM head).

    The source sentence is fed as a "text" modality through
    ``MarianMultiModalProcessor`` (registered above). ``load_weights`` consumes a
    real ``Helsinki-NLP/opus-mt-*`` checkpoint with no missing or unexpected keys.
    """

    # HF Marian checkpoint keys already carry the ``model.`` prefix and use
    # standard names, so the weights mapper is an identity map.
    hf_to_vllm_mapper = WeightsMapper()
    keys_to_ignore_on_load_missing = ["final_logits_bias"]

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        lora_config = vllm_config.lora_config
        # All Helsinki-NLP/opus-mt-* models tie the LM head to the shared
        # embeddings.
        assert config.tie_word_embeddings
        self.config = config
        self.model = MarianModel(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )

        self.unpadded_vocab_size = config.vocab_size
        if lora_config:
            self.unpadded_vocab_size += lora_config.lora_extra_vocab_size

        # HF's Marian lm_head uses the raw (unscaled) tied weight; only the
        # input word embedding is scaled by sqrt(d_model). So the LM head must
        # NOT divide by embed_scale -- doing so shrinks the matmul term while
        # leaving final_logits_bias full-size, distorting the distribution.
        self.lm_head = ParallelLMHead(config.vocab_size, config.d_model)
        # Bias added to logits after the LM head, matching HuggingFace.
        self.register_buffer("final_logits_bias", torch.zeros((1, config.vocab_size)))
        self.logits_processor = LogitsProcessor(
            self.unpadded_vocab_size, config.vocab_size
        )

    def get_language_model(self) -> nn.Module:
        return self.model.decoder

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: MultiModalEmbeddings | None = None,
        is_multimodal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.model.decoder.embed_tokens(input_ids)

    def embed_multimodal(self, **kwargs) -> MultiModalEmbeddings:
        encoder_input_ids_list = self._parse_and_validate_encoder_input(**kwargs)

        if not encoder_input_ids_list:
            raise ValueError(
                "encoder_input_ids_list is empty - this should not happen. "
                "Check that multimodal data is being passed correctly."
            )

        # Process each unique encoder input once; duplicates alias the same
        # output tensor (safe for read-only consumers).
        encoder_outputs: list[torch.Tensor] = []
        cached_outputs: dict[tuple[int, ...], torch.Tensor] = {}
        for encoder_input_ids in encoder_input_ids_list:
            cache_key = tuple(encoder_input_ids.reshape(-1).tolist())
            encoder_output = cached_outputs.get(cache_key)
            if encoder_output is not None:
                encoder_outputs.append(encoder_output)
                continue

            encoder_positions = torch.arange(
                encoder_input_ids.size(-1),
                dtype=torch.long,
                device=encoder_input_ids.device,
            )
            encoder_output = self.model.encoder(
                input_ids=encoder_input_ids.squeeze(0),
                positions=encoder_positions,
            )
            cached_outputs[cache_key] = encoder_output
            encoder_outputs.append(encoder_output)
        return encoder_outputs

    def _parse_and_validate_encoder_input(self, **kwargs: object) -> list[torch.Tensor]:
        encoder_input_ids = kwargs.get("encoder_input_ids", kwargs.get("input_ids"))

        if encoder_input_ids is None:
            return []

        if not isinstance(encoder_input_ids, (torch.Tensor, list)):
            raise ValueError(
                "Incorrect type of encoder input_ids. "
                f"Got type: {type(encoder_input_ids)}"
            )

        if isinstance(encoder_input_ids, list):
            result = []
            for item in encoder_input_ids:
                if isinstance(item, torch.Tensor) and item.dim() == 0:
                    item = item.unsqueeze(0)
                result.append(item)
            return result
        return encoder_input_ids.unsqueeze(1).unbind(dim=0)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        encoder_outputs: list[torch.Tensor] | None = None,
        **kwargs,
    ) -> torch.Tensor:
        # EncoderDecoderModelState passes an empty list on decode steps.
        enc_states = torch.cat(encoder_outputs, dim=0) if encoder_outputs else None

        return self.model(
            input_ids, positions, inputs_embeds, encoder_outputs=enc_states
        )

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        logits = self.logits_processor(self.lm_head, hidden_states)
        if logits is not None:
            logits = logits + self.final_logits_bias
        return logits

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        weights_tuple_list = list(weights)

        shared_embedding_weight = None
        final_logits_bias_weight = None
        for name, loaded_weight in weights_tuple_list:
            if "final_logits_bias" in name:
                final_logits_bias_weight = loaded_weight
            elif (
                "shared.weight" in name
                or "encoder.embed_tokens.weight" in name
                or "decoder.embed_tokens.weight" in name
                or "lm_head.weight" in name
            ):
                if shared_embedding_weight is not None:
                    # opus-mt ties model.shared, both embed_tokens and lm_head;
                    # they alias one tensor, so only the first copy is used.
                    continue
                shared_embedding_weight = loaded_weight

        # ``model.shared.weight`` has no matching parameter in this module, and
        # ``final_logits_bias`` is a buffer -> ignore them as "unexpected".
        loader = AutoWeightsLoader(
            self,
            ignore_unexpected_prefixes=["model.shared.", "final_logits_bias"],
        )
        loaded_params = loader.load_weights(
            weights_tuple_list, mapper=self.hf_to_vllm_mapper
        )

        if final_logits_bias_weight is not None:
            self.final_logits_bias.copy_(final_logits_bias_weight)
            loaded_params.add("final_logits_bias")

        if shared_embedding_weight is not None:
            weight_loader = getattr(
                self.lm_head.weight, "weight_loader", default_weight_loader
            )
            weight_loader(self.lm_head.weight, shared_embedding_weight)

            self.model.encoder.embed_tokens.weight = self.lm_head.weight
            self.model.decoder.embed_tokens.weight = self.lm_head.weight
            loaded_params.update(
                {
                    "model.encoder.embed_tokens.weight",
                    "lm_head.weight",
                    "model.decoder.embed_tokens.weight",
                }
            )

        # Buffers left at their default (init) values shouldn't be reported
        # missing.
        for key in self.keys_to_ignore_on_load_missing:
            loaded_params.add(key)

        return loaded_params
