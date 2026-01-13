# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import enum
import math
from collections.abc import Iterable, Mapping, Sequence
from contextlib import nullcontext
from functools import partial
from typing import Annotated, Literal, cast

import numpy as np
import torch
from torch import nn
from transformers import (
    BatchFeature,
    WhisperConfig,
    WhisperFeatureExtractor,
)
from transformers.models.whisper.modeling_whisper import sinusoids

from vllm.attention.layer import Attention
from vllm.config import CacheConfig, ModelConfig, SpeechToTextConfig, VllmConfig
from vllm.config.multimodal import BaseDummyOptions
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.inputs.data import PromptType
from vllm.logger import init_logger
from vllm.model_executor.layers.activation import get_act_fn
from vllm.model_executor.layers.attention.cross_attention import CrossAttention
from vllm.model_executor.layers.attention.mm_encoder_attention import MMEncoderAttention
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.whisper_utils import (
    ISO639_1_SUPPORTED_LANGS,
    WhisperAttentionWithBlockPooling,
    WhisperCausalConv1d,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (
    MultiModalDataDict,
    MultiModalFieldConfig,
    MultiModalKwargsItems,
)
from vllm.multimodal.parse import MultiModalDataItems, MultiModalDataParser
from vllm.multimodal.processing import (
    BaseProcessingInfo,
    EncDecMultiModalProcessor,
    PromptReplacement,
    PromptUpdate,
)
from vllm.multimodal.profiling import BaseDummyInputsBuilder
from vllm.transformers_utils.processor import cached_processor_from_config
from vllm.utils.jsontree import json_map_leaves
from vllm.utils.tensor_schema import TensorSchema, TensorShape
from vllm.utils.torch_utils import set_default_torch_dtype
from vllm.v1.attention.backend import (
    AttentionType,
)

from .interfaces import MultiModalEmbeddings, SupportsMultiModal, SupportsTranscription
from .utils import (
    AutoWeightsLoader,
    WeightsMapper,
    cast_overflow_tensors,
    make_layers,
    maybe_prefix,
)

logger = init_logger(__name__)


class WhisperPosEmbedType(enum.Enum):
    SINUSOIDAL = "sinusoidal"
    NOPE = "nope"
    LEARNED = "learned"


class WhisperAudioInputs(TensorSchema):
    """
    Dimensions:
        - b: Batch size
        - nmb: Number of mel bins
        - t: Time frames (M)
    """

    input_features: Annotated[
        list[torch.Tensor] | None,
        TensorShape("b", "nmb", "t"),
    ]


class WhisperEncoderAttention(MMEncoderAttention):
    """Multi-headed attention for Whisper encoder with 2D tensor support."""

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        """
        Input shape: batch_size x seq_len x hidden_size
                     or seq_len x hidden_size
        """
        is_2d = query.dim() == 2
        if is_2d:
            query = query.unsqueeze(0)
            key = key.unsqueeze(0)
            value = value.unsqueeze(0)

        # Call the parent forward method
        out = super().forward(query, key, value)

        if is_2d:
            out = out.squeeze(0)

        return out


class WhisperPositionalEmbedding(nn.Embedding):
    def __init__(self, num_positions: int, embedding_dim: int):
        super().__init__(num_positions, embedding_dim)

    def forward(self, position_ids):
        return self.weight[position_ids]


class WhisperAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        bias: bool = True,
        attn_type: AttentionType = AttentionType.DECODER,
        per_layer_sliding_window: int | None = None,
        block_pool_size: int = 1,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.embed_dim = embed_dim
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        if self.total_num_heads >= tp_size:
            # Number of heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_heads % tp_size == 0
        else:
            # Number of heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_heads == 0
        self.num_kv_heads = max(1, self.total_num_heads // tp_size)
        self.head_dim = self.embed_dim // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.attn_type = attn_type

        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: "
                f"{self.embed_dim} and `num_heads`: {num_heads})."
            )
        self.scaling = self.head_dim**-0.5

        self._init_qkv(embed_dim, bias, quant_config, prefix=prefix)
        self.out_proj = RowParallelLinear(
            input_size=embed_dim,
            output_size=embed_dim,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.out_proj",
        )
        if attn_type == AttentionType.ENCODER:
            self.attn = WhisperEncoderAttention(
                self.num_heads,
                self.head_dim,
                self.scaling,
                num_kv_heads=self.num_kv_heads,
            )
        elif self.attn_type == AttentionType.ENCODER_DECODER:
            self.attn = CrossAttention(
                self.num_heads,
                self.head_dim,
                self.scaling,
                num_kv_heads=self.num_kv_heads,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=f"{prefix}.attn",
                attn_type=self.attn_type,
            )
        else:  # AttentionType.DECODER (regular decoder self-attention)
            if block_pool_size > 1:
                attn_cls = partial(
                    WhisperAttentionWithBlockPooling, block_pool_size=block_pool_size
                )
            else:
                attn_cls = Attention

            self.attn = attn_cls(
                self.num_heads,
                self.head_dim,
                self.scaling,
                num_kv_heads=self.num_kv_heads,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=f"{prefix}.attn",
                attn_type=self.attn_type,
                per_layer_sliding_window=per_layer_sliding_window,
            )

    def _init_qkv(
        self,
        embed_dim: int,
        bias: bool = True,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        self.qkv_proj = QKVParallelLinear(
            hidden_size=embed_dim,
            head_size=self.head_dim,
            total_num_heads=self.total_num_heads,
            total_num_kv_heads=self.total_num_heads,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
    ):
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        attn_output = self.attn(q, k, v)

        output, _ = self.out_proj(attn_output)

        return output


class WhisperCrossAttention(WhisperAttention):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        bias: bool = True,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__(
            embed_dim=embed_dim,
            num_heads=num_heads,
            bias=bias,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=prefix,
            attn_type=AttentionType.ENCODER_DECODER,
        )

    def _init_qkv(
        self,
        embed_dim: int,
        bias: bool = True,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        self.q_proj = ColumnParallelLinear(
            input_size=embed_dim,
            output_size=embed_dim,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.q_proj",
        )
        self.kv_proj = QKVParallelLinear(
            hidden_size=embed_dim,
            head_size=self.head_dim,
            total_num_heads=0,
            total_num_kv_heads=self.total_num_heads,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.kv_proj",
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None,
    ):
        q, _ = self.q_proj(hidden_states)

        # Encoder hidden states are only computed once during prefill phase.
        # Afterwards, the keys and values should be available in the kv-cache.
        if encoder_hidden_states is not None:
            kv, _ = self.kv_proj(encoder_hidden_states)
            k, v = kv.split([self.kv_size, self.kv_size], dim=-1)
        else:
            k = v = None

        attn_output = self.attn(q, k, v)

        output, _ = self.out_proj(attn_output)

        return output


class WhisperMLP(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        ffn_dim: int,
        act_fn: str,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()

        self.activation_fn = get_act_fn(act_fn)
        self.fc1 = ColumnParallelLinear(
            input_size=embed_dim,
            output_size=ffn_dim,
            quant_config=quant_config,
            prefix=f"{prefix}.fc1",
        )
        self.fc2 = RowParallelLinear(
            input_size=ffn_dim,
            output_size=embed_dim,
            quant_config=quant_config,
            prefix=f"{prefix}.fc2",
        )

    def forward(self, hidden_states: torch.Tensor):
        hidden_states, _ = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states, _ = self.fc2(hidden_states)
        return hidden_states


class WhisperEncoderLayer(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        is_causal = getattr(config, "is_causal", False)
        sliding_window = getattr(config, "sliding_window", None)
        block_pool_size = getattr(config, "block_pool_size", 1)
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config

        self.embed_dim = config.d_model
        self.self_attn = WhisperAttention(
            embed_dim=self.embed_dim,
            num_heads=config.encoder_attention_heads,
            attn_type=AttentionType.DECODER if is_causal else AttentionType.ENCODER,
            block_pool_size=block_pool_size,
            per_layer_sliding_window=sliding_window,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
        )
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.mlp = WhisperMLP(
            embed_dim=config.d_model,
            ffn_dim=config.encoder_ffn_dim,
            act_fn=config.activation_function,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
    ):
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states = self.self_attn(hidden_states=hidden_states)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        hidden_states = cast_overflow_tensors(hidden_states)

        return hidden_states


class WhisperDecoderLayer(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config

        self.self_attn = WhisperAttention(
            embed_dim=config.d_model,
            num_heads=config.decoder_attention_heads,
            attn_type=AttentionType.DECODER,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
        )
        self.self_attn_layer_norm = nn.LayerNorm(config.d_model)
        self.encoder_attn = WhisperCrossAttention(
            embed_dim=config.d_model,
            num_heads=config.decoder_attention_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.encoder_attn",
        )
        self.encoder_attn_layer_norm = nn.LayerNorm(config.d_model)
        self.mlp = WhisperMLP(
            embed_dim=config.d_model,
            ffn_dim=config.decoder_ffn_dim,
            act_fn=config.activation_function,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )
        self.final_layer_norm = nn.LayerNorm(config.d_model)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None,
    ):
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states = self.self_attn(hidden_states=hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.encoder_attn_layer_norm(hidden_states)
        hidden_states = self.encoder_attn(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class WhisperEncoder(nn.Module):
    def __init__(
        self, *, vllm_config: VllmConfig, prefix: str = "", init_in_fp32: bool = False
    ):
        super().__init__()
        config = vllm_config.model_config.hf_config
        embed_dim = config.d_model

        self.pos_embed_type = WhisperPosEmbedType(
            getattr(config, "pos_embed", "sinusoidal")
        )
        self.num_mel_bins = config.num_mel_bins
        self.max_source_positions = config.max_source_positions
        self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0

        self.is_causal = getattr(config, "is_causal", False)
        Conv1d = (
            WhisperCausalConv1d if self.is_causal else partial(nn.Conv1d, padding=1)
        )

        self.conv1 = Conv1d(self.num_mel_bins, embed_dim, kernel_size=3)
        self.conv2 = Conv1d(embed_dim, embed_dim, stride=2, kernel_size=3)

        self.total_stride = self.conv1.stride[0] * self.conv2.stride[0]
        self.start_layer, self.end_layer, self.layers = make_layers(
            config.encoder_layers,
            lambda prefix: WhisperEncoderLayer(
                vllm_config=vllm_config, prefix=f"{prefix}.layers"
            ),
            prefix=f"{prefix}.layers",
        )
        self.layer_norm = nn.LayerNorm(config.d_model)

        if self.is_causal and self.pos_embed_type != WhisperPosEmbedType.NOPE:
            raise ValueError(
                "Only NOPE position embeddings are supported "
                f"for causal models, but got {self.pos_embed_type}"
            )
        elif self.pos_embed_type in (
            WhisperPosEmbedType.SINUSOIDAL,
            WhisperPosEmbedType.LEARNED,
        ):
            maybe_fp32_init_ctx = (
                set_default_torch_dtype(torch.float32)
                if init_in_fp32
                else nullcontext()
            )

            with (
                torch.no_grad(),
                maybe_fp32_init_ctx,
            ):
                self.embed_positions = nn.Embedding(
                    self.max_source_positions, embed_dim
                )
                self.embed_positions.weight.copy_(
                    sinusoids(*self.embed_positions.weight.shape)
                )

    def forward_conv(
        self, input_features: torch.Tensor | list[torch.Tensor]
    ) -> torch.Tensor:
        hidden_states = []
        input_is_batched = False
        for features in input_features:
            embeds = nn.functional.gelu(self.conv1(features))
            embeds = nn.functional.gelu(self.conv2(embeds))

            if self.pos_embed_type in (
                WhisperPosEmbedType.SINUSOIDAL,
                WhisperPosEmbedType.LEARNED,
            ):
                embeds = embeds.transpose(-1, -2)
                embeds = (
                    embeds + self.embed_positions.weight[: embeds.size(-2), :]
                ).to(embeds.dtype)
            elif self.pos_embed_type == WhisperPosEmbedType.NOPE:
                embeds = embeds.transpose(-1, -2).to(embeds.dtype)
            else:
                raise ValueError(f"Unknown pos_embed_type: {self.pos_embed_type}")

            hidden_states.append(embeds)
            input_is_batched = embeds.ndim > 2
        # Input to MHA must be B x T x D
        if input_is_batched or self.is_causal:
            # Models using WhisperEncoder may handle batching internally.
            # If WhisperEncoder is causal, sequences
            # are not padded to have identical seq length (T)
            # => concat over feature dim
            hidden_states = torch.cat(hidden_states)
        else:
            hidden_states = torch.stack(hidden_states, dim=0)

        return hidden_states

    def forward_layers(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for encoder_layer in self.layers:
            hidden_states = encoder_layer(hidden_states)

        hidden_states = self.layer_norm(hidden_states)
        return hidden_states

    def forward(self, input_features: torch.Tensor | list[torch.Tensor]):
        hidden_states = self.forward_conv(input_features)
        return self.forward_layers(hidden_states)


class WhisperDecoder(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.layerdrop = config.decoder_layerdrop
        self.padding_idx = config.pad_token_id
        self.max_target_positions = config.max_target_positions
        self.max_source_positions = config.max_source_positions
        self.embed_scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0

        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.d_model, self.padding_idx
        )
        self.embed_positions = WhisperPositionalEmbedding(
            self.max_target_positions, config.d_model
        )
        self.start_layer, self.end_layer, self.layers = make_layers(
            config.decoder_layers,
            lambda prefix: WhisperDecoderLayer(
                vllm_config=vllm_config, prefix=f"{prefix}.layers"
            ),
            prefix=f"{prefix}.layers",
        )
        self.layer_norm = nn.LayerNorm(config.d_model)

    def forward(
        self,
        input_ids,
        positions: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None,
    ):
        inputs_embeds = self.embed_input_ids(input_ids)
        positions = self.embed_positions(positions)
        hidden_states = inputs_embeds + positions

        for decoder_layer in self.layers:
            hidden_states = decoder_layer(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
            )

        hidden_states = self.layer_norm(hidden_states)
        return hidden_states

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)


class WhisperModel(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.encoder = WhisperEncoder(
            vllm_config=vllm_config, prefix=f"{prefix}.encoder"
        )
        self.decoder = WhisperDecoder(
            vllm_config=vllm_config, prefix=f"{prefix}.decoder"
        )

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        encoder_outputs: list[torch.Tensor],
    ) -> torch.Tensor:
        enc_states = torch.cat(encoder_outputs, dim=0) if len(encoder_outputs) else None
        decoder_outputs = self.decoder(
            input_ids=input_ids,
            positions=positions,
            encoder_hidden_states=enc_states,
        )
        return decoder_outputs

    def get_encoder_outputs(
        self,
        input_features: torch.Tensor | list[torch.Tensor] | None,
    ) -> torch.Tensor | None:
        if input_features is None:
            return None
        return self.encoder(input_features)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".self_attn.qkv_proj", ".self_attn.q_proj", "q"),
            (".self_attn.qkv_proj", ".self_attn.k_proj", "k"),
            (".self_attn.qkv_proj", ".self_attn.v_proj", "v"),
            (".encoder_attn.kv_proj", ".encoder_attn.k_proj", "k"),
            (".encoder_attn.kv_proj", ".encoder_attn.v_proj", "v"),
        ]
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        for name, loaded_weight in weights:
            for param_name, weight_name, shard_id in stacked_params_mapping:
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
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue

                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params


class WhisperProcessingInfo(BaseProcessingInfo):
    def get_hf_config(self) -> WhisperConfig:
        return self.ctx.get_hf_config(WhisperConfig)

    @property
    def skip_prompt_length_check(self) -> bool:
        return True  # Because the encoder prompt is padded

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"audio": 1}

    def get_feature_extractor(self, **kwargs: object) -> WhisperFeatureExtractor:
        hf_processor = self.get_hf_processor(**kwargs)
        feature_extractor = hf_processor.feature_extractor  # type: ignore
        assert isinstance(feature_extractor, WhisperFeatureExtractor)
        return feature_extractor

    def get_target_channels(self) -> int:
        """Return target audio channels for Whisper models (mono)."""
        return 1

    def get_num_audio_tokens(self) -> int:
        return self.get_hf_config().max_source_positions


class WhisperDummyInputsBuilder(BaseDummyInputsBuilder[WhisperProcessingInfo]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        num_audios = mm_counts.get("audio", 0)

        return "<|startoftranscript|>" * num_audios

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions] | None = None,
    ) -> MultiModalDataDict:
        feature_extractor = self.info.get_feature_extractor()

        sampling_rate = feature_extractor.sampling_rate
        audio_len = feature_extractor.chunk_length * sampling_rate
        num_audios = mm_counts.get("audio", 0)

        audio_overrides = mm_options.get("audio") if mm_options else None

        return {
            "audio": self._get_dummy_audios(
                length=audio_len, num_audios=num_audios, overrides=audio_overrides
            )
        }


class WhisperMultiModalProcessor(EncDecMultiModalProcessor[WhisperProcessingInfo]):
    def _get_data_parser(self) -> MultiModalDataParser:
        feature_extractor = self.info.get_feature_extractor()
        return MultiModalDataParser(
            target_sr=feature_extractor.sampling_rate,
            target_channels=self.info.get_target_channels(),
        )

    def create_encoder_prompt(
        self,
        prompt: str | list[int],
        mm_data: MultiModalDataDict,
    ) -> str | list[int]:
        # Strictly speaking, whisper encoder only accept audio features.
        # We create a dummy encoder prompt here which will be padded to
        # num_audio_tokens. So that we can create dummy data from this
        # for encoder profiling.
        return [0]

    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        if mm_data:
            feature_extractor = self.info.get_feature_extractor(**mm_kwargs)
            mm_data = dict(audio=mm_data.pop("audios"))
            mm_kwargs = dict(
                **mm_kwargs,
                sampling_rate=feature_extractor.sampling_rate,
            )
        processed_outputs = super()._call_hf_processor(
            prompt=prompt,
            mm_data=mm_data,
            mm_kwargs=mm_kwargs,
            tok_kwargs=tok_kwargs,
        )
        if "labels" in processed_outputs:
            processed_outputs["input_ids"] = processed_outputs.pop("labels")
        return processed_outputs

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return dict(input_features=MultiModalFieldConfig.batched("audio"))

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        num_tokens = self.info.get_num_audio_tokens()
        return [
            PromptReplacement(
                modality="audio",
                target=[0],
                replacement=[0] * num_tokens,
            )
        ]


@MULTIMODAL_REGISTRY.register_processor(
    WhisperMultiModalProcessor,
    info=WhisperProcessingInfo,
    dummy_inputs=WhisperDummyInputsBuilder,
)
class WhisperForConditionalGeneration(
    nn.Module, SupportsTranscription, SupportsMultiModal
):
    packed_modules_mapping = {
        "self_attn.qkv_proj": [
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
        ],
        "encoder_attn.kv_proj": ["encoder_attn.k_proj", "encoder_attn.v_proj"],
    }

    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_substr={".fc1.": ".mlp.fc1.", ".fc2.": ".mlp.fc2."}
    )

    # Whisper only supports audio-conditioned generation.
    supports_transcription_only = True
    supports_segment_timestamp = True
    supported_languages = ISO639_1_SUPPORTED_LANGS

    @classmethod
    def validate_language(cls, language: str | None) -> str | None:
        if language is None:
            # TODO language should be optional and can be guessed.
            # For now we default to en. See
            # https://github.com/huggingface/transformers/blob/main/src/transformers/models/whisper/generation_whisper.py#L1520
            logger.warning(
                "Defaulting to language='en'. If you wish to transcribe "
                "audio in a different language, pass the `language` field "
                "in the TranscriptionRequest."
            )
            language = "en"
        return super().validate_language(language)

    @classmethod
    def get_generation_prompt(
        cls,
        audio: np.ndarray,
        model_config: ModelConfig,  # not needed here
        stt_config: SpeechToTextConfig,
        language: str | None,
        task_type: Literal["transcribe", "translate"],
        request_prompt: str,
        to_language: str | None,
    ) -> PromptType:
        if language is None:
            raise ValueError(
                "Language must be specified when creating the Whisper prompt"
            )
        prompt = {
            "encoder_prompt": {
                # Whisper does not support encoder prompt.
                "prompt": "",
                "multi_modal_data": {
                    "audio": (audio, stt_config.sample_rate),
                },
            },
            "decoder_prompt": (
                (f"<|prev|>{request_prompt}" if request_prompt else "")
                + f"<|startoftranscript|><|{language}|>"
                + f"<|{task_type}|><|notimestamps|>"
            ),
        }
        return cast(PromptType, prompt)

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("audio"):
            return None

        raise ValueError("Only audio modality is supported")

    @classmethod
    def get_speech_to_text_config(
        cls, model_config: ModelConfig, task_type: str
    ) -> SpeechToTextConfig:
        processor = cached_processor_from_config(model_config)

        return SpeechToTextConfig(
            max_audio_clip_s=processor.feature_extractor.chunk_length,
            sample_rate=processor.feature_extractor.sampling_rate,
        )

    @classmethod
    def get_num_audio_tokens(
        cls,
        audio_duration_s: float,
        stt_config: SpeechToTextConfig,
        model_config: ModelConfig,
    ) -> int | None:
        processor = cached_processor_from_config(model_config)
        hop_length = processor.feature_extractor.hop_length
        assert hop_length is not None
        # NOTE(NickLucche) user can't pass encoder
        # prompts directly at least not to Whisper.
        # One indicator of the encoder amount of processing
        # is the log-mel spectogram length.
        return math.ceil(audio_duration_s * stt_config.sample_rate / hop_length)

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.dtype = vllm_config.model_config.dtype

        self.model = WhisperModel(vllm_config=vllm_config, prefix=prefix)

        self.proj_out = ParallelLMHead(
            config.vocab_size,
            config.d_model,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "proj_out"),
        )
        self.proj_out = self.proj_out.tie_weights(self.model.decoder.embed_tokens)
        logit_scale = getattr(config, "logit_scale", 1.0)
        self.logits_processor = LogitsProcessor(config.vocab_size, scale=logit_scale)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        encoder_outputs: list[torch.Tensor] | None = None,
        **kwargs,
    ) -> torch.Tensor:
        if encoder_outputs is None:
            encoder_outputs = []
        decoder_outputs = self.model(
            input_ids=input_ids,
            positions=positions,
            encoder_outputs=encoder_outputs,
        )
        return decoder_outputs

    def get_language_model(self) -> torch.nn.Module:
        return self.model.decoder

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        # Required as part of SupportsMultiModal interface.
        audio_input = self._parse_and_validate_audio_input(**kwargs)
        # Split concatenated encoder outputs into one tensor per audio input
        enc_output = self.model.get_encoder_outputs(audio_input["input_features"])
        # The assumption is we can only process whole mm items (audios)
        return enc_output.unbind(dim=0)

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: MultiModalEmbeddings | None = None,
        *,
        is_multimodal: torch.Tensor | None = None,
        handle_oov_mm_token: bool = False,
    ) -> torch.Tensor:
        # This method just returns the decoder sequence embeddings since
        # Whisper does not have encoder text tokens.
        return self.model.decoder.embed_input_ids(input_ids)

    def _parse_and_validate_audio_input(self, **kwargs: object) -> WhisperAudioInputs:
        input_features = kwargs.pop("input_features", None)

        if input_features is not None:
            input_features = json_map_leaves(lambda x: x.to(self.dtype), input_features)

        return WhisperAudioInputs(input_features=input_features)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        logits = self.logits_processor(self.proj_out, hidden_states)
        return logits

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self, skip_prefixes=["proj_out."])

        # add fake zeros bias for k_proj to state_dict
        weights = _create_fake_bias_for_k_proj(weights)
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)


def _create_fake_bias_for_k_proj(
    weights: Iterable[tuple[str, torch.Tensor]],
) -> Iterable[tuple[str, torch.Tensor]]:
    """
    Create full zeros bias for k_proj weight in self-attn and x-attn layers.
    So that the bias for k_proj in qkv_proj can be initialized with zeros.
    """
    for name, weight in weights:
        if name.endswith(".k_proj.weight"):
            bias = torch.zeros(weight.size(0))
            bias_name = name.replace("weight", "bias")
            yield from [(name, weight), (bias_name, bias)]
        yield name, weight
