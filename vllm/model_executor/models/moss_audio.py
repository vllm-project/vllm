# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-only MOSS-Audio model compatible with HuggingFace weights."""

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Annotated, Any

import numpy as np
import regex as re
import torch
import torch.nn.functional as F
from torch import nn
from transformers import BatchFeature, PretrainedConfig, Qwen3Config
from transformers.models.whisper import WhisperFeatureExtractor

from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.config.multimodal import BaseDummyOptions
from vllm.distributed import (
    get_pp_group,
    get_tensor_model_parallel_world_size,
)
from vllm.inputs import ModalityData, MultiModalDataDict
from vllm.model_executor.layers.activation import _ACTIVATION_REGISTRY, SiluAndMul
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (
    AudioItem,
    MultiModalFieldConfig,
    MultiModalKwargsItems,
)
from vllm.multimodal.parse import (
    DictEmbeddingItems,
    ModalityDataItems,
    MultiModalDataItems,
    MultiModalDataParser,
)
from vllm.multimodal.processing import (
    BaseDummyInputsBuilder,
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    PromptReplacement,
    PromptUpdate,
    PromptUpdateDetails,
)
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.repo_utils import get_hf_file_to_dict
from vllm.utils.tensor_schema import TensorSchema, TensorShape

from .interfaces import (
    MultiModalEmbeddings,
    SupportsLoRA,
    SupportsMultiModal,
    SupportsPP,
    _require_is_multimodal,
)
from .module_mapping import MultiModelKeys
from .qwen3 import Qwen3ForCausalLM, Qwen3Model
from .utils import (
    AutoWeightsLoader,
    WeightsMapper,
    _merge_multimodal_embeddings,
    maybe_prefix,
)

MOSS_AUDIO_TOKEN = "<|AUDIO|>"
MOSS_AUDIO_BOS_TOKEN = "<|audio_bos|>"
MOSS_AUDIO_EOS_TOKEN = "<|audio_eos|>"
MOSS_AUDIO_TOKEN_ID = 151654
MOSS_AUDIO_BOS_TOKEN_ID = 151669
MOSS_AUDIO_EOS_TOKEN_ID = 151670
DEFAULT_MAX_AUDIO_SECONDS = 30
DEFAULT_MOSS_AUDIO_MEL_CONFIG = {
    "mel_dim": 128,
    "mel_sr": 16000,
    "mel_hop_length": 160,
    "mel_n_fft": 400,
}
MOSS_AUDIO_PLACEHOLDER = (
    f"{MOSS_AUDIO_BOS_TOKEN}{MOSS_AUDIO_TOKEN}{MOSS_AUDIO_EOS_TOKEN}"
)
MOSS_AUDIO_SPAN_RE = re.compile(
    f"{re.escape(MOSS_AUDIO_BOS_TOKEN)}"
    f"(?:{re.escape(MOSS_AUDIO_TOKEN)})+"
    f"{re.escape(MOSS_AUDIO_EOS_TOKEN)}"
)
MOSS_AUDIO_PROCESSOR_CONFIG_KEYS = {
    "audio_token_id",
    "audio_start_id",
    "audio_end_id",
    "enable_time_marker",
    "mel_config",
}


class MossAudioAudioInputs(TensorSchema):
    """
    Dimensions:
        - b: Batch size
        - nmb: Number of mel bins
        - t: Time frames
    """

    audio_data: Annotated[torch.Tensor, TensorShape("b", "nmb", "t")]
    audio_data_seqlens: Annotated[torch.Tensor, TensorShape("b")]


def _normalize_moss_audio_mel_config(
    mel_config: Mapping[str, object] | None = None,
) -> dict[str, int]:
    config = dict(DEFAULT_MOSS_AUDIO_MEL_CONFIG)
    config.update(_extract_moss_audio_mel_config(mel_config))
    return config


def _extract_moss_audio_mel_config(
    mel_config: Mapping[str, object] | None = None,
) -> dict[str, int]:
    config: dict[str, int] = {}
    if mel_config is None:
        return config

    aliases = {
        "mel_dim": ("mel_dim", "feature_size", "n_mels", "num_mel_bins"),
        "mel_sr": ("mel_sr", "sampling_rate", "sample_rate"),
        "mel_hop_length": ("mel_hop_length", "hop_length"),
        "mel_n_fft": ("mel_n_fft", "n_fft"),
    }
    for target_key, source_keys in aliases.items():
        for source_key in source_keys:
            if source_key in mel_config:
                config[target_key] = int(mel_config[source_key])
                break

    return config


def _filter_moss_audio_processor_config(
    config: Mapping[str, object] | None,
) -> dict[str, object]:
    if not config:
        return {}

    return {
        key: value
        for key, value in config.items()
        if key in MOSS_AUDIO_PROCESSOR_CONFIG_KEYS
    }


def _merge_moss_audio_processor_configs(
    *configs: Mapping[str, object] | None,
) -> dict[str, object]:
    merged: dict[str, object] = {}
    merged_mel_config: dict[str, int] = {}
    for config in configs:
        filtered = _filter_moss_audio_processor_config(config)
        mel_config = filtered.pop("mel_config", None)
        merged.update(filtered)
        if isinstance(mel_config, Mapping):
            merged_mel_config.update(_extract_moss_audio_mel_config(mel_config))

    if merged_mel_config:
        merged["mel_config"] = merged_mel_config
    return merged


@dataclass
class MossAudioEncoderConfig:
    d_model: int = 1280
    output_dim: int = 1280
    num_mel_bins: int = 128
    encoder_layers: int = 32
    encoder_attention_heads: int = 20
    encoder_ffn_dim: int = 5120
    downsample_rate: int = 8
    downsample_hidden_size: int = 480
    encoder_attention_window_size: int = 100
    max_source_positions: int = 1500
    dropout: float = 0.1
    attention_dropout: float = 0.1
    activation_dropout: float = 0.0
    activation_function: str = "gelu"
    layer_norm_eps: float = 1e-5
    _attn_implementation: str = "eager"
    pretrained_path: str = ""
    n_window: int = 200
    conv_chunksize: int = 64
    deepstack_encoder_layer_indexes: list[int] = field(
        default_factory=lambda: [8, 16, 24]
    )

    @classmethod
    def from_config(cls, config: object) -> "MossAudioEncoderConfig":
        if isinstance(config, cls):
            return config
        if isinstance(config, Mapping):
            values = {
                key: value
                for key, value in config.items()
                if key in cls.__dataclass_fields__
            }
        else:
            values = {
                key: getattr(config, key)
                for key in cls.__dataclass_fields__
                if hasattr(config, key)
            }
        return cls(**values)


class MossAudioConfig(PretrainedConfig):
    model_type = "moss_audio"
    is_composition = True

    def __init__(
        self,
        audio_config: Mapping[str, object] | MossAudioEncoderConfig | None = None,
        language_config: Mapping[str, object] | Qwen3Config | None = None,
        adapter_hidden_size: int = 8192,
        ignore_index: int = -100,
        deepstack_num_inject_layers: int | None = None,
        **kwargs: object,
    ) -> None:
        self.audio_config = MossAudioEncoderConfig.from_config(audio_config or {})
        if isinstance(language_config, Qwen3Config):
            self.language_config = language_config
        else:
            self.language_config = Qwen3Config(**(language_config or {}))

        self.adapter_hidden_size = adapter_hidden_size
        self.ignore_index = ignore_index
        self.deepstack_num_inject_layers = deepstack_num_inject_layers

        for key in ("num_hidden_layers", "eos_token_id", "bos_token_id", "vocab_size"):
            kwargs.setdefault(key, getattr(self.language_config, key, None))
        kwargs.setdefault("tie_word_embeddings", False)
        super().__init__(**kwargs)

        for key in (
            "hidden_size",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
            "max_position_embeddings",
            "rms_norm_eps",
        ):
            if hasattr(self.language_config, key):
                setattr(self, key, getattr(self.language_config, key))

    def get_text_config(self, decoder: bool = False) -> Qwen3Config:
        return self.language_config


class SinusoidsPositionEmbedding(nn.Module):
    def __init__(self, num_positions: int, embedding_dim: int) -> None:
        super().__init__()
        del num_positions  # Kept for config compatibility.
        max_timescale = 10000.0
        log_timescale_increment = math.log(max_timescale) / (embedding_dim // 2 - 1)
        inv_timescales = torch.exp(
            -log_timescale_increment * torch.arange(embedding_dim // 2).float()
        )
        self.register_buffer("inv_timescales", inv_timescales, persistent=False)

    def forward(self, seq_len: int, device: torch.device) -> torch.Tensor:
        scaled_time = (
            torch.arange(seq_len, device=device, dtype=self.inv_timescales.dtype)[
                :, None
            ]
            * self.inv_timescales[None, :]
        )
        return torch.cat([scaled_time.sin(), scaled_time.cos()], dim=1).unsqueeze(0)


class MossAudioAttention(nn.Module):
    def __init__(
        self,
        config: MossAudioEncoderConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.embed_dim = config.d_model
        self.num_heads = config.encoder_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"d_model ({self.embed_dim}) must be divisible by "
                f"encoder_attention_heads ({self.num_heads})."
            )

        tp_size = get_tensor_model_parallel_world_size()
        if self.num_heads % tp_size != 0:
            raise ValueError(
                "MOSS-Audio audio encoder attention heads must be divisible by "
                f"tensor parallel size. Got {self.num_heads=} and {tp_size=}."
            )
        self.num_local_heads = self.num_heads // tp_size
        # TODO: can use QKVParallelLinear
        self.q_proj = ColumnParallelLinear(
            input_size=self.embed_dim,
            output_size=self.embed_dim,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.q_proj",
        )
        self.k_proj = ColumnParallelLinear(
            input_size=self.embed_dim,
            output_size=self.embed_dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.k_proj",
        )
        self.v_proj = ColumnParallelLinear(
            input_size=self.embed_dim,
            output_size=self.embed_dim,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.v_proj",
        )
        self.out_proj = RowParallelLinear(
            input_size=self.embed_dim,
            output_size=self.embed_dim,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.out_proj",
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        q, _ = self.q_proj(hidden_states)
        k, _ = self.k_proj(hidden_states)
        v, _ = self.v_proj(hidden_states)
        q = q.view(batch_size, seq_len, -1, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, -1, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, -1, self.head_dim).transpose(1, 2)
        attn_output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attention_mask[:, None, None, :],
            dropout_p=0.0,
            scale=self.head_dim**-0.5,
        )
        output, _ = self.out_proj(
            attn_output.transpose(1, 2).reshape(
                batch_size,
                seq_len,
                -1,
            )
        )
        return output


class MossAudioEncoderLayer(nn.Module):
    def __init__(
        self,
        config: MossAudioEncoderConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.self_attn = MossAudioAttention(
            config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
        )
        self.self_attn_layer_norm = nn.LayerNorm(
            config.d_model, eps=config.layer_norm_eps
        )
        self.activation_fn = _ACTIVATION_REGISTRY[config.activation_function]
        self.activation_dropout = config.activation_dropout
        self.dropout = config.dropout
        self.fc1 = ColumnParallelLinear(
            config.d_model,
            config.encoder_ffn_dim,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.fc1",
        )
        self.fc2 = RowParallelLinear(
            config.encoder_ffn_dim,
            config.d_model,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.fc2",
        )
        self.final_layer_norm = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states = self.self_attn(hidden_states, attention_mask)
        hidden_states = residual + F.dropout(
            hidden_states, p=self.dropout, training=self.training
        )

        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states, _ = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = F.dropout(
            hidden_states, p=self.activation_dropout, training=self.training
        )
        hidden_states, _ = self.fc2(hidden_states)
        hidden_states = residual + F.dropout(
            hidden_states, p=self.dropout, training=self.training
        )
        return hidden_states


class MossAudioEncoder(nn.Module):
    def __init__(
        self,
        config: MossAudioEncoderConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.gelu = nn.GELU()
        self.conv1 = nn.Conv2d(
            1,
            config.downsample_hidden_size,
            kernel_size=(3, 3),
            stride=(2, 2),
            padding=(1, 1),
        )
        self.conv2 = nn.Conv2d(
            config.downsample_hidden_size,
            config.downsample_hidden_size,
            kernel_size=(3, 3),
            stride=(2, 2),
            padding=(1, 1),
        )
        self.conv3 = nn.Conv2d(
            config.downsample_hidden_size,
            config.downsample_hidden_size,
            kernel_size=(3, 3),
            stride=(2, 2),
            padding=(1, 1),
        )

        conv_freq = self._compute_downsampled_length(
            torch.tensor(config.num_mel_bins)
        ).item()
        self.stem_proj = ReplicatedLinear(
            config.downsample_hidden_size * int(conv_freq),
            config.d_model,
            bias=True,
            quant_config=quant_config,
            return_bias=False,
            prefix=f"{prefix}.stem_proj",
        )
        self.embed_positions = SinusoidsPositionEmbedding(
            config.max_source_positions, config.d_model
        )
        self.layers = nn.ModuleList(
            [
                MossAudioEncoderLayer(
                    config,
                    quant_config=quant_config,
                    prefix=f"{prefix}.layers.{layer_idx}",
                )
                for layer_idx in range(config.encoder_layers)
            ]
        )
        self.layer_norm = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        if config.output_dim != config.d_model:
            self.out_proj = ReplicatedLinear(
                config.d_model,
                config.output_dim,
                bias=False,
                quant_config=quant_config,
                return_bias=False,
                prefix=f"{prefix}.out_proj",
            )
        else:
            self.out_proj = nn.Identity()

        self.deepstack_encoder_layer_indexes = list(
            config.deepstack_encoder_layer_indexes or []
        )
        self._deepstack_capture_map = {
            layer_idx: capture_idx
            for capture_idx, layer_idx in enumerate(
                self.deepstack_encoder_layer_indexes
            )
        }
        self.n_window = int(config.n_window)
        self.chunk_frames = int(self.n_window * 2)
        self.conv_chunksize = int(config.conv_chunksize)

    @property
    def dtype(self) -> torch.dtype:
        return self.conv1.weight.dtype

    @staticmethod
    def _compute_downsampled_length(lengths: torch.Tensor) -> torch.Tensor:
        def conv_out_len(length: torch.Tensor) -> torch.Tensor:
            return (length - 1) // 2 + 1

        return conv_out_len(conv_out_len(conv_out_len(lengths)))

    @staticmethod
    def compute_num_audio_tokens(raw_mel_len: int) -> int:
        lengths = torch.tensor(raw_mel_len, dtype=torch.long)
        return int(MossAudioEncoder._compute_downsampled_length(lengths).item())

    def _encode_chunk_batch(
        self,
        input_features: torch.Tensor,
        seq_lengths: torch.Tensor,
        output_deepstack_hidden_states: bool = True,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        if input_features.dim() == 2:
            input_features = input_features.unsqueeze(0)

        downsampled_lengths = self._compute_downsampled_length(seq_lengths)
        x = input_features.unsqueeze(1)
        x = self.gelu(self.conv1(x))
        x = self.gelu(self.conv2(x))
        x = self.gelu(self.conv3(x))
        x = x.permute(0, 3, 1, 2).contiguous().flatten(2)
        x = self.stem_proj(x)

        max_len = int(downsampled_lengths.max().item())
        if x.size(1) > max_len:
            x = x[:, :max_len, :]
        x = x + self.embed_positions(x.shape[1], x.device).to(x.dtype)

        attention_mask = (
            torch.arange(x.size(1), device=x.device)[None, :]
            < downsampled_lengths[:, None]
        )

        deepstack_hidden_states: list[torch.Tensor | None] = []
        if output_deepstack_hidden_states:
            deepstack_hidden_states = [None] * len(self.deepstack_encoder_layer_indexes)
        for layer_idx, layer in enumerate(self.layers):
            x = layer(x, attention_mask)
            if output_deepstack_hidden_states:
                capture_idx = self._deepstack_capture_map.get(layer_idx)
                if capture_idx is not None:
                    deepstack_hidden_states[capture_idx] = x

        x = self.layer_norm(x)
        x = self.out_proj(x)

        if not output_deepstack_hidden_states:
            return x, []

        ordered_deepstack_hidden_states = [
            hidden_states
            for hidden_states in deepstack_hidden_states
            if hidden_states is not None
        ]
        ordered_deepstack_hidden_states = [
            self.out_proj(hidden_states)
            for hidden_states in ordered_deepstack_hidden_states
        ]
        return x, ordered_deepstack_hidden_states

    def forward(
        self,
        input_features: torch.Tensor,
        feature_lens: torch.Tensor | None = None,
        output_deepstack_hidden_states: bool = True,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...] | None]:
        if input_features.dim() == 3:
            if feature_lens is None:
                feature_lens = torch.full(
                    (input_features.size(0),),
                    input_features.size(-1),
                    dtype=torch.long,
                    device=input_features.device,
                )
            else:
                feature_lens = feature_lens.to(
                    device=input_features.device, dtype=torch.long
                )
            valid_chunks = [
                input_features[i, :, : int(feature_lens[i].item())]
                for i in range(int(input_features.shape[0]))
            ]
            input_features = torch.cat(valid_chunks, dim=1)
        elif input_features.dim() != 2:
            raise ValueError(
                f"Expected [n_mels, T] or [B, n_mels, T], got "
                f"{tuple(input_features.shape)}."
            )

        if feature_lens is None:
            feature_lens = torch.tensor(
                [int(input_features.shape[1])],
                device=input_features.device,
                dtype=torch.long,
            )
        else:
            feature_lens = feature_lens.to(
                device=input_features.device, dtype=torch.long
            )

        chunk_num = torch.ceil(
            feature_lens.to(torch.float32) / self.chunk_frames
        ).long()
        chunk_lengths = torch.full(
            (int(chunk_num.sum().item()),),
            self.chunk_frames,
            dtype=torch.long,
            device=feature_lens.device,
        )
        tail_chunk_index = F.pad(chunk_num, (1, 0), value=-1).cumsum(0)[1:]
        chunk_lengths[tail_chunk_index] = feature_lens % self.chunk_frames
        chunk_lengths[chunk_lengths == 0] = self.chunk_frames

        chunk_list = input_features.T.split(chunk_lengths.tolist(), dim=0)
        padded_feature = nn.utils.rnn.pad_sequence(
            chunk_list, batch_first=True
        ).transpose(1, 2)

        feature_lens_after_cnn = self._compute_downsampled_length(chunk_lengths)
        t_down_max = (
            int(feature_lens_after_cnn.max().item())
            if feature_lens_after_cnn.numel() > 0
            else 0
        )
        indices = torch.arange(t_down_max, device=padded_feature.device)
        padded_mask_after_cnn = indices[None, :] < feature_lens_after_cnn[:, None]

        num_deepstack = len(self.deepstack_encoder_layer_indexes)
        should_output_deepstack = output_deepstack_hidden_states and num_deepstack > 0
        padded_embeds: list[torch.Tensor] = []
        deepstack_padded_embeds: list[list[torch.Tensor]] = [
            [] for _ in range(num_deepstack if should_output_deepstack else 0)
        ]
        for feat_chunk, len_chunk in zip(
            padded_feature.split(self.conv_chunksize, dim=0),
            chunk_lengths.split(self.conv_chunksize, dim=0),
        ):
            out, deepstack_outs = self._encode_chunk_batch(
                feat_chunk,
                len_chunk,
                output_deepstack_hidden_states=should_output_deepstack,
            )
            if out.shape[1] < t_down_max:
                out = F.pad(out, (0, 0, 0, t_down_max - out.shape[1]))
            padded_embeds.append(out)

            if should_output_deepstack:
                if len(deepstack_outs) != num_deepstack:
                    raise RuntimeError(
                        "DeepStack output count does not match configured "
                        "layer indexes."
                    )
                for capture_idx, ds in enumerate(deepstack_outs):
                    if ds.shape[1] < t_down_max:
                        ds = F.pad(ds, (0, 0, 0, t_down_max - ds.shape[1]))
                    deepstack_padded_embeds[capture_idx].append(ds)

        if padded_embeds:
            padded_embed = torch.cat(padded_embeds, dim=0)
        else:
            padded_embed = torch.empty(
                (0, t_down_max, self.config.output_dim),
                device=padded_feature.device,
                dtype=padded_feature.dtype,
            )

        last_hidden_state = padded_embed[padded_mask_after_cnn].unsqueeze(0)

        deepstack_states: tuple[torch.Tensor, ...] | None = None
        if should_output_deepstack:
            collected: list[torch.Tensor] = []
            for chunks_list in deepstack_padded_embeds:
                if chunks_list:
                    ds = torch.cat(chunks_list, dim=0)
                    collected.append(ds[padded_mask_after_cnn].unsqueeze(0))
                else:
                    collected.append(
                        torch.empty(
                            (1, 0, self.config.output_dim),
                            device=padded_feature.device,
                            dtype=padded_embed.dtype,
                        )
                    )
            deepstack_states = tuple(collected)

        return last_hidden_state, deepstack_states


class GatedMLP(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            input_size,
            [hidden_size, hidden_size],
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            hidden_size,
            output_size,
            bias=False,
            input_is_parallel=True,
            quant_config=quant_config,
            prefix=f"{prefix}.down_proj",
        )
        self.act_fn = SiluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        "positions": -1,
        "intermediate_tensors": 0,
        "inputs_embeds": 0,
        "deepstack_input_embeds": 0,
    }
)
class MossQwen3Model(Qwen3Model):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        self.deepstack_inject_layer_indices: Iterable[int] = range(0)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        deepstack_input_embeds: IntermediateTensors | None = None,
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

        aux_hidden_states = self._maybe_add_hidden_state([], 0, hidden_states, residual)
        for layer_idx, layer in enumerate(
            self.layers[self.start_layer : self.end_layer],
            start=self.start_layer,
        ):
            hidden_states, residual = layer(positions, hidden_states, residual)
            deepstack_key = f"deepstack_input_embeds_{layer_idx}"
            if (
                deepstack_input_embeds is not None
                and deepstack_key in deepstack_input_embeds.tensors
            ):
                hidden_states = hidden_states + deepstack_input_embeds[deepstack_key]
            self._maybe_add_hidden_state(
                aux_hidden_states,
                layer_idx - self.start_layer + 1,
                hidden_states,
                residual,
            )

        if not get_pp_group().is_last_rank:
            tensors = {"hidden_states": hidden_states, "residual": residual}
            # Keep the DeepStack PP schema config-driven, but only carry
            # payloads needed by downstream injection points across this rank.
            # Missing downstream payloads are zero-filled below to clear
            # receive buffers instead of leaving stale tensors.
            for layer_idx in self.deepstack_inject_layer_indices:
                if layer_idx < self.end_layer:
                    continue
                deepstack_key = f"deepstack_input_embeds_{layer_idx}"
                if (
                    deepstack_input_embeds is not None
                    and deepstack_key in deepstack_input_embeds.tensors
                ):
                    tensors[deepstack_key] = deepstack_input_embeds[deepstack_key]
                else:
                    tensors[deepstack_key] = hidden_states.new_zeros(
                        hidden_states.shape
                    )
            return IntermediateTensors(tensors)

        hidden_states, _ = self.norm(hidden_states, residual)
        if len(aux_hidden_states) > 0:
            return hidden_states, aux_hidden_states
        return hidden_states


class MossQwen3ForCausalLM(Qwen3ForCausalLM):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super(Qwen3ForCausalLM, self).__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config

        self.config = config
        self.vllm_config = vllm_config
        self.quant_config = quant_config
        self.model = MossQwen3Model(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )

        if get_pp_group().is_last_rank:
            if config.tie_word_embeddings:
                self.lm_head = self.model.embed_tokens
            else:
                self.lm_head = ParallelLMHead(
                    config.vocab_size,
                    config.hidden_size,
                    quant_config=quant_config,
                    prefix=maybe_prefix(prefix, "lm_head"),
                )
        else:
            from .utils import PPMissingLayer

            self.lm_head = PPMissingLayer()

        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.deepstack_inject_layer_indices: Iterable[int] = range(0)

    def make_empty_intermediate_tensors(
        self,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> IntermediateTensors:
        intermediate_tensors = self.model.make_empty_intermediate_tensors(
            batch_size, dtype, device
        )
        for layer_idx in self.deepstack_inject_layer_indices:
            # Non-first PP ranks only receive DeepStack payloads for layers
            # at or after their local start layer.
            if layer_idx < self.model.start_layer:
                continue
            intermediate_tensors[f"deepstack_input_embeds_{layer_idx}"] = torch.zeros(
                (batch_size, self.config.hidden_size),
                dtype=dtype,
                device=device,
            )
        return intermediate_tensors

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        deepstack_input_embeds: IntermediateTensors | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        return self.model(
            input_ids,
            positions,
            intermediate_tensors,
            inputs_embeds,
            deepstack_input_embeds=deepstack_input_embeds,
        )


def _moss_audio_field_config(
    hf_inputs: Mapping[str, torch.Tensor],
) -> Mapping[str, MultiModalFieldConfig]:
    return {
        "audio_data": MultiModalFieldConfig.batched("audio"),
        "audio_data_seqlens": MultiModalFieldConfig.batched("audio", keep_on_cpu=True),
    }


class MossAudioMultiModalDataParser(MultiModalDataParser):
    def _parse_audio_data(
        self,
        data: dict[str, torch.Tensor] | ModalityData[AudioItem],
    ) -> ModalityDataItems[Any, Any] | None:
        if isinstance(data, dict):
            return DictEmbeddingItems(
                data,
                modality="audio",
                required_fields={"audio_data", "audio_data_seqlens"},
                fields_factory=_moss_audio_field_config,
            )

        return super()._parse_audio_data(data)


class MossAudioProcessor:
    model_input_names = [
        "input_ids",
        "attention_mask",
        "audio_data",
        "audio_data_seqlens",
    ]

    def __init__(
        self,
        tokenizer: object,
        *,
        audio_token_id: int = MOSS_AUDIO_TOKEN_ID,
        audio_start_id: int = MOSS_AUDIO_BOS_TOKEN_ID,
        audio_end_id: int = MOSS_AUDIO_EOS_TOKEN_ID,
        enable_time_marker: bool = False,
        mel_config: Mapping[str, object] | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.audio_token_id = int(audio_token_id)
        self.audio_start_id = int(audio_start_id)
        self.audio_end_id = int(audio_end_id)
        self.enable_time_marker = bool(enable_time_marker)
        self.mel_config = _normalize_moss_audio_mel_config(mel_config)
        self.feature_extractor = WhisperFeatureExtractor(
            feature_size=self.mel_config["mel_dim"],
            sampling_rate=self.mel_config["mel_sr"],
            hop_length=self.mel_config["mel_hop_length"],
            n_fft=self.mel_config["mel_n_fft"],
        )
        self.audio_tokens_per_second = self.mel_config["mel_sr"] / (
            self.mel_config["mel_hop_length"] * 8
        )
        self.time_marker_every_seconds = 2
        self.time_marker_every_audio_tokens = int(
            self.audio_tokens_per_second * self.time_marker_every_seconds
        )
        self._digit_token_ids = {
            "0": 15,
            "1": 16,
            "2": 17,
            "3": 18,
            "4": 19,
            "5": 20,
            "6": 21,
            "7": 22,
            "8": 23,
            "9": 24,
        }

    @staticmethod
    def conv3_downsample_len(raw_mel_len: int) -> int:
        return MossAudioEncoder.compute_num_audio_tokens(raw_mel_len)

    def _extract_mel(self, audio: np.ndarray | torch.Tensor) -> torch.Tensor:
        if isinstance(audio, torch.Tensor):
            wav = audio.detach().to("cpu", dtype=torch.float32).numpy()
        else:
            wav = np.asarray(audio, dtype=np.float32)
        if wav.size == 0:
            raise ValueError("The audio is too short to be represented.")
        if wav.ndim == 2:
            wav = wav[0]
        feats = self.feature_extractor._np_extract_fbank_features(
            wav[None, ...], device="cpu"
        )
        return torch.from_numpy(feats[0])

    def _get_default_audio_prompt(self) -> str:
        return MOSS_AUDIO_PLACEHOLDER

    def _ensure_audio_placeholders(
        self,
        prompt_text: str,
        num_audios: int,
    ) -> str:
        if num_audios == 0 or MOSS_AUDIO_SPAN_RE.search(prompt_text):
            return prompt_text

        audio_prompt = self._get_default_audio_prompt() * num_audios
        if prompt_text:
            return f"{audio_prompt}\n{prompt_text}"
        return audio_prompt

    def _build_audio_tokens_with_time_markers(self, audio_seq_len: int) -> list[int]:
        total_duration_seconds = audio_seq_len / self.audio_tokens_per_second
        num_full_seconds = int(total_duration_seconds)
        token_ids: list[int] = []
        audio_tokens_consumed = 0
        for second in range(
            self.time_marker_every_seconds,
            num_full_seconds + 1,
            self.time_marker_every_seconds,
        ):
            marker_pos = (
                second // self.time_marker_every_seconds
            ) * self.time_marker_every_audio_tokens
            audio_segment_len = marker_pos - audio_tokens_consumed
            if audio_segment_len > 0:
                token_ids.extend([self.audio_token_id] * audio_segment_len)
                audio_tokens_consumed += audio_segment_len
            token_ids.extend(self._digit_token_ids[digit] for digit in str(second))

        remaining = audio_seq_len - audio_tokens_consumed
        if remaining > 0:
            token_ids.extend([self.audio_token_id] * remaining)
        return token_ids

    def build_audio_placeholder_ids(self, num_audio_tokens: int) -> list[int]:
        if self.enable_time_marker:
            return self._build_audio_tokens_with_time_markers(num_audio_tokens)
        return [self.audio_token_id] * num_audio_tokens

    def __call__(
        self,
        text: str | Sequence[str] | None = None,
        audios: Sequence[np.ndarray | torch.Tensor] | None = None,
        audio: Sequence[np.ndarray | torch.Tensor] | None = None,
        return_tensors: str = "pt",
        **kwargs: object,
    ) -> BatchFeature:
        """Build text tokens and audio tensors for one MossAudio prompt.

        Example:
            text="Describe this.", audio=[waveform]
            -> input_ids contains audio_start, N audio tokens, audio_end
            -> audio_data has shape [1, mel_dim, max_time]
            -> mel_dim is the number of mel filter-bank bins, 128 by default
            -> audio_data_seqlens stores the unpadded mel length
        """
        del kwargs

        # Step 1. Normalize text input; this processor handles one prompt.
        if isinstance(text, (list, tuple)):
            if len(text) != 1:
                raise ValueError(f"Expected text batch size 1, got {len(text)}")
            prompt_text = text[0]
        elif text is None:
            prompt_text = ""
        else:
            prompt_text = text

        # Step 2. Accept either `audios` or `audio` and normalize to a list.
        audio_list = audios if audios is not None else audio
        audio_list = [] if audio_list is None else list(audio_list)

        # Step 3. Convert waveforms to [mel_dim, time] mel features and token
        # counts. mel_dim is the number of mel filter-bank bins.
        mels: list[torch.Tensor] = []
        raw_lengths: list[int] = []
        token_lens: list[int] = []
        for one_audio in audio_list:
            mel = self._extract_mel(one_audio)
            raw_len = int(mel.shape[-1])
            num_tokens = self.conv3_downsample_len(raw_len)
            if raw_len <= 0 or num_tokens <= 0:
                raise ValueError("The audio is too short to be represented.")
            mels.append(mel)
            raw_lengths.append(raw_len)
            token_lens.append(num_tokens)

        # Step 4. Pad variable-length mel features into a batch tensor.
        if mels:
            max_length = max(raw_lengths)
            audio_batch = torch.zeros(
                (len(mels), self.mel_config["mel_dim"], max_length),
                dtype=torch.float32,
            )
            for index, mel in enumerate(mels):
                audio_batch[index, :, : mel.shape[-1]] = mel
            audio_data_seqlens = torch.tensor(raw_lengths, dtype=torch.long)
        else:
            audio_batch = None
            audio_data_seqlens = None

        # Step 5. Ensure each audio item has a placeholder span in the prompt.
        prompt_text = self._ensure_audio_placeholders(prompt_text, len(audio_list))
        input_ids = []
        cursor = 0

        # Step 6. Text-only path: tokenize and preserve placeholder spans.
        if not audio_list:
            for match in MOSS_AUDIO_SPAN_RE.finditer(prompt_text):
                prefix = prompt_text[cursor : match.start()]
                input_ids.extend(
                    self.tokenizer.encode(prefix, add_special_tokens=False)
                )
                input_ids.extend(
                    [self.audio_start_id, self.audio_token_id, self.audio_end_id]
                )
                cursor = match.end()
            suffix = prompt_text[cursor:]
            input_ids.extend(self.tokenizer.encode(suffix, add_special_tokens=False))
            data: dict[str, torch.Tensor] = {
                "input_ids": torch.tensor([input_ids], dtype=torch.long),
                "attention_mask": torch.ones((1, len(input_ids)), dtype=torch.long),
            }
            return BatchFeature(data=data, tensor_type=return_tensors)

        # Step 7. Audio path: expand each placeholder to its audio-token count.
        span_iter = iter(MOSS_AUDIO_SPAN_RE.finditer(prompt_text))
        for item_idx, _ in enumerate(audio_list):
            match = next(span_iter, None)
            if match is None:
                raise ValueError(
                    "Audio placeholder count mismatch: expected one "
                    f"{MOSS_AUDIO_PLACEHOLDER!r} span per audio item."
                )
            prefix = prompt_text[cursor : match.start()]
            input_ids.extend(self.tokenizer.encode(prefix, add_special_tokens=False))
            input_ids.append(self.audio_start_id)
            input_ids.extend(self.build_audio_placeholder_ids(token_lens[item_idx]))
            input_ids.append(self.audio_end_id)
            cursor = match.end()

        # Step 8. Reject extra placeholder spans after all audio items are used.
        suffix = prompt_text[cursor:]
        if MOSS_AUDIO_SPAN_RE.search(suffix):
            raise ValueError(
                "Audio placeholder count mismatch: found more placeholder spans "
                "than audio items."
            )
        input_ids.extend(self.tokenizer.encode(suffix, add_special_tokens=False))

        # Step 9. Return tokenizer output plus audio tensors for embed_multimodal.
        data = {
            "input_ids": torch.tensor([input_ids], dtype=torch.long),
            "attention_mask": torch.ones((1, len(input_ids)), dtype=torch.long),
        }
        if audio_batch is not None and audio_data_seqlens is not None:
            data["audio_data"] = audio_batch
            data["audio_data_seqlens"] = audio_data_seqlens
        return BatchFeature(data=data, tensor_type=return_tensors)

    def decode(self, *args: object, **kwargs: object) -> str:
        return self.tokenizer.decode(*args, **kwargs)

    def batch_decode(self, *args: object, **kwargs: object) -> list[str]:
        return self.tokenizer.batch_decode(*args, **kwargs)


class MossAudioProcessingInfo(BaseProcessingInfo):
    def get_hf_config(self) -> MossAudioConfig:
        config = self.ctx.get_hf_config()
        if isinstance(config, MossAudioConfig):
            return config
        return MossAudioConfig(
            audio_config=getattr(config, "audio_config", None),
            language_config=getattr(config, "language_config", None),
            adapter_hidden_size=getattr(config, "adapter_hidden_size", 8192),
            ignore_index=getattr(config, "ignore_index", -100),
            deepstack_num_inject_layers=getattr(
                config, "deepstack_num_inject_layers", None
            ),
        )

    def _get_processor_config_defaults(self) -> dict[str, object]:
        cached_defaults = getattr(self, "_processor_config_defaults", None)
        if cached_defaults is not None:
            return cached_defaults

        model_config = self.ctx.model_config
        for file_name in ("processor_config.json", "preprocessor_config.json"):
            config = get_hf_file_to_dict(
                file_name,
                model_config.model,
                model_config.revision,
            )
            defaults = _filter_moss_audio_processor_config(config)
            if defaults:
                self._processor_config_defaults = defaults
                return defaults

        defaults = {}
        self._processor_config_defaults = defaults
        return defaults

    @staticmethod
    def _get_processor_cache_key(kwargs: Mapping[str, object]) -> tuple[object, ...]:
        mel_config = _normalize_moss_audio_mel_config(
            kwargs.get("mel_config")
            if isinstance(kwargs.get("mel_config"), Mapping)
            else None
        )
        return (
            int(kwargs.get("audio_token_id", MOSS_AUDIO_TOKEN_ID)),
            int(kwargs.get("audio_start_id", MOSS_AUDIO_BOS_TOKEN_ID)),
            int(kwargs.get("audio_end_id", MOSS_AUDIO_EOS_TOKEN_ID)),
            bool(kwargs.get("enable_time_marker", False)),
            tuple(sorted(mel_config.items())),
        )

    def get_hf_processor(self, **kwargs: object) -> MossAudioProcessor:
        merged_kwargs = _merge_moss_audio_processor_configs(
            self._get_processor_config_defaults(),
            self.ctx.get_merged_mm_kwargs({}),
            kwargs,
        )
        mel_config = _normalize_moss_audio_mel_config(
            merged_kwargs.get("mel_config")
            if isinstance(merged_kwargs.get("mel_config"), Mapping)
            else None
        )
        processor_kwargs = {
            "audio_token_id": int(
                merged_kwargs.get("audio_token_id", MOSS_AUDIO_TOKEN_ID)
            ),
            "audio_start_id": int(
                merged_kwargs.get("audio_start_id", MOSS_AUDIO_BOS_TOKEN_ID)
            ),
            "audio_end_id": int(
                merged_kwargs.get("audio_end_id", MOSS_AUDIO_EOS_TOKEN_ID)
            ),
            "enable_time_marker": bool(merged_kwargs.get("enable_time_marker", False)),
            "mel_config": mel_config,
        }

        cache = getattr(self, "_hf_processor_cache", None)
        if cache is None:
            cache = {}
            self._hf_processor_cache = cache

        cache_key = self._get_processor_cache_key(processor_kwargs)
        processor = cache.get(cache_key)
        if processor is not None:
            return processor

        processor = MossAudioProcessor(
            self.get_tokenizer(),
            **processor_kwargs,
        )
        cache[cache_key] = processor
        return processor

    def get_feature_extractor(self, **kwargs: object) -> WhisperFeatureExtractor:
        return self.get_hf_processor(**kwargs).feature_extractor

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"audio": None}

    def get_data_parser(self) -> MultiModalDataParser:
        processor = self.get_hf_processor()
        return MossAudioMultiModalDataParser(
            target_sr=processor.mel_config["mel_sr"],
            target_channels=1,
            expected_hidden_size=self._get_expected_hidden_size(),
        )

    def get_mm_max_tokens_per_item(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> Mapping[str, int] | None:
        if mm_counts.get("audio", 0) <= 0:
            return {}
        processor = self.get_hf_processor()
        raw_mel_len = math.ceil(
            (processor.mel_config["mel_sr"] * DEFAULT_MAX_AUDIO_SECONDS)
            / processor.mel_config["mel_hop_length"]
        )
        return {"audio": MossAudioEncoder.compute_num_audio_tokens(raw_mel_len)}


class MossAudioDummyInputsBuilder(BaseDummyInputsBuilder[MossAudioProcessingInfo]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        num_audios = mm_counts.get("audio", 0)
        return MOSS_AUDIO_PLACEHOLDER * num_audios

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions],
    ) -> MultiModalDataDict:
        num_audios = mm_counts.get("audio", 0)
        audio_overrides = mm_options.get("audio")
        return {
            "audio": self._get_dummy_audios(
                length=16000,
                num_audios=num_audios,
                overrides=audio_overrides,
            )
        }


class MossAudioMultiModalProcessor(BaseMultiModalProcessor[MossAudioProcessingInfo]):
    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        mm_data = dict(mm_data)
        audios = mm_data.pop("audios", [])
        if audios:
            mm_data["audio"] = audios
        mm_kwargs = dict(mm_kwargs)
        processor_kwargs = _filter_moss_audio_processor_config(mm_kwargs)
        tok_kwargs = {
            key: value
            for key, value in tok_kwargs.items()
            if key not in MOSS_AUDIO_PROCESSOR_CONFIG_KEYS
        }
        return self.info.ctx.call_hf_processor(
            self.info.get_hf_processor(**processor_kwargs),
            dict(text=prompt, **mm_data),
            dict(**tok_kwargs),
        )

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return _moss_audio_field_config(hf_inputs)

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        processor = self.info.get_hf_processor(**hf_processor_mm_kwargs)
        out_mm_data = out_mm_kwargs.get_data()
        audio_data_seqlens = out_mm_data.get("audio_data_seqlens")
        if audio_data_seqlens is None:
            audio_token_lens: list[int] = []
        else:
            if isinstance(audio_data_seqlens, torch.Tensor):
                lens = audio_data_seqlens.reshape(-1).tolist()
            else:
                lens = list(audio_data_seqlens)
            audio_token_lens = [
                MossAudioEncoder.compute_num_audio_tokens(int(length))
                for length in lens
            ]

        def get_replacement(
            item_idx: int,
            suffix_token_ids: list[int] | None = None,
        ) -> PromptUpdateDetails[list[int]]:
            num_tokens = audio_token_lens[item_idx]
            if num_tokens == 0:
                raise ValueError("The audio is too short to be represented.")
            audio_token_ids = processor.build_audio_placeholder_ids(num_tokens)
            suffix_token_ids = suffix_token_ids or []
            is_embed = torch.tensor(
                [token_id == processor.audio_token_id for token_id in audio_token_ids],
                dtype=torch.bool,
            )
            return PromptUpdateDetails(
                full=[
                    processor.audio_start_id,
                    *audio_token_ids,
                    processor.audio_end_id,
                    *suffix_token_ids,
                ],
                is_embed=lambda _tokenizer, _seq: torch.cat(
                    [
                        torch.tensor([False]),
                        is_embed,
                        torch.tensor([False]),
                        torch.zeros(len(suffix_token_ids), dtype=torch.bool),
                    ]
                ),
            )

        prompt_update_specs = [
            (
                [
                    processor.audio_start_id,
                    processor.audio_token_id,
                    processor.audio_end_id,
                ],
                [],
            )
        ]
        for suffix in ("", "\n"):
            tokenizer_target = processor.tokenizer.encode(
                MOSS_AUDIO_PLACEHOLDER + suffix,
                add_special_tokens=False,
            )
            suffix_token_ids = processor.tokenizer.encode(
                suffix,
                add_special_tokens=False,
            )
            if any(target == tokenizer_target for target, _ in prompt_update_specs):
                continue
            prompt_update_specs.append((tokenizer_target, suffix_token_ids))

        return [
            PromptReplacement(
                modality="audio",
                target=target,
                replacement=(
                    lambda item_idx, suffix_token_ids=suffix_token_ids: get_replacement(
                        item_idx,
                        suffix_token_ids,
                    )
                ),
            )
            for target, suffix_token_ids in prompt_update_specs
        ]


@MULTIMODAL_REGISTRY.register_processor(
    MossAudioMultiModalProcessor,
    info=MossAudioProcessingInfo,
    dummy_inputs=MossAudioDummyInputsBuilder,
)
class MossAudioModel(nn.Module, SupportsMultiModal, SupportsPP, SupportsLoRA):
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

    embedding_modules = {
        "embed_tokens": "input_embeddings",
        "lm_head": "output_embeddings",
    }

    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "lm_head.": "language_model.lm_head.",
            "language_model.embed_tokens.": "language_model.model.embed_tokens.",
            "language_model.layers.": "language_model.model.layers.",
            "language_model.norm.": "language_model.model.norm.",
        },
        orig_to_new_stacked={
            ".gate_proj": (".gate_up_proj", 0),
            ".up_proj": (".gate_up_proj", 1),
        },
    )

    def get_mm_mapping(self) -> MultiModelKeys:
        return MultiModelKeys.from_string_field(
            language_model="language_model.",
        )

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("audio"):
            return MOSS_AUDIO_PLACEHOLDER
        raise ValueError("Only audio modality is supported")

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        self.vllm_config = vllm_config
        config = vllm_config.model_config.hf_config
        if not isinstance(config, MossAudioConfig):
            config = MossAudioConfig(
                audio_config=getattr(config, "audio_config", None),
                language_config=getattr(config, "language_config", None),
                adapter_hidden_size=getattr(config, "adapter_hidden_size", 8192),
                ignore_index=getattr(config, "ignore_index", -100),
                deepstack_num_inject_layers=getattr(
                    config, "deepstack_num_inject_layers", None
                ),
            )
        self.config = config
        self.quant_config = vllm_config.quant_config
        self.multimodal_config = vllm_config.model_config.multimodal_config

        parallel_config = vllm_config.parallel_config
        tp_size = parallel_config.tensor_parallel_size
        if self.config.adapter_hidden_size % tp_size != 0:
            raise ValueError(
                "MOSS-Audio adapter_hidden_size must be divisible by tensor "
                f"parallel size. Got adapter_hidden_size="
                f"{self.config.adapter_hidden_size} and tensor_parallel_size="
                f"{tp_size}."
            )

        audio_config = MossAudioEncoderConfig.from_config(self.config.audio_config)
        if audio_config.encoder_attention_heads % tp_size != 0:
            raise ValueError(
                "MOSS-Audio encoder_attention_heads must be divisible by "
                "tensor parallel size. Got encoder_attention_heads="
                f"{audio_config.encoder_attention_heads} and "
                f"tensor_parallel_size={tp_size}."
            )
        language_config = self.config.language_config
        self.audio_token_id = MOSS_AUDIO_TOKEN_ID
        self.deepstack_input_embeds: IntermediateTensors | None = None

        with self._mark_tower_model(vllm_config, "audio"):
            self.audio_encoder = MossAudioEncoder(
                audio_config,
                quant_config=self.quant_config,
                prefix=maybe_prefix(prefix, "audio_encoder"),
            )
            self.audio_adapter = GatedMLP(
                input_size=audio_config.output_dim,
                hidden_size=self.config.adapter_hidden_size,
                output_size=language_config.hidden_size,
                quant_config=self.quant_config,
                prefix=maybe_prefix(prefix, "audio_adapter"),
            )

            deepstack_k = len(audio_config.deepstack_encoder_layer_indexes or [])
            if self.config.deepstack_num_inject_layers is not None:
                deepstack_k = min(
                    deepstack_k,
                    int(self.config.deepstack_num_inject_layers),
                )
            self.deepstack_audio_merger_list = nn.ModuleList(
                [
                    GatedMLP(
                        input_size=audio_config.output_dim,
                        hidden_size=self.config.adapter_hidden_size,
                        output_size=language_config.hidden_size,
                        quant_config=self.quant_config,
                        prefix=maybe_prefix(
                            prefix,
                            f"deepstack_audio_merger_list.{layer_idx}",
                        ),
                    )
                    for layer_idx in range(deepstack_k)
                ]
            )

        with self._mark_language_model(vllm_config):
            self.language_model = MossQwen3ForCausalLM(
                vllm_config=vllm_config.with_hf_config(
                    language_config, architectures=["Qwen3ForCausalLM"]
                ),
                prefix=maybe_prefix(prefix, "language_model"),
            )
            self.language_model.deepstack_inject_layer_indices = range(deepstack_k)
            self.language_model.model.deepstack_inject_layer_indices = range(
                deepstack_k
            )

        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )

    @staticmethod
    def _validate_audio_batch_size(
        audio_batch_size: int, audio_data_seqlens: torch.Tensor
    ) -> None:
        if audio_batch_size != audio_data_seqlens.numel():
            raise ValueError(
                "audio_data batch size does not match audio_data_seqlens: "
                f"{audio_batch_size} != {audio_data_seqlens.numel()}."
            )

    @staticmethod
    def _pad_audio_data_list(
        audio_data: list[torch.Tensor],
        audio_data_seqlens: torch.Tensor,
    ) -> torch.Tensor:
        if len(audio_data) == 0:
            raise ValueError("audio_data list must not be empty.")
        MossAudioModel._validate_audio_batch_size(len(audio_data), audio_data_seqlens)

        # pad_sequence needs every item to share the same trailing feature
        # layout, so validate the mel-major audio tensors before transposing.
        first = audio_data[0]
        if not isinstance(first, torch.Tensor):
            raise TypeError("audio_data list items must be torch.Tensor.")
        if first.ndim != 2:
            raise ValueError("audio_data list items must have shape [mel_dim, time].")

        mel_dim = first.shape[0]
        dtype = first.dtype
        device = first.device
        for item in audio_data[1:]:
            if not isinstance(item, torch.Tensor):
                raise TypeError("audio_data list items must be torch.Tensor.")
            if item.ndim != 2:
                raise ValueError(
                    "audio_data list items must have shape [mel_dim, time]."
                )
            if item.shape[0] != mel_dim:
                raise ValueError("audio_data list items must have the same mel_dim.")
            if item.dtype != dtype:
                raise TypeError("audio_data list items must have the same dtype.")
            if item.device != device:
                raise ValueError("audio_data list items must be on the same device.")

        # Each item arrives as [mel_dim, time]. pad_sequence pads along dim 1
        # after converting to [time, mel_dim], then we restore [batch, mel, time].
        time_major = [item.transpose(0, 1) for item in audio_data]
        padded = torch.nn.utils.rnn.pad_sequence(time_major, batch_first=True)
        return padded.transpose(1, 2).contiguous()

    def _parse_and_validate_audio_input(
        self, **kwargs: object
    ) -> MossAudioAudioInputs | None:
        """Normalize and validate model-side audio kwargs.

        If audio_data is provided, this checks that audio_data_seqlens is also
        present, flattens sequence lengths to a long tensor, pads list inputs
        to [batch, mel_dim, time], validates batch-size/sequence-length
        agreement, and rejects empty, non-positive, or downsampled-zero audio
        lengths.
        """
        audio_data = kwargs.pop("audio_data", None)
        audio_data_seqlens = kwargs.pop("audio_data_seqlens", None)
        if audio_data is None:
            return None
        if audio_data_seqlens is None:
            raise ValueError(
                "audio_data_seqlens is required when audio_data is provided."
            )
        if not isinstance(audio_data_seqlens, torch.Tensor):
            audio_data_seqlens = torch.tensor(audio_data_seqlens, dtype=torch.long)
        audio_data_seqlens = audio_data_seqlens.to(dtype=torch.long).reshape(-1)

        if isinstance(audio_data, list):
            audio_data = self._pad_audio_data_list(audio_data, audio_data_seqlens)
        elif isinstance(audio_data, torch.Tensor):
            if audio_data.ndim == 3:
                self._validate_audio_batch_size(audio_data.shape[0], audio_data_seqlens)
        else:
            raise TypeError("audio_data must be a torch.Tensor or list[torch.Tensor].")

        audio_token_lens = MossAudioEncoder._compute_downsampled_length(
            audio_data_seqlens
        )
        if (
            audio_data_seqlens.numel() == 0
            or torch.any(audio_data_seqlens <= 0).item()
            or torch.any(audio_token_lens <= 0).item()
        ):
            raise ValueError("The audio is too short to be represented.")
        return MossAudioAudioInputs(
            audio_data=audio_data,
            audio_data_seqlens=audio_data_seqlens,
        )

    def _process_audio_input(
        self,
        audio_input: MossAudioAudioInputs,
    ) -> tuple[torch.Tensor, ...]:
        """Run the audio encoder and return one embedding tensor per audio.

        Example:
            audio_data=[2, 128, 1200], audio_data_seqlens=[800, 1200]
            -> returns (audio0_embeds, audio1_embeds), split by token length
            -> DeepStack packs each item as [main, layer0, ...] on dim -1
        """
        audio_data = audio_input["audio_data"]
        audio_data_seqlens = audio_input["audio_data_seqlens"]
        last_hidden_state, deepstack = self.audio_encoder(
            audio_data.to(self.audio_encoder.dtype),
            feature_lens=audio_data_seqlens,
            output_deepstack_hidden_states=len(self.deepstack_audio_merger_list) > 0,
        )
        audio_embeds = self.audio_adapter(last_hidden_state)
        audio_lengths = MossAudioEncoder._compute_downsampled_length(
            audio_data_seqlens.to(device=audio_embeds.device, dtype=torch.long)
        ).tolist()
        main_embeddings = tuple(audio_embeds.squeeze(0).split(audio_lengths, dim=0))

        deepstack_embeddings: list[tuple[torch.Tensor, ...]] = []
        if deepstack is not None:
            if len(deepstack) < len(self.deepstack_audio_merger_list):
                raise RuntimeError(
                    "DeepStack output count does not match configured audio "
                    "merger count."
                )
            for idx, hidden_states in enumerate(
                deepstack[: len(self.deepstack_audio_merger_list)]
            ):
                ds_embeds = self.deepstack_audio_merger_list[idx](hidden_states)
                deepstack_embeddings.append(
                    tuple(ds_embeds.squeeze(0).split(audio_lengths, dim=0))
                )

        if not deepstack_embeddings:
            return main_embeddings

        return tuple(
            torch.cat(
                [
                    main_embedding,
                    *(
                        layer_embeddings[item_idx]
                        for layer_embeddings in deepstack_embeddings
                    ),
                ],
                dim=-1,
            )
            for item_idx, main_embedding in enumerate(main_embeddings)
        )

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        audio_input = self._parse_and_validate_audio_input(**kwargs)
        if audio_input is None:
            return ()
        return self._process_audio_input(audio_input)

    def _split_multimodal_embeddings(
        self,
        multimodal_embeddings: MultiModalEmbeddings,
        hidden_size: int,
    ) -> tuple[tuple[torch.Tensor, ...], tuple[tuple[torch.Tensor, ...], ...]]:
        """Unpack audio embeddings before merging them into token embeddings.

        embed_input_ids calls this on the output of embed_multimodal. Plain
        audio embeddings already have width hidden_size and are returned as the
        main embeddings for _merge_multimodal_embeddings. When DeepStack is
        enabled, _process_audio_input packs each audio item as
        [main, layer0, layer1, ...] along the last dimension so the standard
        multimodal path can carry a single embedding object. This method splits
        that packed layout back into main embeddings plus per-layer DeepStack
        embeddings, which _cache_deepstack_input_embeds scatters and forward
        passes into MossQwen3Model for layer injection.
        """
        if isinstance(multimodal_embeddings, torch.Tensor):
            embeddings = tuple(multimodal_embeddings.unbind(0))
        else:
            embeddings = tuple(multimodal_embeddings)

        if len(embeddings) == 0:
            return (), ()

        deepstack_count = len(self.deepstack_audio_merger_list)
        if all(embedding.shape[-1] == hidden_size for embedding in embeddings):
            return embeddings, ()

        packed_hidden_size = hidden_size * (deepstack_count + 1)
        if deepstack_count == 0 or any(
            embedding.shape[-1] != packed_hidden_size for embedding in embeddings
        ):
            got = [int(embedding.shape[-1]) for embedding in embeddings]
            raise ValueError(
                "MOSS-Audio multimodal embedding width mismatch: expected "
                f"{hidden_size} or {packed_hidden_size}, got {got}."
            )

        split_by_item = [
            torch.split(embedding, hidden_size, dim=-1) for embedding in embeddings
        ]
        main_embeddings = tuple(parts[0] for parts in split_by_item)
        deepstack_embeddings = tuple(
            tuple(parts[layer_idx + 1] for parts in split_by_item)
            for layer_idx in range(deepstack_count)
        )
        return main_embeddings, deepstack_embeddings

    def _cache_deepstack_input_embeds(
        self,
        inputs_embeds: torch.Tensor,
        deepstack_embeddings: tuple[tuple[torch.Tensor, ...], ...],
        is_multimodal: torch.Tensor,
    ) -> None:
        if len(deepstack_embeddings) == 0:
            self.deepstack_input_embeds = None
            return
        flat_by_layer = [
            torch.cat(layer_embeds, dim=0).to(
                device=inputs_embeds.device, dtype=inputs_embeds.dtype
            )
            for layer_embeds in deepstack_embeddings
        ]
        num_mm_tokens = int(is_multimodal.sum().item())
        if any(layer.shape[0] != num_mm_tokens for layer in flat_by_layer):
            got = [int(layer.shape[0]) for layer in flat_by_layer]
            raise ValueError(
                "DeepStack audio token count mismatch: "
                f"expected {num_mm_tokens}, got {got}."
            )
        data = {}
        for layer_idx, layer_embeds in enumerate(flat_by_layer):
            scattered = inputs_embeds.new_zeros(inputs_embeds.shape)
            scattered[is_multimodal] = layer_embeds
            data[f"deepstack_input_embeds_{layer_idx}"] = scattered
        self.deepstack_input_embeds = IntermediateTensors(data)

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: MultiModalEmbeddings | None = None,
        *,
        is_multimodal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        inputs_embeds = self._embed_text_input_ids(
            input_ids,
            self.language_model.embed_input_ids,
            is_multimodal=is_multimodal,
        )

        self.deepstack_input_embeds = None
        if multimodal_embeddings is None or len(multimodal_embeddings) == 0:
            return inputs_embeds
        is_multimodal = _require_is_multimodal(is_multimodal)
        multimodal_embeddings, deepstack_embeddings = self._split_multimodal_embeddings(
            multimodal_embeddings,
            hidden_size=int(inputs_embeds.shape[-1]),
        )

        inputs_embeds = _merge_multimodal_embeddings(
            inputs_embeds=inputs_embeds,
            multimodal_embeddings=multimodal_embeddings,
            is_multimodal=is_multimodal,
        )
        self._cache_deepstack_input_embeds(
            inputs_embeds,
            deepstack_embeddings,
            is_multimodal,
        )
        return inputs_embeds

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors:
        if intermediate_tensors is None:
            deepstack_input_embeds = self.deepstack_input_embeds
        else:
            # Non-first PP ranks consume hidden states from intermediate_tensors.
            # The executor may still pass dummy inputs_embeds during profiling.
            inputs_embeds = None
            deepstack_input_embeds = intermediate_tensors
        hidden_states = self.language_model(
            input_ids,
            positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
            deepstack_input_embeds=deepstack_input_embeds,
        )
        self.deepstack_input_embeds = None
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        return self.language_model.compute_logits(hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=["audio_encoder.embed_positions"],
        )
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)
