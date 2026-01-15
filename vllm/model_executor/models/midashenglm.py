# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright 2025 Horizon team, Xiaomi MiLM Plus.
# Copyright 2024 The Qwen team.
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
"""Inference-only MiDashengLM model compatible with HuggingFace weights."""

import collections
import collections.abc
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Annotated, Any, TypeAlias, cast

import numpy as np
import torch
import torch.nn as nn
import torchaudio.functional as F
from torch.nn.functional import scaled_dot_product_attention
from transformers import BatchFeature

from vllm.config import VllmConfig
from vllm.config.multimodal import BaseDummyOptions
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.layers.activation import get_act_fn
from vllm.model_executor.layers.conv import Conv2dLayer
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (
    MultiModalDataDict,
    MultiModalFieldConfig,
    MultiModalKwargsItems,
)
from vllm.multimodal.parse import MultiModalDataItems, MultiModalDataParser
from vllm.multimodal.processing import (
    BaseDummyInputsBuilder,
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    PromptReplacement,
    PromptUpdate,
    PromptUpdateDetails,
)
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.configs.midashenglm import DashengConfig
from vllm.utils.tensor_schema import TensorSchema, TensorShape

from .interfaces import MultiModalEmbeddings, SupportsMultiModal, SupportsPP
from .utils import AutoWeightsLoader, init_vllm_registered_model, maybe_prefix

_Tuple2: TypeAlias = int | tuple[int, int] | Sequence[int]


def _resolve_tuple2(x: _Tuple2) -> tuple[int, int]:
    if isinstance(x, collections.abc.Sequence):
        assert len(x) == 2, (
            f"Expected a sequence of length 2, got {x} with length {len(x)}"
        )
        return cast(tuple[int, int], tuple(x))
    return (x, x)


def calculate_mel_frames_dasheng(
    audio_length_samples: int,
    n_fft: int = 512,
    hop_size: int = 160,
    dasheng_subsampling: int = 4,
    center=True,
    model_subsampling: int = 5,
) -> int:
    """Calculate the number of Mel-spectrogram frames."""
    if center:
        audio_length_samples = audio_length_samples + n_fft

    return (
        int(1 + ((audio_length_samples - n_fft) / hop_size))
        // dasheng_subsampling
        // model_subsampling
    )


class AudioPatchEmbed(nn.Module):
    def __init__(
        self,
        input_size: _Tuple2 = 64,
        patch_size: _Tuple2 = 16,
        patch_stride: _Tuple2 = 16,
        in_chans: int = 1,
        embed_dim: int = 768,
        norm_layer: Callable | None = None,
        flatten: bool = False,
    ):
        super().__init__()
        self.input_size = _resolve_tuple2(input_size)
        self.patch_size = _resolve_tuple2(patch_size)
        self.patch_stride = _resolve_tuple2(patch_stride)
        self.grid_size = (
            self.input_size[0] // self.patch_stride[0],
            self.input_size[1] // self.patch_stride[1],
        )
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = Conv2dLayer(
            in_chans,
            embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_stride,
        )
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        if self.flatten:
            x = torch.permute(
                torch.flatten(x, 2, 3), (0, 2, 1)
            )  # rearrange(x, "b c f t -> b (f t) c")
        x = self.norm(x)
        return x


class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class DashengMlp(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = ColumnParallelLinear(
            input_size=in_features,
            output_size=hidden_features,
            quant_config=quant_config,
            prefix=f"{prefix}.fc1",
        )
        self.act = get_act_fn("gelu")
        self.fc2 = RowParallelLinear(
            input_size=hidden_features,
            output_size=out_features,
            quant_config=quant_config,
            prefix=f"{prefix}.fc2",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self.fc1(x)
        x = self.act(x)
        x, _ = self.fc2(x)
        return x


class DashengAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.embed_dim = dim
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
        self.scale = self.head_dim**-0.5

        self.qkv = QKVParallelLinear(
            hidden_size=self.embed_dim,
            head_size=self.head_dim,
            total_num_heads=self.total_num_heads,
            total_num_kv_heads=self.total_num_heads,
            bias=qkv_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv",
        )
        self.proj = RowParallelLinear(
            input_size=dim,
            output_size=dim,
            quant_config=quant_config,
            prefix=f"{prefix}.proj",
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None):
        B, N, C = x.shape

        qkv, _ = self.qkv(x)
        qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        x = scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask[:, None, None, :] if mask is not None else None,
        )

        x = x.transpose(1, 2).reshape(B, N, C)
        x, _ = self.proj(x)
        return x


class DashengBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        init_values: float | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = DashengAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
        )
        self.ls1 = (
            LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        )

        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = DashengMlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )
        self.ls2 = (
            LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        )

    # Kwargs usually has a mask parameter that is passed to Attention
    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x + self.ls1(self.attn(self.norm1(x), mask))
        x = x + self.ls2(self.mlp(self.norm2(x)))
        return x


class DashengFrontend(nn.Module):
    def __init__(self, config: DashengConfig):
        super().__init__()
        self.config = config

        spectrogram_window = torch.hann_window(self.config.win_length)
        self.register_buffer(
            "spectrogram_window",
            spectrogram_window,
            persistent=False,
        )
        self.spectrogram_window: torch.Tensor

        melscale_fbanks = F.melscale_fbanks(
            n_freqs=self.config.n_fft // 2 + 1,
            f_min=self.config.f_min,
            f_max=self.config.f_max,
            n_mels=self.config.n_mels,
            sample_rate=self.config.sample_rate,
        )
        self.register_buffer("melscale_fbanks", melscale_fbanks, persistent=False)
        self.melscale_fbanks: torch.Tensor

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        spectrogram = F.spectrogram(
            waveform=waveform.to(torch.float32),
            pad=0,
            window=self.spectrogram_window,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            win_length=self.config.win_length,
            power=2,
            normalized=False,
            center=self.config.center,
        )
        mel_spectrogram = (spectrogram.mT @ self.melscale_fbanks.to(torch.float32)).mT
        # x has shape [batch, freq, time].
        # F.amplitude_to_DB accepts inputs shaped as:
        #   - [freq, time]
        #   - [channel, freq, time]
        #   - [..., channel, freq, time]
        # Here we insert a channel dimension of size 1 before calling it,
        # then remove that extra dimension afterward.
        log_mel_spectrogram = F.amplitude_to_DB(
            mel_spectrogram.unsqueeze(1),
            multiplier=10,
            amin=1e-10,
            db_multiplier=0,
            top_db=120,
        ).squeeze(1)
        return log_mel_spectrogram.to(waveform.dtype)


class DashengAudioTransformer(nn.Module):
    def __init__(
        self,
        config: DashengConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()

        self.target_length = config.target_length
        self.hop_length = config.hop_length

        self.front_end = DashengFrontend(config)

        self.init_bn = nn.BatchNorm2d(config.n_mels, momentum=0.01)

        self.patch_embed = AudioPatchEmbed(
            input_size=(config.n_mels, config.target_length),
            embed_dim=config.embed_dim,
            in_chans=config.input_channels,
            patch_size=config.patch_size,
            flatten=False,
            patch_stride=config.patch_stride,
        )

        self.time_pos_embed = nn.Parameter(
            torch.empty(1, config.embed_dim, 1, self.patch_embed.grid_size[1])
        )
        self.freq_pos_embed = nn.Parameter(
            torch.empty(1, config.embed_dim, self.patch_embed.grid_size[0], 1)
        )
        self.blocks = nn.ModuleList(
            DashengBlock(
                dim=config.embed_dim,
                num_heads=config.num_heads,
                mlp_ratio=config.mlp_ratio,
                qkv_bias=config.qkv_bias,
                init_values=config.init_values,
                quant_config=quant_config,
                prefix=f"{prefix}.blocks.{i}",
            )
            for i in range(config.depth)
        )
        self.norm = nn.LayerNorm(config.embed_dim, eps=1e-6)

    def forward_features(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        t = x.shape[-1]
        x = x + self.time_pos_embed[:, :, :, :t]
        x = (
            x + self.freq_pos_embed[:, :, :, :]
        )  # Just to support __getitem__ in posembed
        x = torch.permute(
            torch.flatten(x, 2, 3), (0, 2, 1)
        )  # rearrange(x, "b c f t -> b (f t) c")
        for block in self.blocks:
            x = block(x, mask)
        x = self.norm(x)
        return x

    def _to_mask(self, lengths: torch.Tensor, max_length: int) -> torch.Tensor:
        batch_size = len(lengths)
        idx = torch.arange(max_length, device=lengths.device)
        idx = idx.repeat(batch_size).view(batch_size, max_length)
        mask = (idx < lengths.unsqueeze(-1)).bool()
        return mask

    def forward(
        self,
        x: torch.Tensor,
        x_length: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        x = self.front_end(x)
        x = x.to(self.time_pos_embed.dtype)
        target_length_in_patches = self.target_length // 4
        x = x.unsqueeze(1)
        x = torch.permute(x, (0, 2, 1, 3))
        x = self.init_bn(x)
        x = torch.permute(x, (0, 2, 1, 3))

        x = self.patch_embed(x)
        t = x.shape[-1]

        input_splits = x.split(target_length_in_patches, dim=-1)

        if x_length is not None:
            assert len(x_length) == len(x), (
                "batchsizes of input x and x_length need to be same"
            )
            assert x_length.ndim == 1, "Lengths are of size (B,)"
            scaled_lengths = (x_length / (self.hop_length * 4)).long()
            mask = self._to_mask(max_length=t, lengths=scaled_lengths)
            split_masks = mask.split(target_length_in_patches, dim=-1)
        else:
            mask = None
            split_masks = [None] * len(input_splits)

        outputs = []

        for split_x, split_mask in zip(input_splits, split_masks):
            forward_kwargs = {}
            forward_kwargs["mask"] = split_mask
            split_x = self.forward_features(split_x, **forward_kwargs)
            outputs.append(split_x)
        x = torch.cat(outputs, dim=1)
        return x, mask


class AudioProjectorSubsample(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        downsample_rate=5,
        dtype: torch.dtype | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.k = downsample_rate
        self.net = nn.Sequential(
            ColumnParallelLinear(
                input_size=in_dim * self.k,
                output_size=out_dim,
                quant_config=quant_config,
                prefix=f"{prefix}.net.0",
                return_bias=False,
            ),
            get_act_fn("gelu"),
            RowParallelLinear(
                input_size=out_dim,
                output_size=out_dim,
                quant_config=quant_config,
                prefix=f"{prefix}.net.2",
                return_bias=False,
            ),
        )

    def forward(self, x, mask=None):
        batch_size, seq_len, dim = x.shape
        num_frames_to_discard = seq_len % self.k
        if num_frames_to_discard > 0:
            x = x[:, :-num_frames_to_discard, :]
            if mask is not None:
                mask = mask[:, :-num_frames_to_discard]
        if mask is None:
            mask = torch.ones(x.shape[:-1], dtype=torch.long, device=x.device)
        x = x.reshape(
            batch_size, -1, self.k * dim
        )  # rearrange(x, "b (s k) d -> b s (k d)", k=self.k)
        for layer in self.net:
            x = layer(x)
        mask = mask.reshape(
            batch_size, -1, self.k
        )  # rearrange(mask, "b (s k) -> b s k", k=self.k)
        mask = mask.any(dim=-1).long()
        return x, mask


# === Audio Inputs === #
class MiDashengLMAudioInputs(TensorSchema):
    """

    Dimensions:
        - bn: Batch size * number of audios
        - p: Number of sampling points
    """

    input_values: Annotated[torch.Tensor, TensorShape("n", "p")]
    audio_length: Annotated[torch.Tensor, TensorShape("n")]


class MiDashengLMProcessingInfo(BaseProcessingInfo):
    def get_hf_config(self):
        return self.ctx.get_hf_config()

    def get_feature_extractor(self):
        hf_processor = self.get_hf_processor()
        feature_extractor = hf_processor.feature_extractor
        return feature_extractor

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"audio": None}

    def get_min_audio_len(self):
        return 3200

    def get_max_audio_len(self):
        return 160000


class MiDashengLMDummyInputsBuilder(BaseDummyInputsBuilder[MiDashengLMProcessingInfo]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        num_audios = mm_counts.get("audio", 0)

        hf_processor = self.info.get_hf_processor()
        audio_token = hf_processor.audio_token
        audio_bos_token = hf_processor.audio_bos_token
        audio_eos_token = hf_processor.audio_eos_token

        single_audio_text = f"{audio_bos_token}{audio_token}{audio_eos_token}"
        return single_audio_text * num_audios

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions] | None = None,
    ) -> MultiModalDataDict:
        num_audios = mm_counts.get("audio", 0)

        audio_overrides = mm_options.get("audio") if mm_options else None

        return {
            "audio": self._get_dummy_audios(
                length=self.info.get_max_audio_len(),
                num_audios=num_audios,
                overrides=audio_overrides,
            )
        }


class MiDashengLMMultiModalProcessor(
    BaseMultiModalProcessor[MiDashengLMProcessingInfo]
):
    def _get_data_parser(self) -> MultiModalDataParser:
        feature_extractor = self.info.get_feature_extractor()
        return MultiModalDataParser(target_sr=feature_extractor.sampling_rate)

    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, Any],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        audios = mm_data.pop("audios", [])

        # + Padding
        min_audio_len = self.info.get_min_audio_len()
        processed_audios = [
            np.pad(
                audio,
                (0, min_audio_len - audio.shape[-1]),
                mode="constant",
                constant_values=0,
            )
            if isinstance(audio, np.ndarray) and audio.shape[-1] < min_audio_len
            else audio
            for audio in audios
        ]

        if processed_audios:
            mm_data["audio"] = processed_audios

        if not mm_data.get("audio", []):
            prompt_ids = self.info.get_tokenizer().encode(prompt)
            prompt_ids = self._apply_hf_processor_tokens_only(prompt_ids)
            return BatchFeature(dict(input_ids=[prompt_ids]), tensor_type="pt")

        mm_kwargs = dict(
            **mm_kwargs,
        )

        return super()._call_hf_processor(
            prompt=prompt,
            mm_data=mm_data,
            mm_kwargs=mm_kwargs,
            tok_kwargs=tok_kwargs,
        )

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return dict(
            input_values=MultiModalFieldConfig.batched("audio"),
            audio_length=MultiModalFieldConfig.batched("audio"),
        )

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        processor = self.info.get_hf_processor(**hf_processor_mm_kwargs)
        tokenizer = self.info.get_tokenizer()
        vocab = tokenizer.get_vocab()

        audio_token = getattr(processor, "audio_token", "<|AUDIO|>")
        audio_token_id = vocab[audio_token]

        out_mm_data = out_mm_kwargs.get_data()
        audio_length = out_mm_data.get("audio_length")
        if audio_length is None:
            audio_output_lengths = []
        else:
            audio_length_np = (
                audio_length.cpu().numpy()
                if isinstance(audio_length, torch.Tensor)
                else audio_length
            )
            audio_output_lengths = [
                max(1, calculate_mel_frames_dasheng(int(length)))  # at least one frame
                for length in audio_length_np
            ]

        def get_replacement_midashenglm(item_idx: int):
            num_features = audio_output_lengths[item_idx]
            audio_tokens = [audio_token_id] * num_features

            return PromptUpdateDetails.select_token_id(
                audio_tokens,
                embed_token_id=audio_token_id,
            )

        return [
            PromptReplacement(
                modality="audio",
                target=audio_token,
                replacement=get_replacement_midashenglm,
            )
        ]


@MULTIMODAL_REGISTRY.register_processor(
    MiDashengLMMultiModalProcessor,
    info=MiDashengLMProcessingInfo,
    dummy_inputs=MiDashengLMDummyInputsBuilder,
)
class MiDashengLMModel(nn.Module, SupportsMultiModal, SupportsPP):
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

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("audio"):
            return "<|audio_bos|><|AUDIO|><|audio_eos|>"

        raise ValueError("Only audio modality is supported")

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config

        # Initialize audio components
        self.audio_encoder = DashengAudioTransformer(
            config.audio_encoder_config,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "audio_encoder"),
        )
        self.audio_projector = AudioProjectorSubsample(
            in_dim=config.audio_encoder_config.embed_dim,
            out_dim=config.text_config.hidden_size,
            downsample_rate=config.subsample_factor,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "audio_projector"),
        )

        # Initialize language model (decoder)
        self.decoder = init_vllm_registered_model(
            vllm_config=vllm_config,
            hf_config=config.text_config,
            prefix=maybe_prefix(prefix, "decoder"),
            architectures=["Qwen2ForCausalLM"],
        )

        self.quant_config = quant_config
        self.make_empty_intermediate_tensors = (
            self.decoder.make_empty_intermediate_tensors
        )

    def _parse_and_validate_audio_input(
        self, **kwargs: object
    ) -> MiDashengLMAudioInputs | None:
        input_values = kwargs.pop("input_values", None)
        audio_length = kwargs.pop("audio_length", None)

        if input_values is None:
            return None

        if isinstance(input_values, list):
            input_values = torch.nn.utils.rnn.pad_sequence(
                input_values,
                batch_first=True,
            )

        return MiDashengLMAudioInputs(
            input_values=input_values,
            audio_length=audio_length,
        )

    def _process_audio_input(
        self,
        audio_input: MiDashengLMAudioInputs,
    ) -> tuple[torch.Tensor, ...]:
        # Process audio through encoder and projector
        input_values = audio_input["input_values"]
        audio_length = audio_input["audio_length"]

        encoder_out, encoder_atts = self.audio_encoder(input_values, audio_length)
        audio_embeddings, _ = self.audio_projector(encoder_out, encoder_atts)
        audio_embeddings = audio_embeddings.to(audio_input["input_values"].dtype)
        batch_size, max_audio_tokens, embed_dim = audio_embeddings.shape

        audio_output_lengths = [
            max(1, calculate_mel_frames_dasheng(int(length)))  # at least one frame
            for length in audio_length.tolist()
        ]
        audio_output_lengths = torch.tensor(
            audio_output_lengths,
            device=audio_embeddings.device,
        )

        audio_feature_mask = torch.arange(
            max_audio_tokens, device=audio_embeddings.device
        ).unsqueeze(0).expand(
            batch_size, max_audio_tokens
        ) < audio_output_lengths.unsqueeze(1)

        masked_audio_features = audio_embeddings[audio_feature_mask].view(-1, embed_dim)

        return torch.split(masked_audio_features, audio_output_lengths.tolist())

    def get_language_model(self) -> torch.nn.Module:
        return self.decoder

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        audio_input = self._parse_and_validate_audio_input(**kwargs)

        if audio_input is None:
            return []
        return self._process_audio_input(audio_input)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors:
        if intermediate_tensors is not None:
            inputs_embeds = None

        return self.decoder.model(
            input_ids,
            positions,
            intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        return self.decoder.compute_logits(hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)
