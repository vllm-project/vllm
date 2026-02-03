# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Inference-only FunAudioChat model compatible with HuggingFace weights.

FunAudioChat is a Qwen3 text model augmented with:
  - a continuous audio encoder (Whisper-mel frontend + transformer)
  - a discrete audio encoder (speech tokenizer + projector)

In the HF implementation, audio features are scattered into `<|AUDIO|>` token
positions via `inputs_embeds`, while `position_ids` (RoPE) remains standard 1D.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Mapping, Sequence
from functools import cached_property
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from transformers import PreTrainedTokenizerFast, WhisperFeatureExtractor
from transformers.activations import get_activation
from transformers.feature_extraction_utils import BatchFeature
from transformers.modeling_outputs import BaseModelOutput

from vllm.config import VllmConfig
from vllm.config.multimodal import BaseDummyOptions
from vllm.model_executor.layers.attention.mm_encoder_attention import MMEncoderAttention
from vllm.model_executor.layers.linear import QKVParallelLinear, RowParallelLinear
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (
    MultiModalDataDict,
    MultiModalFieldConfig,
    MultiModalKwargsItems,
)
from vllm.multimodal.parse import (
    AudioProcessorItems,
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
from vllm.utils.import_utils import _has_module

from .interfaces import MultiModalEmbeddings, SupportsMultiModal, SupportsPP
from .utils import AutoWeightsLoader, init_vllm_registered_model, maybe_prefix


class _SinusoidsPositionEmbedding(nn.Module):
    def __init__(self, length: int, channels: int, max_timescale: float = 10000.0):
        super().__init__()
        if channels % 2 != 0:
            raise ValueError("SinusoidsPositionEmbedding needs even channels input")

        log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
        inv_timescales = torch.exp(
            -log_timescale_increment * torch.arange(channels // 2).float()
        )
        scaled_time = (
            torch.arange(length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
        )
        self.register_buffer(
            "positional_embedding",
            torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1),
            persistent=False,
        )


class FunAudioChatAudioAttention(nn.Module):
    """Multi-headed attention used inside the continuous audio tower."""

    def __init__(self, config: Any):
        super().__init__()
        self.embed_dim = int(config.d_model)
        self.total_num_heads = int(config.encoder_attention_heads)
        self.dropout = float(getattr(config, "attention_dropout", 0.0))
        self.head_dim = self.embed_dim // self.total_num_heads
        self.num_key_value_groups = 1  # needed for eager attention
        self.config = config

        if self.head_dim * self.total_num_heads != self.embed_dim:
            raise ValueError(
                "embed_dim must be divisible by num_heads "
                f"(got embed_dim={self.embed_dim}, "
                f"num_heads={self.total_num_heads})."
            )
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = 0.0
        self.is_decoder = False
        self.is_causal = False

        self.qkv_proj = QKVParallelLinear(
            self.embed_dim,
            self.head_dim,
            self.total_num_heads,
            bias=True,
        )
        self.num_heads = self.qkv_proj.num_heads
        self.num_kv_heads = self.qkv_proj.num_kv_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim

        self.attn = MMEncoderAttention(
            num_heads=self.num_heads,
            head_size=self.head_dim,
            scale=self.scaling,
            num_kv_heads=self.num_kv_heads,
            prefix="funaudiochat_audio_tower.attn",
        )
        self.out_proj = RowParallelLinear(
            self.embed_dim,
            self.embed_dim,
            bias=True,
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
        ]

        params_dict = dict(self.named_parameters())
        with torch.no_grad():
            if self.qkv_proj.bias is not None:
                # HF FunAudioChat uses bias=False for k_proj. Ensure the missing
                # shard starts as zeros, while allowing q/v shards to load.
                self.qkv_proj.bias.zero_()

        loaded_params: set[str] = set()
        for name, loaded_weight in weights:
            for param_name, shard_name, shard_id in stacked_params_mapping:
                if shard_name not in name:
                    continue
                name = name.replace(shard_name, param_name)
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
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

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor:
        del kwargs
        del attention_mask
        seq_length, _ = hidden_states.size()

        qkv, _ = self.qkv_proj(hidden_states)
        query_states, key_states, value_states = qkv.split(
            [self.q_size, self.kv_size, self.kv_size], dim=-1
        )

        max_seqlen: torch.Tensor | None = None
        if cu_seqlens is not None:
            max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max()

        attn_output = self.attn(
            query_states.reshape(1, seq_length, self.q_size),
            key_states.reshape(1, seq_length, self.kv_size),
            value_states.reshape(1, seq_length, self.kv_size),
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        ).reshape(seq_length, -1)

        output, _ = self.out_proj(attn_output)
        return output


class FunAudioChatAudioEncoderLayer(nn.Module):
    def __init__(self, config: Any):
        super().__init__()
        self.embed_dim = int(config.d_model)
        self.self_attn = FunAudioChatAudioAttention(config)
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.dropout = float(config.dropout)
        self.activation_fn = get_activation(str(config.activation_function))
        self.activation_dropout = float(config.activation_dropout)
        self.fc1 = nn.Linear(self.embed_dim, int(config.encoder_ffn_dim))
        self.fc2 = nn.Linear(int(config.encoder_ffn_dim), self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **kwargs: object,
    ) -> tuple[torch.Tensor]:
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            cu_seqlens=cu_seqlens,
            attention_mask=attention_mask,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(
            hidden_states, p=self.activation_dropout, training=self.training
        )
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(
            hidden_states, p=self.dropout, training=self.training
        )
        hidden_states = residual + hidden_states

        return (hidden_states,)


class FunAudioChatAudioEncoder(nn.Module):
    """Continuous audio tower."""

    def __init__(self, config: Any):
        super().__init__()
        self.config = config

        embed_dim = int(config.d_model)
        self.num_mel_bins = int(config.num_mel_bins)
        self.max_source_positions = int(config.max_source_positions)
        self.embed_scale = (embed_dim**0.5) if bool(config.scale_embedding) else 1.0
        self.n_window = int(config.n_window)

        self.conv1 = nn.Conv1d(self.num_mel_bins, embed_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(embed_dim, embed_dim, kernel_size=3, stride=2, padding=1)
        self.layers = nn.ModuleList(
            [
                FunAudioChatAudioEncoderLayer(config)
                for _ in range(int(config.encoder_layers))
            ]
        )
        self.ln_post = nn.LayerNorm(embed_dim)
        self.avg_pooler = nn.AvgPool1d(2, stride=2)
        self.proj = nn.Linear(embed_dim, int(config.output_dim))
        self.positional_embedding = _SinusoidsPositionEmbedding(
            self.max_source_positions, embed_dim
        )

        # Present in HF weights even if unused during S2T.
        self.audio_bos_eos_token = nn.Embedding(2, int(config.output_dim))

    @property
    def dtype(self) -> torch.dtype:
        return self.conv1.weight.dtype

    def _prepare_attention_mask(
        self, inputs_tensor: torch.Tensor, cu_seqlens: torch.Tensor
    ) -> torch.Tensor | None:
        if getattr(self.config, "_attn_implementation", "eager") == "flash_attention_2":
            return None

        seq_length = inputs_tensor.shape[0]
        attention_mask = torch.full(
            (1, 1, seq_length, seq_length),
            torch.finfo(inputs_tensor.dtype).min,
            device=inputs_tensor.device,
            dtype=inputs_tensor.dtype,
        )
        for i in range(1, len(cu_seqlens)):
            start = int(cu_seqlens[i - 1].item())
            end = int(cu_seqlens[i].item())
            attention_mask[..., start:end, start:end] = 0
        return attention_mask

    def forward(
        self,
        input_features: torch.Tensor,
        feature_lens: torch.Tensor,
        aftercnn_lens: torch.Tensor,
        speech_maxlen: int,
        **kwargs: object,
    ) -> BaseModelOutput:
        # For max-length audio (300s => ~7500 speech frames at 25Hz), the
        # Torch SDPA path can be prohibitively memory hungry (~O(n^2) inside the
        # longest chunks). Require FlashAttention for such inputs to avoid OOM
        # and performance cliffs.
        if int(speech_maxlen) >= 7500:
            if not _has_module("flash_attn"):
                raise RuntimeError(
                    "FunAudioChat long audio (~300s) requires FlashAttention-2 "
                    "for the continuous audio tower, but `flash_attn` is not "
                    "installed in the runtime environment."
                )
            if not getattr(
                self.layers[0].self_attn.attn, "is_flash_attn_backend", False
            ):
                raise RuntimeError(
                    "FunAudioChat long audio (~300s) requires FlashAttention for the "
                    "continuous audio tower, but the selected MM encoder attention "
                    "backend is not FlashAttention."
                )

        # Handle empty / invalid items (feature_lens == 0) without crashing.
        original_batch_size = int(feature_lens.size(0))
        device = input_features.device

        valid_mask = feature_lens > 0
        valid_indices = torch.where(valid_mask)[0]

        if valid_indices.numel() == 0:
            output_dim = int(self.proj.out_features)
            return BaseModelOutput(
                last_hidden_state=torch.zeros(
                    (original_batch_size, speech_maxlen, output_dim),
                    device=device,
                    dtype=self.proj.weight.dtype,
                )
            )

        input_features_list = input_features.split(feature_lens.tolist(), dim=1)
        valid_input_features_list = [input_features_list[int(i)] for i in valid_indices]
        valid_input_features = torch.cat(valid_input_features_list, dim=1)

        valid_feature_lens = feature_lens[valid_mask]
        valid_aftercnn_lens = aftercnn_lens[valid_mask]

        chunk_num = torch.ceil(valid_feature_lens / (self.n_window * 2)).long()

        chunk_lengths_list: list[int] = []
        full_chunk_len = self.n_window * 2
        for i, length in enumerate(valid_feature_lens):
            num_chunks_for_sample = int(chunk_num[i].item())
            if num_chunks_for_sample == 0:
                continue
            chunk_lengths_list.extend([full_chunk_len] * (num_chunks_for_sample - 1))
            last_chunk_len = int(length.item()) % full_chunk_len
            if last_chunk_len == 0:
                last_chunk_len = full_chunk_len
            chunk_lengths_list.append(last_chunk_len)

        chunk_lengths = torch.tensor(
            chunk_lengths_list, dtype=torch.long, device=device
        )

        chunk_list = valid_input_features.split(chunk_lengths.tolist(), dim=1)
        padded_feature, padded_mask, padded_mask_after_cnn = (
            self.padded_and_mask_function(
                chunk_list, chunk_lengths, padding_value=0, padding_side="right"
            )
        )

        padded_embed = nn.functional.gelu(self.conv1(padded_feature)) * padded_mask
        padded_embed = nn.functional.gelu(self.conv2(padded_embed)).transpose(1, 2)

        padded_embed = padded_embed + self.positional_embedding.positional_embedding[
            : padded_embed.shape[1], :
        ].unsqueeze(0).to(padded_embed.dtype)

        hidden_states = padded_embed[padded_mask_after_cnn]
        cu_seqlens = torch.cat(
            (
                torch.zeros(1, device=padded_mask_after_cnn.device, dtype=torch.int32),
                padded_mask_after_cnn.sum(1).cumsum(0),
            )
        ).to(torch.int32)

        for encoder_layer in self.layers:
            (hidden_states,) = encoder_layer(
                hidden_states,
                cu_seqlens=cu_seqlens,
                **kwargs,
            )

        hidden_states_list = hidden_states.split(valid_aftercnn_lens.tolist(), dim=0)

        pooled_list: list[torch.Tensor] = []
        pooled_lengths: list[int] = []
        for each_audio_states in hidden_states_list:
            seq_len = int(each_audio_states.shape[0])
            if seq_len >= 2:
                pooled = nn.functional.avg_pool1d(
                    each_audio_states.transpose(0, 1), kernel_size=2, stride=2
                ).transpose(0, 1)
            else:
                pooled = each_audio_states
            pooled_list.append(pooled)
            pooled_lengths.append(int(pooled.shape[0]))

        pooled_concat = torch.cat(pooled_list, dim=0)
        processed_concat = self.proj(self.ln_post(pooled_concat))
        processed_audio_list = list(processed_concat.split(pooled_lengths, dim=0))

        output_dim = (
            int(processed_audio_list[0].shape[-1])
            if processed_audio_list
            else int(self.proj.out_features)
        )
        output_hidden_states = torch.zeros(
            (original_batch_size, speech_maxlen, output_dim),
            dtype=processed_audio_list[0].dtype
            if processed_audio_list
            else self.proj.weight.dtype,
            device=device,
        )

        for valid_idx, processed in zip(valid_indices, processed_audio_list):
            seq_len = min(int(processed.shape[0]), int(speech_maxlen))
            output_hidden_states[int(valid_idx), :seq_len] = processed[:seq_len]

        return BaseModelOutput(last_hidden_state=output_hidden_states)

    def padded_and_mask_function(
        self,
        tensor_list: Sequence[torch.Tensor],
        tensor_len: torch.Tensor,
        padding_value: float = 0.0,
        padding_side: str = "right",
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        max_len = int(tensor_len.max().item())
        dim = int(tensor_list[0].shape[0])
        padded_tensor = torch.full(
            size=(len(tensor_list), dim, max_len),
            fill_value=padding_value,
            dtype=self.dtype,
            device=tensor_list[0].device,
        )

        batch_mask = torch.zeros(
            (len(tensor_len), max_len), dtype=torch.long, device=padded_tensor.device
        )
        for i, length in enumerate(tensor_len):
            length_val = int(length.item())
            batch_mask[i, :length_val] = 1
            padded_tensor[i, :, :length_val] = tensor_list[i]

        feature_lens_after_cnn = (tensor_len - 1) // 2 + 1
        max_len_after_cnn = int(feature_lens_after_cnn.max().item())
        batch_mask_after_cnn = torch.zeros(
            (len(tensor_len), max_len_after_cnn),
            dtype=torch.long,
            device=padded_tensor.device,
        )
        for i, length in enumerate(feature_lens_after_cnn):
            batch_mask_after_cnn[i, : int(length.item())] = 1

        if padding_side != "right":
            raise NotImplementedError("Only right padding is supported.")

        return (
            padded_tensor,
            batch_mask.unsqueeze(1).to(padded_tensor.dtype),
            batch_mask_after_cnn.bool(),
        )

    # From the HF FunAudioChat implementation.
    def _get_feat_extract_output_lengths(
        self, input_lengths: torch.LongTensor
    ) -> tuple[torch.LongTensor, torch.LongTensor]:
        input_lengths = (input_lengths - 1) // 2 + 1
        output_lengths = (input_lengths - 2) // 2 + 1
        return input_lengths, output_lengths


class FunAudioChatDiscreteEncoder(nn.Module):
    """Discrete audio encoder (speech tokenizer -> grouped embeddings)."""

    def __init__(self, config: Any):
        super().__init__()
        self.padding_idx = int(config.pad_token_id)
        self.group_size = int(config.group_size)
        self.hidden_size = int(config.output_dim)
        self.continuous_features_mode = getattr(
            config, "continuous_features_mode", "add"
        )
        self.embed_tokens = nn.Embedding(
            int(config.codebook_size), self.hidden_size, self.padding_idx
        )
        self.output_matching = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.continual_output_matching = nn.Linear(
            self.hidden_size, self.hidden_size, bias=False
        )

    def forward(
        self,
        audio_ids: torch.Tensor,
        continuous_audio_features: torch.Tensor | None = None,
        continuous_audio_output_lengths: torch.Tensor | None = None,
        feature_exist_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del continuous_audio_output_lengths

        inputs_embeds = self.embed_tokens(audio_ids)
        hidden_states = inputs_embeds.reshape(
            inputs_embeds.shape[0], -1, self.group_size * self.hidden_size
        )
        hidden_states = hidden_states.reshape(
            hidden_states.shape[0], -1, self.group_size, self.hidden_size
        ).mean(dim=2)
        hidden_states = self.output_matching(hidden_states)

        if continuous_audio_features is not None:
            continuous_audio_features = continuous_audio_features.reshape(
                continuous_audio_features.shape[0],
                -1,
                self.group_size,
                self.hidden_size,
            ).mean(dim=2)
            continuous_audio_hidden_states = self.continual_output_matching(
                continuous_audio_features
            )

            if feature_exist_mask is None:
                feature_exist_mask = torch.ones(
                    (hidden_states.shape[0],),
                    dtype=torch.bool,
                    device=hidden_states.device,
                )
            if self.continuous_features_mode == "add":
                hidden_states[feature_exist_mask] += continuous_audio_hidden_states
            else:
                hidden_states[feature_exist_mask] = continuous_audio_hidden_states

        return hidden_states

    def _get_feat_extract_output_lengths(
        self, input_lengths: torch.LongTensor
    ) -> tuple[torch.LongTensor, torch.LongTensor]:
        output_lengths = (input_lengths + self.group_size - 1) // self.group_size
        return input_lengths, output_lengths


class FunAudioChatProcessingInfo(BaseProcessingInfo):
    token_fps: int = 25

    @cached_property
    def feature_extractor(self) -> WhisperFeatureExtractor:
        return WhisperFeatureExtractor.from_pretrained(self.model_id)

    @cached_property
    def speech_tokenizer(self) -> PreTrainedTokenizerFast:
        return PreTrainedTokenizerFast.from_pretrained(
            self.model_id, subfolder="speech_tokenizer"
        )

    def get_feature_extractor(self) -> WhisperFeatureExtractor:
        return self.feature_extractor

    def get_speech_tokenizer(self) -> PreTrainedTokenizerFast:
        return self.speech_tokenizer

    def get_data_parser(self):
        return MultiModalDataParser(
            target_sr=int(self.feature_extractor.sampling_rate),
            target_channels=self.get_target_channels(),
            expected_hidden_size=self._get_expected_hidden_size(),
        )

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"audio": None}

    def get_target_channels(self) -> int:
        return 1

    def get_mm_max_tokens_per_item(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> Mapping[str, int] | None:
        # The discrete audio encoder downsamples 25Hz frames with group_size=5,
        # so for a 300s clip the max number of `<|AUDIO|>` placeholders is 1500.
        cfg = self.get_hf_config()
        audio_cfg = getattr(cfg, "audio_config", None)
        max_audio_tokens = int(getattr(audio_cfg, "max_source_positions", 1500))
        return {"audio": max_audio_tokens}

    def get_audio_group_size(self) -> int:
        cfg = self.get_hf_config()
        audio_cfg = getattr(cfg, "audio_config", None)
        return int(getattr(audio_cfg, "group_size", 5))


class FunAudioChatDummyInputsBuilder(
    BaseDummyInputsBuilder[FunAudioChatProcessingInfo]
):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        num_audios = mm_counts.get("audio", 0)
        return "<|audio_bos|><|AUDIO|><|audio_eos|>" * int(num_audios)

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions] | None = None,
    ) -> MultiModalDataDict:
        feature_extractor = self.info.get_feature_extractor()
        sampling_rate = int(feature_extractor.sampling_rate)

        # Dummy inputs are used for profiling; construct the worst-case audio
        # length that maximizes the number of encoder tokens.
        cfg = self.info.get_hf_config()
        audio_cfg = getattr(cfg, "audio_config", None)
        max_audio_tokens = int(getattr(audio_cfg, "max_source_positions", 1500))
        group_size = self.info.get_audio_group_size()
        token_fps = int(getattr(self.info, "token_fps", 25))
        target_num_frames = max(1, max_audio_tokens) * max(1, group_size)
        audio_len = max(
            1,
            (target_num_frames * sampling_rate + token_fps - 1) // token_fps,
        )
        num_audios = int(mm_counts.get("audio", 0))

        audio_overrides = mm_options.get("audio") if mm_options else None
        return {
            "audio": self._get_dummy_audios(
                length=audio_len,
                num_audios=num_audios,
                overrides=audio_overrides,
            )
        }


class FunAudioChatMultiModalProcessor(
    BaseMultiModalProcessor[FunAudioChatProcessingInfo]
):
    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        tokenizer = self.info.get_tokenizer()
        input_ids = torch.tensor([tokenizer.encode(prompt, **tok_kwargs)])

        audios = mm_data.get("audios", [])
        if not audios:
            return BatchFeature({"input_ids": input_ids})

        feature_extractor = self.info.get_feature_extractor()
        sr = int(feature_extractor.sampling_rate)
        min_samples = int(getattr(feature_extractor, "n_fft", 400) or 400)

        wavs: list[np.ndarray] = []
        speech_strs: list[str] = []

        speech_tokenizer = self.info.get_speech_tokenizer()
        pad_token = speech_tokenizer.pad_token or "<|audio_pad|>"
        for audio in audios:
            if isinstance(audio, torch.Tensor):
                audio = audio.detach().cpu().numpy()
            audio_np = np.asarray(audio, dtype=np.float32)

            if min_samples > 0 and audio_np.shape[0] < min_samples:
                audio_np = np.pad(
                    audio_np, (0, min_samples - audio_np.shape[0]), mode="constant"
                )

            wavs.append(audio_np)
            num_frames = int(
                (float(audio_np.shape[0]) / float(sr)) * float(self.info.token_fps)
            )
            speech_strs.append(pad_token * max(1, int(num_frames)))

        audio_group_size = self.info.get_audio_group_size()
        speech_inputs = speech_tokenizer(
            speech_strs,
            return_attention_mask=True,
            return_token_type_ids=False,
            padding=True,
            pad_to_multiple_of=audio_group_size,
            return_tensors="pt",
        )

        wav_inputs = feature_extractor(
            wavs,
            sampling_rate=sr,
            return_attention_mask=True,
            padding="max_length",
            return_tensors="pt",
        )

        mm_inputs: dict[str, torch.Tensor] = {
            "speech_ids": speech_inputs["input_ids"],
            "speech_attention_mask": speech_inputs["attention_mask"],
            "input_features": wav_inputs["input_features"],
            "feature_attention_mask": wav_inputs["attention_mask"],
            "feature_exist_mask": torch.ones((len(wavs),), dtype=torch.bool),
        }

        return BatchFeature({"input_ids": input_ids, **mm_inputs})

    def _hf_processor_applies_updates(
        self,
        prompt_text: str,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        tokenization_kwargs: Mapping[str, object],
    ) -> bool:
        return False

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return {
            "speech_ids": MultiModalFieldConfig.batched("audio"),
            "speech_attention_mask": MultiModalFieldConfig.batched("audio"),
            "input_features": MultiModalFieldConfig.batched("audio"),
            "feature_attention_mask": MultiModalFieldConfig.batched("audio"),
            "feature_exist_mask": MultiModalFieldConfig.batched("audio"),
        }

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        tokenizer = self.info.get_tokenizer()
        vocab = tokenizer.get_vocab()

        audio_token = "<|AUDIO|>"
        audio_token_id = vocab[audio_token]

        out_mm_data = out_mm_kwargs.get_data()
        speech_attention_mask = out_mm_data.get("speech_attention_mask")
        if speech_attention_mask is None:
            audio_output_lengths: list[int] = []
        else:
            assert isinstance(speech_attention_mask, torch.Tensor)
            speech_lengths = speech_attention_mask.sum(-1)
            group_size = self.info.get_audio_group_size()
            audio_output_lengths = (
                (speech_lengths + group_size - 1) // group_size
            ).tolist()

        def get_replacement_funaudiochat(item_idx: int):
            num_features = (
                int(audio_output_lengths[item_idx]) if audio_output_lengths else 1
            )
            if num_features <= 0:
                audios = mm_items.get_items("audio", AudioProcessorItems)
                audio_len = audios.get_audio_length(item_idx)
                raise ValueError(
                    f"The audio (len={audio_len}) is too short to be "
                    "represented inside the model"
                )

            audio_tokens = [audio_token_id] * num_features
            return PromptUpdateDetails.select_token_id(
                audio_tokens,
                embed_token_id=audio_token_id,
            )

        return [
            PromptReplacement(
                modality="audio",
                target=audio_token,
                replacement=get_replacement_funaudiochat,
            )
        ]


@MULTIMODAL_REGISTRY.register_processor(
    FunAudioChatMultiModalProcessor,
    info=FunAudioChatProcessingInfo,
    dummy_inputs=FunAudioChatDummyInputsBuilder,
)
class FunAudioChatForConditionalGeneration(nn.Module, SupportsMultiModal, SupportsPP):
    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("audio"):
            return "<|audio_bos|><|AUDIO|><|audio_eos|>"

        raise ValueError("Only audio modality is supported")

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        multimodal_config = vllm_config.model_config.multimodal_config
        self.config = config
        self.multimodal_config = multimodal_config
        self.quant_config = quant_config

        with self._mark_tower_model(vllm_config, "audio"):
            self.continuous_audio_tower = FunAudioChatAudioEncoder(config.audio_config)
            self.audio_tower = FunAudioChatDiscreteEncoder(config.audio_config)

        with self._mark_language_model(vllm_config):
            self.language_model = init_vllm_registered_model(
                vllm_config=vllm_config,
                hf_config=config.text_config,
                prefix=maybe_prefix(prefix, "language_model"),
                architectures=["Qwen3ForCausalLM"],
            )

        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )

    def get_language_model(self) -> torch.nn.Module:
        return self.language_model

    def _get_continuous_audio_features(
        self,
        input_features: torch.Tensor,
        feature_attention_mask: torch.Tensor,
        speech_maxlen: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Align mask and features to avoid indexing errors when padding differs.
        if (
            input_features.dim() == 3
            and feature_attention_mask.shape[1] != input_features.shape[-1]
        ):
            min_len = min(
                int(feature_attention_mask.shape[1]), int(input_features.shape[-1])
            )
            feature_attention_mask = feature_attention_mask[:, :min_len]
            input_features = input_features[:, :, :min_len]

        feature_lens = torch.sum(feature_attention_mask, dim=1)

        flat_features = input_features.permute(0, 2, 1)[
            feature_attention_mask.bool()
        ].permute(1, 0)

        audio_feat_lengths, audio_output_lengths = (
            self.continuous_audio_tower._get_feat_extract_output_lengths(feature_lens)
        )

        audio_outputs = self.continuous_audio_tower(
            flat_features,
            feature_lens=feature_lens,
            aftercnn_lens=audio_feat_lengths,
            speech_maxlen=speech_maxlen,
        )
        return audio_outputs.last_hidden_state, audio_output_lengths

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        speech_ids = kwargs.get("speech_ids")
        speech_attention_mask = kwargs.get("speech_attention_mask")
        input_features = kwargs.get("input_features")
        feature_attention_mask = kwargs.get("feature_attention_mask")
        feature_exist_mask = kwargs.get("feature_exist_mask")

        if speech_ids is None:
            return []

        pad_id = int(getattr(self.audio_tower, "padding_idx", 0))

        if not isinstance(speech_ids, torch.Tensor):
            if (
                isinstance(speech_ids, (list, tuple))
                and len(speech_ids) > 0
                and all(isinstance(t, torch.Tensor) for t in speech_ids)
            ):
                speech_ids_tensors = []
                for t in speech_ids:
                    if t.dim() == 2 and t.shape[0] == 1:
                        t = t.squeeze(0)
                    if t.dim() != 1:
                        raise TypeError(
                            "FunAudioChat speech_ids must be a 1D tensor per item "
                            f"(got shape={tuple(t.shape)})"
                        )
                    speech_ids_tensors.append(t)
                speech_ids = nn.utils.rnn.pad_sequence(
                    speech_ids_tensors,
                    batch_first=True,
                    padding_value=pad_id,
                )
            else:
                raise TypeError(
                    "FunAudioChat speech_ids must be a Tensor or a sequence of Tensors "
                    f"(got {type(speech_ids)})"
                )

        if speech_attention_mask is None:
            speech_attention_mask = speech_ids.ne(pad_id).to(dtype=torch.int64)

        if not isinstance(speech_attention_mask, torch.Tensor):
            if (
                isinstance(speech_attention_mask, (list, tuple))
                and len(speech_attention_mask) > 0
                and all(isinstance(t, torch.Tensor) for t in speech_attention_mask)
            ):
                mask_tensors = []
                for t in speech_attention_mask:
                    if t.dim() == 2 and t.shape[0] == 1:
                        t = t.squeeze(0)
                    if t.dim() != 1:
                        raise TypeError(
                            "FunAudioChat speech_attention_mask must be a 1D tensor "
                            f"per item (got shape={tuple(t.shape)})"
                        )
                    mask_tensors.append(t)
                speech_attention_mask = nn.utils.rnn.pad_sequence(
                    mask_tensors,
                    batch_first=True,
                    padding_value=0,
                )
            else:
                raise TypeError(
                    "FunAudioChat speech_attention_mask must be a Tensor or a "
                    f"sequence of Tensors (got {type(speech_attention_mask)})"
                )

        debug = os.getenv("VLLM_FUN_AUDIOCHAT_DEBUG", "") == "1"
        if debug:
            print(
                f"[FunAudioChat] embed_multimodal speech_ids={tuple(speech_ids.shape)} "
                f"speech_attention_mask={tuple(speech_attention_mask.shape)}",
                flush=True,
            )
            attn_impl = getattr(
                self.continuous_audio_tower.config, "_attn_implementation", None
            )
            print(
                f"[FunAudioChat] audio_attn_impl={attn_impl}",
                flush=True,
            )
            if hasattr(self.continuous_audio_tower, "conv1"):
                conv1_w = self.continuous_audio_tower.conv1.weight
                print(
                    f"[FunAudioChat] conv1_w_norm={float(conv1_w.norm().item()):.6g}",
                    flush=True,
                )
            try:
                attn0 = self.continuous_audio_tower.layers[0].self_attn
                q_norm = float(attn0.q_proj.weight.norm().item())
                k_norm = float(attn0.k_proj.weight.norm().item())
                v_norm = float(attn0.v_proj.weight.norm().item())
                o_norm = float(attn0.out_proj.weight.norm().item())
                print(
                    f"[FunAudioChat] attn0_q_norm={q_norm:.6g} "
                    f"k_norm={k_norm:.6g} "
                    f"v_norm={v_norm:.6g} "
                    f"o_norm={o_norm:.6g}",
                    flush=True,
                )
            except Exception:
                pass
            if isinstance(input_features, torch.Tensor):
                print(
                    f"[FunAudioChat] input_features={tuple(input_features.shape)}",
                    flush=True,
                )
            if isinstance(feature_attention_mask, torch.Tensor):
                print(
                    "[FunAudioChat] feature_attention_mask="
                    f"{tuple(feature_attention_mask.shape)}",
                    flush=True,
                )

        group_size = int(self.audio_tower.group_size)
        speech_maxlen = int(speech_ids.shape[-1])

        # Ensure token length is divisible by group_size.
        target_len = ((speech_maxlen + group_size - 1) // group_size) * group_size
        if target_len > speech_maxlen:
            pad_id = int(self.audio_tower.padding_idx)
            pad_len = target_len - speech_maxlen
            speech_ids = nn.functional.pad(speech_ids, (0, pad_len), value=pad_id)
            speech_attention_mask = nn.functional.pad(
                speech_attention_mask, (0, pad_len), value=0
            )
            speech_maxlen = int(speech_ids.shape[-1])

        continuous_audio_features = None
        continuous_audio_output_lengths = None
        if input_features is not None and feature_attention_mask is not None:
            assert isinstance(input_features, torch.Tensor)
            assert isinstance(feature_attention_mask, torch.Tensor)
            continuous_audio_features, continuous_audio_output_lengths = (
                self._get_continuous_audio_features(
                    input_features=input_features,
                    feature_attention_mask=feature_attention_mask,
                    speech_maxlen=speech_maxlen,
                )
            )

        if feature_exist_mask is None:
            feature_exist_mask = torch.ones(
                (speech_ids.shape[0],), dtype=torch.bool, device=speech_ids.device
            )
        assert isinstance(feature_exist_mask, torch.Tensor)

        audio_features = self.audio_tower(
            speech_ids,
            continuous_audio_features=continuous_audio_features,
            continuous_audio_output_lengths=continuous_audio_output_lengths,
            feature_exist_mask=feature_exist_mask,
        )

        _, audio_output_lengths = self.audio_tower._get_feat_extract_output_lengths(
            speech_attention_mask.sum(-1)
        )
        lengths = audio_output_lengths.tolist()

        embeds = tuple(
            audio_features[i, : int(length)] for i, length in enumerate(lengths)
        )
        if debug:
            embed_lens = [int(t.shape[0]) for t in embeds]
            print(f"[FunAudioChat] embed_multimodal out_lens={embed_lens}", flush=True)
            if embeds:
                t0 = embeds[0]
                print(
                    f"[FunAudioChat] embed0 dtype={t0.dtype} device={t0.device} "
                    f"nan={bool(torch.isnan(t0).any())} "
                    f"norm={float(t0.norm().item()):.6g}",
                    flush=True,
                )
            dump_path = os.getenv("VLLM_FUN_AUDIOCHAT_DUMP_PATH", "")
            if (
                dump_path
                and speech_ids.shape[0] == 1
                and len(embeds) == 1
                and embed_lens[0] > 10
            ):
                if not os.path.exists(dump_path):
                    np.save(dump_path, embeds[0].detach().float().cpu().numpy())
                    print(f"[FunAudioChat] dumped embeds to {dump_path}", flush=True)
                cont_path = dump_path.replace(".npy", "_cont.npy")
                if continuous_audio_features is not None and not os.path.exists(
                    cont_path
                ):
                    np.save(
                        cont_path,
                        continuous_audio_features.detach().float().cpu().numpy(),
                    )
                    print(
                        f"[FunAudioChat] dumped continuous to {cont_path}", flush=True
                    )
        return embeds

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors:
        del kwargs
        if intermediate_tensors is not None:
            inputs_embeds = None

        return self.language_model.model(
            input_ids,
            positions,
            intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.language_model.compute_logits(hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self, skip_prefixes=["audio_invert_tower."])
        return loader.load_weights(weights)
