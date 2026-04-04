# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Copyright 2026 The vLLM team.
# Copyright 2026 NVIDIA CORPORATION and the HuggingFace Inc. team. All rights
# reserved.
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

from collections.abc import Callable, Mapping, Sequence
from math import pi
from typing import Optional, TypeAlias

import torch
from torch import Tensor, broadcast_tensors, nn
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from transformers.models.musicflamingo import (
    MusicFlamingoConfig,
    MusicFlamingoProcessor,
)

from vllm.config import VllmConfig
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import MultiModalKwargsItems
from vllm.multimodal.parse import MultiModalDataItems
from vllm.multimodal.processing import (
    PromptReplacement,
    PromptUpdate,
    PromptUpdateDetails,
)

from .audioflamingo3 import (
    AudioFlamingo3DummyInputsBuilder,
    AudioFlamingo3EmbeddingInputs,
    AudioFlamingo3Encoder,
    AudioFlamingo3FeatureInputs,
    AudioFlamingo3ForConditionalGeneration,
    AudioFlamingo3MultiModalProcessor,
    AudioFlamingo3MultiModalProjector,
    AudioFlamingo3ProcessingInfo,
    _count_audio_tokens_from_mask,
    _get_audio_post_pool_output_lengths,
)


def rotate_half(x):
    x = x.reshape(*x.shape[:-1], -1, 2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return x.flatten(-2)


def apply_rotary_time_emb(hidden_states, cos, sin):
    original_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float64)
    cos = cos.to(hidden_states)
    sin = sin.to(hidden_states)
    rot_dim = cos.shape[-1]

    rotated = hidden_states[..., :rot_dim]
    passthrough = hidden_states[..., rot_dim:]
    rotated = (rotated * cos) + (rotate_half(rotated) * sin)
    return torch.cat((rotated, passthrough), dim=-1).to(original_dtype)


class MusicFlamingoRotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor

    def __init__(self, config: MusicFlamingoConfig, device=None):
        super().__init__()
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_type = self.config.rope_parameters["rope_type"]
        rope_init_fn: Callable = self.compute_default_rope_parameters
        if self.rope_type != "default":
            rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]
        inv_freq, self.attention_scaling = rope_init_fn(self.config, device)

        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.register_buffer("original_inv_freq", inv_freq.clone(), persistent=False)
        position_angles = self._compute_position_angles(self.inv_freq)
        self.register_buffer("position_angles", position_angles, persistent=False)

    @staticmethod
    def compute_default_rope_parameters(
        config: MusicFlamingoConfig | None = None,
        device: Optional["torch.device"] = None,
        seq_len: int | None = None,
    ) -> tuple["torch.Tensor", float]:
        del seq_len
        base = config.rope_parameters["rope_theta"]
        partial_rotary_factor = config.rope_parameters.get("partial_rotary_factor", 1.0)
        head_dim = getattr(config, "head_dim", None) or (
            config.hidden_size // config.num_attention_heads
        )
        dim = int(head_dim * partial_rotary_factor)
        attention_factor = 1.0

        inv_freq = 1.0 / (
            base
            ** (
                torch.arange(0, dim, 2, dtype=torch.int64).to(
                    device=device,
                    dtype=torch.float,
                )
                / dim
            )
        )
        return inv_freq, attention_factor

    def _compute_position_angles(self, inv_freq):
        positions = torch.arange(
            int(self.max_seq_len_cached),
            device=inv_freq.device,
            dtype=inv_freq.dtype,
        )
        positions = positions / self.max_seq_len_cached * (2 * pi)
        position_angles = positions.unsqueeze(-1) * inv_freq
        position_angles = torch.repeat_interleave(position_angles, 2, dim=-1)
        return position_angles.to(dtype=inv_freq.dtype)

    @torch.no_grad()
    def forward(self, timestamps: Tensor, seq_len: int) -> tuple[Tensor, Tensor]:
        window_starts = timestamps[:, 0].to(
            device=self.inv_freq.device, dtype=self.inv_freq.dtype
        )
        window_duration = self.config.audio_frame_step * 4 * seq_len
        window_positions = (
            torch.round(window_starts / window_duration) / self.max_seq_len_cached
        )
        window_freqs = window_positions.unsqueeze(-1) * self.inv_freq
        window_freqs = torch.repeat_interleave(window_freqs, 2, dim=-1)

        window_freqs = window_freqs[:, None, :]
        time_freqs = self.position_angles[:seq_len][None, :, :]
        window_freqs, time_freqs = broadcast_tensors(window_freqs, time_freqs)
        freqs = torch.cat((window_freqs, time_freqs), dim=-1)
        angle = (-timestamps * 2 * pi).to(freqs)
        freqs = freqs * angle.unsqueeze(-1)
        return freqs.cos(), freqs.sin()


MusicFlamingoFeatureInputs = AudioFlamingo3FeatureInputs
MusicFlamingoEmbeddingInputs = AudioFlamingo3EmbeddingInputs

MusicFlamingoInputs: TypeAlias = (
    MusicFlamingoFeatureInputs | MusicFlamingoEmbeddingInputs
)


class MusicFlamingoEncoder(AudioFlamingo3Encoder):
    pass


class MusicFlamingoMultiModalProjector(AudioFlamingo3MultiModalProjector):
    pass


class MusicFlamingoProcessingInfo(AudioFlamingo3ProcessingInfo):
    def get_hf_config(self) -> MusicFlamingoConfig:
        return self.ctx.get_hf_config(MusicFlamingoConfig)

    def get_hf_processor(self, **kwargs: object) -> MusicFlamingoProcessor:
        return self.ctx.get_hf_processor(MusicFlamingoProcessor, **kwargs)

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"audio": 1}


class MusicFlamingoDummyInputsBuilder(AudioFlamingo3DummyInputsBuilder):
    pass


def _build_audio_timestamps(
    feature_attention_mask: torch.Tensor,
    chunk_counts: list[int],
    max_post_length: int,
    audio_frame_step: float,
) -> torch.Tensor:
    input_lengths = feature_attention_mask.sum(-1).to(torch.long)
    post_lengths = _get_audio_post_pool_output_lengths(input_lengths)
    chunk_count_tensor = torch.as_tensor(
        chunk_counts,
        device=post_lengths.device,
        dtype=torch.long,
    )

    if int(chunk_count_tensor.sum().item()) != post_lengths.shape[0]:
        raise ValueError(
            "chunk_counts do not match the number of encoded audio windows."
        )

    audio_embed_frame_step = audio_frame_step * 4
    frame_offsets = (
        torch.arange(
            max_post_length,
            device=post_lengths.device,
            dtype=torch.float32,
        )
        * audio_embed_frame_step
    )

    sample_indices = torch.repeat_interleave(
        torch.arange(chunk_count_tensor.shape[0], device=post_lengths.device),
        chunk_count_tensor,
    )
    sample_start_rows = torch.searchsorted(
        sample_indices,
        torch.arange(chunk_count_tensor.shape[0], device=post_lengths.device),
    )
    window_indices = (
        torch.arange(post_lengths.shape[0], device=post_lengths.device)
        - sample_start_rows[sample_indices]
    ).to(torch.float32)

    return (
        window_indices.unsqueeze(1) * max_post_length * audio_embed_frame_step
        + frame_offsets
    )


class MusicFlamingoMultiModalProcessor(AudioFlamingo3MultiModalProcessor):
    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        processor = self.info.get_hf_processor(**hf_processor_mm_kwargs)
        tokenizer = self.info.get_tokenizer()
        vocab = tokenizer.get_vocab()

        audio_token = processor.audio_token
        audio_token_id = vocab.get(audio_token, processor.audio_token_id)

        audio_bos_token = processor.audio_bos_token
        audio_bos_token_id = vocab.get(audio_bos_token, processor.audio_bos_token_id)

        audio_eos_token = processor.audio_eos_token
        audio_eos_token_id = vocab.get(audio_eos_token, processor.audio_eos_token_id)

        out_mm_data = out_mm_kwargs.get_data()
        feature_attention_mask = out_mm_data.get("feature_attention_mask")
        chunk_counts = out_mm_data.get("chunk_counts")

        def get_replacement_musicflamingo(item_idx: int):
            if feature_attention_mask is not None:
                num_features = _count_audio_tokens_from_mask(
                    feature_attention_mask,
                    chunk_counts,
                    item_idx,
                )
            else:
                audio_embeds = out_mm_data["audio_embeds"][item_idx]
                num_features = audio_embeds.shape[0]

            if num_features == 0:
                raise ValueError("Audio is too short")

            full_tokens = [
                audio_bos_token_id,
                *([audio_token_id] * int(num_features)),
                audio_eos_token_id,
            ]

            return PromptUpdateDetails.select_token_id(
                full_tokens,
                embed_token_id=audio_token_id,
            )

        return [
            PromptReplacement(
                modality="audio",
                target=audio_token,
                replacement=get_replacement_musicflamingo,
            )
        ]


@MULTIMODAL_REGISTRY.register_processor(
    MusicFlamingoMultiModalProcessor,
    info=MusicFlamingoProcessingInfo,
    dummy_inputs=MusicFlamingoDummyInputsBuilder,
)
class MusicFlamingoForConditionalGeneration(AudioFlamingo3ForConditionalGeneration):
    """vLLM MusicFlamingo model aligned with HF modular_musicflamingo."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        self.audio_tower = MusicFlamingoEncoder(self.config.audio_config)
        self.multi_modal_projector = MusicFlamingoMultiModalProjector(self.config)
        self.pos_emb = MusicFlamingoRotaryEmbedding(self.config)

    def _process_audio_input(
        self, audio_input: MusicFlamingoInputs
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        if audio_input["type"] == "audio_embeds":
            return super()._process_audio_input(audio_input)

        (
            input_features,
            feature_attention_mask,
            chunk_counts,
        ) = self._normalize_audio_feature_inputs(audio_input)
        hidden_states = self._encode_audio_features(
            input_features,
            feature_attention_mask,
        )
        audio_timestamps = _build_audio_timestamps(
            feature_attention_mask,
            chunk_counts,
            hidden_states.shape[-2],
            self.config.audio_frame_step,
        )
        cos, sin = self.pos_emb(
            audio_timestamps.to(hidden_states.device),
            seq_len=hidden_states.shape[-2],
        )
        hidden_states = apply_rotary_time_emb(hidden_states, cos, sin)
        audio_features = self.multi_modal_projector(hidden_states)

        return self._group_audio_embeddings(
            audio_features,
            feature_attention_mask,
            chunk_counts,
        )
