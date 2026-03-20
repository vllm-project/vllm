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
from typing import Annotated, Any, Optional, TypeAlias

import torch
from torch import Tensor, broadcast_tensors, nn
from transformers import BatchFeature
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from transformers.models.musicflamingo import (
    MusicFlamingoConfig,
    MusicFlamingoProcessor,
)

from vllm.config import VllmConfig
from vllm.config.multimodal import BaseDummyOptions
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (
    MultiModalDataDict,
    MultiModalFieldConfig,
    MultiModalKwargsItems,
)
from vllm.multimodal.parse import (
    DictEmbeddingItems,
    ModalityData,
    ModalityDataItems,
    MultiModalDataItems,
    MultiModalDataParser,
)
from vllm.multimodal.processing import (
    PromptReplacement,
    PromptUpdate,
    PromptUpdateDetails,
)
from vllm.utils.tensor_schema import TensorShape

from .audioflamingo3 import (
    AudioFlamingo3DummyInputsBuilder,
    AudioFlamingo3EmbeddingInputs,
    AudioFlamingo3Encoder,
    AudioFlamingo3FeatureInputs,
    AudioFlamingo3ForConditionalGeneration,
    AudioFlamingo3MultiModalDataParser,
    AudioFlamingo3MultiModalProcessor,
    AudioFlamingo3MultiModalProjector,
    AudioFlamingo3ProcessingInfo,
    _audioflamingo3_field_config,
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
    if rot_dim > hidden_states.shape[-1]:
        raise ValueError(
            f"feature dimension {hidden_states.shape[-1]} is not of "
            f"sufficient size to rotate in all the positions {rot_dim}"
        )

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
        dim = getattr(config, "head_dim", None) or (
            config.hidden_size // config.num_attention_heads
        )
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
        batch_positions = torch.arange(
            timestamps.shape[0],
            device=self.inv_freq.device,
            dtype=self.inv_freq.dtype,
        )
        batch_positions = batch_positions / self.max_seq_len_cached
        batch_freqs = batch_positions.unsqueeze(-1) * self.inv_freq
        batch_freqs = torch.repeat_interleave(batch_freqs, 2, dim=-1)

        batch_freqs = batch_freqs[:, None, :]
        time_freqs = self.position_angles[:seq_len][None, :, :]
        batch_freqs, time_freqs = broadcast_tensors(batch_freqs, time_freqs)
        freqs = torch.cat((batch_freqs, time_freqs), dim=-1)
        angle = (-timestamps * 2 * pi).to(freqs)
        freqs = freqs * angle.unsqueeze(-1)
        return freqs.cos(), freqs.sin()


class MusicFlamingoFeatureInputs(AudioFlamingo3FeatureInputs):
    rote_timestamps: Annotated[
        torch.Tensor,
        TensorShape(
            "num_chunks",
            "num_audio_time_steps",
            dynamic_dims={"num_audio_time_steps"},
        ),
    ]


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

    def get_data_parser(self) -> MultiModalDataParser:
        feature_extractor = self.get_feature_extractor()
        return MusicFlamingoMultiModalDataParser(
            target_sr=feature_extractor.sampling_rate,
            expected_hidden_size=self._get_expected_hidden_size(),
        )


class MusicFlamingoDummyInputsBuilder(AudioFlamingo3DummyInputsBuilder):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        num_audios = mm_counts.get("audio", 0)
        hf_processor = self.info.get_hf_processor()
        return hf_processor.audio_token * num_audios

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions] | None = None,
    ) -> MultiModalDataDict:
        hf_processor = self.info.get_hf_processor()
        feature_extractor = self.info.get_feature_extractor()
        sampling_rate = feature_extractor.sampling_rate
        audio_len = int(hf_processor.max_audio_len * sampling_rate)
        num_audios = mm_counts.get("audio", 0)
        audio_overrides = mm_options.get("audio") if mm_options else None

        return {
            "audio": self._get_dummy_audios(
                length=audio_len,
                num_audios=num_audios,
                overrides=audio_overrides,
            )
        }


def _musicflamingo_field_config(hf_inputs: Mapping[str, torch.Tensor]):
    fields = dict(_audioflamingo3_field_config(hf_inputs))
    chunk_counts = hf_inputs.get("chunk_counts")
    if chunk_counts is not None:
        fields["rote_timestamps"] = MultiModalFieldConfig.flat_from_sizes(
            "audio", chunk_counts, dim=0
        )
    else:
        fields["rote_timestamps"] = MultiModalFieldConfig.batched("audio")
    return fields


class MusicFlamingoMultiModalDataParser(AudioFlamingo3MultiModalDataParser):
    def _parse_audio_data(
        self,
        data: dict[str, torch.Tensor] | ModalityData[Any],
    ) -> ModalityDataItems[Any, Any] | None:
        if isinstance(data, dict):
            return DictEmbeddingItems(
                data,
                modality="audio",
                required_fields={"audio_embeds"},
                fields_factory=_musicflamingo_field_config,
            )
        return super()._parse_audio_data(data)


class MusicFlamingoMultiModalProcessor(AudioFlamingo3MultiModalProcessor):
    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: dict[str, object],
        mm_kwargs: Mapping[str, Any],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        outputs = super()._call_hf_processor(
            prompt=prompt,
            mm_data=mm_data,
            mm_kwargs=mm_kwargs,
            tok_kwargs=tok_kwargs,
        )

        audio_data = mm_data.get("audio")
        if audio_data is None:
            return outputs

        audio_list = audio_data if isinstance(audio_data, list) else [audio_data]
        if len(audio_list) == 0:
            return outputs

        processor = self.info.get_hf_processor(**mm_kwargs)
        feature_extractor = processor.feature_extractor
        sampling_rate = feature_extractor.sampling_rate
        chunk_length = feature_extractor.chunk_length
        window_size = int(sampling_rate * chunk_length)
        max_windows = int(processor.max_audio_len // chunk_length)

        chunk_counts = []
        for audio in audio_list:
            n_samples = len(audio) if isinstance(audio, list) else audio.shape[0]
            n_win = max(1, (n_samples + window_size - 1) // window_size)
            chunk_counts.append(min(n_win, max_windows))
        outputs["chunk_counts"] = torch.tensor(chunk_counts, dtype=torch.long)

        if "rote_timestamps" not in outputs:
            raise KeyError(
                "MusicFlamingoProcessor output must include `rote_timestamps`."
            )

        return outputs

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return _musicflamingo_field_config(hf_inputs)

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        base_updates = super()._get_prompt_updates(
            mm_items=mm_items,
            hf_processor_mm_kwargs=hf_processor_mm_kwargs,
            out_mm_kwargs=out_mm_kwargs,
        )
        if len(base_updates) != 1 or not isinstance(base_updates[0], PromptReplacement):
            raise ValueError("Expected exactly one audio PromptReplacement.")
        base_update = base_updates[0]

        processor = self.info.get_hf_processor(**hf_processor_mm_kwargs)
        tokenizer = self.info.get_tokenizer()
        vocab = tokenizer.get_vocab()

        audio_token = processor.audio_token
        audio_token_id = vocab.get(audio_token, processor.audio_token_id)

        audio_bos_token = processor.audio_bos_token
        audio_bos_token_id = vocab.get(audio_bos_token, processor.audio_bos_token_id)

        audio_eos_token = processor.audio_eos_token
        audio_eos_token_id = vocab.get(audio_eos_token, processor.audio_eos_token_id)

        def get_replacement_musicflamingo(item_idx: int):
            base_replacement = base_update.replacement
            base_details = (
                base_replacement(item_idx)
                if callable(base_replacement)
                else base_replacement
            )
            if not isinstance(base_details, PromptUpdateDetails):
                base_details = PromptUpdateDetails.from_seq(base_details)

            base_full = base_details.full
            if not isinstance(base_full, list):
                raise TypeError(
                    "Expected token-id replacement from AudioFlamingo3 prompt update."
                )

            full_tokens = [audio_bos_token_id, *base_full, audio_eos_token_id]

            return PromptUpdateDetails.select_token_id(
                full_tokens,
                embed_token_id=audio_token_id,
            )

        return [
            PromptReplacement(
                modality=base_update.modality,
                target=base_update.target,
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

    def _parse_and_validate_audio_input(
        self, **kwargs: object
    ) -> MusicFlamingoInputs | None:
        rote_timestamps = kwargs.pop("rote_timestamps", None)
        audio_input = super()._parse_and_validate_audio_input(**kwargs)
        if audio_input is None or audio_input["type"] == "audio_embeds":
            return audio_input

        return MusicFlamingoFeatureInputs(
            type="audio_features",
            input_features=audio_input["input_features"],
            feature_attention_mask=audio_input["feature_attention_mask"],
            chunk_counts=audio_input["chunk_counts"],
            rote_timestamps=rote_timestamps,
        )

    def _process_audio_input(
        self, audio_input: MusicFlamingoInputs
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        if audio_input["type"] == "audio_embeds":
            return super()._process_audio_input(audio_input)

        rote_timestamps = audio_input["rote_timestamps"]
        if rote_timestamps is None:
            raise ValueError(
                "MusicFlamingo audio feature inputs must include `rote_timestamps`."
            )
        if isinstance(rote_timestamps, list):
            rote_timestamps = torch.cat(rote_timestamps, dim=0)

        (
            input_features,
            feature_attention_mask,
            chunk_counts,
        ) = self._normalize_audio_feature_inputs(audio_input)
        hidden_states = self._encode_audio_features(
            input_features,
            feature_attention_mask,
        )
        cos, sin = self.pos_emb(
            rote_timestamps.to(hidden_states.device),
            seq_len=hidden_states.shape[-2],
        )
        hidden_states = apply_rotary_time_emb(hidden_states, cos, sin)
        audio_features = self.multi_modal_projector(hidden_states)

        return self._group_audio_embeddings(
            audio_features,
            feature_attention_mask,
            chunk_counts,
        )
