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

from collections.abc import Mapping, Sequence
from math import pi
from typing import Annotated, Any, TypeAlias

import torch
from torch import Tensor, broadcast_tensors, nn
from torch.amp import autocast
from transformers import BatchFeature, PretrainedConfig
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


# === Rotary Embedding === #
def rotate_half(x):
    x = x.reshape(*x.shape[:-1], -1, 2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return x.flatten(-2)


@autocast("cuda", enabled=False)
def apply_rotary_emb(freqs, t, start_index=0, scale=1.0, seq_dim=-2):
    ori_dtype = t.dtype
    embed_dtype = torch.float64
    t = t.to(embed_dtype)
    if t.ndim == 3:
        seq_len = t.shape[seq_dim]
        freqs = freqs[-seq_len:].to(t) if freqs.ndim == 2 else freqs.to(t)

    rot_dim = freqs.shape[-1]
    end_index = start_index + rot_dim

    if rot_dim > t.shape[-1]:
        raise ValueError(
            f"feature dimension {t.shape[-1]} is not of sufficient size to"
            f" rotate in all the positions {rot_dim}"
        )

    t_left, t, t_right = (
        t[..., :start_index],
        t[..., start_index:end_index],
        t[..., end_index:],
    )
    t = (t * freqs.cos() * scale) + (rotate_half(t) * freqs.sin() * scale)
    return torch.cat((t_left, t, t_right), dim=-1).to(ori_dtype)


class MusicFlamingoRotaryEmbedding(nn.Module):
    freqs: torch.Tensor

    def __init__(self, config: MusicFlamingoConfig, device=None):
        super().__init__()

        self.config = config
        self.dim = getattr(config, "rotary_dim", 256)
        self.max_time = getattr(config, "rotary_max_time", 1200.0)

        freqs = self.compute_default_rote_parameters(config, device=device)
        self.freqs = nn.Parameter(freqs, requires_grad=False)

        cached_freqs = self._build_cached_freqs(freqs)
        self.register_buffer("cached_freqs", cached_freqs, persistent=False)

    @staticmethod
    def compute_default_rote_parameters(
        config: MusicFlamingoConfig | None = None,
        device=None,
    ):
        dim = getattr(config, "rotary_dim", 256)
        max_time = getattr(config, "rotary_max_time", 1200.0)
        theta = max_time / (2 * pi) if max_time is not None else 50000
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
        if device is not None:
            freqs = freqs.to(device=device)
        return freqs

    def _build_cached_freqs(self, freqs, device=None, dtype=None):
        if self.max_time is None:
            return None

        positions = torch.arange(
            int(self.max_time),
            device=device,
            dtype=dtype if dtype is not None else freqs.dtype,
        )
        positions = positions / self.max_time * (2 * pi)
        cached_freqs = positions.unsqueeze(-1) * freqs
        return torch.repeat_interleave(cached_freqs, 2, dim=-1)

    def get_axial_freqs(self, *dims):
        Colon = slice(None)
        all_freqs = []

        for ind, dim in enumerate(dims):
            pos = torch.arange(dim, device=self.freqs.device)

            freqs = self.forward(pos, seq_len=dim)

            all_axis = [None] * len(dims)
            all_axis[ind] = Colon

            new_axis_slice = (Ellipsis, *all_axis, Colon)
            all_freqs.append(freqs[new_axis_slice])

        all_freqs = broadcast_tensors(*all_freqs)
        return torch.cat(all_freqs, dim=-1)

    @autocast("cuda", enabled=False)
    def forward(self, t: Tensor, seq_len=None, offset=0):
        if (
            seq_len is not None
            and self.cached_freqs is not None
            and (offset + seq_len) <= self.cached_freqs.shape[0]
        ):
            return self.cached_freqs[offset : (offset + seq_len)].detach()

        freqs = self.freqs

        if self.max_time is not None:
            t = t / self.max_time * (2 * pi)

        freqs = t.type(freqs.dtype).unsqueeze(-1) * freqs
        freqs = torch.repeat_interleave(freqs, 2, dim=-1)
        return freqs


# === Audio Inputs === #
class MusicFlamingoFeatureInputs(AudioFlamingo3FeatureInputs):
    audio_times: Annotated[
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
    def __init__(self, config: PretrainedConfig):
        super().__init__(config)
        self.pos_emb = MusicFlamingoRotaryEmbedding(config)
        self._pending_audio_times: torch.Tensor | None = None

    def forward(
        self,
        input_features: torch.Tensor | list[torch.Tensor],
        attention_mask: torch.Tensor | None = None,
        audio_times: torch.Tensor | None = None,
    ):
        hidden_states = super().forward(
            input_features=input_features,
            attention_mask=attention_mask,
        )

        if audio_times is None:
            audio_times = self._pending_audio_times

        if audio_times is not None:
            times = audio_times.to(hidden_states.device)
            freqs = self.pos_emb.get_axial_freqs(
                times.shape[0], hidden_states.shape[-2]
            ).to(self.conv1.weight.device)
            angle = (-times * 2 * pi).to(self.conv1.weight.device)
            angle_expanded = angle.unsqueeze(2).expand(
                times.shape[0], hidden_states.shape[-2], freqs.shape[-1]
            )
            freqs = freqs * angle_expanded

            hidden_states = apply_rotary_emb(freqs, hidden_states)

        return hidden_states


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
        return (
            hf_processor.audio_bos_token
            + hf_processor.audio_token
            + hf_processor.audio_eos_token
        ) * num_audios

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
        fields["audio_times"] = MultiModalFieldConfig.flat_from_sizes(
            "audio", chunk_counts, dim=0
        )
    else:
        fields["audio_times"] = MultiModalFieldConfig.batched("audio")
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
        chunk_length = mm_kwargs.get("chunk_length", feature_extractor.chunk_length)
        window_size = int(sampling_rate * chunk_length)
        max_windows = int(processor.max_audio_len // chunk_length)

        chunk_counts = []
        for audio in audio_list:
            n_samples = len(audio) if isinstance(audio, list) else audio.shape[0]
            n_win = max(1, (n_samples + window_size - 1) // window_size)
            chunk_counts.append(min(n_win, max_windows))
        outputs["chunk_counts"] = torch.tensor(chunk_counts, dtype=torch.long)

        if "audio_times" not in outputs:
            raise KeyError("MusicFlamingoProcessor output must include `audio_times`.")

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

    def _parse_and_validate_audio_input(
        self, **kwargs: object
    ) -> MusicFlamingoInputs | None:
        audio_times = kwargs.pop("audio_times", None)
        audio_input = super()._parse_and_validate_audio_input(**kwargs)
        if audio_input is None or audio_input["type"] == "audio_embeds":
            return audio_input

        return MusicFlamingoFeatureInputs(
            type="audio_features",
            input_features=audio_input["input_features"],
            feature_attention_mask=audio_input["feature_attention_mask"],
            chunk_counts=audio_input["chunk_counts"],
            audio_times=audio_times,
        )

    def _process_audio_input(
        self, audio_input: MusicFlamingoInputs
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        if audio_input["type"] == "audio_embeds":
            return super()._process_audio_input(audio_input)

        audio_times = audio_input["audio_times"]

        if audio_times is None:
            raise ValueError(
                "MusicFlamingo audio feature inputs must include `audio_times`."
            )
        if isinstance(audio_times, list):
            audio_times = torch.cat(audio_times, dim=0)

        assert isinstance(self.audio_tower, MusicFlamingoEncoder)
        self.audio_tower._pending_audio_times = audio_times
        try:
            return super()._process_audio_input(audio_input)
        finally:
            self.audio_tower._pending_audio_times = None
