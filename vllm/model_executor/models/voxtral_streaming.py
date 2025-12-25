# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
from collections.abc import Mapping

import torch

from vllm.config.vllm import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.models.interfaces import MultiModalEmbeddings
from vllm.model_executor.models.voxtral import (
    VoxtralDummyInputsBuilder,
    VoxtralForConditionalGeneration,
    VoxtralMultiModalProcessor,
    VoxtralProcessingInfo,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.cache import _I, BaseMultiModalProcessorCache
from vllm.multimodal.inputs import (
    MultiModalKwargsOptionalItems,
)
from vllm.multimodal.parse import MultiModalDataItems
from vllm.multimodal.processing import (
    MultiModalPromptUpdates,
    PlaceholderFeaturesInfo,
)
from vllm.multimodal.profiling import BaseDummyInputsBuilder
from vllm.sequence import IntermediateTensors

from .utils import (
    _flatten_embeddings,
)

logger = init_logger(__name__)


class VoxtralStreamingMultiModalProcessor(VoxtralMultiModalProcessor):
    def __init__(
        self,
        info: _I,
        dummy_inputs: BaseDummyInputsBuilder[_I],
        *,
        cache: BaseMultiModalProcessorCache | None = None,
    ) -> None:
        # streaming can't make use of a cache yet
        super().__init__(info, dummy_inputs, cache=None)

    def _maybe_apply_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        prompt_ids: list[int],
        mm_kwargs: MultiModalKwargsOptionalItems,
        mm_prompt_updates: MultiModalPromptUpdates,
        is_update_applied: bool,
    ) -> tuple[list[int], Mapping[str, list[PlaceholderFeaturesInfo]]]:
        # there are no placeholder audio tokens for streaming
        # so we need to build the place placeholder positions manually

        # in streaming there is always only one audio input
        audios = mm_kwargs.get("audio", [])
        assert len(audios) == 1, (
            f"Expected only one audio input for streaming, got {mm_kwargs=}"
        )
        tokenizer = self.info.get_tokenizer()
        audio_config = tokenizer.instruct.audio_encoder.audio_config

        num_audio_samples = audios[0]["audio_arrays"].data.shape[0]
        length = audio_config.num_audio_tokens(num_audio_samples)

        features_info = PlaceholderFeaturesInfo(
            modality="audio",
            item_idx=0,
            start_idx=0,
            tokens=length
            * [0],  # only used for length computation, so we can take dummy inputs
            is_embed=None,
        )
        return prompt_ids, {"audio": [features_info]}


class TimeEmbedding(torch.nn.Module):
    """Sinusoidal Embedding for encoding time"""

    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        self.dim = dim
        self.theta = theta
        inv_freq = torch.exp(
            -math.log(self.theta)
            * torch.arange(self.dim // 2).float()
            / (self.dim // 2)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t = t[..., None]  # (B,) -> (B, 1) or (B, T) -> (B, T, 1)
        inv_freq = self.inv_freq.to(device=t.device, dtype=t.dtype)
        emb = (
            t * inv_freq
        )  # (B, 1) x (D/2,) -> (B, D/2) or (B, T, 1) x (D/2,) -> (B, T, D/2)
        return torch.cat((emb.cos(), emb.sin()), dim=-1)  # (B, D) or (B, T, D)


@MULTIMODAL_REGISTRY.register_processor(
    VoxtralStreamingMultiModalProcessor,
    info=VoxtralProcessingInfo,
    dummy_inputs=VoxtralDummyInputsBuilder,
)
class VoxtralStreamingGeneration(VoxtralForConditionalGeneration):
    requires_raw_input_tokens = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        self.time_embedding: TimeEmbedding = TimeEmbedding(
            dim=self.config.text_config.hidden_size
        )

        audio_config = self.tokenizer.instruct.audio_encoder.audio_config
        _n_delay_tokens = (
            audio_config.frame_rate * audio_config.transcription_delay_ms / 1000
        )
        assert _n_delay_tokens.is_integer(), (
            f"n_delay_tokens must be integer, got {_n_delay_tokens}"
        )

        self.n_delay_tokens = int(_n_delay_tokens)

    @property
    def audio_config(self):
        return self.tokenizer.instruct.audio_encoder.audio_config

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: MultiModalEmbeddings | None = None,
        *,
        is_multimodal: torch.Tensor | None = None,
        # Multi-modal token ID may exceed vocab size
        handle_oov_mm_token: bool = True,
    ) -> torch.Tensor:
        """Pass post-conv embeddings directly as input"""
        # for streaming we simply flatten the multimodal embeddings
        # to be in tensor format, we treat the input ids later
        assert multimodal_embeddings is not None
        assert len(multimodal_embeddings) > 0, (
            "For streaming you must provide a multimodal_embedding at every step."
        )
        mm_embeds_flat = _flatten_embeddings(multimodal_embeddings)
        return mm_embeds_flat

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors:
        assert inputs_embeds is not None
        assert input_ids is not None

        pool_size = self.config.audio_config.block_pool_size
        inputs_embeds = inputs_embeds.view(
            inputs_embeds.shape[0] * pool_size, inputs_embeds.shape[1] // pool_size
        )

        audio_hidden_states = self.whisper_encoder.whisper_encoder.forward_layers(
            inputs_embeds
        )

        num_tokens, audio_hidden_size = audio_hidden_states.shape
        assert num_tokens % self.downsample_factor == 0
        audio_hidden_states = audio_hidden_states.reshape(
            num_tokens // self.downsample_factor,
            audio_hidden_size * self.downsample_factor,
        )
        audio_text_embeds = self.audio_language_adapter(audio_hidden_states)

        text_embeds = self.language_model.embed_input_ids(input_ids)

        # sum pool text and audio embeddings
        inputs_embeds = audio_text_embeds + text_embeds

        time_tensor = torch.tensor(
            [self.n_delay_tokens],
            device=inputs_embeds.device,
            dtype=inputs_embeds.dtype,
        )
        inputs_embeds = inputs_embeds + self.time_embedding(time_tensor)

        hidden_states = self.language_model.model(
            input_ids, positions, intermediate_tensors, inputs_embeds=inputs_embeds
        )

        return hidden_states

    def embed_multimodal(
        self, **kwargs
    ) -> list[torch.Tensor] | torch.Tensor | tuple[torch.Tensor, ...] | None:
        """Transform audio waveforms -> initial whisper post-conv embeddings"""
        audio_inputs = self._parse_and_validate_audio_arrays(**kwargs)

        assert audio_inputs is not None, (
            "For streaming you must provide an audio input at every step."
        )

        multiple_of = self.audio_config.raw_audio_length_per_tok
        assert all(
            (this_audio := audio.shape[0]) % multiple_of == 0 for audio in audio_inputs
        ), (
            f"Every input audio waveform has to be a multiple of {multiple_of}, but"
            f" one is {this_audio} with {(this_audio / multiple_of)=}."
        )

        mel_features = [
            self.whisper_encoder.compute_whisper_melspec(audio).to(
                self.whisper_encoder.dtype
            )
            for audio in audio_inputs
        ]
        seq_lens = [mel.shape[1] for mel in mel_features]
        # [total_num_20ms_frames, hidden_size]
        audio_embeddings = self.whisper_encoder.whisper_encoder.forward_conv(
            mel_features
        )[0]
        conv_stride = self.whisper_encoder.whisper_encoder.total_stride
        audio_embeddings_per_sample = audio_embeddings.split(
            [s // conv_stride for s in seq_lens], dim=0
        )

        # audio_embeddings per sample need to be divisible by 4
        pool_size = self.config.audio_config.block_pool_size
        assert all(
            (this_shape := sample.shape[0]) % pool_size == 0
            for sample in audio_embeddings_per_sample
        ), f"Every audio embedding has to be a multiple of 4, but one is {this_shape}."

        audio_embeddings_per_sample = [
            e.view(e.shape[0] // pool_size, e.shape[1] * pool_size)
            for e in audio_embeddings_per_sample
        ]
        return audio_embeddings_per_sample
