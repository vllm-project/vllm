# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import math
from collections.abc import Mapping
from typing import AsyncGenerator, Literal, cast

import numpy as np
import torch
from mistral_common.protocol.instruct.chunk import RawAudio
from mistral_common.protocol.transcription.request import (
    StreamingMode,
    TranscriptionRequest,
)
from mistral_common.tokens.tokenizers.audio import Audio

from vllm.config import ModelConfig, SpeechToTextConfig, VllmConfig
from vllm.inputs.data import PromptType, TokensPrompt
from vllm.logger import init_logger
from vllm.model_executor.models.interfaces import MultiModalEmbeddings, SupportsRealtime
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
from vllm.multimodal.processing import BaseDummyInputsBuilder
from vllm.multimodal.processing.processor import (
    MultiModalPromptUpdates,
    PlaceholderFeaturesInfo,
)
from vllm.sequence import IntermediateTensors
from vllm.tokenizers import cached_tokenizer_from_config

from .utils import (
    _flatten_embeddings,
)
from mistral_common.tokens.tokenizers.audio import AudioConfig
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer

logger = init_logger(__name__)

_PRE_ALLOCATE_BUFFER_SIZE_IN_S = 30

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


def _expand_tensor(input_tensor: torch.Tensor, scaling: int) -> torch.Tensor:
    # 1. Multiply by the scaling factor (e.g. 4)
    base = input_tensor * scaling

    # 2. Create the offsets, e.g. [0, 1, 2, 3]
    offsets = torch.arange(scaling, device=input_tensor.device)

    # 3. Use broadcasting, e.g. (N, 1) + (4,) results in (N, 4)
    # Then flatten back to 1D
    return (base.unsqueeze(1) + offsets).view(-1)


class VoxtralRealtimeBuffer:
    def __init__(self, config: AudioConfig) -> None:
        self._config = config

        # TODO(Patrick) - move tokenizer config
        self._look_ahead_in_ms = 2.5
        self._look_back_in_ms = 52.5

        self._sampling_rate = self._config.sampling_rate

        self._look_ahead = self._get_len_in_samples(self._look_ahead_in_ms)
        self._look_back = self._get_len_in_samples(self._look_back_in_ms)
        self._streaming_size = self._get_len_in_samples(1000 / self._config.frame_rate)

        # mutable objects
        streaming_delay = self._get_len_in_samples(self._config.transcription_delay_ms)
        n_left_pad_samples = (
            self._config.raw_audio_length_per_tok * self._config.n_left_pad_tokens
        )
        n_left_pad_samples = 0
        self._start = 0
        self._end = streaming_delay + n_left_pad_samples + self._streaming_size

        # always pre-allocate 30 second buffers
        self._buffer_size = _PRE_ALLOCATE_BUFFER_SIZE_IN_S * self._sampling_rate
        self._buffer: np.ndarray = np.empty(self._buffer_size, dtype=np.float32)
        self._filled_buffer_len = 0

    @property
    def start_idx(self):
        return max(self._start - self._look_back, 0)

    @property
    def end_idx(self):
        return self._end + self._look_ahead

    @property
    def is_chunk_complete(self) -> bool:
        return self._filled_buffer_len >= self.end_idx

    def _get_len_in_samples(self, len_in_ms: float) -> int:
        _len_in_s = self._sampling_rate * len_in_ms / 1000
        assert _len_in_s.is_integer(), _len_in_s
        len_in_s = int(_len_in_s)

        return len_in_s

    def _allocate_new_buffer(self) -> None:
        # allocate new buffer
        new_buffer = np.empty(self._buffer_size, dtype=np.float32)
        left_to_copy = max(self._filled_buffer_len - self.start_idx, 0)

        if left_to_copy > 0:
            new_buffer[:left_to_copy] = self._buffer[self.start_idx: self._filled_buffer_len]

        del self._buffer
        self._buffer = new_buffer

        self._filled_buffer_len = left_to_copy
        self._start = self._look_back
        self._end = self._start + self._streaming_size

    def add_audio_chunk(self, audio_chunk: np.ndarray) -> None:
        put_end_idx = self._filled_buffer_len + len(audio_chunk)

        if put_end_idx > self._buffer_size:
            self._allocate_new_buffer()

        self._buffer[self._filled_buffer_len : self._filled_buffer_len + len(audio_chunk)] = audio_chunk
        self._filled_buffer_len += len(audio_chunk)

    def get_audio_chunk(self) -> np.ndarray | None:
        if not self.is_chunk_complete:
            return None

        audio_chunk = self._buffer[self.start_idx: self.end_idx]
        self._start = self._end
        self._end += self._streaming_size

        return audio_chunk


@MULTIMODAL_REGISTRY.register_processor(
    VoxtralStreamingMultiModalProcessor,
    info=VoxtralProcessingInfo,
    dummy_inputs=VoxtralDummyInputsBuilder,
)
class VoxtralStreamingGeneration(VoxtralForConditionalGeneration, SupportsRealtime):
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

    # for realtime transcription
    @classmethod
    async def buffer_realtime_audio(
        cls,
        audio_stream: AsyncGenerator[np.ndarray, None],
        input_stream: asyncio.Queue[list[int]],
        model_config: ModelConfig,
    ) -> AsyncGenerator[PromptType, None]:
        tokenizer = cached_tokenizer_from_config(model_config)
        audio_encoder = tokenizer.instruct.audio_encoder
        config = audio_encoder.audio_config

        buffer = VoxtralRealtimeBuffer(config)
        is_first_yield = True

        count = 0

        async for audio in audio_stream:
            print("audio shape", audio.shape)
            buffer.add_audio_chunk(audio)

            while (new_audio := buffer.get_audio_chunk()) is not None:
                print("new_audio shape", new_audio.shape)

                if is_first_yield:
                    # make sure that input_stream is empty
                    assert input_stream.empty()

                    audio = Audio(new_audio, config.sampling_rate, format="wav")

                    request = TranscriptionRequest(
                        streaming=StreamingMode.ONLINE,
                        audio=RawAudio.from_audio(audio),
                        language=None,
                    )
                    # audio will be correctly padded
                    # and correct token ids will be created
                    audio_enc = tokenizer.mistral.encode_transcription(request)

                    token_ids = audio_enc.tokens
                    new_audio = audio_enc.audios[0].audio_array

                    is_first_yield = False
                else:
                    # pop last element from input_stream
                    all_outputs = await asyncio.wait_for(input_stream.get(), timeout=10.0)
                    token_ids = all_outputs[-1:]

                count += 1
                print(f"COUNT {count}")
                print(20 * "-" + "INPUT" + 20 * "-")
                print("token_ids", token_ids)
                print("audio shape", new_audio.shape)
                print("audio", new_audio)

                multi_modal_data = {"audio": (new_audio, None)}
                yield TokensPrompt(
                    prompt_token_ids=token_ids, multi_modal_data=multi_modal_data
                )

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
        input_ids: torch.Tensor | None,
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

        whisper_positions = _expand_tensor(positions, pool_size)
        audio_hidden_states = self.whisper_encoder.whisper_encoder(
            inputs_embeds, whisper_positions
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
        t_cond = self.time_embedding(time_tensor)

        hidden_states = self.language_model.model(
            input_ids,
            positions,
            intermediate_tensors,
            inputs_embeds=inputs_embeds,
            t_cond=t_cond,
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

        def _truncate_left(
            sample: torch.Tensor, mult_of: int, pos: int
        ) -> torch.Tensor:
            assert pos in [0, 1], pos
            if (ctx := sample.shape[pos] % mult_of) != 0:
                sample = sample[ctx:] if pos == 0 else sample[:, ctx:]
                assert sample.shape[pos] > 0, (
                    f"Sample is empty after truncation with ctx {ctx}"
                )

            return sample

        mel_features = [
            self.whisper_encoder.compute_whisper_melspec(audio).to(
                self.whisper_encoder.dtype
            )
            for audio in audio_inputs
        ]

        # we truncate the left most mel feature
        # if the sequence length in impair
        mel_features = [_truncate_left(mel, 2, 1) for mel in mel_features]

        seq_lens = [mel.shape[1] for mel in mel_features]
        # [total_num_20ms_frames, hidden_size]
        audio_embeddings = self.whisper_encoder.whisper_encoder.forward_conv(
            mel_features
        )
        conv_stride = self.whisper_encoder.whisper_encoder.total_stride
        audio_embeddings_per_sample = audio_embeddings.split(
            [s // conv_stride for s in seq_lens], dim=0
        )

        # audio_embeddings per sample need to be divisible by 4
        pool_size = self.config.audio_config.block_pool_size

        audio_embeddings_per_sample = [
            _truncate_left(sample, pool_size, 0)
            for sample in audio_embeddings_per_sample
        ]

        audio_embeddings_per_sample = [
            e.view(e.shape[0] // pool_size, e.shape[1] * pool_size)
            for e in audio_embeddings_per_sample
        ]
        return audio_embeddings_per_sample

    @classmethod
    def get_speech_to_text_config(
        cls, model_config: ModelConfig, task_type: str
    ) -> SpeechToTextConfig:
        tokenizer = cached_tokenizer_from_config(model_config)
        audio_config = tokenizer.instruct.audio_encoder.audio_config
        sample_rate = audio_config.sampling_rate
        return SpeechToTextConfig(
            max_audio_clip_s=None,  # only limited by memory
            sample_rate=sample_rate,
            min_energy_split_window_size=None,
        )

    @classmethod
    # for speech-to-text transcription
    def get_generation_prompt(
        cls,
        audio: np.ndarray,
        model_config: ModelConfig,
        stt_config: SpeechToTextConfig,
        language: str | None,
        task_type: Literal["transcribe", "translate"],
        request_prompt: str,
        to_language: str | None,
    ) -> PromptType:
        tokenizer = cached_tokenizer_from_config(model_config)
        audio = Audio(audio, int(stt_config.sample_rate), format="wav")  # lossless

        req = TranscriptionRequest(
            model=model_config.model,
            audio=RawAudio.from_audio(audio),
            language=language,
            streaming=StreamingMode.OFFLINE,
        )

        tokenized = tokenizer.instruct.encode_transcription(req)
        audio = (tokenized.audios[0].audio_array, stt_config.sample_rate)
        prompts_dict = {"multi_modal_data": {"audio": audio}}
        prompts_dict["prompt_token_ids"] = tokenized.tokens
        return cast(PromptType, prompts_dict)
