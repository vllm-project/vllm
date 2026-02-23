# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import math
from collections.abc import AsyncGenerator, Iterable, Iterator, Mapping
from typing import Literal

import numpy as np
import torch
from mistral_common.protocol.instruct.chunk import RawAudio
from mistral_common.protocol.transcription.request import (
    StreamingMode,
    TranscriptionRequest,
)
from mistral_common.tokens.tokenizers.audio import Audio, AudioConfig

from vllm.compilation.decorators import support_torch_compile
from vllm.config import ModelConfig, SpeechToTextConfig, VllmConfig
from vllm.engine.protocol import StreamingInput
from vllm.envs import VLLM_ENGINE_ITERATION_TIMEOUT_S
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
from vllm.utils.torch_utils import is_torch_equal_or_newer

from .utils import (
    _flatten_embeddings,
)

logger = init_logger(__name__)


class VoxtralRealtimeMultiModalProcessor(VoxtralMultiModalProcessor):
    def __init__(
        self,
        info: _I,
        dummy_inputs: BaseDummyInputsBuilder[_I],
        *,
        cache: BaseMultiModalProcessorCache | None = None,
    ) -> None:
        # realtime can't make use of a cache yet
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

        # in realtime there is always only one audio input
        audios = mm_kwargs.get("audio", [])
        assert len(audios) == 1, (
            f"Expected only one audio input for realtime, got {mm_kwargs=}"
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
    def __init__(self, config: AudioConfig, prompt_tokens: list[int]) -> None:
        self._config = config

        _look_ahead_in_ms = self._config.streaming_look_ahead_ms
        _look_back_in_ms = self._config.streaming_look_back_ms
        self._look_ahead_in_samples = self._ms_to_samples(_look_ahead_in_ms)
        self._look_back_in_samples = self._ms_to_samples(_look_back_in_ms)

        # None signals the end
        self._audio_queue: asyncio.Queue[np.ndarray | None] = asyncio.Queue()
        self._leftover: np.ndarray | None = None
        self._token_queue: asyncio.Queue[int] = asyncio.Queue()

        self._initial_end = len(prompt_tokens) * self._config.raw_audio_length_per_tok
        for token in prompt_tokens:
            self._token_queue.put_nowait(token)

    def _generate_frame_size_and_num_tokens(self) -> Iterator[tuple[int, int]]:
        streaming_step_size = self._ms_to_samples(1000 / self._config.frame_rate)
        start = 0
        end = self._initial_end
        while True:
            frame_start = max(start - self._look_back_in_samples, 0)
            frame_end = end + self._look_ahead_in_samples
            frame_size = frame_end - frame_start
            num_tokens = (end - start) / self._config.raw_audio_length_per_tok
            assert num_tokens.is_integer()
            yield frame_size, int(num_tokens)
            start = end
            end += streaming_step_size

    def _ms_to_samples(self, ms: float) -> int:
        len_ = self._config.sampling_rate * ms / 1000
        assert len_.is_integer(), len_
        return int(len_)

    async def append_audio(self, audio_array: np.ndarray | None) -> None:
        await self._audio_queue.put(audio_array)

    async def append_tokens(self, tokens: Iterable[int]) -> None:
        for token in tokens:
            await self._token_queue.put(token)

    async def get_input_stream(self) -> AsyncGenerator[StreamingInput]:
        for frame_size, num_tokens in self._generate_frame_size_and_num_tokens():
            next_tokens = [await self._token_queue.get() for _ in range(num_tokens)]

            audio_arrays: list[np.ndarray] = (
                [self._leftover] if self._leftover is not None else []
            )
            while sum(len(arr) for arr in audio_arrays) < frame_size:
                arr = await self._audio_queue.get()
                if arr is None:
                    return
                audio_arrays.append(arr)

            audio_array = np.concatenate(audio_arrays)
            frame = audio_array[:frame_size]

            # The current stride took look_ahead_in_samples audio of the next sample
            # In addition the next sample will take look_back_in_samples audio of
            # the current sample => So let's put both of this into the leftover
            stride = (
                frame_size - self._look_ahead_in_samples - self._look_back_in_samples
            )
            assert stride > 0, f"{stride=} must be positive"

            self._leftover = audio_array[stride:]

            yield StreamingInput(
                TokensPrompt(
                    prompt_token_ids=next_tokens,
                    multi_modal_data={"audio": (frame, None)},
                )
            )


@MULTIMODAL_REGISTRY.register_processor(
    VoxtralRealtimeMultiModalProcessor,
    info=VoxtralProcessingInfo,
    dummy_inputs=VoxtralDummyInputsBuilder,
)
@support_torch_compile
class VoxtralRealtimeGeneration(VoxtralForConditionalGeneration, SupportsRealtime):
    requires_raw_input_tokens = True
    # transformers' currently has limited support for MistralCommon backend
    # and cached_get_processor. Let's skip until fixed
    skip_warmup_audio_preprocessing = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)

        assert (
            not vllm_config.compilation_config.cudagraph_mode.has_full_cudagraphs()
        ), "Voxtral realtime doesn't support full cudagraphs yet. Please use PIECEWISE."

        self.time_embedding: TimeEmbedding = TimeEmbedding(
            dim=self.config.text_config.hidden_size
        )

        audio_config = self.tokenizer.instruct.audio_encoder.audio_config
        self.n_delay_tokens = audio_config.get_num_delay_tokens()

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

        # Get prompt tokens (streaming prefix tokens) without encoding audio
        prompt_tokens = (
            tokenizer.instruct.start() + audio_encoder.encode_streaming_tokens()
        )

        # Get left/right padding audio
        left_pad, right_pad = audio_encoder.get_padding_audio()

        buffer = VoxtralRealtimeBuffer(config, prompt_tokens)

        # Feed audio with padding into buffer in background
        async def feed_audio():
            yielded_first_chunk = False
            async for audio_chunk in audio_stream:
                if not yielded_first_chunk:
                    yielded_first_chunk = True
                    # Prepend left padding before first real audio
                    await buffer.append_audio(left_pad.audio_array)
                await buffer.append_audio(audio_chunk)
            # Append right padding at the end
            await buffer.append_audio(right_pad.audio_array)
            await buffer.append_audio(None)  # signal end

        # Feed output tokens back into buffer in background
        async def feed_tokens():
            while True:
                all_outputs = await asyncio.wait_for(
                    input_stream.get(),
                    timeout=VLLM_ENGINE_ITERATION_TIMEOUT_S,
                )
                await buffer.append_tokens(all_outputs[-1:])

        audio_task = asyncio.create_task(feed_audio())
        token_task = asyncio.create_task(feed_tokens())

        try:
            async for streaming_input in buffer.get_input_stream():
                yield streaming_input.prompt
        finally:
            audio_task.cancel()
            token_task.cancel()

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
        """Pass post-conv embeddings directly as input.

        For realtime models, multimodal embeddings are required at every
        decode step.  If they are missing (e.g. due to an empty audio
        commit, encoder-cache eviction under GPU memory pressure, or a
        client disconnect), return zero embeddings instead of crashing
        the engine so that all other in-flight requests stay alive.
        """
        if multimodal_embeddings is None or len(multimodal_embeddings) == 0:
            logger.warning(
                "Realtime model received empty multimodal embeddings "
                "for %d input tokens. Returning zero embeddings to "
                "avoid engine crash.",
                input_ids.shape[0],
            )
            pool_size = self.config.audio_config.block_pool_size
            embed_dim = self.config.audio_config.d_model * pool_size
            return torch.zeros(
                input_ids.shape[0],
                embed_dim,
                dtype=self.whisper_encoder.dtype,
                device=input_ids.device,
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
        if is_torch_equal_or_newer("2.11"):
            inputs_embeds = inputs_embeds.view(
                inputs_embeds.shape[0] * pool_size, inputs_embeds.shape[1] // pool_size
            )
        else:
            # TODO Use reshape + clone to break the view chain and avoid output
            # aliasing input bug in torch.compile's AOT autograd cache.
            # Without clone(), if any downstream operation returns a view that's
            # connected to this view of inputs_embeds, the AOT autograd cache
            # fails to pickle the ViewMetaSequence containing SymInt shapes.
            # This will be fixed in pytorch 2.11 and beyond.
            # issue: https://github.com/pytorch/pytorch/issues/174299
            inputs_embeds = inputs_embeds.reshape(
                inputs_embeds.shape[0] * pool_size, inputs_embeds.shape[1] // pool_size
            ).clone()

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

        time_tensor = torch.full(
            (1,),
            fill_value=self.n_delay_tokens,
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

        if audio_inputs is None:
            logger.warning(
                "Realtime model received no audio inputs in "
                "embed_multimodal. Returning empty embeddings."
            )
            return []

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

        return TokensPrompt(
            prompt_token_ids=tokenized.tokens,
            multi_modal_data={
                "audio": (tokenized.audios[0].audio_array, stt_config.sample_rate)
            },
        )
