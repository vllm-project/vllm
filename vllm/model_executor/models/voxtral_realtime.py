# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import itertools
import math
from collections.abc import AsyncGenerator, Hashable, Iterable, Iterator, Mapping
from typing import Any

import numpy as np
import torch
from mistral_common.protocol.transcription.request import (
    StreamingMode,
    TranscriptionRequest,
)
from mistral_common.tokens.tokenizers.audio import Audio, AudioConfig

from vllm.compilation.decorators import support_torch_compile
from vllm.config import ModelConfig, SpeechToTextConfig, VllmConfig
from vllm.config.speech_to_text import SpeechToTextParams
from vllm.inputs import PromptType, TokensPrompt
from vllm.logger import init_logger
from vllm.model_executor.models.interfaces import (
    MultiModalEmbeddings,
    SupportsEncoderCudaGraph,
    SupportsRealtime,
)
from vllm.model_executor.models.voxtral import (
    VoxtralDummyInputsBuilder,
    VoxtralForConditionalGeneration,
    VoxtralMultiModalProcessor,
    VoxtralProcessingInfo,
)
from vllm.model_executor.models.whisper_causal import WhisperCausalEncoder
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.parse import MultiModalDataItems
from vllm.multimodal.processing import ProcessorInputs, TimingContext
from vllm.multimodal.processing.processor import (
    MultiModalProcessingResult,
    PlaceholderFeaturesInfo,
)
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors
from vllm.tokenizers import cached_tokenizer_from_config
from vllm.tokenizers.mistral import MistralTokenizer
from vllm.utils.math_utils import round_up_to
from vllm.utils.torch_utils import is_torch_equal_or_newer
from vllm.v1.worker.encoder_cudagraph_defs import (
    ENCODER_CUDAGRAPH_AXIS_KEYS_KWARG,
    EncoderCudaGraphCaptureInputs,
    EncoderCudaGraphConfig,
    EncoderCudaGraphPathConfig,
    EncoderCudaGraphReplayBuffers,
    EncoderItemSpec,
)

from .utils import _flatten_embeddings

logger = init_logger(__name__)

# Largest audio graph batch. Streaming frames arrive one per session per
# step, so larger batches are rare and their launch overhead is amortized.
_MAX_AUDIO_GRAPH_BATCH = 32


def _streaming_prompt_tokens(tokenizer: MistralTokenizer) -> list[int]:
    """The prefix (BOS, left padding, delay) the first audio frame covers."""
    instruct = tokenizer.instruct
    return instruct.start() + instruct.audio_encoder.encode_streaming_tokens()


def _ms_to_samples(config: AudioConfig, ms: float) -> int:
    len_ = config.sampling_rate * ms / 1000
    assert len_.is_integer(), len_
    return int(len_)


def _stream_frames(
    config: AudioConfig, num_prompt_tokens: int
) -> Iterator[tuple[int, int]]:
    """Yield (frame size in samples, new tokens) for each streaming step.

    The first frame covers the prompt, each later one a single step; frames
    extend into their neighbours by the look-back and look-ahead.
    """
    look_ahead = _ms_to_samples(config, config.streaming_look_ahead_ms)
    look_back = _ms_to_samples(config, config.streaming_look_back_ms)
    streaming_step_size = _ms_to_samples(config, 1000 / config.frame_rate)
    start = 0
    end = num_prompt_tokens * config.raw_audio_length_per_tok
    while True:
        frame_start = max(start - look_back, 0)
        frame_end = end + look_ahead
        frame_size = frame_end - frame_start
        num_tokens = (end - start) / config.raw_audio_length_per_tok
        assert num_tokens.is_integer()
        yield frame_size, int(num_tokens)
        start = end
        end += streaming_step_size


class VoxtralRealtimeMultiModalProcessor(VoxtralMultiModalProcessor):
    def _cached_apply_hf_processor(
        self,
        inputs: ProcessorInputs,
        timing_ctx: TimingContext,
    ) -> MultiModalProcessingResult:
        # realtime can't make use of a cache yet
        return self._apply_hf_processor(inputs, timing_ctx)

    def _maybe_apply_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        mm_res: MultiModalProcessingResult,
    ) -> tuple[list[int], Mapping[str, list[PlaceholderFeaturesInfo]]]:
        mm_kwargs = mm_res.kwargs.require_data()

        # there are no placeholder audio tokens for streaming
        # so we need to build the place placeholder positions manually

        # in realtime there is always only one audio input
        audios = mm_kwargs.get("audio", [])
        assert len(audios) == 1, (
            f"Expected only one audio input for realtime, got {mm_kwargs=}"
        )
        tokenizer = self.info.get_tokenizer()
        audio_config = tokenizer.instruct.audio_encoder.audio_config

        audio_array = audios[0]["audio_arrays"].data
        assert isinstance(audio_array, (torch.Tensor, np.ndarray))
        num_audio_samples = audio_array.shape[0]
        length = audio_config.num_audio_tokens(num_audio_samples)

        features_info = PlaceholderFeaturesInfo(
            modality="audio",
            item_idx=0,
            start_idx=0,
            tokens=length
            * [0],  # only used for length computation, so we can take dummy inputs
            is_embed=None,
        )
        return mm_res.prompt_ids, {"audio": [features_info]}


class TimeEmbedding(torch.nn.Module):
    """Sinusoidal Embedding for encoding time."""

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


class VoxtralRealtimeProcessingInfo(VoxtralProcessingInfo):
    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        # Each streaming update carries a single audio frame.
        return {"audio": 1}

    def get_max_audio_array_len(self) -> int:
        # Profile with the largest streaming chunk, the first one: it covers
        # the prompt prefix (BOS + left padding + delay), later chunks are
        # ~1 token. get_max_audio_tokens keeps max_model_len so the encoder
        # cache stays as large as before: each paused session keeps its last
        # chunk cached until it resumes, and a cache sized to one chunk
        # deadlocks a few concurrent sessions.
        tokenizer = self.get_tokenizer()
        audio_config = tokenizer.instruct.audio_encoder.audio_config
        num_prompt_tokens = len(_streaming_prompt_tokens(tokenizer))
        return num_prompt_tokens * audio_config.raw_audio_length_per_tok


class VoxtralRealtimeBuffer:
    def __init__(
        self,
        config: AudioConfig,
        prompt_tokens: list[int],
        max_model_len: int | None = None,
    ) -> None:
        self._config = config
        self._max_model_len = max_model_len

        self._look_ahead_in_samples = _ms_to_samples(
            config, config.streaming_look_ahead_ms
        )
        self._look_back_in_samples = _ms_to_samples(
            config, config.streaming_look_back_ms
        )

        # None signals the end
        self._audio_queue: asyncio.Queue[np.ndarray | None] = asyncio.Queue()
        self._leftover: np.ndarray | None = None
        self._token_queue: asyncio.Queue[int] = asyncio.Queue()

        self._num_prompt_tokens = len(prompt_tokens)
        for token in prompt_tokens:
            self._token_queue.put_nowait(token)

    async def append_audio(self, audio_array: np.ndarray | None) -> None:
        await self._audio_queue.put(audio_array)

    async def append_tokens(self, tokens: Iterable[int]) -> None:
        for token in tokens:
            await self._token_queue.put(token)

    async def get_input_stream(self) -> AsyncGenerator[TokensPrompt]:
        n_tokens = 0
        for frame_size, num_tokens in _stream_frames(
            self._config, self._num_prompt_tokens
        ):
            n_tokens += num_tokens
            if self._max_model_len is not None and n_tokens >= self._max_model_len:
                logger.warning(
                    "Stopping realtime audio stream: reached max_model_len (%d).",
                    self._max_model_len,
                )
                return

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

            yield TokensPrompt(
                prompt_token_ids=next_tokens,
                multi_modal_data={"audio": (frame, None)},
            )


@MULTIMODAL_REGISTRY.register_processor(
    VoxtralRealtimeMultiModalProcessor,
    info=VoxtralRealtimeProcessingInfo,
    dummy_inputs=VoxtralDummyInputsBuilder,
)
@support_torch_compile
class VoxtralRealtimeGeneration(
    VoxtralForConditionalGeneration, SupportsRealtime, SupportsEncoderCudaGraph
):
    requires_raw_input_tokens = True
    # transformers' currently has limited support for MistralCommon backend
    # and cached_get_processor. Let's skip until fixed
    skip_warmup_audio_preprocessing = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)

        # Full cudagraphs are supported for decode-only batches (the encoder's
        # block-pooling attention builder is capture-safe for uniform
        # single-token decode). Mixed/prefill batches are not, so block only
        # pure FULL; FULL_DECODE_ONLY and FULL_AND_PIECEWISE are allowed.
        cudagraph_mode = vllm_config.compilation_config.cudagraph_mode
        assert not cudagraph_mode.mixed_mode().has_full_cudagraphs(), (
            "Voxtral realtime supports full cudagraphs for decode-only batches. "
            "Use cudagraph_mode=FULL_DECODE_ONLY or FULL_AND_PIECEWISE (not FULL)."
        )

        self.time_embedding: TimeEmbedding = TimeEmbedding(
            dim=self.config.text_config.hidden_size
        )

        audio_config = self.tokenizer.instruct.audio_encoder.audio_config
        self.n_delay_tokens = audio_config.get_num_delay_tokens()

        # Audio CUDA graphs cover the two frame lengths a stream produces:
        # the startup frame, which covers the prompt, and the steady frame.
        num_prompt_tokens = len(_streaming_prompt_tokens(self.tokenizer))
        (startup, startup_tokens), (steady, _) = itertools.islice(
            _stream_frames(audio_config, num_prompt_tokens), 2
        )
        self._audio_graph_lengths = (steady, startup)
        self._audio_graph_startup_tokens = startup_tokens
        batch = min(_MAX_AUDIO_GRAPH_BATCH, vllm_config.scheduler_config.max_num_seqs)
        buckets = [1 << i for i in range(batch.bit_length()) if 1 << i < batch]
        self._audio_graph_batch_sizes = (*buckets, batch)
        max_tokens = batch * startup_tokens
        compilation = vllm_config.compilation_config
        self._audio_graph_token_budget = (
            max(compilation.encoder_cudagraph_token_budgets)
            if compilation.encoder_cudagraph_token_budgets
            else max(
                startup_tokens,
                min(
                    max_tokens,
                    compilation.encoder_cudagraph_max_token_budget or max_tokens,
                ),
            )
        )
        # The configurations the graphs were validated against.
        self._audio_graph_enabled = (
            current_platform.is_cuda()
            and vllm_config.model_config.dtype == torch.bfloat16
            and vllm_config.lora_config is None
        )

    def get_encoder_cudagraph_config(self) -> EncoderCudaGraphConfig:
        return EncoderCudaGraphConfig(
            modalities=["audio"] if self._audio_graph_enabled else [],
            buffer_keys=["audio_arrays"],
            out_hidden_size=self.config.audio_config.d_model
            * self.config.audio_config.block_pool_size,
            # The axes fully specify the layout; capture one aggregate budget.
            paths={
                "default": EncoderCudaGraphPathConfig(
                    min_token_budget=self._audio_graph_token_budget
                )
            },
            capture_axes=(
                self._audio_graph_batch_sizes,
                self._audio_graph_lengths,
            ),
        )

    def get_input_modality(self, mm_kwargs: dict[str, Any]) -> str:
        return "audio"

    def get_encoder_cudagraph_budget_range(
        self, vllm_config: VllmConfig
    ) -> tuple[int, int]:
        startup_tokens = self._audio_graph_startup_tokens
        return startup_tokens, startup_tokens * self._audio_graph_batch_sizes[-1]

    def get_encoder_cudagraph_item_specs(
        self, mm_kwargs: dict[str, Any]
    ) -> list[EncoderItemSpec]:
        audios = self._parse_and_validate_audio_arrays(**mm_kwargs)
        if audios is None:
            return []
        stride = int(self.audio_config.raw_audio_length_per_tok)
        return [
            EncoderItemSpec(
                input_size=audio.numel(),
                output_tokens=max(1, audio.numel() // stride),
                batch_key=audio.numel(),
            )
            for audio in audios
        ]

    def select_encoder_cudagraph_items(
        self, mm_kwargs: dict[str, Any], indices: list[int]
    ) -> dict[str, Any]:
        audios = self._parse_and_validate_audio_arrays(**mm_kwargs)
        assert audios is not None
        selected = [audios[i] for i in indices]
        batch = round_up_to(list(self._audio_graph_batch_sizes), len(selected))
        length = selected[0].numel() if selected else None
        return {
            "audio_arrays": selected,
            ENCODER_CUDAGRAPH_AXIS_KEYS_KWARG: (batch, length),
        }

    def prepare_encoder_cudagraph_capture_inputs(
        self,
        token_budget: int,
        max_batch_size: int,
        max_frames_per_batch: int,
        device: torch.device,
        dtype: torch.dtype,
        path: str = "default",
        axis_keys: tuple[Hashable, ...] | None = None,
    ) -> EncoderCudaGraphCaptureInputs:
        assert axis_keys is not None
        batch, length = axis_keys
        assert isinstance(batch, int) and isinstance(length, int)
        return EncoderCudaGraphCaptureInputs(
            values={
                "audio_arrays": torch.zeros(
                    (batch, length), device=device, dtype=torch.float32
                ),
            }
        )

    def prepare_encoder_cudagraph_replay_buffers(
        self,
        mm_kwargs: dict[str, Any],
        max_batch_size: int,
        max_frames_per_batch: int,
        path: str = "default",
    ) -> EncoderCudaGraphReplayBuffers:
        audios = self._parse_and_validate_audio_arrays(**mm_kwargs)
        assert audios is not None
        return EncoderCudaGraphReplayBuffers(
            values={"audio_arrays": torch.stack(audios)}
        )

    def encoder_cudagraph_forward(
        self, inputs: dict[str, torch.Tensor], path: str = "default"
    ) -> torch.Tensor:
        return self._embed_audio(inputs["audio_arrays"])

    def encoder_eager_forward(
        self, mm_kwargs: dict[str, Any], path: str = "default"
    ) -> torch.Tensor:
        outputs = self.embed_multimodal(**mm_kwargs)
        assert isinstance(outputs, (list, tuple))
        return torch.cat(outputs, dim=0)

    def _embed_audio(self, audio: torch.Tensor) -> torch.Tensor:
        """Log-mel, causal convs and block pooling of one waveform
        `(num_samples,)` or a batch of equal-length waveforms
        `(batch, num_samples)`.

        Returns `(num_tokens, d_model * block_pool_size)`, with the items'
        tokens concatenated in order. Shared by the eager and CUDA graph paths.
        """
        whisper = self.whisper_encoder
        assert isinstance(whisper.whisper_encoder, WhisperCausalEncoder)
        mel = whisper.compute_whisper_melspec(audio).to(whisper.dtype)
        # Drop the leftmost mel frame if the frame count is odd.
        mel = mel[..., mel.shape[-1] % 2 :]
        hidden = whisper.whisper_encoder.forward_conv(mel)
        # Drop the leftmost frames that don't fill a pooling block.
        pool = self.config.audio_config.block_pool_size
        hidden = hidden[..., hidden.shape[-2] % pool :, :]
        assert hidden.shape[-2] > 0, (
            f"Audio of {audio.shape[-1]} samples is too short to embed"
        )
        return hidden.reshape(-1, pool * hidden.shape[-1])

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
        prompt_tokens = _streaming_prompt_tokens(tokenizer)

        # Get left/right padding audio
        left_pad, right_pad = audio_encoder.get_padding_audio()

        buffer = VoxtralRealtimeBuffer(
            config, prompt_tokens, max_model_len=model_config.max_model_len
        )

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

        # Feed output tokens back into the buffer. Idle waits are normal here;
        # request cleanup still cancels this task through the finally block.
        async def feed_tokens():
            while True:
                all_outputs = await input_stream.get()
                await buffer.append_tokens(all_outputs[-1:])

        audio_task = asyncio.create_task(feed_audio())
        token_task = asyncio.create_task(feed_tokens())

        try:
            async for prompt in buffer.get_input_stream():
                yield prompt
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
        """Transform audio waveforms -> initial whisper post-conv embeddings."""
        audio_inputs = self._parse_and_validate_audio_arrays(**kwargs)

        if audio_inputs is None:
            logger.warning(
                "Realtime model received no audio inputs in "
                "embed_multimodal. Returning empty embeddings."
            )
            return []

        return [self._embed_audio(audio) for audio in audio_inputs]

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
        stt_params: SpeechToTextParams,
    ) -> PromptType:
        if stt_params.task_type != "transcribe":
            raise ValueError(
                f'Voxtral Realtime only supports "transcribe", '
                f"got {stt_params.task_type}"
            )
        audio = stt_params.audio
        model_config = stt_params.model_config
        stt_config = stt_params.stt_config
        language = stt_params.language

        tokenizer = cached_tokenizer_from_config(model_config)
        audio = Audio(audio, int(stt_config.sample_rate), format="wav")  # lossless

        req = TranscriptionRequest(
            model=model_config.model,
            audio=audio.to_base64(audio.format),
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
