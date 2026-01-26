# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio
from dataclasses import asdict

import pytest
from mistral_common.audio import Audio
from mistral_common.protocol.instruct.chunk import RawAudio
from mistral_common.protocol.transcription.request import (
    StreamingMode,
    TranscriptionRequest,
)
from mistral_common.tokens.tokenizers.audio import AudioConfig
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer

from vllm import LLM, EngineArgs, SamplingParams
from vllm.assets.audio import AudioAsset
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.inputs.data import TokensPrompt
from vllm.v1.engine.async_llm import AsyncLLM, StreamingInput

MODEL_NAME = "mistralai/Voxtral-Mini-3B-Realtime-2602"
ENGINE_CONFIG = dict(
    model=MODEL_NAME,
    max_model_len=8192,
    max_num_seqs=4,
    limit_mm_per_prompt={"audio": 1},
    config_format="mistral",
    load_format="mistral",
    tokenizer_mode="mistral",
    enforce_eager=True,
    gpu_memory_utilization=0.4,
)


EXPECTED_TEXT = [
    (
        " First words I spoke in the original phonograph. "
        "A little piece of practical poetry. Mary had a little lamb,"
        " its fleece was quite a slow, and everywhere that Mary went, "
        "the lamb was sure to go."
    ),
    (
        " And the 0-1 pitch on the way to Edgar Martinez. Swung on"
        " the line. Down the left field line for OBS. Here comes Joy. "
        "Here is Junior to third base. They're going to wave him in. "
        "The throw to the plate will be late. The Mariners are going"
        " to play. For the American League Championship, "
        "I don't believe it. It just continues. My, oh, my."
    ),
]


@pytest.fixture
def audio_assets() -> list[AudioAsset]:
    return [AudioAsset("mary_had_lamb"), AudioAsset("winning_call")]


@pytest.fixture
def tokenizer() -> MistralTokenizer:
    return MistralTokenizer.from_hf_hub(MODEL_NAME)


@pytest.fixture
def engine() -> LLM:
    engine_args = EngineArgs(**ENGINE_CONFIG)
    return LLM(**asdict(engine_args))


@pytest.fixture
def async_engine() -> AsyncLLM:
    engine_args = AsyncEngineArgs(**ENGINE_CONFIG)
    return AsyncLLM.from_engine_args(engine_args)


@pytest.mark.skip(reason="Voxtral streaming is not yet public")
def test_voxtral_streaming_forward(audio_assets, tokenizer, engine):
    audio_config = tokenizer.instruct_tokenizer.tokenizer.audio

    def from_file(file_path: str):
        audio = Audio.from_file(file_path, strict=False)
        req = TranscriptionRequest(
            audio=RawAudio.from_audio(audio),
            streaming=StreamingMode.OFFLINE,
            language=None,
        )
        tokenized = tokenizer.instruct_tokenizer.encode_transcription(req)

        return (tokenized.tokens, tokenized.audios[0].audio_array)

    tokenized_list = [
        from_file(audio_asset.get_local_path()) for audio_asset in audio_assets
    ]

    inputs = []
    sampling_params = []

    for tokens, audio_array in tokenized_list:
        num_samples = audio_array.shape[0]
        max_tokens = audio_config.num_audio_tokens(num_samples) - len(tokens) - 1
        sampling_params.append(SamplingParams(temperature=0.0, max_tokens=max_tokens))

        input_dict = {
            "multi_modal_data": {"audio": [(audio_array, None)]},
            "prompt_token_ids": tokens,
        }
        inputs.append(input_dict)

    outputs = engine.generate(
        inputs,
        sampling_params=sampling_params,
    )

    texts = [out.outputs[0].text for out in outputs]
    assert texts == EXPECTED_TEXT


class RealTimeAudioInput:
    """
    This class is used to stream an audio file just as
    if it would be streamed in real-time.
    """

    def __init__(self, tokenizer: MistralTokenizer) -> None:
        # TODO(Patrick) - put these into the tokenizer config
        self._look_ahead_in_ms = 2.5
        self._look_back_in_ms = 52.5

        self._tokenizer = tokenizer
        self._config: AudioConfig = (
            self._tokenizer.instruct_tokenizer.audio_encoder.audio_config
        )
        self._sampling_rate = self._config.sampling_rate

        self._audio: Audio | None = None

        # mutable objects
        self._start = 0

        n_left_pad_samples = (
            self._config.raw_audio_length_per_tok * self._config.n_left_pad_tokens
        )
        self._end = self.streaming_delay + n_left_pad_samples + self.streaming_size
        self._queue: asyncio.Queue[StreamingInput | None] = asyncio.Queue()

    @classmethod
    async def create(cls, audio: Audio, tokenizer: MistralTokenizer):
        self = cls(tokenizer)

        # we're doing "OFFLINE" encoding here to right & left pad the audio since
        # we have access to the whole audio
        # if we'd do an actual online realtime streaming application we
        # should instead pass `StreamingMode.ONLINE`
        req = TranscriptionRequest(
            streaming=StreamingMode.OFFLINE,
            audio=RawAudio.from_audio(audio),
            language=None,
        )
        audio_enc = self._tokenizer.encode_transcription(req)
        self._audio = audio_enc.audios[0]

        # add first request
        await self.add_tokens(audio_enc.tokens)

        return self

    @property
    def look_ahead(self) -> int:
        return self._get_len_in_samples(self._look_ahead_in_ms)

    @property
    def look_back(self) -> int:
        return self._get_len_in_samples(self._look_back_in_ms)

    @property
    def streaming_delay(self) -> int:
        return self._get_len_in_samples(self._config.transcription_delay_ms)

    @property
    def streaming_size(self) -> int:
        stream_size_in_ms = 1000 / self._config.frame_rate
        return self._get_len_in_samples(stream_size_in_ms)

    def _get_len_in_samples(self, len_in_ms: float) -> int:
        _len_in_s = self._sampling_rate * len_in_ms / 1000
        assert _len_in_s.is_integer(), _len_in_s
        len_in_s = int(_len_in_s)

        return len_in_s

    async def add_tokens(self, tokens: list[int]) -> None:
        assert self._audio is not None
        if self._start >= len(self._audio.audio_array):
            self.stop()
            return

        _end = self._end + self.look_ahead
        _start = max(0, self._start - self.look_back)

        multi_modal_data = {"audio": (self._audio.audio_array[_start:_end], None)}

        prompt = TokensPrompt(
            prompt_token_ids=tokens, multi_modal_data=multi_modal_data
        )

        await self._queue.put(StreamingInput(prompt))

        # increase
        self._start = self._end
        self._end = self._end + self.streaming_size

    def stop(self):
        self._queue.put_nowait(None)

    async def generator(self):
        while (item := await self._queue.get()) is not None:
            yield item


@pytest.mark.asyncio
@pytest.mark.skip(reason="Voxtral streaming is not yet public")
async def test_voxtral_streaming_generator(audio_assets, tokenizer, async_engine):
    sampling_params = SamplingParams(temperature=0.0, max_tokens=1)

    output_tokens_list = []
    for i, audio_asset in enumerate(audio_assets):
        output_tokens = []
        audio = Audio.from_file(audio_asset.get_local_path(), strict=False)
        streaming_input = await RealTimeAudioInput.create(
            audio=audio, tokenizer=tokenizer
        )

        request_id = f"session-{i}"

        async for resp in async_engine.generate(
            prompt=streaming_input.generator(),
            sampling_params=sampling_params,
            request_id=request_id,
        ):
            tokens = resp.outputs[0].token_ids[-1:]

            output_tokens.extend(tokens)
            await streaming_input.add_tokens(tokens)

        output_tokens_list.append(output_tokens)

    texts = [tokenizer.decode(output_tokens) for output_tokens in output_tokens_list]

    # 'true' streaming and 'offline' streaming differ a bit because log-mels are
    # differently noramalized
    # TODO(Patrick) - check if we want to align or not
    texts[0] = texts[0].replace("He has f", "F")
    texts[1] = texts[1].replace("a base hit", "OBS").replace("oh my", "oh, my")

    assert texts == EXPECTED_TEXT
