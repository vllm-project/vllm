# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import asdict

import pytest
from mistral_common.audio import Audio
from mistral_common.protocol.instruct.chunk import RawAudio
from mistral_common.protocol.transcription.request import (
    StreamingMode,
    TranscriptionRequest,
)
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.tokens.tokenizers.tekken import SpecialTokenPolicy

from vllm import LLM, EngineArgs, SamplingParams
from vllm.assets.audio import AudioAsset
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.v1.engine.async_llm import AsyncLLM

MODEL_NAME = "mistralai/Voxtral-Mini-4B-Realtime-2602"
ENGINE_CONFIG = dict(
    model=MODEL_NAME,
    max_model_len=8192,
    max_num_seqs=4,
    limit_mm_per_prompt={"audio": 1},
    config_format="mistral",
    load_format="mistral",
    tokenizer_mode="mistral",
    enforce_eager=True,
    gpu_memory_utilization=0.9,
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


def test_voxtral_realtime_forward(audio_assets, tokenizer, engine):
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


@pytest.mark.asyncio
async def test_voxtral_realtime_generator(audio_assets, tokenizer, async_engine):
    # Lazy import to avoid CUDA-reinitialization error
    from vllm.model_executor.models.voxtral_realtime import VoxtralRealtimeBuffer

    sampling_params = SamplingParams(temperature=0.0, max_tokens=1)
    audio_config = tokenizer.instruct_tokenizer.audio_encoder.audio_config

    output_tokens_list = []
    for i, audio_asset in enumerate(audio_assets):
        output_tokens = []
        audio = Audio.from_file(audio_asset.get_local_path(), strict=False)

        req = TranscriptionRequest(
            streaming=StreamingMode.OFFLINE,
            audio=RawAudio.from_audio(audio),
            language=None,
        )
        audio_enc = tokenizer.encode_transcription(req)

        buffer = VoxtralRealtimeBuffer(audio_config, audio_enc.tokens)
        await buffer.append_audio(audio_enc.audios[0].audio_array)
        await buffer.append_audio(None)

        request_id = f"session-{i}"

        async for resp in async_engine.generate(
            prompt=buffer.get_input_stream(),
            sampling_params=sampling_params,
            request_id=request_id,
        ):
            tokens = resp.outputs[0].token_ids[-1:]
            output_tokens.extend(tokens)
            await buffer.append_tokens(tokens)

        output_tokens_list.append(output_tokens)

    texts = [
        tokenizer.decode(output_tokens, special_token_policy=SpecialTokenPolicy.IGNORE)
        for output_tokens in output_tokens_list
    ]
    texts[1] = texts[1].replace("a base hit", "OBS").replace("oh my", "oh, my")
    assert texts == EXPECTED_TEXT
