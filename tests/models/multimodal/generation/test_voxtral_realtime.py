# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import contextlib

import pytest
import pytest_asyncio
from mistral_common.protocol.transcription.request import (
    StreamingMode,
    TranscriptionRequest,
)
from mistral_common.tokens.tokenizers.audio import Audio
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.tokens.tokenizers.tekken import SpecialTokenPolicy

from vllm import LLM, EngineArgs, SamplingParams
from vllm.assets.audio import AudioAsset
from vllm.distributed import cleanup_dist_env_and_memory
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.utils.math_utils import cdiv
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.kv_cache_interface import SlidingWindowSpec

from ....utils import ROCM_ENGINE_KWARGS, wait_for_rocm_memory_to_settle

MODEL_NAME = "mistralai/Voxtral-Mini-4B-Realtime-2602"
AUDIO_LAYER_NAME = "whisper_encoder.whisper_encoder.layers.0.layers.self_attn.attn"
ENGINE_CONFIG = {
    "model": MODEL_NAME,
    "max_model_len": 8192,
    "max_num_seqs": 4,
    "limit_mm_per_prompt": {"audio": 1},
    "config_format": "mistral",
    "load_format": "mistral",
    "tokenizer_mode": "mistral",
    "enforce_eager": True,
    "gpu_memory_utilization": 0.9,
    **ROCM_ENGINE_KWARGS,
}


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


def _normalize(texts: list[str]) -> list[str]:
    # The model occasionally transcribes "OBS" as "a base hit" and
    # "oh, my" as "oh my", but both are acoustically valid. Normalise so
    # the assertion is stable across runs and hardware.
    texts[1] = texts[1].replace("a base hit", "OBS").replace("oh my", "oh, my")
    return texts


def assert_encoder_kv_cache_spec(engine: LLM) -> None:
    vllm_config = engine.llm_engine.vllm_config
    audio_config = vllm_config.model_config.hf_config.audio_config
    kv_cache_specs_per_rank = engine.llm_engine.model_executor.get_kv_cache_specs()

    assert len(kv_cache_specs_per_rank) == 1
    kv_cache_specs = kv_cache_specs_per_rank[0]
    assert AUDIO_LAYER_NAME in kv_cache_specs, kv_cache_specs.keys()
    spec = kv_cache_specs[AUDIO_LAYER_NAME]

    assert audio_config.sliding_window == 750
    assert audio_config.block_pool_size == 4
    assert isinstance(spec, SlidingWindowSpec)
    assert spec.block_size == 16
    assert spec.num_kv_heads == 128
    # cdiv(750, 4) == 188 pooled tokens cover the model's window; the extra
    # +1 is an eviction margin (see whisper_causal.py get_kv_cache_spec).
    assert spec.sliding_window == cdiv(750, 4) + 1 == 189
    assert (
        spec.max_admission_blocks_per_request(
            max_num_batched_tokens=1,
            max_model_len=vllm_config.model_config.max_model_len,
        )
        == 13
    )


@pytest.fixture
def audio_assets() -> list[AudioAsset]:
    return [AudioAsset("mary_had_lamb"), AudioAsset("winning_call")]


@pytest.fixture
def tokenizer() -> MistralTokenizer:
    return MistralTokenizer.from_hf_hub(MODEL_NAME)


@pytest.fixture
def engine(monkeypatch: pytest.MonkeyPatch):
    # Disable multiprocessing allows us to access model executor from LLM engine
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    engine_args = EngineArgs(**ENGINE_CONFIG)
    llm = LLM.from_engine_args(engine_args)
    try:
        yield llm
    finally:
        with contextlib.suppress(Exception):
            llm.llm_engine.engine_core.shutdown()
        # pytest pins the fixture value, so drop the engine ref for cleanup.
        del llm.llm_engine
        del llm
        cleanup_dist_env_and_memory()
        wait_for_rocm_memory_to_settle()


@pytest_asyncio.fixture
async def async_engine():
    engine_args = AsyncEngineArgs(**ENGINE_CONFIG)
    llm = AsyncLLM.from_engine_args(engine_args)
    try:
        yield llm
    finally:
        llm.shutdown()


def test_voxtral_realtime_forward(audio_assets, tokenizer, engine):
    assert_encoder_kv_cache_spec(engine)
    audio_config = tokenizer.instruct_tokenizer.tokenizer.audio

    def from_file(file_path: str):
        audio = Audio.from_file(file_path, strict=False)
        req = TranscriptionRequest(
            audio=audio.to_base64(audio.format),
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

    texts = _normalize([out.outputs[0].text for out in outputs])
    for i, (got, expected) in enumerate(zip(texts, EXPECTED_TEXT)):
        assert got == expected, (
            f"Output mismatch at index {i}:\n"
            f"  got:      {got!r}\n"
            f"  expected: {expected!r}"
        )


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
            audio=audio.to_base64(audio.format),
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

    texts = _normalize(
        [
            tokenizer.decode(
                output_tokens, special_token_policy=SpecialTokenPolicy.IGNORE
            )
            for output_tokens in output_tokens_list
        ]
    )
    for i, (got, expected) in enumerate(zip(texts, EXPECTED_TEXT)):
        assert got == expected, (
            f"Output mismatch at index {i}:\n"
            f"  got:      {got!r}\n"
            f"  expected: {expected!r}"
        )
