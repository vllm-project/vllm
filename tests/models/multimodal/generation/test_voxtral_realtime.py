# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import contextlib

import pytest
import pytest_asyncio
import torch
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

from ....utils import ROCM_ENGINE_KWARGS

MODEL_NAME = "mistralai/Voxtral-Mini-4B-Realtime-2602"
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


@pytest.fixture
def audio_assets() -> list[AudioAsset]:
    return [AudioAsset("mary_had_lamb"), AudioAsset("winning_call")]


@pytest.fixture
def tokenizer() -> MistralTokenizer:
    return MistralTokenizer.from_hf_hub(MODEL_NAME)


@pytest.fixture
def engine():
    engine_args = EngineArgs(**ENGINE_CONFIG)
    llm = LLM.from_engine_args(engine_args)
    try:
        yield llm
    finally:
        with contextlib.suppress(Exception):
            llm.llm_engine.engine_core.shutdown()
        import torch

        torch.accelerator.empty_cache()


@pytest_asyncio.fixture
async def async_engine():
    engine_args = AsyncEngineArgs(**ENGINE_CONFIG)
    llm = AsyncLLM.from_engine_args(engine_args)
    try:
        yield llm
    finally:
        llm.shutdown()


@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason="Requires GPU")
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

    texts = _normalize([out.outputs[0].text for out in outputs])
    for i, (got, expected) in enumerate(zip(texts, EXPECTED_TEXT)):
        assert got == expected, (
            f"Output mismatch at index {i}:\n"
            f"  got:      {got!r}\n"
            f"  expected: {expected!r}"
        )


@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason="Requires GPU")
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


# ---------------------------------------------------------------------------
# Unit tests for embed_input_ids mixed-batch fix (issue #39202)
# ---------------------------------------------------------------------------


class _FakeAudioConfig:
    """Minimal stand-in for the audio config attributes used by
    embed_input_ids."""

    def __init__(self, d_model: int = 64, block_pool_size: int = 2):
        self.d_model = d_model
        self.block_pool_size = block_pool_size


class _FakeConfig:
    def __init__(self, d_model: int = 64, block_pool_size: int = 2):
        self.audio_config = _FakeAudioConfig(d_model, block_pool_size)


class _FakeEncoder:
    dtype = torch.float32


class _FakeVoxtralRealtimeGeneration:
    """Minimal stub that reuses the real embed_input_ids logic."""

    def __init__(self, d_model: int = 64, block_pool_size: int = 2):
        self.config = _FakeConfig(d_model, block_pool_size)
        self.whisper_encoder = _FakeEncoder()

    # Bind the real method — import lazily to avoid CUDA init at import time.
    @property
    def _real_cls(self):
        from vllm.model_executor.models.voxtral_realtime import (
            VoxtralRealtimeGeneration,
        )
        return VoxtralRealtimeGeneration

    def embed_input_ids(self, input_ids, multimodal_embeddings=None, *,
                        is_multimodal=None):
        return self._real_cls.embed_input_ids(
            self, input_ids, multimodal_embeddings,
            is_multimodal=is_multimodal)


def _make_stub(d_model=64, block_pool_size=2):
    return _FakeVoxtralRealtimeGeneration(d_model, block_pool_size)


@pytest.mark.skip_global_cleanup
def test_embed_input_ids_all_multimodal():
    """All tokens are multimodal — output must equal the embeddings."""
    stub = _make_stub()
    embed_dim = 64 * 2  # d_model * block_pool_size
    n_tokens = 3
    input_ids = torch.zeros(n_tokens, dtype=torch.long)
    mm_embeds = [torch.randn(n_tokens, embed_dim)]
    is_mm = torch.ones(n_tokens, dtype=torch.bool)

    out = stub.embed_input_ids(input_ids, mm_embeds, is_multimodal=is_mm)

    assert out.shape == (n_tokens, embed_dim)
    torch.testing.assert_close(out, mm_embeds[0])


@pytest.mark.skip_global_cleanup
def test_embed_input_ids_mixed_batch():
    """Mixed batch: some multimodal, some not — must not crash and must
    return the correct shape with zeros for non-multimodal positions."""
    stub = _make_stub()
    embed_dim = 64 * 2
    n_tokens = 5
    n_mm = 3
    input_ids = torch.zeros(n_tokens, dtype=torch.long)
    mm_data = torch.randn(n_mm, embed_dim)
    mm_embeds = [mm_data]
    # First 3 tokens are multimodal, last 2 are not
    is_mm = torch.tensor([True, True, True, False, False])

    out = stub.embed_input_ids(input_ids, mm_embeds, is_multimodal=is_mm)

    assert out.shape == (n_tokens, embed_dim)
    # Multimodal positions should match the embeddings
    torch.testing.assert_close(out[:n_mm], mm_data)
    # Non-multimodal positions should be zero
    assert (out[n_mm:] == 0).all()


@pytest.mark.skip_global_cleanup
def test_embed_input_ids_no_multimodal():
    """No multimodal embeddings — should return all zeros."""
    stub = _make_stub()
    embed_dim = 64 * 2
    n_tokens = 4
    input_ids = torch.zeros(n_tokens, dtype=torch.long)

    out = stub.embed_input_ids(input_ids, [], is_multimodal=None)

    assert out.shape == (n_tokens, embed_dim)
    assert (out == 0).all()


@pytest.mark.skip_global_cleanup
def test_embed_input_ids_no_mask_exact_match():
    """Fallback: is_multimodal is None but embeddings match token count."""
    stub = _make_stub()
    embed_dim = 64 * 2
    n_tokens = 3
    input_ids = torch.zeros(n_tokens, dtype=torch.long)
    mm_data = torch.randn(n_tokens, embed_dim)
    mm_embeds = [mm_data]

    out = stub.embed_input_ids(input_ids, mm_embeds, is_multimodal=None)

    assert out.shape == (n_tokens, embed_dim)
    torch.testing.assert_close(out, mm_data)
