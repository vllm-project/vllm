# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import pytest_asyncio
from mistral_common.protocol.transcription.request import (
    StreamingMode,
    TranscriptionRequest,
)
from mistral_common.tokens.tokenizers.audio import Audio
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.tokens.tokenizers.tekken import SpecialTokenPolicy

from vllm import LLM, SamplingParams
from vllm.assets.audio import AudioAsset
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.utils.math_utils import cdiv
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.kv_cache_interface import SlidingWindowSpec

from ....utils import ROCM_ENGINE_KWARGS

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


@pytest_asyncio.fixture
async def async_engine():
    gpu_memory_utilization = ENGINE_CONFIG.get("gpu_memory_utilization", 0.9)
    from vllm.platforms import current_platform

    if current_platform.is_rocm():
        from tests.utils import wait_for_rocm_memory_to_settle

        wait_for_rocm_memory_to_settle(threshold_ratio=1.0 - gpu_memory_utilization)

    engine_args = AsyncEngineArgs(**ENGINE_CONFIG)
    llm = AsyncLLM.from_engine_args(engine_args)
    try:
        yield llm
    finally:
        shutdown_timeout = 60.0 if current_platform.is_rocm() else None
        llm.shutdown(timeout=shutdown_timeout)
        del llm
        import torch

        torch._dynamo.reset()
        from vllm.distributed import cleanup_dist_env_and_memory

        cleanup_dist_env_and_memory()
        from tests.utils import wait_for_rocm_memory_to_settle

        wait_for_rocm_memory_to_settle(threshold_ratio=1.0 - gpu_memory_utilization)


def test_voxtral_realtime_forward(audio_assets, tokenizer, vllm_runner, monkeypatch):
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

    vllm_kwargs = {**ENGINE_CONFIG}
    vllm_kwargs["model_name"] = vllm_kwargs.pop("model")

    with vllm_runner(**vllm_kwargs) as vllm_model:
        assert_encoder_kv_cache_spec(vllm_model.llm)
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
            sampling_params.append(
                SamplingParams(temperature=0.0, max_tokens=max_tokens)
            )

            input_dict = {
                "multi_modal_data": {"audio": [(audio_array, None)]},
                "prompt_token_ids": tokens,
            }
            inputs.append(input_dict)

        outputs = vllm_model.llm.generate(
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


# [EXPERIMENTAL] Narrow, equal text/audio sliding windows so a short clip's
# growing position clock crosses the re-anchor threshold during the session.
# Equal windows also exercise the merged-KV-group path (one mixed 128-NeoX /
# 64-GPT-J group), where the worker reads head_size per layer.
_REANCHOR_WINDOW = {
    "text_config": {"sliding_window": 256},
    "audio_config": {"sliding_window": 256},
}


async def _stream_transcribe(engine_args, tokenizer, audio_asset) -> list[int]:
    """Drive one realtime streaming session to completion, greedily, returning
    the decoded output token ids."""
    import torch

    from vllm.model_executor.models.voxtral_realtime import VoxtralRealtimeBuffer

    llm = AsyncLLM.from_engine_args(engine_args)
    try:
        audio_config = tokenizer.instruct_tokenizer.audio_encoder.audio_config
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

        output_tokens: list[int] = []
        async for resp in llm.generate(
            prompt=buffer.get_input_stream(),
            sampling_params=SamplingParams(temperature=0.0, max_tokens=1),
            request_id="reanchor-parity",
        ):
            tokens = resp.outputs[0].token_ids[-1:]
            output_tokens.extend(tokens)
            await buffer.append_tokens(tokens)
        return output_tokens
    finally:
        llm.shutdown()
        # Free this engine's KV pool before the next engine inits (single GPU).
        torch.accelerator.empty_cache()


_REANCHOR_TEST_MML = 300


@pytest.mark.asyncio
async def test_voxtral_realtime_reanchor_parity(audio_assets, tokenizer):
    """[EXPERIMENTAL] End-to-end parity of RoPE re-anchoring (review of #45022).

    A streaming session whose position clock crosses ``max_model_len - margin``
    is re-anchored down by R(-D) on the live cached keys. This proves the
    re-rotation is transparent end to end (real attention forward, real KV
    layout), not just algebraically (tests/v1/worker/test_reanchor_rotary.py):
    at an identical narrow window, a re-anchor-ON session must decode the SAME
    tokens as a re-anchor-OFF reference. Both runs share the window, so any
    divergence is the re-anchor's doing, not the window's.

    Firing is proven WITHOUT capturing the subprocess scheduler log (it runs in a
    child EngineCore process that neither caplog nor capfd sees reliably): a
    session in a ``max_model_len``-position context cannot emit MORE than
    ``max_model_len`` output tokens unless its RoPE clock was re-anchored down to
    make room, so the reference out-growing the test's cap is itself proof the
    unbounded path engaged. Too-short a clip fails loudly rather than passing
    vacuously.

    GPU-only (the re-anchor path is CUDA-gated in VllmConfig). Validated on an
    RTX 4090 (16 GiB): winning_call at mml=300 / margin=24 / window=256 fires
    ~6-8 re-anchors and matches the reference token-for-token.
    """
    audio_asset = audio_assets[1]  # winning_call (longer clip)

    # Reference: re-anchor OFF, ample room to run the full clip un-re-anchored.
    ref_args = AsyncEngineArgs(
        **{**ENGINE_CONFIG, "max_model_len": 2048, "hf_overrides": _REANCHOR_WINDOW}
    )
    ref_tokens = await _stream_transcribe(ref_args, tokenizer, audio_asset)

    # Test: re-anchor ON with max_model_len well below the session's clock, so the
    # threshold (mml - margin) is crossed repeatedly mid-decode and re-anchors.
    test_args = AsyncEngineArgs(
        **{
            **ENGINE_CONFIG,
            "max_model_len": _REANCHOR_TEST_MML,
            "enable_realtime_unbounded": True,
            "realtime_reanchor_margin_tokens": 24,
            "enable_prefix_caching": False,
            "hf_overrides": _REANCHOR_WINDOW,
        }
    )
    test_tokens = await _stream_transcribe(test_args, tokenizer, audio_asset)

    # Anti-vacuous: the session must out-grow the test's context window, else
    # re-anchor was never needed. Emitting more output tokens than max_model_len
    # positions is only possible once the RoPE clock has been folded down.
    assert len(ref_tokens) > _REANCHOR_TEST_MML, (
        f"clip too short to exercise re-anchor: {len(ref_tokens)} decoded tokens "
        f"<= max_model_len {_REANCHOR_TEST_MML}; use a longer clip or lower mml"
    )

    # ...and the re-anchored decode must match the reference token-for-token.
    def _decode(toks: list[int]) -> str:
        return tokenizer.decode(toks, special_token_policy=SpecialTokenPolicy.IGNORE)

    assert test_tokens == ref_tokens, (
        "re-anchored output diverged from the reference at the same window:\n"
        f"  ref  ({len(ref_tokens)} toks): {_decode(ref_tokens)!r}\n"
        f"  test ({len(test_tokens)} toks): {_decode(test_tokens)!r}"
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
