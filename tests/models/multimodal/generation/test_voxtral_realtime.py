# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from fractions import Fraction
from typing import TYPE_CHECKING

import pytest
import pytest_asyncio
import torch
from mistral_common.protocol.transcription.request import (
    StreamingMode,
    TranscriptionRequest,
)
from mistral_common.tokens.tokenizers.audio import Audio
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.tokens.tokenizers.tekken import SpecialTokenPolicy

from vllm import LLM, SamplingParams
from vllm.assets.audio import AudioAsset
from vllm.config import CUDAGraphMode
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.protocol import StreamingInput
from vllm.platforms import current_platform
from vllm.renderers.inputs.preprocess import parse_model_prompt
from vllm.utils.math_utils import cdiv
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.kv_cache_interface import SlidingWindowSpec

from ....utils import ROCM_ENGINE_KWARGS, create_new_process_for_each_test

if TYPE_CHECKING:
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner

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
    assert spec.num_kv_heads == 32
    assert spec.tokens_per_state == Fraction(1, 4)
    # cdiv(750, 4) == 188 pooled tokens cover the model's window; the extra
    # +1 is an eviction margin (see whisper_causal.py get_kv_cache_spec).
    assert spec.sliding_window == cdiv(750, 4) + 1 == 189
    assert (
        spec.max_admission_blocks_per_request(
            max_in_flight_tokens=1,
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
async def async_engine(request):
    engine_config = {**ENGINE_CONFIG, **getattr(request, "param", {})}
    gpu_memory_utilization = ENGINE_CONFIG.get("gpu_memory_utilization", 0.9)
    from vllm.platforms import current_platform

    if current_platform.is_rocm():
        from tests.utils import wait_for_memory_to_settle

        wait_for_memory_to_settle(threshold_ratio=1.0 - gpu_memory_utilization)

    engine_args = AsyncEngineArgs(**engine_config)
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
        from tests.utils import wait_for_memory_to_settle

        wait_for_memory_to_settle(threshold_ratio=1.0 - gpu_memory_utilization)


def _build_inputs(
    audio_assets: list[AudioAsset], tokenizer: MistralTokenizer
) -> tuple[list[dict], list[SamplingParams]]:
    audio_config = tokenizer.instruct_tokenizer.tokenizer.audio
    inputs = []
    sampling_params = []
    for audio_asset in audio_assets:
        audio = Audio.from_file(audio_asset.get_local_path(), strict=False)
        req = TranscriptionRequest(
            audio=audio.to_base64(audio.format),
            streaming=StreamingMode.OFFLINE,
            language=None,
        )
        tokenized = tokenizer.instruct_tokenizer.encode_transcription(req)
        tokens = tokenized.tokens
        audio_array = tokenized.audios[0].audio_array

        num_samples = audio_array.shape[0]
        max_tokens = audio_config.num_audio_tokens(num_samples) - len(tokens) - 1
        sampling_params.append(SamplingParams(temperature=0.0, max_tokens=max_tokens))
        inputs.append(
            {
                "multi_modal_data": {"audio": [(audio_array, None)]},
                "prompt_token_ids": tokens,
            }
        )
    return inputs, sampling_params


def _assert_expected_text(outputs) -> None:
    texts = _normalize([out.outputs[0].text for out in outputs])
    for i, (got, expected) in enumerate(zip(texts, EXPECTED_TEXT)):
        assert got == expected, (
            f"Output mismatch at index {i}:\n"
            f"  got:      {got!r}\n"
            f"  expected: {expected!r}"
        )


def test_voxtral_realtime_forward(audio_assets, tokenizer, vllm_runner, monkeypatch):
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

    vllm_kwargs = {**ENGINE_CONFIG}
    vllm_kwargs["model_name"] = vllm_kwargs.pop("model")

    with vllm_runner(**vllm_kwargs) as vllm_model:
        assert_encoder_kv_cache_spec(vllm_model.llm)
        inputs, sampling_params = _build_inputs(audio_assets, tokenizer)

        outputs = vllm_model.llm.generate(inputs, sampling_params=sampling_params)
        _assert_expected_text(outputs)


@pytest.mark.skipif(
    not current_platform.is_cuda(), reason="Full cudagraphs require CUDA"
)
@pytest.mark.parametrize(
    "cudagraph_mode",
    [CUDAGraphMode.FULL_DECODE_ONLY, CUDAGraphMode.FULL_AND_PIECEWISE],
)
@pytest.mark.parametrize(
    "frontend_graphs", [False, pytest.param(True, marks=pytest.mark.core_model)]
)
@create_new_process_for_each_test(method="spawn")
def test_voxtral_realtime_cudagraph(
    audio_assets, tokenizer, vllm_runner, monkeypatch, cudagraph_mode, frontend_graphs
):
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

    vllm_kwargs = {**ENGINE_CONFIG}
    vllm_kwargs["model_name"] = vllm_kwargs.pop("model")
    vllm_kwargs["enforce_eager"] = False
    vllm_kwargs["compilation_config"] = {
        "cudagraph_mode": cudagraph_mode,
        "cudagraph_mm_encoder": frontend_graphs,
        # max_num_seqs is 4, so this is the only decode size worth capturing.
        "cudagraph_capture_sizes": [4],
    }
    if frontend_graphs and cudagraph_mode == CUDAGraphMode.FULL_AND_PIECEWISE:
        # The manager keeps its model minimum when an automatic cap is smaller.
        vllm_kwargs["compilation_config"]["encoder_cudagraph_max_token_budget"] = 1

    with vllm_runner(**vllm_kwargs) as vllm_model:
        if frontend_graphs:
            vllm_model.llm.collective_rpc(_check_frontend_graph_replay)
        inputs, sampling_params = _build_inputs(audio_assets, tokenizer)

        outputs = vllm_model.llm.generate(inputs, sampling_params=sampling_params)
        _assert_expected_text(outputs)


@torch.inference_mode()
def _check_frontend_graph_replay(worker):
    """Mixed lengths and later replays must preserve each item's embeddings."""
    manager = worker.model_runner.model_state.encoder_runner.cudagraph_manager
    assert manager is not None and manager.is_captured()
    model = manager.model
    axes = model.get_encoder_cudagraph_config().capture_axes
    short, startup = axes[1]
    assert manager.get_cumulative_stats()["num_budgets"] == len(axes[0]) * len(axes[1])
    torch.manual_seed(0)
    for dtype in (torch.float32, torch.bfloat16):
        # Include uncaptured lengths with the same output token counts.
        lengths = (short, startup, short + 160, startup + 160)
        audios = [
            (torch.randn(lengths[i % 4], device="cuda") * (0.01 + i * 0.005)).to(dtype)
            for i in range(manager.max_batch_size * 4 + 1)
        ]
        expected = model.embed_multimodal(audio_arrays=audios)
        before = manager.get_cumulative_stats()
        actual = manager.execute({"audio_arrays": audios})
        after = manager.get_cumulative_stats()
        assert after["graph_hits"] > before["graph_hits"]
        assert after["graph_misses"] > before["graph_misses"]
        assert len(actual) == len(expected)
        for i, (output, reference) in enumerate(zip(actual, expected)):
            assert output.shape == reference.shape
            assert torch.isfinite(output).all()
            if i % 4 >= 2:
                assert torch.equal(output, reference)
            else:
                relative_error = (
                    output.float() - reference.float()
                ).norm() / reference.float().norm()
                assert relative_error <= 1e-3
        saved = [output.clone() for output in actual]
        manager.execute({"audio_arrays": list(reversed(audios))})
        assert manager.get_cumulative_stats()["num_budgets"] == before["num_budgets"]
        for output, snapshot in zip(actual, saved):
            assert torch.equal(output, snapshot)


@pytest.mark.parametrize(
    ("async_engine", "frontend_graphs"),
    [
        pytest.param({}, False, id="eager"),
        pytest.param(
            {
                "enforce_eager": False,
                "worker_extension_cls": f"{__name__}.FrontendGraphTestWorker",
                "compilation_config": {
                    "cudagraph_mode": CUDAGraphMode.FULL_AND_PIECEWISE,
                    "cudagraph_mm_encoder": True,
                    "cudagraph_capture_sizes": [1, 2, 4],
                },
            },
            True,
            id="frontend_graphs",
            marks=[
                pytest.mark.core_model,
                pytest.mark.skipif(
                    not current_platform.is_cuda(),
                    reason="Frontend graphs require CUDA",
                ),
            ],
        ),
    ],
    indirect=["async_engine"],
)
@pytest.mark.asyncio
async def test_voxtral_realtime_generator(
    audio_assets, tokenizer, async_engine, frontend_graphs
):
    # Lazy import to avoid CUDA-reinitialization error
    from vllm.model_executor.models.voxtral_realtime import VoxtralRealtimeBuffer

    sampling_params = SamplingParams(temperature=0.0, max_tokens=1)
    audio_config = tokenizer.instruct_tokenizer.audio_encoder.audio_config

    async def input_stream(audio_buffer):
        async for prompt in audio_buffer.get_input_stream():
            parsed_prompt = parse_model_prompt(async_engine.model_config, prompt)
            (engine_input,) = await async_engine.renderer.render_cmpl_async(
                [parsed_prompt]
            )
            yield StreamingInput(prompt=engine_input)

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
            prompt=input_stream(buffer),
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
    if frontend_graphs:
        stats = await async_engine.collective_rpc("frontend_graph_stats")
        assert all(rank is not None and rank["graph_hits"] > 0 for rank in stats)


class FrontendGraphTestWorker:
    model_runner: "GPUModelRunner"

    def frontend_graph_stats(self):
        manager = self.model_runner.model_state.encoder_runner.cudagraph_manager
        return None if manager is None else manager.get_cumulative_stats()


@pytest.mark.asyncio
async def test_voxtral_realtime_buffer_stops_at_max_model_len(tokenizer):
    from vllm.model_executor.models.voxtral_realtime import VoxtralRealtimeBuffer

    audio_config = tokenizer.instruct_tokenizer.audio_encoder.audio_config
    audio = Audio.from_file(AudioAsset("winning_call").get_local_path(), strict=False)
    req = TranscriptionRequest(
        streaming=StreamingMode.OFFLINE,
        audio=audio.to_base64(audio.format),
        language=None,
    )
    audio_enc = tokenizer.encode_transcription(req)

    max_model_len = 64
    buffer = VoxtralRealtimeBuffer(
        audio_config, audio_enc.tokens, max_model_len=max_model_len
    )
    await buffer.append_audio(audio_enc.audios[0].audio_array)
    await buffer.append_audio(None)
    # Unlimited feedback tokens so only the length cap can stop the stream.
    await buffer.append_tokens([0] * max_model_len)

    n_tokens = 0
    async for prompt in buffer.get_input_stream():
        n_tokens += len(prompt["prompt_token_ids"])

    assert len(audio_enc.tokens) < n_tokens < max_model_len


@pytest.mark.asyncio
@pytest.mark.parametrize("async_engine", [{"max_model_len": 128}], indirect=True)
async def test_voxtral_realtime_generator_max_model_len(
    audio_assets, tokenizer, async_engine
):
    # Lazy import to avoid CUDA-reinitialization error
    from vllm.model_executor.models.voxtral_realtime import VoxtralRealtimeBuffer

    max_model_len = async_engine.model_config.max_model_len
    sampling_params = SamplingParams(temperature=0.0, max_tokens=1)
    audio_config = tokenizer.instruct_tokenizer.audio_encoder.audio_config

    async def input_stream(audio_buffer):
        async for prompt in audio_buffer.get_input_stream():
            parsed_prompt = parse_model_prompt(async_engine.model_config, prompt)
            (engine_input,) = await async_engine.renderer.render_cmpl_async(
                [parsed_prompt]
            )
            yield StreamingInput(prompt=engine_input)

    # winning_call is ~300 tokens long, well past max_model_len.
    audio = Audio.from_file(audio_assets[1].get_local_path(), strict=False)
    req = TranscriptionRequest(
        streaming=StreamingMode.OFFLINE,
        audio=audio.to_base64(audio.format),
        language=None,
    )
    audio_enc = tokenizer.encode_transcription(req)
    buffer = VoxtralRealtimeBuffer(
        audio_config, audio_enc.tokens, max_model_len=max_model_len
    )
    await buffer.append_audio(audio_enc.audios[0].audio_array)
    await buffer.append_audio(None)

    output_tokens = []
    async for resp in async_engine.generate(
        prompt=input_stream(buffer),
        sampling_params=sampling_params,
        request_id="session-max-len",
    ):
        tokens = resp.outputs[0].token_ids[-1:]
        output_tokens.extend(tokens)
        await buffer.append_tokens(tokens)

    assert resp.finished
    assert 0 < len(output_tokens) < max_model_len
    # The engine is still healthy after the session hit the cap.
    assert not async_engine.errored
