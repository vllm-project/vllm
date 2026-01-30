# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Sequence
from typing import Any

import librosa
import pytest
from transformers import AutoModelForSpeechSeq2Seq

from vllm.assets.audio import AudioAsset
from vllm.platforms import current_platform

from ....conftest import HfRunner, PromptAudioInput, VllmRunner
from ....utils import create_new_process_for_each_test, multi_gpu_test
from ...registry import HF_EXAMPLE_MODELS
from ...utils import check_logprobs_close

VLLM_PROMPT = "<|startoftranscript|><|en|><|transcribe|><|notimestamps|>"
HF_PROMPT = ""
# Whisper expects 16kHz audio
WHISPER_SAMPLE_RATE = 16000


@pytest.fixture(autouse=True)
def use_spawn_for_whisper(monkeypatch):
    """Whisper has issues with forked workers, use spawn instead."""
    monkeypatch.setenv("VLLM_WORKER_MULTIPROC_METHOD", "spawn")


def run_test(
    hf_runner: type[HfRunner],
    vllm_runner: type[VllmRunner],
    inputs: Sequence[tuple[list[str], list[str], PromptAudioInput]],
    model: str,
    *,
    max_model_len: int,
    dtype: str,
    max_tokens: int,
    num_logprobs: int,
    tensor_parallel_size: int,
    distributed_executor_backend: str | None = None,
    enforce_eager: bool = True,
) -> None:
    """Inference result should be the same between hf and vllm.

    All the audio fixtures for the test are from AudioAsset.
    For huggingface runner, we provide the audio as input.
    For vllm runner, we provide MultiModalDataDict objects
    and corresponding MultiModalConfig as input.
    """
    with vllm_runner(
        model,
        dtype=dtype,
        max_model_len=max_model_len,
        tensor_parallel_size=tensor_parallel_size,
        distributed_executor_backend=distributed_executor_backend,
        limit_mm_per_prompt={"audio": 2},
        enforce_eager=enforce_eager,
        disable_custom_all_reduce=True,
    ) as vllm_model:
        vllm_outputs_per_case = [
            vllm_model.generate_greedy_logprobs(
                vllm_prompts,
                max_tokens,
                num_logprobs=num_logprobs,
                audios=audios,
            )
            for vllm_prompts, _, audios in inputs
        ]

    with hf_runner(model, dtype=dtype, auto_cls=AutoModelForSpeechSeq2Seq) as hf_model:
        hf_outputs_per_case = [
            hf_model.generate_greedy_logprobs_limit(
                hf_prompts,
                max_tokens,
                num_logprobs=num_logprobs,
                audios=audios,
            )
            for _, hf_prompts, audios in inputs
        ]

    for hf_outputs, vllm_outputs in zip(hf_outputs_per_case, vllm_outputs_per_case):
        check_logprobs_close(
            outputs_0_lst=hf_outputs,
            outputs_1_lst=vllm_outputs,
            name_0="hf",
            name_1="vllm",
        )


@pytest.fixture
def input_audios() -> list[tuple[list[str], list[str], list[tuple[Any, int]]]]:
    audio_assets = [AudioAsset("mary_had_lamb"), AudioAsset("winning_call")]
    inputs = []
    for asset in audio_assets:
        audio, orig_sr = asset.audio_and_sample_rate
        # Resample to Whisper's expected sample rate (16kHz)
        if orig_sr != WHISPER_SAMPLE_RATE:
            audio = librosa.resample(
                audio, orig_sr=orig_sr, target_sr=WHISPER_SAMPLE_RATE
            )
        # vLLM prompts, HF prompts, audio inputs
        inputs.append(([VLLM_PROMPT], [HF_PROMPT], [(audio, WHISPER_SAMPLE_RATE)]))
    return inputs


def check_model_available(model: str) -> None:
    model_info = HF_EXAMPLE_MODELS.find_hf_info(model)
    model_info.check_available_online(on_fail="skip")
    model_info.check_transformers_version(on_fail="skip")


@pytest.mark.core_model
@pytest.mark.cpu_model
@pytest.mark.parametrize("model", ["openai/whisper-large-v3-turbo"])
@pytest.mark.parametrize("dtype", ["half", "float"])
@pytest.mark.parametrize("num_logprobs", [5])
@pytest.mark.parametrize("enforce_eager", [True, False])
@create_new_process_for_each_test("spawn")
def test_models(
    hf_runner,
    vllm_runner,
    model: str,
    dtype: str,
    num_logprobs: int,
    input_audios,
    enforce_eager: bool,
) -> None:
    check_model_available(model)
    if current_platform.is_cpu() and not enforce_eager:
        pytest.skip("Skipping test for CPU with non-eager mode")
    run_test(
        hf_runner,
        vllm_runner,
        input_audios,
        model,
        dtype=dtype,
        max_model_len=448,
        max_tokens=200,
        num_logprobs=num_logprobs,
        tensor_parallel_size=1,
        enforce_eager=enforce_eager,
    )


@multi_gpu_test(num_gpus=2)
@pytest.mark.core_model
@pytest.mark.parametrize("model", ["openai/whisper-large-v3-turbo"])
@pytest.mark.parametrize("distributed_executor_backend", ["ray", "mp"])
@pytest.mark.parametrize("dtype", ["half"])
@pytest.mark.parametrize("max_tokens", [200])
@pytest.mark.parametrize("num_logprobs", [5])
@create_new_process_for_each_test("spawn")
def test_models_distributed(
    hf_runner,
    vllm_runner,
    model: str,
    distributed_executor_backend: str,
    dtype: str,
    max_tokens: int,
    num_logprobs: int,
    input_audios,
) -> None:
    check_model_available(model)
    run_test(
        hf_runner,
        vllm_runner,
        input_audios,
        model,
        dtype=dtype,
        max_model_len=448,
        max_tokens=max_tokens,
        num_logprobs=num_logprobs,
        tensor_parallel_size=2,
        distributed_executor_backend=distributed_executor_backend,
        enforce_eager=False,
    )


@pytest.mark.core_model
@pytest.mark.parametrize("model", ["openai/whisper-large-v3-turbo"])
def test_encoder_cache_cleanup(
    vllm_runner,
    model: str,
    input_audios,
    monkeypatch,
) -> None:
    """Test that encoder cache is properly cleaned up after requests complete.

    This is a regression test for a bug where encoder cache entries were freed
    in the same scheduling step they were allocated, before the model could use
    them.
    """
    # Set single-process mode to access the model runner's encoder cache directly
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    check_model_available(model)

    with vllm_runner(
        model,
        dtype="half",
        max_model_len=448,
        tensor_parallel_size=1,
        limit_mm_per_prompt={"audio": 2},
        enforce_eager=True,
    ) as vllm_model:
        engine_core = vllm_model.llm.llm_engine.engine_core.engine_core
        model_runner = engine_core.model_executor.driver_worker.worker.model_runner
        encoder_cache = model_runner.encoder_cache

        # Run multiple sequential requests to ensure cache is properly managed
        for vllm_prompts, _, audios in input_audios:
            vllm_model.generate_greedy(vllm_prompts, max_tokens=50, audios=audios)

        # After all requests complete, encoder cache should be empty
        cache_size = len(encoder_cache)
        assert cache_size == 0, (
            f"Encoder cache should be empty after all requests complete, "
            f"but has {cache_size} entries. This indicates encoder cache "
            f"entries are not being properly freed."
        )
