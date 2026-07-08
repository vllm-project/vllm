# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from typing import Any

import numpy as np
import pytest
import pytest_asyncio
from transformers import AutoTokenizer

from ....conftest import AUDIO_ASSETS, AudioTestAssets, VllmRunner
from ....utils import RemoteOpenAIServer
from ...registry import HF_EXAMPLE_MODELS

MODEL_NAME = "sarvamai/shuka-1"

AUDIO_PROMPTS = AUDIO_ASSETS.prompts(
    {
        "mary_had_lamb": "Transcribe this into English.",
        "winning_call": "What is happening in this audio clip?",
    }
)

AudioTuple = tuple[np.ndarray, int]

VLLM_PLACEHOLDER = "<|audio|>"


def params_kwargs_to_cli_args(params_kwargs: dict[str, Any]) -> list[str]:
    """Convert kwargs to CLI args."""
    args = []
    for key, value in params_kwargs.items():
        if isinstance(value, bool):
            if value:
                args.append(f"--{key.replace('_', '-')}")
        else:
            args.append(f"--{key.replace('_', '-')}={value}")
    return args


@pytest.fixture(params=[pytest.param({}, marks=pytest.mark.cpu_model)])
def server(request, audio_assets: AudioTestAssets):
    args = [
        "--dtype",
        "bfloat16",
        "--max-model-len",
        "4096",
        "--enforce-eager",
        "--limit-mm-per-prompt",
        json.dumps({"audio": len(audio_assets)}),
        "--trust-remote-code",
    ] + params_kwargs_to_cli_args(request.param)

    with RemoteOpenAIServer(
        MODEL_NAME, args, env_dict={"VLLM_AUDIO_FETCH_TIMEOUT": "30"}
    ) as remote_server:
        yield remote_server


@pytest_asyncio.fixture
async def client(server):
    async with server.get_async_client() as async_client:
        yield async_client


def _get_prompt(audio_count, question, placeholder):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    placeholder = f"{placeholder}\n" * audio_count

    return tokenizer.apply_chat_template(
        [{"role": "user", "content": f"{placeholder}{question}"}],
        tokenize=False,
        add_generation_prompt=True,
    )


def run_test(
    vllm_runner: type[VllmRunner],
    prompts_and_audios: list[tuple[str, list[AudioTuple]]],
    model: str,
    *,
    dtype: str,
    max_tokens: int,
    num_logprobs: int,
    **kwargs,
):
    model_info = HF_EXAMPLE_MODELS.find_hf_info(model)
    model_info.check_available_online(on_fail="skip")
    model_info.check_transformers_version(on_fail="skip")

    with vllm_runner(
        model,
        dtype=dtype,
        enforce_eager=True,
        limit_mm_per_prompt={
            "audio": max((len(audio) for _, audio in prompts_and_audios))
        },
        **kwargs,
    ) as vllm_model:
        vllm_outputs = vllm_model.generate_greedy_logprobs(
            [prompt for prompt, _ in prompts_and_audios],
            max_tokens,
            num_logprobs=num_logprobs,
            audios=[audios for _, audios in prompts_and_audios],
        )

    assert all(tokens for tokens, *_ in vllm_outputs)


@pytest.mark.core_model
@pytest.mark.parametrize("dtype", ["half"])
@pytest.mark.parametrize("max_tokens", [32])
@pytest.mark.parametrize("num_logprobs", [5])
def test_single_audio(
    vllm_runner,
    audio_assets: AudioTestAssets,
    dtype: str,
    max_tokens: int,
    num_logprobs: int,
) -> None:
    prompts_and_audios = []
    for audio, question in zip(audio_assets, AUDIO_PROMPTS):
        prompt = _get_prompt(1, question, VLLM_PLACEHOLDER)
        prompts_and_audios.append((prompt, [audio.audio_and_sample_rate]))

    run_test(
        vllm_runner,
        prompts_and_audios,
        MODEL_NAME,
        dtype=dtype,
        max_tokens=max_tokens,
        num_logprobs=num_logprobs,
    )


@pytest.mark.core_model
@pytest.mark.parametrize("dtype", ["half"])
@pytest.mark.parametrize("max_tokens", [32])
def test_variable_length_audio_batching(
    vllm_runner,
    audio_assets: AudioTestAssets,
    dtype: str,
    max_tokens: int,
) -> None:
    """Test batching of requests with different audio durations."""
    model_info = HF_EXAMPLE_MODELS.find_hf_info(MODEL_NAME)
    model_info.check_available_online(on_fail="skip")
    model_info.check_transformers_version(on_fail="skip")

    prompts_and_audios = []
    for audio, question in zip(audio_assets, AUDIO_PROMPTS):
        prompt = _get_prompt(1, question, VLLM_PLACEHOLDER)
        prompts_and_audios.append((prompt, [audio.audio_and_sample_rate]))

    with vllm_runner(
        MODEL_NAME,
        dtype=dtype,
        enforce_eager=True,
        limit_mm_per_prompt={"audio": 1},
    ) as vllm_model:
        outputs = vllm_model.generate_greedy(
            [prompt for prompt, _ in prompts_and_audios],
            max_tokens,
            audios=[audios for _, audios in prompts_and_audios],
        )

    assert len(outputs) == len(prompts_and_audios)
    for output in outputs:
        assert len(output[1]) > 0, "Expected non-empty output"


@pytest.mark.asyncio
async def test_online_serving(client, audio_assets: AudioTestAssets):
    """Exercises online serving via the OpenAI-compatible chat endpoint."""
    audio = audio_assets[0]

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "audio_url", "audio_url": {"url": audio.url}},
                {"type": "text", "text": "What's happening in this audio clip?"},
            ],
        }
    ]

    chat_completion = await client.chat.completions.create(
        model=MODEL_NAME, messages=messages, max_tokens=10
    )

    assert len(chat_completion.choices) == 1
    choice = chat_completion.choices[0]
    assert choice.finish_reason == "length"
