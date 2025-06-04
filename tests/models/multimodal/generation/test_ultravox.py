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

MODEL_NAME = "fixie-ai/ultravox-v0_5-llama-3_2-1b"

AUDIO_PROMPTS = AUDIO_ASSETS.prompts({
    "mary_had_lamb":
    "Transcribe this into English.",
    "winning_call":
    "What is happening in this audio clip?",
})

MULTI_AUDIO_PROMPT = "Describe each of the audios above."

AudioTuple = tuple[np.ndarray, int]

VLLM_PLACEHOLDER = "<|audio|>"
HF_PLACEHOLDER = "<|audio|>"

CHUNKED_PREFILL_KWARGS = {
    "enable_chunked_prefill": True,
    "max_num_seqs": 2,
    # Use a very small limit to exercise chunked prefill.
    "max_num_batched_tokens": 16
}


def params_kwargs_to_cli_args(params_kwargs: dict[str, Any]) -> list[str]:
    """Convert kwargs to CLI args."""
    args = []
    for key, value in params_kwargs.items():
        if isinstance(value, bool):
            if value:
                args.append(f"--{key.replace('_','-')}")
        else:
            args.append(f"--{key.replace('_','-')}={value}")
    return args


@pytest.fixture(params=[
    pytest.param({}, marks=pytest.mark.cpu_model),
    pytest.param(CHUNKED_PREFILL_KWARGS),
])
def server(request, audio_assets: AudioTestAssets):
    args = [
        "--dtype", "bfloat16", "--max-model-len", "4096", "--enforce-eager",
        "--limit-mm-per-prompt",
        json.dumps({"audio": len(audio_assets)}), "--trust-remote-code"
    ] + params_kwargs_to_cli_args(request.param)

    with RemoteOpenAIServer(MODEL_NAME,
                            args,
                            env_dict={"VLLM_AUDIO_FETCH_TIMEOUT":
                                      "30"}) as remote_server:
        yield remote_server


@pytest_asyncio.fixture
async def client(server):
    async with server.get_async_client() as async_client:
        yield async_client


def _get_prompt(audio_count, question, placeholder):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    placeholder = f"{placeholder}\n" * audio_count

    return tokenizer.apply_chat_template([{
        'role': 'user',
        'content': f"{placeholder}{question}"
    }],
                                         tokenize=False,
                                         add_generation_prompt=True)


def run_multi_audio_test(
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

    with vllm_runner(model,
                     dtype=dtype,
                     enforce_eager=True,
                     limit_mm_per_prompt={
                         "audio":
                         max((len(audio) for _, audio in prompts_and_audios))
                     },
                     **kwargs) as vllm_model:
        vllm_outputs = vllm_model.generate_greedy_logprobs(
            [prompt for prompt, _ in prompts_and_audios],
            max_tokens,
            num_logprobs=num_logprobs,
            audios=[audios for _, audios in prompts_and_audios])

    # The HuggingFace model doesn't support multiple audios yet, so
    # just assert that some tokens were generated.
    assert all(tokens for tokens, *_ in vllm_outputs)


@pytest.mark.core_model
@pytest.mark.parametrize("dtype", ["half"])
@pytest.mark.parametrize("max_tokens", [128])
@pytest.mark.parametrize("num_logprobs", [5])
@pytest.mark.parametrize("vllm_kwargs", [
    pytest.param({}, marks=pytest.mark.cpu_model),
    pytest.param(CHUNKED_PREFILL_KWARGS),
])
def test_models_with_multiple_audios(vllm_runner,
                                     audio_assets: AudioTestAssets, dtype: str,
                                     max_tokens: int, num_logprobs: int,
                                     vllm_kwargs: dict) -> None:

    vllm_prompt = _get_prompt(len(audio_assets), MULTI_AUDIO_PROMPT,
                              VLLM_PLACEHOLDER)
    run_multi_audio_test(
        vllm_runner,
        [(vllm_prompt, [audio.audio_and_sample_rate
                        for audio in audio_assets])],
        MODEL_NAME,
        dtype=dtype,
        max_tokens=max_tokens,
        num_logprobs=num_logprobs,
        **vllm_kwargs,
    )


@pytest.mark.asyncio
async def test_online_serving(client, audio_assets: AudioTestAssets):
    """Exercises online serving with/without chunked prefill enabled."""

    messages = [{
        "role":
        "user",
        "content": [
            *[{
                "type": "audio_url",
                "audio_url": {
                    "url": audio.url
                }
            } for audio in audio_assets],
            {
                "type":
                "text",
                "text":
                f"What's happening in these {len(audio_assets)} audio clips?"
            },
        ],
    }]

    chat_completion = await client.chat.completions.create(model=MODEL_NAME,
                                                           messages=messages,
                                                           max_tokens=10)

    assert len(chat_completion.choices) == 1
    choice = chat_completion.choices[0]
    assert choice.finish_reason == "length"
