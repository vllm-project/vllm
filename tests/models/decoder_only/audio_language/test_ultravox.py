# SPDX-License-Identifier: Apache-2.0

from typing import Optional

import numpy as np
import pytest
import pytest_asyncio
from transformers import AutoModel, AutoTokenizer

from vllm.multimodal.audio import resample_audio
from vllm.sequence import SampleLogprobs

from ....conftest import HfRunner, VllmRunner
from ....utils import RemoteOpenAIServer
from ...utils import check_logprobs_close

MODEL_NAME = "fixie-ai/ultravox-v0_5-llama-3_2-1b"

AudioTuple = tuple[np.ndarray, int]

VLLM_PLACEHOLDER = "<|audio|>"
HF_PLACEHOLDER = "<|audio|>"

CHUNKED_PREFILL_KWARGS = {
    "enable_chunked_prefill": True,
    "max_num_seqs": 2,
    # Use a very small limit to exercise chunked prefill.
    "max_num_batched_tokens": 16
}


@pytest.fixture(scope="session")
def audio_assets():
    from vllm.assets.audio import AudioAsset
    return [AudioAsset("mary_had_lamb"), AudioAsset("winning_call")]


@pytest.fixture(scope="module", params=("mary_had_lamb", "winning_call"))
def audio(request):
    from vllm.assets.audio import AudioAsset
    return AudioAsset(request.param)


@pytest.fixture(params=[
    pytest.param({}, marks=pytest.mark.cpu_model),
    pytest.param(CHUNKED_PREFILL_KWARGS),
])
def server(request, audio_assets):
    args = [
        "--dtype=bfloat16", "--max-model-len=4096", "--enforce-eager",
        f"--limit-mm-per-prompt=audio={len(audio_assets)}",
        "--trust-remote-code"
    ] + [
        f"--{key.replace('_','-')}={value}"
        for key, value in request.param.items()
    ]

    with RemoteOpenAIServer(MODEL_NAME, args) as remote_server:
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


def vllm_to_hf_output(vllm_output: tuple[list[int], str,
                                         Optional[SampleLogprobs]],
                      model: str):
    """Sanitize vllm output to be comparable with hf output."""
    output_ids, output_str, out_logprobs = vllm_output

    tokenizer = AutoTokenizer.from_pretrained(model)
    eos_token_id = tokenizer.eos_token_id

    hf_output_ids = output_ids[:]
    hf_output_str = output_str
    if hf_output_ids[-1] == eos_token_id:
        hf_output_str = hf_output_str + tokenizer.decode(eos_token_id)

    return hf_output_ids, hf_output_str, out_logprobs


def run_test(
    hf_runner: type[HfRunner],
    vllm_runner: type[VllmRunner],
    prompts_and_audios: list[tuple[str, str, AudioTuple]],
    model: str,
    *,
    dtype: str,
    max_tokens: int,
    num_logprobs: int,
    **kwargs,
):
    """Inference result should be the same between hf and vllm."""
    # NOTE: take care of the order. run vLLM first, and then run HF.
    # vLLM needs a fresh new process without cuda initialization.
    # if we run HF first, the cuda initialization will be done and it
    # will hurt multiprocessing backend with fork method (the default method).

    with vllm_runner(model, dtype=dtype, enforce_eager=True,
                     **kwargs) as vllm_model:
        vllm_outputs_per_audio = [
            vllm_model.generate_greedy_logprobs([vllm_prompt],
                                                max_tokens,
                                                num_logprobs=num_logprobs,
                                                audios=[audio])
            for vllm_prompt, _, audio in prompts_and_audios
        ]

    with hf_runner(model, dtype=dtype, auto_cls=AutoModel) as hf_model:
        hf_outputs_per_audio = [
            hf_model.generate_greedy_logprobs_limit(
                [hf_prompt],
                max_tokens,
                num_logprobs=num_logprobs,
                audios=[(resample_audio(audio[0],
                                        orig_sr=audio[1],
                                        target_sr=16000), 16000)])
            for _, hf_prompt, audio in prompts_and_audios
        ]

    for hf_outputs, vllm_outputs in zip(hf_outputs_per_audio,
                                        vllm_outputs_per_audio):
        check_logprobs_close(
            outputs_0_lst=hf_outputs,
            outputs_1_lst=[
                vllm_to_hf_output(vllm_output, model)
                for vllm_output in vllm_outputs
            ],
            name_0="hf",
            name_1="vllm",
        )


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
@pytest.mark.parametrize("dtype", ["bfloat16"])
@pytest.mark.parametrize("max_tokens", [128])
@pytest.mark.parametrize("num_logprobs", [5])
@pytest.mark.parametrize("vllm_kwargs", [
    pytest.param({}, marks=pytest.mark.cpu_model),
    pytest.param(CHUNKED_PREFILL_KWARGS),
])
def test_models(hf_runner, vllm_runner, audio, dtype: str, max_tokens: int,
                num_logprobs: int, vllm_kwargs: dict) -> None:

    vllm_prompt = _get_prompt(1, "Describe the audio above.", VLLM_PLACEHOLDER)
    hf_prompt = _get_prompt(1, "Describe the audio above.", HF_PLACEHOLDER)
    run_test(
        hf_runner,
        vllm_runner,
        [(vllm_prompt, hf_prompt, audio.audio_and_sample_rate)],
        MODEL_NAME,
        dtype=dtype,
        max_tokens=max_tokens,
        num_logprobs=num_logprobs,
        **vllm_kwargs,
    )


@pytest.mark.core_model
@pytest.mark.parametrize("dtype", ["half"])
@pytest.mark.parametrize("max_tokens", [128])
@pytest.mark.parametrize("num_logprobs", [5])
@pytest.mark.parametrize("vllm_kwargs", [
    pytest.param({}, marks=pytest.mark.cpu_model),
    pytest.param(CHUNKED_PREFILL_KWARGS),
])
def test_models_with_multiple_audios(vllm_runner, audio_assets, dtype: str,
                                     max_tokens: int, num_logprobs: int,
                                     vllm_kwargs: dict) -> None:

    vllm_prompt = _get_prompt(len(audio_assets),
                              "Describe each of the audios above.",
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
async def test_online_serving(client, audio_assets):
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
