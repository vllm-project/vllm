# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json

import pytest
from mistral_common.audio import Audio
from mistral_common.protocol.instruct.chunk import AudioChunk, RawAudio, TextChunk
from mistral_common.protocol.instruct.messages import UserMessage
from transformers import VoxtralForConditionalGeneration

from vllm.tokenizers.mistral import MistralTokenizer

from ....conftest import AudioTestAssets
from ....utils import RemoteOpenAIServer
from .test_ultravox import MULTI_AUDIO_PROMPT, run_multi_audio_test
from .vlm_utils import model_utils

MODEL_NAME = "mistralai/Voxtral-Mini-3B-2507"
MISTRAL_FORMAT_ARGS = [
    "--tokenizer_mode",
    "mistral",
    "--config_format",
    "mistral",
    "--load_format",
    "mistral",
]


def _get_prompt(audio_assets: AudioTestAssets, question: str) -> list[int]:
    """Build a token-ID prompt via mistral_common for vLLM offline inference."""
    tokenizer = MistralTokenizer.from_pretrained(MODEL_NAME)

    audios = [
        Audio.from_file(str(asset.get_local_path()), strict=False)
        for asset in audio_assets
    ]
    audio_chunks = [
        AudioChunk(input_audio=RawAudio.from_audio(audio)) for audio in audios
    ]

    messages = [
        UserMessage(content=[*audio_chunks, TextChunk(text=question)]).to_openai()
    ]
    return tokenizer.apply_chat_template(messages=messages)


@pytest.mark.core_model
@pytest.mark.parametrize("dtype", ["half"])
@pytest.mark.parametrize("max_tokens", [128])
@pytest.mark.parametrize("num_logprobs", [5])
def test_models_with_multiple_audios(
    vllm_runner,
    audio_assets: AudioTestAssets,
    dtype: str,
    max_tokens: int,
    num_logprobs: int,
) -> None:
    vllm_prompt = _get_prompt(audio_assets, MULTI_AUDIO_PROMPT)
    run_multi_audio_test(
        vllm_runner,
        [(vllm_prompt, [a.audio_and_sample_rate for a in audio_assets])],
        MODEL_NAME,
        dtype=dtype,
        max_tokens=max_tokens,
        num_logprobs=num_logprobs,
        tokenizer_mode="mistral",
    )


def test_online_serving(hf_runner, vllm_runner, audio_assets: AudioTestAssets):
    """Three-layer accuracy and serving validation.

    1. Offline vLLM greedy output (runs first to avoid CUDA fork issues
       with multiprocessing - see vlm_utils/core.py).
    2. HF Transformers greedy output as independent ground truth - uses
       a completely separate model loading path (AutoProcessor +
       VoxtralForConditionalGeneration) to validate that vLLM's
       mistral-format weight loading and audio preprocessing are correct.
    3. Online OpenAI-compatible API output must match offline - validates
       that the serving path (chat template, audio encoding, tokenization)
       does not corrupt anything.

    Steps run sequentially so each releases the GPU before the next starts.
    """

    question = f"What's happening in these {len(audio_assets)} audio clips?"
    max_tokens = 10
    audio_data = [asset.audio_and_sample_rate for asset in audio_assets]

    vllm_prompt = _get_prompt(audio_assets, question)
    with vllm_runner(
        MODEL_NAME,
        dtype="half",
        enforce_eager=True,
        tokenizer_mode="mistral",
        config_format="mistral",
        load_format="mistral",
        limit_mm_per_prompt={"audio": len(audio_assets)},
    ) as vllm_model:
        offline_outputs = vllm_model.generate_greedy(
            [vllm_prompt],
            max_tokens,
            audios=[audio_data],
        )

    offline_text = offline_outputs[0][1]
    assert offline_text, "Offline vLLM inference produced empty output"

    with hf_runner(
        MODEL_NAME,
        dtype="half",
        auto_cls=VoxtralForConditionalGeneration,
    ) as hf_model:
        hf_model = model_utils.voxtral_patch_hf_runner(hf_model)
        hf_outputs = hf_model.generate_greedy(
            [question],
            max_tokens,
            audios=[audio_data],
        )

    hf_text = hf_outputs[0][1]
    assert hf_text, "HF Transformers produced empty output"
    assert offline_text == hf_text, (
        f"vLLM offline output does not match HF Transformers.\n"
        f"  vLLM: {offline_text!r}\n"
        f"  HF:   {hf_text!r}"
    )

    def _asset_to_openai_chunk(asset):
        audio = Audio.from_file(str(asset.get_local_path()), strict=False)
        audio.format = "wav"
        return AudioChunk.from_audio(audio).to_openai()

    messages = [
        {
            "role": "user",
            "content": [
                *[_asset_to_openai_chunk(a) for a in audio_assets],
                {"type": "text", "text": question},
            ],
        }
    ]

    server_args = [
        "--enforce-eager",
        "--limit-mm-per-prompt",
        json.dumps({"audio": len(audio_assets)}),
        *MISTRAL_FORMAT_ARGS,
    ]

    with RemoteOpenAIServer(
        MODEL_NAME,
        server_args,
        env_dict={"VLLM_AUDIO_FETCH_TIMEOUT": "30"},
    ) as remote_server:
        client = remote_server.get_client()
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            max_tokens=max_tokens,
            temperature=0,
        )

    assert len(completion.choices) == 1
    choice = completion.choices[0]
    assert choice.finish_reason == "length"
    assert choice.message.content == offline_text, (
        f"Online serving output does not match offline inference.\n"
        f"  Online:  {choice.message.content!r}\n"
        f"  Offline: {offline_text!r}"
    )
