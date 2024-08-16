import math
import sys
import time
from typing import Dict, List, Optional, Tuple, Union, cast
from unittest.mock import patch

import librosa
import numpy as np
import openai
import pytest
import requests
import torch

from vllm import ModelRegistry
from vllm.config import MultiModalConfig
from vllm.inputs import INPUT_REGISTRY
from vllm.inputs.data import LLMInputs
from vllm.inputs.registry import InputContext
from vllm.model_executor.models.interfaces import SupportsMultiModal
from vllm.model_executor.models.opt import OPTForCausalLM
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.base import MultiModalInputs
from vllm.multimodal.image import (cached_get_tokenizer,
                                   repeat_and_pad_image_tokens)
from vllm.multimodal.utils import encode_audio_base64, fetch_audio
from vllm.utils import get_open_port

from ...utils import VLLM_PATH

chatml_jinja_path = VLLM_PATH / "examples/template_chatml.jinja"
assert chatml_jinja_path.exists()

MODEL_NAME = "facebook/opt-125m"
TEST_AUDIO_URLS = [
    "https://upload.wikimedia.org/wikipedia/en/b/bf/Dave_Niehaus_Winning_Call_1995_AL_Division_Series.ogg",
]


def server_function(port):

    def fake_input_mapper(ctx: InputContext, data: object):
        assert isinstance(data, tuple)
        (audio, sr) = cast(Tuple[np.ndarray, Union[float, int]], data)

        # Resample it to 1 sample per second
        audio = librosa.resample(audio, orig_sr=sr, target_sr=1)
        return MultiModalInputs({"processed_audio": torch.from_numpy(audio)})

    def fake_input_processor(ctx: InputContext, llm_inputs: LLMInputs):
        multi_modal_data = llm_inputs.get("multi_modal_data")
        if multi_modal_data is None or "audio" not in multi_modal_data:
            return llm_inputs

        audio, sr = multi_modal_data.get("audio")
        audio_duration = math.ceil(len(audio) / sr)

        new_prompt, new_token_ids = repeat_and_pad_image_tokens(
            cached_get_tokenizer(ctx.model_config.tokenizer),
            llm_inputs.get("prompt"),
            llm_inputs["prompt_token_ids"],
            image_token_id=62,  # "_"
            repeat_count=audio_duration)

        return LLMInputs(prompt_token_ids=new_token_ids,
                         prompt=new_prompt,
                         multi_modal_data=multi_modal_data)

    @MULTIMODAL_REGISTRY.register_input_mapper("audio", fake_input_mapper)
    @MULTIMODAL_REGISTRY.register_max_multimodal_tokens(
        "audio", lambda *_, **__: 100)
    @INPUT_REGISTRY.register_input_processor(fake_input_processor)
    class FakeAudioModel(OPTForCausalLM, SupportsMultiModal):

        def __init__(self, *args, multimodal_config: MultiModalConfig,
                     **kwargs):
            assert multimodal_config is not None
            super().__init__(*args, **kwargs)

        def forward(
            self,
            *args,
            processed_audio: Optional[torch.Tensor] = None,
            **kwargs,
        ) -> torch.Tensor:
            return super().forward(*args, **kwargs)

    ModelRegistry.register_model("OPTForCausalLM", FakeAudioModel)

    with patch("vllm.entrypoints.chat_utils._mm_token_str",
               lambda *_, **__: "_"):
        sys.argv = ["placeholder.py"] + \
            (f"--model {MODEL_NAME} --gpu-memory-utilization 0.10 "
            "--dtype bfloat16 --enforce-eager --api-key token-abc123 "
            f"--port {port} --chat-template {chatml_jinja_path} "
            "--disable-frontend-multiprocessing").split()
        import runpy
        runpy.run_module('vllm.entrypoints.openai.api_server',
                         run_name='__main__')


@pytest.fixture(scope="module")
def client():
    port = get_open_port()
    ctx = torch.multiprocessing.get_context("spawn")
    server = ctx.Process(target=server_function, args=(port, ))
    server.start()
    MAX_SERVER_START_WAIT_S = 60
    client = openai.AsyncOpenAI(
        base_url=f"http://localhost:{port}/v1",
        api_key="token-abc123",
    )
    # run health check
    health_url = f"http://localhost:{port}/health"
    start = time.time()
    while True:
        try:
            if requests.get(health_url).status_code == 200:
                break
        except Exception as err:
            result = server.exitcode
            if result is not None:
                raise RuntimeError("Server exited unexpectedly.") from err

            time.sleep(0.5)
            if time.time() - start > MAX_SERVER_START_WAIT_S:
                raise RuntimeError("Server failed to start in time.") from err

    try:
        yield client
    finally:
        server.kill()


@pytest.fixture(scope="session")
def base64_encoded_audio() -> Dict[str, str]:
    return {
        audio_url: encode_audio_base64(*fetch_audio(audio_url))
        for audio_url in TEST_AUDIO_URLS
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", [MODEL_NAME])
@pytest.mark.parametrize("audio_url", TEST_AUDIO_URLS)
async def test_single_chat_session_audio(client: openai.AsyncOpenAI,
                                         model_name: str, audio_url: str):
    messages = [{
        "role":
        "user",
        "content": [
            {
                "type": "audio_url",
                "audio_url": {
                    "url": audio_url
                }
            },
            {
                "type": "text",
                "text": "What's happening in this audio?"
            },
        ],
    }]

    # test single completion
    chat_completion = await client.chat.completions.create(model=model_name,
                                                           messages=messages,
                                                           max_tokens=10,
                                                           logprobs=True,
                                                           top_logprobs=5)
    assert len(chat_completion.choices) == 1

    choice = chat_completion.choices[0]
    assert choice.finish_reason == "length"
    assert chat_completion.usage == openai.types.CompletionUsage(
        completion_tokens=10, prompt_tokens=36, total_tokens=46)

    message = choice.message
    message = chat_completion.choices[0].message
    assert message.content is not None and len(message.content) >= 10
    assert message.role == "assistant"
    messages.append({"role": "assistant", "content": message.content})

    # test multi-turn dialogue
    messages.append({"role": "user", "content": "express your result in json"})
    chat_completion = await client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_tokens=10,
    )
    message = chat_completion.choices[0].message
    assert message.content is not None and len(message.content) >= 0


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", [MODEL_NAME])
@pytest.mark.parametrize("audio_url", TEST_AUDIO_URLS)
async def test_single_chat_session_audio_base64encoded(
        client: openai.AsyncOpenAI, model_name: str, audio_url: str,
        base64_encoded_audio: Dict[str, str]):

    messages = [{
        "role":
        "user",
        "content": [
            {
                "type": "audio_url",
                "audio_url": {
                    "url":
                    f"data:audio/wav;base64,{base64_encoded_audio[audio_url]}"
                }
            },
            {
                "type": "text",
                "text": "What's happening in this audio?"
            },
        ],
    }]

    # test single completion
    chat_completion = await client.chat.completions.create(model=model_name,
                                                           messages=messages,
                                                           max_tokens=10,
                                                           logprobs=True,
                                                           top_logprobs=5)
    assert len(chat_completion.choices) == 1

    choice = chat_completion.choices[0]
    assert choice.finish_reason == "length"
    assert chat_completion.usage == openai.types.CompletionUsage(
        completion_tokens=10, prompt_tokens=36, total_tokens=46)

    message = choice.message
    message = chat_completion.choices[0].message
    assert message.content is not None and len(message.content) >= 10
    assert message.role == "assistant"
    messages.append({"role": "assistant", "content": message.content})

    # test multi-turn dialogue
    messages.append({"role": "user", "content": "express your result in json"})
    chat_completion = await client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_tokens=10,
    )
    message = chat_completion.choices[0].message
    assert message.content is not None and len(message.content) >= 0


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", [MODEL_NAME])
@pytest.mark.parametrize("audio_url", TEST_AUDIO_URLS)
async def test_chat_streaming_audio(client: openai.AsyncOpenAI,
                                    model_name: str, audio_url: str):
    messages = [{
        "role":
        "user",
        "content": [
            {
                "type": "audio_url",
                "audio_url": {
                    "url": audio_url
                }
            },
            {
                "type": "text",
                "text": "What's happening in this audio?"
            },
        ],
    }]

    # test single completion
    chat_completion = await client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_tokens=10,
        temperature=0.0,
    )
    output = chat_completion.choices[0].message.content
    stop_reason = chat_completion.choices[0].finish_reason

    # test streaming
    stream = await client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_tokens=10,
        temperature=0.0,
        stream=True,
    )
    chunks: List[str] = []
    finish_reason_count = 0
    async for chunk in stream:
        delta = chunk.choices[0].delta
        if delta.role:
            assert delta.role == "assistant"
        if delta.content:
            chunks.append(delta.content)
        if chunk.choices[0].finish_reason is not None:
            finish_reason_count += 1
    # finish reason should only return in last block
    assert finish_reason_count == 1
    assert chunk.choices[0].finish_reason == stop_reason
    assert delta.content
    assert "".join(chunks) == output


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", [MODEL_NAME])
@pytest.mark.parametrize("audio_url", TEST_AUDIO_URLS)
async def test_multi_audio_input(client: openai.AsyncOpenAI, model_name: str,
                                 audio_url: str):

    messages = [{
        "role":
        "user",
        "content": [
            {
                "type": "audio_url",
                "audio_url": {
                    "url": audio_url
                }
            },
            {
                "type": "audio_url",
                "audio_url": {
                    "url": audio_url
                }
            },
            {
                "type": "text",
                "text": "What's happening in this audio?"
            },
        ],
    }]

    with pytest.raises(openai.BadRequestError):  # test multi-audio input
        await client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=10,
            temperature=0.0,
        )

    # the server should still work afterwards
    completion = await client.completions.create(
        model=model_name,
        prompt=[0, 0, 0, 0, 0],
        max_tokens=5,
        temperature=0.0,
    )
    completion = completion.choices[0].text
    assert completion is not None and len(completion) >= 0
