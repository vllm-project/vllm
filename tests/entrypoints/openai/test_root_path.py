# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import contextlib
import os
from typing import Any, NamedTuple

import openai  # use the official client for correctness check
import pytest

from ...utils import RemoteOpenAIServer

# # any model with a chat template should work here
MODEL_NAME = "Qwen/Qwen2-1.5B-Instruct"
API_KEY = "abc-123"
ERROR_API_KEY = "abc"
ROOT_PATH = "llm"


@pytest.fixture(scope="module")
def server():
    args = [
        # use half precision for speed and memory savings in CI environment
        "--dtype",
        "float16",
        "--enforce-eager",
        "--max-model-len",
        "4080",
        "--root-path",  # use --root-path=/llm for testing
        "/" + ROOT_PATH,
    ]
    envs = os.environ.copy()

    envs["VLLM_API_KEY"] = API_KEY
    with RemoteOpenAIServer(MODEL_NAME, args, env_dict=envs) as remote_server:
        yield remote_server


class TestCase(NamedTuple):
    model_name: str
    base_url: list[str]
    api_key: str
    expected_error: Any


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            model_name=MODEL_NAME,
            base_url=["v1"],  # http://localhost:8000/v1
            api_key=ERROR_API_KEY,
            expected_error=openai.AuthenticationError),
        TestCase(
            model_name=MODEL_NAME,
            base_url=[ROOT_PATH, "v1"],  # http://localhost:8000/llm/v1
            api_key=ERROR_API_KEY,
            expected_error=openai.AuthenticationError),
        TestCase(
            model_name=MODEL_NAME,
            base_url=["v1"],  # http://localhost:8000/v1
            api_key=API_KEY,
            expected_error=None),
        TestCase(
            model_name=MODEL_NAME,
            base_url=[ROOT_PATH, "v1"],  # http://localhost:8000/llm/v1
            api_key=API_KEY,
            expected_error=None),
    ],
)
async def test_chat_session_root_path_with_api_key(server: RemoteOpenAIServer,
                                                   test_case: TestCase):
    saying: str = "Here is a common saying about apple. An apple a day, keeps"
    ctx = contextlib.nullcontext()
    if test_case.expected_error is not None:
        ctx = pytest.raises(test_case.expected_error)
    with ctx:
        client = openai.AsyncOpenAI(
            api_key=test_case.api_key,
            base_url=server.url_for(*test_case.base_url),
            max_retries=0)
        chat_completion = await client.chat.completions.create(
            model=test_case.model_name,
            messages=[{
                "role": "user",
                "content": "tell me a common saying"
            }, {
                "role": "assistant",
                "content": saying
            }],
            extra_body={
                "continue_final_message": True,
                "add_generation_prompt": False
            })

        assert chat_completion.id is not None
        assert len(chat_completion.choices) == 1
        choice = chat_completion.choices[0]
        assert choice.finish_reason == "stop"
        message = choice.message
        assert len(message.content) > 0
        assert message.role == "assistant"
