# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import subprocess
import tempfile

import pytest

from vllm.entrypoints.openai.run_batch import BatchRequestOutput

CHAT_MODEL_NAME = "hmellor/tiny-random-LlamaForCausalLM"
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-small"
RERANKER_MODEL_NAME = "BAAI/bge-reranker-v2-m3"
REASONING_MODEL_NAME = "Qwen/Qwen3-0.6B"

# ruff: noqa: E501
INPUT_BATCH = (
    '{{"custom_id": "request-1", "method": "POST", "url": "/v1/chat/completions", "body": {{"model": "{0}", "messages": [{{"role": "system", "content": "You are a helpful assistant."}},{{"role": "user", "content": "Hello world!"}}],"max_tokens": 1000}}}}\n'
    '{{"custom_id": "request-2", "method": "POST", "url": "/v1/chat/completions", "body": {{"model": "{0}", "messages": [{{"role": "system", "content": "You are an unhelpful assistant."}},{{"role": "user", "content": "Hello world!"}}],"max_tokens": 1000}}}}\n'
    '{{"custom_id": "request-3", "method": "POST", "url": "/v1/chat/completions", "body": {{"model": "NonExistModel", "messages": [{{"role": "system", "content": "You are an unhelpful assistant."}},{{"role": "user", "content": "Hello world!"}}],"max_tokens": 1000}}}}\n'
    '{{"custom_id": "request-4", "method": "POST", "url": "/bad_url", "body": {{"model": "{0}", "messages": [{{"role": "system", "content": "You are an unhelpful assistant."}},{{"role": "user", "content": "Hello world!"}}],"max_tokens": 1000}}}}\n'
    '{{"custom_id": "request-5", "method": "POST", "url": "/v1/chat/completions", "body": {{"stream": "True", "model": "{0}", "messages": [{{"role": "system", "content": "You are an unhelpful assistant."}},{{"role": "user", "content": "Hello world!"}}],"max_tokens": 1000}}}}'
).format(CHAT_MODEL_NAME)

INVALID_INPUT_BATCH = (
    '{{"invalid_field": "request-1", "method": "POST", "url": "/v1/chat/completions", "body": {{"model": "{0}", "messages": [{{"role": "system", "content": "You are a helpful assistant."}},{{"role": "user", "content": "Hello world!"}}],"max_tokens": 1000}}}}\n'
    '{{"custom_id": "request-2", "method": "POST", "url": "/v1/chat/completions", "body": {{"model": "{0}", "messages": [{{"role": "system", "content": "You are an unhelpful assistant."}},{{"role": "user", "content": "Hello world!"}}],"max_tokens": 1000}}}}'
).format(CHAT_MODEL_NAME)

INPUT_EMBEDDING_BATCH = (
    f'{{"custom_id": "request-1", "method": "POST", "url": "/v1/embeddings", "body": {{"model": "{EMBEDDING_MODEL_NAME}", "input": "You are a helpful assistant."}}}}\n'
    f'{{"custom_id": "request-2", "method": "POST", "url": "/v1/embeddings", "body": {{"model": "{EMBEDDING_MODEL_NAME}", "input": "You are an unhelpful assistant."}}}}\n'
    f'{{"custom_id": "request-3", "method": "POST", "url": "/v1/embeddings", "body": {{"model": "{EMBEDDING_MODEL_NAME}", "input": "Hello world!"}}}}\n'
    '{"custom_id": "request-4", "method": "POST", "url": "/v1/embeddings", "body": {"model": "NonExistModel", "input": "Hello world!"}}'
)

INPUT_SCORE_BATCH = f"""{{"custom_id": "request-1", "method": "POST", "url": "/score", "body": {{"model": "{RERANKER_MODEL_NAME}", "queries": "What is the capital of France?", "documents": ["The capital of Brazil is Brasilia.", "The capital of France is Paris."]}}}}
{{"custom_id": "request-2", "method": "POST", "url": "/v1/score", "body": {{"model": "{RERANKER_MODEL_NAME}", "queries": "What is the capital of France?", "documents": ["The capital of Brazil is Brasilia.", "The capital of France is Paris."]}}}}"""

INPUT_RERANK_BATCH = f"""{{"custom_id": "request-1", "method": "POST", "url": "/rerank", "body": {{"model": "{RERANKER_MODEL_NAME}", "query": "What is the capital of France?", "documents": ["The capital of Brazil is Brasilia.", "The capital of France is Paris."]}}}}
{{"custom_id": "request-2", "method": "POST", "url": "/v1/rerank", "body": {{"model": "{RERANKER_MODEL_NAME}", "query": "What is the capital of France?", "documents": ["The capital of Brazil is Brasilia.", "The capital of France is Paris."]}}}}
{{"custom_id": "request-2", "method": "POST", "url": "/v2/rerank", "body": {{"model": "{RERANKER_MODEL_NAME}", "query": "What is the capital of France?", "documents": ["The capital of Brazil is Brasilia.", "The capital of France is Paris."]}}}}"""

INPUT_REASONING_BATCH = f"""{{"custom_id": "request-1", "method": "POST", "url": "/v1/chat/completions", "body": {{"model": "{REASONING_MODEL_NAME}", "messages": [{{"role": "system", "content": "You are a helpful assistant."}},{{"role": "user", "content": "Solve this math problem: 2+2=?"}}]}}}}
{{"custom_id": "request-2", "method": "POST", "url": "/v1/chat/completions", "body": {{"model": "{REASONING_MODEL_NAME}", "messages": [{{"role": "system", "content": "You are a helpful assistant."}},{{"role": "user", "content": "What is the capital of France?"}}]}}}}"""

# Tool definition for testing tool calling
WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_current_weather",
        "description": "Get the current weather in a given location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and state, e.g. San Francisco, CA",
                },
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                },
            },
            "required": ["location"],
        },
    },
}

INPUT_TOOL_CALLING_BATCH = f"""{{"custom_id": "request-1", "method": "POST", "url": "/v1/chat/completions", "body": {{"model": "{REASONING_MODEL_NAME}", "messages": [{{"role": "user", "content": "What is the weather in San Francisco?"}}], "tools": [{json.dumps(WEATHER_TOOL)}], "tool_choice": "required", "max_tokens": 1000}}}}"""


def test_empty_file():
    with (
        tempfile.NamedTemporaryFile("w") as input_file,
        tempfile.NamedTemporaryFile("r") as output_file,
    ):
        input_file.write("")
        input_file.flush()
        proc = subprocess.Popen(
            [
                "vllm",
                "run-batch",
                "-i",
                input_file.name,
                "-o",
                output_file.name,
                "--model",
                EMBEDDING_MODEL_NAME,
            ],
        )
        proc.communicate()
        proc.wait()
        assert proc.returncode == 0, f"{proc=}"

        contents = output_file.read()
        assert contents.strip() == ""


def test_completions():
    with (
        tempfile.NamedTemporaryFile("w") as input_file,
        tempfile.NamedTemporaryFile("r") as output_file,
    ):
        input_file.write(INPUT_BATCH)
        input_file.flush()
        proc = subprocess.Popen(
            [
                "vllm",
                "run-batch",
                "-i",
                input_file.name,
                "-o",
                output_file.name,
                "--model",
                CHAT_MODEL_NAME,
            ],
        )
        proc.communicate()
        proc.wait()
        assert proc.returncode == 0, f"{proc=}"

        contents = output_file.read()
        for line in contents.strip().split("\n"):
            # Ensure that the output format conforms to the openai api.
            # Validation should throw if the schema is wrong.
            BatchRequestOutput.model_validate_json(line)


def test_completions_invalid_input():
    """
    Ensure that we fail when the input doesn't conform to the openai api.
    """
    with (
        tempfile.NamedTemporaryFile("w") as input_file,
        tempfile.NamedTemporaryFile("r") as output_file,
    ):
        input_file.write(INVALID_INPUT_BATCH)
        input_file.flush()
        proc = subprocess.Popen(
            [
                "vllm",
                "run-batch",
                "-i",
                input_file.name,
                "-o",
                output_file.name,
                "--model",
                CHAT_MODEL_NAME,
            ],
        )
        proc.communicate()
        proc.wait()
        assert proc.returncode != 0, f"{proc=}"


def test_embeddings():
    with (
        tempfile.NamedTemporaryFile("w") as input_file,
        tempfile.NamedTemporaryFile("r") as output_file,
    ):
        input_file.write(INPUT_EMBEDDING_BATCH)
        input_file.flush()
        proc = subprocess.Popen(
            [
                "vllm",
                "run-batch",
                "-i",
                input_file.name,
                "-o",
                output_file.name,
                "--model",
                EMBEDDING_MODEL_NAME,
            ],
        )
        proc.communicate()
        proc.wait()
        assert proc.returncode == 0, f"{proc=}"

        contents = output_file.read()
        for line in contents.strip().split("\n"):
            # Ensure that the output format conforms to the openai api.
            # Validation should throw if the schema is wrong.
            BatchRequestOutput.model_validate_json(line)


@pytest.mark.parametrize("input_batch", [INPUT_SCORE_BATCH, INPUT_RERANK_BATCH])
def test_score(input_batch):
    with (
        tempfile.NamedTemporaryFile("w") as input_file,
        tempfile.NamedTemporaryFile("r") as output_file,
    ):
        input_file.write(input_batch)
        input_file.flush()
        proc = subprocess.Popen(
            [
                "vllm",
                "run-batch",
                "-i",
                input_file.name,
                "-o",
                output_file.name,
                "--model",
                RERANKER_MODEL_NAME,
            ],
        )
        proc.communicate()
        proc.wait()
        assert proc.returncode == 0, f"{proc=}"

        contents = output_file.read()
        for line in contents.strip().split("\n"):
            # Ensure that the output format conforms to the openai api.
            # Validation should throw if the schema is wrong.
            BatchRequestOutput.model_validate_json(line)

            # Ensure that there is no error in the response.
            line_dict = json.loads(line)
            assert isinstance(line_dict, dict)
            assert line_dict["error"] is None


def test_reasoning_parser():
    """
    Test that reasoning_parser parameter works correctly in run_batch.
    """
    with (
        tempfile.NamedTemporaryFile("w") as input_file,
        tempfile.NamedTemporaryFile("r") as output_file,
    ):
        input_file.write(INPUT_REASONING_BATCH)
        input_file.flush()
        proc = subprocess.Popen(
            [
                "vllm",
                "run-batch",
                "-i",
                input_file.name,
                "-o",
                output_file.name,
                "--model",
                REASONING_MODEL_NAME,
                "--reasoning-parser",
                "qwen3",
            ],
        )
        proc.communicate()
        proc.wait()
        assert proc.returncode == 0, f"{proc=}"

        contents = output_file.read()
        for line in contents.strip().split("\n"):
            # Ensure that the output format conforms to the openai api.
            # Validation should throw if the schema is wrong.
            BatchRequestOutput.model_validate_json(line)

            # Ensure that there is no error in the response.
            line_dict = json.loads(line)
            assert isinstance(line_dict, dict)
            assert line_dict["error"] is None

            # Check that reasoning is present and not empty
            reasoning = line_dict["response"]["body"]["choices"][0]["message"][
                "reasoning"
            ]
            assert reasoning is not None
            assert len(reasoning) > 0


def test_tool_calling():
    """
    Test that tool calling works correctly in run_batch.
    Verifies that requests with tools return tool_calls in the response.
    """
    with (
        tempfile.NamedTemporaryFile("w") as input_file,
        tempfile.NamedTemporaryFile("r") as output_file,
    ):
        input_file.write(INPUT_TOOL_CALLING_BATCH)
        input_file.flush()
        proc = subprocess.Popen(
            [
                "vllm",
                "run-batch",
                "-i",
                input_file.name,
                "-o",
                output_file.name,
                "--model",
                REASONING_MODEL_NAME,
                "--enable-auto-tool-choice",
                "--tool-call-parser",
                "hermes",
            ],
        )
        proc.communicate()
        proc.wait()
        assert proc.returncode == 0, f"{proc=}"

        contents = output_file.read()
        for line in contents.strip().split("\n"):
            if not line.strip():  # Skip empty lines
                continue
            # Ensure that the output format conforms to the openai api.
            # Validation should throw if the schema is wrong.
            BatchRequestOutput.model_validate_json(line)

            # Ensure that there is no error in the response.
            line_dict = json.loads(line)
            assert isinstance(line_dict, dict)
            assert line_dict["error"] is None

            # Check that tool_calls are present in the response
            # With tool_choice="required", the model must call a tool
            response_body = line_dict["response"]["body"]
            assert response_body is not None
            message = response_body["choices"][0]["message"]
            assert "tool_calls" in message
            tool_calls = message.get("tool_calls")
            # With tool_choice="required", tool_calls must be present and non-empty
            assert tool_calls is not None
            assert isinstance(tool_calls, list)
            assert len(tool_calls) > 0
            # Verify tool_calls have the expected structure
            for tool_call in tool_calls:
                assert "id" in tool_call
                assert "type" in tool_call
                assert tool_call["type"] == "function"
                assert "function" in tool_call
                assert "name" in tool_call["function"]
                assert "arguments" in tool_call["function"]
                # Verify the tool name matches our tool definition
                assert tool_call["function"]["name"] == "get_current_weather"
