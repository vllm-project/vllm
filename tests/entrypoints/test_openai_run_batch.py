import subprocess
import sys
import tempfile

from vllm.entrypoints.openai.protocol import BatchRequestOutput

# ruff: noqa: E501
INPUT_BATCH = """{"custom_id": "request-1", "method": "POST", "url": "/v1/chat/completions", "body": {"model": "NousResearch/Meta-Llama-3-8B-Instruct", "messages": [{"role": "system", "content": "You are a helpful assistant."},{"role": "user", "content": "Hello world!"}],"max_tokens": 1000}}
{"custom_id": "request-2", "method": "POST", "url": "/v1/chat/completions", "body": {"model": "NousResearch/Meta-Llama-3-8B-Instruct", "messages": [{"role": "system", "content": "You are an unhelpful assistant."},{"role": "user", "content": "Hello world!"}],"max_tokens": 1000}}"""

INVALID_INPUT_BATCH = """{"invalid_field": "request-1", "method": "POST", "url": "/v1/chat/completions", "body": {"model": "NousResearch/Meta-Llama-3-8B-Instruct", "messages": [{"role": "system", "content": "You are a helpful assistant."},{"role": "user", "content": "Hello world!"}],"max_tokens": 1000}}
{"custom_id": "request-2", "method": "POST", "url": "/v1/chat/completions", "body": {"model": "NousResearch/Meta-Llama-3-8B-Instruct", "messages": [{"role": "system", "content": "You are an unhelpful assistant."},{"role": "user", "content": "Hello world!"}],"max_tokens": 1000}}"""


def test_e2e():
    with tempfile.NamedTemporaryFile(
            "w") as input_file, tempfile.NamedTemporaryFile(
                "r") as output_file:
        input_file.write(INPUT_BATCH)
        input_file.flush()
        proc = subprocess.Popen([
            sys.executable, "-m", "vllm.entrypoints.openai.run_batch", "-i",
            input_file.name, "-o", output_file.name, "--model",
            "NousResearch/Meta-Llama-3-8B-Instruct"
        ], )
        proc.communicate()
        proc.wait()
        assert proc.returncode == 0, f"{proc=}"

        contents = output_file.read()
        for line in contents.strip().split("\n"):
            # Ensure that the output format conforms to the openai api.
            # Validation should throw if the schema is wrong.
            BatchRequestOutput.model_validate_json(line)


def test_e2e_invalid_input():
    """
    Ensure that we fail when the input doesn't conform to the openai api.
    """
    with tempfile.NamedTemporaryFile(
            "w") as input_file, tempfile.NamedTemporaryFile(
                "r") as output_file:
        input_file.write(INVALID_INPUT_BATCH)
        input_file.flush()
        proc = subprocess.Popen([
            sys.executable, "-m", "vllm.entrypoints.openai.run_batch", "-i",
            input_file.name, "-o", output_file.name, "--model",
            "NousResearch/Meta-Llama-3-8B-Instruct"
        ], )
        proc.communicate()
        proc.wait()
        assert proc.returncode != 0, f"{proc=}"
