
import tempfile
import subprocess
import sys

from vllm.entrypoints.openai.protocol import (
    BatchRequestOutput)

# ruff: noqa: E501
INPUT_BATCH = """{"custom_id": "request-1", "method": "POST", "url": "/v1/chat/completions", "body": {"model": "NousResearch/Meta-Llama-3-8B-Instruct", "messages": [{"role": "system", "content": "You are a helpful assistant."},{"role": "user", "content": "Hello world!"}],"max_tokens": 1000}}
{"custom_id": "request-2", "method": "POST", "url": "/v1/chat/completions", "body": {"model": "NousResearch/Meta-Llama-3-8B-Instruct", "messages": [{"role": "system", "content": "You are an unhelpful assistant."},{"role": "user", "content": "Hello world!"}],"max_tokens": 1000}}"""


def test_e2e():
    with tempfile.NamedTemporaryFile("w") as input_file, tempfile.NamedTemporaryFile("r") as output_file:
            input_file.write(INPUT_BATCH)
            input_file.flush()
            proc = subprocess.Popen([
                sys.executable, "-m", "vllm.entrypoints.openai.run_batch",
                "-i", input_file.name, "-o", output_file.name, "--model",
                "NousResearch/Meta-Llama-3-8B-Instruct"
            ], )
            proc.communicate()
            proc.wait()
            assert proc.returncode == 0, f"{proc=}"

            contents = output_file.read()
            for line in contents.strip().split("\n"):
                BatchRequestOutput.model_validate_json(line)
