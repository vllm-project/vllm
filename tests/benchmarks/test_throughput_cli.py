# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import subprocess

import pytest

from vllm.benchmarks.datasets import SampleRequest
from vllm.benchmarks.throughput import (
    _run_vllm_chat_requests,
    add_cli_args,
)
from vllm.utils.argparse_utils import FlexibleArgumentParser

MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"


@pytest.mark.benchmark
def test_bench_throughput():
    command = [
        "vllm",
        "bench",
        "throughput",
        "--model",
        MODEL_NAME,
        "--input-len",
        "32",
        "--output-len",
        "1",
        "--enforce-eager",
        "--load-format",
        "dummy",
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    print(result.stdout)
    print(result.stderr)

    assert result.returncode == 0, f"Benchmark failed: {result.stderr}"


def test_bench_throughput_accepts_custom_audio_args():
    parser = FlexibleArgumentParser()
    add_cli_args(parser)

    args = parser.parse_args(
        [
            "--dataset-name",
            "custom_audio",
            "--dataset-path",
            "audio.jsonl",
            "--no-oversample",
            "--custom-output-len",
            "32",
            "--enable-multimodal-chat",
        ]
    )

    assert args.dataset_name == "custom_audio"
    assert args.no_oversample
    assert args.custom_output_len == 32
    assert args.enable_multimodal_chat


def test_vllm_chat_requests_include_multimodal_content():
    class FakeLLM:
        def __init__(self):
            self.prompts = None

        def chat(self, prompts, sampling_params, use_tqdm):
            del sampling_params, use_tqdm
            self.prompts = prompts
            return []

    llm = FakeLLM()
    audio_content = {
        "type": "input_audio",
        "input_audio": {"data": "abc", "format": "wav"},
    }
    request = SampleRequest(
        prompt="Transcribe this audio.",
        prompt_len=1,
        expected_output_len=8,
        multi_modal_data=audio_content,
    )

    _run_vllm_chat_requests(
        llm,
        [request],
        n=1,
        disable_detokenize=False,
        do_profile=False,
        prequeue_requests=False,
    )

    assert llm.prompts == [
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Transcribe this audio."},
                    audio_content,
                ],
            }
        ]
    ]
