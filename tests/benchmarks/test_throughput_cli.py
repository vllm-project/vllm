# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import subprocess
from unittest.mock import Mock

import pytest

from vllm.benchmarks.datasets import MMVUDataset, SampleRequest
from vllm.benchmarks.throughput import (
    _run_vllm_chat_requests,
    add_cli_args,
    get_requests,
    validate_args,
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


def test_bench_throughput_accepts_mmvu_dataset():
    parser = FlexibleArgumentParser()
    add_cli_args(parser)

    args = parser.parse_args(
        [
            "--model",
            MODEL_NAME,
            "--backend",
            "vllm-chat",
            "--dataset-name",
            "hf",
            "--dataset-path",
            "yale-nlp/MMVU",
        ]
    )

    validate_args(args)


def test_bench_throughput_loads_mmvu_dataset(monkeypatch: pytest.MonkeyPatch):
    parser = FlexibleArgumentParser()
    add_cli_args(parser)
    args = parser.parse_args(
        [
            "--model",
            MODEL_NAME,
            "--backend",
            "vllm-chat",
            "--dataset-name",
            "hf",
            "--dataset-path",
            "yale-nlp/MMVU",
        ]
    )
    captured = {}

    class FakeMMVUDataset:
        SUPPORTED_DATASET_PATHS = {"yale-nlp/MMVU": None}

        def __init__(self, **kwargs):
            captured["init"] = kwargs

        def sample(self, **kwargs):
            captured["sample"] = kwargs
            return []

    monkeypatch.setattr(
        "vllm.benchmarks.throughput.MMVUDataset", FakeMMVUDataset
    )

    assert get_requests(args, Mock()) == []
    assert captured["init"]["dataset_split"] == "validation"
    assert captured["sample"]["enable_multimodal_chat"]


def test_mmvu_dataset_embeds_local_video_bytes(tmp_path):
    video_path = tmp_path / "video.mp4"
    video_path.write_bytes(b"video-bytes")
    dataset = MMVUDataset.__new__(MMVUDataset)
    dataset._remote_path_root = "https://huggingface.co/datasets/yale-nlp/MMVU/resolve/main"
    dataset._local_path_root = str(tmp_path)
    dataset.hf_name = "yale-nlp/MMVU"
    dataset.data = [
        {
            "question": "What happens?",
            "choices": {"A": "Something", "B": "Nothing"},
            "video": f"{dataset._remote_path_root}/video.mp4",
        }
    ]
    tokenizer = Mock()
    tokenizer.encode.return_value = []

    request = dataset.sample(tokenizer, num_requests=1)[0]

    assert request.multi_modal_data == {
        "type": "video_url",
        "video_url": {"url": "data:video/mp4;base64,dmlkZW8tYnl0ZXM="},
    }


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
