# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json
from argparse import Namespace
from pathlib import Path
from typing import Any

import pytest

from vllm.benchmarks.datasets import CustomImageDataset, get_samples
from vllm.benchmarks.lib.endpoint_request_func import (
    RequestFuncInput,
    _get_chat_content,
    _get_chat_messages,
)

pytestmark = pytest.mark.skip_global_cleanup


class _TokenizedPrompt:
    def __init__(self, prompt: str) -> None:
        self.input_ids = prompt.split()


class _Tokenizer:
    def __call__(self, prompt: str) -> _TokenizedPrompt:
        return _TokenizedPrompt(prompt)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _args_for_custom_image(dataset_path: Path) -> Namespace:
    return Namespace(
        dataset_name="custom_image",
        dataset_path=str(dataset_path),
        disable_shuffle=True,
        seed=0,
        num_prompts=2,
        custom_output_len=32,
        enable_multimodal_chat=False,
        request_id_prefix="req-",
        no_oversample=False,
    )


@pytest.mark.benchmark
def test_get_samples_custom_image_cli_path_supports_multi_image_and_content(
    tmp_path: Path,
) -> None:
    image_a = tmp_path / "chart_a.png"
    image_b = tmp_path / "chart_b.png"
    image_c = tmp_path / "chart_c.png"
    jsonl = tmp_path / "images.jsonl"
    _write_jsonl(
        jsonl,
        [
            {
                "prompt": "Compare the first two charts.",
                "image_files": [str(image_a), str(image_b)],
            },
            {
                "content": [
                    {"type": "text", "text": "Now compare "},
                    {"type": "image", "image": str(image_c)},
                ],
            },
        ],
    )

    samples = get_samples(_args_for_custom_image(jsonl), _Tokenizer())

    assert len(samples) == 2
    assert samples[0].request_id == "req-0"
    assert isinstance(samples[0].multi_modal_data, list)
    assert [part["image_url"]["url"] for part in samples[0].multi_modal_data] == [
        f"file://{image_a}",
        f"file://{image_b}",
    ]

    assert samples[1].request_id == "req-1"
    assert samples[1].multi_modal_data is None
    assert isinstance(samples[1].prompt, list)
    assert samples[1].prompt[0] == {"type": "text", "text": "Now compare "}
    assert samples[1].prompt[1]["image_url"]["url"] == f"file://{image_c}"


@pytest.mark.benchmark
def test_custom_image_dataset_uses_all_image_files(tmp_path: Path) -> None:
    image_a = tmp_path / "chart_a.png"
    image_b = tmp_path / "chart_b.png"
    jsonl = tmp_path / "images.jsonl"
    _write_jsonl(
        jsonl,
        [
            {
                "prompt": "Compare the charts.",
                "image_files": [str(image_a), str(image_b)],
            }
        ],
    )

    dataset = CustomImageDataset(dataset_path=str(jsonl), disable_shuffle=True)
    samples = dataset.sample(
        tokenizer=_Tokenizer(),
        num_requests=1,
        output_len=32,
    )

    assert len(samples) == 1
    sample = samples[0]
    assert sample.prompt == "Compare the charts."
    assert sample.prompt_len == 3
    assert isinstance(sample.multi_modal_data, list)
    assert [part["image_url"]["url"] for part in sample.multi_modal_data] == [
        f"file://{image_a}",
        f"file://{image_b}",
    ]


@pytest.mark.benchmark
def test_custom_image_dataset_preserves_interleaved_content_order(
    tmp_path: Path,
) -> None:
    image_a = tmp_path / "chart_a.png"
    image_b = tmp_path / "chart_b.png"
    jsonl = tmp_path / "images.jsonl"
    _write_jsonl(
        jsonl,
        [
            {
                "content": [
                    {"type": "text", "text": "Compare "},
                    {"type": "image", "image": str(image_a)},
                    {"type": "text", "text": " with "},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": str(image_b),
                            "detail": "low",
                        },
                    },
                ],
            }
        ],
    )

    dataset = CustomImageDataset(dataset_path=str(jsonl), disable_shuffle=True)
    samples = dataset.sample(
        tokenizer=_Tokenizer(),
        num_requests=1,
        output_len=32,
    )

    assert len(samples) == 1
    sample = samples[0]
    assert sample.multi_modal_data is None
    assert sample.prompt_len == 2
    assert isinstance(sample.prompt, list)
    assert [part["type"] for part in sample.prompt] == [
        "text",
        "image_url",
        "text",
        "image_url",
    ]
    assert sample.prompt[1]["image_url"]["url"] == f"file://{image_a}"
    assert sample.prompt[3]["image_url"] == {
        "url": f"file://{image_b}",
        "detail": "low",
    }

    request_input = RequestFuncInput(
        prompt=sample.prompt,
        api_url="http://localhost:8000/v1/chat/completions",
        prompt_len=sample.prompt_len,
        output_len=32,
        model="test-model",
    )
    assert _get_chat_content(request_input) == sample.prompt


@pytest.mark.benchmark
def test_custom_image_dataset_wraps_interleaved_content_for_multimodal_chat(
    tmp_path: Path,
) -> None:
    image = tmp_path / "chart.png"
    jsonl = tmp_path / "images.jsonl"
    _write_jsonl(
        jsonl,
        [
            {
                "content": [
                    {"type": "text", "text": "Describe "},
                    {"type": "image", "image": str(image)},
                ],
            }
        ],
    )

    dataset = CustomImageDataset(dataset_path=str(jsonl), disable_shuffle=True)
    samples = dataset.sample(
        tokenizer=_Tokenizer(),
        num_requests=1,
        output_len=32,
        enable_multimodal_chat=True,
    )

    sample = samples[0]
    assert sample.multi_modal_data is None
    assert sample.prompt == [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe "},
                {"type": "image_url", "image_url": {"url": f"file://{image}"}},
            ],
        }
    ]

    request_input = RequestFuncInput(
        prompt=sample.prompt,
        api_url="http://localhost:8000/v1/chat/completions",
        prompt_len=sample.prompt_len,
        output_len=32,
        model="test-model",
    )
    assert _get_chat_messages(request_input) == sample.prompt


@pytest.mark.benchmark
def test_custom_image_dataset_rejects_invalid_content_part(
    tmp_path: Path,
) -> None:
    jsonl = tmp_path / "images.jsonl"
    _write_jsonl(jsonl, [{"content": [{"type": "audio", "audio": "clip.wav"}]}])

    dataset = CustomImageDataset(dataset_path=str(jsonl), disable_shuffle=True)
    with pytest.raises(ValueError, match="type 'text', 'image', or 'image_url'"):
        dataset.sample(
            tokenizer=_Tokenizer(),
            num_requests=1,
            output_len=32,
        )
