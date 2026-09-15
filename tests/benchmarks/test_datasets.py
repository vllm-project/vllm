# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json
import tempfile
from pathlib import Path

import pytest
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from benchmarks.datasets import (
    SampleRequest,
    add_dataset_parser,
    get_samples,
    is_valid_sequence,
    sample_random_requests,
    sample_sharegpt_requests,
    sample_sonnet_requests,
)
from vllm.utils.argparse_utils import FlexibleArgumentParser


@pytest.fixture(scope="session")
def tokenizer() -> PreTrainedTokenizerBase:
    return AutoTokenizer.from_pretrained("openai-community/gpt2")


def test_sample_request_dataclass():
    req = SampleRequest(
        prompt="Hello world",
        prompt_len=2,
        expected_output_len=10,
        request_id="req_0",
    )
    assert req.prompt == "Hello world"
    assert req.prompt_len == 2
    assert req.expected_output_len == 10
    assert req.request_id == "req_0"
    assert req.multi_modal_data is None


def test_is_valid_sequence():
    # Valid sequence
    assert is_valid_sequence(
        prompt_len=100,
        output_len=50,
        min_len=4,
        max_prompt_len=1024,
        max_total_len=2048,
    )

    # Prompt too short
    assert not is_valid_sequence(prompt_len=2, output_len=50, min_len=4)

    # Output too short
    assert not is_valid_sequence(prompt_len=100, output_len=2, min_len=4)

    # Output too short but check skipped
    assert is_valid_sequence(
        prompt_len=100, output_len=2, min_len=4, skip_min_output_len_check=True
    )

    # Prompt too long
    assert not is_valid_sequence(prompt_len=2000, output_len=50, max_prompt_len=1024)

    # Combined sequence too long
    assert not is_valid_sequence(prompt_len=1500, output_len=1000, max_total_len=2048)


def test_sample_random_requests(tokenizer):
    requests = sample_random_requests(
        tokenizer=tokenizer,
        num_requests=5,
        input_len=32,
        output_len=16,
        random_seed=42,
        request_id_prefix="random_",
    )
    assert len(requests) == 5
    for i, req in enumerate(requests):
        assert isinstance(req, SampleRequest)
        assert req.request_id == f"random_{i}"
        assert req.expected_output_len == 16
        assert req.prompt_len > 0


def test_sample_sharegpt_requests(tokenizer):
    sharegpt_data = [
        {
            "conversations": [
                {"from": "human", "value": "What is the capital of France?"},
                {"from": "gpt", "value": "The capital of France is Paris."},
            ]
        },
        {
            "conversations": [
                {"from": "human", "value": "Tell me a joke about programming."},
                {
                    "from": "gpt",
                    "value": (
                        "Why do programmers prefer dark mode? Because light attracts"
                        " bugs."
                    ),
                },
            ]
        },
        {
            # Single turn conversation, should be filtered out
            "conversations": [
                {"from": "human", "value": "Single turn only"},
            ]
        },
    ]

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(sharegpt_data, f)
        temp_path = f.name

    try:
        requests = sample_sharegpt_requests(
            dataset_path=temp_path,
            tokenizer=tokenizer,
            num_requests=2,
            random_seed=42,
            request_id_prefix="sharegpt_",
        )
        assert len(requests) == 2
        assert requests[0].request_id == "sharegpt_0"
        assert requests[1].request_id == "sharegpt_1"
    finally:
        Path(temp_path).unlink(missing_ok=True)


def test_sample_sharegpt_requests_oversampling(tokenizer):
    sharegpt_data = [
        {
            "conversations": [
                {"from": "human", "value": "What is machine learning?"},
                {
                    "from": "gpt",
                    "value": (
                        "Machine learning is a field of artificial intelligence."
                    ),
                },
            ]
        },
    ]

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(sharegpt_data, f)
        temp_path = f.name

    try:
        requests = sample_sharegpt_requests(
            dataset_path=temp_path,
            tokenizer=tokenizer,
            num_requests=3,
            random_seed=42,
            request_id_prefix="sg_",
        )
        assert len(requests) == 3
        assert requests[0].request_id == "sg_0"
        assert requests[1].request_id == "sg_1"
        assert requests[2].request_id == "sg_2"
    finally:
        Path(temp_path).unlink(missing_ok=True)


def test_sample_sonnet_requests(tokenizer):
    if not getattr(tokenizer, "chat_template", None):
        tokenizer.chat_template = (
            "{% for message in messages %}{{ message['content'] }}{% endfor %}"
        )

    sonnet_content = """First line of poem
Second line of poem
Third line of poem
Fourth line of poem
Fifth line of poem
Sixth line of poem
Seventh line of poem
Eighth line of poem
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write(sonnet_content)
        temp_path = f.name

    try:
        requests = sample_sonnet_requests(
            dataset_path=temp_path,
            tokenizer=tokenizer,
            num_requests=2,
            input_len=50,
            output_len=10,
            random_seed=42,
            request_id_prefix="sonnet_",
        )

        assert len(requests) == 2
        assert requests[0].request_id == "sonnet_0"
    finally:
        Path(temp_path).unlink(missing_ok=True)



def test_add_dataset_parser_and_get_samples(tokenizer):
    parser = FlexibleArgumentParser()
    add_dataset_parser(parser)
    args = parser.parse_args(
        ["--dataset-name", "random", "--num-prompts", "3", "--seed", "42"]
    )
    requests = get_samples(args, tokenizer)
    assert len(requests) == 3
