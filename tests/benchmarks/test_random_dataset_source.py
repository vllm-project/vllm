# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json
import os
import tempfile
from typing import Any

import pytest
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from vllm.benchmarks.datasets import RandomDataset, SampleRequest


@pytest.fixture(scope="session")
def hf_tokenizer() -> PreTrainedTokenizerBase:
    return AutoTokenizer.from_pretrained("openai-community/gpt2")


@pytest.fixture
def sharegpt_file(tmp_path) -> str:
    """Create a temporary ShareGPT-format JSON file."""
    data = [
        {
            "conversations": [
                {"from": "human", "value": "Hello, how are you today?"},
                {"from": "gpt", "value": "I am fine, thank you!"},
            ]
        },
        {
            "conversations": [
                {"from": "human", "value": "What is the weather like?"},
                {"from": "gpt", "value": "It is sunny today."},
            ]
        },
        {
            "conversations": [
                {"from": "human", "value": "Tell me a joke."},
                {"from": "gpt", "value": "Why did the chicken cross the road?"},
            ]
        },
    ]
    filepath = tmp_path / "sharegpt.json"
    filepath.write_text(json.dumps(data), encoding="utf-8")
    return str(filepath)


@pytest.fixture
def jsonl_file(tmp_path) -> str:
    """Create a temporary JSONL file with prompt field."""
    lines = [
        json.dumps({"prompt": "The quick brown fox jumps over the lazy dog."}),
        json.dumps({"prompt": "Machine learning is a subset of artificial intelligence."}),
        json.dumps({"prompt": "Python is a popular programming language."}),
    ]
    filepath = tmp_path / "custom.jsonl"
    filepath.write_text("\n".join(lines), encoding="utf-8")
    return str(filepath)


@pytest.fixture
def txt_file(tmp_path) -> str:
    """Create a temporary plain text file."""
    lines = [
        "To be or not to be, that is the question.",
        "All that glitters is not gold.",
        "The only thing we have to fear is fear itself.",
    ]
    filepath = tmp_path / "plain.txt"
    filepath.write_text("\n".join(lines), encoding="utf-8")
    return str(filepath)


def _collect_samples(
    dataset: RandomDataset,
    tokenizer: PreTrainedTokenizerBase,
    num_requests: int = 4,
    input_len: int = 32,
    output_len: int = 16,
    **kwargs: Any,
) -> list[SampleRequest]:
    return dataset.sample(
        tokenizer=tokenizer,
        num_requests=num_requests,
        input_len=input_len,
        output_len=output_len,
        **kwargs,
    )


@pytest.mark.benchmark
def test_random_dataset_with_sharegpt_source(
    hf_tokenizer: PreTrainedTokenizerBase, sharegpt_file: str
) -> None:
    """Test that random dataset samples tokens from ShareGPT file."""
    ds = RandomDataset(random_seed=42, dataset_path=sharegpt_file)
    samples = _collect_samples(ds, hf_tokenizer, num_requests=4, input_len=32)

    assert len(samples) == 4
    for s in samples:
        assert s.prompt_len > 0
        assert s.expected_output_len == 16
        # prompt should contain real text, not random token IDs
        assert len(s.prompt) > 0


@pytest.mark.benchmark
def test_random_dataset_with_jsonl_source(
    hf_tokenizer: PreTrainedTokenizerBase, jsonl_file: str
) -> None:
    """Test that random dataset samples tokens from JSONL file."""
    ds = RandomDataset(random_seed=42, dataset_path=jsonl_file)
    samples = _collect_samples(ds, hf_tokenizer, num_requests=4, input_len=32)

    assert len(samples) == 4
    for s in samples:
        assert s.prompt_len > 0


@pytest.mark.benchmark
def test_random_dataset_with_txt_source(
    hf_tokenizer: PreTrainedTokenizerBase, txt_file: str
) -> None:
    """Test that random dataset samples tokens from plain text file."""
    ds = RandomDataset(random_seed=42, dataset_path=txt_file)
    samples = _collect_samples(ds, hf_tokenizer, num_requests=4, input_len=32)

    assert len(samples) == 4
    for s in samples:
        assert s.prompt_len > 0


@pytest.mark.benchmark
def test_random_dataset_with_dataset_path_respects_input_len(
    hf_tokenizer: PreTrainedTokenizerBase, sharegpt_file: str
) -> None:
    """Test that sampled prompts respect the requested input_len."""
    ds = RandomDataset(random_seed=42, dataset_path=sharegpt_file)
    target_len = 64
    samples = _collect_samples(
        ds, hf_tokenizer, num_requests=4, input_len=target_len
    )

    assert len(samples) == 4
    for s in samples:
        # prompt_len should be close to target_len (may differ slightly
        # due to tokenization roundtrip)
        assert abs(s.prompt_len - target_len) <= 5


@pytest.mark.benchmark
def test_random_dataset_without_dataset_path_unchanged(
    hf_tokenizer: PreTrainedTokenizerBase,
) -> None:
    """Test that without dataset_path, behavior is unchanged (random tokens)."""
    ds = RandomDataset(random_seed=42)
    samples = _collect_samples(ds, hf_tokenizer, num_requests=4, input_len=32)

    assert len(samples) == 4
    for s in samples:
        assert s.prompt_len > 0
        assert s.expected_output_len == 16


@pytest.mark.benchmark
def test_random_dataset_same_seed_same_output_with_source(
    hf_tokenizer: PreTrainedTokenizerBase, sharegpt_file: str
) -> None:
    """Test that same seed produces same output when using dataset_path."""
    ds_a = RandomDataset(random_seed=123, dataset_path=sharegpt_file)
    ds_b = RandomDataset(random_seed=123, dataset_path=sharegpt_file)

    a = _collect_samples(ds_a, hf_tokenizer, num_requests=4, input_len=32)
    b = _collect_samples(ds_b, hf_tokenizer, num_requests=4, input_len=32)

    assert [s.prompt for s in a] == [s.prompt for s in b]


@pytest.mark.benchmark
def test_random_dataset_unsupported_extension(
    hf_tokenizer: PreTrainedTokenizerBase, tmp_path: Any
) -> None:
    """Test that unsupported file extension raises ValueError."""
    filepath = tmp_path / "data.csv"
    filepath.write_text("a,b,c", encoding="utf-8")

    ds = RandomDataset(random_seed=42, dataset_path=str(filepath))
    with pytest.raises(ValueError, match="Unsupported dataset file extension"):
        _collect_samples(ds, hf_tokenizer, num_requests=1, input_len=32)


@pytest.mark.benchmark
def test_random_dataset_empty_sharegpt_file(
    hf_tokenizer: PreTrainedTokenizerBase, tmp_path: Any
) -> None:
    """Test that empty ShareGPT file raises ValueError."""
    filepath = tmp_path / "empty.json"
    filepath.write_text("[]", encoding="utf-8")

    ds = RandomDataset(random_seed=42, dataset_path=str(filepath))
    with pytest.raises(ValueError, match="No valid data found"):
        _collect_samples(ds, hf_tokenizer, num_requests=1, input_len=32)
