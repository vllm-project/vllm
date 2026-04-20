# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json
from pathlib import Path

import pytest
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from vllm.benchmarks.datasets import CustomDataset
from vllm.benchmarks.datasets.create_txt_slices_dataset import create_txt_slices_jsonl


@pytest.fixture(scope="session")
def hf_tokenizer() -> PreTrainedTokenizerBase:
    # Use a small, commonly available tokenizer
    return AutoTokenizer.from_pretrained("gpt2")


text_content = """
Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor
incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud
exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.
Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat
nulla pariatur. Excepteur sint occaecat cupidatat non proident,
sunt in culpa qui officia deserunt mollit anim id est laborum.
"""


@pytest.mark.benchmark
def test_create_txt_slices_jsonl(
    hf_tokenizer: PreTrainedTokenizerBase, tmp_path: Path
) -> None:
    """Test that create_txt_slices_jsonl produces valid JSONL for CustomDataset."""
    txt_path = tmp_path / "input.txt"
    jsonl_path = tmp_path / "input.txt.jsonl"

    txt_path.write_text(text_content)

    create_txt_slices_jsonl(
        input_path=str(txt_path),
        output_path=str(jsonl_path),
        tokenizer_name="gpt2",
        num_prompts=10,
        input_len=10,
        output_len=10,
    )

    # Verify the JSONL file is valid and has the expected structure
    records = [json.loads(line) for line in jsonl_path.read_text().splitlines()]

    assert len(records) == 10
    for record in records:
        assert "prompt" in record
        assert "output_tokens" in record
        assert isinstance(record["prompt"], str)
        assert record["output_tokens"] == 10

    # Verify the JSONL file can be loaded by CustomDataset
    dataset = CustomDataset(dataset_path=str(jsonl_path))
    samples = dataset.sample(
        tokenizer=hf_tokenizer,
        num_requests=10,
        output_len=10,
        skip_chat_template=True,
    )

    assert len(samples) == 10
    assert all(sample.expected_output_len == 10 for sample in samples)
