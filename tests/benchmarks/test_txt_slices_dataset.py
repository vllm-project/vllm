# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json
import os
import tempfile

import pytest
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from vllm.benchmarks.create_txt_slices_dataset import create_txt_slices_jsonl
from vllm.benchmarks.datasets import CustomDataset


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
def test_create_txt_slices_jsonl(hf_tokenizer: PreTrainedTokenizerBase) -> None:
    """Test that create_txt_slices_jsonl produces valid JSONL for CustomDataset."""
    # Write the text content to a temporary file
    # Use delete=False for Python 3.10 compatibility (delete_on_close is 3.12+)
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
        f.write(text_content)
        f.close()
        txt_path = f.name

    jsonl_path = txt_path + ".jsonl"

    try:
        create_txt_slices_jsonl(
            input_path=txt_path,
            output_path=jsonl_path,
            tokenizer_name="gpt2",
            num_prompts=10,
            input_len=10,
            output_len=10,
        )

        # Verify the JSONL file is valid and has the expected structure
        with open(jsonl_path) as jf:
            records = [json.loads(line) for line in jf]

        assert len(records) == 10
        for record in records:
            assert "prompt" in record
            assert "output_tokens" in record
            assert isinstance(record["prompt"], str)
            assert record["output_tokens"] == 10

        # Verify the JSONL file can be loaded by CustomDataset
        dataset = CustomDataset(dataset_path=jsonl_path)
        samples = dataset.sample(
            tokenizer=hf_tokenizer,
            num_requests=10,
            output_len=10,
            skip_chat_template=True,
        )

        assert len(samples) == 10
        assert all(sample.expected_output_len == 10 for sample in samples)
    finally:
        os.unlink(txt_path)
        if os.path.exists(jsonl_path):
            os.unlink(jsonl_path)
