# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
import tempfile

import pytest
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from vllm.benchmarks.datasets import TxtSlicesDataset


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
def test_txt_slices(hf_tokenizer: PreTrainedTokenizerBase) -> None:
    # Write the text content to a temporary file
    # Use delete=False for Python 3.10 compatibility (delete_on_close is 3.12+)
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
        f.write(text_content)
        f.close()
        temp_file_path = f.name

    try:
        dataset = TxtSlicesDataset(dataset_path=temp_file_path)

        samples = dataset.sample(
            hf_tokenizer, num_requests=10, input_len=10, output_len=10
        )

        assert len(samples) == 10
        assert all(sample.prompt_len == 10 for sample in samples)
        assert all(sample.expected_output_len == 10 for sample in samples)

        for sample in samples:
            tokenized_prompt = hf_tokenizer(
                sample.prompt, add_special_tokens=True
            ).input_ids
            assert len(tokenized_prompt) == 10
    finally:
        os.unlink(f.name)
