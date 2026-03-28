# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from vllm.benchmarks.datasets import ShareGPTDataset


class DummyTokenizer:

    def __call__(self, text: str) -> SimpleNamespace:
        return SimpleNamespace(input_ids=list(range(len(text.split()))))


@pytest.mark.benchmark
@pytest.mark.skip_global_cleanup
def test_sharegpt_sample_explains_fixed_output_length_constraint(
    tmp_path: Path,
) -> None:
    dataset_path = tmp_path / "sharegpt.json"
    dataset_path.write_text(
        json.dumps([
            {
                "conversations": [
                    {
                        "from": "human",
                        "value": "this prompt has enough words to be valid alone",
                    },
                    {
                        "from": "gpt",
                        "value": "this completion also has enough words",
                    },
                ]
            }
        ]),
        encoding="utf-8",
    )

    dataset = ShareGPTDataset(
        dataset_path=str(dataset_path),
        random_seed=0,
        disable_shuffle=True,
    )

    with pytest.raises(
        ValueError,
        match=r"output_len=2048.*prompt_len \+ output_len <= 2048",
    ):
        dataset.sample(
            tokenizer=DummyTokenizer(),
            num_requests=1,
            output_len=2048,
        )


@pytest.mark.benchmark
@pytest.mark.skip_global_cleanup
def test_sharegpt_sample_still_oversamples_valid_requests(tmp_path: Path) -> None:
    dataset_path = tmp_path / "sharegpt.json"
    dataset_path.write_text(
        json.dumps([
            {
                "conversations": [
                    {
                        "from": "human",
                        "value": "this prompt has enough words",
                    },
                    {
                        "from": "gpt",
                        "value": "this completion also has enough words",
                    },
                ]
            }
        ]),
        encoding="utf-8",
    )

    dataset = ShareGPTDataset(
        dataset_path=str(dataset_path),
        random_seed=0,
        disable_shuffle=True,
    )

    samples = dataset.sample(
        tokenizer=DummyTokenizer(),
        num_requests=3,
        output_len=16,
    )

    assert len(samples) == 3
    assert len({sample.request_id for sample in samples}) == 3
