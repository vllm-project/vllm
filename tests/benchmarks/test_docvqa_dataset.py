# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import argparse
from unittest.mock import patch

import pytest
from PIL import Image
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from vllm.benchmarks.datasets import DocVQADataset, get_samples

_ROWS = [
    {
        "questionId": str(i),
        "question": f"  What is the total on page {i}?  ",
        "image": Image.new("RGB", (64, 80), "white"),
        "answers": [str(i)],
    }
    for i in range(3)
]


@pytest.fixture(scope="session")
def hf_tokenizer() -> PreTrainedTokenizerBase:
    return AutoTokenizer.from_pretrained("openai-community/gpt2")


def _dataset(**kwargs) -> DocVQADataset:
    with patch(
        "vllm.benchmarks.datasets.datasets.load_dataset", return_value=list(_ROWS)
    ) as load:
        ds = DocVQADataset(
            dataset_path="lmms-lab/DocVQA",
            dataset_split="validation",
            dataset_subset="DocVQA",
            no_stream=True,
            disable_shuffle=True,
            **kwargs,
        )
    load.assert_called_once()
    assert load.call_args.kwargs["name"] == "DocVQA"
    return ds


def test_sample_plain_prompt(hf_tokenizer):
    reqs = _dataset().sample(hf_tokenizer, num_requests=2, no_oversample=True)
    assert len(reqs) == 2
    for i, r in enumerate(reqs):
        assert r.prompt == (
            f"What is the total on page {i}?"
            "\nAnswer the question using a single word or phrase."
        )
        assert r.prompt_len == len(hf_tokenizer(r.prompt).input_ids)
        assert r.expected_output_len == DocVQADataset.DEFAULT_OUTPUT_LEN
        assert r.multi_modal_data["type"] == "image_url"
        assert r.multi_modal_data["image_url"]["url"].startswith("data:image/")


def test_sample_chat_prompt_and_output_len(hf_tokenizer):
    reqs = _dataset().sample(
        hf_tokenizer,
        num_requests=1,
        output_len=7,
        enable_multimodal_chat=True,
        no_oversample=True,
    )
    assert len(reqs) == 1
    r = reqs[0]
    assert r.expected_output_len == 7
    assert r.multi_modal_data is None
    content = r.prompt[0]["content"]
    assert [c["type"] for c in content] == ["text", "image_url"]


def test_get_samples_defaults(hf_tokenizer):
    args = argparse.Namespace(
        dataset_name="hf",
        dataset_path="lmms-lab/DocVQA",
        hf_name=None,
        hf_split=None,
        hf_subset=None,
        hf_output_len=None,
        num_prompts=2,
        seed=0,
        no_stream=True,
        no_oversample=True,
        disable_shuffle=True,
        trust_remote_code=False,
        request_id_prefix="",
        backend="openai-chat",
        skip_chat_template=False,
        enable_multimodal_chat=False,
    )
    with patch(
        "vllm.benchmarks.datasets.datasets.load_dataset", return_value=list(_ROWS)
    ) as load:
        reqs = get_samples(args, hf_tokenizer)
    assert args.hf_split == "validation"
    assert args.hf_subset == "DocVQA"
    assert load.call_args.kwargs["split"] == "validation"
    assert len(reqs) == 2
