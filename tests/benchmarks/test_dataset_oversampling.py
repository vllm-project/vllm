# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import argparse
import random
from dataclasses import replace
from unittest.mock import patch

import pytest
from datasets import Dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import TokenizersBackend

from vllm.benchmarks.datasets import RandomDataset, SampleRequest, get_samples


@pytest.mark.parametrize("prefix", ["", "bench-"])
@pytest.mark.parametrize(
    "filtered,no_oversample", [(True, False), (False, False), (True, True)]
)
def test_get_samples_oversampling_preserves_requests_with_unique_ids(
    prefix: str, filtered: bool, no_oversample: bool
) -> None:
    """Filtering must not make oversampling reuse an existing row's ID."""
    tokenizer = TokenizersBackend(
        tokenizer_object=Tokenizer(WordLevel({"[UNK]": 0}, unk_token="[UNK]")),
        unk_token="[UNK]",
    )
    rows = Dataset.from_list(
        [
            {
                "code": f"print({i})",
                "change_request": f"Change {i}",
                "norm_distance": distance,
            }
            for i, distance in enumerate([0.5, 1.0, 1.0, 0.5, 0.5])
        ]
    )
    args = argparse.Namespace(
        dataset_name="hf",
        dataset_path="vdaita/edit_5k_char",
        hf_name=None,
        hf_subset=None,
        hf_split=None,
        disable_shuffle=True,
        num_prompts=6,
        hf_output_len=32,
        blazedit_min_distance=0.1 if filtered else 0.0,
        blazedit_max_distance=0.9 if filtered else 1.0,
        seed=7,
        no_stream=True,
        backend="openai-chat",
        trust_remote_code=False,
        enable_multimodal_chat=False,
        request_id_prefix=prefix,
        no_oversample=True,
        skip_chat_template=True,
    )
    with patch("vllm.benchmarks.datasets.datasets.load_dataset", return_value=rows):
        originals = get_samples(args, tokenizer)
        args.no_oversample = no_oversample
        samples = get_samples(args, tokenizer)

    original_indices = [0, 3, 4] if filtered else list(range(5))
    assert [sample.request_id for sample in originals] == [
        prefix + str(i) for i in original_indices
    ]
    additional_indices = [] if no_oversample else list(range(5, 5 + 6 - len(originals)))
    rng = random.Random(args.seed)
    expected = originals + [
        replace(rng.choice(originals), request_id=prefix + str(i))
        for i in additional_indices
    ]
    assert samples == expected
    assert len({sample.request_id for sample in samples}) == len(samples)


@pytest.mark.parametrize("num_requests", [2, 4])
def test_oversampling_rejects_duplicate_input_ids(num_requests: int) -> None:
    requests = [
        SampleRequest(prompt="first", prompt_len=1, request_id="bench-0"),
        SampleRequest(prompt="second", prompt_len=1, request_id="bench-0"),
    ]
    with pytest.raises(ValueError, match="Duplicate request_id"):
        RandomDataset(random_seed=7).maybe_oversample_requests(
            requests, num_requests=num_requests, request_id_prefix="bench-"
        )
