# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression tests for HuggingFace benchmark datasets loaded without a split.

See https://github.com/vllm-project/vllm/issues/49480: when ``--hf-split`` is
omitted, ``load_dataset`` returns a split *mapping* (``DatasetDict`` /
``IterableDatasetDict``). Iterating that mapping yields split names (``str``)
rather than examples, so sampling used to raise
``TypeError: string indices must be integers``.
"""

import logging
from unittest.mock import patch

import pytest
from datasets import Dataset, DatasetDict, IterableDatasetDict
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from vllm.benchmarks.datasets import datasets as datasets_mod
from vllm.benchmarks.datasets.datasets import ConversationDataset


@pytest.fixture(scope="session")
def hf_tokenizer() -> PreTrainedTokenizerBase:
    return AutoTokenizer.from_pretrained("openai-community/gpt2")


def _conversation_rows(n: int) -> list[dict]:
    return [
        {
            "conversations": [
                {"from": "human", "value": f"question {i}"},
                {"from": "gpt", "value": f"answer {i} with enough content"},
            ]
        }
        for i in range(n)
    ]


def _dataset_dict(n: int, splits: tuple[str, ...] = ("train",)) -> DatasetDict:
    return DatasetDict(
        {split: Dataset.from_list(_conversation_rows(n)) for split in splits}
    )


def _iterable_dataset_dict(
    n: int, splits: tuple[str, ...] = ("train",)
) -> IterableDatasetDict:
    return IterableDatasetDict(
        {
            split: Dataset.from_list(_conversation_rows(n)).to_iterable_dataset()
            for split in splits
        }
    )


def _make_dataset(mock_return, dataset_split=None) -> ConversationDataset:
    """Build a ConversationDataset while mocking the HuggingFace download."""
    with patch.object(datasets_mod, "load_dataset", return_value=mock_return):
        return ConversationDataset(
            dataset_path="Aeala/ShareGPT_Vicuna_unfiltered",
            dataset_split=dataset_split,
            random_seed=0,
        )


@pytest.mark.benchmark
@pytest.mark.parametrize("factory", [_dataset_dict, _iterable_dataset_dict])
def test_missing_split_does_not_crash_sampling(
    hf_tokenizer: PreTrainedTokenizerBase, factory
) -> None:
    """Omitting --hf-split must not crash sampling (issue #49480).

    Covers both the eager (``DatasetDict``) and streaming
    (``IterableDatasetDict``) mappings ``load_dataset`` can return.
    """
    dataset = _make_dataset(factory(16))

    requests = dataset.sample(
        tokenizer=hf_tokenizer,
        num_requests=5,
        request_id_prefix="req-",
        output_len=32,
    )

    assert len(requests) == 5
    assert all(req.prompt for req in requests)


@pytest.mark.benchmark
def test_missing_split_prefers_train(
    hf_tokenizer: PreTrainedTokenizerBase, caplog
) -> None:
    """With several splits, "train" is chosen even when it isn't first."""
    mapping = DatasetDict(
        {
            "validation": Dataset.from_list(_conversation_rows(4)),
            "train": Dataset.from_list(_conversation_rows(16)),
        }
    )

    with caplog.at_level(logging.WARNING):
        dataset = _make_dataset(mapping)

    # The mapping was collapsed to a concrete split, not left as a dict.
    assert not isinstance(dataset.data, (DatasetDict, IterableDatasetDict))
    # "train" (16 rows) was chosen over the first key "validation" (4 rows).
    assert len(dataset.data) == 16
    assert "defaulting to 'train'" in caplog.text


@pytest.mark.benchmark
def test_missing_split_without_train_uses_first(
    hf_tokenizer: PreTrainedTokenizerBase,
) -> None:
    """When there is no "train" split, fall back to the first available one."""
    mapping = _dataset_dict(6, splits=("test",))

    dataset = _make_dataset(mapping)

    assert not isinstance(dataset.data, (DatasetDict, IterableDatasetDict))
    assert len(dataset.data) == 6


@pytest.mark.benchmark
def test_explicit_split_is_left_untouched(
    hf_tokenizer: PreTrainedTokenizerBase,
) -> None:
    """A concrete --hf-split yields a Dataset (not a mapping); leave it as-is."""
    single_split = Dataset.from_list(_conversation_rows(12))

    dataset = _make_dataset(single_split, dataset_split="train")

    assert len(dataset.data) == 12
    requests = dataset.sample(
        tokenizer=hf_tokenizer,
        num_requests=4,
        request_id_prefix="",
        output_len=8,
    )
    assert len(requests) == 4
