# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import argparse
import json
from pathlib import Path

import pytest
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from vllm.benchmarks.datasets import get_samples


@pytest.fixture(scope="session")
def hf_tokenizer() -> PreTrainedTokenizerBase:
    return AutoTokenizer.from_pretrained("gpt2")


def _write_jsonl(path: Path, n_rows: int) -> None:
    with path.open("w") as f:
        for i in range(n_rows):
            f.write(json.dumps({"prompt": f"row {i}: unique prompt content."}) + "\n")


def _args_for_custom(dataset_path: str, seed: int) -> argparse.Namespace:
    return argparse.Namespace(
        dataset_name="custom",
        dataset_path=dataset_path,
        disable_shuffle=False,
        num_prompts=30,
        custom_output_len=32,
        skip_chat_template=True,
        no_oversample=False,
        seed=seed,
        request_id_prefix="",
    )


@pytest.mark.benchmark
def test_custom_dataset_seed_propagates(
    hf_tokenizer: PreTrainedTokenizerBase, tmp_path: Path
) -> None:
    """--seed must control the CustomDataset shuffle used by get_samples.

    Without the fix, CustomDataset was instantiated without random_seed,
    so its load-time shuffle always used DEFAULT_SEED=0 regardless of
    args.seed, causing every run with --dataset-name custom to pick the
    same subset of rows from a larger file.
    """
    jsonl = tmp_path / "data.jsonl"
    _write_jsonl(jsonl, n_rows=60)

    samples_a = get_samples(_args_for_custom(str(jsonl), seed=0), hf_tokenizer)
    samples_b = get_samples(_args_for_custom(str(jsonl), seed=42), hf_tokenizer)

    prompts_a = {s.prompt for s in samples_a}
    prompts_b = {s.prompt for s in samples_b}

    assert len(prompts_a) == 30
    assert len(prompts_b) == 30
    assert prompts_a != prompts_b


@pytest.mark.benchmark
def test_custom_dataset_same_seed_is_deterministic(
    hf_tokenizer: PreTrainedTokenizerBase, tmp_path: Path
) -> None:
    """Same --seed must yield the same CustomDataset subset."""
    jsonl = tmp_path / "data.jsonl"
    _write_jsonl(jsonl, n_rows=60)

    samples_a = get_samples(_args_for_custom(str(jsonl), seed=7), hf_tokenizer)
    samples_b = get_samples(_args_for_custom(str(jsonl), seed=7), hf_tokenizer)

    prompts_a = [s.prompt for s in samples_a]
    prompts_b = [s.prompt for s in samples_b]

    assert prompts_a == prompts_b
