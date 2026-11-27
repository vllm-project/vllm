# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json
from pathlib import Path

import pytest
from transformers import AutoTokenizer

from vllm.benchmarks.datasets import get_samples
from vllm.benchmarks.serve import add_cli_args
from vllm.utils.argparse_utils import FlexibleArgumentParser


@pytest.mark.benchmark
@pytest.mark.parametrize(
    "hf_name", ["Aeala/ShareGPT_Vicuna_unfiltered", "lmms-lab/LLaVA-OneVision-Data"]
)
def test_conversation_dataset_defaults_to_train_split(
    tmp_path: Path, hf_name: str
) -> None:
    """Omitting --hf-split must fall back to the train split instead of crashing."""
    prompts = [f"Question {i}: what is the capital of France?" for i in range(3)]
    with (tmp_path / "data.jsonl").open("w") as f:
        for prompt in prompts:
            turns = [
                {"from": "human", "value": prompt},
                {"from": "gpt", "value": "The capital of France is Paris."},
            ]
            f.write(json.dumps({"conversations": turns}) + "\n")
    parser = FlexibleArgumentParser()
    add_cli_args(parser)
    args = parser.parse_args(
        ["--backend", "openai-chat", "--dataset-name", "hf"]
        + ["--dataset-path", str(tmp_path), "--hf-name", hf_name, "--num-prompts", "3"]
    )
    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")

    samples = get_samples(args, tokenizer)

    assert sorted(s.prompt for s in samples) == prompts
