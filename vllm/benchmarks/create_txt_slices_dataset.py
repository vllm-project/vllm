# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Convert a plain-text file (local path or URL) into a JSONL dataset
compatible with ``CustomDataset`` (``--dataset-name custom``).

Each line of the output JSONL contains a ``prompt`` (decoded from a random
slice of the tokenized source text) and an ``output_tokens`` count.

Usage
-----
::

    python -m vllm.benchmarks.create_txt_slices_dataset \\
        --input  sonnet.txt \\
        --output sonnet_dataset.jsonl \\
        --tokenizer gpt2 \\
        --num-prompts 1000 \\
        --input-len 1024 \\
        --output-len 128

The resulting JSONL file can then be used with the serving benchmark::

    python -m vllm.benchmarks.serve \\
        --dataset-name custom \\
        --dataset-path sonnet_dataset.jsonl \\
        ...
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import urllib.request

import numpy as np
from transformers import AutoTokenizer

from vllm.benchmarks.shared import get_sampling_params

logger = logging.getLogger(__name__)


def load_text(path: str) -> str:
    """Load text from a local file or URL."""
    if path.startswith(("http://", "https://")):
        with urllib.request.urlopen(path) as response:
            return response.read().decode("utf-8")
    with open(path, encoding="utf-8") as f:
        return f.read()


def create_txt_slices_jsonl(
    *,
    input_path: str,
    output_path: str,
    tokenizer_name: str,
    num_prompts: int,
    input_len: int,
    output_len: int,
    range_ratio: float = 0.0,
    input_range_ratio: float | None = None,
    output_range_ratio: float | None = None,
    seed: int = 0,
    trust_remote_code: bool = False,
) -> None:
    """Read *input_path*, slice it into prompts, and write JSONL to
    *output_path*."""

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name, trust_remote_code=trust_remote_code
    )

    text = load_text(input_path)
    if not text:
        raise ValueError("The text file is empty and cannot be sampled from.")

    token_ids = tokenizer(text, add_special_tokens=False).input_ids
    if not token_ids:
        raise ValueError("Tokenizing the text produced zero tokens; cannot sample.")

    resolved_input_rr = (
        input_range_ratio if input_range_ratio is not None else range_ratio
    )
    resolved_output_rr = (
        output_range_ratio if output_range_ratio is not None else range_ratio
    )

    rng_np = np.random.default_rng(seed)
    rng_py = random.Random(seed)

    input_lens, output_lens, _ = get_sampling_params(
        rng_np,
        num_prompts,
        resolved_input_rr,
        resolved_output_rr,
        input_len,
        output_len,
        tokenizer,
    )

    num_available_tokens = len(token_ids)

    records: list[dict[str, object]] = []
    for i in range(num_prompts):
        req_input_len = int(input_lens[i])
        req_output_len = int(output_lens[i])

        # Randomly select a start position and slice with cycling
        start_pos = rng_py.randint(0, num_available_tokens - 1)
        prompt_token_ids = [
            token_ids[(start_pos + j) % num_available_tokens]
            for j in range(req_input_len)
        ]
        prompt = tokenizer.decode(prompt_token_ids, skip_special_tokens=False)

        records.append({"prompt": prompt, "output_tokens": req_output_len})

    with open(output_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    logger.info(
        "Wrote %d prompts to %s",
        len(records),
        output_path,
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Convert a plain-text file into a JSONL dataset "
        "for CustomDataset (--dataset-name custom).",
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path or URL to the source text file.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path for the output JSONL file.",
    )
    parser.add_argument(
        "--tokenizer",
        required=True,
        help="HuggingFace tokenizer name or path.",
    )
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=1000,
        help="Number of prompt samples to generate (default: 1000).",
    )
    parser.add_argument(
        "--input-len",
        type=int,
        default=1024,
        help="Target number of input tokens per prompt (default: 1024).",
    )
    parser.add_argument(
        "--output-len",
        type=int,
        default=128,
        help="Target number of output tokens per prompt (default: 128).",
    )
    parser.add_argument(
        "--range-ratio",
        type=float,
        default=0.0,
        help="Range ratio for both input and output length sampling "
        "(default: 0.0). Must be in [0, 1).",
    )
    parser.add_argument(
        "--input-range-ratio",
        type=float,
        default=None,
        help="Range ratio for input length sampling. "
        "Overrides --range-ratio for inputs.",
    )
    parser.add_argument(
        "--output-range-ratio",
        type=float,
        default=None,
        help="Range ratio for output length sampling. "
        "Overrides --range-ratio for outputs.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducibility (default: 0).",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Trust remote code from HuggingFace.",
    )

    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO)

    create_txt_slices_jsonl(
        input_path=args.input,
        output_path=args.output,
        tokenizer_name=args.tokenizer,
        num_prompts=args.num_prompts,
        input_len=args.input_len,
        output_len=args.output_len,
        range_ratio=args.range_ratio,
        input_range_ratio=args.input_range_ratio,
        output_range_ratio=args.output_range_ratio,
        seed=args.seed,
        trust_remote_code=args.trust_remote_code,
    )


if __name__ == "__main__":
    main()
