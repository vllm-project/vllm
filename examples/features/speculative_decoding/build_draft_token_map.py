# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build a frequency-ranked draft vocabulary for `draft_token_map`.

Counts token frequencies over text files and keeps the most frequent ids plus
every special and added token. The text should look like the traffic you
serve, ideally text the target model generated.

Example:
    python examples/features/speculative_decoding/build_draft_token_map.py \\
        --tokenizer Qwen/Qwen3.5-0.8B --text-file my_outputs.jsonl \\
        --num-tokens 32768 --output draft_vocab.pt

    vllm serve Qwen/Qwen3.5-0.8B --speculative-config \\
        '{"method": "mtp", "num_speculative_tokens": 3,
          "draft_token_map": "draft_vocab.pt"}'

The `.pt` output uses SGLang's `--speculative-token-map` format; a `.json`
output is also accepted by vLLM.

"""

import argparse
import json
from collections import Counter

import torch
from transformers import AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument(
        "--text-file",
        action="append",
        required=True,
        help="Plain text, or .jsonl with one string per line; may be repeated.",
    )
    parser.add_argument("--num-tokens", type=int, default=32768)
    parser.add_argument("--output", required=True, help=".pt or .json")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    counts: Counter[int] = Counter()
    for path in args.text_file:
        with open(path) as f:
            texts = (
                [json.loads(line) for line in f]
                if path.endswith(".jsonl")
                else [f.read()]
            )
        for text in texts:
            counts.update(tokenizer.encode(text, add_special_tokens=False))

    keep = set(tokenizer.all_special_ids) | set(tokenizer.get_added_vocab().values())
    for token_id, _ in counts.most_common():
        if len(keep) >= args.num_tokens:
            break
        keep.add(token_id)
    ids = sorted(keep)

    if args.output.endswith(".json"):
        with open(args.output, "w") as f:
            json.dump(ids, f)
    else:
        torch.save(ids, args.output)
    print(f"Wrote {len(ids)} ids ({len(counts)} distinct ids seen) to {args.output}")


if __name__ == "__main__":
    main()
