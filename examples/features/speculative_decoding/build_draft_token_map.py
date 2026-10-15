# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build a frequency-ranked draft vocabulary for `draft_token_map`.

Counts token frequencies over a text corpus and keeps the most frequent ids,
plus every special/added token and, optionally, all ids below a floor. The
corpus should look like the traffic you serve: ideally text the target model
generated, otherwise chat, code and tool-call data. The last `--holdout`
fraction of documents is kept out of the counts and used to report how much of
unseen text the list covers.

Example:
    python examples/features/speculative_decoding/build_draft_token_map.py \\
        --tokenizer Qwen/Qwen3-8B \\
        --dataset HuggingFaceH4/ultrachat_200k:train_sft \\
        --text-file my_outputs.jsonl \\
        --num-tokens 32768 --id-floor 4096 --output draft_vocab.pt

    vllm serve Qwen/Qwen3-8B --speculative-config \\
        '{"method": "mtp", "num_speculative_tokens": 3,
          "draft_token_map": "draft_vocab.pt"}'

The `.pt` output uses SGLang's `--speculative-token-map` format; a `.json`
output is also accepted by vLLM.

"""

import argparse
import json
from collections import Counter
from collections.abc import Iterator

import torch
from transformers import AutoTokenizer


def _row_text(row: dict, tokenizer) -> str:
    for key in ("messages", "conversations"):
        turns = row.get(key)
        if isinstance(turns, list) and turns and isinstance(turns[0], dict):
            messages = [
                {
                    "role": t.get("role") or t.get("from", "user"),
                    "content": str(t.get("content") or t.get("value") or ""),
                }
                for t in turns
            ]
            if tokenizer.chat_template:
                return tokenizer.apply_chat_template(messages, tokenize=False)
            return "\n".join(m["content"] for m in messages)
    return "\n".join(str(v) for v in row.values() if isinstance(v, str))


def _iter_texts(args, tokenizer) -> Iterator[str]:
    for spec in args.dataset:
        from datasets import load_dataset

        name, _, split = spec.partition(":")
        ds = load_dataset(name, split=split or "train", streaming=True)
        for i, row in enumerate(ds):
            if i >= args.max_docs:
                break
            yield _row_text(row, tokenizer)
    for path in args.text_file:
        with open(path) as f:
            if path.endswith(".jsonl"):
                for line in f:
                    row = json.loads(line)
                    yield _row_text(row, tokenizer) if isinstance(row, dict) else row
            else:
                yield f.read()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument(
        "--dataset",
        action="append",
        default=[],
        help="HF dataset as name[:split]; may be repeated.",
    )
    parser.add_argument(
        "--text-file",
        action="append",
        default=[],
        help="Plain text, or .jsonl of strings or chat rows; may be repeated.",
    )
    parser.add_argument("--max-docs", type=int, default=20000)
    parser.add_argument("--num-tokens", type=int, default=32768)
    parser.add_argument(
        "--id-floor",
        type=int,
        default=0,
        help="Always include ids below this value (BPE merges are roughly "
        "frequency ordered).",
    )
    parser.add_argument("--holdout", type=float, default=0.05)
    parser.add_argument("--output", required=True, help=".pt or .json")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    texts = list(_iter_texts(args, tokenizer))
    if not texts:
        parser.error("No text: pass --dataset and/or --text-file.")
    num_held_out = int(len(texts) * args.holdout)
    train = texts[: len(texts) - num_held_out]
    held_out = texts[len(texts) - num_held_out :]

    counts: Counter[int] = Counter()
    for text in train:
        counts.update(tokenizer.encode(text, add_special_tokens=False))

    keep = set(range(min(args.id_floor, len(tokenizer))))
    keep.update(tokenizer.all_special_ids)
    keep.update(tokenizer.get_added_vocab().values())
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

    print(
        f"Wrote {len(ids)} ids ({len(counts)} distinct ids seen in "
        f"{sum(counts.values())} tokens of {len(train)} documents) to {args.output}"
    )
    if held_out:
        held = [t for text in held_out for t in tokenizer.encode(text)]
        covered = sum(t in keep for t in held) / max(len(held), 1)
        print(f"Held-out coverage: {100 * covered:.2f}% of {len(held)} tokens")


if __name__ == "__main__":
    main()
