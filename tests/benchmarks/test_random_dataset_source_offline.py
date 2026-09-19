# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Offline regression tests for RandomDataset --dataset-path source sampling.

These complement tests/benchmarks/test_random_dataset_source.py (which needs a
real HF tokenizer download) by using a tiny deterministic fake tokenizer, so
they run with no network. They target the specific defects fixed alongside
PR #52537:

  * global RNG must not be touched (the class documents RNG isolation),
  * blank lines in .jsonl must not abort the run,
  * malformed ShareGPT entries (missing "value") must be skipped,
  * sampled prompt tokens must actually come from the source file's token pool.
"""
import json
import random

import numpy as np
import pytest

from vllm.benchmarks.datasets import RandomDataset


class FakeTokenizer:
    """A reversible byte-level tokenizer: token id == code point.

    decode/encode round-trip exactly, so gen_prompt_decode_to_target_len
    converges immediately and sampled prompts map 1:1 back to token ids.
    """

    vocab_size = 256
    all_special_ids: list[int] = []

    def num_special_tokens_to_add(self) -> int:
        return 0

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [b for b in text.encode("latin-1", errors="replace")]

    def decode(self, token_ids) -> str:
        return bytes(int(t) % 256 for t in token_ids).decode("latin-1")


def _write(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return str(p)


def _source_pool_ids(texts):
    tok = FakeTokenizer()
    ids = set()
    for t in texts:
        ids.update(tok.encode(t))
    return ids


@pytest.mark.benchmark
def test_source_sampling_does_not_touch_global_rng(tmp_path):
    """The class promises RNG isolation; sampling from a file must not perturb
    Python's global `random` state (the bug: random.seed()/random.shuffle())."""
    # Must be a .json (ShareGPT) file: the global-RNG violation being guarded
    # against (random.seed()/random.shuffle()) lived in the .json branch.
    texts = ["alpha beta gamma", "delta epsilon zeta", "eta theta iota"]
    data = [
        {
            "conversations": [
                {"from": "human", "value": t},
                {"from": "gpt", "value": "ok"},
            ]
        }
        for t in texts
    ]
    path = _write(tmp_path, "src.json", json.dumps(data))

    random.seed(999)
    before = random.random()

    random.seed(999)
    ds = RandomDataset(random_seed=42, dataset_path=path)
    ds.sample(
        tokenizer=FakeTokenizer(), num_requests=4, input_len=16, output_len=8
    )
    after = random.random()

    # If the dataset had re-seeded / consumed the global RNG, these would differ.
    assert before == after


@pytest.mark.benchmark
def test_jsonl_blank_lines_are_skipped(tmp_path):
    """A trailing/interior blank line must not raise JSONDecodeError."""
    lines = [
        json.dumps({"prompt": "first prompt here"}),
        "",
        "   ",
        json.dumps({"prompt": "second prompt here"}),
        "",
    ]
    path = _write(tmp_path, "blanks.jsonl", "\n".join(lines))

    ds = RandomDataset(random_seed=42, dataset_path=path)
    samples = ds.sample(
        tokenizer=FakeTokenizer(), num_requests=3, input_len=16, output_len=8
    )
    assert len(samples) == 3


@pytest.mark.benchmark
def test_sharegpt_missing_value_key_is_skipped(tmp_path):
    """Malformed ShareGPT entries (no 'value') must be skipped, not KeyError."""
    data = [
        {"conversations": [{"from": "human"}, {"from": "gpt", "value": "y"}]},
        {
            "conversations": [
                {"from": "human", "value": "a well formed human turn"},
                {"from": "gpt", "value": "response"},
            ]
        },
    ]
    path = _write(tmp_path, "sharegpt.json", json.dumps(data))

    ds = RandomDataset(random_seed=42, dataset_path=path)
    samples = ds.sample(
        tokenizer=FakeTokenizer(), num_requests=2, input_len=16, output_len=8
    )
    assert len(samples) == 2


@pytest.mark.benchmark
def test_sampled_tokens_come_from_source_pool(tmp_path):
    """Core behavior: prompt tokens are drawn from the file's token pool.

    Uses a restricted alphabet in the source file and a zero prefix so the
    sampled window is exactly source tokens; asserts every prompt token is a
    member of the source pool (allowing the small tail that
    gen_prompt_decode_to_target_len may pad when a pool is short -- here the
    pool is long enough that no padding occurs)."""
    # Only bytes for "abc " appear in the source.
    texts = ["abc abc abc abc abc abc abc abc" for _ in range(4)]
    path = _write(tmp_path, "plain.txt", "\n".join(texts))
    pool = _source_pool_ids(texts)

    ds = RandomDataset(random_seed=7, dataset_path=path)
    tok = FakeTokenizer()
    samples = ds.sample(
        tokenizer=tok,
        num_requests=3,
        input_len=16,
        output_len=8,
        prefix_len=0,
    )
    assert len(samples) == 3
    for s in samples:
        ids = tok.encode(s.prompt)
        # Every token in the prompt must be from the restricted source alphabet.
        assert set(ids).issubset(pool), (
            f"prompt contained tokens outside the source pool: "
            f"{set(ids) - pool}"
        )


@pytest.mark.benchmark
def test_same_seed_reproducible_with_source(tmp_path):
    """Same seed + same file -> identical prompts (isolated-RNG determinism)."""
    texts = ["one two three", "four five six", "seven eight nine"]
    path = _write(
        tmp_path, "src.jsonl", "\n".join(json.dumps({"prompt": t}) for t in texts)
    )

    a = RandomDataset(random_seed=123, dataset_path=path).sample(
        tokenizer=FakeTokenizer(), num_requests=4, input_len=16, output_len=8
    )
    b = RandomDataset(random_seed=123, dataset_path=path).sample(
        tokenizer=FakeTokenizer(), num_requests=4, input_len=16, output_len=8
    )
    assert [s.prompt for s in a] == [s.prompt for s in b]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
