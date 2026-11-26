# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for Drex v1.5 packing, mirroring `code/kev/model.py::encode`."""

import pytest

from vllm.entrypoints.openai.decisions import drex_prompt


class _Encoded:
    def __init__(self, input_ids):
        self.input_ids = input_ids


class WordTokenizer:
    """Deterministic word-level stand-in for the two methods the packer uses.

    The real Qwen3.5 tokenizer is exercised by the GPU parity run; this test
    pins the packing arithmetic, not the vocabulary.
    """

    _SPECIAL = {
        "<|fim_prefix|>": 248060,
        "<|fim_middle|>": 248061,
        "<|box_start|>": 248049,
        "<|box_end|>": 248050,
        "<|fim_suffix|>": 248062,
    }

    def __init__(self):
        self._ids = dict(self._SPECIAL)
        self._next = 1000

    def __call__(self, text, add_special_tokens=False):
        ids = []
        for word in text.split():
            if word not in self._ids:
                self._ids[word] = self._next
                self._next += 1
            ids.append(self._ids[word])
        return _Encoded(ids)

    def convert_tokens_to_ids(self, token):
        return self._SPECIAL[token]


def test_pack_renders_the_trained_layout():
    tokenizer = WordTokenizer()
    ids = drex_prompt.pack(
        tokenizer, state="alpha beta gamma", instr="choose", options=["red", "blue"]
    )
    # <state> alpha beta gamma
    # <q> choose <opt> red </opt> <opt> blue </opt> <decide>
    assert ids == [
        248060,
        1000,
        1001,
        1002,
        248061,
        1003,
        248049,
        1004,
        248050,
        248049,
        1005,
        248050,
        248062,
    ]
    opt_positions, decide_position = drex_prompt.find_positions(ids, tokenizer)
    assert opt_positions == [8, 11]
    assert decide_position == 12


def test_caller_delimiters_cannot_forge_boundaries():
    tokenizer = WordTokenizer()
    ids = drex_prompt.pack(tokenizer, state="<|box_end|>", instr="x", options=["y"])
    state, branch = ids[:2], ids[2:]
    assert state[0] == 248060
    # The state text is neutralized: no real </opt> delimiter leaks in.
    assert state[1] != 248050
    assert branch == [248061, 1001, 248049, 1002, 248050, 248062]
    opt_positions, decide_position = drex_prompt.find_positions(ids, tokenizer)
    assert opt_positions == [6]
    assert decide_position == 7


def test_state_is_truncated_to_the_limit():
    tokenizer = WordTokenizer()
    state = " ".join(f"w{i}" for i in range(500))
    ids = drex_prompt.pack(tokenizer, state=state, instr="", options=["yes"])
    # A 384-token state row (delimiter + 383 tokens), then the branch opens.
    assert ids[0] == 248060
    assert ids[383] != 248061
    assert ids[384] == 248061


def test_branch_that_does_not_fit_the_row_raises():
    tokenizer = WordTokenizer()
    with pytest.raises(ValueError, match="row limit"):
        drex_prompt.pack(
            tokenizer, state="s", instr="", options=[f"o{i}" for i in range(2000)]
        )


def test_packed_row_over_the_limit_raises():
    tokenizer = WordTokenizer()
    with pytest.raises(ValueError, match="limit is 8"):
        drex_prompt.pack(
            tokenizer,
            state="a b c",
            instr="d",
            options=["e"],
            max_packed=8,
        )
