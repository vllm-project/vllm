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


def test_option_text_appends_a_description_only_when_present():
    assert drex_prompt.option_text("red", None) == "red"
    assert drex_prompt.option_text("red", "") == "red"
    assert drex_prompt.option_text("red", "a warm color") == "red: a warm color"


def test_choice_options_keep_criteria_order():
    texts, order = drex_prompt.model_options("choice", [("A", "first"), ("B", None)])
    assert texts == ["A: first", "B"]
    assert order == [0, 1]


def test_noul_options_are_packed_no_then_yes():
    # The question type stores options as (yes, no); the model reads (no, yes).
    texts, order = drex_prompt.model_options(
        "noul", [("yes", "it is true"), ("no", "it is false")]
    )
    assert texts == ["no: it is false", "yes: it is true"]
    assert order == [1, 0]


def test_score_options_are_level_labels_in_order():
    texts, order = drex_prompt.model_options(
        "score", [("0", "low"), ("1", "mid"), ("2", "high")]
    )
    assert texts == ["0", "1", "2"]
    assert order == [0, 1, 2]


def test_unknown_question_type_is_rejected():
    with pytest.raises(ValueError, match="does not answer"):
        drex_prompt.model_options("weird", [("a", None)])


def test_apply_order_keeps_identity_order():
    assert drex_prompt.apply_order([0.25, 0.75], [0, 1]) == [0.25, 0.75]


def test_apply_order_reads_a_noul_answer_as_yes_then_no():
    # The model reads (no, yes); the caller reads (yes, no).
    assert drex_prompt.apply_order([0.25, 0.75], [1, 0]) == [0.75, 0.25]


def test_apply_order_rejects_a_length_mismatch():
    with pytest.raises(ValueError, match="2 options but 3 positions"):
        drex_prompt.apply_order([0.5, 0.5], [0, 1, 2])


def test_render_keeps_a_string_as_is():
    assert drex_prompt.render("hello\nworld") == "hello\nworld"


def test_render_flattens_a_mapping_into_labelled_lines():
    assert drex_prompt.render({"hp": 3, "name": "x"}) == "hp: 3\nname: x"


def test_render_indents_nested_structures():
    assert (
        drex_prompt.render({"a": {"b": 1}, "c": [1, 2]})
        == "a:\n  b: 1\nc:\n  - 1\n  - 2"
    )


def test_render_empty_values_are_empty():
    assert drex_prompt.render(None) == ""


def test_check_budget_allows_a_request_within_the_window():
    drex_prompt.check_budget([[1, 2], [3]], max_model_len=2)


def test_check_budget_allows_a_request_exactly_at_the_limit():
    # Two rows of two tokens reach 2 * max_model_len without exceeding it.
    drex_prompt.check_budget([[1, 2], [3, 4]], max_model_len=2)


def test_check_budget_rejects_a_request_that_amplifies_one_call():
    # Every question prefills the state again, so the rows, not the question
    # count, are what bounds the work a single caller can ask for.
    with pytest.raises(ValueError, match="over the 4-token budget"):
        drex_prompt.check_budget([[1, 2], [3, 4], [5]], max_model_len=2)
