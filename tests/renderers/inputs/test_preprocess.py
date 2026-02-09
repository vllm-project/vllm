# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from vllm.renderers.inputs.preprocess import prompt_to_seq


def test_empty_input():
    assert prompt_to_seq([]) == []
    assert prompt_to_seq([[]]) == [[]]
    assert prompt_to_seq([[], []]) == [[], []]


def test_text_input():
    assert prompt_to_seq("foo") == ["foo"]
    assert prompt_to_seq(["foo"]) == ["foo"]
    assert prompt_to_seq(["foo", "bar"]) == ["foo", "bar"]


def test_token_input():
    assert prompt_to_seq([1, 2]) == [[1, 2]]
    assert prompt_to_seq([[1, 2]]) == [[1, 2]]
    assert prompt_to_seq([[1, 2], [3, 4]]) == [[1, 2], [3, 4]]


def test_text_token_input():
    assert prompt_to_seq([[1, 2], "foo"]) == [[1, 2], "foo"]
    assert prompt_to_seq(["foo", [1, 2]]) == ["foo", [1, 2]]


def test_bytes_input():
    assert prompt_to_seq(b"foo") == [b"foo"]
    assert prompt_to_seq([b"foo"]) == [b"foo"]
    assert prompt_to_seq([b"foo", b"bar"]) == [b"foo", b"bar"]


def test_dict_input():
    assert prompt_to_seq({"prompt": "foo"}) == [{"prompt": "foo"}]
    assert prompt_to_seq([{"prompt": "foo"}]) == [{"prompt": "foo"}]
    assert prompt_to_seq([{"prompt": "foo"}, {"prompt_token_ids": [1, 2]}]) == [
        {"prompt": "foo"},
        {"prompt_token_ids": [1, 2]},
    ]
