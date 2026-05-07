# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest

from vllm.renderers.inputs.preprocess import (
    parse_dec_only_prompt,
    parse_enc_dec_prompt,
    prompt_to_seq,
)


def test_empty_input():
    assert prompt_to_seq([]) == []
    assert prompt_to_seq([[]]) == [[]]
    assert prompt_to_seq([[], []]) == [[], []]


def test_text_input():
    assert prompt_to_seq("foo") == ["foo"]
    assert prompt_to_seq(["foo"]) == ["foo"]
    assert prompt_to_seq(["foo", "bar"]) == ["foo", "bar"]


def test_tokens_input():
    assert prompt_to_seq([1, 2]) == [[1, 2]]
    assert prompt_to_seq([[1, 2]]) == [[1, 2]]
    assert prompt_to_seq([[1, 2], [3, 4]]) == [[1, 2], [3, 4]]


def test_text_tokens_input():
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


def test_parse_dec_only_prompt_rejects_non_string_prompt_field():
    with pytest.raises(TypeError, match="Prompt text should be a string"):
        parse_dec_only_prompt({"prompt": [1, 2, 3], "cache_salt": "abc"})


def test_parse_dec_only_prompt_rejects_non_string_prompt_list():
    with pytest.raises(TypeError, match="Prompt text should be a string"):
        parse_dec_only_prompt({"prompt": [1, "x"]})


def test_parse_enc_dec_prompt_rejects_nested_non_string_prompt_field():
    with pytest.raises(TypeError, match="Prompt text should be a string"):
        parse_enc_dec_prompt(
            {
                "encoder_prompt": {"prompt": [1, 2, 3]},
                "decoder_prompt": {"prompt": [4, 5]},
            }
        )
