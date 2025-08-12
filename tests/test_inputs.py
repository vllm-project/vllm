from typing import List

import pytest

from vllm.inputs import parse_and_batch_prompt

STRING_INPUTS = [
    '',
    'foo',
    'foo bar',
    'foo baz bar',
    'foo bar qux baz',
]

TOKEN_INPUTS = [
    [-1],
    [1],
    [1, 2],
    [1, 3, 4],
    [1, 2, 4, 3],
]

INPUTS_SLICES = [
    slice(None, None, -1),
    slice(None, None, 2),
    slice(None, None, -2),
]


def test_parse_single_batch_empty():
    with pytest.raises(ValueError, match="at least one prompt"):
        parse_and_batch_prompt([])

    with pytest.raises(ValueError, match="at least one prompt"):
        parse_and_batch_prompt([[]])


@pytest.mark.parametrize('string_input', STRING_INPUTS)
def test_parse_single_batch_string_consistent(string_input: str):
    assert parse_and_batch_prompt(string_input) \
        == parse_and_batch_prompt([string_input])


@pytest.mark.parametrize('token_input', TOKEN_INPUTS)
def test_parse_single_batch_token_consistent(token_input: List[int]):
    assert parse_and_batch_prompt(token_input) \
        == parse_and_batch_prompt([token_input])


@pytest.mark.parametrize('inputs_slice', INPUTS_SLICES)
def test_parse_single_batch_string_slice(inputs_slice: slice):
    assert parse_and_batch_prompt(STRING_INPUTS)[inputs_slice] \
        == parse_and_batch_prompt(STRING_INPUTS[inputs_slice])
