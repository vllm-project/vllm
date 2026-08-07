# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from vllm.inputs import tokens_input
from vllm.v1.engine.input_processor import InputProcessor


def make_input_processor(vocab_size: int) -> InputProcessor:
    processor = object.__new__(InputProcessor)
    processor.model_config = SimpleNamespace(get_vocab_size=lambda: vocab_size)
    processor.renderer = SimpleNamespace(
        tokenizer=SimpleNamespace(max_token_id=vocab_size - 1)
    )
    processor._validate_prompt_len = Mock()
    return processor


@pytest.mark.parametrize("token_ids", [[0], [0, 9]])
def test_pretokenized_prompt_ids_accept_valid_non_negative_ids(token_ids: list[int]):
    processor = make_input_processor(vocab_size=10)

    processor._validate_model_input(tokens_input(token_ids), prompt_type="decoder")


@pytest.mark.parametrize("token_ids", [[-1], [1, -1], [-1034240]])
def test_pretokenized_prompt_ids_reject_negative_ids_before_model_execution(
    token_ids: list[int],
):
    processor = make_input_processor(vocab_size=10)

    with pytest.raises(ValueError, match="out of vocabulary"):
        processor._validate_model_input(tokens_input(token_ids), prompt_type="decoder")
