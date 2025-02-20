# SPDX-License-Identifier: Apache-2.0
from unittest.mock import Mock

from vllm.transformers_utils.detokenizer_utils import (
    convert_prompt_ids_to_tokens)

return_tokens = [
    "�this", "�is", "�a", "�test", "�for", "�initial", "�incremental",
    "�detokenization", "�offset"
]


def mock_convert_ids_to_tokens(ids: list[int], *args, **kwargs):
    return return_tokens[-len(ids):]


def test_intial_incremental_detokenization_offset():
    mock_tokenizer = Mock()
    mock_tokenizer.convert_ids_to_tokens = mock_convert_ids_to_tokens
    new_tokens, prefix_offset, read_offset = convert_prompt_ids_to_tokens(
        mock_tokenizer,
        prompt_ids=list(range(len(return_tokens))),
    )
    assert prefix_offset == 2
    assert read_offset == 7
    assert new_tokens == return_tokens[-7:]

    new_tokens, prefix_offset, read_offset = convert_prompt_ids_to_tokens(
        mock_tokenizer,
        prompt_ids=list(range(len(return_tokens))),
        intial_incremental_detokenization_offset=0,
    )
    assert prefix_offset == 2
    assert read_offset == 2
