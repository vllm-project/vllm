# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest

from vllm.entrypoints.openai.chat_completion.serving import _usage_prompt_tokens


@pytest.mark.parametrize(
    ("prompt_token_ids", "generation_prefix_len", "encoder_ids", "expected"),
    [
        ([1, 2, 3, 4, 5], None, None, 5),
        ([1, 2, 3, 4, 5], 0, None, 5),
        ([1, 2, 3, 4, 5], 3, None, 2),
        ([1, 2, 3], 3, None, 0),
        ([1, 2, 3], 3, [9, 8], 2),
        # Invalid prefix lengths are ignored (fall back to full prompt length).
        ([1, 2, 3], -1, None, 3),
        ([1, 2, 3], 4, None, 3),
    ],
)
def test_usage_prompt_tokens(
    prompt_token_ids: list[int],
    generation_prefix_len: int | None,
    encoder_ids: list[int] | None,
    expected: int,
):
    assert (
        _usage_prompt_tokens(
            prompt_token_ids,
            generation_prefix_len=generation_prefix_len,
            encoder_prompt_token_ids=encoder_ids,
        )
        == expected
    )
