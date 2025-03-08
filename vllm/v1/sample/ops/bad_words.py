# SPDX-License-Identifier: Apache-2.0

import torch

_SMALLEST_LOGIT = float("-inf")


def _apply_bad_words_single_batch(
    logits: torch.Tensor,
    bad_words_token_ids: list[list[int]],
    past_tokens_ids: list[int],
) -> None:
    for bad_word_ids in bad_words_token_ids:
        if len(bad_word_ids) > len(past_tokens_ids) + 1:
            continue

        prefix_length = len(bad_word_ids) - 1
        last_token_id = bad_word_ids[-1]
        if prefix_length > 0:
            actual_prefix = past_tokens_ids[-prefix_length:]
        else:
            actual_prefix = []
        expected_prefix = bad_word_ids[:prefix_length]

        assert len(actual_prefix) == len(expected_prefix)

        if actual_prefix == expected_prefix:
            logits[last_token_id] = _SMALLEST_LOGIT


def apply_bad_words(
    logits: torch.Tensor,
    bad_words_token_ids: dict[int, list[list[int]]],
    past_tokens_ids: list[list[int]],
) -> None:
    for i, bad_words_ids in bad_words_token_ids.items():
        _apply_bad_words_single_batch(logits[i], bad_words_ids,
                                      past_tokens_ids[i])
