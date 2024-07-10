from typing import Callable, List, Union

import torch

LogitsProcessor = Union[Callable[[List[int], torch.Tensor], torch.Tensor],
                        Callable[[List[int], List[int], torch.Tensor],
                                 torch.Tensor]]
"""LogitsProcessor is a function that takes a list
of previously generated tokens, the logits tensor
for the next token and, optionally, prompt tokens as a
first argument, and returns a modified tensor of logits
to sample from."""


class NoBadWordsLogitsProcessor:
    _SMALLEST_LOGIT = float("-inf")
    _NEUTRAL_LOGIT = 0.0

    def __init__(self, bad_words_ids: List[List[int]]):
        self.bad_words_ids = bad_words_ids
        self.word_bias: torch.FloatTensor = None

    def __call__(
        self,
        past_tokens_ids: List[int],
        logits: torch.FloatTensor,
    ) -> torch.Tensor:
        if self.word_bias is None:
            self._init_word_bias(logits=logits)

        last_token_bias = torch.zeros_like(logits)

        for bad_word_ids in self.bad_words_ids:
            if len(bad_word_ids) == 1:  # 1-token words already processed
                continue

            if len(bad_word_ids) > len(past_tokens_ids) + 1:
                continue

            prefix_length = len(bad_word_ids) - 1
            last_token_id = bad_word_ids[-1]
            actual_prefix = past_tokens_ids[-prefix_length:]
            expected_prefix = bad_word_ids[:prefix_length]
            is_match = actual_prefix == expected_prefix

            last_token_bias[last_token_id] += (self._SMALLEST_LOGIT if is_match
                                               else self._NEUTRAL_LOGIT)

        logits = logits + self.word_bias + last_token_bias

        return logits

    def _init_word_bias(self, logits: torch.FloatTensor) -> None:
        # Code based on NoBadWordsLogitsProcessor and SequenceBiasLogitsProcessor  # noqa: E501
        # from https://github.com/huggingface/transformers/blob/main/src/transformers/generation/logits_process.py

        vocab_size = logits.shape[-1]

        self._check_token_ids_bounds(vocab_size=vocab_size)

        self.word_bias = torch.zeros((vocab_size, ),
                                     dtype=torch.float,
                                     device=logits.device)

        for bad_word_ids in self.bad_words_ids:
            if len(bad_word_ids) == 1:
                bad_word_id = bad_word_ids[-1]
                self.word_bias[bad_word_id] = self._SMALLEST_LOGIT

    def _check_token_ids_bounds(self, vocab_size: int) -> None:
        invalid_token_ids = []

        for bad_word_ids in self.bad_words_ids:
            for token_id in bad_word_ids:
                if token_id < 0 or token_id >= vocab_size:
                    invalid_token_ids.append(token_id)

        if len(invalid_token_ids) > 0:
            raise ValueError(
                f"The model vocabulary size is {vocab_size},"
                f" but the following tokens"
                f" were specified as bad: {invalid_token_ids}."
                f" All token id values should be integers satisfying:"
                f" 0 <= token_id < {vocab_size}.")
