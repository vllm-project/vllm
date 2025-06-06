# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from enum import Enum
from typing import Optional

import numpy as np
import regex as re

from vllm import CompletionOutput


class BatchLogprobsComposition(Enum):
    """Types of logprobs configs to include in test batch"""
    NONE = 0
    SAMPLE = 1
    PROMPT = 2
    SAMPLE_PROMPT = 3


BatchLogprobsSpecType = list[tuple[Optional[int], Optional[int]]]


def get_test_batch(
    batch_logprobs_composition: BatchLogprobsComposition
) -> BatchLogprobsSpecType:
    """Generate logprobs configs for a batch of requests

    A given request's logprobs configuration is (1) num_sample_logprobs and (2)
    num_prompt_logprobs. The batch logprobs configuration is the list of request
    logprobs configs.

    batch_logprobs_composition == NONE yields a batch with no sample or prompt
    logprobs

    batch_logprobs_composition == SAMPLE yields a batch with some requests
    configured for sample logprobs only, and others configured for no logprobs

    batch_logprobs_composition == PROMPT yields a batch with some requests
    configured for prompt logprobs only, and others configured for no logprobs

    batch_logprobs_composition == SAMPLE_PROMPT yields a batch with some
    requests configured for sample logprobs and prompt logprobs, some configured
    for only sample logprobs or only prompt logprobs, and some configured for
    no logprobs

    Args:
      batch_logprobs_composition: types of logprobs configs to include in batch

    Returns:

      list of (Optional[num_sample_logprobs], Optional[num_prompt_logprobs])
      tuples
    """
    if batch_logprobs_composition == BatchLogprobsComposition.NONE:
        # No requests with sample or prompt logprobs
        return [(None, None)]
    elif batch_logprobs_composition == BatchLogprobsComposition.SAMPLE:
        # Requests requiring sample logprobs or no logprobs
        return [
            (None, None),
            (0, None),
            (5, None),
            (3, None),
        ]
    elif batch_logprobs_composition == BatchLogprobsComposition.PROMPT:
        # Requests requiring prompt logprobs or no logprobs
        return [
            (None, None),
            (None, 0),
            (None, 6),
            (None, 5),
        ]
    elif batch_logprobs_composition == BatchLogprobsComposition.SAMPLE_PROMPT:
        # Requests requiring either no logprobs, just
        # sample logprobs, just prompt logprobs, or
        # both sample and prompt logprobs
        return [
            (None, None),
            (0, None),
            (5, None),
            (3, None),
            (0, 3),
            (6, 0),
            (6, 3),
            (None, 6),
            (None, 5),
            (None, 0),
        ]
    else:
        raise ValueError("Invalid logprobs batch configuration for test.")


def assert_incr_detok_str_matches_non_incr_detok_str(
    incremental_detokenization_str: str,
    non_incremental_detokenization_str: str,
    msg: str,
) -> None:
    """Compare incrementally detok. text to non-incrementally detok. text

    Fail if the strings mismatch after non-alphanumeric characters are stripped
    out.

    Rationale: incremental detokenization in the text generation process allows
    the tokenizer to adjust the next token text output based on the token's
    context in the string. However, logprobs detokenization detokenizes each
    token individually, and the resultant strings may include some
    non-alphanumeric placeholder characters where there could be i.e.
    whitespace. So, this function compares only the alphanumeric text
    between two strings and fails if there is a mismatch, which helps
    with validating logprobs detokenization.

    Args:
      incremental_detokenization_str: incrementally-detokenized generated text
      non_incremental_detokenization_str: non-incrementally-detokenized logprob
                                          tokens
      msg: error message if `assert` fails
    """
    rgx = r'[^a-zA-Z0-9]+'
    assert (re.sub(rgx, '', incremental_detokenization_str) == re.sub(
        rgx, '', non_incremental_detokenization_str)), (msg)


def compute_correct_cumulative_logprob(
        completion_output: CompletionOutput) -> float:
    """Compute known-good value for evaluating cumulative logprob

    Args:
      completion_output: completion output from engine

    Returns:
      Known-good cumulative logprob value
    """
    token_ids = completion_output.token_ids
    logprobs = completion_output.logprobs
    assert logprobs is not None
    return sum([lp[tok_id].logprob for tok_id, lp in zip(token_ids, logprobs)])


def create_weighted_output_token_list(
        batch_size: int,
        vocab_size: int,
        min_freq: int = 1) -> tuple[list[list[int]], list[list[int]]]:
    """
    Creates an output token list where each token occurs a distinct
    number of times.

    For each batch, a random subset of token IDs is selected from the
    vocabulary. The n selected tokens are then added to the output token
    list, each with a different frequency from [min_freq, min_freq+n].

    Returns:
        tuple[list[list[int]], list[list[int]]]:
            - The first element is the output token list, where each sublist
              corresponds to a batch and contains tokens with weighted
              frequencies.
            - The second element is a list of distinct token IDs for each
              batch, ordered by their frequency in the corresponding output
              list.
    """
    output_token_ids: list[list[int]] = []
    sorted_token_ids_in_output: list[list[int]] = []
    for _ in range(batch_size):
        distinct_token_ids = np.random.choice(vocab_size,
                                              size=np.random.randint(5, 10),
                                              replace=False).tolist()
        sorted_token_ids_in_output.append(distinct_token_ids)
        output_token_ids_for_batch = []
        for index, token_id in enumerate(distinct_token_ids):
            output_token_ids_for_batch.extend(
                [token_id for _ in range(index + min_freq)])
        output_token_ids.append(output_token_ids_for_batch)
    return output_token_ids, sorted_token_ids_in_output
