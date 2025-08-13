# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import numpy as np

from vllm.config import ModelConfig, SpeculativeConfig, VllmConfig
from vllm.v1.spec_decode.ngram_proposer import (
    NgramProposer, _find_longest_matched_ngram_and_propose_tokens)


def test_find_longest_matched_ngram_and_propose_tokens():
    tokens = np.array([1, 2, 3, 4, 1, 2, 3, 5, 6])
    assert _find_longest_matched_ngram_and_propose_tokens(origin_tokens=tokens,
                                                          min_ngram=2,
                                                          max_ngram=2,
                                                          max_model_len=1024,
                                                          k=2) is None

    tokens = np.array([1, 2, 3, 4, 1, 2, 3])
    np.testing.assert_array_equal(
        _find_longest_matched_ngram_and_propose_tokens(origin_tokens=tokens,
                                                       min_ngram=2,
                                                       max_ngram=2,
                                                       max_model_len=1024,
                                                       k=3),
        np.array([4, 1, 2]))
    np.testing.assert_array_equal(
        _find_longest_matched_ngram_and_propose_tokens(origin_tokens=tokens,
                                                       min_ngram=2,
                                                       max_ngram=2,
                                                       max_model_len=1024,
                                                       k=2), np.array([4, 1]))
    np.testing.assert_array_equal(
        _find_longest_matched_ngram_and_propose_tokens(origin_tokens=tokens,
                                                       min_ngram=1,
                                                       max_ngram=1,
                                                       max_model_len=1024,
                                                       k=3),
        np.array([4, 1, 2]))
    np.testing.assert_array_equal(
        _find_longest_matched_ngram_and_propose_tokens(origin_tokens=tokens,
                                                       min_ngram=1,
                                                       max_ngram=1,
                                                       max_model_len=1024,
                                                       k=2), np.array([4, 1]))

    tokens = np.array([1, 3, 6, 2, 3, 4, 1, 2, 3])
    np.testing.assert_array_equal(
        _find_longest_matched_ngram_and_propose_tokens(origin_tokens=tokens,
                                                       min_ngram=2,
                                                       max_ngram=2,
                                                       max_model_len=1024,
                                                       k=3),
        np.array([4, 1, 2]))
    # Return on the first match
    np.testing.assert_array_equal(
        _find_longest_matched_ngram_and_propose_tokens(origin_tokens=tokens,
                                                       min_ngram=1,
                                                       max_ngram=1,
                                                       max_model_len=1024,
                                                       k=2), np.array([6, 2]))


def test_ngram_proposer():

    def ngram_proposer(min_n: int, max_n: int, k: int) -> NgramProposer:
        # Dummy model config. Just to set max_model_len.
        model_config = ModelConfig(model="facebook/opt-125m")
        return NgramProposer(
            vllm_config=VllmConfig(model_config=model_config,
                                   speculative_config=SpeculativeConfig(
                                       prompt_lookup_min=min_n,
                                       prompt_lookup_max=max_n,
                                       num_speculative_tokens=k,
                                       method="ngram",
                                   )))

    # No match.
    result = ngram_proposer(
        min_n=2, max_n=2,
        k=2).propose(context_token_ids=np.array([1, 2, 3, 4, 5]))
    assert result is None

    # No match for 4-gram.
    result = ngram_proposer(
        min_n=4, max_n=4,
        k=2).propose(context_token_ids=np.array([1, 2, 3, 4, 1, 2, 3]))
    assert result is None

    # No match for 4-gram but match for 3-gram.
    result = ngram_proposer(
        min_n=3, max_n=4,
        k=2).propose(context_token_ids=np.array([1, 2, 3, 4, 1, 2, 3]))
    assert np.array_equal(result, np.array([4, 1]))

    # Match for both 4-gram and 3-gram.
    # In this case, the proposer should return the 4-gram match.
    result = ngram_proposer(min_n=3, max_n=4, k=2).propose(
        context_token_ids=np.array([2, 3, 4, 5, 1, 2, 3, 4, 1, 2, 3, 4]))
    assert np.array_equal(result, np.array([1, 2]))  # Not [5, 1]

    # Match for 2-gram and 3-gram, but not 4-gram.
    result = ngram_proposer(min_n=2, max_n=4, k=2).propose(
        context_token_ids=np.array([3, 4, 5, 2, 3, 4, 1, 2, 3, 4]))
    assert np.array_equal(result, np.array([1, 2]))  # Not [5, 2]

    # Multiple 3-gram matched, but always pick the first one.
    result = ngram_proposer(
        min_n=3, max_n=3, k=2).propose(context_token_ids=np.array(
            [1, 2, 3, 100, 1, 2, 3, 200, 1, 2, 3, 300, 1, 2, 3]))
    assert np.array_equal(result, np.array([100, 1]))
