# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import numpy as np

from vllm.config import (
    ModelConfig,
    SpeculativeConfig,
    VllmConfig,
)
from vllm.v1.spec_decode.ngram_proposer import (
    NgramProposer,
    _find_longest_matched_ngram_and_propose_tokens,
)


def test_find_longest_matched_ngram_and_propose_tokens():
    tokens = np.array([1, 2, 3, 4, 1, 2, 3, 5, 6])
    result = _find_longest_matched_ngram_and_propose_tokens(
        origin_tokens=tokens, min_ngram=2, max_ngram=2, max_model_len=1024, k=2
    )
    assert len(result) == 0

    tokens = np.array([1, 2, 3, 4, 1, 2, 3])
    np.testing.assert_array_equal(
        _find_longest_matched_ngram_and_propose_tokens(
            origin_tokens=tokens, min_ngram=2, max_ngram=2, max_model_len=1024, k=3
        ),
        np.array([4, 1, 2]),
    )
    np.testing.assert_array_equal(
        _find_longest_matched_ngram_and_propose_tokens(
            origin_tokens=tokens, min_ngram=2, max_ngram=2, max_model_len=1024, k=2
        ),
        np.array([4, 1]),
    )
    np.testing.assert_array_equal(
        _find_longest_matched_ngram_and_propose_tokens(
            origin_tokens=tokens, min_ngram=1, max_ngram=1, max_model_len=1024, k=3
        ),
        np.array([4, 1, 2]),
    )
    np.testing.assert_array_equal(
        _find_longest_matched_ngram_and_propose_tokens(
            origin_tokens=tokens, min_ngram=1, max_ngram=1, max_model_len=1024, k=2
        ),
        np.array([4, 1]),
    )

    tokens = np.array([1, 3, 6, 2, 3, 4, 1, 2, 3])
    np.testing.assert_array_equal(
        _find_longest_matched_ngram_and_propose_tokens(
            origin_tokens=tokens, min_ngram=2, max_ngram=2, max_model_len=1024, k=3
        ),
        np.array([4, 1, 2]),
    )
    # Return on the first match
    np.testing.assert_array_equal(
        _find_longest_matched_ngram_and_propose_tokens(
            origin_tokens=tokens, min_ngram=1, max_ngram=1, max_model_len=1024, k=2
        ),
        np.array([6, 2]),
    )


def test_ngram_proposer():
    def get_ngram_proposer(min_n: int, max_n: int, k: int) -> NgramProposer:
        # Dummy model config. Just to set max_model_len.
        model_config = ModelConfig(model="facebook/opt-125m")
        return NgramProposer(
            vllm_config=VllmConfig(
                model_config=model_config,
                speculative_config=SpeculativeConfig(
                    prompt_lookup_min=min_n,
                    prompt_lookup_max=max_n,
                    num_speculative_tokens=k,
                    method="ngram",
                ),
            )
        )

    # No match.
    token_ids_cpu = np.array([[1, 2, 3, 4, 5]])
    result = get_ngram_proposer(min_n=2, max_n=2, k=2).propose(
        num_speculative_tokens=2,
        sampled_token_ids=[[0]],
        num_tokens_no_spec=np.array([len(c) for c in token_ids_cpu]),
        token_ids_cpu=token_ids_cpu,
    )
    assert len(result[0]) == 0

    # No match for 4-gram.
    token_ids_cpu = np.array([[1, 2, 3, 4, 1, 2, 3]])
    result = get_ngram_proposer(min_n=4, max_n=4, k=2).propose(
        num_speculative_tokens=2,
        sampled_token_ids=[[0]],
        num_tokens_no_spec=np.array([len(c) for c in token_ids_cpu]),
        token_ids_cpu=token_ids_cpu,
    )
    assert len(result[0]) == 0

    # No match for 4-gram but match for 3-gram.
    token_ids_cpu = np.array([[1, 2, 3, 4, 1, 2, 3]])
    result = get_ngram_proposer(min_n=3, max_n=4, k=2).propose(
        num_speculative_tokens=2,
        sampled_token_ids=[[0]],
        num_tokens_no_spec=np.array([len(c) for c in token_ids_cpu]),
        token_ids_cpu=token_ids_cpu,
    )
    assert np.array_equal(result, np.array([[4, 1]]))

    # Match for both 4-gram and 3-gram.
    # In this case, the proposer should return the 4-gram match.
    token_ids_cpu = np.array([[2, 3, 4, 5, 1, 2, 3, 4, 1, 2, 3, 4]])
    result = get_ngram_proposer(min_n=3, max_n=4, k=2).propose(
        num_speculative_tokens=2,
        sampled_token_ids=[[0]],
        num_tokens_no_spec=np.array([len(c) for c in token_ids_cpu]),
        token_ids_cpu=token_ids_cpu,
    )
    assert np.array_equal(result, np.array([[1, 2]]))  # Not [5, 1]]

    # Match for 2-gram and 3-gram, but not 4-gram.
    token_ids_cpu = np.array([[3, 4, 5, 2, 3, 4, 1, 2, 3, 4]])
    result = get_ngram_proposer(min_n=2, max_n=4, k=2).propose(
        num_speculative_tokens=2,
        sampled_token_ids=[[0]],
        num_tokens_no_spec=np.array([len(c) for c in token_ids_cpu]),
        token_ids_cpu=token_ids_cpu,
    )
    assert np.array_equal(result, np.array([[1, 2]]))  # Not [5, 2]]

    # Multiple 3-gram matched, but always pick the first one.
    token_ids_cpu = np.array([[1, 2, 3, 100, 1, 2, 3, 200, 1, 2, 3, 300, 1, 2, 3]])
    result = get_ngram_proposer(min_n=3, max_n=3, k=2).propose(
        num_speculative_tokens=2,
        sampled_token_ids=[[0]],
        num_tokens_no_spec=np.array([len(c) for c in token_ids_cpu]),
        token_ids_cpu=token_ids_cpu,
    )
    assert np.array_equal(result, np.array([[100, 1]]))

    # check empty input
    token_ids_cpu = np.array([[]])
    result = get_ngram_proposer(min_n=2, max_n=2, k=2).propose(
        num_speculative_tokens=2,
        sampled_token_ids=[[0]],
        num_tokens_no_spec=np.array([len(c) for c in token_ids_cpu]),
        token_ids_cpu=token_ids_cpu,
    )
    assert len(result[0]) == 0

    # check multibatch input
    # first request has 5 tokens and a match
    # second request has 3 tokens and no match. Padded with -1 for max len 5
    token_ids_cpu = np.array([[1, 2, 3, 1, 2], [4, 5, 6, -1, -1]])
    result = get_ngram_proposer(min_n=2, max_n=2, k=2).propose(
        num_speculative_tokens=2,
        sampled_token_ids=[[0], [1]],
        num_tokens_no_spec=np.array([5, 3]),
        token_ids_cpu=token_ids_cpu,
    )
    assert len(result[0]) == 2
    assert np.array_equal(result[0], np.array([3, 1]))
    assert np.array_equal(result[1], np.array([]))

    # Test non-contiguous indices: requests 0 and 2 need proposals,
    # request 1 is in prefill
    proposer = get_ngram_proposer(min_n=2, max_n=2, k=2)
    max_model_len = 20
    token_ids_cpu = np.zeros((3, max_model_len), dtype=np.int32)
    token_ids_cpu[0, :5] = [1, 2, 3, 1, 2]
    token_ids_cpu[1, :3] = [4, 5, 6]
    token_ids_cpu[2, :5] = [7, 8, 9, 7, 8]
    num_tokens_no_spec = np.array([5, 3, 5], dtype=np.int32)
    sampled_token_ids = [[2], [], [8]]  # Empty list for request 1 simulates prefill
    result = proposer.propose(
        num_speculative_tokens=2,
        sampled_token_ids=sampled_token_ids,
        num_tokens_no_spec=num_tokens_no_spec,
        token_ids_cpu=token_ids_cpu,
    )
    assert len(result) == 3
    assert np.array_equal(result[0], [3, 1])
    assert len(result[1]) == 0
    assert np.array_equal(result[2], [9, 7])
    # Verify internal arrays written to correct indices
    assert proposer.valid_ngram_num_drafts[0] == 2
    assert proposer.valid_ngram_num_drafts[1] == 0
    assert proposer.valid_ngram_num_drafts[2] == 2
    assert np.array_equal(proposer.valid_ngram_draft[0, :2], [3, 1])
    assert np.array_equal(proposer.valid_ngram_draft[2, :2], [9, 7])

    # test if 0 threads available: can happen if TP size > CPU count
    ngram_proposer = get_ngram_proposer(min_n=2, max_n=2, k=2)
    ngram_proposer.num_numba_thread_available = 0
    # set max_model_len to 2 * threshold to ensure multithread is used
    num_tokens_threshold = ngram_proposer.num_tokens_threshold
    ngram_proposer.max_model_len = 2 * num_tokens_threshold
    # using multibatch test
    middle_integer = num_tokens_threshold // 2
    input_1 = [_ for _ in range(num_tokens_threshold)]
    input_1 += [middle_integer, middle_integer + 1]
    input_2 = [-1] * len(input_1)
    input_2[:3] = [4, 5, 6]
    token_ids_cpu = np.array([input_1, input_2])
    result = ngram_proposer.propose(
        num_speculative_tokens=2,
        sampled_token_ids=[[0], [1]],
        num_tokens_no_spec=np.array([len(input_1), 3]),
        token_ids_cpu=token_ids_cpu,
    )
    assert len(result[0]) == 2
    assert np.array_equal(result[0], np.array([middle_integer + 2, middle_integer + 3]))
    assert np.array_equal(result[1], np.array([]))


def test_ngram_global_cache_proposer():
    model_config = ModelConfig(model="facebook/opt-125m")
    proposer = NgramProposer(
        vllm_config=VllmConfig(
            model_config=model_config,
            speculative_config=SpeculativeConfig(
                prompt_lookup_min=2,
                prompt_lookup_max=2,
                prompt_lookup_cache_scope="global",
                prompt_lookup_global_branch_length=6,
                num_speculative_tokens=3,
                method="ngram",
            ),
        )
    )

    first_request_tokens = np.array([[1, 2, 3, 4, 5, 6]], dtype=np.int32)
    proposer.propose(
        sampled_token_ids=[[6]],
        num_tokens_no_spec=np.array([first_request_tokens.shape[1]], dtype=np.int32),
        token_ids_cpu=first_request_tokens,
        req_ids=["req-0"],
    )

    second_request_tokens = np.array([[9, 8, 3, 4]], dtype=np.int32)
    result = proposer.propose(
        sampled_token_ids=[[4]],
        num_tokens_no_spec=np.array([second_request_tokens.shape[1]], dtype=np.int32),
        token_ids_cpu=second_request_tokens,
        req_ids=["req-1"],
    )

    assert result == [[5, 6]]


def test_ngram_global_cache_respects_max_model_len():
    model_config = ModelConfig(model="facebook/opt-125m", max_model_len=6)
    proposer = NgramProposer(
        vllm_config=VllmConfig(
            model_config=model_config,
            speculative_config=SpeculativeConfig(
                prompt_lookup_min=2,
                prompt_lookup_max=2,
                prompt_lookup_cache_scope="global",
                prompt_lookup_global_branch_length=6,
                num_speculative_tokens=3,
                method="ngram",
            ),
        )
    )

    first_request_tokens = np.array([[1, 2, 3, 4, 5]], dtype=np.int32)
    proposer.propose(
        sampled_token_ids=[[5]],
        num_tokens_no_spec=np.array([first_request_tokens.shape[1]], dtype=np.int32),
        token_ids_cpu=first_request_tokens,
        req_ids=["req-0"],
    )

    second_request_tokens = np.array([[9, 8, 1, 2]], dtype=np.int32)
    result = proposer.propose(
        sampled_token_ids=[[2]],
        num_tokens_no_spec=np.array([second_request_tokens.shape[1]], dtype=np.int32),
        token_ids_cpu=second_request_tokens,
        req_ids=["req-1"],
    )

    assert result == [[3, 4]]


def test_ngram_global_cache_evicts_old_entries():
    model_config = ModelConfig(model="facebook/opt-125m")
    proposer = NgramProposer(
        vllm_config=VllmConfig(
            model_config=model_config,
            speculative_config=SpeculativeConfig(
                prompt_lookup_min=1,
                prompt_lookup_max=1,
                prompt_lookup_cache_scope="global",
                prompt_lookup_global_branch_length=3,
                prompt_lookup_global_max_entries=1,
                num_speculative_tokens=1,
                method="ngram",
            ),
        )
    )

    proposer.propose(
        sampled_token_ids=[[3]],
        num_tokens_no_spec=np.array([3], dtype=np.int32),
        token_ids_cpu=np.array([[1, 2, 3]], dtype=np.int32),
        req_ids=["req-0"],
    )
    proposer.propose(
        sampled_token_ids=[[6]],
        num_tokens_no_spec=np.array([3], dtype=np.int32),
        token_ids_cpu=np.array([[4, 5, 6]], dtype=np.int32),
        req_ids=["req-1"],
    )

    result = proposer.propose(
        sampled_token_ids=[[1]],
        num_tokens_no_spec=np.array([1], dtype=np.int32),
        token_ids_cpu=np.array([[1]], dtype=np.int32),
        req_ids=["req-2"],
    )

    assert result == [[]]


def test_ngram_global_cache_does_not_fallback_to_local():
    model_config = ModelConfig(model="facebook/opt-125m")
    proposer = NgramProposer(
        vllm_config=VllmConfig(
            model_config=model_config,
            speculative_config=SpeculativeConfig(
                prompt_lookup_min=2,
                prompt_lookup_max=2,
                prompt_lookup_cache_scope="global",
                prompt_lookup_global_branch_length=3,
                num_speculative_tokens=2,
                method="ngram",
            ),
        )
    )

    token_ids_cpu = np.array([[0, 0, 0]], dtype=np.int32)
    proposer.propose(
        sampled_token_ids=[[0]],
        num_tokens_no_spec=np.array([3], dtype=np.int32),
        token_ids_cpu=token_ids_cpu,
        req_ids=["req-0"],
    )

    # Local ngram would find the earlier [1, 2] in the complete context and
    # propose [3, 4]. Global mode only indexes the configured suffix window
    # after the first full-context insert, so this remains a cache miss.
    token_ids_cpu = np.array([[0, 0, 0, 1, 2, 3, 4, 5, 1, 2]], dtype=np.int32)
    result = proposer.propose(
        sampled_token_ids=[[2]],
        num_tokens_no_spec=np.array([10], dtype=np.int32),
        token_ids_cpu=token_ids_cpu,
        req_ids=["req-0"],
    )

    assert result == [[]]


def test_ngram_global_cache_indexes_full_context_once():
    model_config = ModelConfig(model="facebook/opt-125m")
    proposer = NgramProposer(
        vllm_config=VllmConfig(
            model_config=model_config,
            speculative_config=SpeculativeConfig(
                prompt_lookup_min=2,
                prompt_lookup_max=2,
                prompt_lookup_cache_scope="global",
                prompt_lookup_global_branch_length=3,
                num_speculative_tokens=2,
                method="ngram",
            ),
        )
    )

    # The earlier [1, 2] is outside the configured suffix window of length 3.
    # The first time a request is indexed, global mode still inserts the full
    # visible context so prompt-local matches can be reused without a local
    # fallback scan.
    token_ids_cpu = np.array([[1, 2, 3, 4, 5, 1, 2]], dtype=np.int32)
    result = proposer.propose(
        sampled_token_ids=[[2]],
        num_tokens_no_spec=np.array([7], dtype=np.int32),
        token_ids_cpu=token_ids_cpu,
        req_ids=["req-0"],
    )

    assert result == [[3, 4]]


def test_ngram_global_cache_skips_full_context_when_cache_is_small():
    model_config = ModelConfig(model="facebook/opt-125m")
    proposer = NgramProposer(
        vllm_config=VllmConfig(
            model_config=model_config,
            speculative_config=SpeculativeConfig(
                prompt_lookup_min=2,
                prompt_lookup_max=2,
                prompt_lookup_cache_scope="global",
                prompt_lookup_global_branch_length=3,
                prompt_lookup_global_max_entries=1,
                num_speculative_tokens=2,
                method="ngram",
            ),
        )
    )

    token_ids_cpu = np.array([[1, 2, 3, 4, 5, 1, 2]], dtype=np.int32)
    result = proposer.propose(
        sampled_token_ids=[[2]],
        num_tokens_no_spec=np.array([7], dtype=np.int32),
        token_ids_cpu=token_ids_cpu,
        req_ids=["req-0"],
    )

    assert result == [[]]
