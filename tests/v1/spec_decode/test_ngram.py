# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import numpy as np
import pytest
import torch

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


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA required for NgramProposerGPU tests",
)
def test_ngram_proposer_gpu_dynamic_sd():
    """Unit tests for NgramProposerGPU supporting dynamic speculative decoding.

    Tests that NgramProposerGPU handles variable runtime num_speculative_tokens:
    - num_speculative_tokens == k (full static k)
    - num_speculative_tokens < k (dynamic schedule k < static k, slices output)
    - num_speculative_tokens == 0 (dynamic schedule k = 0, returns empty drafts)
    - preserves scattered tokens in token_ids_gpu across k=0 steps
    - raises AssertionError when num_speculative_tokens > k or < 0
    - handles multibatch with mixed matching states
    """
    from vllm.config import SchedulerConfig, set_current_vllm_config
    from vllm.v1.spec_decode.ngram_proposer_gpu import NgramProposerGPU

    device = torch.device("cuda:0")
    k = 4
    min_n = 2
    max_n = 4
    max_model_len = 64
    batch_size = 3

    model_config = ModelConfig(
        model="facebook/opt-125m",
        max_model_len=max_model_len,
        enforce_eager=True,
    )
    scheduler_config = SchedulerConfig.default_factory(
        max_num_seqs=batch_size,
        max_model_len=max_model_len,
    )
    spec_config = SpeculativeConfig(
        method="ngram_gpu",
        prompt_lookup_min=min_n,
        prompt_lookup_max=max_n,
        num_speculative_tokens=k,
    )
    vllm_config = VllmConfig(
        model_config=model_config,
        scheduler_config=scheduler_config,
        speculative_config=spec_config,
    )

    with set_current_vllm_config(vllm_config):
        proposer = NgramProposerGPU(vllm_config, device=device)

        # Batch setup:
        # req 0: [1, 2, 3, 4, 1] + [2] -> matches [1, 2] at pos 0 -> draft [3, 4, 1, 2]
        # req 1: [7, 8, 9, 10, 11] + [12] -> no match -> all -1
        # req 2: [20, 21, 99, 20] + [21] -> matches [20, 21] at pos 0
        #        -> draft [99, 20, 21, -1]
        num_tokens = torch.tensor([5, 5, 4], dtype=torch.int32, device=device)
        token_ids = torch.zeros(
            (batch_size, max_model_len), dtype=torch.int32, device=device
        )
        token_ids[0, :5] = torch.tensor(
            [1, 2, 3, 4, 1], dtype=torch.int32, device=device
        )
        token_ids[1, :5] = torch.tensor(
            [7, 8, 9, 10, 11], dtype=torch.int32, device=device
        )
        token_ids[2, :4] = torch.tensor(
            [20, 21, 99, 20], dtype=torch.int32, device=device
        )

        valid_sampled = torch.tensor(
            [[2], [12], [21]], dtype=torch.int32, device=device
        )
        valid_count = torch.tensor([1, 1, 1], dtype=torch.int32, device=device)

        # 1. Full k = 4
        d4, c4 = proposer.propose(
            4, num_tokens, token_ids.clone(), valid_sampled, valid_count
        )
        assert d4.shape == (batch_size, 4)
        assert torch.equal(
            d4[0], torch.tensor([3, 4, 1, 2], dtype=torch.int32, device=device)
        )
        assert torch.equal(
            d4[1], torch.tensor([-1, -1, -1, -1], dtype=torch.int32, device=device)
        )
        assert torch.equal(
            d4[2], torch.tensor([99, 20, 21, -1], dtype=torch.int32, device=device)
        )
        assert c4[0].item() == 4
        assert c4[1].item() == 0
        assert c4[2].item() == 3

        # 2. Dynamic schedule k = 2 (< self.k)
        d2, c2 = proposer.propose(
            2, num_tokens, token_ids.clone(), valid_sampled, valid_count
        )
        assert d2.shape == (batch_size, 2)
        assert torch.equal(
            d2[0], torch.tensor([3, 4], dtype=torch.int32, device=device)
        )
        assert torch.equal(
            d2[1], torch.tensor([-1, -1], dtype=torch.int32, device=device)
        )
        assert torch.equal(
            d2[2], torch.tensor([99, 20], dtype=torch.int32, device=device)
        )
        assert c2[0].item() == 2
        assert c2[1].item() == 0
        assert c2[2].item() == 2

        # 3. Dynamic schedule k = 0 (scheduler picks 0 drafts for large batch)
        token_ids_step0 = token_ids.clone()
        d0, c0 = proposer.propose(
            0, num_tokens, token_ids_step0, valid_sampled, valid_count
        )
        assert d0.shape == (batch_size, 0)
        assert d0.dtype == torch.int32
        assert torch.equal(c0, torch.zeros(batch_size, dtype=c4.dtype, device=device))
        # Verify newly sampled tokens were correctly scattered in-place
        assert token_ids_step0[0, 5].item() == 2
        assert token_ids_step0[1, 5].item() == 12
        assert token_ids_step0[2, 4].item() == 21

        # 4. Sequential next step with k > 0 after k = 0
        num_tokens_step1 = torch.tensor([6, 6, 5], dtype=torch.int32, device=device)
        valid_sampled_step1 = torch.tensor(
            [[3], [13], [22]], dtype=torch.int32, device=device
        )
        d3, c3 = proposer.propose(
            3, num_tokens_step1, token_ids_step0, valid_sampled_step1, valid_count
        )
        assert d3.shape == (batch_size, 3)
        assert torch.equal(
            d3[0], torch.tensor([4, 1, 2], dtype=torch.int32, device=device)
        )
        assert c3[0].item() == 3

        # 5. Invalid k bounds
        with pytest.raises(AssertionError):
            proposer.propose(
                5, num_tokens, token_ids.clone(), valid_sampled, valid_count
            )
        with pytest.raises(AssertionError):
            proposer.propose(
                -1, num_tokens, token_ids.clone(), valid_sampled, valid_count
            )
