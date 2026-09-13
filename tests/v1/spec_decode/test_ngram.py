# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import numpy as np
import pytest

from vllm.config import (
    ModelConfig,
    ParallelConfig,
    SpeculativeConfig,
    VllmConfig,
)
from vllm.utils.torch_utils import available_cpu_count
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


def _make_proposer(
    parallel_config: ParallelConfig | None = None,
    min_n: int = 2,
    max_n: int = 2,
    k: int = 2,
    # 16 attention heads, so every tensor_parallel_size below divides evenly.
    model: str = "Qwen/Qwen3-0.6B",
) -> NgramProposer:
    return NgramProposer(
        vllm_config=VllmConfig(
            model_config=ModelConfig(model=model),
            parallel_config=parallel_config or ParallelConfig(),
            speculative_config=SpeculativeConfig(
                prompt_lookup_min=min_n,
                prompt_lookup_max=max_n,
                num_speculative_tokens=k,
                method="ngram",
            ),
        )
    )


def _parallel_config(rank: int, **kwargs) -> ParallelConfig:
    parallel_config = ParallelConfig(**kwargs)
    # Set as the worker does (WorkerWrapperBase.init_worker).
    parallel_config.rank = rank
    return parallel_config


_TP8_PP4 = {"tensor_parallel_size": 8, "pipeline_parallel_size": 4, "nnodes": 4}
_TP16 = {"tensor_parallel_size": 16, "nnodes": 2}
_EXTERNAL = {
    "tensor_parallel_size": 4,
    "distributed_executor_backend": "external_launcher",
}


@pytest.mark.parametrize(
    "rank,kwargs,expected_leader",
    [
        # Single rank: always the leader.
        (0, {}, True),
        # TP only: the engine core reads drafts from rank 0 alone.
        (0, {"tensor_parallel_size": 4}, True),
        (1, {"tensor_parallel_size": 4}, False),
        (3, {"tensor_parallel_size": 4}, False),
        # TP+PP: Executor._get_output_rank() is world_size - tp_size = 24 for
        # TP=8/PP=4, which `rank % tp_size == 0` must not exclude. Ranks 0/8/16
        # also match but sit in earlier PP stages, where the model runner never
        # builds a proposer. First TP rank on last PP will be the draft leader.
        (24, _TP8_PP4, True),
        (25, _TP8_PP4, False),
        (31, _TP8_PP4, False),
        # Multi-node TP group: rank arithmetic is unchanged, so the leader is
        # the first rank of the group (on node 0) and every other rank skips.
        (0, _TP16, True),
        (8, _TP16, False),
        # external_launcher runs one scheduler per rank with no draft
        # broadcast, so every rank must draft for itself.
        (0, _EXTERNAL, True),
        (3, _EXTERNAL, True),
    ],
)
def test_ngram_leader_rank_selection(rank, kwargs, expected_leader):
    proposer = _make_proposer(_parallel_config(rank, **kwargs))
    assert proposer.is_leader is expected_leader


@pytest.mark.parametrize("rank,expected_leader", [(0, True), (1, True)])
def test_ngram_leader_selection_with_expert_parallel(rank, expected_leader):
    """EP adds no ranks and does not change the layout. With TP=1 every rank is
    its own DP engine, so each one stays a leader."""
    proposer = NgramProposer(
        vllm_config=VllmConfig(
            model_config=ModelConfig(model="Qwen/Qwen3-30B-A3B", enforce_eager=True),
            parallel_config=_parallel_config(
                rank, tensor_parallel_size=1, enable_expert_parallel=True
            ),
            speculative_config=SpeculativeConfig(
                prompt_lookup_min=2,
                prompt_lookup_max=2,
                num_speculative_tokens=2,
                method="ngram",
            ),
        )
    )
    assert proposer.is_leader is expected_leader


def test_ngram_non_leader_skips_lookup():
    """A non-leader rank returns empty drafts and leaves its buffers alone."""
    token_ids_cpu = np.array([[1, 2, 3, 1, 2], [7, 8, 9, 7, 8]], dtype=np.int32)
    num_tokens_no_spec = np.array([5, 5], dtype=np.int32)
    kwargs = {"tensor_parallel_size": 4}

    leader = _make_proposer(_parallel_config(0, **kwargs))
    leader_result = leader.propose(
        num_speculative_tokens=2,
        sampled_token_ids=[[0], [1]],
        num_tokens_no_spec=num_tokens_no_spec,
        token_ids_cpu=token_ids_cpu,
    )
    assert np.array_equal(leader_result[0], [3, 1])
    assert np.array_equal(leader_result[1], [9, 7])

    follower = _make_proposer(_parallel_config(1, **kwargs))
    follower_result = follower.propose(
        num_speculative_tokens=2,
        sampled_token_ids=[[0], [1]],
        num_tokens_no_spec=num_tokens_no_spec,
        token_ids_cpu=token_ids_cpu,
    )
    assert follower_result == [[], []]
    assert not follower.valid_ngram_num_drafts[:2].any()


@pytest.mark.parametrize("tp_size,nnodes", [(1, 1), (2, 1), (4, 1), (8, 2), (16, 4)])
def test_ngram_thread_budget_is_not_split_across_tp_ranks(tp_size, nnodes):
    """Only the leader drafts, so its budget must not shrink with TP size."""
    proposer = _make_proposer(
        _parallel_config(0, tensor_parallel_size=tp_size, nnodes=nnodes)
    )
    assert proposer.num_numba_thread_available == max(
        1, min(8, available_cpu_count() // 2)
    )
    # Never 0, which used to happen once tp_size exceeded the core count.
    assert proposer.num_numba_thread_available >= 1


def test_ngram_thread_budget_shared_when_every_rank_drafts():
    """external_launcher ranks draft concurrently, so they share the node."""
    parallel_config = _parallel_config(
        0,
        tensor_parallel_size=4,
        distributed_executor_backend="external_launcher",
    )
    proposer = _make_proposer(parallel_config)
    assert proposer.num_numba_thread_available == max(
        1, min(8, available_cpu_count() // 2 // parallel_config.local_world_size)
    )


@pytest.mark.parametrize("batch", [1, 8, 64])
def test_ngram_drafts_do_not_depend_on_thread_count(batch):
    """Raising the thread cap must not change a single drafted token: the numba
    kernel parallelises over requests, so each request's draft is independent."""
    rng = np.random.default_rng(1234)
    ctx = 4096
    token_ids_cpu = rng.integers(0, 32000, size=(batch, ctx), dtype=np.int32)
    # Make the suffix repeat an earlier span so every request finds a match.
    for row in range(batch):
        src = int(rng.integers(0, ctx - 128))
        token_ids_cpu[row, ctx - 64 :] = token_ids_cpu[row, src : src + 64]
    num_tokens_no_spec = np.full(batch, ctx, dtype=np.int32)
    sampled_token_ids = [[1]] * batch

    results = []
    for threads in (1, 8):
        proposer = _make_proposer(min_n=3, max_n=5, k=5)
        proposer.valid_ngram_draft = np.zeros((batch, proposer.k), dtype=np.int32)
        proposer.valid_ngram_num_drafts = np.zeros(batch, dtype=np.int32)
        proposer.max_model_len = 2 * ctx
        proposer.num_numba_thread_available = threads
        results.append(
            proposer.propose(
                num_speculative_tokens=5,
                sampled_token_ids=sampled_token_ids,
                num_tokens_no_spec=num_tokens_no_spec,
                token_ids_cpu=token_ids_cpu,
            )
        )
    assert results[0] == results[1]
    # Guard against the test passing because nothing was drafted at all.
    assert any(len(draft) for draft in results[0])
