# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import numpy as np
import torch

from vllm.config import (
    ModelConfig,
    SpeculativeConfig,
    VllmConfig,
)
from vllm.v1.core.sched.output import CachedRequestData, SchedulerOutput
from vllm.v1.spec_decode.ngram_proposer import (
    NgramProposer,
    _find_longest_matched_ngram_and_propose_tokens,
)
from vllm.v1.spec_decode.ngram_proposer_gpu import (
    update_scheduler_for_invalid_drafts,
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


def _spec_scheduler_output(
    req_ids: list[str],
    spec_lens: dict[str, int],
) -> SchedulerOutput:
    """A decode-step SchedulerOutput carrying `spec_lens[req_id]` draft slots each."""
    spec_tokens = {
        req_id: list(range(1000, 1000 + n)) for req_id, n in spec_lens.items() if n > 0
    }
    num_scheduled = {r: 1 + spec_lens.get(r, 0) for r in req_ids}
    return SchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=CachedRequestData(
            req_ids=list(req_ids),
            resumed_req_ids=set(),
            new_token_ids=[[] for _ in req_ids],
            all_token_ids={},
            new_block_ids=[None] * len(req_ids),
            num_computed_tokens=[0] * len(req_ids),
            num_output_tokens=[0] * len(req_ids),
        ),
        num_scheduled_tokens=num_scheduled,
        total_num_scheduled_tokens=sum(num_scheduled.values()),
        scheduled_spec_decode_tokens=spec_tokens,
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=[],
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
    )


class _RecordingEvent:
    """Stands in for the D2H-completion event and records whether it was awaited."""

    def __init__(self):
        self.syncs = 0

    def synchronize(self):
        self.syncs += 1


def test_update_scheduler_for_invalid_drafts_trims_to_valid_counts():
    """Each request's spec slots are trimmed to its valid draft count.

    The totals are the part worth pinning: `total_num_scheduled_tokens` has to stay
    equal to `sum(num_scheduled_tokens.values())`, because the model runner sizes its
    input buffers from it.
    """
    k = 3
    req_ids = [f"req-{i}" for i in range(4)]
    out = _spec_scheduler_output(req_ids, {r: k for r in req_ids})
    req_id_to_index = {r: i for i, r in enumerate(req_ids)}
    # Fully valid, partly valid, nothing valid, and a count above what was scheduled.
    valid = torch.tensor([3, 1, 0, 7], dtype=torch.int32)
    event = _RecordingEvent()

    update_scheduler_for_invalid_drafts(event, valid, out, req_id_to_index)

    assert event.syncs == 1
    assert out.scheduled_spec_decode_tokens["req-0"] == [1000, 1001, 1002]
    assert out.scheduled_spec_decode_tokens["req-1"] == [1000]
    # A request with no valid drafts is dropped, not left with an empty list.
    assert "req-2" not in out.scheduled_spec_decode_tokens
    # A count above the scheduled width is clamped, never used to grow the request.
    assert out.scheduled_spec_decode_tokens["req-3"] == [1000, 1001, 1002]
    assert out.num_scheduled_tokens == {
        "req-0": 4,
        "req-1": 2,
        "req-2": 1,
        "req-3": 4,
    }
    assert out.total_num_scheduled_tokens == sum(out.num_scheduled_tokens.values())


def test_update_scheduler_for_invalid_drafts_ignores_unknown_requests():
    """Requests absent from the batch index or from the spec dict are skipped."""
    req_ids = ["in-batch", "no-spec", "not-in-index"]
    out = _spec_scheduler_output(req_ids, {"in-batch": 2, "not-in-index": 2})
    # "not-in-index" was speculated but has since left the running batch.
    req_id_to_index = {"in-batch": 0, "no-spec": 1}
    valid = torch.tensor([1, 0, 0], dtype=torch.int32)

    before_total = out.total_num_scheduled_tokens
    update_scheduler_for_invalid_drafts(_RecordingEvent(), valid, out, req_id_to_index)

    assert out.scheduled_spec_decode_tokens["in-batch"] == [1000]
    assert out.scheduled_spec_decode_tokens["not-in-index"] == [1000, 1001]
    assert out.num_scheduled_tokens["no-spec"] == 1
    assert out.total_num_scheduled_tokens == before_total - 1


def test_update_scheduler_for_invalid_drafts_no_speculation_is_a_noop():
    """With nothing speculated there is nothing to trim and nothing to wait for.

    `scheduled_spec_decode_tokens` omits requests without draft tokens entirely, so an
    empty mapping means the whole step speculated nothing. Awaiting the D2H copy in
    that case buys nothing, and the buffer is never read.
    """
    req_ids = ["req-0", "req-1"]
    out = _spec_scheduler_output(req_ids, {})
    assert out.scheduled_spec_decode_tokens == {}
    before = (dict(out.num_scheduled_tokens), out.total_num_scheduled_tokens)
    event = _RecordingEvent()

    update_scheduler_for_invalid_drafts(
        event, torch.zeros(2, dtype=torch.int32), out, {"req-0": 0, "req-1": 1}
    )

    assert (dict(out.num_scheduled_tokens), out.total_num_scheduled_tokens) == before
    assert out.scheduled_spec_decode_tokens == {}
    assert event.syncs == 0
