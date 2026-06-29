# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for per-request speculative-decoding stats accumulation.

These cover the pure histogram/per-step math (no GPU / no model): the engine-core
accumulator ``RequestSpecDecodeStats`` and its ``to_dict`` payload surfaced per
output sequence as ``choices[].speculative_decoding_stats``.
"""

import msgspec
import pytest

from vllm.outputs import CompletionOutput
from vllm.v1.engine import EngineCoreOutput
from vllm.v1.metrics.stats import RequestSpecDecodeStats


def _stats(pairs, num_spec_tokens=3, detailed=False):
    s = RequestSpecDecodeStats.new(num_spec_tokens)
    for k, j in pairs:
        s.observe(num_draft_tokens=k, num_accepted=j, detailed=detailed)
    return s


def test_new_allocates_dense_histogram_of_k_plus_one():
    s = RequestSpecDecodeStats.new(num_spec_tokens=3)
    assert s.num_spec_tokens == 3
    assert s.histogram == [0, 0, 0, 0]
    assert s.num_draft_tokens == 0
    assert s.per_step_accepted == []


def test_observe_buckets_by_accepted_draft_count():
    s = _stats([(3, 0), (3, 3), (3, 2), (3, 3), (3, 1)])
    # j=0 ->1, j=1 ->1, j=2 ->1, j=3 ->2
    assert s.histogram == [1, 1, 1, 2]
    assert s.num_draft_tokens == 15
    # summary level does not record the ordered per-step arrays
    assert s.per_step_accepted == []
    assert s.per_step_drafted == []


def test_observe_detailed_records_ordered_per_step_arrays():
    # Distinct step count (3), max draft length (k=4), and per-step drafted
    # counts (4, 2, 4) so no accidental "everything is 3" pattern is implied.
    s = _stats([(4, 3), (2, 2), (4, 0)], num_spec_tokens=4, detailed=True)
    assert s.per_step_accepted == [3, 2, 0]
    assert s.per_step_drafted == [4, 2, 4]
    # histogram (indexed by accepted j, length k+1=5) is still maintained
    assert s.histogram == [1, 0, 1, 1, 0]


def test_to_dict_summary_omits_per_step_arrays():
    d = _stats([(3, 0), (3, 3), (3, 2), (3, 3), (3, 1)]).to_dict()
    assert d == {
        "mean_acceptance_length": pytest.approx(1 + 9 / 5),  # j+1
        "draft_acceptance_rate": pytest.approx(9 / 15),
        "acceptance_histogram": {"0": 1, "1": 1, "2": 1, "3": 2},  # string keys
        "num_spec_steps": 5,
        "num_accepted_draft_tokens": 9,
        "num_draft_tokens": 15,
        "num_spec_tokens": 3,
    }


def test_to_dict_detailed_appends_per_step_arrays():
    d = _stats([(3, 3), (3, 2), (3, 0)], detailed=True).to_dict()
    assert d["per_step_accepted"] == [3, 2, 0]
    assert d["per_step_drafted"] == [3, 3, 3]
    # summary fields still present in detailed mode
    assert d["num_spec_steps"] == 3
    assert d["num_accepted_draft_tokens"] == 5


def test_to_dict_histogram_is_sparse_keyed_by_j():
    # dense would be [2, 0, 0, 1]; to_dict drops zero buckets, keys stringified
    d = _stats([(3, 0), (3, 0), (3, 3)]).to_dict()
    assert d["acceptance_histogram"] == {"0": 2, "3": 1}


def test_all_rejected_gives_mean_one_and_rate_zero():
    d = _stats([(2, 0), (2, 0), (2, 0), (2, 0)], num_spec_tokens=2).to_dict()
    assert d["num_spec_steps"] == 4
    assert d["num_accepted_draft_tokens"] == 0
    assert d["acceptance_histogram"] == {"0": 4}
    assert d["mean_acceptance_length"] == pytest.approx(1.0)
    assert d["draft_acceptance_rate"] == pytest.approx(0.0)


def test_empty_stats_do_not_divide_by_zero():
    d = RequestSpecDecodeStats.new(3).to_dict()
    assert d["num_spec_steps"] == 0
    assert d["num_draft_tokens"] == 0
    assert d["draft_acceptance_rate"] == 0.0
    assert d["mean_acceptance_length"] == 1.0
    assert "per_step_accepted" not in d


def test_observe_records_proposed_and_accepted_independently():
    # observe() takes proposed and accepted as independent inputs: the histogram
    # is keyed by accepted, num_draft_tokens sums the proposed as given. (The
    # grammar-invalidated-draft subtraction happens in the scheduler before
    # observe() -- see test_per_request_spec_decode_subtracts_invalid_drafts.)
    s = RequestSpecDecodeStats.new(num_spec_tokens=3)
    s.observe(num_draft_tokens=2, num_accepted=1)
    s.observe(num_draft_tokens=3, num_accepted=1)
    d = s.to_dict()
    assert d["acceptance_histogram"] == {"1": 2}  # both steps accepted 1
    assert d["num_draft_tokens"] == 5  # proposed summed independently: 2 + 3


def test_engine_core_output_round_trips_spec_decode_stats():
    # The accumulator rides EngineCoreOutput (msgspec, array_like) to the
    # frontend; verify it serializes (incl. per-step arrays) and is omitted
    # when absent. ``new_token_ids`` is EngineCoreOutput's required "tokens
    # generated this step" field -- a dummy value here since we only exercise
    # spec_decode_stats.
    stats = _stats([(3, 0), (3, 3), (3, 2)], detailed=True)
    out = EngineCoreOutput(
        request_id="r1", new_token_ids=[1, 2], spec_decode_stats=stats
    )
    decoder = msgspec.msgpack.Decoder(EngineCoreOutput)
    decoded = decoder.decode(msgspec.msgpack.encode(out))
    assert decoded.spec_decode_stats.histogram == [1, 0, 1, 1]
    assert decoded.spec_decode_stats.num_draft_tokens == 9
    assert decoded.spec_decode_stats.per_step_accepted == [0, 3, 2]

    without = EngineCoreOutput(request_id="r2", new_token_ids=[1])
    decoded_without = decoder.decode(msgspec.msgpack.encode(without))
    assert decoded_without.spec_decode_stats is None


def _completion_output(**kwargs):
    return CompletionOutput(
        index=0,
        text="",
        token_ids=[],
        cumulative_logprob=None,
        logprobs=None,
        **kwargs,
    )


def test_completion_output_carries_spec_decode_stats():
    # Stats are per output sequence, so they ride the CompletionOutput
    # (=> choices[i]), not the request-level RequestOutput.
    stats = _stats([(3, 3), (3, 1)])
    out = _completion_output(spec_decode_stats=stats)
    assert out.spec_decode_stats is stats
    assert _completion_output().spec_decode_stats is None
