# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest

from vllm.v1.engine import FinishReason
from vllm.v1.metrics.stats import (
    IterationStats,
    PrefillStats,
    PromptTokenStats,
    RequestStateStats,
    compute_timing_intervals,
)


def test_iteration_stats_repr():
    iteration_stats = IterationStats()
    assert repr(iteration_stats).startswith("IterationStats(")


def test_prefill_kv_computed_with_cache():
    """Test that prefill KV compute correctly excludes cached tokens."""
    iteration_stats = IterationStats()
    req_stats = RequestStateStats(arrival_time=0.0)
    req_stats.scheduled_ts = 0.1
    req_stats.first_token_ts = 0.5
    req_stats.last_token_ts = 5.0
    req_stats.num_generation_tokens = 50

    # Case 1: With prefix cache (1200 tokens cached)
    iteration_stats.update_from_finished_request(
        finish_reason=FinishReason.STOP,
        request_id="test-req-001",
        num_prompt_tokens=10000,
        max_tokens_param=100,
        req_stats=req_stats,
        num_cached_tokens=1200,
    )

    finished_req = iteration_stats.finished_requests[0]
    assert finished_req.num_prompt_tokens == 10000
    assert finished_req.num_cached_tokens == 1200
    assert finished_req.request_id == "test-req-001"

    # Verify calculation: prefill KV = prompt tokens - cached tokens
    prefill_kv_computed = finished_req.num_prompt_tokens - max(
        finished_req.num_cached_tokens, 0
    )
    assert prefill_kv_computed == 8800  # 10000 - 1200


def test_prefill_kv_computed_no_cache():
    """Test prefill KV compute without prefix caching."""
    iteration_stats = IterationStats()
    req_stats = RequestStateStats(arrival_time=0.0)
    req_stats.scheduled_ts = 0.1
    req_stats.first_token_ts = 0.5
    req_stats.last_token_ts = 2.0
    req_stats.num_generation_tokens = 10

    # Case 2: No prefix cache
    iteration_stats.update_from_finished_request(
        finish_reason=FinishReason.STOP,
        request_id="test-req-002",
        num_prompt_tokens=2000,
        max_tokens_param=100,
        req_stats=req_stats,
        num_cached_tokens=0,
    )

    finished_req = iteration_stats.finished_requests[0]
    assert finished_req.num_prompt_tokens == 2000
    assert finished_req.num_cached_tokens == 0
    assert finished_req.request_id == "test-req-002"

    # Verify calculation: prefill KV = full prompt when no cache
    prefill_kv_computed = finished_req.num_prompt_tokens - max(
        finished_req.num_cached_tokens, 0
    )
    assert prefill_kv_computed == 2000


def test_prefill_kv_computed_edge_cases():
    """Test edge cases for prefill KV compute calculation."""
    iteration_stats = IterationStats()
    req_stats = RequestStateStats(arrival_time=0.0)
    req_stats.scheduled_ts = 0.1
    req_stats.first_token_ts = 0.5
    req_stats.last_token_ts = 1.0
    req_stats.num_generation_tokens = 1

    # Case 3: Negative num_cached_tokens (shouldn't happen, but handle gracefully)
    iteration_stats.update_from_finished_request(
        finish_reason=FinishReason.STOP,
        request_id="test-req-003",
        num_prompt_tokens=100,
        max_tokens_param=10,
        req_stats=req_stats,
        num_cached_tokens=-1,
    )

    finished_req = iteration_stats.finished_requests[0]
    # max() should handle negative values
    prefill_kv_computed = finished_req.num_prompt_tokens - max(
        finished_req.num_cached_tokens, 0
    )
    assert prefill_kv_computed == 100  # Should treat negative as 0
    assert finished_req.request_id == "test-req-003"

    # Case 4: All tokens cached (shouldn't happen in practice)
    iteration_stats2 = IterationStats()
    iteration_stats2.update_from_finished_request(
        finish_reason=FinishReason.STOP,
        request_id="test-req-004",
        num_prompt_tokens=100,
        max_tokens_param=10,
        req_stats=req_stats,
        num_cached_tokens=100,
    )

    finished_req2 = iteration_stats2.finished_requests[0]
    prefill_kv_computed2 = finished_req2.num_prompt_tokens - max(
        finished_req2.num_cached_tokens, 0
    )
    assert prefill_kv_computed2 == 0  # All cached, nothing computed
    assert finished_req2.request_id == "test-req-004"


def test_prompt_token_stats_all_computed():
    """Test all tokens computed locally, no caching."""
    stats = PromptTokenStats()

    # Case 1: No caching (All tokens computed locally)
    prefill_stats = PrefillStats()
    prefill_stats.set(
        num_prompt_tokens=1000,
        num_local_cached_tokens=0,
        num_external_cached_tokens=0,
    )
    stats.update_from_output(prefill_stats)

    assert stats.computed == 1000
    assert stats.local_cache_hit == 0
    assert stats.external_kv_transfer == 0
    assert stats.cached_tokens == 0
    assert stats.total == 1000


def test_prompt_token_stats_partial_local_cache():
    """Test partial local prefix cache hit."""
    stats = PromptTokenStats()

    # Case 2: Partial local cache
    prefill_stats = PrefillStats()
    prefill_stats.set(
        num_prompt_tokens=1000,
        num_local_cached_tokens=300,
        num_external_cached_tokens=0,
    )
    stats.update_from_output(prefill_stats)

    assert stats.computed == 700
    assert stats.local_cache_hit == 300
    assert stats.external_kv_transfer == 0
    assert stats.cached_tokens == 300
    assert stats.total == 1000


def test_prompt_token_stats_partial_external_transfer():
    """Test partial external KV transfer."""
    stats = PromptTokenStats()

    # Case 3: Partial external transfer
    prefill_stats = PrefillStats()
    prefill_stats.set(
        num_prompt_tokens=1000,
        num_local_cached_tokens=0,
        num_external_cached_tokens=500,
    )
    stats.update_from_output(prefill_stats)

    assert stats.computed == 500
    assert stats.local_cache_hit == 0
    assert stats.external_kv_transfer == 500
    assert stats.cached_tokens == 500
    assert stats.total == 1000


def test_prompt_token_stats_mixed_sources():
    """Test mix of local cache and external transfer."""
    stats = PromptTokenStats()

    # Case 4: Mixed sources
    prefill_stats = PrefillStats()
    prefill_stats.set(
        num_prompt_tokens=1000,
        num_local_cached_tokens=400,
        num_external_cached_tokens=200,
    )
    stats.update_from_output(prefill_stats)

    assert stats.computed == 400
    assert stats.local_cache_hit == 400
    assert stats.external_kv_transfer == 200
    assert stats.cached_tokens == 600
    assert stats.total == 1000


def test_prompt_token_stats_full_local_cache_recompute():
    """Test full local cache triggers last token recomputation.

    When all tokens are cached, the scheduler forces the model to recompute
    the last token (num_computed_tokens=1), with the rest from cache.
    """
    stats = PromptTokenStats()

    # Case 5: Full local cache (999 cached, 1 recomputed)
    prefill_stats = PrefillStats()
    prefill_stats.set(
        num_prompt_tokens=1000,
        num_local_cached_tokens=999,
        num_external_cached_tokens=0,
    )
    stats.update_from_output(prefill_stats)

    assert stats.computed == 1
    assert stats.local_cache_hit == 999
    assert stats.external_kv_transfer == 0
    assert stats.cached_tokens == 999
    assert stats.total == 1000


def test_prompt_token_stats_full_external_transfer_recompute():
    """Test full external transfer triggers last token recomputation."""
    stats = PromptTokenStats()

    # Case 6: Full external transfer (999 from external, 1 recomputed)
    prefill_stats = PrefillStats()
    prefill_stats.set(
        num_prompt_tokens=1000,
        num_local_cached_tokens=0,
        num_external_cached_tokens=999,
    )
    stats.update_from_output(prefill_stats)

    assert stats.computed == 1
    assert stats.local_cache_hit == 0
    assert stats.external_kv_transfer == 999
    assert stats.cached_tokens == 999
    assert stats.total == 1000


def test_update_from_finished_request_returns_finished_stats():
    """update_from_finished_request returns the same FinishedRequestStats it appends."""
    iteration_stats = IterationStats()
    req_stats = RequestStateStats(
        arrival_time=100.0,
        queued_ts=100.05,
        scheduled_ts=100.10,
        first_token_ts=100.20,
        last_token_ts=100.50,
        num_generation_tokens=5,
    )

    returned = iteration_stats.update_from_finished_request(
        finish_reason=FinishReason.STOP,
        request_id="req-1",
        num_prompt_tokens=10,
        max_tokens_param=None,
        req_stats=req_stats,
        num_cached_tokens=3,
    )

    assert returned is not None
    assert iteration_stats.finished_requests[-1] is returned
    assert returned.request_id == "req-1"
    assert returned.num_cached_tokens == 3


def test_update_from_finished_request_no_negative_intervals():
    # Aborted before first token: first_token_ts/last_token_ts == 0.0.
    iteration_stats = IterationStats()
    req_stats = RequestStateStats(arrival_time=0.0)
    req_stats.queued_ts = 1.0
    req_stats.scheduled_ts = 1.5
    # first_token_ts and last_token_ts remain 0.0

    finished = iteration_stats.update_from_finished_request(
        finish_reason=FinishReason.ABORT,
        request_id="aborted-req",
        num_prompt_tokens=10,
        max_tokens_param=None,
        req_stats=req_stats,
    )

    # Previously these were negative (0.0 - 1.5). Now clamped to 0.0.
    assert finished.prefill_time == 0.0
    assert finished.decode_time == 0.0
    assert finished.inference_time == 0.0
    assert finished.mean_time_per_output_token == 0.0
    assert finished.queued_time == pytest.approx(0.5)  # measured, unchanged


def test_update_from_finished_request_populated_unchanged():
    iteration_stats = IterationStats()
    req_stats = RequestStateStats(arrival_time=0.0)
    req_stats.queued_ts = 1.0
    req_stats.scheduled_ts = 1.5
    req_stats.first_token_ts = 2.0
    req_stats.last_token_ts = 3.0
    req_stats.num_generation_tokens = 2

    finished = iteration_stats.update_from_finished_request(
        finish_reason=FinishReason.STOP,
        request_id="ok-req",
        num_prompt_tokens=10,
        max_tokens_param=None,
        req_stats=req_stats,
    )

    assert finished.queued_time == pytest.approx(0.5)
    assert finished.prefill_time == pytest.approx(0.5)
    assert finished.decode_time == pytest.approx(1.0)
    assert finished.inference_time == pytest.approx(1.5)
    assert finished.mean_time_per_output_token == pytest.approx(1.0)  # 1.0/(2-1)


def test_update_from_finished_request_single_token_mean_is_zero():
    # A request that finishes with exactly one output token has no
    # inter-token interval: the core returns None for mean_per_output_token
    # (n-1 == 0), which the engine path must coerce to 0.0 for Prometheus.
    iteration_stats = IterationStats()
    req_stats = RequestStateStats(arrival_time=0.0)
    req_stats.queued_ts = 1.0
    req_stats.scheduled_ts = 1.5
    req_stats.first_token_ts = 2.0
    req_stats.last_token_ts = 2.0
    req_stats.num_generation_tokens = 1

    finished = iteration_stats.update_from_finished_request(
        finish_reason=FinishReason.STOP,
        request_id="single-token-req",
        num_prompt_tokens=10,
        max_tokens_param=None,
        req_stats=req_stats,
    )

    assert finished.mean_time_per_output_token == 0.0
    assert finished.queued_time == pytest.approx(0.5)
    assert finished.prefill_time == pytest.approx(0.5)
    # decode == last - first == 0.0 (measured but zero, single token)
    assert finished.decode_time == 0.0


def test_compute_timing_intervals_fully_populated():
    stats = RequestStateStats(
        queued_ts=1.0,
        scheduled_ts=1.5,
        first_token_ts=2.0,
        last_token_ts=3.0,
        num_generation_tokens=2,
    )
    iv = compute_timing_intervals(stats, num_generation_tokens=10)
    assert iv.queue == pytest.approx(0.5)
    assert iv.prefill == pytest.approx(0.5)
    assert iv.decode == pytest.approx(1.0)
    assert iv.inference == pytest.approx(1.5)
    assert iv.mean_per_output_token == pytest.approx(1.0 / 9)
    assert iv.tokens_per_second == pytest.approx(10.0 / 1.5)


def test_compute_timing_intervals_missing_first_token_is_none():
    # Aborted before first token: first_token_ts/last_token_ts never set.
    stats = RequestStateStats(queued_ts=1.0, scheduled_ts=1.5)
    iv = compute_timing_intervals(stats, num_generation_tokens=0)
    assert iv.queue == pytest.approx(0.5)
    assert iv.prefill is None  # not negative (was 0 - 1.5 = -1.5)
    assert iv.decode is None
    assert iv.inference is None
    assert iv.mean_per_output_token is None
    assert iv.tokens_per_second is None


def test_compute_timing_intervals_mean_itl_needs_two_tokens():
    stats = RequestStateStats(
        queued_ts=1.0, scheduled_ts=1.5, first_token_ts=2.0, last_token_ts=3.0
    )
    iv = compute_timing_intervals(stats, num_generation_tokens=1)
    assert iv.mean_per_output_token is None  # n-1 == 0
