# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from vllm.v1.engine import FinishReason
from vllm.v1.metrics.stats import IterationStats, PromptTokenStats, RequestStateStats


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
        num_prompt_tokens=10000,
        max_tokens_param=100,
        req_stats=req_stats,
        num_cached_tokens=1200,
    )

    finished_req = iteration_stats.finished_requests[0]
    assert finished_req.num_prompt_tokens == 10000
    assert finished_req.num_cached_tokens == 1200

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
        num_prompt_tokens=2000,
        max_tokens_param=100,
        req_stats=req_stats,
        num_cached_tokens=0,
    )

    finished_req = iteration_stats.finished_requests[0]
    assert finished_req.num_prompt_tokens == 2000
    assert finished_req.num_cached_tokens == 0

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

    # Case 4: All tokens cached (shouldn't happen in practice)
    iteration_stats2 = IterationStats()
    iteration_stats2.update_from_finished_request(
        finish_reason=FinishReason.STOP,
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


def test_prompt_token_stats_all_computed():
    """Test all tokens computed locally, no caching."""
    stats = PromptTokenStats()

    # Case 1: No caching (All tokens computed locally)
    stats.update_from_output(
        num_cached_tokens=0,
        num_external_computed_tokens=0,
        prompt_len=1000,
    )

    assert stats.computed == 1000
    assert stats.local_cache_hit == 0
    assert stats.external_kv_transfer == 0
    assert stats.total == 1000


def test_prompt_token_stats_partial_local_cache():
    """Test partial local prefix cache hit."""
    stats = PromptTokenStats()

    # Case 2: Partial local cache
    stats.update_from_output(
        num_cached_tokens=300,
        num_external_computed_tokens=0,
        prompt_len=1000,
    )

    assert stats.computed == 700
    assert stats.local_cache_hit == 300
    assert stats.external_kv_transfer == 0


def test_prompt_token_stats_partial_external_transfer():
    """Test partial external KV transfer."""
    stats = PromptTokenStats()

    # Case 3: Partial external transfer
    stats.update_from_output(
        num_cached_tokens=500,
        num_external_computed_tokens=500,
        prompt_len=1000,
    )

    assert stats.computed == 500
    assert stats.local_cache_hit == 0
    assert stats.external_kv_transfer == 500


def test_prompt_token_stats_mixed_sources():
    """Test mix of local cache and external transfer."""
    stats = PromptTokenStats()

    # Case 4: Mixed sources
    stats.update_from_output(
        num_cached_tokens=600,
        num_external_computed_tokens=200,
        prompt_len=1000,
    )

    assert stats.computed == 400
    assert stats.local_cache_hit == 400
    assert stats.external_kv_transfer == 200


def test_prompt_token_stats_full_local_cache_recompute():
    """Test full local cache triggers last token recomputation.

    When all tokens are cached, the scheduler reduces num_cached_tokens by 1
    to force the model to recompute the last token.
    """
    stats = PromptTokenStats()

    # Case 5: Full local cache (999 cached after reduction, 1 recomputed)
    stats.update_from_output(
        num_cached_tokens=999,
        num_external_computed_tokens=0,
        prompt_len=1000,
    )

    assert stats.computed == 1
    assert stats.local_cache_hit == 1000
    assert stats.recomputed_tokens == 1


def test_prompt_token_stats_full_external_transfer_recompute():
    """Test full external transfer triggers last token recomputation."""
    stats = PromptTokenStats()

    # Case 6: Full external transfer (999 cached after reduction, 1 recomputed)
    stats.update_from_output(
        num_cached_tokens=999,
        num_external_computed_tokens=1000,
        prompt_len=1000,
    )

    assert stats.computed == 1
    assert stats.local_cache_hit == 0
    assert stats.external_kv_transfer == 1000
    assert stats.recomputed_tokens == 1


def test_prompt_token_stats_pd_disagg_external_exceeds_cached():
    """Test P/D disagg case where external tokens exceed cached tokens.

    In P/D disaggregation, the decode instance may receive more tokens via
    external KV transfer than it has cached locally. This previously caused
    negative local_cache_hit values which crashed Prometheus counters.
    """
    stats = PromptTokenStats()

    # Case: Decode receives 7000 tokens from prefill, but has 0 local cache
    # This should NOT result in negative local_cache_hit
    stats.update_from_output(
        num_cached_tokens=0,
        num_external_computed_tokens=7000,
        prompt_len=7000,
    )

    assert stats.computed == 7000  # prompt_len - num_cached_tokens
    assert stats.local_cache_hit == 0  # Should be clamped to 0, not -7000
    assert stats.external_kv_transfer == 7000
    assert stats.total == 7000

    # Verify all values are non-negative (required for Prometheus counters)
    assert stats.computed >= 0
    assert stats.local_cache_hit >= 0
    assert stats.external_kv_transfer >= 0
    assert stats.cached_tokens >= 0
    assert stats.recomputed_tokens >= 0


def test_prompt_token_stats_pd_disagg_partial_overlap():
    """Test P/D disagg with partial overlap between external and cached."""
    stats = PromptTokenStats()

    # Case: Some local cache, but more external transfer
    # num_cached_tokens=100, num_external=500
    # Old behavior: local_cache_hit = 100 - 500 = -400 (BUG!)
    # New behavior: local_cache_hit = max(0, 100 - 500) = 0
    stats.update_from_output(
        num_cached_tokens=100,
        num_external_computed_tokens=500,
        prompt_len=1000,
    )

    assert stats.computed == 900  # 1000 - 100
    assert stats.local_cache_hit == 0  # Clamped from -400
    assert stats.external_kv_transfer == 500
    assert stats.local_cache_hit >= 0  # Must be non-negative for Prometheus
