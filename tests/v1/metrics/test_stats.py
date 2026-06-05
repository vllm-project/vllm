# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from unittest.mock import MagicMock, patch

from vllm.v1.engine import FinishReason
from vllm.v1.metrics.loggers import (
    AggregatedLoggingStatLogger,
    LoggingStatLogger,
    StatLoggerManager,
)
from vllm.v1.metrics.stats import (
    IterationStats,
    PrefillStats,
    PromptTokenStats,
    RequestStateStats,
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


# ---------------------------------------------------------------------------
# StatLoggerManager multi-API-server logging tests (no GPU required)
# ---------------------------------------------------------------------------

_LOGGERS_MODULE = "vllm.v1.metrics.loggers"


def _logging_factory_was_used(client_count: int, client_index: int) -> bool:
    """Return True if LoggingStatLogger was registered as a factory.
    Patches out PrometheusStatLogger (needs real config) and
    LoggingStatLogger (needs real config) so the test runs without GPU.
    Checks whether the patched LoggingStatLogger mock was called as a
    factory — which only happens when logging is enabled for this server.
    """
    mock_config = MagicMock()
    with (
        patch(f"{_LOGGERS_MODULE}.PrometheusStatLogger"),
        patch(f"{_LOGGERS_MODULE}.LoggingStatLogger") as mock_logging_cls,
    ):
        StatLoggerManager(
            vllm_config=mock_config,
            enable_default_loggers=True,
            client_count=client_count,
            client_index=client_index,
        )
    return mock_logging_cls.called


def test_stat_logger_manager_single_server():
    """With api_server_count=1 the default logging factory must be active."""
    assert _logging_factory_was_used(client_count=1, client_index=0), (
        "LoggingStatLogger should be active for a single API server"
    )


def test_stat_logger_manager_multi_server_primary():
    """With api_server_count>1 the primary server (index 0) must log."""
    assert _logging_factory_was_used(client_count=4, client_index=0), (
        "LoggingStatLogger should be active on the primary API server (index 0)"
    )


def test_stat_logger_manager_multi_server_non_primary():
    """With api_server_count>1 non-primary servers must NOT log."""
    for idx in range(1, 4):
        assert not _logging_factory_was_used(client_count=4, client_index=idx), (
            f"LoggingStatLogger should be silent on server index {idx}"
        )