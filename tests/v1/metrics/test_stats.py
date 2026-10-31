# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.v1.core.sched.output import ScheduledEncoderInputStats, SchedulerOutput
from vllm.v1.engine import EngineCoreOutput, EngineCoreOutputs, FinishReason
from vllm.v1.metrics.cache_hit_source import CacheHitSource
from vllm.v1.metrics.stats import (
    IterationStats,
    LoRARequestStates,
    PrefillStats,
    PrefixCacheStats,
    PromptTokenStats,
    RequestStateStats,
    SchedulerIterationDetails,
    SchedulerStats,
)
from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder
from vllm.v1.utils import compute_iteration_details


def test_iteration_stats_repr():
    iteration_stats = IterationStats()
    assert repr(iteration_stats).startswith("IterationStats(")


@pytest.mark.parametrize("enabled", [False, True])
def test_iteration_stats_preserve_api_surface_in_mixed_batch(enabled):
    """Retain request identity and collect API token totals only when enabled."""
    stats = IterationStats(enable_api_surface_metrics=enabled)
    stats.iteration_timestamp = 102.0
    api_surfaces = ("messages", "chat_completions", "completions", "unknown")
    for index, api_surface in enumerate(api_surfaces):
        req_stats = RequestStateStats(
            arrival_time=100.0 - index, api_surface=api_surface
        )
        prefill_stats = PrefillStats()
        prefill_stats.set(10 + index, 2, 3)
        output = EngineCoreOutput(
            request_id=api_surface,
            new_token_ids=[1, 2],
            prefill_stats=prefill_stats,
        )
        for is_prefilling, timestamp in ((True, 10.0), (False, 11.0 + index)):
            stats.update_from_output(
                output,
                timestamp,
                is_prefilling,
                req_stats,
                LoRARequestStates(),
                lora_name=None,
            )
        stats.update_from_finished_request(
            FinishReason.STOP, api_surface, 10, None, req_stats
        )

    assert stats.time_to_first_tokens_iter == [
        (api_surface, 2.0 + index) for index, api_surface in enumerate(api_surfaces)
    ]
    assert stats.inter_token_latencies_iter == [
        (api_surface, 1.0 + index) for index, api_surface in enumerate(api_surfaces)
    ]
    assert [request.api_surface for request in stats.finished_requests] == list(
        api_surfaces
    )
    assert stats.num_generation_tokens == 16
    assert stats.num_prompt_tokens == sum(10 + index for index in range(4))
    assert stats.prompt_token_stats.cached_tokens == 20
    if enabled:
        assert stats.num_generation_tokens_by_api_surface == dict.fromkeys(
            api_surfaces, 4
        )
        assert stats.prompt_token_stats_by_api_surface is not None
        for index, api_surface in enumerate(api_surfaces):
            assert stats.prompt_token_stats_by_api_surface[
                api_surface
            ] == PromptTokenStats(
                computed=5 + index,
                local_cache_hit=2,
                external_kv_transfer=3,
                cached_tokens=5,
                total=10 + index,
            )
    else:
        assert stats.num_generation_tokens_by_api_surface is None
        assert stats.prompt_token_stats_by_api_surface is None


def test_scheduler_iteration_details_serialization():
    iteration_details = SchedulerIterationDetails(
        iteration_index=1,
        num_ctx_requests=2,
        num_ctx_tokens=3,
        num_generation_requests=4,
        num_generation_tokens=5,
        elapsed_ms=6.7,
        num_encoder_inputs=2,
        num_encoder_output_tokens=392,
    )
    outputs = EngineCoreOutputs(
        scheduler_stats=SchedulerStats(
            kv_cache_usage=0.5,
            iteration_details=iteration_details,
        )
    )

    encoded = MsgpackEncoder().encode(outputs)
    decoded = MsgpackDecoder(EngineCoreOutputs).decode(encoded)

    assert decoded.scheduler_stats is not None
    assert decoded.scheduler_stats.kv_cache_usage == 0.5
    assert decoded.scheduler_stats.iteration_details == iteration_details


def test_prefix_cache_stats_hits_by_source_accumulate():
    stats = PrefixCacheStats()
    stats.record(
        num_tokens=8,
        num_hits=5,
        preempted=False,
        hits_by_source={CacheHitSource.HOST: 3, CacheHitSource.DISK: 2},
    )
    stats.record(
        num_tokens=8,
        num_hits=6,
        preempted=False,
        hits_by_source={CacheHitSource.P2P: 5, CacheHitSource.HOST: 1},
    )
    # Zero counts never create an entry.
    stats.record(
        num_tokens=8,
        num_hits=4,
        preempted=False,
        hits_by_source={CacheHitSource.DISK: 4, CacheHitSource.P2P: 0},
    )

    assert stats.hits_by_source == {
        CacheHitSource.HOST: 4,
        CacheHitSource.DISK: 6,
        CacheHitSource.P2P: 5,
    }
    assert sum(stats.hits_by_source.values()) == stats.hits == 15


def test_prefix_cache_stats_record_hits_by_source():
    stats = PrefixCacheStats()
    stats.record(num_tokens=32, num_hits=16, preempted=False)
    stats.record(
        num_tokens=32,
        num_hits=24,
        preempted=False,
        hits_by_source={CacheHitSource.HOST: 16, CacheHitSource.DISK: 8},
    )
    # Preempted re-admissions never contribute to hits or the split.
    stats.record(
        num_tokens=32,
        num_hits=32,
        preempted=True,
        hits_by_source={CacheHitSource.P2P: 32},
    )
    # A connector miscount is reported as unspecified, not dropped or raised.
    stats.record(
        num_tokens=8,
        num_hits=8,
        preempted=False,
        hits_by_source={CacheHitSource.HOST: 4},
    )

    assert stats.requests == 3
    assert stats.hits == 48
    assert stats.preempted_hits == 32
    assert stats.hits_by_source == {
        CacheHitSource.HOST: 16,
        CacheHitSource.DISK: 8,
        CacheHitSource.EXTERNAL_UNSPECIFIED: 8,
    }
    assert sum(stats.hits_by_source.values()) == 48 - 16


def test_prefix_cache_stats_hits_by_source_serialization():
    connector_stats = PrefixCacheStats()
    connector_stats.record(
        num_tokens=16,
        num_hits=8,
        preempted=False,
        hits_by_source={CacheHitSource.P2P: 4, CacheHitSource.HOST: 4},
    )
    outputs = EngineCoreOutputs(
        scheduler_stats=SchedulerStats(
            connector_prefix_cache_stats=connector_stats,
        )
    )

    encoded = MsgpackEncoder().encode(outputs)
    decoded = MsgpackDecoder(EngineCoreOutputs).decode(encoded)

    assert decoded.scheduler_stats is not None
    assert decoded.scheduler_stats.connector_prefix_cache_stats is not None
    assert decoded.scheduler_stats.connector_prefix_cache_stats.hits_by_source == {
        CacheHitSource.P2P: 4,
        CacheHitSource.HOST: 4,
    }


def test_compute_iteration_details_includes_encoder_stats():
    scheduler_output = SchedulerOutput.make_empty()
    scheduler_output.scheduled_encoder_input_stats = ScheduledEncoderInputStats(
        num_inputs=2,
        output_tokens=392,
    )

    iteration_details = compute_iteration_details(scheduler_output)

    assert iteration_details.num_encoder_inputs == 2
    assert iteration_details.num_encoder_output_tokens == 392


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
