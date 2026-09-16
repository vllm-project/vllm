# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest

from vllm.v1.cache_hit_source import CacheHitSource
from vllm.v1.core.sched.output import ScheduledEncoderInputStats, SchedulerOutput
from vllm.v1.engine import EngineCoreOutput, EngineCoreOutputs, FinishReason
from vllm.v1.metrics.stats import (
    IterationStats,
    PrefillStats,
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


def test_prefill_cache_sources_serialization():
    prefill_stats = PrefillStats()
    prefill_stats.set(
        num_prompt_tokens=16,
        num_local_cached_tokens=4,
        num_external_cached_tokens=8,
        external_cached_token_sources=[("p2p", 4), ("host", 4)],
    )
    outputs = EngineCoreOutputs(
        outputs=[
            EngineCoreOutput(
                request_id="request",
                new_token_ids=[1],
                prefill_stats=prefill_stats,
            )
        ]
    )

    encoded = MsgpackEncoder().encode(outputs)
    decoded = MsgpackDecoder(EngineCoreOutputs).decode(encoded)

    assert decoded.outputs[0].prefill_stats is not None
    assert decoded.outputs[0].prefill_stats.external_cached_token_sources == [
        ("p2p", 4),
        ("host", 4),
    ]


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
    assert stats.cached_tokens_by_source == {}
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
    assert stats.cached_tokens_by_source == {"device": 300}
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
    assert stats.cached_tokens_by_source == {"external": 500}
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
        external_cached_token_sources=[("host", 100), ("disk", 100)],
    )
    stats.update_from_output(prefill_stats)

    assert stats.computed == 400
    assert stats.local_cache_hit == 400
    assert stats.external_kv_transfer == 200
    assert stats.cached_tokens == 600
    assert stats.cached_tokens_by_source == {
        "device": 400,
        "host": 100,
        "disk": 100,
    }
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
    assert stats.cached_tokens_by_source == {"device": 999}
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
    assert stats.cached_tokens_by_source == {"external": 999}
    assert stats.total == 1000


def test_prefill_stats_truncates_failed_external_source_segments():
    prefill_stats = PrefillStats()
    prefill_stats.set(
        num_prompt_tokens=1000,
        num_local_cached_tokens=100,
        num_external_cached_tokens=400,
        external_cached_token_sources=[("p2p", 200), ("host", 200)],
    )

    prefill_stats.truncate_external_cached_tokens(250)

    assert prefill_stats.num_computed_tokens == 650
    assert prefill_stats.num_cached_tokens == 350
    assert prefill_stats.num_local_cached_tokens == 100
    assert prefill_stats.num_external_cached_tokens == 250
    assert prefill_stats.external_cached_token_sources == [
        ("p2p", 200),
        ("host", 50),
    ]

    stats = PromptTokenStats()
    stats.update_from_output(prefill_stats)
    assert stats.cached_tokens_by_source == {
        "device": 100,
        "p2p": 200,
        "host": 50,
    }


def test_prefill_stats_coalesces_external_source_segments():
    prefill_stats = PrefillStats()
    prefill_stats.set(
        num_prompt_tokens=16,
        num_local_cached_tokens=4,
        num_external_cached_tokens=8,
        external_cached_token_sources=[
            ("p2p", 0),
            ("host", 3),
            ("host", 5),
            ("disk", 0),
        ],
    )

    assert prefill_stats.external_cached_token_sources == [("host", 8)]

    prefill_stats.truncate_external_cached_tokens(0)

    assert prefill_stats.external_cached_token_sources == []
    assert prefill_stats.num_external_cached_tokens == 0
    assert prefill_stats.num_cached_tokens == 4
    assert prefill_stats.num_computed_tokens == 12


@pytest.mark.parametrize("source", list(CacheHitSource))
@pytest.mark.parametrize("as_string", [False, True])
def test_prefill_stats_accepts_canonical_sources(source, as_string):
    prefill_stats = PrefillStats()
    prefill_stats.set(
        num_prompt_tokens=16,
        num_local_cached_tokens=0,
        num_external_cached_tokens=16,
        external_cached_token_sources=[(source.value if as_string else source, 16)],
    )

    assert prefill_stats.external_cached_token_sources == [(source.value, 16)]


@pytest.mark.parametrize(
    "source", ["gpu", "cpu", "dram", "file", "fs", "nvme", "HOST", "obj", "custom"]
)
@pytest.mark.parametrize("num_tokens", [0, 8])
def test_prefill_stats_rejects_noncanonical_sources(source, num_tokens):
    prefill_stats = PrefillStats()

    with pytest.raises(ValueError):
        prefill_stats.set(
            num_prompt_tokens=16,
            num_local_cached_tokens=0,
            num_external_cached_tokens=num_tokens,
            external_cached_token_sources=[(source, num_tokens)],
        )


@pytest.mark.parametrize(
    "sources",
    [
        [("", 8)],
        [("host", -1), ("disk", 9)],
        [("host", 7)],
        [("host", 9)],
    ],
)
def test_prefill_stats_rejects_malformed_external_source_segments(
    sources: list[tuple[str, int]],
):
    prefill_stats = PrefillStats()

    with pytest.raises(AssertionError):
        prefill_stats.set(
            num_prompt_tokens=16,
            num_local_cached_tokens=4,
            num_external_cached_tokens=8,
            external_cached_token_sources=sources,
        )


def test_prompt_token_stats_accumulates_sources_across_outputs():
    stats = PromptTokenStats()
    first = PrefillStats()
    first.set(
        num_prompt_tokens=16,
        num_local_cached_tokens=4,
        num_external_cached_tokens=4,
        external_cached_token_sources=[("host", 4)],
    )
    second = PrefillStats()
    second.set(
        num_prompt_tokens=16,
        num_local_cached_tokens=2,
        num_external_cached_tokens=6,
        external_cached_token_sources=[("host", 2), ("disk", 4)],
    )

    stats.update_from_output(first)
    stats.update_from_output(second)

    assert stats.cached_tokens_by_source == {
        "device": 6,
        "host": 6,
        "disk": 4,
    }
    assert sum(stats.cached_tokens_by_source.values()) == stats.cached_tokens == 16
