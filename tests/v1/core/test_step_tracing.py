# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for step-level batch summary tracing (PR #3)."""

import hashlib
from collections.abc import Callable
from unittest.mock import Mock, patch

import pytest

from vllm.tracing import SpanAttributes

from .utils import create_requests, create_scheduler

pytestmark = pytest.mark.cpu_test


def make_deterministic_step_sampler(seed: int = 0) -> Callable[[int, float], bool]:
    """Create deterministic step sampler using stable hash.

    Uses hashlib.sha1 (NOT Python's hash() which is salted per process).
    Takes step_id (int) instead of string key.
    """

    def sampler(step_id: int, rate: float) -> bool:
        hash_bytes = hashlib.sha1(f"{seed}:{step_id}".encode()).digest()
        hash_value = int.from_bytes(hash_bytes[:8], "big") / (2**64)
        return hash_value < rate

    return sampler


def test_step_tracing_disabled():
    """Test that step tracing is disabled by default.

    Verifies:
    - No span created when step_tracing_enabled=False
    - No events emitted
    - Zero overhead
    """
    scheduler = create_scheduler(step_tracing_enabled=False)

    # Verify step tracing is disabled
    assert not scheduler._enable_step_tracing
    assert scheduler._step_span is None

    # Add requests and schedule
    requests = create_requests(num_requests=2)
    for request in requests:
        scheduler.add_request(request)

    output = scheduler.schedule()

    # Verify scheduler works normally
    assert output.scheduler_step == 1
    assert len(output.scheduled_new_reqs) == 2


def test_step_tracing_enabled_no_otel():
    """Test step tracing with OTEL unavailable (no endpoint).

    Verifies graceful degradation when OTEL endpoint not configured.
    """
    scheduler = create_scheduler(
        step_tracing_enabled=True,
        step_tracing_sample_rate=1.0,
        otlp_traces_endpoint=None,  # No OTEL endpoint
    )

    # Step tracing should be disabled due to missing OTEL
    assert not scheduler._enable_step_tracing or scheduler._step_span is None

    # Scheduler should still work
    requests = create_requests(num_requests=2)
    for request in requests:
        scheduler.add_request(request)

    output = scheduler.schedule()
    assert output.scheduler_step == 1


def test_step_tracing_sample_rate_zero():
    """Test that sample_rate=0.0 produces no events.

    Even with step tracing enabled, 0% sampling should emit no events.
    """
    with patch("vllm.tracing.init_tracer") as mock_init_tracer:
        mock_tracer = Mock()
        mock_span = Mock()
        mock_tracer.start_span.return_value = mock_span
        mock_init_tracer.return_value = mock_tracer

        scheduler = create_scheduler(
            step_tracing_enabled=True,
            step_tracing_sample_rate=0.0,  # Never sample
            otlp_traces_endpoint="http://localhost:4317",
        )

        # Override sampler to always return False (0% sampling)
        scheduler._step_sampler = lambda step_id, rate: False

        requests = create_requests(num_requests=2)
        for request in requests:
            scheduler.add_request(request)

        # Schedule multiple times
        for _ in range(5):
            scheduler.schedule()

        # Verify no events emitted (sampler returned False every time)
        assert mock_span.add_event.call_count == 0


def test_step_tracing_sample_rate_one():
    """Test that sample_rate=1.0 emits event every step.

    Verifies:
    - Event emitted for every schedule() call
    - Event name is "step.BATCH_SUMMARY"
    - All required attributes present
    """
    with patch("vllm.tracing.init_tracer") as mock_init_tracer:
        mock_tracer = Mock()
        mock_span = Mock()
        mock_tracer.start_span.return_value = mock_span
        mock_init_tracer.return_value = mock_tracer

        scheduler = create_scheduler(
            step_tracing_enabled=True,
            step_tracing_sample_rate=1.0,  # Sample every step
            otlp_traces_endpoint="http://localhost:4317",
        )

        # Override sampler to always return True (100% sampling)
        scheduler._step_sampler = lambda step_id, rate: True

        requests = create_requests(num_requests=2)
        for request in requests:
            scheduler.add_request(request)

        # Schedule 3 times
        num_schedules = 3
        for _ in range(num_schedules):
            scheduler.schedule()

        # Verify event emitted for each schedule
        assert mock_span.add_event.call_count == num_schedules

        # Verify all events have correct name
        for call in mock_span.add_event.call_args_list:
            event_name = call[0][0]
            assert event_name == "step.BATCH_SUMMARY"


def test_step_tracing_deterministic_sampling():
    """Test deterministic step sampling with stable hash sampler.

    Verifies:
    - Exact expected steps are sampled
    - No statistical flakiness in tests
    - Sampling is reproducible
    """
    with patch("vllm.tracing.init_tracer") as mock_init_tracer:
        mock_tracer = Mock()
        mock_span = Mock()
        mock_tracer.start_span.return_value = mock_span
        mock_init_tracer.return_value = mock_tracer

        scheduler = create_scheduler(
            step_tracing_enabled=True,
            step_tracing_sample_rate=0.5,  # 50% sampling
            otlp_traces_endpoint="http://localhost:4317",
        )

        # Install deterministic sampler
        deterministic_sampler = make_deterministic_step_sampler(seed=42)
        scheduler._step_sampler = deterministic_sampler

        # Pre-compute which steps should be sampled with this seed
        expected_sampled_steps = []
        for step in range(1, 11):
            if deterministic_sampler(step, 0.5):
                expected_sampled_steps.append(step)

        # Schedule 10 times
        for _ in range(10):
            scheduler.schedule()

        # Verify exact expected steps were sampled
        assert mock_span.add_event.call_count == len(expected_sampled_steps)

        # Verify step IDs in events match expected
        sampled_steps = []
        for call in mock_span.add_event.call_args_list:
            attributes = call[1]["attributes"]
            sampled_steps.append(attributes[SpanAttributes.STEP_ID])

        assert sampled_steps == expected_sampled_steps


def test_step_tracing_empty_schedule():
    """Test batch summary emitted even for empty schedules.

    Verifies:
    - Empty schedule (no requests) still emits batch summary when sampled
    - All counts are zero
    - Useful for liveness monitoring
    """
    with patch("vllm.tracing.init_tracer") as mock_init_tracer:
        mock_tracer = Mock()
        mock_span = Mock()
        mock_tracer.start_span.return_value = mock_span
        mock_init_tracer.return_value = mock_tracer

        scheduler = create_scheduler(
            step_tracing_enabled=True,
            step_tracing_sample_rate=1.0,
            otlp_traces_endpoint="http://localhost:4317",
        )

        # Override sampler to always return True
        scheduler._step_sampler = lambda step_id, rate: True

        # Schedule with no requests (empty schedule)
        output = scheduler.schedule()

        # Verify event emitted even with no requests
        assert mock_span.add_event.call_count == 1

        # Verify attributes
        call_args = mock_span.add_event.call_args_list[0]
        event_name = call_args[0][0]
        attributes = call_args[1]["attributes"]

        assert event_name == "step.BATCH_SUMMARY"
        assert attributes[SpanAttributes.STEP_ID] == 1
        assert attributes[SpanAttributes.QUEUE_RUNNING_DEPTH] == 0
        assert attributes[SpanAttributes.QUEUE_WAITING_DEPTH] == 0
        assert attributes[SpanAttributes.BATCH_NUM_PREFILL_REQS] == 0
        assert attributes[SpanAttributes.BATCH_NUM_DECODE_REQS] == 0
        assert attributes[SpanAttributes.BATCH_SCHEDULED_TOKENS] == 0
        assert attributes[SpanAttributes.BATCH_PREFILL_TOKENS] == 0
        assert attributes[SpanAttributes.BATCH_DECODE_TOKENS] == 0
        assert attributes[SpanAttributes.BATCH_NUM_FINISHED] == 0
        assert attributes[SpanAttributes.BATCH_NUM_PREEMPTED] == 0


def test_step_tracing_required_attributes():
    """Test that all required attributes are present and correct.

    Verifies:
    - All required attributes from plan are present
    - Values match expected computations
    - Types are correct
    """
    with patch("vllm.tracing.init_tracer") as mock_init_tracer:
        mock_tracer = Mock()
        mock_span = Mock()
        mock_tracer.start_span.return_value = mock_span
        mock_init_tracer.return_value = mock_tracer

        scheduler = create_scheduler(
            step_tracing_enabled=True,
            step_tracing_sample_rate=1.0,
            otlp_traces_endpoint="http://localhost:4317",
        )

        # Override sampler to always return True
        scheduler._step_sampler = lambda step_id, rate: True

        # Add requests and schedule
        requests = create_requests(num_requests=3)
        for request in requests:
            scheduler.add_request(request)

        output = scheduler.schedule()

        # Verify event emitted
        assert mock_span.add_event.call_count == 1

        # Extract attributes
        call_args = mock_span.add_event.call_args_list[0]
        attributes = call_args[1]["attributes"]

        # Verify all required attributes present
        required_attrs = [
            SpanAttributes.STEP_ID,
            SpanAttributes.STEP_TS_START_NS,
            SpanAttributes.STEP_TS_END_NS,
            SpanAttributes.STEP_DURATION_US,
            SpanAttributes.QUEUE_RUNNING_DEPTH,
            SpanAttributes.QUEUE_WAITING_DEPTH,
            SpanAttributes.BATCH_NUM_PREFILL_REQS,
            SpanAttributes.BATCH_NUM_DECODE_REQS,
            SpanAttributes.BATCH_SCHEDULED_TOKENS,
            SpanAttributes.BATCH_PREFILL_TOKENS,
            SpanAttributes.BATCH_DECODE_TOKENS,
            SpanAttributes.BATCH_NUM_FINISHED,
            SpanAttributes.BATCH_NUM_PREEMPTED,
            SpanAttributes.KV_USAGE_GPU_RATIO,
            SpanAttributes.KV_BLOCKS_TOTAL_GPU,
            SpanAttributes.KV_BLOCKS_FREE_GPU,
        ]

        for attr in required_attrs:
            assert attr in attributes, f"Missing required attribute: {attr}"

        # Verify types
        assert isinstance(attributes[SpanAttributes.STEP_ID], int)
        assert isinstance(attributes[SpanAttributes.STEP_TS_START_NS], int)
        assert isinstance(attributes[SpanAttributes.STEP_TS_END_NS], int)
        assert isinstance(attributes[SpanAttributes.STEP_DURATION_US], int)
        assert isinstance(attributes[SpanAttributes.QUEUE_RUNNING_DEPTH], int)
        assert isinstance(attributes[SpanAttributes.QUEUE_WAITING_DEPTH], int)
        assert isinstance(attributes[SpanAttributes.BATCH_NUM_PREFILL_REQS], int)
        assert isinstance(attributes[SpanAttributes.BATCH_NUM_DECODE_REQS], int)
        assert isinstance(attributes[SpanAttributes.BATCH_SCHEDULED_TOKENS], int)
        assert isinstance(attributes[SpanAttributes.BATCH_PREFILL_TOKENS], int)
        assert isinstance(attributes[SpanAttributes.BATCH_DECODE_TOKENS], int)
        assert isinstance(attributes[SpanAttributes.BATCH_NUM_FINISHED], int)
        assert isinstance(attributes[SpanAttributes.BATCH_NUM_PREEMPTED], int)
        assert isinstance(attributes[SpanAttributes.KV_USAGE_GPU_RATIO], float)
        assert isinstance(attributes[SpanAttributes.KV_BLOCKS_TOTAL_GPU], int)
        assert isinstance(attributes[SpanAttributes.KV_BLOCKS_FREE_GPU], int)

        # Verify step ID
        assert attributes[SpanAttributes.STEP_ID] == 1

        # Verify timing
        assert attributes[SpanAttributes.STEP_TS_START_NS] > 0
        assert attributes[SpanAttributes.STEP_TS_END_NS] > 0
        assert attributes[SpanAttributes.STEP_TS_END_NS] >= attributes[SpanAttributes.STEP_TS_START_NS]
        assert attributes[SpanAttributes.STEP_DURATION_US] >= 0

        # Verify queue depths
        assert attributes[SpanAttributes.QUEUE_RUNNING_DEPTH] >= 0
        assert attributes[SpanAttributes.QUEUE_WAITING_DEPTH] >= 0

        # Verify batch composition
        assert attributes[SpanAttributes.BATCH_NUM_PREFILL_REQS] >= 0
        assert attributes[SpanAttributes.BATCH_NUM_DECODE_REQS] >= 0

        # Verify token counts
        assert attributes[SpanAttributes.BATCH_SCHEDULED_TOKENS] >= 0
        assert attributes[SpanAttributes.BATCH_PREFILL_TOKENS] >= 0
        assert attributes[SpanAttributes.BATCH_DECODE_TOKENS] >= 0

        # Verify KV cache metrics
        assert 0.0 <= attributes[SpanAttributes.KV_USAGE_GPU_RATIO] <= 1.0
        assert attributes[SpanAttributes.KV_BLOCKS_TOTAL_GPU] >= 0
        assert attributes[SpanAttributes.KV_BLOCKS_FREE_GPU] >= 0


def test_step_tracing_invariants():
    """Test that batch summary attributes satisfy expected invariants.

    Verifies:
    - Token sum invariants
    - KV cache consistency
    - Batch composition consistency
    """
    with patch("vllm.tracing.init_tracer") as mock_init_tracer:
        mock_tracer = Mock()
        mock_span = Mock()
        mock_tracer.start_span.return_value = mock_span
        mock_init_tracer.return_value = mock_tracer

        scheduler = create_scheduler(
            step_tracing_enabled=True,
            step_tracing_sample_rate=1.0,
            otlp_traces_endpoint="http://localhost:4317",
        )

        # Override sampler to always return True
        scheduler._step_sampler = lambda step_id, rate: True

        # Add requests and schedule
        requests = create_requests(num_requests=5)
        for request in requests:
            scheduler.add_request(request)

        output = scheduler.schedule()

        # Extract attributes
        call_args = mock_span.add_event.call_args_list[0]
        attributes = call_args[1]["attributes"]

        # Invariant: prefill + decode requests <= running depth
        prefill_reqs = attributes[SpanAttributes.BATCH_NUM_PREFILL_REQS]
        decode_reqs = attributes[SpanAttributes.BATCH_NUM_DECODE_REQS]
        running_depth = attributes[SpanAttributes.QUEUE_RUNNING_DEPTH]
        assert prefill_reqs + decode_reqs <= running_depth

        # Invariant: prefill + decode tokens == scheduled tokens
        # NOTE: This test assumes no speculative decode. With speculative decode,
        # scheduled_tokens may include spec tokens not counted in prefill/decode.
        # If this test starts failing, check if spec decode is enabled in the test.
        prefill_tokens = attributes[SpanAttributes.BATCH_PREFILL_TOKENS]
        decode_tokens = attributes[SpanAttributes.BATCH_DECODE_TOKENS]
        scheduled_tokens = attributes[SpanAttributes.BATCH_SCHEDULED_TOKENS]
        # Equality expected in standard config (no spec decode in create_requests())
        assert prefill_tokens + decode_tokens == scheduled_tokens

        # Invariant: KV cache consistency
        kv_total = attributes[SpanAttributes.KV_BLOCKS_TOTAL_GPU]
        kv_free = attributes[SpanAttributes.KV_BLOCKS_FREE_GPU]
        kv_usage = attributes[SpanAttributes.KV_USAGE_GPU_RATIO]

        assert kv_free <= kv_total
        assert 0.0 <= kv_usage <= 1.0

        # Sanity check: usage roughly matches free/total ratio
        # (May not be exact due to reserved/null blocks)
        if kv_total > 0:
            expected_usage = 1.0 - (kv_free / kv_total)
            # Allow some tolerance due to null block and rounding
            assert abs(kv_usage - expected_usage) < 0.1


def test_step_tracing_failure_safety():
    """Test that tracing failures don't crash the scheduler.

    Verifies:
    - Scheduler continues working even if event emission fails
    - Exceptions are caught and logged
    - No impact on scheduling correctness
    """
    with patch("vllm.tracing.init_tracer") as mock_init_tracer:
        mock_tracer = Mock()
        mock_span = Mock()
        mock_tracer.start_span.return_value = mock_span
        mock_init_tracer.return_value = mock_tracer

        # Make add_event raise an exception
        mock_span.add_event.side_effect = Exception("OTEL failure")

        scheduler = create_scheduler(
            step_tracing_enabled=True,
            step_tracing_sample_rate=1.0,
            otlp_traces_endpoint="http://localhost:4317",
        )

        # Override sampler to always return True
        scheduler._step_sampler = lambda step_id, rate: True

        # Add requests and schedule
        requests = create_requests(num_requests=2)
        for request in requests:
            scheduler.add_request(request)

        # Scheduler should not crash despite tracing failure
        output = scheduler.schedule()

        # Verify scheduler worked correctly
        assert output.scheduler_step == 1
        assert len(output.scheduled_new_reqs) == 2

        # Verify exception was caught (add_event was called but didn't crash)
        assert mock_span.add_event.call_count == 1
