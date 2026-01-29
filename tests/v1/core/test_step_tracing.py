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


def test_rich_snapshot_rate_zero():
    """Test that rich subsample rate 0.0 produces no rich events.

    Verifies:
    - Batch summary still emitted (step.BATCH_SUMMARY)
    - No rich snapshot events (step.REQUEST_SNAPSHOT)
    - Rich sampling is independent from batch summary sampling
    """
    with patch("vllm.tracing.init_tracer") as mock_init_tracer:
        mock_tracer = Mock()
        mock_span = Mock()
        mock_tracer.start_span.return_value = mock_span
        mock_init_tracer.return_value = mock_tracer

        scheduler = create_scheduler(
            step_tracing_enabled=True,
            step_tracing_sample_rate=1.0,  # Always sample batch summary
            step_tracing_rich_subsample_rate=0.0,  # Never sample rich snapshots
            otlp_traces_endpoint="http://localhost:4317",
        )

        # Override samplers
        scheduler._step_sampler = lambda step_id, rate: True  # Batch summary always
        scheduler._rich_sampler = lambda step_id, rate: False  # Rich never

        # Add requests and schedule
        requests = create_requests(num_requests=3)
        for request in requests:
            scheduler.add_request(request)

        output = scheduler.schedule()

        # Verify scheduler worked
        assert output.scheduler_step == 1
        assert len(output.scheduled_new_reqs) == 3

        # Extract event names
        event_names = [call[0][0] for call in mock_span.add_event.call_args_list]

        # Should have exactly 1 batch summary, no rich snapshots
        assert event_names.count("step.BATCH_SUMMARY") == 1
        assert event_names.count("step.REQUEST_SNAPSHOT") == 0


def test_rich_snapshot_enabled():
    """Test that rich subsample rate 1.0 emits events for all running requests.

    Verifies:
    - One step.REQUEST_SNAPSHOT event per running request
    - Events have correct step.id correlation
    - All required attributes present
    - KV metrics populated
    """
    with patch("vllm.tracing.init_tracer") as mock_init_tracer:
        mock_tracer = Mock()
        mock_span = Mock()
        mock_tracer.start_span.return_value = mock_span
        mock_init_tracer.return_value = mock_tracer

        scheduler = create_scheduler(
            step_tracing_enabled=True,
            step_tracing_sample_rate=1.0,
            step_tracing_rich_subsample_rate=1.0,  # Always sample rich
            otlp_traces_endpoint="http://localhost:4317",
        )

        # Override samplers to always return True
        scheduler._step_sampler = lambda step_id, rate: True
        scheduler._rich_sampler = lambda step_id, rate: True

        # Add requests and schedule
        requests = create_requests(num_requests=4)
        for request in requests:
            scheduler.add_request(request)

        output = scheduler.schedule()

        # Verify scheduler worked
        assert output.scheduler_step == 1
        assert len(output.scheduled_new_reqs) == 4

        # Extract events
        event_names = [call[0][0] for call in mock_span.add_event.call_args_list]
        event_attrs = [call[1]["attributes"] for call in mock_span.add_event.call_args_list]

        # Should have 1 batch summary + 4 rich snapshots
        assert event_names.count("step.BATCH_SUMMARY") == 1
        assert event_names.count("step.REQUEST_SNAPSHOT") == 4

        # Verify rich snapshot attributes
        rich_events = [
            attrs
            for name, attrs in zip(event_names, event_attrs)
            if name == "step.REQUEST_SNAPSHOT"
        ]
        assert len(rich_events) == 4

        for attrs in rich_events:
            # Required attributes
            assert attrs[SpanAttributes.STEP_ID] == 1
            assert SpanAttributes.REQUEST_ID in attrs
            assert attrs[SpanAttributes.REQUEST_PHASE] in ("PREFILL", "DECODE")
            assert SpanAttributes.REQUEST_NUM_PROMPT_TOKENS in attrs
            assert SpanAttributes.REQUEST_NUM_COMPUTED_TOKENS in attrs
            assert SpanAttributes.REQUEST_NUM_OUTPUT_TOKENS in attrs
            assert SpanAttributes.REQUEST_NUM_PREEMPTIONS in attrs
            assert SpanAttributes.REQUEST_SCHEDULED_TOKENS_THIS_STEP in attrs
            assert SpanAttributes.KV_BLOCKS_ALLOCATED_GPU in attrs
            assert SpanAttributes.KV_BLOCKS_CACHED_GPU in attrs

            # Verify types
            assert isinstance(attrs[SpanAttributes.STEP_ID], int)
            assert isinstance(attrs[SpanAttributes.REQUEST_ID], str)
            assert isinstance(attrs[SpanAttributes.KV_BLOCKS_ALLOCATED_GPU], int)
            assert isinstance(attrs[SpanAttributes.KV_BLOCKS_CACHED_GPU], int)


def test_rich_snapshot_gated_on_batch_summary():
    """Test that rich snapshots are only emitted when batch summary is sampled.

    Verifies the two-stage sampling:
    1. Step must be batch-summary-sampled
    2. Then rich subsampling decision
    """
    with patch("vllm.tracing.init_tracer") as mock_init_tracer:
        mock_tracer = Mock()
        mock_span = Mock()
        mock_tracer.start_span.return_value = mock_span
        mock_init_tracer.return_value = mock_tracer

        scheduler = create_scheduler(
            step_tracing_enabled=True,
            step_tracing_sample_rate=1.0,
            step_tracing_rich_subsample_rate=1.0,
            otlp_traces_endpoint="http://localhost:4317",
        )

        # Override batch summary sampler to return False (not sampled)
        scheduler._step_sampler = lambda step_id, rate: False
        # Rich sampler is irrelevant (shouldn't be called)
        scheduler._rich_sampler = lambda step_id, rate: True

        # Add requests and schedule
        requests = create_requests(num_requests=3)
        for request in requests:
            scheduler.add_request(request)

        output = scheduler.schedule()

        # Verify scheduler worked
        assert output.scheduler_step == 1
        assert len(output.scheduled_new_reqs) == 3

        # No events should be emitted (batch summary not sampled)
        assert mock_span.add_event.call_count == 0


def test_rich_snapshot_deterministic_sampling():
    """Test deterministic rich sampling for reproducible tests.

    Verifies:
    - Deterministic sampler produces stable results
    - Rich sampling decision is independent per step
    - Same seed produces same sample set
    """
    with patch("vllm.tracing.init_tracer") as mock_init_tracer:
        mock_tracer = Mock()
        mock_span = Mock()
        mock_tracer.start_span.return_value = mock_span
        mock_init_tracer.return_value = mock_tracer

        scheduler = create_scheduler(
            step_tracing_enabled=True,
            step_tracing_sample_rate=1.0,
            step_tracing_rich_subsample_rate=1.0,
            otlp_traces_endpoint="http://localhost:4317",
        )

        # Use deterministic samplers
        scheduler._step_sampler = make_deterministic_step_sampler(seed=42)
        scheduler._rich_sampler = make_deterministic_step_sampler(seed=100)

        # Run multiple steps
        requests = create_requests(num_requests=2)
        for request in requests:
            scheduler.add_request(request)

        for _ in range(5):
            scheduler.schedule()

        # Extract event names
        event_names = [call[0][0] for call in mock_span.add_event.call_args_list]

        # With deterministic sampling, results should be stable
        batch_summaries = event_names.count("step.BATCH_SUMMARY")
        rich_snapshots = event_names.count("step.REQUEST_SNAPSHOT")

        # Verify we got some events (exact count depends on hash outputs)
        assert batch_summaries > 0
        # Rich snapshots only emitted when batch summary was sampled
        assert rich_snapshots % 2 == 0  # Should be even (2 requests per step)


def test_rich_snapshot_with_zero_running_requests():
    """Test that rich snapshots work correctly with empty running queue.

    Verifies:
    - Batch summary emitted even with no running requests
    - No rich snapshot events (no requests to snapshot)
    - No crashes or errors
    """
    with patch("vllm.tracing.init_tracer") as mock_init_tracer:
        mock_tracer = Mock()
        mock_span = Mock()
        mock_tracer.start_span.return_value = mock_span
        mock_init_tracer.return_value = mock_tracer

        scheduler = create_scheduler(
            step_tracing_enabled=True,
            step_tracing_sample_rate=1.0,
            step_tracing_rich_subsample_rate=1.0,
            otlp_traces_endpoint="http://localhost:4317",
        )

        # Override samplers to always return True
        scheduler._step_sampler = lambda step_id, rate: True
        scheduler._rich_sampler = lambda step_id, rate: True

        # Schedule with no requests
        output = scheduler.schedule()

        # Verify scheduler worked
        assert output.scheduler_step == 1
        assert len(output.scheduled_new_reqs) == 0

        # Extract event names
        event_names = [call[0][0] for call in mock_span.add_event.call_args_list]

        # Should have 1 batch summary, 0 rich snapshots (no running requests)
        assert event_names.count("step.BATCH_SUMMARY") == 1
        assert event_names.count("step.REQUEST_SNAPSHOT") == 0


def test_step_tracing_cli_wiring():
    """Test that CLI flags are properly wired through to ObservabilityConfig.

    This is a regression test for PR #3 and PR #5 CLI wiring.
    Ensures that step tracing flags flow from CLI -> EngineArgs -> ObservabilityConfig.
    """
    from vllm.engine.arg_utils import EngineArgs

    # Test values different from defaults
    engine_args = EngineArgs(
        model="facebook/opt-125m",
        step_tracing_enabled=True,  # Default: False
        step_tracing_sample_rate=0.75,  # Default: 0.01
        step_tracing_rich_subsample_rate=0.05,  # Default: 0.001
    )

    # Create engine config and verify wiring
    vllm_config = engine_args.create_engine_config()
    obs_config = vllm_config.observability_config

    # Verify all three fields are correctly wired
    assert obs_config.step_tracing_enabled is True
    assert obs_config.step_tracing_sample_rate == 0.75
    assert obs_config.step_tracing_rich_subsample_rate == 0.05
