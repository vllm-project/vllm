# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from unittest.mock import MagicMock

import pytest

from vllm.config import CUDAGraphMode, ProfilerConfig
from vllm.config.profiler import _is_uri_path
from vllm.profiler.wrapper import WorkerProfiler
from vllm.v1.core.sched.output import CachedRequestData
from vllm.v1.worker.dp_utils import DPProfilerSync
from vllm.v1.worker.gpu_model_runner import GPUModelRunner
from vllm.v1.worker.gpu_worker import Worker


class ConcreteWorkerProfiler(WorkerProfiler):
    """
    A basic implementation of a worker profiler for testing purposes.
    """

    def __init__(self, profiler_config: ProfilerConfig):
        self.start_call_count = 0
        self.stop_call_count = 0
        self.should_fail_start = False
        super().__init__(profiler_config)

    def _start(self) -> None:
        if self.should_fail_start:
            raise RuntimeError("Simulated start failure")
        self.start_call_count += 1

    def _stop(self) -> None:
        self.stop_call_count += 1


@pytest.fixture
def default_profiler_config():
    return ProfilerConfig(
        profiler="torch",
        torch_profiler_dir="/tmp/mock",
        delay_iterations=0,
        max_iterations=0,
    )


def test_immediate_start_stop(default_profiler_config):
    """Test standard start without delay."""
    profiler = ConcreteWorkerProfiler(default_profiler_config)
    profiler.start()
    assert profiler._running is True
    assert profiler._active is True
    assert profiler.start_call_count == 1

    profiler.stop()
    assert profiler._running is False
    assert profiler._active is False
    assert profiler.stop_call_count == 1


def test_delayed_start(default_profiler_config):
    """Test that profiler waits for N steps before actually starting."""
    default_profiler_config.delay_iterations = 2
    profiler = ConcreteWorkerProfiler(default_profiler_config)

    # User requests start
    profiler.start()

    # Should be active (request accepted) but not running (waiting for delay)
    assert profiler._active is True
    assert profiler._running is False
    assert profiler.start_call_count == 0

    # Step 1
    profiler.step()
    assert profiler._running is False

    # Step 2 (Threshold reached)
    profiler.step()
    assert profiler._running is True
    assert profiler.start_call_count == 1


def test_max_iterations(default_profiler_config):
    """Test that profiler stops automatically after max iterations."""
    default_profiler_config.max_iterations = 2
    profiler = ConcreteWorkerProfiler(default_profiler_config)

    profiler.start()
    assert profiler._running is True

    # Iteration 1
    profiler.step()  # profiling_count becomes 1
    assert profiler._running is True

    # Iteration 2
    profiler.step()  # profiling_count becomes 2
    assert profiler._running is True

    # Iteration 3 (Exceeds max)
    profiler.step()  # profiling_count becomes 3

    # Should have stopped now
    assert profiler._running is False
    assert profiler.stop_call_count == 1


def test_delayed_start_and_max_iters(default_profiler_config):
    """Test combined delayed start and max iterations."""
    default_profiler_config.delay_iterations = 2
    default_profiler_config.max_iterations = 2
    profiler = ConcreteWorkerProfiler(default_profiler_config)
    profiler.start()

    # Step 1
    profiler.step()
    assert profiler._running is False
    assert profiler._active is True

    # Step 2 (Starts now)
    profiler.step()
    assert profiler._profiling_for_iters == 1
    assert profiler._running is True
    assert profiler._active is True

    # Next iteration
    profiler.step()
    assert profiler._profiling_for_iters == 2
    assert profiler._running is True

    # Iteration 2 (exceeds max)
    profiler.step()

    # Should have stopped now
    assert profiler._running is False
    assert profiler.stop_call_count == 1


def test_idempotency(default_profiler_config):
    """Test that calling start/stop multiple times doesn't break logic."""
    profiler = ConcreteWorkerProfiler(default_profiler_config)

    # Double Start
    profiler.start()
    profiler.start()
    assert profiler.start_call_count == 1  # Should only start once

    # Double Stop
    profiler.stop()
    profiler.stop()
    assert profiler.stop_call_count == 1  # Should only stop once


def test_step_inactive(default_profiler_config):
    """Test that stepping while inactive does nothing."""
    default_profiler_config.delay_iterations = 2
    profiler = ConcreteWorkerProfiler(default_profiler_config)

    # Not started yet
    profiler.step()
    profiler.step()

    # Even though we stepped 2 times, start shouldn't happen because active=False
    assert profiler.start_call_count == 0


def test_start_failure(default_profiler_config):
    """Test behavior when the underlying _start method raises exception."""
    profiler = ConcreteWorkerProfiler(default_profiler_config)
    profiler.should_fail_start = True

    profiler.start()

    # Exception caught in _call_start
    assert profiler._running is False  # Should not mark as running
    assert profiler._active is True  # Request is still considered active
    assert profiler.start_call_count == 0  # Logic failed inside start


def test_shutdown(default_profiler_config):
    """Test that shutdown calls stop only if running."""
    profiler = ConcreteWorkerProfiler(default_profiler_config)

    # Case 1: Not running
    profiler.shutdown()
    assert profiler.stop_call_count == 0

    # Case 2: Running
    profiler.start()
    profiler.shutdown()
    assert profiler.stop_call_count == 1


def test_mixed_delay_and_stop(default_profiler_config):
    """Test manual stop during the delay period."""
    default_profiler_config.delay_iterations = 5
    profiler = ConcreteWorkerProfiler(default_profiler_config)

    profiler.start()
    profiler.step()
    profiler.step()

    # User cancels before delay finishes
    profiler.stop()
    assert profiler._active is False

    # Further steps should not trigger start
    profiler.step()
    profiler.step()
    profiler.step()

    assert profiler.start_call_count == 0


class TestIsUriPath:
    """Tests for the _is_uri_path helper function."""

    @pytest.mark.parametrize(
        "path,expected",
        [
            # Valid URI schemes - should return True
            ("gs://bucket/path", True),
            ("s3://bucket/path", True),
            ("hdfs://cluster/path", True),
            ("abfs://container/path", True),
            ("http://example.com/path", True),
            ("https://example.com/path", True),
            # Local paths - should return False
            ("/tmp/local/path", False),
            ("./relative/path", False),
            ("relative/path", False),
            ("/absolute/path", False),
            # Windows drive letters - should return False (single char scheme)
            ("C://windows/path", False),
            ("D://drive/path", False),
            # Edge cases
            ("", False),
            ("no-scheme", False),
            ("scheme-no-slashes:", False),
            ("://no-scheme", False),
        ],
    )
    def test_is_uri_path(self, path, expected):
        """Test that _is_uri_path correctly identifies URI vs local paths."""
        assert _is_uri_path(path) == expected


class TestAnnotateProfile:
    """Tests for Worker.annotate_profile() annotation string formatting."""

    def _annotate(self, detailed: bool) -> str:
        worker = MagicMock()
        worker.vllm_config.profiler_config.detailed_trace_annotation = detailed
        worker.profiler = MagicMock()

        ctx_req = MagicMock(req_id="ctx1", num_computed_tokens=0)
        cached = CachedRequestData(
            req_ids=["gen1"],
            resumed_req_ids=set(),
            new_token_ids=[],
            all_token_ids={},
            new_block_ids=[],
            num_computed_tokens=[10],
            num_output_tokens=[1],
        )
        sched = MagicMock(
            scheduled_new_reqs=[ctx_req],
            scheduled_cached_reqs=cached,
            num_scheduled_tokens={"ctx1": 4, "gen1": 1},
        )

        Worker.annotate_profile(worker, sched)
        return worker.profiler.annotate_context_manager.call_args[0][0]

    def test_simple_format_mixed(self):
        assert self._annotate(detailed=False) == (
            "execute_context_1(4)_generation_1(1)"
        )

    def test_detailed_format_mixed(self):
        # ctx1: sq=4, sk=4, sqsq=16, sqsk=16 | gen1: sq=1, sk=11, sqsq=1, sqsk=11 | bs=5
        assert self._annotate(detailed=True) == (
            "execute_5_context_1(sq4sk4sqsq16sqsk16)_generation_1(sq1sk11sqsq1sqsk11)"
        )


class TestDPProfilerSync:
    """Unit tests for the OR-reduced, deferred profiler start used by
    VLLM_ENABLE_MULTINODE_PROFILING (see DPProfilerSync). observe(consensus)
    stands in for the value read back from the per-step DP all-reduce."""

    def test_local_request_reaches_consensus(self):
        sync = DPProfilerSync()
        assert sync.consume_start() is False  # nothing requested yet

        sync.request_start()
        assert sync._pending is True
        # Reduce carries this rank's request; every rank observes the OR.
        sync.observe(consensus=True)

        assert sync.consume_start() is True
        # Consumed exactly once, and the request is cleared afterwards.
        assert sync.consume_start() is False
        assert sync._pending is False

    def test_remote_request_starts_this_rank(self):
        """A rank that never received start_profile still starts once the OR
        across DP ranks is True (only one rank needs the HTTP call)."""
        sync = DPProfilerSync()
        assert sync._pending is False

        sync.observe(consensus=True)

        assert sync.consume_start() is True

    def test_no_request_no_start(self):
        sync = DPProfilerSync()
        sync.observe(consensus=False)
        assert sync.consume_start() is False

    def test_latch_survives_later_false_observation(self):
        """After consensus, a second reduce in the same step (PP+SP) reports
        False once _pending is cleared; that must not drop the latch before the
        worker consumes it."""
        sync = DPProfilerSync()
        sync.request_start()
        sync.observe(consensus=True)
        sync.observe(consensus=False)  # second reduce, pending already cleared
        assert sync.consume_start() is True

    def test_cancel_drops_pending_request(self):
        """stop_profile before consensus cancels a pending start."""
        sync = DPProfilerSync()
        sync.request_start()
        sync.cancel()
        assert sync._pending is False
        # A stale reduce value must not resurrect a cancelled request as a start
        # this rank acts on; but if the OR is genuinely True from another rank it
        # still starts. Here nothing else requested, so no start.
        sync.observe(consensus=False)
        assert sync.consume_start() is False


class TestWorkerSyncedProfileStart:
    """Worker._maybe_start_synced_profile drives the deferred start from the
    per-step consensus latch."""

    def _worker(self, dp_profiler_sync, profiler):
        worker = MagicMock()
        worker.model_runner.dp_profiler_sync = dp_profiler_sync
        worker.profiler = profiler
        return worker

    def test_starts_on_consensus(self):
        sync = DPProfilerSync()
        sync.request_start()
        sync.observe(consensus=True)
        profiler = MagicMock()
        worker = self._worker(sync, profiler)

        Worker._maybe_start_synced_profile(worker)
        profiler.start.assert_called_once()

        # Latch consumed: a later step does not start again.
        Worker._maybe_start_synced_profile(worker)
        profiler.start.assert_called_once()

    def test_no_start_without_consensus(self):
        sync = DPProfilerSync()
        sync.request_start()  # requested but reduce has not agreed yet
        profiler = MagicMock()
        worker = self._worker(sync, profiler)

        Worker._maybe_start_synced_profile(worker)
        profiler.start.assert_not_called()

    def test_noop_when_sync_disabled(self):
        profiler = MagicMock()
        worker = self._worker(None, profiler)

        Worker._maybe_start_synced_profile(worker)
        profiler.start.assert_not_called()

    def test_creates_profiler_on_remote_rank(self):
        """A rank that reached consensus via the OR-reduce but never received
        start_profile has no profiler wrapper yet; it must be created and
        started here, not silently skipped. Regression: an early
        ``self.profiler is None`` return dropped every rank that did not get the
        HTTP call, so only the called node captured (single-node traces)."""
        sync = DPProfilerSync()
        sync.observe(consensus=True)  # remote request; this rank never asked
        worker = self._worker(sync, profiler=None)
        worker.profiler_config = MagicMock(profiler="torch")

        created = MagicMock()

        def _create(*args, **kwargs):
            worker.profiler = created

        worker._create_profiler.side_effect = _create

        Worker._maybe_start_synced_profile(worker)

        worker._create_profiler.assert_called_once()
        created.start.assert_called_once()

    def test_skips_when_profiling_not_configured(self):
        """Consensus on a rank where profiling is not configured must no-op
        (and not build a profiler), rather than crash."""
        sync = DPProfilerSync()
        sync.observe(consensus=True)
        worker = self._worker(sync, profiler=None)
        worker.profiler_config = None

        Worker._maybe_start_synced_profile(worker)

        worker._create_profiler.assert_not_called()


class TestWorkerExecuteDummyBatch:
    """execute_dummy_batch must advance the profiler on idle steps so a DP rank
    that started capture via the OR-reduced synced start but then only services
    dummy batches still counts these iterations, reaches max_iterations, stops,
    and dumps its trace. Regression: dummy batches started the profiler (via
    _maybe_start_synced_profile) but never stepped it (step runs only in
    execute_model's annotate_profile), so idle ranks never exported and
    multi-node capture silently degraded to the busy ranks only."""

    def _worker(self, profiler):
        worker = MagicMock()
        worker.profiler = profiler
        worker.model_runner.uniform_decode_query_len = 1
        return worker

    def test_steps_profiler_on_dummy_batch(self):
        profiler = MagicMock()
        worker = self._worker(profiler)

        Worker.execute_dummy_batch(worker)

        profiler.step.assert_called_once()
        worker.model_runner._dummy_run.assert_called_once()
        worker._maybe_start_synced_profile.assert_called_once()

    def test_no_step_when_profiler_absent(self):
        worker = self._worker(profiler=None)

        # Must not raise when there is no profiler yet, and must still run the
        # dummy batch and check for a synced start.
        Worker.execute_dummy_batch(worker)

        worker.model_runner._dummy_run.assert_called_once()
        worker._maybe_start_synced_profile.assert_called_once()


def test_profiler_entered_during_capture():
    """Profiler is used as a context manager in _warmup_and_capture,
    confirming it is active during the actual graph capture run."""
    runner = MagicMock()
    runner.compilation_config.cudagraph_num_of_warmups = 0
    mock_profiler = MagicMock()

    GPUModelRunner._warmup_and_capture(
        runner,
        desc=MagicMock(num_tokens=4, uniform=True),
        cudagraph_runtime_mode=CUDAGraphMode.FULL,
        profiler=mock_profiler,
    )

    mock_profiler.__enter__.assert_called_once()
    mock_profiler.__exit__.assert_called_once()
