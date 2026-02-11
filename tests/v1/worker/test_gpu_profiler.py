# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest

from vllm.config import ProfilerConfig
from vllm.config.profiler import _is_uri_path
from vllm.profiler.wrapper import WorkerProfiler


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
