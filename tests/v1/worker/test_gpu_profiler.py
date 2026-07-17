# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest

from vllm.config import ProfilerConfig
from vllm.config.profiler import _is_uri_path
from vllm.profiler.wrapper import TorchProfilerWrapper, WorkerProfiler


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


class FakeTorchProfile:
    def __init__(self, kwargs: dict[str, object] | None = None) -> None:
        self.kwargs = kwargs or {}
        self.start_call_count = 0
        self.stop_call_count = 0
        self.step_call_count = 0

    def start(self) -> None:
        self.start_call_count += 1

    def stop(self) -> None:
        self.stop_call_count += 1

    def step(self) -> None:
        self.step_call_count += 1

    def key_averages(self) -> "FakeTorchProfile":
        return self

    def table(self, sort_by: str, row_limit: int | None = None) -> str:
        return f"{sort_by}:{row_limit}"


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


def _multi_window_config(windows: list[tuple[int, int]]) -> ProfilerConfig:
    """Build a ProfilerConfig with multiple (delay, max) windows."""
    delays = ",".join(str(d) for d, _ in windows)
    maxes = ",".join(str(m) for _, m in windows)
    return ProfilerConfig(
        profiler="cuda",
        delay_iterations=delays,
        max_iterations=maxes,
    )


@pytest.mark.parametrize(
    ("windows", "idle_steps_after_first_window", "second_window_extra_steps"),
    [
        ([(0, 2), (5, 3)], 2, 2),
        ([(0, 2), (3, 2)], 0, 1),
    ],
)
def test_multi_window_two_windows(
    windows: list[tuple[int, int]],
    idle_steps_after_first_window: int,
    second_window_extra_steps: int,
):
    """Two profiling windows fire in order, with or without an idle gap."""
    config = _multi_window_config(windows)
    profiler = ConcreteWorkerProfiler(config)

    # First window: delay=0, max=2. Starts immediately on start().
    profiler.start()
    assert profiler._running is True
    assert profiler.start_call_count == 1

    profiler.step()  # iter 1, active iter 1
    profiler.step()  # iter 2, active iter 2
    assert profiler._running is False
    assert profiler.stop_call_count == 1
    assert profiler._active is True

    for _ in range(idle_steps_after_first_window):
        profiler.step()
        assert profiler._running is False

    profiler.step()  # Window 1 reaches its delay and starts.
    assert profiler._running is True
    assert profiler.start_call_count == 2

    for _ in range(second_window_extra_steps):
        profiler.step()
    assert profiler._running is False
    assert profiler.stop_call_count == 2


def test_multi_window_three_windows_with_delayed_first_window():
    """Three windows can fire in order when the first one is delayed."""
    config = _multi_window_config([(1, 1), (3, 2), (6, 1)])
    profiler = ConcreteWorkerProfiler(config)

    profiler.start()
    assert profiler._running is False
    assert profiler.start_call_count == 0

    profiler.step()  # iter 1: window 0 starts and completes
    assert profiler._running is False
    assert profiler.start_call_count == 1
    assert profiler.stop_call_count == 1

    profiler.step()  # iter 2: gap before window 1
    assert profiler._running is False

    profiler.step()  # iter 3: window 1 starts
    assert profiler._running is True
    assert profiler.start_call_count == 2

    profiler.step()  # iter 4: window 1 completes
    assert profiler._running is False
    assert profiler.stop_call_count == 2

    profiler.step()  # iter 5: gap before window 2
    assert profiler._running is False

    profiler.step()  # iter 6: window 2 starts and completes
    assert profiler._running is False
    assert profiler.start_call_count == 3
    assert profiler.stop_call_count == 3


def test_multi_window_stop_mid_sequence():
    """Manual stop during window 0 cancels remaining windows."""
    config = _multi_window_config([(0, 5), (10, 3)])
    profiler = ConcreteWorkerProfiler(config)

    profiler.start()
    profiler.step()
    profiler.step()

    profiler.stop()
    assert profiler._active is False
    assert profiler.stop_call_count == 1

    # Subsequent steps must not start window 1.
    for _ in range(15):
        profiler.step()
    assert profiler.start_call_count == 1


def test_multi_window_stop_during_later_window_cancels_remaining_windows():
    """Manual stop during a later window cancels remaining windows."""
    config = _multi_window_config([(0, 1), (3, 5), (10, 1)])
    profiler = ConcreteWorkerProfiler(config)

    profiler.start()
    profiler.step()  # window 0 completes
    profiler.step()  # gap before window 1
    profiler.step()  # window 1 starts
    profiler.step()  # window 1 is still running
    assert profiler._running is True
    assert profiler._current_window == 1

    profiler.stop()
    assert profiler._active is False
    assert profiler._running is False
    assert profiler._current_window == 1
    assert profiler.stop_call_count == 2

    for _ in range(10):
        profiler.step()
    assert profiler.start_call_count == 2
    assert profiler.stop_call_count == 2


def test_multi_window_parses_comma_separated_strings():
    """Comma-separated strings should normalize to list[int] windows."""
    config = ProfilerConfig(
        profiler="cuda",
        delay_iterations="30,100",
        max_iterations="10,20",
    )
    assert config.get_iteration_windows() == [(30, 10), (100, 20)]


def test_multi_window_allows_torch_profiler(tmp_path):
    """Multi-window profiling is supported by the torch profiler."""
    config = ProfilerConfig(
        profiler="torch",
        torch_profiler_dir=str(tmp_path),
        delay_iterations="0,50",
        max_iterations="5,5",
    )
    assert config.get_iteration_windows() == [(0, 5), (50, 5)]


def test_torch_multi_window_uses_new_profiler_per_window(tmp_path, monkeypatch):
    """Each torch profiling window needs a fresh profiler object."""
    fake_profiles: list[FakeTorchProfile] = []

    def fake_profile(**kwargs: object) -> FakeTorchProfile:
        assert "activities" in kwargs
        profile = FakeTorchProfile(kwargs)
        fake_profiles.append(profile)
        return profile

    monkeypatch.setattr("vllm.profiler.wrapper.torch.profiler.profile", fake_profile)

    config = ProfilerConfig(
        profiler="torch",
        torch_profiler_dir=str(tmp_path),
        torch_profiler_dump_cuda_time_total=False,
        delay_iterations="0,3",
        max_iterations="2,2",
    )
    profiler = TorchProfilerWrapper(
        config,
        worker_name="worker",
        local_rank=1,
        activities=["CPU"],
        on_trace_ready=lambda _: None,
    )

    profiler.start()
    profiler.step()
    profiler.step()
    assert len(fake_profiles) == 1
    assert fake_profiles[0].stop_call_count == 1
    assert (tmp_path / "profiler_out_1_window_0.txt").exists()
    assert profiler.profiler is None

    profiler.step()
    assert len(fake_profiles) == 2
    assert fake_profiles[1].start_call_count == 1

    profiler.step()
    assert fake_profiles[1].stop_call_count == 1
    assert (tmp_path / "profiler_out_1_window_1.txt").exists()
    assert profiler.profiler is None


def test_torch_multi_window_resets_schedule_per_window(tmp_path, monkeypatch):
    """Torch wait/warmup schedule accounting resets for every profiling window."""
    fake_profiles: list[FakeTorchProfile] = []

    def fake_profile(**kwargs: object) -> FakeTorchProfile:
        profile = FakeTorchProfile(kwargs)
        fake_profiles.append(profile)
        return profile

    monkeypatch.setattr("vllm.profiler.wrapper.torch.profiler.profile", fake_profile)

    config = ProfilerConfig(
        profiler="torch",
        torch_profiler_dir=str(tmp_path),
        torch_profiler_dump_cuda_time_total=False,
        delay_iterations="0,3",
        max_iterations="1,1",
        wait_iterations=1,
        warmup_iterations=1,
    )
    profiler = TorchProfilerWrapper(
        config,
        worker_name="worker",
        local_rank=1,
        activities=["CPU"],
        on_trace_ready=lambda _: None,
    )

    profiler.start()
    profiler.step()
    assert len(fake_profiles) == 1
    assert fake_profiles[0].step_call_count == 1
    assert fake_profiles[0].stop_call_count == 0
    assert fake_profiles[0].kwargs["schedule"] is not None

    profiler.step()
    assert fake_profiles[0].step_call_count == 2
    assert fake_profiles[0].stop_call_count == 1
    assert profiler.profiler is None

    profiler.step()
    assert len(fake_profiles) == 2
    assert fake_profiles[1].step_call_count == 1
    assert fake_profiles[1].stop_call_count == 0
    assert fake_profiles[1].kwargs["schedule"] is not None

    profiler.step()
    assert fake_profiles[1].step_call_count == 2
    assert fake_profiles[1].stop_call_count == 1
    assert profiler.profiler is None


def test_multi_window_rejects_length_mismatch():
    with pytest.raises(ValueError, match="same number"):
        ProfilerConfig(
            profiler="cuda",
            delay_iterations="0,50",
            max_iterations="5",
        )


def test_multi_window_rejects_overlap():
    """Window 1 must not start before window 0 finishes."""
    with pytest.raises(ValueError, match="must not overlap"):
        ProfilerConfig(
            profiler="cuda",
            delay_iterations="0,4",
            max_iterations="5,3",
        )


def test_multi_window_rejects_non_monotonic_delay():
    with pytest.raises(ValueError, match="must not overlap"):
        ProfilerConfig(
            profiler="cuda",
            delay_iterations="10,5",
            max_iterations="2,2",
        )


def test_multi_window_rejects_negative_element():
    with pytest.raises(ValueError, match="must be non-negative"):
        ProfilerConfig(
            profiler="cuda",
            delay_iterations="0,-5",
            max_iterations="2,2",
        )


def test_multi_window_rejects_zero_max_before_last_window():
    """Only the final multi-window max can be 0 (unlimited)."""
    with pytest.raises(ValueError, match="must be > 0 in multi-window mode"):
        ProfilerConfig(
            profiler="cuda",
            delay_iterations="0,10",
            max_iterations="0,5",
        )


def test_multi_window_allows_zero_max_for_last_window():
    """The final window may run until stop_profile is called."""
    config = _multi_window_config([(0, 2), (5, 0)])
    assert config.get_iteration_windows() == [(0, 2), (5, 0)]

    profiler = ConcreteWorkerProfiler(config)

    profiler.start()
    profiler.step()  # window 0 iter 1
    profiler.step()  # window 0 iter 2, then stop
    assert profiler._running is False
    assert profiler.stop_call_count == 1

    profiler.step()  # iter 3, still waiting for delay=5
    profiler.step()  # iter 4, still waiting for delay=5
    profiler.step()  # iter 5, final unlimited window starts
    assert profiler._running is True
    assert profiler.start_call_count == 2

    for _ in range(5):
        profiler.step()
    assert profiler._running is True
    assert profiler.stop_call_count == 1

    profiler.stop()
    assert profiler._running is False
    assert profiler.stop_call_count == 2


def test_single_window_via_string():
    """A single-value string should behave identically to a scalar."""
    config = ProfilerConfig(
        profiler="cuda",
        delay_iterations="2",
        max_iterations="2",
    )
    assert config.get_iteration_windows() == [(2, 2)]


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
