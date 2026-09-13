# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
import threading
import time
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, call, patch
from uuid import UUID

import pytest
from pydantic import ValidationError

from vllm.config import (
    CompilationConfig,
    CUDAGraphMode,
    ProfilerConfig,
    VllmConfig,
)
from vllm.config.profiler import _is_uri_path
from vllm.platforms import current_platform
from vllm.profiler.wrapper import (
    ProtonProfilerWrapper,
    TorchProfilerWrapper,
    WorkerProfiler,
)
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
    # And fully reset, not just paused -- a later start_profile must not be a
    # permanent no-op just because max_iterations already fired once.
    assert profiler._active is False
    assert profiler._active_iteration_count == 0
    assert profiler._profiling_for_iters == 0


def test_restart_after_max_iterations(default_profiler_config):
    """A start_profile after an auto-stop must actually restart, not be
    silently ignored (regression test: auto-stop used to leave _active
    True forever, so start() always bailed out early)."""
    default_profiler_config.max_iterations = 2
    profiler = ConcreteWorkerProfiler(default_profiler_config)

    profiler.start()
    profiler.step()
    profiler.step()
    profiler.step()  # exceeds max, auto-stops
    assert profiler._running is False
    assert profiler.start_call_count == 1

    profiler.start()
    assert profiler._active is True
    assert profiler._running is True
    assert profiler.start_call_count == 2

    profiler.step()
    assert profiler._running is True


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

    def test_skips_annotation_work_after_profiler_stops(self):
        worker = MagicMock()
        worker.profiler.is_running = False

        context = Worker.annotate_profile(worker, scheduler_output=None)

        worker.profiler.step.assert_called_once_with()
        worker.profiler.annotate_context_manager.assert_not_called()
        assert isinstance(context, nullcontext)


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
        # A stale reduce must not resurrect a cancelled request.
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

    def test_creates_profiler_on_remote_consensus(self):
        """A rank that never received /start_profile has no profiler wrapper
        yet. Reaching consensus via the OR-reduce must build one and start it,
        not silently consume the latch (regression: only the rank hit by the
        HTTP call ever captured)."""
        sync = DPProfilerSync()
        sync.observe(consensus=True)  # request came from another DP rank
        worker = self._worker(sync, profiler=None)
        worker.profiler = None

        def create():
            worker.profiler = MagicMock()

        worker._create_profiler.side_effect = create

        Worker._maybe_start_synced_profile(worker)

        worker._create_profiler.assert_called_once()
        worker.profiler.start.assert_called_once()

    def test_no_create_when_profiling_unconfigured(self):
        """Consensus with no profiler and no profiler config: skip, don't crash."""
        sync = DPProfilerSync()
        sync.observe(consensus=True)
        worker = self._worker(sync, profiler=None)
        worker.profiler = None
        worker.profiler_config = None

        Worker._maybe_start_synced_profile(worker)

        worker._create_profiler.assert_not_called()


class TestTorchProfilerWrapperAsyncExport:
    """Tests that trace export runs off the calling thread instead of
    blocking stop()/step() (see TorchProfilerWrapper._async_trace_ready).
    Uses CPU-only activities so these run without a GPU."""

    def _make_wrapper(self, tmp_path, on_trace_ready):
        config = ProfilerConfig(
            profiler="torch",
            torch_profiler_dir=str(tmp_path),
            torch_profiler_dump_cuda_time_total=False,
        )
        return TorchProfilerWrapper(
            profiler_config=config,
            worker_name="test-worker",
            local_rank=0,
            activities=["CPU"],
            on_trace_ready=on_trace_ready,
        )

    def test_stop_does_not_block_on_slow_export(self, tmp_path):
        calling_thread = threading.current_thread()
        handler_thread: dict[str, threading.Thread] = {}
        handler_done = threading.Event()

        def slow_handler(prof):
            handler_thread["thread"] = threading.current_thread()
            time.sleep(0.3)
            handler_done.set()

        wrapper = self._make_wrapper(tmp_path, slow_handler)
        wrapper.start()

        start = time.perf_counter()
        wrapper.stop()
        elapsed = time.perf_counter() - start

        assert elapsed < 0.2, "stop() should not block on the trace export"
        assert handler_done.wait(timeout=2.0), "background export never completed"
        assert handler_thread["thread"] is not calling_thread

    def test_export_errors_are_caught_not_raised(self, tmp_path):
        handler_done = threading.Event()

        def failing_handler(prof):
            handler_done.set()
            raise RuntimeError("boom")

        wrapper = self._make_wrapper(tmp_path, failing_handler)
        wrapper.start()

        wrapper.stop()  # must not raise even though the handler will

        assert handler_done.wait(timeout=2.0)


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


def make_proton(session_id: int | None = 7):
    data = SimpleNamespace(
        advance_phase=Mock(side_effect=range(1, 100)),
        clear=Mock(),
        get=Mock(return_value={"traceEvents": []}),
        get_msgpack=Mock(return_value=b"profile"),
    )
    return SimpleNamespace(
        start=Mock(return_value=session_id),
        activate=Mock(),
        deactivate=Mock(),
        finalize=Mock(),
        scope=Mock(return_value=nullcontext()),
        data=data,
    )


def make_proton_wrapper(
    tmp_path, proton=None, triton_version="3.7.0", **config_overrides
):
    proton = proton or make_proton()
    config = ProfilerConfig(
        profiler="proton",
        proton_profiler_dir=str(tmp_path),
        **config_overrides,
    )

    def import_module(name):
        if name == "triton.profiler":
            return proton
        assert name == "triton"
        return SimpleNamespace(__version__=triton_version)

    with patch(
        "vllm.profiler.wrapper.importlib.import_module", side_effect=import_module
    ):
        wrapper = ProtonProfilerWrapper(config, worker_name="rank_3")
    return wrapper, proton


_requires_cuda_for_proton = pytest.mark.skipif(
    not current_platform.is_cuda(),
    reason="Proton profiling tests require an NVIDIA CUDA platform.",
)


@_requires_cuda_for_proton
class TestProtonConfig:
    def test_normalizes_local_output_directory(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        config = ProfilerConfig(profiler="proton", proton_profiler_dir="profiles")
        assert config.proton_profiler_dir == os.path.join(tmp_path, "profiles")

    @pytest.mark.parametrize(
        ("options", "message"),
        [
            ({"proton_profiler_dir": ""}, "must be set"),
            ({"proton_profiler_dir": "s3://bucket/profiles"}, "local directory"),
            (
                {"proton_data": "tree", "proton_output_format": "chrome_trace"},
                "requires proton_data",
            ),
            (
                {"proton_data": "trace", "proton_output_format": "hatchet"},
                "requires proton_data",
            ),
            (
                {
                    "proton_data": "trace",
                    "proton_output_format": "hatchet_msgpack",
                },
                "requires proton_data",
            ),
            (
                {
                    "proton_data": "trace",
                    "proton_graph_attribution": True,
                },
                "requires proton_data='tree'",
            ),
        ],
    )
    def test_rejects_invalid_option_combinations(self, tmp_path, options, message):
        kwargs = {"proton_profiler_dir": str(tmp_path), **options}
        with pytest.raises(ValueError, match=message):
            ProfilerConfig(profiler="proton", **kwargs)

    @pytest.mark.parametrize(
        "field",
        [
            "proton_context",
            "proton_data",
            "proton_backend",
            "proton_hook",
            "proton_output_format",
        ],
    )
    def test_rejects_invalid_typed_options(self, field, tmp_path):
        with pytest.raises(ValidationError):
            ProfilerConfig(
                profiler="proton",
                proton_profiler_dir=str(tmp_path),
                **{field: "invalid"},
            )

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("proton_profiler_dir", "profiles"),
            ("proton_context", "python"),
            ("proton_data", "trace"),
            ("proton_backend", "cupti"),
            ("proton_mode", "pcsampling"),
            ("proton_hook", "triton"),
            ("proton_output_format", "chrome_trace"),
            ("proton_graph_attribution", True),
        ],
    )
    def test_rejects_proton_options_for_other_profilers(self, field, value):
        with pytest.raises(ValueError, match=f"{field} only applicable"):
            ProfilerConfig(**{field: value})

    def test_allows_proton_when_cuda_graphs_are_disabled(self, tmp_path):
        config = VllmConfig(
            profiler_config=ProfilerConfig(
                profiler="proton", proton_profiler_dir=str(tmp_path)
            ),
            compilation_config=CompilationConfig(cudagraph_mode=CUDAGraphMode.NONE),
        )

        assert config.compilation_config.cudagraph_mode == CUDAGraphMode.NONE

    def test_rejects_proton_on_non_cuda_platforms(self, tmp_path):
        with (
            patch("vllm.platforms.current_platform.is_cuda", return_value=False),
            pytest.raises(ValueError, match="supports NVIDIA CUDA only"),
        ):
            VllmConfig(
                profiler_config=ProfilerConfig(
                    profiler="proton", proton_profiler_dir=str(tmp_path)
                ),
                compilation_config=CompilationConfig(cudagraph_mode=CUDAGraphMode.NONE),
            )

    @pytest.mark.parametrize("encoder_only", [False, True])
    @pytest.mark.parametrize("attribution", [False, True])
    def test_cuda_graphs_require_attribution(self, tmp_path, encoder_only, attribution):
        # Encoder graphs are independent of the decoder's cudagraph_mode.
        expected = (
            nullcontext()
            if attribution
            else pytest.raises(
                ValueError, match="requires proton_graph_attribution=True"
            )
        )
        with expected:
            VllmConfig(
                profiler_config=ProfilerConfig(
                    profiler="proton",
                    proton_profiler_dir=str(tmp_path),
                    proton_graph_attribution=attribution,
                ),
                compilation_config=CompilationConfig(
                    cudagraph_mode=(
                        CUDAGraphMode.NONE if encoder_only else CUDAGraphMode.FULL
                    ),
                    cudagraph_mm_encoder=encoder_only,
                ),
            )

    def test_validates_default_cuda_graph_mode_after_resolution(self, tmp_path):
        with pytest.raises(ValueError, match="requires proton_graph_attribution=True"):
            VllmConfig(
                profiler_config=ProfilerConfig(
                    profiler="proton", proton_profiler_dir=str(tmp_path)
                ),
            )

    @pytest.mark.parametrize(
        "mode",
        [
            "periodic_flushing",
            "periodic_flushing:format=hatchet",
            "PERIODIC_FLUSHING:format=hatchet",
        ],
    )
    def test_graph_attribution_rejects_periodic_flushing(self, tmp_path, mode):
        # Reject before Proton's native phase manager can abort the worker.
        with pytest.raises(ValueError, match="incompatible with periodic_flushing"):
            ProfilerConfig(
                profiler="proton",
                proton_profiler_dir=str(tmp_path),
                proton_graph_attribution=True,
                proton_mode=mode,
            )

    @pytest.mark.parametrize("attribution", [False, True])
    @pytest.mark.parametrize(
        "mode", ["pcsampling", "pcsampling:interval=100", "PcSampling:interval=100"]
    )
    @pytest.mark.parametrize("encoder_only", [False, True])
    def test_pcsampling_requires_graphs_disabled(
        self, tmp_path, attribution, mode, encoder_only
    ):
        with pytest.raises(ValueError, match="PC sampling requires CUDA graphs"):
            VllmConfig(
                profiler_config=ProfilerConfig(
                    profiler="proton",
                    proton_profiler_dir=str(tmp_path),
                    proton_graph_attribution=attribution,
                    proton_mode=mode,
                ),
                compilation_config=CompilationConfig(
                    cudagraph_mode=CUDAGraphMode.NONE
                    if encoder_only
                    else CUDAGraphMode.FULL,
                    cudagraph_mm_encoder=encoder_only,
                ),
            )

    def test_ordinary_proton_keeps_mrv1_graph_restriction(self, tmp_path, monkeypatch):
        monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "0")
        with pytest.raises(ValueError, match="requires proton_graph_attribution=True"):
            VllmConfig(
                profiler_config=ProfilerConfig(
                    profiler="proton", proton_profiler_dir=str(tmp_path)
                ),
                compilation_config=CompilationConfig(cudagraph_mode=CUDAGraphMode.FULL),
            )

    def test_graph_attribution_rejects_mrv1(self, tmp_path, monkeypatch):
        monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "0")
        with pytest.raises(ValueError, match="requires the V2 model runner"):
            VllmConfig(
                profiler_config=ProfilerConfig(
                    profiler="proton",
                    proton_profiler_dir=str(tmp_path),
                    proton_graph_attribution=True,
                )
            )


@_requires_cuda_for_proton
class TestProtonProfilerWrapper:
    def test_passes_config_and_global_rank_name_to_proton(self, tmp_path):
        wrapper, proton = make_proton_wrapper(
            tmp_path,
            proton_context="python",
            proton_data="trace",
            proton_backend="cupti",
            proton_mode="pcsampling",
            proton_hook="triton",
            proton_output_format="chrome_trace",
        )

        wrapper.start()

        start_args = proton.start.call_args.kwargs
        assert start_args["name"].startswith(os.path.join(tmp_path, "proton_rank_3_"))
        assert start_args["name"].endswith("_run0")
        assert start_args | {"name": None} == {
            "name": None,
            "context": "python",
            "data": "trace",
            "backend": "cupti",
            "mode": "pcsampling",
            "hook": "triton",
        }
        wrapper.stop()
        proton.finalize.assert_called_once_with(session=7, output_format="chrome_trace")
        assert tmp_path.is_dir()

    def test_finalizes_each_profile_with_unique_output_names(self, tmp_path):
        proton = make_proton()
        proton.start.side_effect = [7, 8]
        wrapper, proton = make_proton_wrapper(tmp_path, proton)

        wrapper.start()
        wrapper.start()
        wrapper.stop()
        wrapper.start()
        wrapper.stop()

        names = [c.kwargs["name"] for c in proton.start.call_args_list]
        assert len(names) == len(set(names)) == 2
        assert names[0].endswith("_run0")
        assert names[1].endswith("_run1")
        assert proton.deactivate.call_count == 2
        assert proton.finalize.call_args_list == [call(session=7), call(session=8)]

    def test_output_names_are_unique_across_worker_restarts(self, tmp_path):
        with patch(
            "vllm.profiler.wrapper.uuid4",
            side_effect=[UUID(int=1), UUID(int=2)],
        ):
            first, first_proton = make_proton_wrapper(tmp_path)
            second, second_proton = make_proton_wrapper(tmp_path)

        first.start()
        second.start()

        first_name = first_proton.start.call_args.kwargs["name"]
        second_name = second_proton.start.call_args.kwargs["name"]
        assert first_name != second_name
        assert first_name.endswith(f"_{UUID(int=1).hex}_run0")
        assert second_name.endswith(f"_{UUID(int=2).hex}_run0")

    @pytest.mark.parametrize(
        ("option", "value", "feature"),
        [
            ("proton_output_format", "hatchet_msgpack", "hatchet_msgpack"),
            ("proton_mode", "periodic_flushing", "periodic flushing"),
        ],
    )
    def test_newer_features_reject_triton_3_6(self, tmp_path, option, value, feature):
        with pytest.raises(RuntimeError, match=feature):
            make_proton_wrapper(tmp_path, triton_version="3.6.0", **{option: value})

    @pytest.mark.parametrize("version", ["3.6.0", "unknown"])
    def test_graph_attribution_requires_phase_api(self, tmp_path, version):
        with pytest.raises(RuntimeError, match="requires Triton >= 3.7"):
            make_proton_wrapper(
                tmp_path, triton_version=version, proton_graph_attribution=True
            )

    def test_ordinary_profiling_still_supports_triton_3_6(self, tmp_path):
        wrapper, proton = make_proton_wrapper(tmp_path, triton_version="3.6.0")
        wrapper.start()
        wrapper.stop()
        proton.finalize.assert_called_once_with(session=7)

    @pytest.mark.parametrize(
        ("option", "value"),
        [
            ("proton_output_format", "hatchet_msgpack"),
            ("proton_mode", "periodic_flushing"),
        ],
    )
    def test_triton_3_7_features(self, tmp_path, option, value):
        make_proton_wrapper(tmp_path, triton_version="3.7.0", **{option: value})

    def test_rejects_output_format_when_finalize_lacks_capability(self, tmp_path):
        proton = make_proton()
        proton.finalize = lambda session=None: None

        with pytest.raises(RuntimeError, match="does not support selecting"):
            make_proton_wrapper(tmp_path, proton, proton_output_format="hatchet")

    def test_cuda_graph_tree_phase_is_written_at_stop(self, tmp_path):
        wrapper, proton = make_proton_wrapper(
            tmp_path,
            proton_context="python",
            proton_graph_attribution=True,
        )
        with wrapper.capture_cuda_graphs():
            proton.start.assert_called_once()
            proton.deactivate.assert_not_called()

        capture_args = proton.start.call_args.kwargs
        assert capture_args["context"] == "python"
        assert capture_args["data"] == "tree"
        proton.data.advance_phase.assert_called_once_with(7)
        proton.deactivate.assert_called_once_with(session=7, flushing=True)
        proton.data.clear.assert_called_once_with(7, 0)

        wrapper.start()
        wrapper.stop()
        wrapper.start()
        wrapper.stop()

        assert proton.data.get.call_args_list == [call(7, 1), call(7, 2)]
        proton.data.clear.assert_has_calls([call(7, 0), call(7, 1), call(7, 2)])
        output_names = sorted(tmp_path.glob("proton_rank_3_*.hatchet"))
        assert len(output_names) == 2
        assert output_names[0].name.endswith("_run0.hatchet")
        assert output_names[1].name.endswith("_run1.hatchet")

    def test_capture_is_noop_without_opt_in(self, tmp_path):
        wrapper, proton = make_proton_wrapper(tmp_path)
        with wrapper.capture_cuda_graphs():
            pass
        proton.start.assert_not_called()

    @pytest.mark.parametrize("delay", [0, 2])
    def test_duplicate_start_preserves_output_prefix(self, tmp_path, delay):
        wrapper, proton = make_proton_wrapper(
            tmp_path, proton_graph_attribution=True, delay_iterations=delay
        )
        with wrapper.capture_cuda_graphs():
            pass
        wrapper.set_output_name("first")
        wrapper.start()
        wrapper.set_output_name("duplicate")
        wrapper.start()
        for _ in range(delay):
            wrapper.step()
        wrapper.stop()
        assert len(list(tmp_path.glob("proton_first_*.hatchet"))) == 1
        assert not list(tmp_path.glob("proton_duplicate_*"))
        wrapper.set_output_name("second")
        wrapper.start()
        for _ in range(delay):
            wrapper.step()
        wrapper.stop()
        assert len(list(tmp_path.glob("proton_second_*.hatchet"))) == 1

    def test_failed_export_clears_activity_and_allows_next_interval(self, tmp_path):
        wrapper, proton = make_proton_wrapper(tmp_path, proton_graph_attribution=True)
        with wrapper.capture_cuda_graphs():
            pass
        proton.data.get.side_effect = [OSError("disk full"), []]
        wrapper.start()
        wrapper.stop()
        proton.data.clear.assert_called_with(7, 1)
        wrapper.start()
        wrapper.stop()
        proton.data.clear.assert_called_with(7, 2)
        assert len(list(tmp_path.glob("*_run1.hatchet"))) == 1

    def test_cuda_graph_context_deactivates_after_capture_error(self, tmp_path):
        wrapper, proton = make_proton_wrapper(tmp_path, proton_graph_attribution=True)

        with (
            pytest.raises(RuntimeError, match="capture failed"),
            wrapper.capture_cuda_graphs(),
        ):
            raise RuntimeError("capture failed")

        proton.deactivate.assert_called_once_with(session=7, flushing=True)

    def test_cuda_graph_capture_deactivates_when_phase_advance_fails(self, tmp_path):
        wrapper, proton = make_proton_wrapper(tmp_path, proton_graph_attribution=True)
        proton.data.advance_phase.side_effect = RuntimeError("advance failed")

        with (
            pytest.raises(RuntimeError, match="advance failed"),
            wrapper.capture_cuda_graphs(),
        ):
            pass

        proton.deactivate.assert_called_once_with(session=7, flushing=True)
        proton.data.clear.assert_not_called()

    def test_cuda_graph_stop_deactivates_when_phase_advance_fails(self, tmp_path):
        wrapper, proton = make_proton_wrapper(tmp_path, proton_graph_attribution=True)
        with wrapper.capture_cuda_graphs():
            pass
        proton.data.clear.reset_mock()
        proton.data.advance_phase.side_effect = RuntimeError("advance failed")

        wrapper._start()
        with pytest.raises(RuntimeError, match="advance failed"):
            wrapper._stop()

        proton.deactivate.assert_called_with(session=7, flushing=True)
        proton.data.clear.assert_not_called()

    def test_shutdown_finalizes_cuda_graph_capture_session(self, tmp_path):
        wrapper, proton = make_proton_wrapper(tmp_path, proton_graph_attribution=True)
        with wrapper.capture_cuda_graphs():
            pass

        wrapper.shutdown()
        wrapper.shutdown()

        proton.finalize.assert_called_once_with(session=7)

    def test_missing_proton_has_actionable_error(self, tmp_path):
        config = ProfilerConfig(profiler="proton", proton_profiler_dir=str(tmp_path))
        with (
            patch(
                "vllm.profiler.wrapper.importlib.import_module",
                side_effect=ImportError,
            ),
            pytest.raises(RuntimeError, match="requires a Triton installation"),
        ):
            ProtonProfilerWrapper(config, worker_name="rank_0")

    def test_scope_annotations_delegate_to_proton(self, tmp_path):
        wrapper, proton = make_proton_wrapper(tmp_path)
        wrapper.start()

        context = wrapper.annotate_context_manager("decode")

        proton.scope.assert_called_once_with("decode")
        assert context is not None


@_requires_cuda_for_proton
def test_gpu_worker_creates_proton_profiler():
    worker = MagicMock()
    worker.rank = 1
    worker.profiler = None
    worker.profiler_config.profiler = "proton"

    with (
        patch(
            "vllm.distributed.utils.get_worker_rank_suffix",
            return_value="rank1",
        ),
        patch("vllm.v1.worker.gpu_worker.ProtonProfilerWrapper") as wrapper,
    ):
        Worker.profile(worker)

    wrapper.assert_called_once_with(worker.profiler_config, worker_name="rank1")
    worker.profiler.start.assert_called_once_with()


@_requires_cuda_for_proton
def test_gpu_worker_recreates_proton_profiler_for_each_run():
    worker = MagicMock()
    worker.rank = 1
    worker.profiler = None
    worker.profiler_config.profiler = "proton"

    with (
        patch(
            "vllm.distributed.utils.get_worker_rank_suffix",
            return_value="rank1",
        ),
        patch("vllm.v1.worker.gpu_worker.ProtonProfilerWrapper") as wrapper,
    ):
        wrapper.return_value.has_cuda_graph_session = False
        Worker.profile(worker, profile_prefix="first")
        Worker.profile(worker, is_start=False)
        Worker.profile(worker, profile_prefix="second")

    assert wrapper.call_args_list == [
        call(worker.profiler_config, worker_name="first_rank1"),
        call(worker.profiler_config, worker_name="second_rank1"),
    ]
    assert wrapper.return_value.start.call_count == 2


@_requires_cuda_for_proton
def test_gpu_worker_reuses_cuda_graph_proton_session():
    worker = MagicMock()
    worker.rank = 1
    worker.profiler = MagicMock(spec=ProtonProfilerWrapper)
    worker.profiler.has_cuda_graph_session = True
    worker.profiler_config.profiler = "proton"

    with patch(
        "vllm.distributed.utils.get_worker_rank_suffix",
        return_value="rank1",
    ):
        Worker.profile(worker, profile_prefix="first")
        Worker.profile(worker, is_start=False)

    worker.profiler.set_output_name.assert_called_once_with("first_rank1")
    worker.profiler.start.assert_called_once_with()
    worker.profiler.stop.assert_called_once_with()


@_requires_cuda_for_proton
@pytest.mark.parametrize("runner", ["attribution_off", "v1", "no_capture"])
def test_proton_not_initialized_without_capture(runner):
    worker = MagicMock()
    worker.profiler = None
    worker.profiler_config.profiler = "proton"
    worker.profiler_config.proton_graph_attribution = runner != "attribution_off"
    worker.use_v2_model_runner = runner != "v1"
    worker.model_runner.needs_cudagraph_capture.return_value = runner != "no_capture"

    context = Worker._get_cudagraph_capture_context(worker)

    assert worker.profiler is None
    with context:
        pass


@_requires_cuda_for_proton
def test_proton_initializes_before_cuda_graph_capture():
    class FakeProtonProfiler:
        def __init__(self, config, worker_name):
            self.config = config
            self.worker_name = worker_name
            self.capture_context = nullcontext()

        def capture_cuda_graphs(self):
            return self.capture_context

    worker = MagicMock()
    worker.rank = 2
    worker.profiler = None
    worker.profiler_config.profiler = "proton"
    worker.profiler_config.proton_graph_attribution = True
    worker.use_v2_model_runner = True
    worker.model_runner.needs_cudagraph_capture.return_value = True

    with (
        patch(
            "vllm.distributed.utils.get_worker_rank_suffix",
            return_value="rank2",
        ),
        patch(
            "vllm.profiler.wrapper.ProtonProfilerWrapper",
            FakeProtonProfiler,
        ),
    ):
        context = Worker._get_cudagraph_capture_context(worker)

    assert worker.profiler.config is worker.profiler_config
    assert worker.profiler.worker_name == "rank2"
    assert context is worker.profiler.capture_context


@_requires_cuda_for_proton
@pytest.mark.parametrize("context", ["shadow", "python"])
@pytest.mark.parametrize("output_format", ["hatchet", "hatchet_msgpack"])
def test_proton_cuda_graph_replay_attribution_on_gpu(tmp_path, context, output_format):
    """Both intervals contain replay kernels, without capture-only activity."""
    import json

    import torch
    import triton
    import triton.profiler as proton
    from packaging.version import Version

    if Version(triton.__version__) < Version("3.7"):
        pytest.skip("Graph attribution requires Triton >= 3.7")

    wrapper = ProtonProfilerWrapper(
        ProfilerConfig(
            profiler="proton",
            proton_profiler_dir=str(tmp_path),
            proton_graph_attribution=True,
            proton_context=context,
            proton_output_format=output_format,
        ),
        worker_name="gpu",
    )
    worker = SimpleNamespace(
        profiler=wrapper,
        use_v2_model_runner=True,
    )
    x = torch.ones(1024, device="cuda")
    graph = torch.cuda.CUDAGraph()

    def capture_only():
        x.add_(1)

    def captured_add():
        x.add_(1)

    def kernel_metrics(value):
        if isinstance(value, list):
            return [metric for child in value for metric in kernel_metrics(child)]
        if isinstance(value, dict):
            metrics = value.get("metrics", {})
            return ([metrics] if metrics.get("count", 0) else []) + [
                metric for child in value.values() for metric in kernel_metrics(child)
            ]
        return []

    try:
        with Worker._get_cudagraph_capture_context(worker):
            with proton.scope("capture_only"):
                capture_only()
            torch.accelerator.synchronize()
            with torch.cuda.graph(graph), proton.scope("captured_add"):
                captured_add()
        for run in range(2):
            wrapper.start()
            with wrapper.annotate_context_manager(f"replay_{run}"):
                graph.replay()
            wrapper.stop()
            (path,) = tmp_path.glob(f"*_run{run}.{output_format}")
            if output_format == "hatchet_msgpack":
                import msgpack

                data = msgpack.unpackb(path.read_bytes())
            else:
                data = json.loads(path.read_text())
            serialized = json.dumps(data)
            assert "<captured_at>" in serialized
            assert "captured_add" in serialized
            assert "capture_only" not in serialized
            assert f"replay_{1 - run}" not in serialized
            metrics = kernel_metrics(data)
            assert sum(metric["count"] for metric in metrics) == 1
            assert sum(metric["time (ns)"] for metric in metrics) > 0
        torch.testing.assert_close(x, torch.full_like(x, 4))
    finally:
        wrapper.shutdown()
    assert not list(tmp_path.glob(".proton_cuda_graph_session*"))
