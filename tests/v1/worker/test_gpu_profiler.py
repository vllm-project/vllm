# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
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
from vllm.profiler.wrapper import ProtonProfilerWrapper, WorkerProfiler
from vllm.v1.core.sched.output import CachedRequestData
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

    def _annotate(self, detailed: bool):
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
        return worker.profiler.annotate_context_manager.call_args

    def test_simple_format_mixed(self):
        assert self._annotate(detailed=False).args[0] == (
            "execute_context_1(4)_generation_1(1)"
        )

    def test_detailed_format_mixed(self):
        # ctx1: sq=4, sk=4, sqsq=16, sqsk=16 | gen1: sq=1, sk=11, sqsq=1, sqsk=11 | bs=5
        assert self._annotate(detailed=True).args[0] == (
            "execute_5_context_1(sq4sk4sqsq16sqsk16)_generation_1(sq1sk11sqsq1sqsk11)"
        )

    def test_numeric_metrics(self):
        assert self._annotate(detailed=False).kwargs["metrics"] == {
            "num_context_requests": 1,
            "num_context_tokens": 4,
            "num_generation_requests": 1,
            "num_generation_tokens": 1,
        }

    def test_skips_annotations_outside_profile_window(self):
        worker = MagicMock()
        worker.profiler.is_running = False

        context = Worker.annotate_profile(worker, MagicMock())

        worker.profiler.step.assert_called_once_with()
        worker.profiler.annotate_context_manager.assert_not_called()
        with context:
            pass


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
    return SimpleNamespace(
        start=Mock(return_value=session_id),
        activate=Mock(),
        deactivate=Mock(),
        finalize=Mock(),
        scope=Mock(return_value=nullcontext()),
    )


def make_proton_wrapper(
    tmp_path, proton=None, triton_version="3.6.0", **config_overrides
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

    def test_rejects_proton_when_cuda_graphs_are_enabled(self, tmp_path):
        with pytest.raises(ValueError, match="requires CUDA graphs to be disabled"):
            VllmConfig(
                profiler_config=ProfilerConfig(
                    profiler="proton", proton_profiler_dir=str(tmp_path)
                ),
                compilation_config=CompilationConfig(
                    cudagraph_mode=CUDAGraphMode.PIECEWISE
                ),
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
            make_proton_wrapper(tmp_path, **{option: value})

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

    @pytest.mark.parametrize("fail_cleanup", [False, True])
    def test_shutdown_finalizes_active_session(self, tmp_path, fail_cleanup):
        proton = make_proton()
        wrapper, proton = make_proton_wrapper(tmp_path, proton)
        wrapper.start()
        if fail_cleanup:
            proton.deactivate.side_effect = RuntimeError("deactivate failed")
            proton.finalize.side_effect = RuntimeError("finalize failed")

        wrapper.shutdown()
        wrapper.shutdown()

        proton.deactivate.assert_called_once_with(session=7)
        proton.finalize.assert_called_once_with(session=7)

    @pytest.mark.parametrize(
        ("first_result", "message"),
        [
            (None, "did not create"),
            (RuntimeError("CUPTI unavailable"), "CUPTI unavailable"),
        ],
    )
    def test_recovers_from_start_errors(self, tmp_path, first_result, message):
        proton = make_proton()
        proton.start.side_effect = [first_result, 8]
        wrapper, _ = make_proton_wrapper(tmp_path, proton)

        with pytest.raises(RuntimeError, match=message):
            wrapper.start()

        wrapper.start()
        assert wrapper.is_running
        wrapper.stop()
        proton.finalize.assert_called_once_with(session=8)

    @pytest.mark.parametrize(
        ("failing_call", "message"),
        [("finalize", "write failed"), ("deactivate", "deactivate failed")],
    )
    def test_recovers_from_stop_errors(self, tmp_path, failing_call, message):
        proton = make_proton()
        proton.start.side_effect = [7, 8]
        getattr(proton, failing_call).side_effect = RuntimeError(message)
        wrapper, _ = make_proton_wrapper(tmp_path, proton)
        wrapper.start()

        with pytest.raises(RuntimeError, match=message):
            wrapper.stop()

        getattr(proton, failing_call).side_effect = None
        wrapper.start()
        wrapper.stop()
        assert proton.finalize.call_args_list == [call(session=7), call(session=8)]

    def test_automatic_stop_errors_do_not_fail_inference(self, tmp_path):
        proton = make_proton()
        proton.finalize.side_effect = RuntimeError("write failed")
        wrapper, _ = make_proton_wrapper(tmp_path, proton, max_iterations=1)
        wrapper.start()

        wrapper.step()
        wrapper.step()
        worker = MagicMock(profiler=wrapper)
        context = Worker.annotate_profile(worker, scheduler_output=None)

        assert isinstance(context, nullcontext)
        proton.scope.assert_not_called()
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

        proton.scope.assert_called_once_with("decode", metrics=None)
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
        Worker.profile(worker, profile_prefix="first")
        Worker.profile(worker, is_start=False)
        Worker.profile(worker, profile_prefix="second")

    assert wrapper.call_args_list == [
        call(worker.profiler_config, worker_name="first_rank1"),
        call(worker.profiler_config, worker_name="second_rank1"),
    ]
    assert wrapper.return_value.start.call_count == 2
