# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
from contextlib import nullcontext
from threading import Event
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
from vllm.profiler.proton import ProtonPhaseManager
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
        is_phase_complete=Mock(return_value=True),
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
            ("proton_flush_interval", 2),
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
    def test_graph_attribution_supports_periodic_flushing(self, tmp_path, mode):
        config = VllmConfig(
            profiler_config=ProfilerConfig(
                profiler="proton",
                proton_profiler_dir=str(tmp_path),
                proton_graph_attribution=True,
                proton_mode=mode,
            ),
        )
        assert config.profiler_config.proton_mode == mode

    @pytest.mark.parametrize(
        "options",
        [
            {"proton_data": "trace"},
            {"proton_mode": "periodic_flushing:format=chrome_trace"},
            {"proton_mode": "periodic_flushing:interval=1"},
            {
                "proton_mode": "periodic_flushing:format=hatchet",
                "proton_output_format": "hatchet_msgpack",
            },
            {"proton_flush_interval": 0},
            {"proton_mode": None, "proton_flush_interval": 2},
        ],
    )
    def test_periodic_flushing_rejects_invalid_options(self, tmp_path, options):
        config = dict(
            profiler="proton",
            proton_profiler_dir=str(tmp_path),
            proton_mode="periodic_flushing",
        )
        with pytest.raises(ValueError):
            ProfilerConfig(**(config | options))

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

    @pytest.mark.parametrize("capture", [False, True])
    def test_periodic_profiles_are_written_before_stop_and_isolate_runs(
        self, tmp_path, capture
    ):
        wrapper, proton = make_proton_wrapper(
            tmp_path,
            proton_mode="PERIODIC_FLUSHING:format=hatchet",
            proton_graph_attribution=capture,
            proton_flush_interval=2,
        )
        if capture:
            with wrapper.capture_cuda_graphs():
                pass
        for run in range(2):
            wrapper.start()
            wrapper.start()
            assert wrapper.has_cuda_graph_session == capture
            wrapper.step()
            wrapper.step()
            assert not list(tmp_path.glob(f"*_run{run}.part_*.hatchet"))
            wrapper.step()
            wrapper._phase_manager.poll(wait=True)
            assert len(list(tmp_path.glob(f"*_run{run}.part_0.hatchet"))) == 1
            assert wrapper.is_running
            wrapper.stop()
            wrapper.stop()
            assert len(list(tmp_path.glob(f"*_run{run}.part_*.hatchet"))) == 2
        assert all(c.kwargs["mode"] is None for c in proton.start.call_args_list)
        assert proton.start.call_count == (1 if capture else 2)
        assert proton.finalize.call_count == (0 if capture else 2)
        wrapper.shutdown()
        assert proton.finalize.call_count == (1 if capture else 2)

    def test_periodic_export_failure_keeps_collecting_without_partial_files(
        self, tmp_path
    ):
        wrapper, proton = make_proton_wrapper(
            tmp_path,
            proton_mode="periodic_flushing",
            proton_flush_interval=1,
        )
        wrapper.start()
        wrapper.step()
        proton.data.get.side_effect = [OSError("disk full"), [], [], []]
        proton.data.is_phase_complete.return_value = False
        wrapper.step()
        assert wrapper.is_running
        proton.data.is_phase_complete.return_value = True
        with pytest.raises(OSError, match="disk full"):
            wrapper._phase_manager.poll(wait=True)
        proton.deactivate.assert_not_called()
        proton.data.clear.assert_not_called()
        assert not list(tmp_path.glob("*.hatchet"))
        assert not list(tmp_path.glob("*.tmp"))
        wrapper.step()
        wrapper.stop()
        assert len(list(tmp_path.glob("*_run0.part_0.hatchet"))) == 1
        assert len(list(tmp_path.glob("*_run0.part_1.hatchet"))) == 1
        assert len(list(tmp_path.glob("*_run0.part_2.hatchet"))) == 1

    def test_periodic_flushing_respects_delay_and_iteration_limit(self, tmp_path):
        wrapper, proton = make_proton_wrapper(
            tmp_path,
            proton_mode="periodic_flushing",
            proton_flush_interval=1,
            delay_iterations=2,
            max_iterations=2,
        )
        wrapper.start()
        wrapper.step()
        proton.start.assert_not_called()
        wrapper.step()
        assert not list(tmp_path.glob("*.hatchet"))
        wrapper.step()
        wrapper._phase_manager.poll(wait=True)
        assert len(list(tmp_path.glob("*.hatchet"))) == 1
        wrapper.step()
        assert not wrapper.is_running
        assert len(list(tmp_path.glob("*.hatchet"))) == 2

    def test_periodic_waits_for_complete_phase_and_exports_in_background(
        self, tmp_path
    ):
        """Incomplete GPU data and slow disk writes must not block worker steps."""
        wrapper, proton = make_proton_wrapper(
            tmp_path, proton_mode="periodic_flushing", proton_flush_interval=2
        )
        entered, release = Event(), Event()

        def slow_read(*args):
            entered.set()
            assert release.wait(timeout=10)
            return []

        proton.data.is_phase_complete.return_value = False
        proton.data.get.side_effect = slow_read
        try:
            wrapper.start()
            for _ in range(3):
                wrapper.step()
            proton.data.get.assert_not_called()
            proton.data.clear.assert_not_called()
            proton.data.is_phase_complete.return_value = True
            wrapper.step()
            assert entered.wait(timeout=10)
            # The exporter is still blocked, but another step can proceed.
            wrapper.step()
            assert wrapper.is_running
            proton.deactivate.assert_not_called()
            proton.data.clear.assert_not_called()
        finally:
            release.set()
            wrapper.stop()
            wrapper.shutdown()
        assert len(list(tmp_path.glob("*_run0.part_*.hatchet"))) == 3

    @pytest.mark.parametrize("capture", [False, True])
    @pytest.mark.parametrize("recover_at_shutdown", [False, True])
    @pytest.mark.parametrize("failure", ["flush", "advance", "write"])
    def test_failed_periodic_stop_retains_phases_until_recovery(
        self, tmp_path, monkeypatch, capture, recover_at_shutdown, failure
    ):
        """Worker stop must retain failed sessions for the next run or shutdown."""
        wrapper, proton = make_proton_wrapper(
            tmp_path,
            proton_mode="periodic_flushing",
            proton_graph_attribution=capture,
            proton_flush_interval=1,
        )
        if capture:
            with wrapper.capture_cuda_graphs():
                pass
        worker = SimpleNamespace(
            rank=0,
            profiler=wrapper,
            profiler_config=SimpleNamespace(profiler="proton"),
            elastic_ep_executor=Mock(),
        )
        monkeypatch.setattr(
            "vllm.distributed.utils.get_worker_rank_suffix", lambda **kwargs: "rank0"
        )
        monkeypatch.setattr("vllm.v1.worker.gpu_worker.gc.unfreeze", lambda: None)
        monkeypatch.setattr(
            "vllm.v1.worker.gpu_worker.ensure_kv_transfer_shutdown", None
        )
        monkeypatch.setattr(
            "vllm.v1.worker.gpu_worker.ensure_ec_transfer_shutdown", None
        )
        monkeypatch.setattr(current_platform, "is_cuda_alike", lambda: False)
        proton.data.clear.reset_mock()
        proton.data.is_phase_complete.return_value = False
        Worker.profile(worker, profile_prefix="first")
        parts = 1 if failure == "write" else 2
        for _ in range(parts):
            worker.profiler.step()
        failing_call = {
            "flush": proton.deactivate,
            "advance": proton.data.advance_phase,
            "write": proton.data.get,
        }[failure]
        original_effect = failing_call.side_effect
        failing_call.side_effect = RuntimeError(f"{failure} failed")
        Worker.profile(worker, is_start=False)
        assert worker.profiler is wrapper
        if failure != "write":
            proton.data.get.assert_not_called()
        proton.data.clear.assert_not_called()
        proton.finalize.assert_not_called()
        failing_call.side_effect = original_effect
        if not recover_at_shutdown:
            Worker.profile(worker, profile_prefix="second")
            worker.profiler.step()
            Worker.profile(worker, is_start=False)
            assert (worker.profiler is wrapper) == capture
        Worker.shutdown(worker)
        first_phase = 1 if capture else 0
        expected = range(first_phase, first_phase + parts + (not recover_at_shutdown))
        assert proton.data.clear.call_args_list == [
            call(7, phase) for phase in expected
        ]
        assert len(list(tmp_path.glob("proton_first_*_run0.part_*.hatchet"))) == parts
        assert len(list(tmp_path.glob("proton_second_*_run1.part_*.hatchet"))) == (
            0 if recover_at_shutdown else 1
        )
        proton.start.assert_called_once()
        proton.finalize.assert_called_once()

    def test_completed_phase_is_retained_when_clear_fails(self, tmp_path):
        proton = make_proton()
        manager = ProtonPhaseManager(proton, 7, None, asynchronous=True)
        proton.data.clear.side_effect = [RuntimeError("clear failed"), None]
        try:
            manager.rotate(str(tmp_path / "part0"))
            with pytest.raises(RuntimeError, match="clear failed"):
                manager.poll(wait=True)
            manager.poll(wait=True)
            assert proton.data.clear.call_args_list == [call(7, 0), call(7, 0)]
            proton.data.get.assert_called_once_with(7, 0)
        finally:
            manager.close()

    def test_capture_is_noop_without_opt_in(self, tmp_path):
        wrapper, proton = make_proton_wrapper(tmp_path)
        with wrapper.capture_cuda_graphs():
            pass
        proton.start.assert_not_called()

    def test_repeated_capture_discards_only_capture_activity(self, tmp_path):
        wrapper, proton = make_proton_wrapper(tmp_path, proton_graph_attribution=True)

        for _ in range(2):
            with wrapper.capture_cuda_graphs():
                pass
            wrapper.start()
            wrapper.stop()

        proton.start.assert_called_once()
        assert proton.data.get.call_args_list == [call(7, 1), call(7, 3)]
        assert proton.data.clear.call_args_list == [
            call(7, phase) for phase in range(4)
        ]
        assert len(list(tmp_path.glob("proton_rank_3_*.hatchet"))) == 2

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

    def test_failed_export_retries_activity_in_next_interval(self, tmp_path):
        wrapper, proton = make_proton_wrapper(tmp_path, proton_graph_attribution=True)
        with wrapper.capture_cuda_graphs():
            pass
        proton.data.get.side_effect = [OSError("disk full"), [], []]
        proton.data.clear.reset_mock()
        wrapper.start()
        wrapper.stop()
        proton.data.clear.assert_not_called()
        wrapper.start()
        wrapper.stop()
        proton.data.clear.assert_called_with(7, 2)
        assert len(list(tmp_path.glob("*_run0.hatchet"))) == 1
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

    @pytest.mark.parametrize("capture", [True, False])
    def test_failed_flush_preserves_incomplete_activity(self, tmp_path, capture):
        """Neither capture discard nor interval export may clear unflushed data."""
        wrapper, proton = make_proton_wrapper(tmp_path, proton_graph_attribution=True)
        if not capture:
            with wrapper.capture_cuda_graphs():
                pass
            proton.data.clear.reset_mock()
            wrapper.start()
        proton.deactivate.side_effect = RuntimeError("flush failed")

        if capture:
            with (
                pytest.raises(RuntimeError, match="flush failed"),
                wrapper.capture_cuda_graphs(),
            ):
                pass
        else:
            wrapper.stop()

        proton.data.get.assert_not_called()
        proton.data.clear.assert_not_called()
        assert not list(tmp_path.glob("*.hatchet"))

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
        wrapper.return_value.has_retained_session = False
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
    worker.profiler.has_retained_session = True
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
@pytest.mark.parametrize("periodic", [False, True])
@pytest.mark.parametrize("capture", [False, True])
def test_proton_cuda_graph_replay_attribution_on_gpu(
    tmp_path, context, output_format, periodic, capture
):
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
            proton_graph_attribution=capture,
            proton_context=context,
            proton_output_format=output_format,
            proton_mode="periodic_flushing" if periodic else None,
            proton_flush_interval=2 if periodic else 100,
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
        if capture:
            with Worker._get_cudagraph_capture_context(worker):
                with proton.scope("capture_only"):
                    capture_only()
                torch.accelerator.synchronize()
                with torch.cuda.graph(graph), proton.scope("captured_add"):
                    captured_add()
        for run in range(2):
            wrapper.start()
            for _ in range(3):
                wrapper.step()
                with wrapper.annotate_context_manager(f"replay_{run}"):
                    if capture:
                        graph.replay()
                    else:
                        captured_add()
            wrapper.stop()
            paths = sorted(tmp_path.glob(f"*_run{run}*.{output_format}"))
            counts = [2, 1] if periodic else [3]
            assert len(paths) == len(counts)
            for path, count in zip(paths, counts):
                if output_format == "hatchet_msgpack":
                    import msgpack

                    data = msgpack.unpackb(path.read_bytes())
                else:
                    data = json.loads(path.read_text())
                serialized = json.dumps(data)
                if capture:
                    assert "<captured_at>" in serialized
                    assert "captured_add" in serialized
                assert "capture_only" not in serialized
                assert f"replay_{1 - run}" not in serialized
                metrics = kernel_metrics(data)
                assert sum(metric["count"] for metric in metrics) == count
                assert sum(metric["time (ns)"] for metric in metrics) > 0
        torch.testing.assert_close(x, torch.full_like(x, 8 if capture else 7))
    finally:
        wrapper.shutdown()
    assert not list(tmp_path.glob(".proton_cuda_graph_session*"))
