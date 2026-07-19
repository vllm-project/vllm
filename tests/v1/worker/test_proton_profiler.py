# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import pytest
import torch
from pydantic import ValidationError

from vllm.config import CUDAGraphMode, ProfilerConfig
from vllm.profiler.wrapper import ProtonProfilerWrapper
from vllm.v1.worker.cpu_worker import CPUWorker
from vllm.v1.worker.gpu_worker import Worker
from vllm.v1.worker.xpu_worker import XPUWorker


def make_proton(
    session_id: int | None = 7,
):
    return SimpleNamespace(
        start=Mock(return_value=session_id),
        activate=Mock(),
        deactivate=Mock(),
        finalize=Mock(),
        scope=Mock(return_value=nullcontext()),
    )


def make_wrapper(tmp_path, proton=None, triton_version="3.6.0", **config_overrides):
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


class TestProtonConfig:
    def test_requires_output_directory(self):
        with pytest.raises(ValueError, match="proton_profiler_dir must be set"):
            ProfilerConfig(profiler="proton")

    def test_normalizes_local_output_directory(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        config = ProfilerConfig(profiler="proton", proton_profiler_dir="profiles")
        assert config.proton_profiler_dir == os.path.join(tmp_path, "profiles")

    def test_rejects_remote_output_directory(self):
        with pytest.raises(ValueError, match="must be a local directory"):
            ProfilerConfig(
                profiler="proton", proton_profiler_dir="s3://bucket/profiles"
            )

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("proton_context", "invalid"),
            ("proton_data", "invalid"),
            ("proton_backend", "invalid"),
            ("proton_backend", "instrumentation"),
            ("proton_hook", "invalid"),
            ("proton_output_format", "invalid"),
        ],
    )
    def test_rejects_invalid_typed_options(self, field, value, tmp_path):
        with pytest.raises(ValidationError):
            ProfilerConfig(
                profiler="proton",
                proton_profiler_dir=str(tmp_path),
                **{field: value},
            )

    def test_accepts_all_options(self, tmp_path):
        config = ProfilerConfig(
            profiler="proton",
            proton_profiler_dir=str(tmp_path),
            proton_context="python",
            proton_data="trace",
            proton_backend="rocprofiler",
            proton_mode="pcsampling",
            proton_hook="triton",
            proton_output_format="chrome_trace",
        )
        assert config.proton_context == "python"
        assert config.proton_data == "trace"
        assert config.proton_backend == "rocprofiler"
        assert config.proton_mode == "pcsampling"
        assert config.proton_hook == "triton"
        assert config.proton_output_format == "chrome_trace"

    @pytest.mark.parametrize(
        ("data", "output_format"),
        [
            ("tree", "chrome_trace"),
            ("trace", "hatchet"),
            ("trace", "hatchet_msgpack"),
        ],
    )
    def test_rejects_incompatible_data_and_output_format(
        self, tmp_path, data, output_format
    ):
        with pytest.raises(ValueError, match="output requires proton_data"):
            ProfilerConfig(
                profiler="proton",
                proton_profiler_dir=str(tmp_path),
                proton_data=data,
                proton_output_format=output_format,
            )

    def test_rejects_delayed_start(self, tmp_path):
        with pytest.raises(ValueError, match="delay_iterations is not supported"):
            ProfilerConfig(
                profiler="proton",
                proton_profiler_dir=str(tmp_path),
                delay_iterations=1,
            )


class TestProtonProfilerWrapper:
    def test_passes_config_and_global_rank_name_to_proton(self, tmp_path):
        wrapper, proton = make_wrapper(
            tmp_path,
            proton_context="python",
            proton_data="trace",
            proton_backend="cupti",
            proton_mode="pcsampling",
            proton_hook="triton",
        )

        wrapper.start()

        proton.start.assert_called_once_with(
            name=os.path.join(tmp_path, "proton_rank_3_run0"),
            context="python",
            data="trace",
            backend="cupti",
            mode="pcsampling",
            hook="triton",
        )
        assert tmp_path.is_dir()

    def test_finalizes_each_profile_with_unique_output_names(self, tmp_path):
        proton = make_proton()
        proton.start.side_effect = [7, 8]
        wrapper, proton = make_wrapper(tmp_path, proton)

        wrapper.start()
        wrapper.stop()
        wrapper.start()
        wrapper.stop()

        assert proton.start.call_count == 2
        assert proton.start.call_args_list[0].kwargs["name"] == os.path.join(
            tmp_path, "proton_rank_3_run0"
        )
        assert proton.start.call_args_list[1].kwargs["name"] == os.path.join(
            tmp_path, "proton_rank_3_run1"
        )
        assert proton.deactivate.call_count == 2
        assert proton.finalize.call_count == 2
        proton.finalize.assert_any_call(session=7)
        proton.finalize.assert_any_call(session=8)

    def test_passes_explicit_output_format_to_finalize(self, tmp_path):
        wrapper, proton = make_wrapper(
            tmp_path,
            proton_data="trace",
            proton_output_format="chrome_trace",
        )

        wrapper.start()
        wrapper.stop()

        proton.finalize.assert_called_once_with(session=7, output_format="chrome_trace")

    @pytest.mark.parametrize(
        ("option", "value", "feature"),
        [
            ("proton_output_format", "hatchet_msgpack", "hatchet_msgpack"),
            ("proton_mode", "periodic_flushing", "periodic flushing"),
            ("proton_backend", "rocprofiler", "rocprofiler backend"),
        ],
    )
    def test_newer_features_require_triton_3_8(self, tmp_path, option, value, feature):
        with pytest.raises(RuntimeError, match=feature):
            make_wrapper(tmp_path, **{option: value})

    def test_rejects_output_format_when_finalize_lacks_capability(self, tmp_path):
        proton = make_proton()
        proton.finalize = lambda session=None: None

        with pytest.raises(RuntimeError, match="does not support selecting"):
            make_wrapper(tmp_path, proton, proton_output_format="hatchet")

    def test_duplicate_start_does_not_rename_active_profile(self, tmp_path):
        wrapper, proton = make_wrapper(tmp_path)
        wrapper.start()

        wrapper.set_worker_name("ignored_rank_3")
        wrapper.start()
        wrapper.stop()

        proton.start.assert_called_once()
        assert proton.start.call_args.kwargs["name"] == os.path.join(
            tmp_path, "proton_rank_3_run0"
        )

    def test_cuda_graph_capture_prepares_dormant_session(self, tmp_path):
        proton = make_proton()
        proton.start.side_effect = [7, 8]
        wrapper, proton = make_wrapper(tmp_path, proton, proton_mode="custom")

        with wrapper.capture_cuda_graphs():
            assert wrapper._capture_session_id == 7

        proton.start.assert_called_once()
        assert proton.start.call_args.kwargs["mode"] == "custom"
        proton.deactivate.assert_called_once_with(session=7)

        wrapper.start()
        assert wrapper._session_id == 8
        assert proton.start.call_count == 2
        proton.activate.assert_not_called()

    def test_pcsampling_rejects_cuda_graph_capture(self, tmp_path):
        wrapper, proton = make_wrapper(tmp_path, proton_mode="pcsampling")

        with (
            pytest.raises(ValueError, match="disable CUDA graphs"),
            wrapper.capture_cuda_graphs(),
        ):
            pass

        proton.start.assert_not_called()

    def test_cuda_graph_context_deactivates_after_capture_error(self, tmp_path):
        wrapper, proton = make_wrapper(tmp_path)

        with (
            pytest.raises(RuntimeError, match="capture failed"),
            wrapper.capture_cuda_graphs(),
        ):
            raise RuntimeError("capture failed")

        proton.deactivate.assert_called_once_with(session=7)

    def test_shutdown_stops_running_session_before_finalize(self, tmp_path):
        proton = make_proton()
        proton.start.side_effect = [7, 8]
        wrapper, proton = make_wrapper(tmp_path, proton)
        with wrapper.capture_cuda_graphs():
            pass
        wrapper.start()

        wrapper.shutdown()

        proton.deactivate.assert_any_call(session=8)
        proton.finalize.assert_any_call(session=8)
        proton.finalize.assert_any_call(session=7)
        assert wrapper._session_id is None
        assert wrapper._capture_session_id is None
        assert wrapper._running is False

    def test_shutdown_contains_proton_errors_and_finishes_cleanup(self, tmp_path):
        proton = make_proton()
        proton.start.side_effect = [7, 8]
        wrapper, proton = make_wrapper(tmp_path, proton)
        with wrapper.capture_cuda_graphs():
            pass
        wrapper.start()
        proton.deactivate.side_effect = RuntimeError("deactivate failed")
        proton.finalize.side_effect = RuntimeError("finalize failed")

        wrapper.shutdown()

        proton.finalize.assert_any_call(session=8)
        proton.finalize.assert_any_call(session=7)
        assert wrapper._session_id is None
        assert wrapper._capture_session_id is None
        assert wrapper._running is False

    def test_disabled_proton_session_is_reported_as_start_failure(self, tmp_path):
        wrapper, _ = make_wrapper(tmp_path, make_proton(session_id=None))

        with pytest.raises(RuntimeError, match="did not create"):
            wrapper.start()

        assert wrapper._running is False
        assert wrapper._active is False
        assert wrapper._session_id is None

    def test_start_error_propagates_and_resets_state(self, tmp_path):
        proton = make_proton()
        proton.start.side_effect = RuntimeError("CUPTI unavailable")
        wrapper, _ = make_wrapper(tmp_path, proton)

        with pytest.raises(RuntimeError, match="CUPTI unavailable"):
            wrapper.start()

        assert wrapper._active is False
        assert wrapper._running is False

    def test_finalize_error_propagates_and_resets_state(self, tmp_path):
        proton = make_proton()
        proton.finalize.side_effect = RuntimeError("write failed")
        wrapper, _ = make_wrapper(tmp_path, proton)
        wrapper.start()

        with pytest.raises(RuntimeError, match="write failed"):
            wrapper.stop()

        assert wrapper._active is False
        assert wrapper._running is False
        assert wrapper._session_id is None

    def test_finalize_runs_when_deactivate_fails(self, tmp_path):
        proton = make_proton()
        proton.deactivate.side_effect = RuntimeError("deactivate failed")
        wrapper, _ = make_wrapper(tmp_path, proton)
        wrapper.start()

        with pytest.raises(RuntimeError, match="deactivate failed"):
            wrapper.stop()

        proton.finalize.assert_called_once_with(session=7)
        assert wrapper._running is False
        assert wrapper._session_id is None

    def test_rejects_amd_visibility_environment(self, tmp_path):
        wrapper, proton = make_wrapper(tmp_path)

        with (
            patch.object(torch.version, "hip", "6.0"),
            patch.dict(os.environ, {"HIP_VISIBLE_DEVICES": "0"}, clear=True),
            pytest.raises(RuntimeError, match="ROCR_VISIBLE_DEVICES"),
        ):
            wrapper.start()

        proton.start.assert_not_called()
        assert wrapper._active is False

    @pytest.mark.parametrize("environment", [{}, {"ROCR_VISIBLE_DEVICES": ""}])
    def test_requires_nonempty_rocr_visible_devices(self, tmp_path, environment):
        wrapper, proton = make_wrapper(tmp_path)

        with (
            patch.object(torch.version, "hip", "6.0"),
            patch.dict(os.environ, environment, clear=True),
            pytest.raises(RuntimeError, match="non-empty ROCR_VISIBLE_DEVICES"),
        ):
            wrapper.start()

        proton.start.assert_not_called()
        assert wrapper._active is False

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
        wrapper, proton = make_wrapper(tmp_path)

        metrics = {"num_tokens": 32, "num_requests": 2}
        context = wrapper.annotate_context_manager("decode", metrics=metrics)

        proton.scope.assert_called_once_with("decode", metrics=metrics)
        assert context is not None


def test_proton_is_not_initialized_when_cuda_graphs_are_disabled():
    worker = MagicMock()
    worker.vllm_config.compilation_config.cudagraph_mode = CUDAGraphMode.NONE
    worker.profiler_config.profiler = "proton"

    context = Worker._get_proton_capture_context(worker)

    worker._get_or_create_profiler.assert_not_called()
    with context:
        pass


def test_proton_is_not_initialized_when_v2_runner_has_no_graphs_to_capture():
    worker = MagicMock()
    worker.use_v2_model_runner = True
    worker.vllm_config.compilation_config.cudagraph_mode = CUDAGraphMode.FULL
    worker.profiler_config.profiler = "proton"
    worker.model_runner.cudagraph_manager.needs_capture.return_value = False

    context = Worker._get_proton_capture_context(worker)

    worker._get_or_create_profiler.assert_not_called()
    with context:
        pass


def test_proton_is_not_initialized_when_v1_runner_has_no_graphs_to_capture():
    worker = MagicMock()
    worker.use_v2_model_runner = False
    worker.vllm_config.compilation_config.cudagraph_mode = CUDAGraphMode.FULL
    worker.vllm_config.compilation_config.cudagraph_mm_encoder = False
    worker.profiler_config.profiler = "proton"
    worker.model_runner.cudagraph_dispatcher.get_capture_descs.return_value = []

    context = Worker._get_proton_capture_context(worker)

    worker._get_or_create_profiler.assert_not_called()
    with context:
        pass


def test_cpu_worker_rejects_proton():
    config = SimpleNamespace(
        profiler_config=SimpleNamespace(profiler="proton"),
    )

    with pytest.raises(ValueError, match="not supported by CPU workers"):
        CPUWorker(config, 0, 0, "tcp://localhost:1")


def test_xpu_worker_rejects_proton():
    config = SimpleNamespace(
        profiler_config=SimpleNamespace(profiler="proton"),
    )

    def initialize_worker(worker, *_args, **_kwargs):
        worker.profiler_config = config.profiler_config

    with (
        patch.object(Worker, "__init__", autospec=True, side_effect=initialize_worker),
        pytest.raises(ValueError, match="not supported by XPU workers"),
    ):
        XPUWorker(config, 0, 0, "tcp://localhost:1")
