# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, call, patch

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
    def test_normalizes_local_output_directory(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        config = ProfilerConfig(profiler="proton", proton_profiler_dir="profiles")
        assert config.proton_profiler_dir == os.path.join(tmp_path, "profiles")

    @pytest.mark.parametrize(
        ("options", "message"),
        [
            ({"proton_profiler_dir": ""}, "must be set"),
            (
                {"proton_profiler_dir": "s3://bucket/profiles"},
                "local directory",
            ),
            ({"delay_iterations": 1}, "delay_iterations"),
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


class TestProtonProfilerWrapper:
    def test_passes_config_and_global_rank_name_to_proton(self, tmp_path):
        wrapper, proton = make_wrapper(
            tmp_path,
            proton_context="python",
            proton_data="trace",
            proton_backend="cupti",
            proton_mode="pcsampling",
            proton_hook="triton",
            proton_output_format="chrome_trace",
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
        wrapper.stop()
        proton.finalize.assert_called_once_with(session=7, output_format="chrome_trace")
        assert tmp_path.is_dir()

    def test_finalizes_each_profile_with_unique_output_names(self, tmp_path):
        proton = make_proton()
        proton.start.side_effect = [7, 8]
        wrapper, proton = make_wrapper(tmp_path, proton)

        wrapper.start()
        wrapper.set_worker_name("ignored_rank_3")
        wrapper.start()  # Duplicate starts neither create nor rename a session.
        wrapper.stop()
        wrapper.start()
        wrapper.stop()

        assert [c.kwargs["name"] for c in proton.start.call_args_list] == [
            os.path.join(tmp_path, f"proton_rank_3_run{run}") for run in range(2)
        ]
        assert proton.deactivate.call_count == 2
        assert proton.finalize.call_args_list == [call(session=7), call(session=8)]

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

    @pytest.mark.parametrize("fail_cleanup", [False, True])
    def test_shutdown_finalizes_all_sessions_and_resets_state(
        self, tmp_path, fail_cleanup
    ):
        proton = make_proton()
        proton.start.side_effect = [7, 8]
        wrapper, proton = make_wrapper(tmp_path, proton)
        with wrapper.capture_cuda_graphs():
            pass
        wrapper.start()
        if fail_cleanup:
            proton.deactivate.side_effect = RuntimeError("deactivate failed")
            proton.finalize.side_effect = RuntimeError("finalize failed")

        wrapper.shutdown()

        proton.deactivate.assert_any_call(session=8)
        proton.finalize.assert_any_call(session=8)
        proton.finalize.assert_any_call(session=7)
        assert (
            wrapper._session_id,
            wrapper._capture_session_id,
            wrapper._running,
        ) == (None, None, False)

    @pytest.mark.parametrize(
        ("session_id", "start_error", "message"),
        [
            (None, None, "did not create"),
            (7, RuntimeError("CUPTI unavailable"), "CUPTI unavailable"),
        ],
    )
    def test_start_errors_propagate_and_reset_state(
        self, tmp_path, session_id, start_error, message
    ):
        proton = make_proton(session_id)
        if start_error:
            proton.start.side_effect = start_error
        wrapper, _ = make_wrapper(tmp_path, proton)

        with pytest.raises(RuntimeError, match=message):
            wrapper.start()

        assert (wrapper._active, wrapper._running, wrapper._session_id) == (
            False,
            False,
            None,
        )

    @pytest.mark.parametrize(
        ("failing_call", "message"),
        [("finalize", "write failed"), ("deactivate", "deactivate failed")],
    )
    def test_stop_errors_propagate_and_reset_state(
        self, tmp_path, failing_call, message
    ):
        proton = make_proton()
        getattr(proton, failing_call).side_effect = RuntimeError(message)
        wrapper, _ = make_wrapper(tmp_path, proton)
        wrapper.start()

        with pytest.raises(RuntimeError, match=message):
            wrapper.stop()

        proton.finalize.assert_called_once_with(session=7)
        assert (wrapper._active, wrapper._running, wrapper._session_id) == (
            False,
            False,
            None,
        )

    @pytest.mark.parametrize(
        ("environment", "message"),
        [
            ({"HIP_VISIBLE_DEVICES": "0"}, "ROCR_VISIBLE_DEVICES"),
            ({}, "non-empty ROCR_VISIBLE_DEVICES"),
            ({"ROCR_VISIBLE_DEVICES": ""}, "non-empty ROCR_VISIBLE_DEVICES"),
        ],
    )
    def test_rejects_invalid_amd_environment(self, tmp_path, environment, message):
        wrapper, proton = make_wrapper(tmp_path)

        with (
            patch.object(torch.version, "hip", "6.0"),
            patch.dict(os.environ, environment, clear=True),
            pytest.raises(RuntimeError, match=message),
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


@pytest.mark.parametrize("runner", ["disabled", "v1", "v2"])
def test_proton_is_not_initialized_without_cuda_graph_capture(runner):
    worker = MagicMock()
    worker.profiler_config.profiler = "proton"
    worker.vllm_config.compilation_config.cudagraph_mode = (
        CUDAGraphMode.NONE if runner == "disabled" else CUDAGraphMode.FULL
    )
    if runner == "v1":
        worker.use_v2_model_runner = False
        worker.model_runner.cudagraph_dispatcher.get_capture_descs.return_value = []
        worker.model_runner.encoder_cudagraph_manager = None
    elif runner == "v2":
        worker.use_v2_model_runner = True
        worker.model_runner.cudagraph_manager.needs_capture.return_value = False

    context = Worker._get_proton_capture_context(worker)

    worker._get_or_create_profiler.assert_not_called()
    if runner == "v1":
        worker.model_runner._maybe_init_encoder_cudagraph_manager.assert_called_once()
    with context:
        pass


@pytest.mark.parametrize(
    ("worker_cls", "device"), [(CPUWorker, "CPU"), (XPUWorker, "XPU")]
)
def test_non_gpu_worker_rejects_proton(worker_cls, device):
    config = SimpleNamespace(profiler_config=SimpleNamespace(profiler="proton"))

    def initialize_worker(worker, *_args, **_kwargs):
        worker.profiler_config = config.profiler_config

    init_patch = (
        patch.object(Worker, "__init__", autospec=True, side_effect=initialize_worker)
        if worker_cls is XPUWorker
        else nullcontext()
    )
    with init_patch, pytest.raises(ValueError, match=f"not supported by {device}"):
        worker_cls(config, 0, 0, "tcp://localhost:1")
