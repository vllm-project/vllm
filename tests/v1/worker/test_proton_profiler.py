# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, call, patch
from uuid import UUID

import pytest
import torch
from pydantic import ValidationError

from vllm.config import ProfilerConfig
from vllm.profiler.wrapper import ProtonProfilerWrapper
from vllm.v1.worker.gpu_worker import Worker


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
            name=os.path.join(tmp_path, f"proton_rank_3_{wrapper._instance_id}_run0"),
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
        wrapper.start()  # Duplicate starts do not create another session.
        wrapper.stop()
        wrapper.start()
        wrapper.stop()

        assert [c.kwargs["name"] for c in proton.start.call_args_list] == [
            os.path.join(
                tmp_path,
                f"proton_rank_3_{wrapper._instance_id}_run{run}",
            )
            for run in range(2)
        ]
        assert proton.deactivate.call_count == 2
        assert proton.finalize.call_args_list == [call(session=7), call(session=8)]

    def test_output_names_are_unique_across_worker_restarts(self, tmp_path):
        with patch(
            "vllm.profiler.wrapper.uuid4",
            side_effect=[UUID(int=1), UUID(int=2)],
        ):
            first, first_proton = make_wrapper(tmp_path)
            second, second_proton = make_wrapper(tmp_path)

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

    @pytest.mark.parametrize("fail_cleanup", [False, True])
    def test_shutdown_finalizes_active_session(self, tmp_path, fail_cleanup):
        proton = make_proton()
        wrapper, proton = make_wrapper(tmp_path, proton)
        wrapper.start()
        if fail_cleanup:
            proton.deactivate.side_effect = RuntimeError("deactivate failed")
            proton.finalize.side_effect = RuntimeError("finalize failed")

        wrapper.shutdown()

        proton.deactivate.assert_called_once_with(session=7)
        proton.finalize.assert_called_once_with(session=7)
        assert (wrapper._session_id, wrapper._running) == (None, False)

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

    def test_automatic_stop_errors_do_not_fail_inference(self, tmp_path):
        proton = make_proton()
        proton.finalize.side_effect = RuntimeError("write failed")
        wrapper, _ = make_wrapper(tmp_path, proton, max_iterations=1)
        wrapper.start()

        wrapper.step()
        wrapper.step()

        assert (wrapper._active, wrapper._running, wrapper._session_id) == (
            True,
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
        wrapper.start()

        context = wrapper.annotate_context_manager("decode")

        proton.scope.assert_called_once_with("decode")
        assert context is not None


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
    worker.profiler.start.assert_called_once()
