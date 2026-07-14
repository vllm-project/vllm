# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
from pydantic import ValidationError

from vllm.config import ProfilerConfig
from vllm.profiler.wrapper import ProtonProfilerWrapper


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


def make_wrapper(tmp_path, proton=None, **config_overrides):
    proton = proton or make_proton()
    config = ProfilerConfig(
        profiler="proton",
        proton_profiler_dir=str(tmp_path),
        **config_overrides,
    )
    with patch("vllm.profiler.wrapper.importlib.import_module", return_value=proton):
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
            ("proton_hook", "invalid"),
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
            proton_backend="cupti",
            proton_mode="pcsampling",
            proton_hook="triton",
        )
        assert config.proton_context == "python"
        assert config.proton_data == "trace"
        assert config.proton_backend == "cupti"
        assert config.proton_mode == "pcsampling"
        assert config.proton_hook == "triton"


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
            name=os.path.join(tmp_path, "proton_rank_3"),
            context="python",
            data="trace",
            backend="cupti",
            mode="pcsampling",
            hook="triton",
        )
        assert tmp_path.is_dir()

    def test_finalizes_each_profile_and_uses_next_prefix(self, tmp_path):
        proton = make_proton()
        proton.start.side_effect = [7, 8]
        wrapper, proton = make_wrapper(tmp_path, proton)

        wrapper.start()
        wrapper.stop()
        wrapper.set_worker_name("request_rank_3")
        wrapper.start()
        wrapper.stop()

        assert proton.start.call_count == 2
        assert proton.start.call_args_list[1].kwargs["name"] == os.path.join(
            tmp_path, "proton_request_rank_3"
        )
        assert proton.deactivate.call_count == 2
        assert proton.finalize.call_count == 2
        proton.finalize.assert_any_call(session=7)
        proton.finalize.assert_any_call(session=8)

    def test_duplicate_start_does_not_rename_active_profile(self, tmp_path):
        wrapper, proton = make_wrapper(tmp_path)
        wrapper.start()

        wrapper.set_worker_name("ignored_rank_3")
        wrapper.start()
        wrapper.stop()

        proton.start.assert_called_once()
        assert proton.start.call_args.kwargs["name"] == os.path.join(
            tmp_path, "proton_rank_3"
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
            pytest.raises(ValueError, match="enable eager execution"),
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

    def test_disabled_proton_session_is_reported_as_start_failure(self, tmp_path):
        wrapper, _ = make_wrapper(tmp_path, make_proton(session_id=None))

        wrapper.start()

        assert wrapper._running is False
        assert wrapper._session_id is None

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

        context = wrapper.annotate_context_manager("decode")

        proton.scope.assert_called_once_with("decode")
        assert context is not None
