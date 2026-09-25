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

import vllm.v1.worker.gpu_model_runner as gpu_model_runner_module
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
    create_worker_profiler,
    validate_worker_profiler_config,
)
from vllm.v1.core.sched.output import CachedRequestData
from vllm.v1.worker.gpu_model_runner import GPUModelRunner
from vllm.v1.worker.gpu_worker import Worker
from vllm.v1.worker.xpu_worker import XPUWorker


def _make_worker(profiler, *, synchronize_iterations=False, dp_size=1) -> Worker:
    worker = object.__new__(Worker)
    worker.rank = 0
    worker.profiler = profiler
    worker.profiler_config = ProfilerConfig(
        profiler="cuda",
        synchronize_iterations_across_dp=synchronize_iterations,
    )
    worker.vllm_config = SimpleNamespace(profiler_config=worker.profiler_config)
    worker.parallel_config = SimpleNamespace(data_parallel_size=dp_size)
    worker._dp_profiler_requested = False
    return worker


class ConcreteWorkerProfiler(WorkerProfiler):
    """A basic implementation of a worker profiler for testing purposes."""

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


def test_torch_profiler_rebuilds_one_shot_profiler_each_round(tmp_path):
    config = ProfilerConfig(
        profiler="torch",
        torch_profiler_dir=str(tmp_path),
        torch_profiler_activities=["CUDA"],
        torch_profiler_dump_cuda_time_total=False,
        warmup_iterations=2,
    )
    profilers = [MagicMock(), MagicMock()]
    for profiler in profilers:
        profiler.profiler = MagicMock()

    with patch(
        "vllm.profiler.wrapper.torch.profiler.profile", side_effect=profilers
    ) as profile:
        wrapper = TorchProfilerWrapper(
            config, worker_name="worker", local_rank=0, activities=["CUDA"]
        )
        profile.assert_not_called()

        wrapper.start()
        assert not wrapper.should_annotate
        assert not wrapper._profiler_step()
        wrapper.stop()
        wrapper.start()
        assert not wrapper._profiler_step()
        wrapper.stop()

    assert profile.call_count == 2
    assert profile.call_args.kwargs["activities"] == [
        torch.profiler.ProfilerActivity.CUDA
    ]
    for profiler in profilers:
        profiler.start.assert_called_once_with()
        profiler.step.assert_called_once_with()
        profiler.stop.assert_called_once_with()


def test_torch_profiler_records_each_profile_round(tmp_path):
    traces: list[torch.profiler.profile] = []
    wrapper = TorchProfilerWrapper(
        ProfilerConfig(
            profiler="torch",
            torch_profiler_dir=str(tmp_path),
            torch_profiler_dump_cuda_time_total=False,
        ),
        worker_name="worker",
        local_rank=1,
        activities=["CPU"],
        on_trace_ready=traces.append,
    )

    for run in range(2):
        wrapper.start()
        with torch.profiler.record_function(f"run_{run}"):
            pass
        wrapper.stop()

    assert len(traces) == 2
    for run, trace in enumerate(traces):
        assert {
            event.name for event in trace.events() if event.name.startswith("run_")
        } == {f"run_{run}"}


@pytest.mark.parametrize(
    "activities", [["CPU", "CUDA"], ["CUDA"], ["CPU", "XPU"], ["XPU"]]
)
@pytest.mark.parametrize("dump_device_time", [True, False])
def test_torch_profiler_device_summary(tmp_path, capsys, activities, dump_device_time):
    """Device summaries honor the dump option on both CUDA and XPU."""
    config = ProfilerConfig(
        profiler="torch",
        torch_profiler_dir=str(tmp_path),
        torch_profiler_dump_cuda_time_total=dump_device_time,
    )
    with patch("vllm.profiler.wrapper.torch.profiler.profile") as profile:
        profile.return_value.key_averages.return_value.table.return_value = (
            "device times"
        )
        wrapper = TorchProfilerWrapper(
            config, worker_name="worker", local_rank=0, activities=activities
        )
        wrapper.start()
        wrapper.stop()

    summary = tmp_path / "profiler_out_0.txt"
    assert summary.exists() == dump_device_time
    assert ("device times" in capsys.readouterr().out) == dump_device_time
    if dump_device_time:
        assert summary.read_text() == "device times\n"


@pytest.mark.parametrize(
    "activities",
    [[], ["CPU", "CPU"], ["INVALID"]],
)
def test_torch_profiler_activities_reject_invalid_values(activities):
    with pytest.raises(ValueError, match="torch_profiler_activities"):
        ProfilerConfig(
            profiler="torch",
            torch_profiler_dir="/tmp/mock",
            torch_profiler_activities=activities,
        )


def test_torch_profiler_activities_require_torch_profiler():
    with pytest.raises(ValueError, match="only applicable"):
        ProfilerConfig(torch_profiler_activities=["CPU"])


@pytest.mark.parametrize(
    ("device_type", "activities"),
    [
        ("cuda", ["XPU"]),
        ("xpu", ["CUDA"]),
        ("cpu", ["CUDA"]),
        ("cpu", ["XPU"]),
    ],
)
def test_worker_rejects_unsupported_activities_at_startup(device_type, activities):
    config = ProfilerConfig(
        profiler="torch",
        torch_profiler_dir="/tmp/mock",
        torch_profiler_activities=activities,
    )

    with (
        patch.object(current_platform, "device_type", device_type),
        pytest.raises(ValueError, match="Unsupported torch profiler activities"),
    ):
        validate_worker_profiler_config(config)


@pytest.mark.parametrize("device_type", ["cpu", "xpu"])
def test_worker_rejects_cuda_profiler_on_other_devices(device_type):
    with (
        patch.object(current_platform, "device_type", device_type),
        pytest.raises(ValueError, match="Unsupported profiler type"),
    ):
        validate_worker_profiler_config(ProfilerConfig(profiler="cuda"))


@pytest.mark.parametrize(
    ("device_type", "activities", "expected"),
    [
        ("cuda", None, ("CPU", "CUDA")),
        ("xpu", None, ("CPU", "XPU")),
        ("cpu", None, ("CPU",)),
        ("cuda", ["CUDA"], ("CUDA",)),
        ("xpu", ["XPU"], ("XPU",)),
        ("cuda", ["CPU"], ("CPU",)),
        ("xpu", ["CPU"], ("CPU",)),
        ("cpu", ["CPU"], ("CPU",)),
    ],
)
def test_worker_creates_platform_torch_profiler(device_type, activities, expected):
    config = ProfilerConfig(
        profiler="torch",
        torch_profiler_dir="/tmp/mock",
        torch_profiler_activities=activities,
    )

    with (
        patch.object(current_platform, "device_type", device_type),
        patch("vllm.profiler.wrapper.TorchProfilerWrapper") as wrapper,
    ):
        validate_worker_profiler_config(config)
        profiler = create_worker_profiler(
            config,
            worker_name="rank0",
            local_rank=0,
        )

    assert profiler is wrapper.return_value
    wrapper.assert_called_once_with(
        config,
        worker_name="rank0",
        local_rank=0,
        activities=expected,
    )


@pytest.mark.parametrize("worker_type", [Worker, XPUWorker])
def test_worker_reuses_torch_wrapper_across_profile_rounds(worker_type):
    worker = object.__new__(worker_type)
    worker.rank = 0
    worker.local_rank = 0
    worker.profiler = None
    worker.profiler_config = ProfilerConfig(
        profiler="torch", torch_profiler_dir="/tmp/mock"
    )

    with (
        patch("vllm.distributed.utils.get_worker_rank_suffix", return_value="rank0"),
        patch("vllm.profiler.wrapper.TorchProfilerWrapper") as wrapper,
    ):
        worker.profile()
        worker.profile(is_start=False)
        worker.profile()

    assert worker.profiler is wrapper.return_value
    wrapper.assert_called_once()
    assert wrapper.return_value.start.call_count == 2
    wrapper.return_value.stop.assert_called_once_with()


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


def test_synchronized_iterations_disabled_by_default():
    assert ProfilerConfig().synchronize_iterations_across_dp is False


def test_synchronized_iterations_accept_cuda_profiler():
    config = ProfilerConfig(
        profiler="cuda",
        synchronize_iterations_across_dp=True,
    )

    assert config.synchronize_iterations_across_dp is True


def test_synchronized_iterations_require_profiler():
    with pytest.raises(ValueError, match="requires profiler to be set"):
        ProfilerConfig(synchronize_iterations_across_dp=True)


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
        worker = _make_worker(MagicMock())
        worker.vllm_config.profiler_config.detailed_trace_annotation = detailed

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

    def test_skips_annotation_work_when_profiler_does_not_annotate(self):
        worker = _make_worker(MagicMock())
        worker.profiler.should_annotate = False
        worker.profiler.is_running = False

        with patch(
            "vllm.v1.worker.gpu_worker.compute_iteration_details"
        ) as compute_iteration_details:
            context = Worker.annotate_profile(worker, scheduler_output=None)

        worker.profiler.step.assert_called_once_with()
        compute_iteration_details.assert_not_called()
        worker.profiler.annotate_context_manager.assert_not_called()
        assert isinstance(context, nullcontext)

    def test_synchronized_mode_suppresses_rank_local_step(self):
        worker = _make_worker(MagicMock(), synchronize_iterations=True, dp_size=2)
        worker.profiler.is_running = False
        worker.profiler.should_annotate = False

        Worker.annotate_profile(worker, scheduler_output=None)

        worker.profiler.step.assert_not_called()


class TestDPSynchronizedProfiler:
    def _worker(self, profiler):
        worker = _make_worker(profiler, synchronize_iterations=True, dp_size=2)
        worker._dp_profiler_requested = True
        return worker

    def test_waits_until_every_rank_is_ready(self, default_profiler_config):
        profiler = ConcreteWorkerProfiler(default_profiler_config)
        worker = self._worker(profiler)

        Worker._advance_dp_synchronized_profiler(worker, False)

        assert profiler.start_call_count == 0
        assert profiler._active_iteration_count == 0

        Worker._advance_dp_synchronized_profiler(worker, True)

        assert profiler.start_call_count == 1
        assert profiler._active_iteration_count == 1

    def test_profile_request_arms_without_starting_locally(self):
        worker = _make_worker(MagicMock(), synchronize_iterations=True, dp_size=2)

        with patch(
            "vllm.distributed.utils.get_worker_rank_suffix",
            return_value="rank0",
        ):
            Worker.profile(worker)

        assert worker._dp_profiler_requested is True
        worker.profiler.start.assert_not_called()

    @pytest.mark.parametrize("synchronize_iterations", [False, True])
    def test_single_rank_keeps_rank_local_profiling(self, synchronize_iterations):
        worker = _make_worker(
            MagicMock(), synchronize_iterations=synchronize_iterations
        )

        with (
            patch(
                "vllm.distributed.utils.get_worker_rank_suffix", return_value="rank0"
            ),
            patch("vllm.v1.worker.gpu_worker.logger") as logger,
        ):
            Worker.profile(worker)

        worker.profiler.start.assert_called_once_with()
        assert worker._dp_profiler_requested is False
        if synchronize_iterations:
            logger.info_once.assert_called_once()
            assert "data_parallel_size=1" in logger.info_once.call_args.args[0]
        else:
            logger.info_once.assert_not_called()

    def test_delayed_start_counts_shared_boundaries(self, default_profiler_config):
        default_profiler_config.delay_iterations = 2
        profiler = ConcreteWorkerProfiler(default_profiler_config)
        worker = self._worker(profiler)

        Worker._advance_dp_synchronized_profiler(worker, True)
        assert not profiler.is_running

        Worker._advance_dp_synchronized_profiler(worker, True)
        assert profiler.is_running
        assert profiler.start_call_count == 1

    def test_auto_stop_resets_synchronized_session(self, default_profiler_config):
        default_profiler_config.max_iterations = 1
        profiler = ConcreteWorkerProfiler(default_profiler_config)
        worker = self._worker(profiler)

        Worker._advance_dp_synchronized_profiler(worker, True)
        Worker._advance_dp_synchronized_profiler(worker, True)

        assert not profiler.is_armed
        assert profiler.stop_call_count == 1
        assert worker._dp_profiler_requested is False

    def test_restart_after_auto_stop(self, default_profiler_config):
        default_profiler_config.max_iterations = 1
        profiler = ConcreteWorkerProfiler(default_profiler_config)
        worker = self._worker(profiler)

        Worker._advance_dp_synchronized_profiler(worker, True)
        Worker._advance_dp_synchronized_profiler(worker, True)
        worker._dp_profiler_requested = True
        Worker._advance_dp_synchronized_profiler(worker, True)

        assert profiler.start_call_count == 2
        assert profiler.is_armed

    def test_peer_stop_ends_local_capture(self, default_profiler_config):
        profiler = ConcreteWorkerProfiler(default_profiler_config)
        worker = self._worker(profiler)
        Worker._advance_dp_synchronized_profiler(worker, True)

        Worker._advance_dp_synchronized_profiler(worker, False)

        assert profiler.stop_call_count == 1
        assert worker._dp_profiler_requested is False
        assert not profiler.is_armed


class _BoundaryObserved(Exception):
    pass


class _StopAfterBoundaryBatch:
    @property
    def num_tokens(self):
        raise _BoundaryObserved


def _v1_boundary_result(profiler_ready=True):
    return (
        CUDAGraphMode.NONE,
        _StopAfterBoundaryBatch(),
        False,
        None,
        None,
        profiler_ready,
    )


@pytest.mark.parametrize("profiler_ready", [True, False, None])
def test_v1_execute_model_advances_profiler_only_after_dp_agreement(profiler_ready):
    runner = GPUModelRunner.__new__(GPUModelRunner)
    runner.execute_model_state = None
    runner.speculative_config = None
    runner.synchronize_input_prep = nullcontext
    runner._update_states = Mock()
    runner.cache_config = SimpleNamespace(kv_sharing_fast_prefill=False)
    runner.num_prompt_logprobs = 0
    runner.input_batch = SimpleNamespace(num_reqs=1, req_ids=["request"])
    runner._prepare_inputs = Mock(return_value=(None, None, None))
    runner.cascade_attn_enabled = False
    runner.parallel_config = SimpleNamespace(use_ubatching=False)
    runner._allow_microbatching = Mock(return_value=False)
    runner._determine_batch_execution_and_padding = Mock(
        return_value=_v1_boundary_result(profiler_ready)
    )
    runner.dp_profiler_advance = Mock()
    scheduler_output = MagicMock(
        total_num_scheduled_tokens=1,
        num_scheduled_tokens={"request": 1},
        scheduled_encoder_inputs={},
    )

    with (
        patch.object(
            gpu_model_runner_module,
            "has_kv_transfer_group",
            return_value=False,
        ),
        pytest.raises(_BoundaryObserved),
    ):
        GPUModelRunner.execute_model(runner, scheduler_output)

    if profiler_ready is None:
        runner.dp_profiler_advance.assert_not_called()
    else:
        runner.dp_profiler_advance.assert_called_once_with(profiler_ready)


@pytest.mark.parametrize("profiler_ready", [True, False, None])
def test_v1_dummy_run_advances_profiler_only_after_dp_agreement(profiler_ready):
    runner = GPUModelRunner.__new__(GPUModelRunner)
    runner.vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(multimodal_config=None)
    )
    runner.max_num_tokens = 4
    runner.max_num_reqs = 4
    runner.scheduler_config = SimpleNamespace(max_num_seqs=4)
    runner.uniform_decode_query_len = 1
    runner._determine_batch_execution_and_padding = Mock(
        return_value=_v1_boundary_result(profiler_ready)
    )
    runner.dp_profiler_advance = Mock()

    with pytest.raises(_BoundaryObserved):
        GPUModelRunner._dummy_run(runner, 1, skip_eplb=True)

    if profiler_ready is None:
        runner.dp_profiler_advance.assert_not_called()
    else:
        runner.dp_profiler_advance.assert_called_once_with(profiler_ready)


@pytest.mark.parametrize("dummy_run", [False, True])
@pytest.mark.parametrize("profiler_ready", [True, False, None])
def test_v2_forward_advances_profiler_only_after_dp_agreement(
    dummy_run, profiler_ready
):
    from vllm.v1.worker.gpu import model_runner as v2

    runner = MagicMock()
    runner.gather_batch_req_state.return_value = (None, None)
    runner.lora_config = None
    runner.aux_output_connector = None
    runner.is_encoder_decoder = False
    scheduler_output = MagicMock(
        total_num_scheduled_tokens=1, num_scheduled_tokens={"request": 1}
    )
    with (
        patch.object(
            v2,
            "dispatch_cg_and_sync_dp",
            return_value=(
                _StopAfterBoundaryBatch(),
                SimpleNamespace(profiler_ready=profiler_ready),
            ),
        ),
        pytest.raises(_BoundaryObserved),
    ):
        v2.GPUModelRunner.execute_model(runner, scheduler_output, dummy_run=dummy_run)
    if profiler_ready is None:
        runner.dp_profiler_advance.assert_not_called()
    else:
        runner.dp_profiler_advance.assert_called_once_with(profiler_ready)


@pytest.mark.parametrize("runner_version", [1, 2])
def test_skipped_dp_coordination_does_not_claim_profiler_readiness(runner_version):
    from vllm.v1.worker import dp_utils
    from vllm.v1.worker.gpu import dp_utils as v2_dp
    from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor

    with (
        dp_utils.skip_dp_coordination(),
        patch.object(torch.distributed, "all_reduce") as all_reduce,
        patch.object(
            v2_dp, "get_dp_group", return_value=SimpleNamespace(cpu_group=None)
        ),
    ):
        if runner_version == 1:
            *_, readiness = dp_utils.coordinate_batch_across_dp(
                4, False, SimpleNamespace(data_parallel_size=2), profiler_ready=True
            )
        else:
            _, sync = v2_dp.sync_cudagraph_and_dp_padding(
                None,
                BatchExecutionDescriptor(CUDAGraphMode.NONE, 4, 1),
                num_tokens=4,
                num_reqs=1,
                uniform_token_count=None,
                dp_size=2,
                dp_rank=0,
                profiler_ready=True,
            )
            readiness = sync.profiler_ready
    assert readiness is None
    all_reduce.assert_not_called()


@pytest.mark.parametrize(
    ("local_ready", "readiness", "expected"),
    [
        (None, [0, 0], None),
        (False, [0, 1], False),
        (True, [1, 0], False),
        (True, [1, 1], True),
    ],
)
def test_legacy_dp_sync_carries_profiler_readiness(local_ready, readiness, expected):
    from vllm.v1.worker import dp_utils

    reduced = torch.tensor(
        [
            [64, 64],
            [64, 64],
            [0, 0],
            [0, 0],
            readiness,
        ],
        dtype=torch.int32,
    )
    parallel_config = SimpleNamespace(
        disable_nccl_for_dp_synchronization=True,
        num_ubatches=1,
        data_parallel_size=2,
        data_parallel_rank=0,
    )

    def reduce(tensor, group):
        # Disabled, unarmed and armed profiling must use the same wire shape.
        assert tensor.shape == reduced.shape
        assert tensor[4, 0].item() == int(bool(local_ready))
        tensor.copy_(reduced)

    with (
        patch.object(dp_utils, "_get_device_and_group", return_value=("cpu", None)),
        patch.object(torch.distributed, "all_reduce", side_effect=reduce) as all_reduce,
    ):
        *_, profiler_ready = dp_utils._synchronize_dp_ranks(
            num_tokens_unpadded=64,
            num_tokens_padded=64,
            should_attempt_ubatching=False,
            cudagraph_mode=0,
            parallel_config=parallel_config,
            profiler_ready=local_ready,
        )

    all_reduce.assert_called_once()
    assert profiler_ready is expected


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

# Proton subscribes to CUPTI itself, and a process gets one CUPTI subscriber.
# CI's kernel-launch recorder is another, injected through CUDA_INJECTION64_PATH
# on recording runs, and there Proton's cuptiSubscribe fails with error 39.
_requires_no_injected_cupti_tool = pytest.mark.skipif(
    bool(os.environ.get("CUDA_INJECTION64_PATH")),
    reason="Another CUPTI tool is injected (CUDA_INJECTION64_PATH); "
    "Proton has to be the only one.",
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
    worker = object.__new__(Worker)
    worker.rank = 1
    worker.local_rank = 1
    worker.profiler = None
    worker.profiler_config = MagicMock(profiler="proton")
    worker.profiler_config.synchronize_iterations_across_dp = False
    worker.parallel_config = SimpleNamespace(data_parallel_size=1)

    with (
        patch(
            "vllm.distributed.utils.get_worker_rank_suffix",
            return_value="rank1",
        ),
        patch("vllm.profiler.wrapper.ProtonProfilerWrapper") as wrapper,
    ):
        Worker.profile(worker)

    wrapper.assert_called_once_with(worker.profiler_config, worker_name="rank1")
    worker.profiler.start.assert_called_once_with()


@_requires_cuda_for_proton
def test_gpu_worker_recreates_proton_profiler_for_each_run():
    worker = object.__new__(Worker)
    worker.rank = 1
    worker.local_rank = 1
    worker.profiler = None
    worker.profiler_config = MagicMock(profiler="proton")
    worker.profiler_config.synchronize_iterations_across_dp = False
    worker.parallel_config = SimpleNamespace(data_parallel_size=1)

    with (
        patch(
            "vllm.distributed.utils.get_worker_rank_suffix",
            return_value="rank1",
        ),
        patch("vllm.profiler.wrapper.ProtonProfilerWrapper") as wrapper,
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
@_requires_no_injected_cupti_tool
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
