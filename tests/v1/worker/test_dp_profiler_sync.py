# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from unittest.mock import MagicMock, Mock, patch

import pytest

from vllm.config import ProfilerConfig
from vllm.v1.engine.core import DPEngineCoreProc, EngineCoreOutputs
from vllm.v1.worker.dp_profiler_sync import DPProfilerSync
from vllm.v1.worker.gpu_worker import Worker


@pytest.mark.skip_global_cleanup
class TestDPProfilerSync:
    """Profiler control piggybacks on the per-step DP coordination."""

    def test_remote_start_receives_settings(self):
        source = DPProfilerSync()
        remote = DPProfilerSync()
        source.request_start("capture", 3, 8)
        source_group = Mock(rank_in_group=0)
        source_group.broadcast_object.return_value = ("capture", 3, 8)
        remote_group = Mock(rank_in_group=1)
        remote_group.broadcast_object.return_value = ("capture", 3, 8)

        source.observe([1, 0], [0, 0], source_group)
        remote.observe([1, 0], [0, 0], remote_group)

        source_group.broadcast_object.assert_called_once_with(("capture", 3, 8), src=0)
        remote_group.broadcast_object.assert_called_once_with(None, src=0)
        assert source.consume_start() == ("capture", 3, 8)
        assert remote.consume_start() == ("capture", 3, 8)
        assert source.consume_start() is None
        assert remote.consume_start() is None
        assert not source.pending_start

    def test_stop_wins_over_start_on_all_ranks(self):
        sync = DPProfilerSync()
        sync.request_start("capture", 1, 2)
        sync.observe([1, 0], [0, 1], Mock(rank_in_group=0))

        assert sync.consume_stop()
        assert not sync.consume_stop()
        assert sync.consume_start() is None

    def test_stop_cancels_local_start(self):
        sync = DPProfilerSync()
        sync.request_start("capture", 1, 2)
        sync.request_stop()

        assert not sync.pending_start
        assert sync.pending_stop
        sync.observe([0, 0], [1, 0], Mock(rank_in_group=0))
        assert sync.consume_stop()
        assert not sync.pending_stop

    def test_latch_survives_later_empty_observation(self):
        sync = DPProfilerSync()
        group = Mock(rank_in_group=0)
        group.broadcast_object.return_value = (None, None, None)
        sync.request_start()
        sync.observe([1, 0], [0, 0], group)
        sync.observe([0, 0], [0, 0], group)
        assert sync.consume_start() == (None, None, None)


@pytest.mark.skip_global_cleanup
class TestWorkerSyncedProfileStart:
    """Worker applies the profiler transition after the DP step."""

    def _worker(self, dp_profiler_sync, profiler):
        worker = MagicMock()
        worker._dp_profiler_sync = dp_profiler_sync
        worker.profiler = profiler
        worker._create_profiler.return_value = profiler
        return worker

    def test_starts_on_consensus(self):
        sync = DPProfilerSync()
        sync.request_start("capture", 2, 9)
        group = Mock(rank_in_group=0)
        group.broadcast_object.return_value = ("capture", 2, 9)
        sync.observe([1, 0], [0, 0], group)
        profiler = MagicMock()
        worker = self._worker(sync, profiler)

        Worker._maybe_apply_synced_profile(worker)
        worker._create_profiler.assert_called_once_with("capture")
        profiler.start.assert_called_once_with(delay_iterations=2, max_iterations=9)

        # Latch consumed: a later step does not start again.
        Worker._maybe_apply_synced_profile(worker)
        profiler.start.assert_called_once()

    def test_no_start_without_consensus(self):
        sync = DPProfilerSync()
        sync.request_start()  # requested but reduce has not agreed yet
        profiler = MagicMock()
        worker = self._worker(sync, profiler)

        Worker._maybe_apply_synced_profile(worker)
        profiler.start.assert_not_called()

    def test_invalid_bounds_fail_before_dp_consensus(self):
        sync = DPProfilerSync()
        worker = self._worker(sync, profiler=None)
        worker.profiler_config = ProfilerConfig(
            profiler="torch", torch_profiler_dir="/tmp/mock"
        )

        with pytest.raises(ValueError, match="max_iterations"):
            Worker.profile(worker, max_iterations=-1)

        assert not sync.pending_start

    def test_noop_when_sync_disabled(self):
        profiler = MagicMock()
        worker = self._worker(None, profiler)

        Worker._maybe_apply_synced_profile(worker)
        profiler.start.assert_not_called()

    def test_creates_profiler_on_remote_consensus(self):
        """A rank that never received /start_profile has no profiler wrapper
        yet. Reaching consensus via the OR-reduce must build one and start it,
        not silently consume the latch (regression: only the rank hit by the
        HTTP call ever captured)."""
        sync = DPProfilerSync()
        group = Mock(rank_in_group=1)
        group.broadcast_object.return_value = ("remote", 4, 10)
        sync.observe([1, 0], [0, 0], group)
        worker = self._worker(sync, profiler=None)
        worker.profiler = None

        def create(profile_prefix, profiler_kwargs=None):
            worker.profiler = MagicMock()
            return worker.profiler

        worker._create_profiler.side_effect = create

        Worker._maybe_apply_synced_profile(worker)

        worker._create_profiler.assert_called_once_with("remote")
        worker.profiler.start.assert_called_once_with(
            delay_iterations=4, max_iterations=10
        )

    def test_no_create_when_profiling_unconfigured(self):
        """Consensus with no profiler and no profiler config: skip, don't crash."""
        sync = DPProfilerSync()
        group = Mock(rank_in_group=1)
        group.broadcast_object.return_value = (None, None, None)
        sync.observe([1, 0], [0, 0], group)
        worker = self._worker(sync, profiler=None)
        worker.profiler = None
        worker.profiler_config = None

        Worker._maybe_apply_synced_profile(worker)

        worker._create_profiler.assert_not_called()

    def test_remote_stop_closes_profiler(self):
        sync = DPProfilerSync()
        sync.observe([0, 0], [1, 0], Mock(rank_in_group=1))
        worker = self._worker(sync, MagicMock())

        Worker._maybe_apply_synced_profile(worker)

        worker._stop_profiler.assert_called_once_with()


@pytest.mark.skip_global_cleanup
def test_idle_dp_engine_wakes_peers_for_profile_stop():
    core = object.__new__(DPEngineCoreProc)
    core.has_coordinator = True
    core.engines_running = False
    core.ignore_start_dp_wave = False
    core.current_wave = 7
    core.output_queue = MagicMock()

    with (
        patch("vllm.v1.engine.core.EngineCore.profile") as profile,
        patch("vllm.v1.engine.core.envs.VLLM_ENABLE_MULTINODE_PROFILING", True),
    ):
        core.profile(False)

    profile.assert_called_once_with(False, None, None, None, None)
    core.output_queue.put_nowait.assert_called_once()
    client_index, output = core.output_queue.put_nowait.call_args.args[0]
    assert client_index == -1
    assert isinstance(output, EngineCoreOutputs)
    assert output.start_wave == 7
    assert core.engines_running
