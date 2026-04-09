# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for EP fault tolerance.

These tests validate the fault-tolerance logic without requiring GPUs by
mocking the heavy infrastructure (engine managers, executors, ZMQ, etc.).
"""

import asyncio
import multiprocessing
import weakref
from dataclasses import dataclass, field
from datetime import timedelta
from threading import Event
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import msgspec.msgpack
import pytest
import torch

from vllm.config import ParallelConfig
from vllm.distributed.elastic_ep.elastic_execute import (
    ElasticEPScalingExecutor,
    rebuild_eplb_derived_maps,
    strip_dead_columns,
)
from vllm.distributed.elastic_ep.elastic_state import (
    ElasticEPScalingState,
    ScaleDownRemainingEngineState,
    _BarrierTimeoutError,
)
from vllm.distributed.elastic_ep.standby_state import (
    _build_surviving_rank_tensor,
)
from vllm.v1.engine import (
    EngineCoreRequest,
    ReconfigureDistributedRequest,
    ReconfigureRankType,
)
from vllm.v1.engine.core_client import (
    DPLBAsyncMPClient,
    ElasticScalingCache,
    MPClient,
)
from vllm.v1.engine.utils import CoreEngineActorManager, CoreEngineProcManager

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_parallel_config(**overrides) -> MagicMock:
    """Create a mock ParallelConfig with sane defaults."""
    pc = MagicMock(spec=ParallelConfig)
    pc.enable_elastic_ep = overrides.get("enable_elastic_ep", True)
    pc.enable_ep_fault_tolerance = overrides.get("enable_ep_fault_tolerance", True)
    pc.data_parallel_backend = overrides.get("data_parallel_backend", "ray")
    pc.tensor_parallel_size = overrides.get("tensor_parallel_size", 1)
    pc.data_parallel_size = overrides.get("data_parallel_size", 4)
    pc.data_parallel_master_ip = "127.0.0.1"
    pc.data_parallel_master_port = 29500
    pc._data_parallel_master_port_list = [29501, 29502, 29503]
    pc._coord_store_port = 29600
    return pc


def _make_vllm_config(**pc_overrides) -> MagicMock:
    config = MagicMock()
    config.parallel_config = _make_parallel_config(**pc_overrides)
    config.model_config = None
    return config


@dataclass
class FakeResources:
    engine_dead: bool = False
    engine_manager: Any = None
    coordinator: Any = None


@dataclass
class FakeTCPStore:
    """Minimal TCP store substitute for barrier tests."""

    _data: dict = field(default_factory=dict)

    def set(self, key: str, value: bytes) -> None:
        self._data[key] = value

    def get(self, key: str) -> bytes:
        return self._data.get(key, b"0")

    def check(self, keys: list[str]) -> bool:
        return all(k in self._data for k in keys)

    def delete_key(self, key: str) -> None:
        self._data.pop(key, None)

    def add(self, key: str, value: int) -> int:
        cur = int(self._data.get(key, b"0"))
        new_val = cur + value
        self._data[key] = str(new_val).encode()
        return new_val

    def compare_set(self, key: str, expected: str, value: bytes) -> bytes:
        cur = self._data.get(key, "")
        if cur == expected or cur == expected.encode():
            self._data[key] = value
        return self._data.get(key, b"")


# ===========================================================================
# 1. CoreEngineProcManager._dp_rank_from_proc_name
# ===========================================================================


class TestDPRankFromProcName:
    def test_standard_dp_name(self):
        assert CoreEngineProcManager._dp_rank_from_proc_name("EngineCore_DP0") == 0
        assert CoreEngineProcManager._dp_rank_from_proc_name("EngineCore_DP3") == 3
        assert CoreEngineProcManager._dp_rank_from_proc_name("EngineCore_DP15") == 15

    def test_non_dp_name(self):
        assert CoreEngineProcManager._dp_rank_from_proc_name("EngineCore") == -1
        assert CoreEngineProcManager._dp_rank_from_proc_name("SomeProcess") == -1


# ===========================================================================
# 2. DPLBAsyncMPClient._is_fault_tolerant
# ===========================================================================


class TestIsFaultTolerant:
    def _make_client_stub(self, **pc_overrides):
        """Create a bare DPLBAsyncMPClient instance without running __init__."""
        client = object.__new__(DPLBAsyncMPClient)
        client.vllm_config = _make_vllm_config(**pc_overrides)
        return client

    def test_true_when_flag_enabled(self):
        client = self._make_client_stub(enable_ep_fault_tolerance=True)
        assert client._is_fault_tolerant() is True

    def test_false_when_flag_disabled(self):
        client = self._make_client_stub(enable_ep_fault_tolerance=False)
        assert client._is_fault_tolerant() is False


# ===========================================================================
# 3. DPLBAsyncMPClient._on_engine_process_died
# ===========================================================================


class TestOnEngineProcessDied:
    def _make_client_stub(self, dp_size: int = 4, ft_in_progress: bool = False):
        client = object.__new__(DPLBAsyncMPClient)
        client.vllm_config = _make_vllm_config(data_parallel_size=dp_size)
        client.resources = FakeResources()
        client.core_engines = [i.to_bytes(2, "little") for i in range(dp_size)]
        client._ft_scaling_in_progress = ft_in_progress
        client._dead_engine_identities = set()
        client._ft_event_loop = None
        client.shutdown = MagicMock()
        client.reqs_in_flight = {}
        return client

    def test_returns_false_when_only_one_engine(self):
        client = self._make_client_stub(dp_size=1)
        result = client._on_engine_process_died("EngineCore_DP0", 0)
        assert result is False
        assert client.resources.engine_dead is True
        client.shutdown.assert_called_once()

    def test_returns_false_when_scaling_in_progress(self):
        client = self._make_client_stub(dp_size=4, ft_in_progress=True)
        result = client._on_engine_process_died("EngineCore_DP2", 2)
        assert result is False
        assert client.resources.engine_dead is True

    def test_schedules_scale_down_and_returns_true(self):
        client = self._make_client_stub(dp_size=4)

        loop = asyncio.new_event_loop()
        scheduled_coros = []
        loop.call_soon_threadsafe = lambda fn: scheduled_coros.append(fn)
        client._ft_event_loop = loop

        result = client._on_engine_process_died("EngineCore_DP2", 2)

        assert result is True
        assert client._ft_scaling_in_progress is True
        assert len(scheduled_coros) == 1
        # Verify dead engine identity is tracked for routing avoidance.
        expected_identity = (2).to_bytes(2, "little")
        assert expected_identity in client._dead_engine_identities
        loop.close()

    def test_duplicate_death_notification_is_ignored(self):
        """Ray may report the same actor death multiple times. The second
        notification for the same rank should be a no-op, not trigger a
        shutdown due to 'scaling already in progress'."""
        client = self._make_client_stub(dp_size=4)

        loop = asyncio.new_event_loop()
        scheduled_coros = []
        loop.call_soon_threadsafe = lambda fn: scheduled_coros.append(fn)
        client._ft_event_loop = loop

        # First notification — triggers scale-down.
        result1 = client._on_engine_process_died("EngineCore_DP2", 2)
        assert result1 is True
        assert client._ft_scaling_in_progress is True

        # Second notification for the SAME rank — should be ignored.
        result2 = client._on_engine_process_died("EngineCore_DP2", 2)
        assert result2 is True  # keep monitoring
        assert client.resources.engine_dead is False  # NOT shut down
        assert len(scheduled_coros) == 1  # only one scale-down scheduled
        client.shutdown.assert_not_called()
        loop.close()

    def test_returns_false_when_no_event_loop(self):
        client = self._make_client_stub(dp_size=4)
        # _ft_event_loop is already None from _make_client_stub
        assert client._ft_event_loop is None
        result = client._on_engine_process_died("EngineCore_DP1", 1)
        assert result is False
        assert client.resources.engine_dead is True


# ===========================================================================
# 4. DPLBAsyncMPClient._fail_inflight_requests_for_engine
# ===========================================================================


class TestFailInflightRequests:
    def test_removes_matching_requests(self):
        client = object.__new__(DPLBAsyncMPClient)
        engine_0 = (0).to_bytes(2, "little")
        engine_1 = (1).to_bytes(2, "little")
        client.reqs_in_flight = {
            "req-1": engine_0,
            "req-2": engine_1,
            "req-3": engine_0,
            "req-4": engine_1,
        }
        client._fail_inflight_requests_for_engine(engine_0)
        assert "req-1" not in client.reqs_in_flight
        assert "req-3" not in client.reqs_in_flight
        assert "req-2" in client.reqs_in_flight
        assert "req-4" in client.reqs_in_flight

    def test_noop_when_no_matching_requests(self):
        client = object.__new__(DPLBAsyncMPClient)
        engine_0 = (0).to_bytes(2, "little")
        engine_2 = (2).to_bytes(2, "little")
        client.reqs_in_flight = {"req-1": engine_0}
        client._fail_inflight_requests_for_engine(engine_2)
        assert len(client.reqs_in_flight) == 1

    def test_returns_dead_request_ids(self):
        """Verify that the method returns the IDs it dropped, so callers
        can propagate errors to waiting clients."""
        client = object.__new__(DPLBAsyncMPClient)
        engine_0 = (0).to_bytes(2, "little")
        engine_1 = (1).to_bytes(2, "little")
        client.reqs_in_flight = {
            "req-1": engine_0,
            "req-2": engine_1,
            "req-3": engine_0,
        }
        dead_ids = client._fail_inflight_requests_for_engine(engine_0)
        assert set(dead_ids) == {"req-1", "req-3"}


# ===========================================================================
# 4b. register_ft_abort_callback
# ===========================================================================


class TestRegisterFTAbortCallback:
    def test_registers_and_stores_callback(self):
        client = object.__new__(DPLBAsyncMPClient)
        client._ft_abort_callback = None

        callback = MagicMock()
        client.register_ft_abort_callback(callback)
        assert client._ft_abort_callback is callback

    def test_callback_is_callable(self):
        client = object.__new__(DPLBAsyncMPClient)
        client._ft_abort_callback = None

        called_with = []
        client.register_ft_abort_callback(lambda ids: called_with.extend(ids))
        client._ft_abort_callback(["req-1", "req-2"])
        assert called_with == ["req-1", "req-2"]


# ===========================================================================
# 5. ElasticEPScalingState barrier logic with dead ranks
# ===========================================================================


class TestElasticEPScalingStateBarriers:
    """Test the barrier and progress logic with dead ranks."""

    def _make_scaling_state(
        self, dead_dp_ranks: list[int] | None = None, dp_size: int = 4
    ):
        """Build a minimal ElasticEPScalingState for testing barriers."""
        reconfig = ReconfigureDistributedRequest(
            new_data_parallel_size=dp_size - 1,
            new_data_parallel_rank=ReconfigureRankType.KEEP_CURRENT_RANK,
            new_data_parallel_rank_local=ReconfigureRankType.KEEP_CURRENT_RANK,
            new_data_parallel_master_ip="127.0.0.1",
            new_data_parallel_master_port=29500,
            new_data_parallel_master_port_list=[29501],
            coord_store_port=29600,
            dead_dp_ranks=dead_dp_ranks or [],
        )

        fake_dp_group = MagicMock()
        fake_dp_group.rank.return_value = 0
        fake_dp_group.size.return_value = dp_size

        fake_store = FakeTCPStore()

        state = object.__new__(ElasticEPScalingState)
        state.old_dp_group = fake_dp_group
        state.old_dp_store = fake_store
        state.new_dp_group = None
        state.new_dp_store = None
        state.reconfig_request = reconfig
        state.dead_dp_ranks = set(dead_dp_ranks or [])
        state.worker_type = "existing"
        state.scale_type = "scale_down"
        state.state = ScaleDownRemainingEngineState.PREPARE
        state.vllm_config = _make_vllm_config()
        state.model_executor_ref = MagicMock(return_value=MagicMock())
        state.engine_core_ref = MagicMock(return_value=MagicMock())

        return state

    @pytest.mark.parametrize(
        ("dead", "dp_size", "expected_alive", "expected_first_alive"),
        [
            ([2], 4, 3, 0),
            ([], 4, 4, 0),
            ([0], 4, 3, 1),
            ([0, 1], 4, 2, 2),
        ],
    )
    def test_alive_group_helpers(
        self, dead, dp_size, expected_alive, expected_first_alive
    ):
        state = self._make_scaling_state(dead_dp_ranks=dead, dp_size=dp_size)
        assert state._alive_group_size() == expected_alive
        assert state._first_alive_rank() == expected_first_alive

    def test_tcp_store_barrier_skips_dead_ranks(self):
        state = self._make_scaling_state(dead_dp_ranks=[2], dp_size=4)
        store = state.old_dp_store

        # Simulate ranks 0, 1, 3 arriving (rank 2 is dead).
        for rank in [0, 1, 3]:
            store.set(f"arrival_test_barrier_{rank}", b"1")

        # Should complete without waiting for rank 2.
        state._execute_tcp_store_barrier(
            store,
            group_rank=0,
            group_size=4,
            barrier_id="test_barrier",
            timeout=None,
            skip_ranks={2},
        )

    def test_tcp_store_barrier_times_out_without_alive_rank(self):
        """Alive ranks 1 and 3 never arrive — barrier should time out."""
        state = self._make_scaling_state(dead_dp_ranks=[2], dp_size=4)
        state.old_dp_store.set("arrival_test_barrier_0", b"1")

        with pytest.raises(_BarrierTimeoutError):
            state._execute_tcp_store_barrier(
                state.old_dp_store, group_rank=0, group_size=4,
                barrier_id="test_barrier",
                timeout=timedelta(seconds=0.1), skip_ranks={2},
            )


# ===========================================================================
# 7. _synthesize_shutdown_complete_for_dead_ranks
# ===========================================================================


class TestSynthesizeShutdownComplete:
    def _make_client_stub(
        self, dp_size: int = 4, dead_dp_ranks: set[int] | None = None
    ):
        """Create a DPLBAsyncMPClient stub with eep_scaling_cache set up."""
        client = object.__new__(DPLBAsyncMPClient)
        mock_mgr = MagicMock(spec=CoreEngineActorManager)
        mock_mgr.scale_down_elastic_ep = MagicMock()
        client.resources = SimpleNamespace(engine_manager=mock_mgr)

        num_removed = len(dead_dp_ranks or set())
        client.eep_scaling_cache = ElasticScalingCache(
            existing_core_engines=[i.to_bytes(2, "little") for i in range(dp_size)],
            num_new_core_engines=-num_removed,
            pending_notifications={},
        )
        return client

    def test_injects_notifications_for_dead_ranks(self):
        dead = {3}
        client = self._make_client_stub(dp_size=4, dead_dp_ranks=dead)
        client._synthesize_shutdown_complete_for_dead_ranks(dead)

        cache = client.eep_scaling_cache
        # Cache should be cleared after all notifications are resolved.
        assert cache is None

    def test_partial_notification_does_not_clear_cache(self):
        # 2 ranks being removed, only 1 dead — cache should remain.
        client = self._make_client_stub(dp_size=4, dead_dp_ranks={3})
        client.eep_scaling_cache.num_new_core_engines = -2
        client._synthesize_shutdown_complete_for_dead_ranks({3})
        assert client.eep_scaling_cache is not None


# ===========================================================================
# 8. _scale_down_elastic_ep skips dead engines
# ===========================================================================


class TestScaleDownSkipsDeadEngines:
    def _make_client_stub(self, dp_size: int = 4):
        client = object.__new__(DPLBAsyncMPClient)
        client.vllm_config = _make_vllm_config(data_parallel_size=dp_size)
        client.core_engines = [i.to_bytes(2, "little") for i in range(dp_size)]
        client.lb_engines = [[0, 0] for _ in range(dp_size)]
        mock_mgr = MagicMock(spec=CoreEngineActorManager)
        mock_mgr.remove_run_refs_for_scale_down = MagicMock()
        mock_mgr.scale_down_elastic_ep = MagicMock()
        mock_mgr.placement_group_is_local = [True] * dp_size
        client.resources = SimpleNamespace(engine_manager=mock_mgr)
        client.eep_scaling_cache = None
        client.utility_results = {}
        client._ensure_stats_update_task = MagicMock()
        client._setup_elastic_ep_reconfig_bootstrap = MagicMock(
            return_value=("127.0.0.1", 29600)
        )
        client._coord_store = MagicMock(port=29600)
        client._eep_wait_for_setup_switch_complete = AsyncMock()
        client.first_req_send_socket = MagicMock()
        client.first_req_send_socket.send = AsyncMock()
        return client

    @pytest.mark.asyncio
    async def test_skips_reconfigure_for_dead_rank(self):
        """Verify that _scale_down_elastic_ep does not send reconfigure
        messages to dead engines."""
        client = self._make_client_stub()
        sent_engines = []

        async def mock_call_utility(method, *args, engine=None):
            sent_engines.append(engine)
            return None

        client._call_utility_async = mock_call_utility

        await client._scale_down_elastic_ep(
            cur_data_parallel_size=4,
            new_data_parallel_size=3,
            dead_dp_ranks={3},
        )

        dead_engine_id = (3).to_bytes(2, "little")
        assert dead_engine_id not in sent_engines
        assert len(sent_engines) == 3


# ===========================================================================
# 9. has_unfinished_dp fault_tolerant flag
# ===========================================================================


class TestHasUnfinishedDPFaultTolerant:
    def test_fault_tolerant_returns_true_on_exception(self):
        fake_group = MagicMock()
        with patch(
            "torch.distributed.all_reduce", side_effect=RuntimeError("gloo timeout")
        ):
            result = ParallelConfig.has_unfinished_dp(
                fake_group, True, fault_tolerant=True
            )
        assert result is True

    def test_non_fault_tolerant_raises_on_exception(self):
        fake_group = MagicMock()
        with (
            patch(
                "torch.distributed.all_reduce", side_effect=RuntimeError("gloo timeout")
            ),
            pytest.raises(RuntimeError, match="gloo timeout"),
        ):
            ParallelConfig.has_unfinished_dp(fake_group, True, fault_tolerant=False)

    def test_normal_operation_returns_correct_value(self):
        fake_group = MagicMock()

        def fake_all_reduce(tensor, **kwargs):
            tensor.fill_(1)

        with patch("torch.distributed.all_reduce", side_effect=fake_all_reduce):
            result = ParallelConfig.has_unfinished_dp(
                fake_group, False, fault_tolerant=False
            )
        assert result is True


# ===========================================================================
# 10. get_core_engine_for_request skips dead engines
# ===========================================================================


class TestGetCoreEngineSkipsDeadEngines:
    def _make_client_stub(self, dp_size: int = 4):
        client = object.__new__(DPLBAsyncMPClient)
        client.core_engines = [i.to_bytes(2, "little") for i in range(dp_size)]
        client.lb_engines = [[0, 0] for _ in range(dp_size)]
        client.eng_start_index = 0
        client.client_count = 1
        client._dead_engine_identities = set()
        client.reqs_in_flight = {}
        return client

    def _make_request(self, request_id: str = "req-1"):
        req = MagicMock(spec=EngineCoreRequest)
        req.request_id = request_id
        req.data_parallel_rank = None
        req.pooling_params = None
        return req

    def test_skips_dead_engine_even_when_cheapest(self):
        """Dead engine has the lowest score but must be skipped;
        router should pick the best alive engine instead."""
        client = self._make_client_stub(dp_size=4)
        dead_identity = (2).to_bytes(2, "little")
        client._dead_engine_identities.add(dead_identity)
        # Engine 2 looks cheapest, but it's dead → engine 1 should win.
        client.lb_engines = [[10, 10], [1, 1], [0, 0], [10, 10]]

        req = self._make_request()
        chosen = client.get_core_engine_for_request(req)
        assert chosen != dead_identity
        assert chosen == (1).to_bytes(2, "little")

    def test_routes_normally_with_no_dead_engines(self):
        client = self._make_client_stub(dp_size=4)
        client.lb_engines = [[10, 10], [0, 0], [5, 5], [5, 5]]

        req = self._make_request()
        chosen = client.get_core_engine_for_request(req)
        assert chosen == (1).to_bytes(2, "little")

    def test_records_request_in_flight(self):
        client = self._make_client_stub(dp_size=2)
        req = self._make_request("req-42")
        chosen = client.get_core_engine_for_request(req)
        assert client.reqs_in_flight["req-42"] == chosen

    def test_explicit_dp_rank_targeting_dead_engine_raises(self):
        """When request.data_parallel_rank explicitly targets a dead engine,
        a ValueError is raised so the client can retry on another rank."""
        client = self._make_client_stub(dp_size=4)
        client._dead_engine_identities.add((2).to_bytes(2, "little"))

        req = self._make_request()
        req.data_parallel_rank = 2

        with pytest.raises(ValueError, match="dead"):
            client.get_core_engine_for_request(req)


# ===========================================================================
# 11. _fault_triggered_scale_down async flow
# ===========================================================================


class TestFaultTriggeredScaleDown:
    def _make_client_stub(self, dp_size: int = 4):
        client = object.__new__(DPLBAsyncMPClient)
        client.vllm_config = _make_vllm_config(data_parallel_size=dp_size)
        client.core_engines = [i.to_bytes(2, "little") for i in range(dp_size)]
        client.lb_engines = [[0, 0] for _ in range(dp_size)]
        client.resources = FakeResources()
        client._ft_scaling_in_progress = True
        client._dead_engine_identities = set()
        client._ft_abort_callback = None
        client.reqs_in_flight = {
            "req-on-dead": (2).to_bytes(2, "little"),
            "req-on-alive": (0).to_bytes(2, "little"),
        }
        client.shutdown = MagicMock()
        return client

    @pytest.mark.asyncio
    async def test_success_calls_scale_down_and_clears_state(self):
        client = self._make_client_stub()
        client._scale_down_elastic_ep = AsyncMock()

        abort_callback = MagicMock()
        client._ft_abort_callback = abort_callback

        await client._fault_triggered_scale_down(dead_dp_rank=2)

        # Verify scale-down was called with correct args.
        client._scale_down_elastic_ep.assert_awaited_once_with(4, 3, dead_dp_ranks={2})
        # In-flight request on dead engine should be removed.
        assert "req-on-dead" not in client.reqs_in_flight
        assert "req-on-alive" in client.reqs_in_flight
        # Abort callback should have been invoked with the dead request IDs.
        # In production, AsyncLLM registers output_processor.abort_requests()
        # here, which sends FinishReason.ABORT to each request's client queue.
        abort_callback.assert_called_once_with(["req-on-dead"])
        # State should be cleared in finally block.
        assert client._ft_scaling_in_progress is False
        assert len(client._dead_engine_identities) == 0
        # Should NOT have set engine_dead on success.
        assert client.resources.engine_dead is False

    @pytest.mark.asyncio
    async def test_no_callback_does_not_crash(self):
        """When no abort callback is registered, dead requests are still
        removed from reqs_in_flight but no abort is propagated."""
        client = self._make_client_stub()
        client._scale_down_elastic_ep = AsyncMock()

        await client._fault_triggered_scale_down(dead_dp_rank=2)

        assert "req-on-dead" not in client.reqs_in_flight
        client._scale_down_elastic_ep.assert_awaited_once()
        assert client.resources.engine_dead is False

    @pytest.mark.asyncio
    async def test_failure_sets_engine_dead_and_shuts_down(self):
        client = self._make_client_stub()
        client._scale_down_elastic_ep = AsyncMock(
            side_effect=RuntimeError("scale-down failed")
        )

        await client._fault_triggered_scale_down(dead_dp_rank=2)

        assert client.resources.engine_dead is True
        client.shutdown.assert_called_once()
        # State should still be cleared in finally block.
        assert client._ft_scaling_in_progress is False


# ===========================================================================
# 12. monitor_engine_liveness_fault_tolerant (CoreEngineProcManager)
# ===========================================================================


class TestMonitorEngineLivenessFaultTolerant:
    def test_calls_on_engine_died_when_process_exits_nonzero(self):
        """Simulate a process dying with non-zero exit code."""
        mgr = object.__new__(CoreEngineProcManager)
        mgr.manager_stopped = Event()

        # Create a real subprocess that exits with code 1.
        proc = multiprocessing.Process(target=lambda: exit(1), name="EngineCore_DP2")
        proc.start()
        proc.join()  # Wait for it to finish.

        mgr.processes = [proc]
        died_reports = []

        def on_died(proc_name: str, dp_rank: int) -> bool:
            died_reports.append((proc_name, dp_rank))
            return False  # Stop monitoring.

        # shutdown should be called when on_died returns False.
        mgr.shutdown = MagicMock()

        mgr.monitor_engine_liveness_fault_tolerant(on_died)

        assert len(died_reports) == 1
        assert died_reports[0] == ("EngineCore_DP2", 2)

    def test_continues_monitoring_when_on_died_returns_true(self):
        """When on_died returns True, remaining sentinels should be monitored."""
        mgr = object.__new__(CoreEngineProcManager)
        mgr.manager_stopped = Event()

        # Two processes: one dies immediately, one exits cleanly.
        proc1 = multiprocessing.Process(target=lambda: exit(1), name="EngineCore_DP0")
        proc2 = multiprocessing.Process(target=lambda: exit(0), name="EngineCore_DP1")
        proc1.start()
        proc2.start()
        proc1.join()
        proc2.join()

        mgr.processes = [proc1, proc2]
        died_reports = []

        def on_died(proc_name: str, dp_rank: int) -> bool:
            died_reports.append((proc_name, dp_rank))
            return True  # Keep monitoring.

        mgr.shutdown = MagicMock()
        mgr.failed_proc_name = None

        mgr.monitor_engine_liveness_fault_tolerant(on_died)

        # Only the process with non-zero exit should be reported.
        assert len(died_reports) == 1
        assert died_reports[0][0] == "EngineCore_DP0"

    def test_skips_report_when_manager_stopped(self):
        """When the server is shutting down (SIGTERM), manager_stopped is set.
        Process deaths during shutdown should not trigger fault recovery."""
        mgr = object.__new__(CoreEngineProcManager)
        mgr.manager_stopped = Event()
        mgr.manager_stopped.set()  # Already stopped.

        proc = multiprocessing.Process(target=lambda: exit(1), name="EngineCore_DP0")
        proc.start()
        proc.join()

        mgr.processes = [proc]
        died_reports = []

        def on_died(proc_name: str, dp_rank: int) -> bool:
            died_reports.append((proc_name, dp_rank))
            return True

        mgr.shutdown = MagicMock()
        mgr.monitor_engine_liveness_fault_tolerant(on_died)

        # Should not report because manager_stopped was set.
        assert len(died_reports) == 0


# ===========================================================================
# 13. enable_ep_fault_tolerance config validation
# ===========================================================================


class TestEPFaultToleranceConfigValidation:
    def test_requires_elastic_ep(self):
        with pytest.raises(ValueError, match="enable_elastic_ep=True"):
            ParallelConfig(
                enable_ep_fault_tolerance=True,
                enable_elastic_ep=False,
                data_parallel_size=2,
            )

    def test_requires_tp_1(self):
        with pytest.raises(ValueError, match="tensor_parallel_size=1"):
            ParallelConfig(
                enable_ep_fault_tolerance=True,
                enable_elastic_ep=True,
                enable_eplb=True,
                tensor_parallel_size=2,
                data_parallel_size=2,
            )

    def test_requires_dp_greater_than_1(self):
        with pytest.raises(ValueError, match="data_parallel_size > 1"):
            ParallelConfig(
                enable_ep_fault_tolerance=True,
                enable_elastic_ep=True,
                enable_eplb=True,
                data_parallel_size=1,
            )

    def test_valid_config_accepted(self):
        """dp_size > 1 with elastic EP, EPLB, EP, and TP=1 should succeed."""
        pc = ParallelConfig(
            enable_ep_fault_tolerance=True,
            enable_elastic_ep=True,
            enable_eplb=True,
            enable_expert_parallel=True,
            data_parallel_size=2,
            tensor_parallel_size=1,
        )
        assert pc.enable_ep_fault_tolerance is True


# ===========================================================================
# 14. _first_alive_rank edge case: all ranks dead
# ===========================================================================


class TestFirstAliveRankAllDead:
    def test_raises_when_all_ranks_dead(self):
        state = object.__new__(ElasticEPScalingState)
        fake_dp_group = MagicMock()
        fake_dp_group.size.return_value = 3
        state.old_dp_group = fake_dp_group
        state.dead_dp_ranks = {0, 1, 2}

        with pytest.raises(RuntimeError, match="All ranks are dead"):
            state._first_alive_rank()


# ===========================================================================
# 15. _synthesize_shutdown_complete: verify engine_manager call
# ===========================================================================


class TestSynthesizeShutdownCompleteVerifySideEffects:
    def test_calls_engine_manager_scale_down(self):
        """Verify that scale_down_elastic_ep is called on the engine manager
        when all expected SHUTDOWN_COMPLETE notifications are present."""
        client = object.__new__(DPLBAsyncMPClient)
        mock_mgr = MagicMock(spec=CoreEngineActorManager)
        client.resources = SimpleNamespace(engine_manager=mock_mgr)

        client.eep_scaling_cache = ElasticScalingCache(
            existing_core_engines=[i.to_bytes(2, "little") for i in range(4)],
            num_new_core_engines=-1,
            pending_notifications={},
        )

        client._synthesize_shutdown_complete_for_dead_ranks({3})

        # Verify engine_manager was called with correct old/new sizes
        # and the removed rank set.
        mock_mgr.scale_down_elastic_ep.assert_called_once_with(
            4, 3, removed_dp_ranks={3},
        )
        # Cache should be cleared.
        assert client.eep_scaling_cache is None

    def test_does_not_call_engine_manager_when_not_all_notified(self):
        """When fewer than expected notifications arrive, engine_manager should
        NOT be called and cache should remain."""
        client = object.__new__(DPLBAsyncMPClient)
        mock_mgr = MagicMock(spec=CoreEngineActorManager)
        client.resources = SimpleNamespace(engine_manager=mock_mgr)

        client.eep_scaling_cache = ElasticScalingCache(
            existing_core_engines=[i.to_bytes(2, "little") for i in range(4)],
            num_new_core_engines=-2,  # Expecting 2 shutdowns.
            pending_notifications={},
        )

        client._synthesize_shutdown_complete_for_dead_ranks({3})  # Only 1.

        mock_mgr.scale_down_elastic_ep.assert_not_called()
        assert client.eep_scaling_cache is not None

    def test_noop_when_cache_is_none(self):
        client = object.__new__(DPLBAsyncMPClient)
        client.eep_scaling_cache = None
        # Should not raise.
        client._synthesize_shutdown_complete_for_dead_ranks({3})


# ===========================================================================
# 16. start_engine_core_monitor dispatch
# ===========================================================================


class TestStartEngineMonitorDispatch:
    def test_dispatches_to_fault_tolerant_monitor(self):
        client = object.__new__(MPClient)
        client.resources = SimpleNamespace(
            engine_manager=MagicMock(),
        )
        client._is_fault_tolerant = MagicMock(return_value=True)
        client._start_fault_tolerant_monitor = MagicMock()
        client._start_standard_monitor = MagicMock()

        client.start_engine_core_monitor()

        client._start_fault_tolerant_monitor.assert_called_once()
        client._start_standard_monitor.assert_not_called()

    def test_dispatches_to_standard_monitor(self):
        client = object.__new__(MPClient)
        client.resources = SimpleNamespace(
            engine_manager=MagicMock(),
        )
        client._is_fault_tolerant = MagicMock(return_value=False)
        client._start_fault_tolerant_monitor = MagicMock()
        client._start_standard_monitor = MagicMock()

        client.start_engine_core_monitor()

        client._start_standard_monitor.assert_called_once()
        client._start_fault_tolerant_monitor.assert_not_called()

    def test_noop_when_no_engine_manager(self):
        client = object.__new__(MPClient)
        client.resources = SimpleNamespace(engine_manager=None)
        client._is_fault_tolerant = MagicMock(return_value=True)
        client._start_fault_tolerant_monitor = MagicMock()

        client.start_engine_core_monitor()

        client._start_fault_tolerant_monitor.assert_not_called()


# ===========================================================================
# 17. Mid-rank death: _scale_down_elastic_ep rank compaction
# ===========================================================================


class TestScaleDownMidRankDeath:
    """Verify _scale_down_elastic_ep handles both fault-triggered
    (mid-rank death with compaction) and graceful (highest rank removed)."""

    def _make_client_stub(self, dp_size: int = 4):
        client = object.__new__(DPLBAsyncMPClient)
        client.vllm_config = _make_vllm_config(data_parallel_size=dp_size)
        client.core_engines = [i.to_bytes(2, "little") for i in range(dp_size)]
        client.lb_engines = [[i, i] for i in range(dp_size)]
        mock_mgr = MagicMock(spec=CoreEngineActorManager)
        mock_mgr.remove_run_refs_for_scale_down = MagicMock()
        mock_mgr.scale_down_elastic_ep = MagicMock()
        mock_mgr.placement_group_is_local = [True] * dp_size
        client.resources = SimpleNamespace(engine_manager=mock_mgr)
        client.eep_scaling_cache = None
        client.utility_results = {}
        client._ensure_stats_update_task = MagicMock()
        client._setup_elastic_ep_reconfig_bootstrap = MagicMock(
            return_value=("127.0.0.1", 29600)
        )
        client._coord_store = MagicMock(port=29600)
        client._eep_wait_for_setup_switch_complete = AsyncMock()
        client.first_req_send_socket = MagicMock()
        client.first_req_send_socket.send = AsyncMock()
        return client

    @pytest.mark.asyncio
    async def test_fault_triggered_mid_rank_death(self):
        """Rank 1 dies from [0,1,2,3]: dead rank removed, survivors get
        compacted ranks, no alive engine gets SHUTDOWN, coordinator
        marker includes removed rank."""
        client = self._make_client_stub(dp_size=4)
        reconfig_by_engine: dict[bytes, ReconfigureDistributedRequest] = {}

        async def mock_call_utility(method, *args, engine=None):
            if method == "reinitialize_distributed" and args:
                reconfig_by_engine[engine] = args[0]
            return None

        client._call_utility_async = mock_call_utility

        await client._scale_down_elastic_ep(
            cur_data_parallel_size=4,
            new_data_parallel_size=3,
            dead_dp_ranks={1},
        )

        # core_engines: dead rank removed, not truncated.
        assert client.core_engines == [
            r.to_bytes(2, "little") for r in [0, 2, 3]
        ]
        # lb_engines: dead entry removed.
        assert client.lb_engines == [[0, 0], [2, 2], [3, 3]]
        # No alive engine told to SHUTDOWN; compacted ranks assigned.
        assert (1).to_bytes(2, "little") not in reconfig_by_engine
        for engine, req in reconfig_by_engine.items():
            assert req.new_data_parallel_rank != (
                ReconfigureRankType.SHUTDOWN_CURRENT_RANK
            )
        assert reconfig_by_engine[(0).to_bytes(2, "little")].new_data_parallel_rank == 0
        assert reconfig_by_engine[(2).to_bytes(2, "little")].new_data_parallel_rank == 1
        assert reconfig_by_engine[(3).to_bytes(2, "little")].new_data_parallel_rank == 2
        # Coordinator marker includes removed rank.
        marker = msgspec.msgpack.decode(
            client.first_req_send_socket.send.call_args[0][0]
        )
        assert marker == ["SCALE_ELASTIC_EP", 3, [1]]

    @pytest.mark.asyncio
    async def test_graceful_scale_down_unchanged(self):
        """Graceful (no dead ranks): highest rank gets SHUTDOWN,
        others keep current rank, core_engines truncated normally."""
        client = self._make_client_stub(dp_size=4)
        reconfig_by_engine: dict[bytes, ReconfigureDistributedRequest] = {}

        async def mock_call_utility(method, *args, engine=None):
            if method == "reinitialize_distributed" and args:
                reconfig_by_engine[engine] = args[0]
            return None

        client._call_utility_async = mock_call_utility

        await client._scale_down_elastic_ep(
            cur_data_parallel_size=4,
            new_data_parallel_size=3,
            dead_dp_ranks=None,
        )

        assert (
            reconfig_by_engine[(3).to_bytes(2, "little")].new_data_parallel_rank
            == ReconfigureRankType.SHUTDOWN_CURRENT_RANK
        )
        for r in [0, 1, 2]:
            assert (
                reconfig_by_engine[r.to_bytes(2, "little")].new_data_parallel_rank
                == ReconfigureRankType.KEEP_CURRENT_RANK
            )
        assert client.core_engines == [
            i.to_bytes(2, "little") for i in range(3)
        ]


# ===========================================================================
# 18. _build_surviving_rank_tensor
# ===========================================================================


class TestBuildSurvivingRankTensor:
    @pytest.mark.parametrize(
        ("new_dp", "pp", "tp", "dead", "world_ranks",
         "expected_ranks", "expected_shape"),
        [
            # First scale-down: contiguous world ranks
            (3, 1, 1, {1}, [0, 1, 2, 3], [0, 2, 3], (1, 3, 1, 1)),
            (2, 1, 1, {0}, [0, 1, 2], [1, 2], (1, 2, 1, 1)),
            (3, 1, 2, {1}, [0, 1, 2, 3, 4, 5, 6, 7],
             [0, 1, 4, 5, 6, 7], (1, 3, 1, 2)),
            # Second scale-down: non-contiguous world ranks
            (2, 1, 1, {2}, [0, 2, 3], [0, 2], (1, 2, 1, 1)),
        ],
    )
    def test_surviving_ranks(
        self, new_dp, pp, tp, dead, world_ranks,
        expected_ranks, expected_shape,
    ):
        result = _build_surviving_rank_tensor(
            new_dp_size=new_dp, pp_size=pp, tp_size=tp,
            dead_dp_ranks=dead,
            current_world_ranks=world_ranks,
        )
        assert sorted(result.flatten().tolist()) == expected_ranks
        assert result.shape == expected_shape


# ===========================================================================
# 19. perform_scale_down_eplb_reshuffle rank mapping
# ===========================================================================


class TestScaleDownEPLBReshuffleMapping:
    """Verify rank_mapping maps dead ranks to -1 and compacts survivors."""

    def _make_executor(self, tp_size: int, dp_size: int):
        executor = object.__new__(ElasticEPScalingExecutor)
        worker = MagicMock()
        worker.vllm_config.parallel_config.tensor_parallel_size = tp_size
        worker.vllm_config.parallel_config.data_parallel_size = dp_size
        executor.worker_ref = weakref.ref(worker)
        executor._worker_strong_ref = worker

        captured = {}

        def mock_reshuffle(rank_mapping=None):
            captured["rank_mapping"] = rank_mapping

        executor._perform_eplb_reshuffle = mock_reshuffle
        executor._set_eplb_suppressed = MagicMock()
        return executor, captured

    @pytest.mark.parametrize(
        ("tp", "dp", "new_dp", "dead", "expected_mapping"),
        [
            # TP=1: rank 1 dies → dead=-1, compact rest.
            (1, 4, 3, [1], {0: 0, 1: -1, 2: 1, 3: 2}),
            # TP=1: graceful → highest rank removed.
            (1, 4, 3, None, {0: 0, 1: 1, 2: 2, 3: -1}),
            # TP=2: DP rank 1 dies → EP ranks 2,3 dead.
            (2, 3, 2, [1], {0: 0, 1: 1, 2: -1, 3: -1, 4: 2, 5: 3}),
        ],
    )
    def test_rank_mapping(self, tp, dp, new_dp, dead, expected_mapping):
        executor, captured = self._make_executor(tp_size=tp, dp_size=dp)
        executor.perform_scale_down_eplb_reshuffle(
            new_dp_size=new_dp, dead_dp_ranks=dead
        )
        assert captured["rank_mapping"] == expected_mapping


# ===========================================================================
# 20. reassign_missing_experts
# ===========================================================================


class TestReassignMissingExperts:
    """Verify reassign_missing_experts compacts p2l, replaces missing
    logical experts with redundant replicas, and rebuilds derived maps."""

    def _make_executor_with_eplb(
        self,
        p2l: "torch.Tensor",
        num_logical: int,
        tp_size: int,
        new_ep_size: int,
        dead_dp_ranks: list[int],
    ):
        executor = object.__new__(ElasticEPScalingExecutor)
        worker = MagicMock()
        worker.vllm_config.parallel_config.tensor_parallel_size = tp_size
        executor.worker_ref = weakref.ref(worker)
        executor._worker_strong_ref = worker

        reconfig = MagicMock()
        reconfig.dead_dp_ranks = dead_dp_ranks
        executor.reconfig_request = reconfig

        ep_group = MagicMock()
        ep_group.rank = 0
        ep_group.world_size = new_ep_size

        num_moe_layers = p2l.shape[0]
        state = SimpleNamespace(
            physical_to_logical_map=p2l.clone(),
            logical_replica_count=torch.zeros(
                num_moe_layers, num_logical, dtype=torch.long
            ),
            logical_to_physical_map=torch.full(
                (num_moe_layers, num_logical, 4), -1, dtype=torch.long,
            ),
            communicator=MagicMock(),
        )
        worker.model_runner.eplb_state = SimpleNamespace(
            model_states={"h": state},
        )
        worker.model_runner.model_config.compute_hash.return_value = "h"
        worker.model_runner.get_model.return_value = MagicMock(
            expert_weights=[],
        )

        return executor, state, ep_group

    _REARRANGE_PATH = (
        "vllm.distributed.eplb.rebalance_execute"
        ".rearrange_expert_weights_inplace"
    )
    _EP_GROUP_PATH = (
        "vllm.distributed.elastic_ep.elastic_execute.get_ep_group"
    )

    def test_compaction_and_missing_expert_replacement(self):
        """p2l already compacted by _apply_new_config.  Expert 2 was only
        on the dead rank.  Verifies: expert 2 placed into a redundant
        slot, derived maps consistent."""
        # After compaction (rank 1 removed):
        # rank0=[0,1], rank2=[0,1], rank3=[0,3]
        # Expert 2 missing.  Expert 0 has 3 replicas.
        p2l = torch.tensor([[0, 1, 0, 1, 0, 3]])
        executor, state, ep_group = self._make_executor_with_eplb(
            p2l=p2l, num_logical=4, tp_size=1,
            new_ep_size=3, dead_dp_ranks=[1],
        )

        with (
            patch(self._EP_GROUP_PATH, return_value=ep_group),
            patch(self._REARRANGE_PATH),
        ):
            executor.reassign_missing_experts()

        final_p2l = state.physical_to_logical_map[0]

        # Still 6 columns (already compacted).
        assert final_p2l.shape[0] == 6
        # Expert 2 is now present.
        assert 2 in final_p2l.tolist()
        # Expert 0 lost one redundant replica (3 → 2).
        assert final_p2l.tolist().count(0) == 2
        # All 4 logical experts covered.
        assert set(final_p2l.tolist()) == {0, 1, 2, 3}
        # Derived maps consistent.
        for lid in range(4):
            expected = (final_p2l == lid).sum().item()
            assert state.logical_replica_count[0, lid].item() == expected
            for r in range(expected):
                phys = state.logical_to_physical_map[0, lid, r].item()
                assert phys >= 0
                assert final_p2l[phys].item() == lid

    def test_no_reassignment_when_all_present(self):
        """Kill rank 3 — all experts still have replicas on survivors."""
        # After compaction (rank 3 removed):
        # rank0=[0,1], rank1=[2,3], rank2=[0,2]
        p2l = torch.tensor([[0, 1, 2, 3, 0, 2]])
        executor, state, ep_group = self._make_executor_with_eplb(
            p2l=p2l, num_logical=4, tp_size=1,
            new_ep_size=3, dead_dp_ranks=[3],
        )

        with (
            patch(self._EP_GROUP_PATH, return_value=ep_group),
            patch(self._REARRANGE_PATH),
        ):
            executor.reassign_missing_experts()

        # All 4 experts present, no eviction needed.
        assert set(state.physical_to_logical_map[0].tolist()) == {0, 1, 2, 3}

    def test_derived_maps_correct_after_compaction_no_missing(self):
        """After p2l compaction with no missing experts,
        rebuild_eplb_derived_maps must produce l2p entries that
        reference the NEW compacted physical indices, not the old
        pre-compaction indices.

        This reproduces the bug that caused garbled output:
        _apply_new_config compacted p2l but didn't rebuild derived
        maps, so the router used stale slot indices.
        """
        # BEFORE compaction (4 ranks, 2 local each = 8 physical):
        # rank0=[0,1], rank1=[2,3], rank2=[0,2], rank3=[1,3]
        old_p2l = torch.tensor([[0, 1, 2, 3, 0, 2, 1, 3]])
        num_logical = 4
        max_replicas = 4

        # Compact using the real production function.
        new_p2l = strip_dead_columns(old_p2l, dead_ep_ranks={1}, num_local=2)
        num_physical = new_p2l.shape[1]  # 6

        # Build an eplb_model_state with stale derived maps (old indices).
        state = SimpleNamespace(
            physical_to_logical_map=new_p2l,
            logical_replica_count=torch.zeros(1, num_logical,
                                              dtype=torch.long),
            logical_to_physical_map=torch.full(
                (1, num_logical, max_replicas), -1, dtype=torch.long,
            ),
        )

        # Call the real production function.
        rebuild_eplb_derived_maps(state)

        # All physical indices in l2p must be valid (< num_physical).
        for lid in range(num_logical):
            count = state.logical_replica_count[0, lid].item()
            assert count >= 1, f"Expert {lid} has 0 replicas"
            for r in range(count):
                phys = state.logical_to_physical_map[0, lid, r].item()
                assert 0 <= phys < num_physical, (
                    f"Expert {lid} replica {r} points to slot {phys} "
                    f"but only {num_physical} slots exist"
                )
                assert new_p2l[0, phys].item() == lid

    def test_no_donor_drained_to_zero(self):
        """Multiple missing experts must not drain all replicas from a
        single donor.  Expert 0 has 3 replicas and 2 experts are
        missing — naive greedy could evict all copies of expert 0."""
        # After compaction (rank 1 removed):
        # rank0=[0,1], rank2=[0,1], rank3=[0,1]
        # Experts 2 and 3 are missing.  Expert 0 has 3 replicas,
        # expert 1 has 3 replicas.
        p2l = torch.tensor([[0, 1, 0, 1, 0, 1]])
        executor, state, ep_group = self._make_executor_with_eplb(
            p2l=p2l, num_logical=4, tp_size=1,
            new_ep_size=3, dead_dp_ranks=[1],
        )

        with (
            patch(self._EP_GROUP_PATH, return_value=ep_group),
            patch(self._REARRANGE_PATH),
        ):
            executor.reassign_missing_experts()

        final = state.physical_to_logical_map[0].tolist()
        # All 4 logical experts must be present.
        assert set(final) == {0, 1, 2, 3}
        # No expert should have 0 replicas.
        for lid in range(4):
            assert final.count(lid) >= 1, (
                f"Expert {lid} has 0 replicas: {final}"
            )
