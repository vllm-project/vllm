# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio
import gc
from datetime import timedelta
from queue import Queue
from threading import Event
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from torch.distributed import TCPStore

from vllm.config import ParallelConfig, VllmConfig
from vllm.distributed import utils as distributed_utils
from vllm.distributed.elastic_ep.elastic_state import (
    ElasticEPScalingState,
    ScaleUpExistingEngineState,
)
from vllm.distributed.elastic_ep.external_elastic_ep import (
    ExternalElasticEPScaleCoordinator,
)
from vllm.utils.network_utils import get_open_port


def _scale_coordinator(**parallel_kwargs):
    config = VllmConfig(parallel_config=ParallelConfig(**parallel_kwargs))
    return ExternalElasticEPScaleCoordinator(
        MagicMock(client_index=0, vllm_config=config)
    )


def test_migration_master_retry_keeps_completed_peer_ready(monkeypatch):
    master, peer = _scale_coordinator(), _scale_coordinator()
    peer_done = Event()
    port = get_open_port()

    class FailOnceStore(TCPStore):
        def __init__(self, **kwargs):
            self.fail_once = kwargs["is_master"]
            super().__init__(**kwargs)

        def wait(self, keys, timeout):
            super().wait(keys, timedelta(seconds=5))
            if self.fail_once:
                self.fail_once = False
                assert peer_done.wait(5)
                raise ConnectionError("injected master wait failure")

    monkeypatch.setattr(distributed_utils, "TCPStore", FailOnceStore)

    async def run():
        async def fail_master():
            with pytest.raises(ConnectionError, match="injected"):
                await master.migrate_coord_store("127.0.0.1", port, 0, 2, "ft")

        async def finish_peer():
            await peer.migrate_coord_store("127.0.0.1", port, 1, 2, "ft")
            peer_done.set()

        await asyncio.gather(fail_master(), finish_peer())
        gc.collect()  # Drop exception-local references to the server.
        # A completed peer does not run migration again on the same request.
        await master.migrate_coord_store("127.0.0.1", port, 0, 2, "ft")
        assert peer.client._coord_store.get(master.key("ft", "ft", "ready", 0)) == b"1"

    asyncio.run(run())


def test_sparse_scale_down_rejected_before_bootstrap():
    coordinator = _scale_coordinator(data_parallel_size=4, data_parallel_rank=1)
    with pytest.raises(ValueError, match="sparse.*restore"):
        asyncio.run(coordinator.prepare(2, 1, existing_dp_ranks=[1, 3]))
    assert coordinator.active_epoch is None
    coordinator.client.call_utility_async.assert_not_called()


def _commit_state_after_fault(new_size=4):
    from vllm.v1.fault_tolerance.engine_core_sentinel import EngineCoreSentinel

    parallel_config = SimpleNamespace(
        data_parallel_rank=3,
        data_parallel_size=4,
        data_parallel_master_ip="127.0.0.1",
        _data_parallel_master_rank=0,
        fault_tolerance_config=SimpleNamespace(engine_recovery_timeout_sec=300),
    )
    engine = SimpleNamespace(
        engine_index=3,
        vllm_config=SimpleNamespace(parallel_config=parallel_config),
        output_queue=Queue(),
        dp_store=MagicMock(),
        model_executor=SimpleNamespace(failure_callback=None),
    )
    sentinel = engine.ft_sentinel = EngineCoreSentinel(engine, parallel_config)
    sentinel._recover_and_vote = MagicMock()
    sentinel._dead_dp_ranks = {2}

    state = object.__new__(ElasticEPScalingState)
    state.engine_core_ref = lambda: engine
    state.vllm_config = engine.vllm_config
    state.scale_type = "scale_up"
    state.old_dp_group = object()
    state.state = ScaleUpExistingEngineState.COMMIT_SCALE_UP
    state.commit_requested = True
    state.reconfig_request = SimpleNamespace(
        new_data_parallel_size=new_size,
        new_data_parallel_rank=3,
        new_data_parallel_rank_local=0,
        new_data_parallel_master_ip="127.0.0.1",
        new_data_parallel_master_port=29860,
        new_data_parallel_master_port_list=[],
        coord_store_port=29861,
        existing_dp_ranks=[0, 1, 3],
        data_parallel_master_rank=0,
    )
    state._commit_new_dp_group = MagicMock()
    state._collective_rpc = MagicMock()
    state._send_reconfigure_finished = MagicMock()
    return state, sentinel


@pytest.mark.parametrize("new_size,removed_rank", [(4, 2), (5, 4)])
def test_scale_up_rebases_fault_state_for_the_next_failure(new_size, removed_rank):
    from vllm.v1.fault_tolerance.utils import FaultToleranceRequest

    state, sentinel = _commit_state_after_fault(new_size)
    assert state._progress_existing_engine()
    request = FaultToleranceRequest(
        "scale_down", {"removed_dp_ranks": [removed_rank]}, request_id="shrink-2"
    )
    sentinel.scale_down(request)

    # A replacement may reuse a previously dead rank. Only this cycle's
    # failure belongs in the worker recovery request.
    assert request.params["dead_dp_ranks"] == [removed_rank]
    _, output = sentinel.engine.output_queue.get_nowait()
    assert output.engine_index == 3
    assert output.utility_output.result.result == {
        "id": 3,
        "status": "healthy",
    }


def test_failed_scale_up_keeps_fault_state():
    state, sentinel = _commit_state_after_fault()
    state._collective_rpc.side_effect = RuntimeError("worker commit failed")
    with pytest.raises(RuntimeError, match="worker commit failed"):
        state._progress_existing_engine()
    assert sentinel._dead_dp_ranks == {2}
    assert sentinel.parallel_config.data_parallel_rank == 3
    assert sentinel.engine.output_queue.empty()
