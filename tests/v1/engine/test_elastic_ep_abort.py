# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from vllm.distributed.elastic_ep.elastic_state import ElasticEPScalingState
from vllm.v1.engine.core import DPEngineCoreProc
from vllm.v1.engine.core_client import CoreEngineActorManager, DPLBAsyncMPClient


class _FakeEngineCore:
    def __init__(self) -> None:
        self.dp_group = MagicMock()
        self.dp_group.rank.return_value = 0
        self.dp_store = MagicMock()


def _make_scaling_state(init_fn) -> ElasticEPScalingState:
    engine_core = _FakeEngineCore()
    new_parallel_config = MagicMock()
    new_parallel_config.stateless_init_dp_group.side_effect = init_fn
    vllm_config = SimpleNamespace(parallel_config=MagicMock())
    return ElasticEPScalingState(
        model_executor=MagicMock(),
        engine_core=engine_core,  # type: ignore[arg-type]
        vllm_config=vllm_config,  # type: ignore[arg-type]
        new_parallel_config=new_parallel_config,
        worker_type="existing",
        scale_type="scale_up",
    )


def test_elastic_ep_state_abort_prevents_late_future_crash():
    """Aborting a stuck prepare future prevents late store timeout from raising."""
    unblock = threading.Event()
    finished = threading.Event()

    def slow_fail_init(return_store: bool = True):
        try:
            unblock.wait(timeout=5)
            raise RuntimeError("DistStoreError: wait timeout after 1800000ms")
        finally:
            finished.set()

    state = _make_scaling_state(slow_fail_init)
    assert state.progress() is False
    assert state._prepare_future is not None

    state.abort()
    assert state._prepare_future is None

    # Unblock background worker so its exception completes, then step progress().
    unblock.set()
    assert finished.wait(timeout=5)
    assert state.progress() is False


def test_engine_core_abort_prepared_elastic_ep_resets_state():
    """DPEngineCoreProc.abort_prepared_elastic_ep aborts and clears scaling state."""
    state = MagicMock()
    core = SimpleNamespace(eep_scaling_state=state, process_input_queue_block=True)

    DPEngineCoreProc.abort_prepared_elastic_ep(core)  # type: ignore[arg-type]

    state.abort.assert_called_once_with()
    assert core.eep_scaling_state is None
    assert core.process_input_queue_block is False

    # Idempotent when no scaling is in progress.
    DPEngineCoreProc.abort_prepared_elastic_ep(core)  # type: ignore[arg-type]
    assert core.eep_scaling_state is None


@pytest.mark.asyncio
async def test_prepare_scale_up_aborts_engines_on_coord_store_failure():
    """Client sends abort_prepared_elastic_ep to all engines when wait fails."""
    client = object.__new__(DPLBAsyncMPClient)
    client.core_engines = [b"\x00\x00", b"\x01\x00"]
    client.vllm_config = SimpleNamespace(parallel_config=MagicMock())
    client._setup_elastic_ep_reconfig_bootstrap = MagicMock()
    client._make_reconfig_request = MagicMock(return_value="reconfig_req")
    client._coord_store = MagicMock()
    client._coord_store.wait.side_effect = TimeoutError("wait timeout after 300000ms")

    engine_manager = MagicMock(spec=CoreEngineActorManager)
    client.resources = SimpleNamespace(engine_manager=engine_manager)

    utility_calls: list[tuple[str, bytes]] = []

    async def fake_call_utility(method: str, *args, engine: bytes, **kwargs):
        utility_calls.append((method, engine))
        if method == "reinitialize_distributed":
            return f"eep_ready/{int.from_bytes(engine, 'little')}"
        return None

    client._call_utility_async = AsyncMock(side_effect=fake_call_utility)

    with pytest.raises(TimeoutError, match="300000ms"):
        await client._prepare_scale_up_elastic_ep(
            new_data_parallel_size=4, num_redundant_experts=0
        )

    assert ("abort_prepared_elastic_ep", b"\x00\x00") in utility_calls
    assert ("abort_prepared_elastic_ep", b"\x01\x00") in utility_calls
