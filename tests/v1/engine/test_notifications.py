# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the engine-notification buffer and cross-rank gather.

Bare `EngineCore` instances: the helpers only touch `_pending_notifications`,
so no model is needed.
"""

import queue
from types import SimpleNamespace

import msgspec
import pytest

from vllm.v1.engine import EngineCoreOutputs
from vllm.v1.engine.core import EngineCore, EngineCoreProc
from vllm.v1.notifications import (
    CustomNotification,
    publish_worker_notification,
    take_worker_notifications,
)


@pytest.fixture(autouse=True)
def _drain_worker_buffer():
    """Process-local buffer; don't leak events between tests."""
    take_worker_notifications()
    yield
    take_worker_notifications()


def _bare_engine_core() -> EngineCore:
    engine_core = EngineCore.__new__(EngineCore)
    engine_core._pending_notifications = []
    return engine_core


def test_notifications_accumulate_additively():
    """Additive: no coalescing, or counter increments would get lost."""
    engine_core = _bare_engine_core()

    first = CustomNotification(key="my_plugin", payload={"n": 1})
    second = CustomNotification(key="my_plugin", payload={"n": 2})
    engine_core._publish_notifications([first])
    engine_core._publish_notifications([second])

    outputs: dict[int, EngineCoreOutputs] = {}
    engine_core._flush_notifications(outputs)

    assert outputs[0].engine_notifications == [first, second]


def test_flush_clears_buffer_between_steps():
    """A flush drains the buffer; later events are independent."""
    engine_core = _bare_engine_core()

    engine_core._publish_notifications([CustomNotification(key="my_plugin")])
    first_outputs: dict[int, EngineCoreOutputs] = {}
    engine_core._flush_notifications(first_outputs)

    second_outputs: dict[int, EngineCoreOutputs] = {}
    engine_core._flush_notifications(second_outputs)
    assert second_outputs == {}

    later = CustomNotification(key="my_plugin", payload={"n": 3})
    engine_core._publish_notifications([later])
    third_outputs: dict[int, EngineCoreOutputs] = {}
    engine_core._flush_notifications(third_outputs)
    assert third_outputs[0].engine_notifications == [later]


def test_flush_reuses_existing_outputs():
    """Notifications attach to an existing per-engine outputs entry."""
    engine_core = _bare_engine_core()

    event = CustomNotification(key="my_plugin")
    engine_core._publish_notifications([event])

    existing = EngineCoreOutputs(engine_index=2)
    outputs = {2: existing}
    engine_core._flush_notifications(outputs)

    assert outputs == {2: existing}
    assert existing.engine_notifications == [event]


def test_flush_noop_when_empty():
    """Nothing is added to outputs when the buffer is empty."""
    engine_core = _bare_engine_core()

    outputs: dict[int, EngineCoreOutputs] = {}
    engine_core._flush_notifications(outputs)
    assert outputs == {}


def test_worker_publish_take_roundtrip():
    """The out-of-tree producer path."""
    event = CustomNotification(key="my_plugin", payload={"bytes": 1})
    publish_worker_notification(event)

    assert take_worker_notifications() == [event]
    # None rather than [], so the quiet path allocates nothing.
    assert take_worker_notifications() is None


def test_empty_drain_allocates_nothing():
    """No producer installed builds no list."""
    assert take_worker_notifications() is None


def test_worker_notifications_survive_until_the_first_drain():
    """Load-time producers publish before any gather runs."""
    first = CustomNotification(key="my_plugin", payload={"n": 1})
    second = CustomNotification(key="my_plugin", payload={"n": 2})
    publish_worker_notification(first)
    publish_worker_notification(second)

    assert take_worker_notifications() == [first, second]


def _gathering_engine_core(per_rank) -> EngineCore:
    """EngineCore whose executor answers take_notifications with per_rank."""
    engine_core = _bare_engine_core()
    executor = SimpleNamespace(collective_rpc=lambda method: per_rank)
    engine_core.model_executor = executor
    return engine_core


def test_gather_keeps_every_rank():
    """A rank-0-only reply would drop per-rank producers entirely."""
    rank0 = CustomNotification(key="my_plugin", payload={"rank": 0})
    rank1 = CustomNotification(key="my_plugin", payload={"rank": 1})
    rank2 = CustomNotification(key="my_plugin", payload={"rank": 2})
    engine_core = _gathering_engine_core([[rank0], [rank1], [rank2]])

    engine_core.gather_worker_notifications()

    outputs: dict[int, EngineCoreOutputs] = {}
    engine_core._flush_notifications(outputs)
    assert outputs[0].engine_notifications == [rank0, rank1, rank2]


def test_gather_publishes_nothing_when_all_ranks_are_quiet():
    engine_core = _gathering_engine_core([[], [], []])

    engine_core.gather_worker_notifications()

    outputs: dict[int, EngineCoreOutputs] = {}
    engine_core._flush_notifications(outputs)
    assert outputs == {}


def test_gather_tolerates_ranks_that_cannot_report():
    """A worker without the method replies None."""
    event = CustomNotification(key="my_plugin", payload={"rank": 1})
    engine_core = _gathering_engine_core([None, [event]])

    engine_core.gather_worker_notifications()

    outputs: dict[int, EngineCoreOutputs] = {}
    engine_core._flush_notifications(outputs)
    assert outputs[0].engine_notifications == [event]


def test_gather_never_raises_into_the_engine():
    """A metrics channel must not take down an engine operation."""
    engine_core = _bare_engine_core()

    def boom(method):
        raise RuntimeError("executor is gone")

    engine_core.model_executor = SimpleNamespace(collective_rpc=boom)
    engine_core.gather_worker_notifications()

    outputs: dict[int, EngineCoreOutputs] = {}
    engine_core._flush_notifications(outputs)
    assert outputs == {}


def test_engine_core_outputs_round_trip_over_the_wire():
    """array_like, so the field is positional and always present.

    omit_defaults does not trim trailing fields, and non-Python frontends
    reject an array longer than they know.
    """
    event = CustomNotification(key="my_plugin", payload={"count": 5})
    encoded = msgspec.msgpack.encode(
        EngineCoreOutputs(engine_index=1, engine_notifications=[event])
    )

    as_array = msgspec.msgpack.decode(encoded)
    assert isinstance(as_array, list)
    assert as_array[-1] == [
        {"type": "custom", "key": "my_plugin", "payload": {"count": 5}}
    ]

    decoded = msgspec.msgpack.decode(encoded, type=EngineCoreOutputs)
    assert decoded.engine_notifications == [event]


def test_proc_broadcasts_to_every_frontend():
    """Every API server gets the event, not just one."""
    proc = EngineCoreProc.__new__(EngineCoreProc)
    proc.addresses = SimpleNamespace(outputs=["a", "b", "c"])
    proc.output_queue = queue.Queue()

    event = CustomNotification(key="my_plugin")
    proc._publish_notifications([event])

    delivered = [proc.output_queue.get_nowait() for _ in range(3)]
    assert [client_index for client_index, _ in delivered] == [0, 1, 2]
    assert all(out.engine_notifications == [event] for _, out in delivered)
    assert proc.output_queue.empty()
