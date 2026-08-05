# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the in-process engine-notification buffer.

Hits the real `EngineCore`/`Scheduler` buffer helpers without a model: they
only touch `_pending_notifications`, so a bare instance is enough.
"""

from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.engine import EngineCoreOutputs
from vllm.v1.engine.core import EngineCore
from vllm.v1.notifications import CustomNotification, LoRALoadEvent
from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT


def _bare_engine_core() -> EngineCore:
    engine_core = EngineCore.__new__(EngineCore)
    engine_core._pending_notifications = []
    return engine_core


def _bare_scheduler() -> Scheduler:
    scheduler = Scheduler.__new__(Scheduler)
    scheduler._pending_notifications = []
    return scheduler


def test_notifications_accumulate_additively():
    """Everything queued before a flush comes out, in order.

    The additive contract: no coalescing (e.g. latest-per-type), or counter
    increments would get lost.
    """
    engine_core = _bare_engine_core()

    first = LoRALoadEvent(gpu_adapters=["a"], cpu_adapters=["a"])
    second = LoRALoadEvent(gpu_adapters=["a", "b"], cpu_adapters=["a", "b"])
    engine_core._publish_notifications([first])
    engine_core._publish_notifications([second])

    outputs: dict[int, EngineCoreOutputs] = {}
    engine_core._flush_notifications(outputs)

    assert outputs[0].engine_notifications == [first, second]


def test_flush_clears_buffer_between_steps():
    """A flush drains the buffer; later events are independent."""
    engine_core = _bare_engine_core()

    engine_core._publish_notifications([LoRALoadEvent(gpu_adapters=["a"])])
    first_outputs: dict[int, EngineCoreOutputs] = {}
    engine_core._flush_notifications(first_outputs)

    second_outputs: dict[int, EngineCoreOutputs] = {}
    engine_core._flush_notifications(second_outputs)
    assert second_outputs == {}

    later = LoRALoadEvent(gpu_adapters=["b"])
    engine_core._publish_notifications([later])
    third_outputs: dict[int, EngineCoreOutputs] = {}
    engine_core._flush_notifications(third_outputs)
    assert third_outputs[0].engine_notifications == [later]


def test_flush_reuses_existing_outputs():
    """Notifications attach to an existing per-engine outputs entry."""
    engine_core = _bare_engine_core()

    event = LoRALoadEvent(gpu_adapters=["a"])
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


def test_scheduler_publish_take_roundtrip():
    """publish_notification queues; take_notifications drains exactly once."""
    scheduler = _bare_scheduler()

    lora = LoRALoadEvent(gpu_adapters=["a"])
    custom = CustomNotification(key="my_plugin", payload={"count": 1})
    scheduler.publish_notification(lora)
    scheduler.publish_notification(custom)

    assert scheduler.take_notifications() == [lora, custom]
    # Second take is empty, it drained.
    assert scheduler.take_notifications() == []


def test_collect_drains_scheduler_notifications():
    """Scheduler-sourced events flow out through EngineCore too.

    Worker and scheduler notifications share the same channel: scheduler events
    ride alongside worker_notifications.
    """
    engine_core = _bare_engine_core()
    engine_core.scheduler = _bare_scheduler()

    event = CustomNotification(key="my_plugin", payload={"n": 1})
    engine_core.scheduler.publish_notification(event)

    outputs: dict[int, EngineCoreOutputs] = {}
    engine_core._collect_step_notifications(EMPTY_MODEL_RUNNER_OUTPUT, outputs)

    assert outputs[0].engine_notifications == [event]
    # Draining it was a side effect of the collect.
    assert engine_core.scheduler.take_notifications() == []
