# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TYPE_CHECKING

from vllm.v1.core.sched.plugins.builtin import (
    FCFSSchedulerPlugin,
    PrioritySchedulerPlugin,
)
from vllm.v1.core.sched.plugins.interface import (
    PreemptionPlugin,
    QueueSortPlugin,
    SchedulerPlugin,
    WaitingQueue,
)
from vllm.v1.core.sched.request_queue import RequestQueue, SchedulingPolicy

if TYPE_CHECKING:
    from collections.abc import Sequence

    from vllm.v1.request import Request


BUILTIN_SCHEDULER_PLUGINS: dict[str, type[SchedulerPlugin]] = {
    FCFSSchedulerPlugin.name: FCFSSchedulerPlugin,
    PrioritySchedulerPlugin.name: PrioritySchedulerPlugin,
}


class SchedulerPluginManager:
    """Runs scheduler extension points for one scheduler instance."""

    def __init__(self, policy: str) -> None:
        plugin_cls = BUILTIN_SCHEDULER_PLUGINS.get(policy)
        if plugin_cls is None:
            raise ValueError(f"Unknown scheduling policy: {policy}")
        plugin = plugin_cls()
        assert isinstance(plugin, QueueSortPlugin)
        assert isinstance(plugin, PreemptionPlugin)
        self.plugins = (plugin,)
        self.queue_sort_plugin = plugin
        self.preemption_plugin = plugin
        self.policy = SchedulingPolicy(policy)

    def create_request_queue(self) -> RequestQueue:
        return self.queue_sort_plugin.create_request_queue()

    def select_queue(
        self,
        waiting: RequestQueue,
        skipped: RequestQueue,
    ) -> RequestQueue | None:
        waiting_head = waiting.peek_request() if waiting else None
        skipped_head = skipped.peek_request() if skipped else None
        selected = self.queue_sort_plugin.select_queue(waiting_head, skipped_head)
        if selected == WaitingQueue.WAITING:
            return waiting
        if selected == WaitingQueue.SKIPPED:
            return skipped
        return None

    def select_preemption_victim(
        self,
        running: "Sequence[Request]",
    ) -> "Request":
        _, victim = max(
            enumerate(running),
            key=lambda item: self.preemption_plugin.preemption_key(item[1], item[0]),
        )
        return victim
