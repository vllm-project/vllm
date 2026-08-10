# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
import time
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from vllm.config.scheduler import SchedulerPluginProfile, SchedulerPluginSpec
from vllm.v1.core.sched.plugins.builtin import (
    FCFSSchedulerPlugin,
    PrioritySchedulerPlugin,
)
from vllm.v1.core.sched.plugins.interface import (
    CandidateInfo,
    CandidateSelection,
    FilterPlugin,
    PreemptionPlugin,
    QueueSortPlugin,
    SchedulerPlugin,
    SchedulingCycleState,
    ScorePlugin,
    WaitingQueue,
)
from vllm.v1.core.sched.request_queue import RequestQueue, SchedulingPolicy
from vllm.v1.request import RequestStatus

if TYPE_CHECKING:
    from collections.abc import Sequence

    from vllm.v1.request import Request


BUILTIN_SCHEDULER_PLUGINS: dict[str, type[SchedulerPlugin]] = {
    FCFSSchedulerPlugin.name: FCFSSchedulerPlugin,
    PrioritySchedulerPlugin.name: PrioritySchedulerPlugin,
}


def register_scheduler_plugin(plugin_cls: type[SchedulerPlugin]) -> None:
    """Register a scheduler plugin class."""
    name = plugin_cls.name
    if name in BUILTIN_SCHEDULER_PLUGINS:
        raise ValueError(f"Scheduler plugin is already registered: {name}")
    BUILTIN_SCHEDULER_PLUGINS[name] = plugin_cls


class SchedulerPluginManager:
    """Runs scheduler extension points for one scheduler instance."""

    def __init__(
        self,
        policy: str,
        profile: SchedulerPluginProfile | None = None,
    ) -> None:
        self.policy = SchedulingPolicy(policy)
        self.profile = profile or SchedulerPluginProfile()
        if self.profile.queue_sort is not None and policy != "fcfs":
            raise ValueError(
                "An explicit QueueSort plugin cannot be combined with a "
                f"non-default scheduling policy: {policy}"
            )

        self._plugins_by_name: dict[str, tuple[SchedulerPlugin, dict[str, Any]]] = {}
        queue_sort_spec = self.profile.queue_sort or SchedulerPluginSpec(name=policy)
        queue_sort_plugin = self._get_or_create_plugin(queue_sort_spec)
        if not isinstance(queue_sort_plugin, QueueSortPlugin):
            raise TypeError(
                f"Scheduler plugin {queue_sort_spec.name!r} does not implement "
                "QueueSort"
            )
        self.queue_sort_plugin = queue_sort_plugin

        preemption_spec = self.profile.preemption or SchedulerPluginSpec(name=policy)
        preemption_plugin = self._get_or_create_plugin(preemption_spec)
        if not isinstance(preemption_plugin, PreemptionPlugin):
            raise TypeError(
                f"Scheduler plugin {preemption_spec.name!r} does not implement "
                "Preemption"
            )
        self.preemption_plugin = preemption_plugin

        self.filter_plugins = tuple(
            self._require_filter_plugin(spec) for spec in self.profile.filters
        )
        self.score_plugins = tuple(
            (self._require_score_plugin(spec), spec.weight)
            for spec in self.profile.scores
        )
        self.plugins = tuple(plugin for plugin, _ in self._plugins_by_name.values())
        self.candidate_window = self.profile.candidate_window
        self.has_candidate_plugins = bool(self.filter_plugins or self.score_plugins)

    def _get_or_create_plugin(self, spec: SchedulerPluginSpec) -> SchedulerPlugin:
        existing = self._plugins_by_name.get(spec.name)
        if existing is not None:
            plugin, args = existing
            if args != spec.args:
                raise ValueError(
                    f"Scheduler plugin {spec.name!r} is configured with "
                    "different arguments at multiple extension points"
                )
            return plugin

        plugin_cls = BUILTIN_SCHEDULER_PLUGINS.get(spec.name)
        if plugin_cls is None:
            raise ValueError(f"Unknown scheduler plugin: {spec.name}")
        plugin = plugin_cls(**spec.args)
        self._plugins_by_name[spec.name] = (plugin, spec.args)
        return plugin

    def _require_filter_plugin(self, spec: SchedulerPluginSpec) -> FilterPlugin:
        plugin = self._get_or_create_plugin(spec)
        if not isinstance(plugin, FilterPlugin):
            raise TypeError(f"Scheduler plugin {spec.name!r} does not implement Filter")
        return plugin

    def _require_score_plugin(self, spec: SchedulerPluginSpec) -> ScorePlugin:
        plugin = self._get_or_create_plugin(spec)
        if not isinstance(plugin, ScorePlugin):
            raise TypeError(f"Scheduler plugin {spec.name!r} does not implement Score")
        return plugin

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

    def select_candidate(
        self,
        waiting: RequestQueue,
        skipped: RequestQueue,
        *,
        block_size: int,
        token_budget: int,
        encoder_budget: int,
        num_running_requests: int,
        now: float | None = None,
    ) -> CandidateSelection | None:
        """Select one candidate without mutating either request queue."""
        if not self.has_candidate_plugins:
            queue = self.select_queue(waiting, skipped)
            if queue is None:
                return None
            queue_name = (
                WaitingQueue.WAITING if queue is waiting else WaitingQueue.SKIPPED
            )
            return CandidateSelection(queue.peek_request(), queue_name)

        now = time.time() if now is None else now
        ordered = self.queue_sort_plugin.order_candidates(
            tuple(waiting), tuple(skipped)
        )[: self.candidate_window]
        if not ordered:
            return None
        if ordered[0].request.status == RequestStatus.PREEMPTED:
            ordered = ordered[:1]

        candidate_info = {
            selection.request.request_id: CandidateInfo(
                queue=selection.queue,
                queue_position=position,
                waiting_time=max(0.0, now - selection.request.arrival_time),
            )
            for position, selection in enumerate(ordered)
        }
        state = SchedulingCycleState(
            now=now,
            block_size=block_size,
            token_budget=token_budget,
            encoder_budget=encoder_budget,
            num_running_requests=num_running_requests,
            candidates=MappingProxyType(candidate_info),
        )

        best: CandidateSelection | None = None
        best_score = -math.inf
        for selection in ordered:
            request = selection.request
            if any(
                not plugin.filter(request, state).allowed
                for plugin in self.filter_plugins
            ):
                continue
            score = sum(
                plugin.score(request, state) * weight
                for plugin, weight in self.score_plugins
            )
            if not math.isfinite(score):
                raise ValueError(
                    f"Scheduler plugins returned a non-finite score for "
                    f"request {request.request_id!r}: {score}"
                )
            if best is None or score > best_score:
                best = selection
                best_score = score
        return best

    def select_preemption_victim(
        self,
        running: "Sequence[Request]",
    ) -> "Request":
        _, victim = max(
            enumerate(running),
            key=lambda item: self.preemption_plugin.preemption_key(item[1], item[0]),
        )
        return victim
