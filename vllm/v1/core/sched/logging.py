# SPDX-License-Identifier: Apache-2.0
from vllm.v1.engine import EngineCoreEvent, EngineCoreEventType
from vllm.v1.request import Request


def record_queued(request: Request) -> None:
    request.events.append(EngineCoreEvent.new_event(
        EngineCoreEventType.QUEUED))


def record_scheduled(request: Request, timestamp: float) -> None:
    request.events.append(
        EngineCoreEvent.new_event(EngineCoreEventType.SCHEDULED, timestamp))


def record_preempted(request: Request, timestamp: float) -> None:
    request.events.append(
        EngineCoreEvent.new_event(EngineCoreEventType.PREEMPTED, timestamp))
