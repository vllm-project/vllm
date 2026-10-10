# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Engine-level event notifications.

Engine-scoped state transitions, forwarded to frontends on
`EngineCoreOutputs.engine_notifications`. Tagged msgspec structs, map-encoded
with a `"type"` field; an unknown tag fails the decode rather than being
skipped, so the union stays version-lockstep with the engine. The Rust frontend
mirrors it. Plugins use the open `custom` tag. Delivery is additive: everything
queued arrives in emission order.

Worker-side producers publish into a process-local buffer that the engine core
drains from every rank:

    def on_weights_loaded(rank, elapsed_s, num_bytes):
        publish_worker_notification(
            CustomNotification(
                key="my_loader",
                payload={"rank": rank, "gbps": num_bytes * 8 / elapsed_s / 1e9},
            )
        )

Consumers read them off `EngineCoreOutputs.engine_notifications` in the
frontend process. The gather flattens all ranks into one list, so rank
identity has to live in the payload. Producers that publish after model load
need `VLLM_WORKER_NOTIFICATION_POLL_INTERVAL`; nothing else prompts a gather.
"""

import threading
from collections import deque
from typing import Any

import msgspec

from vllm.logger import init_logger

logger = init_logger(__name__)


class CustomNotification(
    msgspec.Struct,
    tag="custom",
    omit_defaults=True,  # type: ignore[call-arg]
):
    """Open escape hatch for out-of-tree producers.

    Plugins cannot add a tag to the union, so they namespace under `key` and
    put event data in `payload`. Frontends ignore keys they do not know. Left
    GC-tracked, unlike its siblings: `payload` is plugin-supplied, and a cycle
    routed through an untracked struct is never collected.

    `payload` must hold string keys and msgpack-encodable values; this is
    checked only when the engine serializes it, not at construction.
    """

    key: str
    payload: dict[str, Any] = {}


EngineNotification = CustomNotification

MAX_BUFFERED_NOTIFICATIONS = 1024

_worker_notifications: deque[EngineNotification] = deque(
    maxlen=MAX_BUFFERED_NOTIFICATIONS
)
_worker_notifications_lock = threading.Lock()


def publish_worker_notification(notification: EngineNotification) -> None:
    """Queue a notification from inside the worker process.

    Safe to call from any thread; producers such as transfer callbacks
    publish concurrently with the engine's drain."""
    with _worker_notifications_lock:
        if len(_worker_notifications) == MAX_BUFFERED_NOTIFICATIONS:
            logger.warning_once(
                "Worker notification buffer is full (%d); dropping the oldest "
                "notification. Set VLLM_WORKER_NOTIFICATION_POLL_INTERVAL so "
                "the engine drains the buffer.",
                MAX_BUFFERED_NOTIFICATIONS,
            )
        _worker_notifications.append(notification)


def take_worker_notifications() -> list[EngineNotification] | None:
    """Drain what this worker process queued since the last call."""
    with _worker_notifications_lock:
        if not _worker_notifications:
            return None
        pending = list(_worker_notifications)
        _worker_notifications.clear()
        return pending
