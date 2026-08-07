# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Engine-level event notifications.

Engine-scoped state transitions (unlike per-request `EngineCoreEvent` or
per-step `SchedulerStats`), forwarded to frontends on
`EngineCoreOutputs.engine_notifications`. A producer fires one whenever its
state changes.

Each event is a tagged msgspec Struct, map-encoded with a `"type"` field. The
union is version-lockstep with the engine like the rest of `EngineCoreOutputs`:
an unknown tag fails the decode loudly instead of getting skipped. The Rust
frontend mirrors it. Plugins that can't add a tag use the open `custom` one.

Events are additive (like `SchedulerStats`): everything queued gets delivered
in order, nothing dropped. So an event can be a full snapshot (consumer
replaces its state) or a delta (consumer applies each one). Producers throttle
themselves.

Worker-side producers publish into a process-local buffer, which the engine
core drains from every rank. A weight-loader plugin reporting per-rank
transfer throughput, for example:

    from vllm.v1.notifications import (
        CustomNotification,
        publish_worker_notification,
    )

    def on_weights_loaded(rank, elapsed_s, num_bytes):
        publish_worker_notification(
            CustomNotification(
                key="my_loader",
                payload={
                    "rank": rank,
                    "gbps": num_bytes * 8 / elapsed_s / 1e9,
                },
            )
        )

Rank identity belongs in the payload: the gather flattens every rank's events
into one list, so an event that does not say where it came from cannot be told
apart from its peers.

The frontend side is a stat logger, which sees the events from every rank:

    class MyLoaderStatLogger(StatLoggerBase):
        def record(self, *, engine_notifications=None, **kwargs):
            for event in engine_notifications or ():
                if isinstance(event, CustomNotification) and event.key == "my_loader":
                    self.gbps.labels(rank=event.payload["rank"]).set(
                        event.payload["gbps"]
                    )

Delivery: producers that publish during model load are gathered once before
the engine serves, and in-tree producers gather on their own state changes.
A producer that publishes at arbitrary later times needs
`VLLM_WORKER_NOTIFICATION_POLL_INTERVAL` set, since nothing else prompts a
gather on its behalf.
"""

from typing import Any

import msgspec


class CustomNotification(
    msgspec.Struct,
    tag="custom",
    omit_defaults=True,  # type: ignore[call-arg]
):
    """Open escape hatch for out-of-tree producers (plugins).

    The union fails fast on unknown tags, so plugins can't add their own struct
    type. Instead they emit this: namespace under `key`, and place event data
    in `payload`. Frontends that don't know the `key` just ignore it.

    Left GC-tracked, unlike its siblings: `payload` is arbitrary
    plugin-supplied data, and a cycle routed through an untracked struct would
    never be collected.
    """

    key: str
    """Producer-chosen namespace, e.g. the plugin name."""

    payload: dict[str, Any] = {}
    """Arbitrary msgpack-encodable event data, opaque to anyone who doesn't
    know `key`."""


# All engine-level event types; msgspec dispatches on each struct's tag.
EngineNotification = CustomNotification


# Process-local: worker-side producers get no handle to the runner. A model
# loader or platform plugin runs several frames below load_model().
_worker_notifications: list[EngineNotification] = []


def publish_worker_notification(notification: EngineNotification) -> None:
    """Queue a notification from inside the worker process.

    The model runner forwards it on the next `ModelRunnerOutput`. Publishing
    before the first step is fine, which is the point for producers that only
    run during model load. Additive and unbounded; producers throttle
    themselves.
    """
    _worker_notifications.append(notification)


def take_worker_notifications() -> list[EngineNotification] | None:
    """Drain everything queued in this worker process since the last call.

    Returns None rather than an empty list so the common case (no producer
    installed) allocates nothing on the per-step path.
    """
    global _worker_notifications
    if not _worker_notifications:
        return None
    pending = _worker_notifications
    _worker_notifications = []
    return pending
