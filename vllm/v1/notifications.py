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
"""

from typing import Any

import msgspec


class CustomNotification(
    msgspec.Struct,
    tag="custom",
    omit_defaults=True,  # type: ignore[call-arg]
    gc=False,
):  # type: ignore[call-arg]
    """Open escape hatch for out-of-tree producers (plugins).

    The union fails fast on unknown tags, so plugins can't add their own struct
    type. Instead they emit this: namespace under `key`, and place event data
    in `payload`. Frontends that don't know the `key` just ignore it.
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


def take_worker_notifications() -> list[EngineNotification]:
    """Drain everything queued in this worker process since the last call."""
    global _worker_notifications
    pending = _worker_notifications
    _worker_notifications = []
    return pending
