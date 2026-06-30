# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Engine-level event notifications.

These are engine-scoped state transitions (as opposed to the per-request
lifecycle events in `EngineCoreEvent` and the per-step sampling in
`SchedulerStats`). Events originate in the worker or engine core and are
forwarded to frontends on `EngineCoreOutputs.engine_notifications`. They are
typically infrequent (LoRA load/unload), but the channel does not assume so:
a producer may emit on every step.

Each event is a msgspec Struct with an explicit `tag`, so the union is
encoded as a map with a `"type"` discriminator field. Like the rest of
`EngineCoreOutputs`, the union is version-lockstep between engine and
frontend: an unknown tag is a deployment error and fails the decode loudly
rather than being skipped. The Rust frontend
(`rust/src/engine-core-client/src/protocol/notifications.rs`) mirrors the
union and the same fail-fast behavior.

Out-of-tree producers (plugins) can't add their own tags without forking the
protocol, so the union reserves a single open `CustomNotification` tag: a
plugin namespaces its event under `key` and carries an arbitrary payload.
The tag is known to every frontend (so it never trips fail-fast), but an
unrecognized `key` is ignored rather than fatal.

Channel contract: notifications accumulate additively (like `SchedulerStats`).
The engine delivers every queued event in emission order and never drops one,
so a notification may carry an absolute snapshot (consumers replace prior
state, as `LoRALoadEvent` does) or an incremental delta (consumers apply each
event, e.g. a counter increment). Throttling redundant events is the
producer's job: `LoRALoadEvent` only emits when the loaded set changed.
"""

from typing import Any

import msgspec


class LoRALoadEvent(
    msgspec.Struct,
    tag="lora_load_event",
    omit_defaults=True,  # type: ignore[call-arg]
    gc=False,
):  # type: ignore[call-arg]
    """The set of loaded LoRA adapters changed.

    Emitted only when the loaded set changed (on a step, or directly on
    dynamic adapter load/unload). Each event carries the complete current
    state, so consumers should replace (not merge with) the previously
    observed snapshot.

    Adapters are identified by name. If an adapter id is reused for a
    different adapter (which `LoRARequest` forbids but does not enforce),
    the change detector may miss the swap until the next load event.
    """

    gpu_adapters: list[str] = []
    """Names of adapters currently activated into GPU slots (sorted)."""

    cpu_adapters: list[str] = []
    """Names of adapters resident in the CPU cache (sorted). This is a
    superset of `gpu_adapters`."""

    pinned_adapters: list[str] = []
    """Names of adapters pinned in the caches (sorted)."""


class CustomNotification(
    msgspec.Struct,
    tag="custom",
    omit_defaults=True,  # type: ignore[call-arg]
    gc=False,
):  # type: ignore[call-arg]
    """An open-schema notification for out-of-tree producers (plugins).

    The in-tree union fails fast on unknown tags, so a plugin can't add its
    own struct type without forking the protocol. This reserved tag is the
    escape hatch: emit a `CustomNotification`, namespace it under `key`, and
    carry whatever the consumer needs in `payload`. Frontends that don't
    recognize `key` ignore the event (per the channel's additive contract,
    deltas a consumer doesn't apply are simply no-ops).
    """

    key: str
    """Producer-chosen namespace (e.g. the plugin name). Consumers match on
    this to decide whether and how to interpret `payload`."""

    payload: dict[str, Any] = {}
    """Arbitrary msgpack-encodable event data, opaque to the engine and to
    consumers that don't recognize `key`."""


# Union of all engine-level event types. Grows as new event types are added
# (e.g. graceful shutdown progress); msgspec dispatches on each struct's tag.
EngineNotification = LoRALoadEvent | CustomNotification
