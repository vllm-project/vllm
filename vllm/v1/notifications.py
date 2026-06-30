# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Engine-level event notifications.

Engine-scoped state transitions (unlike per-request `EngineCoreEvent` or
per-step `SchedulerStats`), forwarded to frontends on
`EngineCoreOutputs.engine_notifications`. A producer fires one whenever its
state changes, e.g. `LoRALoadEvent` on a LoRA load/unload.

Each event is a tagged msgspec Struct, map-encoded with a `"type"` field. The
union is version-lockstep with the engine like the rest of `EngineCoreOutputs`:
an unknown tag fails the decode loudly instead of getting skipped. The Rust
frontend mirrors it. Plugins that can't add a tag use the open `custom` one.

Events are additive (like `SchedulerStats`): everything queued gets delivered
in order, nothing dropped. So an event can be a full snapshot (consumer
replaces its state) or a delta (consumer applies each one). Producers throttle
themselves; `LoRALoadEvent` only fires when the loaded set actually changes.
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

    Carries the full current state, so consumers replace their snapshot rather
    than merge. Adapters are keyed by name; if an id gets reused for a
    different adapter (which `LoRARequest` forbids but doesn't enforce), the
    differ can miss the swap until the next event.
    """

    gpu_adapters: list[str] = []
    """Adapters activated into GPU slots (sorted)."""

    cpu_adapters: list[str] = []
    """Adapters resident in the CPU cache, a superset of `gpu_adapters`."""

    pinned_adapters: list[str] = []
    """Adapters pinned in the caches (sorted)."""


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
EngineNotification = LoRALoadEvent | CustomNotification
