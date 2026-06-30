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

Channel contract: notifications accumulate additively (like `SchedulerStats`).
The engine delivers every queued event in emission order and never drops one,
so a notification may carry an absolute snapshot (consumers replace prior
state, as `LoRALoadEvent` does) or an incremental delta (consumers apply each
event, e.g. a counter increment). Throttling redundant events is the
producer's job: `LoRALoadEvent` only emits when the loaded set changed.
"""

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


# Union of all engine-level event types. Grows as new event types are added
# (e.g. graceful shutdown progress); msgspec dispatches on each struct's tag.
EngineNotification = LoRALoadEvent
