# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""LoRA adapter residency, reported from the worker to frontends.

LoRA state on a worker changes only inside an engine step or inside an RPC the
engine core issued, so the engine core picks up the snapshot at those two
points: from the output rank's `ModelRunnerOutput`, and right after the RPC
returns. It then forwards it on `EngineCoreOutputs.lora_load_event`. The Rust
frontend mirrors these structs.
"""

import msgspec


class LoRALoadTiming(
    msgspec.Struct,
    omit_defaults=True,  # type: ignore[call-arg]
    gc=False,
):
    """One measured adapter transition on the worker."""

    adapter_name: str
    transition: str
    """`load`: read from disk into the CPU cache. `activate`: moved from the
    CPU cache into a GPU slot."""
    seconds: float


class LoRALoadEvent(
    msgspec.Struct,
    omit_defaults=True,  # type: ignore[call-arg]
    gc=False,
):
    """The set of loaded LoRA adapters changed, or an adapter transition
    completed.

    A full snapshot of the worker's adapter caches, so consumers replace their
    state rather than merge. Adapters are keyed by name. `loads` carries the
    transitions measured since the previous event.
    """

    gpu_adapters: list[str] = []
    """Adapters activated into GPU slots (sorted)."""

    cpu_adapters: list[str] = []
    """Adapters resident in the CPU cache (sorted, superset of gpu_adapters)."""

    pinned_adapters: list[str] = []
    """Adapters pinned in the caches (sorted)."""

    ranks: dict[str, int] = {}
    """LoRA rank of each resident adapter, keyed by name."""

    loads: list[LoRALoadTiming] = []
    """Adapter transitions completed since the previous event, in order."""
