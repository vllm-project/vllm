# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""LoRA adapter residency, sent from workers to frontends. Mirrored by
`rust/src/engine-core-client/src/protocol/lora.rs`."""

from typing import Literal

import msgspec

LoRATransition = Literal["load", "activate"]


class LoRALoadTiming(
    msgspec.Struct,
    omit_defaults=True,  # type: ignore[call-arg]
    gc=False,
):
    """One measured adapter transition on the worker."""

    adapter_name: str
    transition: LoRATransition
    """`load`: read from disk into the CPU cache. `activate`: moved from the
    CPU cache into a GPU slot."""
    seconds: float


class LoRALoadEvent(
    msgspec.Struct,
    omit_defaults=True,  # type: ignore[call-arg]
    gc=False,
):
    """Full snapshot of a worker's adapter caches, keyed by adapter name."""

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
