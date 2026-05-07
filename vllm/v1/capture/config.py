# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Configuration dataclasses for the capture-consumer framework.

``CaptureConsumersConfig`` is wired into ``VllmConfig`` and read by
the runner at engine init.  It enumerates the ordered set of consumer
instances to construct, each identified by a registry ``name`` plus an
optional ``instance_name`` disambiguator and opaque ``params`` dict.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any


@dataclass
class CaptureConsumerSpec:
    """One consumer entry from config."""

    name: str
    """Entry-point name (e.g., ``"filesystem"``)."""

    instance_name: str | None = None
    """Optional unique alias for this consumer instance."""

    params: dict[str, Any] = field(default_factory=dict)
    """Arbitrary key-value parameters forwarded to the consumer factory."""


@dataclass
class CaptureConsumersConfig:
    """Top-level capture-consumers configuration.

    ``consumers`` lists config-driven entries (entry-point name + params).
    ``instances`` carries pre-constructed ``CaptureConsumer`` instances
    passed directly to ``LLM(capture_consumers=[...])``.  Instances ride
    on this config so they survive the ``EngineArgs → VllmConfig`` plumbing
    and reach the runner alongside the dict-form entries.  Only
    ``location='driver'`` instances are permitted (the runner's registry
    enforces this).  Instances don't contribute to ``compute_hash``
    because they're per-run driver-side state, not compile-cache inputs.
    """

    consumers: list[CaptureConsumerSpec]
    instances: list[Any] = field(default_factory=list)

    def compute_hash(self) -> str:
        """Deterministic hash for ``VllmConfig.compute_hash()``."""
        h = hashlib.md5(usedforsecurity=False)
        for spec in self.consumers:
            h.update(spec.name.encode())
            if spec.instance_name:
                h.update(spec.instance_name.encode())
            for k in sorted(spec.params):
                h.update(f"{k}={spec.params[k]}".encode())
        return h.hexdigest()[:16]


def parse_consumer_spec(shorthand: str) -> CaptureConsumerSpec:
    """Parse CLI shorthand: ``'name:key=val,key=val'`` into a spec.

    Simple values only — no commas or equals in values.  Use YAML for
    complex params.

    Raises:
        ValueError: If *shorthand* is empty or a key=value pair is malformed.
    """
    if not shorthand or not shorthand.strip():
        raise ValueError("Consumer spec must not be empty")

    shorthand = shorthand.strip()

    # Split on first ':' — left is name, right is key=val pairs
    if ":" in shorthand:
        name, params_str = shorthand.split(":", 1)
    else:
        name = shorthand
        params_str = ""

    name = name.strip()
    if not name:
        raise ValueError("Consumer spec name must not be empty")

    params: dict[str, Any] = {}
    if params_str.strip():
        for pair in params_str.split(","):
            pair = pair.strip()
            if not pair:
                continue
            if "=" not in pair:
                raise ValueError(
                    f"Malformed key=value pair {pair!r} in consumer spec "
                    f"{shorthand!r}; expected 'key=value'"
                )
            key, value = pair.split("=", 1)
            key = key.strip()
            if not key:
                raise ValueError(f"Empty key in consumer spec {shorthand!r}")
            params[key] = value.strip()

    return CaptureConsumerSpec(name=name, params=params)


def validate_consumer_specs(specs: list[CaptureConsumerSpec]) -> None:
    """Validate consumer specs: non-empty names, unique instance names.

    Raises:
        ValueError: On empty names or duplicate instance names.
    """
    for spec in specs:
        if not spec.name or not spec.name.strip():
            raise ValueError("Consumer spec name must not be empty")

    # Check for duplicate instance names (only among specs that have one)
    instance_names: list[str] = []
    for spec in specs:
        effective_name = spec.instance_name if spec.instance_name else spec.name
        instance_names.append(effective_name)

    seen: set[str] = set()
    for iname in instance_names:
        if iname in seen:
            raise ValueError(
                f"Duplicate consumer instance name {iname!r}; use "
                f"'instance_name' to disambiguate multiple consumers "
                f"with the same entry-point"
            )
        seen.add(iname)
