# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The ``kv.pin`` KV-hint action: bounded-TTL protection of offloaded chunks.

A harness or orchestrator attaches a ``kv.pin`` action to a request to ask the
CPU offloading tier to keep that request's chunks instead of letting ordinary
eviction reclaim them. The payload is ``{"ttl_seconds": <number>}``. The pin
is soft: it only reorders eviction while it is alive, lapses at its deadline,
and the tier caps how much of its capacity may be pinned at once.

Action-specific schema and execution live with the executor, as RFC #53421
asks. Invalid or unsupported actions are skipped without affecting the other
actions of the same envelope.
"""

import math
from dataclasses import dataclass

from vllm.v1.kv_hints.protocol import KvHintsEnvelope

PIN_ACTION_TYPE = "kv.pin"
PIN_ACTION_VERSION = "1.0"
PIN_ACTION_MAJOR_VERSION = 1
MAX_PIN_TTL_SECONDS = 3600.0

_PIN_FIELDS = frozenset({"ttl_seconds"})


@dataclass(frozen=True, slots=True)
class PinRequest:
    """One accepted ``kv.pin`` action."""

    ttl_seconds: float


@dataclass(frozen=True, slots=True)
class PinHints:
    """The ``kv.pin`` actions of one envelope, split by parse outcome."""

    pins: tuple[PinRequest, ...] = ()
    # Actions with an unsupported action_version.
    unsupported: int = 0
    # Actions with an invalid payload.
    rejected: int = 0


def _major_version(version: str) -> int | None:
    head = version.split(".", 1)[0]
    return int(head) if head.isdigit() else None


def _parse_ttl(payload: dict) -> float | None:
    if set(payload) - _PIN_FIELDS:
        return None
    value = payload.get("ttl_seconds")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        ttl = float(value)
    except (TypeError, OverflowError):
        return None
    if not (math.isfinite(ttl) and 0 < ttl <= MAX_PIN_TTL_SECONDS):
        return None
    return ttl


def parse_pin_hints(envelope: KvHintsEnvelope | None) -> PinHints:
    """Collect the ``kv.pin`` actions of an envelope.

    Args:
        envelope: the request's KV hints, or None.

    Returns:
        The accepted pin requests and the counts of unsupported and invalid
        ``kv.pin`` actions, for outcome reporting.

    """
    if envelope is None:
        return PinHints()
    pins: list[PinRequest] = []
    unsupported = 0
    rejected = 0
    for action in envelope.actions:
        if action.action_type != PIN_ACTION_TYPE:
            continue
        if _major_version(action.action_version) != PIN_ACTION_MAJOR_VERSION:
            unsupported += 1
            continue
        ttl = _parse_ttl(action.payload)
        if ttl is None:
            rejected += 1
            continue
        pins.append(PinRequest(ttl_seconds=ttl))
    return PinHints(
        pins=tuple(pins),
        unsupported=unsupported,
        rejected=rejected,
    )
