# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
from dataclasses import dataclass
from typing import Annotated, Literal

import msgspec


class SetPriority(  # type: ignore[call-arg]
    msgspec.Struct, frozen=True, forbid_unknown_fields=True
):
    """Declare priority on this request's cached copies at successful completion."""

    target: Literal["request_cached"]
    location: Literal["local_g1"]
    claim_id: Annotated[str, msgspec.Meta(min_length=1, max_length=256)]
    revision: Annotated[int, msgspec.Meta(ge=0)]
    value: int | None
    ttl_seconds: float | None = None

    def __post_init__(self) -> None:
        if self.value is None:
            if self.ttl_seconds is not None:
                raise ValueError("Clearing a claim must omit ttl_seconds")
        elif (
            self.ttl_seconds is None
            or not math.isfinite(self.ttl_seconds)
            or self.ttl_seconds <= 0
        ):
            raise ValueError("Setting a priority requires a finite positive TTL")


class KVLocation(  # type: ignore[call-arg]
    msgspec.Struct, frozen=True, forbid_unknown_fields=True
):
    """One side of a COPY_TO/MOVE_TO: a storage tier, optionally remote."""

    tier: Literal["gpu", "cpu"]
    # Omitted means the explicitly addressed engine (the receiving one).
    engine_id: str | None = None


class CopyTo(  # type: ignore[call-arg]
    msgspec.Struct, frozen=True, forbid_unknown_fields=True
):
    """Load offloaded copies of the target prefix into local GPU (G1).

    `target` lists one GPU block-hash hex string per GPU block of the prefix,
    in order from the start - the same hashes the engine publishes in its
    BlockStored KV events. Scope is captured at submission time. Only local
    CPU -> GPU copies are supported; the transfer runs request-free and its
    completion is reported through the normal KV residency events, not
    through this action's result.
    """

    target: list[str]
    src: KVLocation
    dst: KVLocation


@dataclass(frozen=True)
class KvHintResult:
    """Executor-local outcome; block IDs describe the scope at application time."""

    status: Literal["applied", "deferred", "duplicate", "rejected", "unsupported"]
    block_ids: tuple[int, ...] = ()
    reason: str | None = None
