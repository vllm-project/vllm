# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""RFC #59708 primitive contracts; no allocator or connector dependencies."""

import math
from typing import Literal, TypeAlias

import msgspec

from vllm.v1.kv_hints.native import Identifier, Target, Tier, TokenPosition

PRIMITIVE_VERSION = "1.0"
PrimitiveTarget: TypeAlias = Target | Literal["request_cached"]


class CacheLocation(msgspec.Struct, frozen=True, forbid_unknown_fields=True):  # type: ignore[call-arg]
    tier: Tier
    engine_id: Identifier | None = None


class SetPriority(msgspec.Struct, frozen=True, forbid_unknown_fields=True):  # type: ignore[call-arg]
    """Copy-scoped claim contract aligned with PR #59714's SetPriority."""

    target: PrimitiveTarget
    location: Literal["local_g1"]
    claim_id: Identifier
    revision: TokenPosition
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


class CopyTo(msgspec.Struct, frozen=True, forbid_unknown_fields=True):  # type: ignore[call-arg]
    target: PrimitiveTarget
    src: CacheLocation
    dst: CacheLocation


class MoveTo(msgspec.Struct, frozen=True, forbid_unknown_fields=True):  # type: ignore[call-arg]
    target: PrimitiveTarget
    src: CacheLocation
    dst: CacheLocation


Primitive: TypeAlias = SetPriority | CopyTo | MoveTo
PRIMITIVES: dict[str, type[Primitive]] = {
    "vllm.set_priority": SetPriority,
    "vllm.copy_to": CopyTo,
    "vllm.move_to": MoveTo,
}
