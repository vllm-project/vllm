# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Payload schemas for vLLM's native consumer of the merged hint envelope."""

import math
from typing import Annotated, Any, Literal, TypeAlias

import msgspec

from vllm.v1.kv_hints.control import ControlError
from vllm.v1.kv_hints.protocol import KvHintAction

Identifier: TypeAlias = Annotated[str, msgspec.Meta(min_length=1, max_length=256)]
TokenPosition: TypeAlias = Annotated[int, msgspec.Meta(ge=0)]
PositiveSeconds: TypeAlias = Annotated[float, msgspec.Meta(gt=0)]
Tier: TypeAlias = Literal["gpu", "cpu", "mooncake"]

NATIVE_PROTOCOL_VERSION = "0.1"
NATIVE_ACTION_VERSION = "1"


class ContentSpan(msgspec.Struct, frozen=True, forbid_unknown_fields=True):  # type: ignore[call-arg]
    """Logical group content; no allocator block IDs or residency claim."""

    block_hash: Annotated[str, msgspec.Meta(pattern="^[0-9a-f]{2,128}$")]
    group_id: Annotated[int, msgspec.Meta(ge=0, le=(1 << 32) - 1)]
    start_token: TokenPosition
    end_token: TokenPosition


class HashTarget(  # type: ignore[call-arg]
    msgspec.Struct,
    tag="hashes",
    tag_field="kind",
    frozen=True,
    forbid_unknown_fields=True,
):
    namespace: Identifier
    blocks: Annotated[tuple[ContentSpan, ...], msgspec.Meta(max_length=16384)]
    start_token: TokenPosition = 0
    end_token: TokenPosition | None = None


class SessionTarget(  # type: ignore[call-arg]
    msgspec.Struct,
    tag="session",
    tag_field="kind",
    frozen=True,
    forbid_unknown_fields=True,
):
    session_id: Identifier
    continuation_id: Identifier | None = None
    start_token: TokenPosition = 0
    end_token: TokenPosition | None = None


class RequestTarget(  # type: ignore[call-arg]
    msgspec.Struct,
    tag="request",
    tag_field="kind",
    frozen=True,
    forbid_unknown_fields=True,
):
    request_id: Identifier
    start_token: TokenPosition = 0
    end_token: TokenPosition | None = None


Target: TypeAlias = HashTarget | SessionTarget | RequestTarget


class RetainPayload(msgspec.Struct, frozen=True, forbid_unknown_fields=True):  # type: ignore[call-arg]
    target: Target
    owner_id: Identifier
    declaration_id: Identifier
    priority: Annotated[int, msgspec.Meta(ge=0, le=100)] = 0
    ttl_seconds: PositiveSeconds | None = None
    tier: Tier = "gpu"
    scope: Literal["snapshot", "follow_growth"] = "snapshot"


class ReleasePayload(msgspec.Struct, frozen=True, forbid_unknown_fields=True):  # type: ignore[call-arg]
    owner_id: Identifier
    declaration_id: Identifier


class EvictPayload(msgspec.Struct, frozen=True, forbid_unknown_fields=True):  # type: ignore[call-arg]
    target: Target
    tier: Tier = "gpu"


class DemotePayload(msgspec.Struct, frozen=True, forbid_unknown_fields=True):  # type: ignore[call-arg]
    target: Target
    destination: Literal["cpu", "mooncake"]


class PrefetchPayload(msgspec.Struct, frozen=True, forbid_unknown_fields=True):  # type: ignore[call-arg]
    target: Target
    source: Literal["cpu", "mooncake"]


class SharePayload(msgspec.Struct, frozen=True, forbid_unknown_fields=True):  # type: ignore[call-arg]
    target: Target
    destination_engine_id: Identifier


class StorePayload(msgspec.Struct, frozen=True):  # type: ignore[call-arg]
    """Internal lowering of COPY from G1; not a legacy wire action."""

    target: Target
    destination: Literal["cpu", "mooncake"]


class ParentPrefix(msgspec.Struct, frozen=True, forbid_unknown_fields=True):  # type: ignore[call-arg]
    session_id: Identifier
    continuation_id: Identifier
    end_token: TokenPosition


class AssociatePayload(msgspec.Struct, frozen=True, forbid_unknown_fields=True):  # type: ignore[call-arg]
    session_id: Identifier
    continuation_id: Identifier
    expected_revision: TokenPosition
    target: HashTarget | RequestTarget
    task_parent_id: Identifier | None = None
    parent_prefix: ParentPrefix | None = None
    observe_growth: bool = False


class FinishTaskPayload(msgspec.Struct, frozen=True, forbid_unknown_fields=True):  # type: ignore[call-arg]
    session_id: Identifier
    expected_revision: TokenPosition


class SignalPayload(msgspec.Struct, frozen=True, forbid_unknown_fields=True):  # type: ignore[call-arg]
    target: Target
    signal: Literal["tool_wait", "resume", "importance", "resume_prediction"]
    importance: Annotated[int, msgspec.Meta(ge=0, le=100)] | None = None
    resume_after_seconds: PositiveSeconds | None = None


NativePayload: TypeAlias = (
    RetainPayload
    | ReleasePayload
    | EvictPayload
    | DemotePayload
    | PrefetchPayload
    | SharePayload
    | AssociatePayload
    | FinishTaskPayload
    | SignalPayload
    | StorePayload
)

# These are built-in semantics, not a user-defined hint-handler registry.
_PAYLOAD_TYPES: dict[str, type[NativePayload]] = {
    "retain": RetainPayload,
    "release": ReleasePayload,
    "evict": EvictPayload,
    "demote": DemotePayload,
    "prefetch": PrefetchPayload,
    "share": SharePayload,
    "associate": AssociatePayload,
    "finish_task": FinishTaskPayload,
    "signal": SignalPayload,
}


class ActionCapability(msgspec.Struct, frozen=True):  # type: ignore[call-arg]
    action_type: str
    action_version: str
    targets: tuple[Literal["hashes", "session", "request", "request_cached"], ...]
    tiers: tuple[Tier, ...]
    request_free: bool
    limitations: tuple[str, ...] = ()
    directions: tuple[str, ...] = ()


class ConsumerCapabilities(msgspec.Struct, frozen=True):  # type: ignore[call-arg]
    engine_id: str
    engine_epoch: str
    consumer: Literal["native", "backend"]
    protocol_versions: tuple[str, ...]
    actions: tuple[ActionCapability, ...]
    namespace: str
    result_ttl_seconds: float
    max_actions: int
    max_envelope_bytes: int
    cache_profile: dict[str, Any] | None = None


def validate_target(target: Target) -> None:
    if target.end_token is not None and target.end_token <= target.start_token:
        raise ControlError("invalid_payload", "Target end must be greater than start")
    if isinstance(target, HashTarget):
        seen: set[tuple[int, int, int]] = set()
        for block in target.blocks:
            if len(block.block_hash) % 2:
                raise ControlError(
                    "invalid_payload", "Block hash must encode whole bytes"
                )
            if block.end_token <= block.start_token:
                raise ControlError("invalid_payload", "Content span must be nonempty")
            position = (block.group_id, block.start_token, block.end_token)
            if position in seen:
                raise ControlError("invalid_payload", "Duplicate content position")
            seen.add(position)
        per_group: dict[int, list[ContentSpan]] = {}
        for block in target.blocks:
            per_group.setdefault(block.group_id, []).append(block)
        for blocks in per_group.values():
            ordered = sorted(
                blocks, key=lambda block: (block.start_token, block.end_token)
            )
            if any(
                a.end_token > b.start_token and a.start_token != b.start_token
                for a, b in zip(ordered, ordered[1:])
            ):
                raise ControlError(
                    "invalid_payload", "Content spans overlap in a group"
                )


def parse_native_action(action: KvHintAction) -> NativePayload:
    """Validate a native payload without resolving content or approving resources."""
    payload_type = _PAYLOAD_TYPES.get(action.action_type)
    if payload_type is None:
        raise ControlError(
            "unsupported_action", f"Unknown native action {action.action_type}"
        )
    if action.action_version != NATIVE_ACTION_VERSION:
        raise ControlError("unsupported_version", "Unsupported native action version")
    try:
        payload = msgspec.convert(action.payload, type=payload_type, strict=True)
    except msgspec.ValidationError as error:
        raise ControlError("invalid_payload", str(error)) from error
    if isinstance(
        payload,
        (
            RetainPayload,
            EvictPayload,
            DemotePayload,
            PrefetchPayload,
            SharePayload,
            AssociatePayload,
            SignalPayload,
        ),
    ):
        validate_target(payload.target)
    if isinstance(payload, RetainPayload):
        if payload.ttl_seconds is not None and not math.isfinite(payload.ttl_seconds):
            raise ControlError("invalid_payload", "Retention TTL must be finite")
        if payload.scope == "follow_growth" and not (
            isinstance(payload.target, SessionTarget)
            and payload.target.continuation_id is not None
        ):
            raise ControlError(
                "invalid_payload", "Growth requires an explicit continuation"
            )
    if (
        isinstance(payload, AssociatePayload)
        and payload.observe_growth
        and not isinstance(payload.target, RequestTarget)
    ):
        raise ControlError(
            "invalid_payload", "Request growth requires a request target"
        )
    if (
        isinstance(payload, SignalPayload)
        and payload.resume_after_seconds is not None
        and not math.isfinite(payload.resume_after_seconds)
    ):
        raise ControlError("invalid_payload", "Resume prediction must be finite")
    if (
        isinstance(payload, AssociatePayload)
        and payload.task_parent_id == payload.session_id
    ):
        raise ControlError("invalid_payload", "A task cannot parent itself")
    return payload
