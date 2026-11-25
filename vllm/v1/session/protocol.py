# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Messages between the session resolver (content) and the KV state store
(physical memory).

The resolver only hands out references to content it has compared token by
token; the store only checks that a referenced state still exists. Every
message is plain data so the resolver can later run in another process.
"""

from dataclasses import dataclass
from typing import NamedTuple


class NodeId(NamedTuple):
    """Identity of an immutable span of KV content. `origin` is drawn at
    random whenever a store starts or resets, so an id is never reused across
    stores, restarts or resets, and a node id with a length always names the
    same tokens wherever it is seen (another engine, a KV connector, a
    router's index). `local` counts nodes within one origin."""

    origin: int
    local: int


@dataclass(frozen=True)
class Candidate:
    """The first `length` tokens of the request equal the path ending in
    `node_id`."""

    node_id: NodeId
    length: int


@dataclass(frozen=True)
class SessionBinding:
    epoch: int
    session_id: int
    # Longest first. The store takes the first one it can still resume from.
    candidates: tuple[Candidate, ...]
    # Last store event the resolver had applied when it produced this binding.
    seq: int
    # Longest prefix this request shares with any state, resumable or not.
    match_length: int = 0


@dataclass(frozen=True)
class NodePublished:
    seq: int
    node_id: NodeId
    parent_id: NodeId | None
    # The node holds path positions [start, end); `start` is the position on
    # the parent's path where it attaches.
    start: int
    end: int
    # Packed int32 tokens [start, end).
    tokens: bytes
    # Positions a request may resume at. None means any position in
    # (start, end] (attention-only models).
    commit_points: tuple[int, ...] | None
    request_id: str


@dataclass(frozen=True)
class NodeExtended:
    """The publishing request computed more tokens: `[start, end)` now follow
    the node's previous end `start`. Earlier content is unchanged. A hybrid
    node becomes resumable when it is sealed with `commit_points`."""

    seq: int
    node_id: NodeId
    start: int
    end: int
    tokens: bytes
    commit_points: tuple[int, ...] | None = None


@dataclass(frozen=True)
class NodeTruncated:
    seq: int
    node_id: NodeId
    # New exclusive end; `end <= start` means the node was removed.
    end: int
    commit_points: tuple[int, ...] | None


@dataclass(frozen=True)
class RequestFinished:
    seq: int
    request_id: str
    # The node ending at the request's final committed position, or None if
    # the request committed nothing (aborted, failed, or preempted for good).
    node_id: NodeId | None


@dataclass(frozen=True)
class StoreReset:
    seq: int
    epoch: int


StoreEvent = NodePublished | NodeExtended | NodeTruncated | RequestFinished | StoreReset
