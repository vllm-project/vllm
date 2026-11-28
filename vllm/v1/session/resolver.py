# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Content side of session KV reuse.

The resolver mirrors the token content of every state node the KV store has
published in a radix tree, assigns each request to a session, and hands the
store candidate `(node, length)` references whose content it has compared
token by token. It never sees physical blocks. If it lags behind the store it
can only be more optimistic (the store rejects dead references), never wrong.
"""

import enum
import time
from array import array
from collections.abc import Hashable, Sequence
from dataclasses import dataclass, field

from vllm.v1.request import Request
from vllm.v1.session.protocol import (
    Candidate,
    NodeExtended,
    NodeId,
    NodePublished,
    NodeTruncated,
    RequestFinished,
    SessionBinding,
    StoreEvent,
    StoreReset,
)

MAX_CANDIDATES = 4
_ITEMSIZE = 4


def pack_tokens(tokens: Sequence[int]) -> bytes:
    return array("i", tokens).tobytes()


def _common_prefix(a: bytes, b: bytes) -> int:
    """Number of leading tokens two packed sequences share."""
    n = min(len(a), len(b)) // _ITEMSIZE
    if a[: n * _ITEMSIZE] == b[: n * _ITEMSIZE]:
        return n
    lo, hi = 0, n - 1
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if a[: mid * _ITEMSIZE] == b[: mid * _ITEMSIZE]:
            lo = mid
        else:
            hi = mid - 1
    return lo


def _token_at(tokens: bytes, pos: int) -> int:
    return int.from_bytes(
        tokens[pos * _ITEMSIZE : (pos + 1) * _ITEMSIZE], "little", signed=True
    )


class _Vertex:
    """A radix edge and the vertex it leads to. Covers path positions
    `[depth - len(label), depth)`."""

    __slots__ = ("label", "depth", "parent", "children", "owners")

    def __init__(self, label: bytes, depth: int, parent: "_Vertex | None") -> None:
        self.label = label
        self.depth = depth
        self.parent = parent
        self.children: dict[int, _Vertex] = {}
        # Store nodes holding the KV of exactly this span on this path.
        self.owners: set[NodeId] = set()

    @property
    def start(self) -> int:
        return self.depth - len(self.label) // _ITEMSIZE


@dataclass
class _Node:
    parent: NodeId | None
    start: int
    end: int
    tokens: bytes
    commit_points: tuple[int, ...] | None
    namespace: Hashable
    vertices: list[_Vertex] = field(default_factory=list)


class SessionState(enum.Enum):
    IDLE = "idle"
    ACTIVE = "active"


@dataclass
class Session:
    session_id: int
    namespace: Hashable
    cursor: NodeId | None = None
    state: SessionState = SessionState.IDLE
    writer: str | None = None
    client_id: str | None = None


class SessionResolver:
    """Single-threaded; colocated with the engine core in this version."""

    def __init__(self) -> None:
        self.epoch = 0
        self.seq = 0
        self.roots: dict[Hashable, _Vertex] = {}
        self.nodes: dict[NodeId, _Node] = {}
        self.sessions: dict[int, Session] = {}
        self.cursor_owner: dict[NodeId, int] = {}
        self.aliases: dict[tuple[Hashable, str], int] = {}
        # States published for each client session id, so a caller can
        # release a session it knows has ended.
        self.client_nodes: dict[str, set[NodeId]] = {}
        self.node_client: dict[NodeId, str] = {}
        self.request_session: dict[str, int] = {}
        self._packed: dict[str, bytes] = {}
        self._next_session_id = 1
        self.num_resolves = 0
        self.num_continued = 0
        self.resolve_seconds = 0.0

    @staticmethod
    def namespace(request: Request) -> Hashable:
        lora = request.lora_request.lora_name if request.lora_request else None
        return (request.cache_salt, lora)

    # ------------------------------------------------------------------ resolve
    def resolve(self, request: Request) -> SessionBinding:
        start_time = time.perf_counter()
        namespace = self.namespace(request)
        tokens = self._packed_tokens(request)
        limit = request.get_max_cache_hit_length()
        path, matched = self._match(namespace, tokens, limit)
        candidates = self._candidates(path, matched)

        session_id = self.request_session.get(request.request_id)
        if session_id is None:
            session_id = self._assign_session(request, namespace, candidates)
        self.resolve_seconds += time.perf_counter() - start_time
        self.num_resolves += 1
        return SessionBinding(self.epoch, session_id, candidates, self.seq, matched)

    def _packed_tokens(self, request: Request) -> bytes:
        """Pack a request's tokens once; re-resolves only pack new tokens."""
        tokens = request.all_token_ids
        packed = self._packed.get(request.request_id, b"")
        num_packed = len(packed) // _ITEMSIZE
        if num_packed > len(tokens):
            packed, num_packed = b"", 0
        if num_packed < len(tokens):
            packed += pack_tokens(tokens[num_packed:])
            self._packed[request.request_id] = packed
        return packed

    def _match(
        self, namespace: Hashable, tokens: bytes, limit: int
    ) -> tuple[list[_Vertex], int]:
        """Vertices along the longest match and the matched length."""
        vertex = self.roots.get(namespace)
        path: list[_Vertex] = []
        matched = 0
        while vertex is not None and matched < limit:
            child = vertex.children.get(_token_at(tokens, matched))
            if child is None:
                break
            n = _common_prefix(child.label, tokens[matched * _ITEMSIZE :])
            path.append(child)
            matched = min(matched + n, limit)
            if n < len(child.label) // _ITEMSIZE:
                break
            vertex = child
        return path, matched

    def _candidates(self, path: list[_Vertex], matched: int) -> tuple[Candidate, ...]:
        found: dict[NodeId, int] = {}
        for vertex in reversed(path):
            reach = min(vertex.depth, matched)
            for node_id in vertex.owners:
                if node_id in found:
                    continue
                length = self._resume_length(self.nodes[node_id], reach)
                if length:
                    found[node_id] = length
            if len(found) >= MAX_CANDIDATES:
                break
        ranked = sorted(found.items(), key=lambda kv: (-kv[1], -kv[0].local))
        return tuple(Candidate(n, length) for n, length in ranked[:MAX_CANDIDATES])

    @staticmethod
    def _resume_length(node: _Node, reach: int) -> int:
        if node.commit_points is None:
            length = min(reach, node.end)
            return length if length > node.start else 0
        points = [c for c in node.commit_points if node.start < c <= reach]
        return max(points, default=0)

    def _assign_session(
        self,
        request: Request,
        namespace: Hashable,
        candidates: tuple[Candidate, ...],
    ) -> int:
        client_id = request.session_id
        preferred = (
            self.aliases.get((namespace, client_id)) if client_id is not None else None
        )
        # Continue the session whose whole cursor state this request extends;
        # candidates are longest first.
        chosen: Session | None = None
        for cand in candidates:
            if (
                chosen is not None
                and chosen.cursor is not None
                and cand.length < self.nodes[chosen.cursor].end
            ):
                break
            owner = self.cursor_owner.get(cand.node_id)
            if owner is None or cand.length != self.nodes[cand.node_id].end:
                continue
            session = self.sessions[owner]
            if session.state != SessionState.IDLE:
                continue
            if chosen is None or owner == preferred:
                chosen = session
        if chosen is None:
            chosen = Session(self._next_session_id, namespace, client_id=client_id)
            self._next_session_id += 1
            self.sessions[chosen.session_id] = chosen
        else:
            self.num_continued += 1
        if client_id is not None:
            self.aliases[namespace, client_id] = chosen.session_id
        chosen.state = SessionState.ACTIVE
        chosen.writer = request.request_id
        self.request_session[request.request_id] = chosen.session_id
        return chosen.session_id

    # ------------------------------------------------------------------- events
    def apply(self, event: StoreEvent) -> None:
        assert event.seq == self.seq + 1, (event.seq, self.seq)
        self.seq = event.seq
        if isinstance(event, NodePublished):
            self._insert(event)
        elif isinstance(event, NodeExtended):
            self._extend(event)
        elif isinstance(event, NodeTruncated):
            self._truncate(event.node_id, event.end, event.commit_points)
        elif isinstance(event, RequestFinished):
            self._finish(event.request_id, event.node_id)
        elif isinstance(event, StoreReset):
            self._reset(event.epoch)

    def _path_tokens(self, parent: NodeId | None, start: int) -> bytes:
        parts: list[bytes] = []
        while parent is not None and start > 0:
            node = self.nodes[parent]
            parts.append(node.tokens[: (start - node.start) * _ITEMSIZE])
            start, parent = node.start, node.parent
        assert start == 0, "parent chain does not reach the root"
        return b"".join(reversed(parts))

    def _insert(self, event: NodePublished) -> None:
        request_session = self.request_session.get(event.request_id)
        namespace = (
            self.sessions[request_session].namespace
            if request_session is not None
            else None
        )
        node = _Node(
            event.parent_id,
            event.start,
            event.end,
            event.tokens,
            event.commit_points,
            namespace,
        )
        self.nodes[event.node_id] = node
        client_id = (
            self.sessions[request_session].client_id
            if request_session is not None
            else None
        )
        if client_id is not None:
            self.node_client[event.node_id] = client_id
            self.client_nodes.setdefault(client_id, set()).add(event.node_id)
        path = self._path_tokens(event.parent_id, event.start) + event.tokens
        self._add_span(event.node_id, path, event.start)

    def _extend(self, event: NodeExtended) -> None:
        node = self.nodes[event.node_id]
        assert node.end == event.start
        if event.commit_points is not None:
            node.commit_points = event.commit_points
        if event.end == event.start:
            return
        node.tokens += event.tokens
        node.end = event.end
        path = self._path_tokens(node.parent, node.start) + node.tokens
        self._add_span(event.node_id, path, event.start)

    def _add_span(self, node_id: NodeId, path: bytes, own_from: int) -> None:
        """Insert `path` and mark `node_id` as holding `[own_from, end)`."""
        node = self.nodes[node_id]
        end = len(path) // _ITEMSIZE
        vertex = self.roots.setdefault(node.namespace, _Vertex(b"", 0, None))
        depth = 0
        while depth < end:
            first = _token_at(path, depth)
            child = vertex.children.get(first)
            if child is None:
                assert depth >= own_from, "ancestor content is missing"
                child = _Vertex(path[depth * _ITEMSIZE :], end, vertex)
                vertex.children[first] = child
            else:
                cut = _common_prefix(child.label, path[depth * _ITEMSIZE :])
                if depth < own_from < depth + cut:
                    cut = own_from - depth
                if cut < len(child.label) // _ITEMSIZE:
                    child = self._split(child, cut)
            if child.start >= own_from:
                child.owners.add(node_id)
                node.vertices.append(child)
            vertex, depth = child, child.depth

    def _split(self, vertex: _Vertex, cut: int) -> _Vertex:
        """Split `vertex` after `cut` tokens; return the upper half."""
        assert vertex.parent is not None and cut > 0
        upper = _Vertex(vertex.label[: cut * _ITEMSIZE], vertex.start + cut, None)
        upper.parent = vertex.parent
        upper.owners = set(vertex.owners)
        vertex.parent.children[_token_at(vertex.label, 0)] = upper
        vertex.label = vertex.label[cut * _ITEMSIZE :]
        vertex.parent = upper
        upper.children[_token_at(vertex.label, 0)] = vertex
        for node_id in upper.owners:
            vertices = self.nodes[node_id].vertices
            vertices.insert(vertices.index(vertex), upper)
        return upper

    def _truncate(
        self, node_id: NodeId, end: int, commit_points: tuple[int, ...] | None
    ) -> None:
        node = self.nodes.get(node_id)
        if node is None:
            return
        kept: list[_Vertex] = []
        for vertex in list(node.vertices):
            if vertex.depth <= end:
                kept.append(vertex)
                continue
            if vertex.start < end:
                kept.append(self._split(vertex, end - vertex.start))
            vertex.owners.discard(node_id)
            self._prune(vertex)
        node.vertices = kept
        node.end = min(node.end, end)
        node.commit_points = commit_points
        node.tokens = node.tokens[: max(0, end - node.start) * _ITEMSIZE]
        if end <= node.start:
            del self.nodes[node_id]
            client_id = self.node_client.pop(node_id, None)
            if client_id is not None:
                published = self.client_nodes[client_id]
                published.discard(node_id)
                if not published:
                    del self.client_nodes[client_id]
            session_id = self.cursor_owner.pop(node_id, None)
            if session_id is not None:
                self._retire_cursor(self.sessions[session_id])

    def _prune(self, vertex: _Vertex) -> None:
        while vertex.parent is not None and not vertex.owners and not vertex.children:
            parent = vertex.parent
            del parent.children[_token_at(vertex.label, 0)]
            vertex = parent

    def _retire_cursor(self, session: Session) -> None:
        session.cursor = None
        if session.state == SessionState.IDLE:
            # Nothing left to continue; a new request starts a new session.
            self._drop_session(session)

    def _finish(self, request_id: str, node_id: NodeId | None) -> None:
        self._packed.pop(request_id, None)
        session_id = self.request_session.pop(request_id, None)
        if session_id is None:
            return
        session = self.sessions[session_id]
        if session.writer != request_id:
            return
        session.writer = None
        session.state = SessionState.IDLE
        if node_id is not None and node_id in self.nodes:
            if session.cursor is not None:
                self.cursor_owner.pop(session.cursor, None)
            previous = self.cursor_owner.get(node_id)
            if previous is not None and previous != session_id:
                self._retire_cursor(self.sessions[previous])
            session.cursor = node_id
            self.cursor_owner[node_id] = session_id
        elif session.cursor is None:
            self._drop_session(session)

    def _drop_session(self, session: Session) -> None:
        del self.sessions[session.session_id]
        if session.client_id is not None:
            key = (session.namespace, session.client_id)
            if self.aliases.get(key) == session.session_id:
                del self.aliases[key]

    def release(self, client_ids: list[str]) -> list[NodeId]:
        """Forget the sessions a caller names as ended and return the states
        they published, deepest first, for the store to free what it can."""
        released: set[NodeId] = set()
        for client_id in client_ids:
            released |= self.client_nodes.pop(client_id, set())
            for session in list(self.sessions.values()):
                if session.client_id != client_id:
                    continue
                self.aliases.pop((session.namespace, client_id), None)
                session.client_id = None
                if session.state == SessionState.IDLE:
                    if session.cursor is not None:
                        self.cursor_owner.pop(session.cursor, None)
                    self._drop_session(session)
        for node_id in released:
            self.node_client.pop(node_id, None)
        return sorted(released, key=lambda n: -self.nodes[n].start)

    def _reset(self, epoch: int) -> None:
        self.epoch = epoch
        self.roots.clear()
        self.nodes.clear()
        self.client_nodes.clear()
        self.node_client.clear()
        self.cursor_owner.clear()
        for session in list(self.sessions.values()):
            session.cursor = None
            if session.state == SessionState.IDLE:
                self._drop_session(session)

    def stats(self) -> dict[str, float]:
        return {
            "resolves": self.num_resolves,
            "continued_sessions": self.num_continued,
            "resolve_seconds": self.resolve_seconds,
            "nodes": len(self.nodes),
            "sessions": len(self.sessions),
        }
