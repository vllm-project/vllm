# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Standalone speculator server: sessions, sequence registry, and the
adapter contract a concrete drafter implements.

The server never operates on tensors, so protocol and registry logic can
be tested without a GPU; adapters receive and return data-plane frames and
do their own device handling. Requests on one connection are processed serially,
which yields the initial invariants directly: no two proposals run on the
same sequence at once, batch IDs increase monotonically per session, and
a response is only sent after its result frames.
"""

import argparse
import contextlib
import itertools
import math
import threading
import time
import uuid
import zlib
from abc import ABC, abstractmethod
from collections.abc import Collection, Iterator
from dataclasses import dataclass
from typing import Any, NamedTuple

import msgspec

from vllm.logger import init_logger
from vllm.v1.spec_decode.remote.capabilities import (
    REMOTE_DRAFT_SUPPORTED_METHODS,
    SpeculatorPlacementCapabilities,
    TargetFeatureKind,
)
from vllm.v1.spec_decode.remote.protocol import (
    PROTOCOL_MAJOR,
    PROTOCOL_MINOR,
    AdvanceAndPropose,
    CancelBatch,
    CloseSequence,
    ErrorReply,
    FeatureSlot,
    Hello,
    HelloAck,
    OpenSequence,
    Ping,
    Pong,
    PrefillChunk,
    ProposalResponse,
    ProtocolError,
    RemoteServerLimits,
    SequenceAck,
    SequenceKey,
    SpeculatorStatusCode,
    TargetFeatureSchema,
    decode_envelope,
    decode_payload,
    encode_message,
    inline_feature_frame_rejections,
    inline_receive_budgets,
    limit_shortfalls,
)
from vllm.v1.spec_decode.remote.state import SequenceState
from vllm.v1.spec_decode.remote.transport import (
    FRAME_CONTROL,
    INLINE_TRANSPORT,
    MAX_CONTROL_FRAME_BYTES,
    MAX_HANDSHAKE_BYTES,
    ConnectionClosed,
    DataFrame,
    FramedConnection,
    Listener,
    TransportError,
    TransportTimeout,
    decode_data_frame,
    frame_to_ints,
    ints_to_frame,
    is_local_endpoint,
    listen,
    validate_frame,
    validate_integer_frame,
)

logger = init_logger(__name__)

DEFAULT_LIMITS = RemoteServerLimits(
    max_batch_size=256,
    max_feature_tokens=16384,
    max_sequences=1024,
    max_model_len=32768,
)
SUPPORTED_TRANSPORTS = (INLINE_TRANSPORT,)

_ACCEPT_POLL_S = 0.2
_RESULT_FRAMES = 3
# Input frames an AdvanceAndPropose may reference besides its feature
# slots: accepted counts, recovery tokens, query lengths, round ids.
_MAX_INPUT_FRAMES = 4


# ----------------------------------------------------------------------
# Adapter contract
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class PrefillInputs:
    """One verified prompt chunk in negotiated schema order.

    Token-level feature frames hold ``num_tokens`` rows; per-sequence
    frames hold one row.
    """

    offset: int
    num_tokens: int
    is_final: bool
    features: tuple[DataFrame | None, ...]


@dataclass(frozen=True)
class StandaloneProposalBatch:
    """One advance-and-propose round as seen by an adapter.

    Rows not in ``active_rows`` were rejected by the registry (stale,
    desynced, not prefilled); the adapter must still return a row for them
    but must not advance their state.

    Per-sequence feature frames hold one row per key. Token-level feature
    frames hold ``sum(query_lens)`` rows: key ``i`` owns the
    ``query_lens[i]`` rows after those of keys ``0..i-1``, inactive keys
    included.
    """

    batch_id: int
    keys: tuple[SequenceKey, ...]
    active_rows: tuple[int, ...]
    num_speculative_tokens: int
    accepted_counts: DataFrame
    recovery_tokens: DataFrame
    features: tuple[DataFrame | None, ...]
    query_lens: tuple[int, ...]


@dataclass(frozen=True)
class StandaloneProposalResult:
    """Adapter output for every row of a StandaloneProposalBatch."""

    token_ids: DataFrame
    """Integer frame of shape [len(keys), num_speculative_tokens]."""
    valid_lengths: DataFrame
    """Integer frame of shape [len(keys)]."""
    row_statuses: tuple[SpeculatorStatusCode, ...]


class StandaloneDraftAdapter(ABC):
    """Model-specific drafter behind the server.

    The server serializes all calls, so implementations need no locking.
    Capabilities must describe the adapter and checkpoint actually loaded,
    never be inferred from the method name.
    """

    @abstractmethod
    def capabilities(self) -> SpeculatorPlacementCapabilities: ...

    @abstractmethod
    def feature_schema(self) -> TargetFeatureSchema: ...

    @abstractmethod
    def supported_methods(self) -> Collection[str]: ...

    def draft_checkpoint_fingerprint(self) -> str:
        return ""

    def warmup(self) -> None:  # noqa: B027
        """Compile and capture before the server accepts connections, so the
        first verifier rounds are not spent on one-time setup."""

    @abstractmethod
    def open_sequence(self, key: SequenceKey) -> None: ...

    @abstractmethod
    def prefill(self, key: SequenceKey, inputs: PrefillInputs) -> None: ...

    @abstractmethod
    def advance_and_propose(
        self, batch: StandaloneProposalBatch
    ) -> StandaloneProposalResult: ...

    @abstractmethod
    def close_sequences(self, keys: tuple[SequenceKey, ...]) -> None:
        """Retire every key before returning; an exception fences the server."""
        ...


class FakeDraftAdapter(StandaloneDraftAdapter):
    """Deterministic drafter for protocol tests.

    ``proposal[i, j] = (recovery_token[i] + j + 1) % vocab_size``, so a
    verifier can check end to end that the right rows came back in the
    right order. Feature slot 0 must be token-level TOKEN_IDS.
    ``hold_batches`` makes a batch block until ``release`` is called, which
    is how slow-server and timeout paths are exercised.
    """

    def __init__(
        self,
        *,
        vocab_size: int = 32000,
        capabilities: SpeculatorPlacementCapabilities | None = None,
        methods: Collection[str] = REMOTE_DRAFT_SUPPORTED_METHODS,
        hold_timeout_s: float = 10.0,
        draft_checkpoint_fingerprint: str = "",
        feature_schema: TargetFeatureSchema | None = None,
    ) -> None:
        self._vocab_size = vocab_size
        self._draft_checkpoint_fingerprint = draft_checkpoint_fingerprint
        self._capabilities = capabilities or SpeculatorPlacementCapabilities(
            state_dependency="own_kv",
            required_features=(TargetFeatureKind.TOKEN_IDS.value,),
            supports_parallel_drafting=True,
            standalone_weights="complete",
        )
        self._methods = frozenset(methods)
        self._schema = feature_schema or TargetFeatureSchema(
            schema_id=1,
            slots=(
                FeatureSlot(
                    kind=TargetFeatureKind.TOKEN_IDS.value,
                    dtype="int32",
                    trailing_shape=(),
                ),
            ),
        )
        self._prefixes: dict[SequenceKey, list[int]] = {}
        self.round_tokens: dict[SequenceKey, list[int]] = {}
        """Token-level TOKEN_IDS rows each active key received last round."""
        self._hold_timeout_s = hold_timeout_s
        self._release = threading.Event()
        self.hold_batches: set[int] = set()
        self.proposed_batches: list[int] = []

    def capabilities(self) -> SpeculatorPlacementCapabilities:
        return self._capabilities

    def feature_schema(self) -> TargetFeatureSchema:
        return self._schema

    def supported_methods(self) -> Collection[str]:
        return self._methods

    def draft_checkpoint_fingerprint(self) -> str:
        return self._draft_checkpoint_fingerprint

    def prefix(self, key: SequenceKey) -> list[int] | None:
        return self._prefixes.get(key)

    def release(self) -> None:
        self._release.set()

    def open_sequence(self, key: SequenceKey) -> None:
        self._prefixes[key] = []

    def prefill(self, key: SequenceKey, inputs: PrefillInputs) -> None:
        tokens = inputs.features[0]
        assert tokens is not None
        self._prefixes[key].extend(frame_to_ints(tokens))

    def advance_and_propose(
        self, batch: StandaloneProposalBatch
    ) -> StandaloneProposalResult:
        if batch.batch_id in self.hold_batches:
            self._release.wait(self._hold_timeout_s)
        self.proposed_batches.append(batch.batch_id)
        token_ids = batch.features[0]
        if token_ids is not None:
            values = frame_to_ints(token_ids)
            start = 0
            for i, length in enumerate(batch.query_lens):
                if i in batch.active_rows:
                    self.round_tokens[batch.keys[i]] = values[start : start + length]
                start += length
        recovery = frame_to_ints(batch.recovery_tokens)
        num_rows = len(batch.keys)
        num_tokens = batch.num_speculative_tokens
        flat: list[int] = []
        for i in range(num_rows):
            last = recovery[i]
            flat.extend((last + j + 1) % self._vocab_size for j in range(num_tokens))
            if i in batch.active_rows:
                self._prefixes[batch.keys[i]].append(last)
        return StandaloneProposalResult(
            token_ids=ints_to_frame(
                0, flat, dtype="int64", shape=(num_rows, num_tokens)
            ),
            valid_lengths=ints_to_frame(0, [num_tokens] * num_rows),
            row_statuses=(SpeculatorStatusCode.OK,) * num_rows,
        )

    def close_sequences(self, keys: tuple[SequenceKey, ...]) -> None:
        for key in keys:
            self._prefixes.pop(key, None)


# ----------------------------------------------------------------------
# Sequence registry
# ----------------------------------------------------------------------


@dataclass
class SequenceEntry:
    """Server-side bookkeeping for one (verifier, sequence, generation)."""

    key: SequenceKey
    owner: str = ""
    """session_id of the connection that opened this generation."""
    state: SequenceState = SequenceState.OPENING
    applied_offset: int = 0
    last_chunk: tuple[int, int, int, bool] | None = None
    """(offset, num_tokens, checksum, is_final) of the last applied chunk."""
    prefix_length: int = 0
    last_batch_id: int = -1
    last_activity: float = 0.0


class OpenOutcome(NamedTuple):
    status: SpeculatorStatusCode
    detail: str
    created: bool
    replaced: SequenceKey | None


class SequenceRegistry:
    """Idempotent sequence lifecycle keyed by (verifier, sequence_id).

    A newer generation replaces the entry; commands carrying an older
    generation are rejected as STALE_GENERATION. Duplicate open, duplicate
    prefill chunk (same offset and checksum), and duplicate close are all
    acknowledged without side effects. Entries record the session that
    opened them, so the delayed disconnect of a superseded connection can
    only release its own leftovers, never a reconnected verifier's state.
    """

    def __init__(self, clock=time.monotonic) -> None:
        self._entries: dict[tuple[str, int], SequenceEntry] = {}
        self._counts: dict[str, int] = {}
        self._clock = clock

    def __len__(self) -> int:
        return len(self._entries)

    def count(self, verifier_instance_id: str) -> int:
        """Open sequences of one verifier; safe to call without the lock."""
        return self._counts.get(verifier_instance_id, 0)

    def get(self, key: SequenceKey) -> SequenceEntry | None:
        entry = self._entries.get(_registry_key(key))
        return entry if entry is not None and entry.key == key else None

    def has_identity(self, key: SequenceKey) -> bool:
        """Whether this verifier and sequence ID already consume one slot."""
        return _registry_key(key) in self._entries

    def replaced_by(self, key: SequenceKey) -> SequenceKey | None:
        """Return the current generation that opening this key would replace."""
        entry = self._entries.get(_registry_key(key))
        if entry is not None and key.generation > entry.key.generation:
            return entry.key
        return None

    def open(self, key: SequenceKey, *, owner: str) -> OpenOutcome:
        entry = self._entries.get(_registry_key(key))
        if entry is not None:
            if key.generation < entry.key.generation:
                return OpenOutcome(
                    SpeculatorStatusCode.STALE_GENERATION,
                    f"generation {key.generation} superseded by {entry.key.generation}",
                    False,
                    None,
                )
            if key.generation == entry.key.generation:
                if entry.owner != owner:
                    return OpenOutcome(
                        SpeculatorStatusCode.SEQUENCE_DESYNC,
                        "generation is owned by another session",
                        False,
                        None,
                    )
                entry.last_activity = self._clock()
                return OpenOutcome(SpeculatorStatusCode.OK, "already open", False, None)
        replaced = entry.key if entry is not None else None
        if replaced is None:
            self._add_count(key.verifier_instance_id, 1)
        self._entries[_registry_key(key)] = SequenceEntry(
            key=key, owner=owner, last_activity=self._clock()
        )
        return OpenOutcome(SpeculatorStatusCode.OK, "", True, replaced)

    def prefill(
        self,
        key: SequenceKey,
        *,
        owner: str,
        offset: int,
        num_tokens: int,
        checksum: int,
        is_final: bool,
    ) -> tuple[SpeculatorStatusCode, str, bool]:
        """Returns (status, detail, apply): apply is False for duplicates."""
        status, detail, entry = self._lookup(key, owner)
        if entry is None:
            return status, detail, False
        entry.last_activity = self._clock()
        if entry.last_chunk is not None and offset == entry.last_chunk[0]:
            if (num_tokens, checksum, is_final) == entry.last_chunk[1:]:
                return SpeculatorStatusCode.OK, "duplicate chunk", False
            return (
                SpeculatorStatusCode.SEQUENCE_DESYNC,
                (
                    f"chunk at offset {offset} was already applied with "
                    "different content or finality"
                ),
                False,
            )
        if entry.state is SequenceState.READY:
            return (
                SpeculatorStatusCode.SEQUENCE_DESYNC,
                "prefill after final chunk",
                False,
            )
        if offset != entry.applied_offset:
            return (
                SpeculatorStatusCode.SEQUENCE_DESYNC,
                f"expected offset {entry.applied_offset}, got {offset}",
                False,
            )
        entry.applied_offset += num_tokens
        entry.prefix_length = entry.applied_offset
        entry.last_chunk = (offset, num_tokens, checksum, is_final)
        entry.state = SequenceState.READY if is_final else SequenceState.PREFILLING
        return SpeculatorStatusCode.OK, "", True

    def begin_round(
        self, key: SequenceKey, batch_id: int, *, owner: str
    ) -> tuple[SpeculatorStatusCode, str]:
        status, detail, entry = self._lookup(key, owner)
        if entry is None:
            return status, detail
        if entry.state is SequenceState.DESYNCED:
            return (
                SpeculatorStatusCode.SEQUENCE_DESYNC,
                "an earlier round failed for this sequence",
            )
        if entry.state is not SequenceState.READY:
            return SpeculatorStatusCode.SEQUENCE_DESYNC, "sequence is not prefilled"
        if batch_id <= entry.last_batch_id:
            return (
                SpeculatorStatusCode.ROUND_MISMATCH,
                f"batch {batch_id} is not newer than {entry.last_batch_id}",
            )
        entry.last_batch_id = batch_id
        entry.last_activity = self._clock()
        return SpeculatorStatusCode.OK, ""

    def advance(self, key: SequenceKey, accepted: int, *, owner: str) -> None:
        entry = self.get(key)
        if entry is not None and entry.owner == owner:
            # accepted drafts plus the bonus/recovery token
            entry.prefix_length += accepted + 1

    def desync(self, key: SequenceKey, *, owner: str) -> None:
        """Refuse further rounds after a failed one.

        The adapter did not advance the row, so its draft state no longer
        matches the verifier's; a later round must not silently extend it.
        """
        entry = self.get(key)
        if entry is not None and entry.owner == owner:
            entry.state = SequenceState.DESYNCED

    def close(self, key: SequenceKey, *, owner: str) -> SequenceKey | None:
        """Remove an exact key only when this connection still owns it."""
        entry = self._entries.get(_registry_key(key))
        if entry is None or entry.key != key or entry.owner != owner:
            return None
        self._remove(key)
        return entry.key

    def close_all(self, owner: str) -> tuple[SequenceKey, ...]:
        """Remove every entry still owned by one connection's session."""
        removed = self.owned_keys(owner)
        for key in removed:
            self._remove(key)
        return removed

    def owned_keys(self, owner: str) -> tuple[SequenceKey, ...]:
        """Keep ownership recorded while the adapter retires these entries."""
        return tuple(
            entry.key for entry in self._entries.values() if entry.owner == owner
        )

    def _remove(self, key: SequenceKey) -> None:
        del self._entries[_registry_key(key)]
        self._add_count(key.verifier_instance_id, -1)

    def _add_count(self, verifier_instance_id: str, delta: int) -> None:
        count = self._counts.get(verifier_instance_id, 0) + delta
        if count:
            self._counts[verifier_instance_id] = count
        else:
            self._counts.pop(verifier_instance_id, None)

    def _lookup(
        self, key: SequenceKey, owner: str
    ) -> tuple[SpeculatorStatusCode, str, SequenceEntry | None]:
        entry = self._entries.get(_registry_key(key))
        if entry is None:
            return SpeculatorStatusCode.SEQUENCE_DESYNC, "unknown sequence", None
        if key.generation < entry.key.generation:
            return (
                SpeculatorStatusCode.STALE_GENERATION,
                f"generation {key.generation} superseded by {entry.key.generation}",
                None,
            )
        if key.generation > entry.key.generation:
            return (
                SpeculatorStatusCode.SEQUENCE_DESYNC,
                f"generation {key.generation} was never opened",
                None,
            )
        if entry.owner != owner:
            return (
                SpeculatorStatusCode.SEQUENCE_DESYNC,
                "generation is owned by another session",
                None,
            )
        return SpeculatorStatusCode.OK, "", entry


def _registry_key(key: SequenceKey) -> tuple[str, int]:
    return key.verifier_instance_id, key.sequence_id


# ----------------------------------------------------------------------
# Server
# ----------------------------------------------------------------------


class ServerShutdownTimeout(RuntimeError):
    """Workers remain live; retain the adapter and restart the owning process."""


class SequenceCleanupFailed(RuntimeError):
    """Retirement is incomplete; retain owners and restart the owning process."""


class AdapterQueueFull(RuntimeError):
    """This connection cannot enqueue another adapter operation."""


class RemoteDraftServer:
    """Accepts verifier sessions and drives one adapter.

    ``start`` binds the endpoint and serves in background threads (one
    acceptor plus one per connection); ``stop`` tears everything down and
    joins within a bounded grace period. Adapter and registry access is serialized by
    one lock; liveness
    probes never take it, so they answer while the adapter is busy.
    """

    def __init__(
        self,
        adapter: StandaloneDraftAdapter,
        *,
        server_id: str | None = None,
        limits: RemoteServerLimits = DEFAULT_LIMITS,
        target_fingerprint: str = "",
        tokenizer_fingerprint: str = "",
        handshake_timeout_s: float = 30.0,
        send_timeout_s: float = 30.0,
        max_pending_bytes: int = 1 << 30,
        max_connections: int = 128,
        max_handshakes: int = 16,
        max_adapter_waiters: int = 64,
        partial_frame_timeout_s: float = 30.0,
        shutdown_timeout_s: float = 5.0,
        allow_remote_connections: bool = False,
        clock=time.monotonic,
    ) -> None:
        if max_pending_bytes <= 0:
            raise ValueError("max_pending_bytes must be positive")
        for name, value in (
            ("max_connections", max_connections),
            ("max_handshakes", max_handshakes),
            ("max_adapter_waiters", max_adapter_waiters),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        for name, timeout in (
            ("handshake_timeout_s", handshake_timeout_s),
            ("send_timeout_s", send_timeout_s),
            ("partial_frame_timeout_s", partial_frame_timeout_s),
            ("shutdown_timeout_s", shutdown_timeout_s),
        ):
            if not math.isfinite(timeout) or timeout <= 0:
                raise ValueError(f"{name} must be finite and positive")
        self.max_pending_bytes = max_pending_bytes
        self.max_connections = max_connections
        self.max_handshakes = max_handshakes
        self.max_adapter_waiters = max_adapter_waiters
        self.partial_frame_timeout_s = partial_frame_timeout_s
        self.shutdown_timeout_s = shutdown_timeout_s
        self.allow_remote_connections = allow_remote_connections
        self._shutdown_failed = False
        self._cleanup_failed = False
        self._draining = False
        self._connections_lock = threading.Lock()
        self._handshaking: set[FramedConnection] = set()
        self.adapter = adapter
        self.server_id = server_id or f"draft-{uuid.uuid4().hex[:12]}"
        self.limits = limits
        self.target_fingerprint = target_fingerprint
        self.tokenizer_fingerprint = tokenizer_fingerprint
        self.handshake_timeout_s = handshake_timeout_s
        self.send_timeout_s = send_timeout_s
        self.registry = SequenceRegistry(clock)
        self.lock = threading.Lock()
        self._queue_lock = threading.Lock()
        self._queue_depth = 0
        self._epochs = itertools.count(1)
        self._listener: Listener | None = None
        self._acceptor: threading.Thread | None = None
        self._workers: list[threading.Thread] = []
        self._connections: set[FramedConnection] = set()
        self._stopping = threading.Event()

    @property
    def endpoint(self) -> str | None:
        return self._listener.endpoint if self._listener is not None else None

    @property
    def queue_depth(self) -> int:
        """Adapter calls running or waiting for their turn."""
        return self._queue_depth

    @contextlib.contextmanager
    def adapter_turn(self, *, cleanup: bool = False) -> Iterator[None]:
        """Bound foreground waiters; admitted connections retain a cleanup turn."""
        with self._queue_lock:
            if not cleanup and self._queue_depth >= self.max_adapter_waiters + 1:
                raise AdapterQueueFull("remote adapter queue is full")
            self._queue_depth += 1
        acquired = False
        try:
            while not acquired:
                if self._shutdown_failed:
                    raise ServerShutdownTimeout("runtime requires process restart")
                if self._cleanup_failed:
                    raise SequenceCleanupFailed("runtime requires process restart")
                if not cleanup and self._draining:
                    raise ConnectionClosed("remote draft server is stopping")
                acquired = self.lock.acquire(timeout=0.1)
            if self._shutdown_failed:
                raise ServerShutdownTimeout("runtime requires process restart")
            if self._cleanup_failed:
                raise SequenceCleanupFailed("runtime requires process restart")
            if not cleanup and self._draining:
                raise ConnectionClosed("remote draft server is stopping")
            yield
        finally:
            if acquired:
                self.lock.release()
            with self._queue_lock:
                self._queue_depth -= 1

    def _handshake_complete(self, conn: FramedConnection) -> None:
        with self._connections_lock:
            self._handshaking.discard(conn)

    def _retire_sequences(self, keys: tuple[SequenceKey, ...]) -> None:
        """Retire under the adapter lock before relinquishing registry ownership."""
        if self._cleanup_failed:
            raise SequenceCleanupFailed("runtime requires process restart")
        if not keys:
            return
        try:
            self.adapter.close_sequences(keys)
        except Exception:
            self._cleanup_failed = True
            self._stopping.set()
            raise
        for key in keys:
            entry = self.registry.get(key)
            if entry is not None:
                self.registry.close(key, owner=entry.owner)

    def start(self, endpoint: str) -> str:
        """Warm up, bind, and serve in the background.

        Returns the resolved endpoint. The endpoint does not accept
        connections until the adapter has finished warming up.
        """
        if self._shutdown_failed:
            raise ServerShutdownTimeout("runtime requires process restart")
        if self._cleanup_failed:
            raise SequenceCleanupFailed("runtime requires process restart")
        if self._listener is not None:
            raise RuntimeError("server is already running")
        if not is_local_endpoint(endpoint) and not self.allow_remote_connections:
            raise ValueError(
                "remote draft endpoints require loopback or UNIX sockets; "
                "external binding requires allow_remote_connections=True"
            )
        self.adapter.warmup()
        self._listener = listen(endpoint)
        if not is_local_endpoint(self._listener.endpoint):
            logger.warning(
                "Remote draft server listens on %s, which is reachable from "
                "other hosts; the protocol has no authentication, so bind to "
                "a loopback address or a UNIX socket.",
                self._listener.endpoint,
            )
        self._stopping.clear()
        self._acceptor = threading.Thread(
            target=self._accept_loop, name="remote-draft-acceptor", daemon=True
        )
        self._acceptor.start()
        capabilities = self.adapter.capabilities()
        logger.info(
            "Remote draft server %s ready at %s: protocol %d.%d, methods %s, "
            "draft fingerprint %r, required features %s, transports %s, "
            "max_batch_size %d, max_model_len %d",
            self.server_id,
            self._listener.endpoint,
            PROTOCOL_MAJOR,
            PROTOCOL_MINOR,
            sorted(self.adapter.supported_methods()),
            self.adapter.draft_checkpoint_fingerprint(),
            list(capabilities.required_features),
            list(SUPPORTED_TRANSPORTS),
            self.limits.max_batch_size,
            self.limits.max_model_len,
        )
        return self._listener.endpoint

    def wait(self, timeout: float | None = None) -> bool:
        """Block until ``stop`` is called; returns False on timeout."""
        return self._stopping.wait(timeout)

    def stop(self) -> None:
        """Stop within the grace period or require process replacement.

        A timed-out worker may still use the adapter/GPU. Keep every owner alive,
        forbid restart, and let the process supervisor terminate the process.
        """
        deadline = time.monotonic() + self.shutdown_timeout_s
        self._draining = True
        self._stopping.set()
        listener, self._listener = self._listener, None
        if listener is not None:
            listener.close()
        with self._connections_lock:
            connections = tuple(self._connections)
        for conn in connections:
            conn.close()
        if self._acceptor is not None:
            self._acceptor.join(max(0.0, deadline - time.monotonic()))
        with self._connections_lock:
            workers = tuple(self._workers)
        for worker in workers:
            worker.join(max(0.0, deadline - time.monotonic()))
        if (self._acceptor is not None and self._acceptor.is_alive()) or any(
            worker.is_alive() for worker in workers
        ):
            self._shutdown_failed = True
            raise ServerShutdownTimeout(
                "remote draft shutdown grace period expired; live adapter/GPU "
                "resources are retained and the owning process must be restarted"
            )
        self._acceptor = None
        self._workers.clear()
        self._draining = False

    def next_epoch(self) -> int:
        return next(self._epochs)

    def _accept_loop(self) -> None:
        listener = self._listener
        assert listener is not None
        while not self._stopping.is_set():
            try:
                conn = listener.accept(timeout=_ACCEPT_POLL_S)
            except TransportTimeout:
                continue
            except TransportError:
                break
            with self._connections_lock:
                if (
                    self._stopping.is_set()
                    or len(self._connections) >= self.max_connections
                    or len(self._handshaking) >= self.max_handshakes
                ):
                    conn.close()
                    continue
                self._connections.add(conn)
                self._handshaking.add(conn)
                self._workers = [w for w in self._workers if w.is_alive()]
                worker = threading.Thread(
                    target=self._serve_connection,
                    args=(conn,),
                    name="remote-draft-session",
                    daemon=True,
                )
                self._workers.append(worker)
                worker.start()

    def _serve_connection(self, conn: FramedConnection) -> None:
        session = _ServerSession(self, conn)
        try:
            session.run()
        except AdapterQueueFull:
            logger.warning(
                "Remote draft session %s: adapter queue full", session.session_id
            )
        except (ConnectionClosed, ServerShutdownTimeout, SequenceCleanupFailed):
            pass
        except Exception:
            logger.exception("Remote draft session %s crashed", session.session_id)
        finally:
            try:
                if not self._shutdown_failed and not self._cleanup_failed:
                    session.cleanup()
            except Exception:
                logger.exception(
                    "Failed to clean up remote draft session %s", session.session_id
                )
            finally:
                try:
                    conn.close()
                finally:
                    with self._connections_lock:
                        self._connections.discard(conn)
                        self._handshaking.discard(conn)


class _ServerSession:
    """One verifier connection: handshake, then a serial request loop."""

    def __init__(self, server: RemoteDraftServer, conn: FramedConnection) -> None:
        self.server = server
        self.conn = conn
        self.session_id = ""
        self.session_epoch = 0
        self.verifier_instance_id = ""
        self.num_speculative_tokens = 0
        self.inbound: dict[int, DataFrame] = {}
        self.last_batch_id = -1
        self.sequence_number = 0
        self.next_result_slot = 0
        self.cancelled_batches = 0
        self.max_pending_frames = 0
        self.max_pending_bytes = 0
        self.max_data_frame_bytes = 0
        self.pending_bytes = 0

    def run(self) -> None:
        if not self._handshake():
            return
        self.server._handshake_complete(self.conn)
        while True:
            try:
                frame = self.conn.recv()
            except ConnectionClosed:
                return
            except TransportError as e:
                logger.warning(
                    "Session %s: unrecoverable frame error: %s", self.session_id, e
                )
                return
            if frame.kind == FRAME_CONTROL:
                self._handle_control(frame.body)
                # Frames precede the message that consumes them, so whatever
                # it left behind can never be referenced again.
                self.inbound.clear()
                self.pending_bytes = 0
                self._set_input_budget()
                continue
            try:
                data = decode_data_frame(frame.body)
            except TransportError as e:
                self._send(ErrorReply(SpeculatorStatusCode.PROTOCOL_ERROR, str(e)), 0)
                continue
            if data.slot in self.inbound:
                raise TransportError("duplicate pending data slot")
            self.pending_bytes += len(frame.body)
            self.inbound[data.slot] = data
            self._set_input_budget()
            if len(self.inbound) > self.max_pending_frames:
                self._send(
                    ErrorReply(
                        SpeculatorStatusCode.PROTOCOL_ERROR,
                        f"more than {self.max_pending_frames} data frames "
                        "without a control message",
                    ),
                    0,
                )
                return

    def _set_input_budget(self) -> None:
        remaining = self.max_pending_bytes - self.pending_bytes
        self.conn.set_receive_limits(
            max_data_bytes=min(self.max_data_frame_bytes, remaining),
            max_buffered_bytes=MAX_CONTROL_FRAME_BYTES + remaining + 10,
            partial_frame_timeout=self.server.partial_frame_timeout_s,
        )

    def cleanup(self) -> None:
        if not self.session_id:
            return
        with self.server.adapter_turn(cleanup=True):
            self.server._retire_sequences(
                self.server.registry.owned_keys(self.session_id)
            )

    # -- handshake -------------------------------------------------------

    def _handshake(self) -> bool:
        self.conn.set_receive_limits(
            max_control_bytes=MAX_HANDSHAKE_BYTES,
            max_data_bytes=0,
            max_buffered_bytes=MAX_HANDSHAKE_BYTES + 5,
        )
        deadline = time.monotonic() + self.server.handshake_timeout_s
        try:
            frame = self.conn.recv(timeout=max(0.0, deadline - time.monotonic()))
        except TransportError:
            return False
        if frame.kind != FRAME_CONTROL:
            self._send(
                ErrorReply(SpeculatorStatusCode.PROTOCOL_ERROR, "expected hello"),
                0,
                deadline=deadline,
            )
            return False
        try:
            envelope = decode_envelope(frame.body)
            message = decode_payload(envelope)
        except ProtocolError as e:
            self._send(
                ErrorReply(SpeculatorStatusCode.PROTOCOL_ERROR, str(e)),
                0,
                deadline=deadline,
            )
            return False
        if not isinstance(message, Hello):
            self._send(
                ErrorReply(
                    SpeculatorStatusCode.PROTOCOL_ERROR,
                    f"expected hello, got {envelope.message_type}",
                ),
                envelope.request_id,
                deadline=deadline,
            )
            return False
        reasons = self._hello_rejections(message)
        if reasons:
            self._send(
                ErrorReply(
                    SpeculatorStatusCode.UNSUPPORTED_FEATURE, "; ".join(reasons)
                ),
                envelope.request_id,
                deadline=deadline,
            )
            return False
        selected = next(
            t for t in message.supported_transports if t in SUPPORTED_TRANSPORTS
        )
        self.session_id = uuid.uuid4().hex
        self.session_epoch = self.server.next_epoch()
        self.verifier_instance_id = message.verifier_instance_id
        self.num_speculative_tokens = message.num_speculative_tokens
        adapter = self.server.adapter
        self.max_pending_frames = _MAX_INPUT_FRAMES + len(
            adapter.feature_schema().slots
        )
        self.max_data_frame_bytes, self.max_pending_bytes = inline_receive_budgets(
            message, adapter.feature_schema(), self.server.limits
        )
        self._set_input_budget()
        self._send(
            HelloAck(
                server_id=self.server.server_id,
                session_id=self.session_id,
                session_epoch=self.session_epoch,
                selected_transport=selected,
                capabilities=adapter.capabilities(),
                feature_schema=adapter.feature_schema(),
                limits=self.server.limits,
            ),
            envelope.request_id,
            deadline=deadline,
        )
        logger.info(
            "Session %s (epoch %d) opened for verifier %s over %s",
            self.session_id,
            self.session_epoch,
            self.verifier_instance_id,
            selected,
        )
        return True

    def _hello_rejections(self, hello: Hello) -> list[str]:
        server = self.server
        reasons: list[str] = []
        if hello.method not in server.adapter.supported_methods():
            reasons.append(
                f"method {hello.method!r} is not served by this adapter "
                f"(supported: {sorted(server.adapter.supported_methods())})"
            )
        if (
            server.target_fingerprint
            and hello.target_fingerprint != server.target_fingerprint
        ):
            reasons.append(
                f"target fingerprint {hello.target_fingerprint!r} does not match "
                f"the server's {server.target_fingerprint!r}"
            )
        if (
            server.tokenizer_fingerprint
            and hello.tokenizer_fingerprint != server.tokenizer_fingerprint
        ):
            reasons.append(
                f"tokenizer fingerprint {hello.tokenizer_fingerprint!r} does not "
                f"match the server's {server.tokenizer_fingerprint!r}"
            )
        expected_draft = server.adapter.draft_checkpoint_fingerprint()
        if expected_draft and hello.draft_checkpoint_fingerprint != expected_draft:
            reasons.append(
                f"draft checkpoint fingerprint "
                f"{hello.draft_checkpoint_fingerprint!r} does not match the "
                f"server's {expected_draft!r}"
            )
        reasons.extend(limit_shortfalls(hello, server.limits))
        selected = next(
            (t for t in hello.supported_transports if t in SUPPORTED_TRANSPORTS), None
        )
        if selected == INLINE_TRANSPORT:
            reasons.extend(
                inline_feature_frame_rejections(
                    hello, server.adapter.feature_schema(), server.limits
                )
            )
            if not reasons:
                _, pending_bytes = inline_receive_budgets(
                    hello, server.adapter.feature_schema(), server.limits
                )
                if pending_bytes > server.max_pending_bytes:
                    reasons.append(
                        f"inline batch requires {pending_bytes} bytes, exceeds "
                        f"max_pending_bytes={server.max_pending_bytes}"
                    )
        if not any(t in SUPPORTED_TRANSPORTS for t in hello.supported_transports):
            reasons.append(
                f"no common transport: verifier offers "
                f"{list(hello.supported_transports)}, server supports "
                f"{list(SUPPORTED_TRANSPORTS)}"
            )
        return reasons

    # -- request loop ----------------------------------------------------

    def _send(
        self,
        message: Any,
        request_id: int,
        *,
        deadline: float | None = None,
        frames: list[DataFrame] | None = None,
    ) -> None:
        if deadline is None:
            deadline = time.monotonic() + self.server.send_timeout_s
        try:
            for frame in frames or ():
                self.conn.send_data(frame, deadline=deadline)
            self.conn.send_control(
                encode_message(
                    message, session_id=self.session_id, request_id=request_id
                ),
                deadline=deadline,
            )
        except TransportError:
            if frames is not None:
                self.conn.close()

    def _handle_control(self, body: bytes) -> None:
        try:
            envelope = decode_envelope(body)
            message = decode_payload(envelope)
        except ProtocolError as e:
            self._send(ErrorReply(SpeculatorStatusCode.PROTOCOL_ERROR, str(e)), 0)
            return
        request_id = envelope.request_id
        if envelope.session_id != self.session_id:
            self._send(
                ErrorReply(
                    SpeculatorStatusCode.PROTOCOL_ERROR,
                    f"session id {envelope.session_id!r} does not match this "
                    "connection",
                ),
                request_id,
            )
            return
        if isinstance(message, OpenSequence):
            self._open(message, request_id)
        elif isinstance(message, PrefillChunk):
            self._prefill(message, request_id)
        elif isinstance(message, AdvanceAndPropose):
            self._propose(message, request_id)
        elif isinstance(message, CloseSequence):
            self._close(message, request_id)
        elif isinstance(message, CancelBatch):
            self.cancelled_batches += 1
            logger.debug(
                "Session %s: cancel for batch %d (already served or never seen)",
                self.session_id,
                message.batch_id,
            )
        elif isinstance(message, Ping):
            self._send(
                Pong(
                    nonce=message.nonce,
                    queue_depth=self.server.queue_depth,
                    active_sequences=self.server.registry.count(
                        self.verifier_instance_id
                    ),
                ),
                request_id,
            )
        else:
            self._send(
                ErrorReply(
                    SpeculatorStatusCode.PROTOCOL_ERROR,
                    f"unexpected {envelope.message_type} from a verifier",
                ),
                request_id,
            )

    def _owns(self, key: SequenceKey) -> bool:
        return key.verifier_instance_id == self.verifier_instance_id

    def _ack(
        self,
        key: SequenceKey,
        request_id: int,
        status: SpeculatorStatusCode = SpeculatorStatusCode.OK,
        detail: str = "",
        *,
        deadline: float | None = None,
    ) -> None:
        self._send(
            SequenceAck(key=key, status=status, detail=detail),
            request_id,
            deadline=deadline,
        )

    def _open(self, message: OpenSequence, request_id: int) -> None:
        key = message.key
        if not self._owns(key):
            self._ack(
                key,
                request_id,
                SpeculatorStatusCode.PROTOCOL_ERROR,
                "key belongs to another verifier",
            )
            return
        server = self.server
        with server.adapter_turn():
            registry = server.registry
            if (
                not registry.has_identity(key)
                and len(registry) >= server.limits.max_sequences
            ):
                status = SpeculatorStatusCode.QUEUE_FULL
                detail = f"max_sequences={server.limits.max_sequences} reached"
            else:
                replaced = registry.replaced_by(key)
                if replaced is not None:
                    server._retire_sequences((replaced,))
                outcome = registry.open(key, owner=self.session_id)
                status, detail = outcome.status, outcome.detail
                if outcome.created:
                    server.adapter.open_sequence(key)
        self._ack(key, request_id, status, detail)

    def _prefill(self, message: PrefillChunk, request_id: int) -> None:
        key = message.key
        schema = self.server.adapter.feature_schema()
        frames = self._take_frames(message.feature_slot, len(schema.slots))
        if not self._owns(key):
            self._ack(
                key,
                request_id,
                SpeculatorStatusCode.PROTOCOL_ERROR,
                "key belongs to another verifier",
            )
            return
        limits = self.server.limits
        if message.num_tokens > limits.max_feature_tokens:
            self._ack(
                key,
                request_id,
                SpeculatorStatusCode.PROTOCOL_ERROR,
                f"chunk of {message.num_tokens} tokens exceeds "
                f"max_feature_tokens={limits.max_feature_tokens}",
            )
            return
        if message.offset + message.num_tokens > limits.max_model_len:
            self._ack(
                key,
                request_id,
                SpeculatorStatusCode.OUT_OF_MEMORY,
                f"prefill to offset {message.offset + message.num_tokens} "
                f"exceeds max_model_len={limits.max_model_len}",
            )
            return
        problem = _frame_problems(frames, schema, message.num_tokens, 1)
        if problem is not None:
            self._ack(key, request_id, SpeculatorStatusCode.PROTOCOL_ERROR, problem)
            return
        if _checksum(frames) != message.checksum:
            self._ack(
                key,
                request_id,
                SpeculatorStatusCode.SEQUENCE_DESYNC,
                "payload checksum does not match the chunk header",
            )
            return
        with self.server.adapter_turn():
            status, detail, apply = self.server.registry.prefill(
                key,
                owner=self.session_id,
                offset=message.offset,
                num_tokens=message.num_tokens,
                checksum=message.checksum,
                is_final=message.is_final,
            )
            if apply:
                self.server.adapter.prefill(
                    key,
                    PrefillInputs(
                        offset=message.offset,
                        num_tokens=message.num_tokens,
                        is_final=message.is_final,
                        features=tuple(frames),
                    ),
                )
        self._ack(key, request_id, status, detail)

    def _propose(self, message: AdvanceAndPropose, request_id: int) -> None:
        schema = self.server.adapter.feature_schema()
        accepted = self.inbound.pop(message.accepted_counts_slot, None)
        recovery = self.inbound.pop(message.recovery_tokens_slot, None)
        frames = self._take_frames(message.feature_slot, len(schema.slots))
        lens_frame = None
        if message.query_lens_slot is not None:
            lens_frame = self.inbound.pop(message.query_lens_slot, None)
        sequence_number = self.sequence_number
        self.sequence_number += 1

        def fail(status: SpeculatorStatusCode, detail: str) -> None:
            logger.warning(
                "Session %s: batch %d rejected: %s %s",
                self.session_id,
                message.batch_id,
                status.name,
                detail,
            )
            self._send(
                ProposalResponse(
                    batch_id=message.batch_id,
                    sequence_number=sequence_number,
                    status=status,
                ),
                request_id,
            )

        if message.batch_id <= self.last_batch_id:
            fail(
                SpeculatorStatusCode.ROUND_MISMATCH,
                f"batch id is not newer than {self.last_batch_id}",
            )
            return
        self.last_batch_id = message.batch_id
        num_rows = len(message.keys)
        if num_rows == 0 or num_rows > self.server.limits.max_batch_size:
            fail(
                SpeculatorStatusCode.PROTOCOL_ERROR,
                f"{num_rows} rows outside 1..{self.server.limits.max_batch_size}",
            )
            return
        if any(not self._owns(key) for key in message.keys):
            fail(SpeculatorStatusCode.PROTOCOL_ERROR, "key belongs to another verifier")
            return
        if accepted is None or recovery is None:
            fail(SpeculatorStatusCode.PROTOCOL_ERROR, "missing input frames")
            return
        if accepted.shape != (num_rows,) or recovery.shape != (num_rows,):
            fail(SpeculatorStatusCode.PROTOCOL_ERROR, "input frames do not match rows")
            return
        try:
            accepted_counts = frame_to_ints(accepted)
            validate_integer_frame(recovery)
        except TransportError as e:
            fail(SpeculatorStatusCode.PROTOCOL_ERROR, f"integer input frame: {e}")
            return
        num_tokens = self.num_speculative_tokens
        if any(not 0 <= count <= num_tokens for count in accepted_counts):
            fail(
                SpeculatorStatusCode.PROTOCOL_ERROR,
                f"accepted counts must be within 0..{num_tokens}",
            )
            return
        query_lens = [1] * num_rows
        if message.query_lens_slot is not None:
            if lens_frame is None or lens_frame.shape != (num_rows,):
                fail(
                    SpeculatorStatusCode.PROTOCOL_ERROR,
                    "query_lens frame is missing or does not match rows",
                )
                return
            try:
                query_lens = frame_to_ints(lens_frame)
            except TransportError as e:
                fail(SpeculatorStatusCode.PROTOCOL_ERROR, f"query_lens frame: {e}")
                return
            if any(length < 0 for length in query_lens):
                fail(SpeculatorStatusCode.PROTOCOL_ERROR, "negative query_lens")
                return
        num_feature_tokens = sum(query_lens)
        max_feature_tokens = self.server.limits.max_feature_tokens
        if num_feature_tokens > max_feature_tokens:
            fail(
                SpeculatorStatusCode.PROTOCOL_ERROR,
                f"{num_feature_tokens} feature tokens exceed "
                f"max_feature_tokens={max_feature_tokens}",
            )
            return
        problem = _frame_problems(frames, schema, num_feature_tokens, num_rows)
        if problem is not None:
            fail(SpeculatorStatusCode.PROTOCOL_ERROR, problem)
            return

        failure: str | None = None
        with self.server.adapter_turn():
            registry = self.server.registry
            begun = [
                registry.begin_round(key, message.batch_id, owner=self.session_id)[0]
                for key in message.keys
            ]
            row_statuses = list(begun)
            max_model_len = self.server.limits.max_model_len
            for i, seq_key in enumerate(message.keys):
                if row_statuses[i] is not SpeculatorStatusCode.OK:
                    continue
                entry = registry.get(seq_key)
                if entry is not None and (
                    entry.prefix_length + accepted_counts[i] + 1 > max_model_len
                ):
                    row_statuses[i] = SpeculatorStatusCode.OUT_OF_MEMORY
            active = tuple(
                i
                for i, status in enumerate(row_statuses)
                if status is SpeculatorStatusCode.OK
            )
            if active:
                active_keys = tuple(message.keys[i] for i in active)
                try:
                    result = self.server.adapter.advance_and_propose(
                        StandaloneProposalBatch(
                            batch_id=message.batch_id,
                            keys=message.keys,
                            active_rows=active,
                            num_speculative_tokens=num_tokens,
                            accepted_counts=accepted,
                            recovery_tokens=recovery,
                            features=tuple(frames),
                            query_lens=tuple(query_lens),
                        )
                    )
                except Exception:
                    logger.exception(
                        "Session %s: adapter failed on batch %d",
                        self.session_id,
                        message.batch_id,
                    )
                    failure = "adapter raised"
                else:
                    if not _result_well_formed(result, num_rows, num_tokens):
                        failure = "adapter returned a malformed result"
                    else:
                        for i in active:
                            row_statuses[i] = result.row_statuses[i]
                            if row_statuses[i] is SpeculatorStatusCode.OK:
                                registry.advance(
                                    message.keys[i],
                                    accepted_counts[i],
                                    owner=self.session_id,
                                )
                        valid_lengths = frame_to_ints(result.valid_lengths)
                        for i, status in enumerate(row_statuses):
                            if status is not SpeculatorStatusCode.OK:
                                valid_lengths[i] = 0
                        tokens = result.token_ids
                        valid = ints_to_frame(
                            0, valid_lengths, dtype=result.valid_lengths.dtype
                        )
                if failure is not None:
                    self._discard_owned_sequences(active_keys)
            else:
                tokens = ints_to_frame(
                    0,
                    [0] * (num_rows * num_tokens),
                    dtype="int64",
                    shape=(num_rows, num_tokens),
                )
                valid = ints_to_frame(0, [0] * num_rows)
            for seq_key, began, status in zip(message.keys, begun, row_statuses):
                if began is SpeculatorStatusCode.OK and status is not began:
                    registry.desync(seq_key, owner=self.session_id)

        if failure is not None:
            fail(SpeculatorStatusCode.INTERNAL_ERROR, failure)
            return
        base = self.next_result_slot
        self.next_result_slot += _RESULT_FRAMES
        self._send(
            ProposalResponse(
                batch_id=message.batch_id,
                sequence_number=sequence_number,
                result_slot=base,
            ),
            request_id,
            frames=[
                msgspec.structs.replace(tokens, slot=base),
                msgspec.structs.replace(valid, slot=base + 1),
                ints_to_frame(base + 2, [int(s) for s in row_statuses]),
            ],
        )

    def _close(self, message: CloseSequence, request_id: int) -> None:
        with self.server.adapter_turn():
            self._discard_owned_sequences(
                tuple(key for key in message.keys if self._owns(key))
            )
        deadline = time.monotonic() + self.server.send_timeout_s
        for key in message.keys:
            if self._owns(key):
                self._ack(key, request_id, deadline=deadline)
            else:
                self._ack(
                    key,
                    request_id,
                    SpeculatorStatusCode.PROTOCOL_ERROR,
                    "key belongs to another verifier",
                    deadline=deadline,
                )

    def _discard_owned_sequences(self, keys: tuple[SequenceKey, ...]) -> None:
        """Drop adapter and registry state after an ambiguous adapter failure."""
        self.server._retire_sequences(
            tuple(
                key
                for key in keys
                if (entry := self.server.registry.get(key)) is not None
                and entry.owner == self.session_id
            )
        )

    def _take_frames(self, base: int, count: int) -> list[DataFrame | None]:
        return [self.inbound.pop(base + i, None) for i in range(count)]


def _frame_problems(
    frames: list[DataFrame | None],
    schema: TargetFeatureSchema,
    num_tokens: int,
    num_sequences: int,
) -> str | None:
    for i, (spec, frame) in enumerate(zip(schema.slots, frames)):
        if frame is None:
            if spec.optional:
                continue
            return f"required feature slot {i} ({spec.kind}) is missing"
        num_rows = num_sequences if spec.per_sequence else num_tokens
        if frame.shape[:1] != (num_rows,):
            return (
                f"feature slot {i} ({spec.kind}) has shape {frame.shape}, "
                f"expected {num_rows} rows"
            )
        if tuple(frame.shape[1:]) != tuple(spec.trailing_shape):
            return (
                f"feature slot {i} ({spec.kind}) has trailing shape "
                f"{tuple(frame.shape[1:])}, schema says {spec.trailing_shape}"
            )
        if frame.dtype != spec.dtype:
            return (
                f"feature slot {i} ({spec.kind}) has dtype {frame.dtype}, "
                f"schema says {spec.dtype}"
            )
        try:
            validate_frame(frame)
        except TransportError as e:
            return f"feature slot {i} ({spec.kind}): {e}"
    return None


def _result_well_formed(
    result: StandaloneProposalResult, num_rows: int, num_tokens: int
) -> bool:
    if (
        result.token_ids.shape != (num_rows, num_tokens)
        or result.valid_lengths.shape != (num_rows,)
        or len(result.row_statuses) != num_rows
        or result.token_ids.dtype not in ("int32", "int64")
        or result.valid_lengths.dtype not in ("int32", "int64")
        or any(
            not isinstance(status, SpeculatorStatusCode)
            for status in result.row_statuses
        )
    ):
        return False
    try:
        validate_integer_frame(result.token_ids)
        valid_lengths = frame_to_ints(result.valid_lengths)
    except TransportError:
        return False
    return all(0 <= length <= num_tokens for length in valid_lengths)


def _checksum(frames: list[DataFrame | None]) -> int:
    value = 0
    for frame in frames:
        if frame is not None:
            value = zlib.crc32(frame.payload, value)
    return value & 0x7FFFFFFF


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    """Serve a fake drafter; model adapters register here once they exist."""
    parser = argparse.ArgumentParser(
        description="Standalone speculator server for remote draft placement."
    )
    parser.add_argument(
        "--endpoint", required=True, help="tcp://host:port or unix:///path"
    )
    parser.add_argument(
        "--adapter", choices=("fake",), default="fake", help="drafter adapter to serve"
    )
    parser.add_argument("--vocab-size", type=int, default=32000)
    parser.add_argument("--allow-remote-connections", action="store_true")
    parser.add_argument("--max-pending-bytes", type=int, default=1 << 30)
    parser.add_argument("--max-connections", type=int, default=128)
    parser.add_argument("--max-handshakes", type=int, default=16)
    parser.add_argument("--max-adapter-waiters", type=int, default=64)
    parser.add_argument("--partial-frame-timeout-s", type=float, default=30.0)
    parser.add_argument("--shutdown-timeout-s", type=float, default=5.0)
    parser.add_argument("--target-fingerprint", default="")
    parser.add_argument("--tokenizer-fingerprint", default="")
    parser.add_argument(
        "--max-batch-size", type=int, default=DEFAULT_LIMITS.max_batch_size
    )
    parser.add_argument(
        "--max-model-len", type=int, default=DEFAULT_LIMITS.max_model_len
    )
    args = parser.parse_args(argv)
    limits = RemoteServerLimits(
        max_batch_size=args.max_batch_size,
        max_feature_tokens=DEFAULT_LIMITS.max_feature_tokens,
        max_sequences=DEFAULT_LIMITS.max_sequences,
        max_model_len=args.max_model_len,
    )
    server = RemoteDraftServer(
        FakeDraftAdapter(vocab_size=args.vocab_size),
        limits=limits,
        target_fingerprint=args.target_fingerprint,
        tokenizer_fingerprint=args.tokenizer_fingerprint,
        allow_remote_connections=args.allow_remote_connections,
        max_pending_bytes=args.max_pending_bytes,
        max_connections=args.max_connections,
        max_handshakes=args.max_handshakes,
        max_adapter_waiters=args.max_adapter_waiters,
        partial_frame_timeout_s=args.partial_frame_timeout_s,
        shutdown_timeout_s=args.shutdown_timeout_s,
    )
    endpoint = server.start(args.endpoint)
    # Parent processes read the resolved endpoint (tcp port 0 is common
    # in tests) from the first stdout line.
    print(endpoint, flush=True)
    try:
        server.wait()
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()


if __name__ == "__main__":
    main()
