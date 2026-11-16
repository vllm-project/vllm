# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Verifier-side session with one standalone speculator server.

The session owns the connection, the negotiated schema and limits, the
per-sequence client state machine, and the slot bookkeeping that keeps
stale data-plane frames from ever being read as current results. Runner
code hands it placement-neutral RemoteProposalBatch objects and receives
SpeculatorOutput back; it never sees protocol structs or frames.

Failure semantics follow the MVP policy: a row whose remote state is in
doubt (timeout, desync, stale generation) stops receiving remote proposals
for the rest of the request and reports ``valid_length=0`` every round.
Automatic re-prefill recovery is a later step; ``open_sequence`` on a
desynced sequence already bumps the generation so that step can build on
it. A lost connection is re-established with backoff when new sequences
are opened, so only requests that were running at the time lose
speculation.
"""

import contextlib
import itertools
import math
import time
import zlib
from collections.abc import Callable, Collection, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch

from vllm.logger import init_logger
from vllm.v1.spec_decode.proposal import ProposalHandle, SpeculatorOutput
from vllm.v1.spec_decode.remote.capabilities import (
    SpeculatorPlacementCapabilities,
    placement_incompatibilities,
)
from vllm.v1.spec_decode.remote.protocol import (
    AdvanceAndPropose,
    CloseSequence,
    ErrorReply,
    Hello,
    HelloAck,
    MessageEnvelope,
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
    limit_shortfalls,
)
from vllm.v1.spec_decode.remote.state import (
    FeatureBatch,
    HandleState,
    InvalidStateTransition,
    RemoteProposalBatch,
    RemoteProposalResult,
    SequenceState,
    can_transition_sequence,
    transition_handle,
    transition_sequence,
)
from vllm.v1.spec_decode.remote.transport import (
    FRAME_DATA,
    INLINE_TRANSPORT,
    MAX_CONTROL_FRAME_BYTES,
    MAX_HANDSHAKE_BYTES,
    ConnectionClosed,
    DataFrame,
    FramedConnection,
    TransportError,
    TransportTimeout,
    check_data_frame_size,
    decode_data_frame,
    frame_to_ints,
)
from vllm.v1.spec_decode.remote.transport import connect as transport_connect

if TYPE_CHECKING:
    from vllm.config.speculative import RemoteDraftConfig

logger = init_logger(__name__)

# A completed round is delivered as three data frames at consecutive slots:
# draft tokens, per-row valid lengths, per-row status codes.
RESULT_FRAMES = 3

_RECONNECT_BACKOFF_S = 1.0
_MAX_RECONNECT_BACKOFF_S = 30.0

_DESYNC_STATUSES = frozenset(
    {
        SpeculatorStatusCode.STALE_GENERATION,
        SpeculatorStatusCode.SEQUENCE_DESYNC,
        SpeculatorStatusCode.ROUND_MISMATCH,
    }
)


class RemoteDraftError(Exception):
    """Remote speculator failure the configured policy does not absorb."""


def tensor_to_frame(slot: int, tensor: torch.Tensor) -> DataFrame:
    """Serialize a tensor into an inline data frame (host copy)."""
    dtype = str(tensor.dtype).removeprefix("torch.")
    shape = tuple(tensor.shape)
    check_data_frame_size(slot, dtype, shape)
    flat = tensor.detach().reshape(-1)
    if flat.device.type != "cpu":
        flat = flat.cpu()
    flat = flat.contiguous()
    payload = flat.view(torch.uint8).numpy().tobytes() if flat.numel() else b""
    return DataFrame(
        slot=slot,
        dtype=dtype,
        shape=shape,
        payload=payload,
    )


def frame_to_tensor(
    frame: DataFrame, device: torch.device | str | None = None
) -> torch.Tensor:
    """Materialize an inline data frame as a tensor on ``device``."""
    dtype = getattr(torch, frame.dtype, None)
    if not isinstance(dtype, torch.dtype):
        raise TransportError(f"unknown dtype {frame.dtype!r} in data frame")
    numel = math.prod(frame.shape)
    if numel == 0:
        tensor = torch.empty(frame.shape, dtype=dtype)
    else:
        expected = numel * dtype.itemsize
        if len(frame.payload) != expected:
            raise TransportError(
                f"data frame {frame.slot} has {len(frame.payload)} payload "
                f"bytes, expected {expected} for {frame.dtype} {frame.shape}"
            )
        tensor = torch.frombuffer(bytearray(frame.payload), dtype=dtype)
        tensor = tensor.reshape(frame.shape)
    return tensor if device is None else tensor.to(device)


@dataclass(frozen=True)
class VerifierIdentity:
    """What the verifier reports about itself in HELLO."""

    verifier_instance_id: str
    target_fingerprint: str
    tokenizer_fingerprint: str
    method: str
    num_speculative_tokens: int
    parallel_drafting: bool
    provided_features: tuple[str, ...]
    """TargetFeatureKind values the verifier can transport every round."""
    max_num_seqs: int
    max_num_batched_tokens: int
    max_model_len: int
    vocab_size: int
    """Target vocabulary size; draft tokens outside it are rejected."""
    draft_sample_method: str = "greedy"
    draft_checkpoint_fingerprint: str = ""
    uses_multi_module: bool = False


@dataclass
class _SequenceRecord:
    key: SequenceKey
    state: SequenceState
    prefilled_tokens: int = 0
    last_status: SpeculatorStatusCode = SpeculatorStatusCode.OK


@dataclass
class RemoteProposalHandle(ProposalHandle):
    """Dispatch receipt for one remote round; collect exactly once."""

    session_epoch: int
    batch_id: int
    keys: tuple[SequenceKey, ...]
    active_rows: tuple[int, ...]
    """Rows that were actually sent; the rest report valid_length=0."""
    num_speculative_tokens: int
    device: torch.device
    deadline: float
    session_id: str = ""
    request_id: int = -1
    input_slot: int = -1
    state: HandleState = HandleState.CREATED
    failed_rows: dict[int, SpeculatorStatusCode] = field(default_factory=dict)
    """Rows that failed at dispatch, reported once by collect."""


class RemoteDraftSession:
    """One verifier's connection to one standalone speculator server.

    Not thread-safe: the model runner drives it from its own thread. The
    ``clock`` is injectable so timeout behaviour is testable without
    sleeping; it must be monotonic and in seconds.
    """

    def __init__(
        self,
        endpoint: str,
        identity: VerifierIdentity,
        *,
        transport: str = "auto",
        failure_policy: str = "target_only",
        request_timeout_ms: int = 100,
        startup_timeout_ms: int = 30_000,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if transport not in ("auto", INLINE_TRANSPORT):
            raise NotImplementedError(
                f"remote_draft.transport={transport!r} has no data-plane "
                "implementation yet; use 'auto' or 'inline'"
            )
        if failure_policy not in ("error", "target_only"):
            raise ValueError(f"unknown failure_policy {failure_policy!r}")
        self._endpoint = endpoint
        self._identity = identity
        self._offered_transports = (INLINE_TRANSPORT,)
        self._failure_policy = failure_policy
        self._request_timeout_s = request_timeout_ms / 1000.0
        self._startup_timeout_s = startup_timeout_ms / 1000.0
        self._clock = clock
        self._reconnect_at = -math.inf
        self._reconnect_backoff_s = _RECONNECT_BACKOFF_S

        self._conn: FramedConnection | None = None
        self._alive = False
        self._session_id = ""
        self._session_epoch = -1
        self._peer_protocol_minor = 0
        self._capabilities: SpeculatorPlacementCapabilities | None = None
        self._feature_schema: TargetFeatureSchema | None = None
        self._limits: RemoteServerLimits | None = None
        self._selected_transport = ""

        self._next_request_id = 1
        self._next_slot = 0
        self._last_batch_id = -1
        self._last_sequence_number = -1
        self._inflight_batch_id: int | None = None
        self._inbound: dict[int, DataFrame] = {}
        self._inbound_sizes: dict[int, int] = {}
        self._inbound_bytes = 0
        self._max_pending_frames = 2 * RESULT_FRAMES
        self._max_pending_bytes = 0
        self._max_result_frame_bytes = 0
        self._records: dict[int, _SequenceRecord] = {}
        self._retiring: set[SequenceKey] = set()
        # Survives reconnects so closed requests need no per-id tombstones.
        self._next_generation = 0
        self._abandoned_batches: set[int] = set()

    @classmethod
    def from_config(
        cls,
        config: "RemoteDraftConfig",
        identity: VerifierIdentity,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> "RemoteDraftSession":
        return cls(
            config.endpoint,
            identity,
            transport=config.transport,
            failure_policy=config.failure_policy,
            request_timeout_ms=config.request_timeout_ms,
            startup_timeout_ms=config.startup_timeout_ms,
            clock=clock,
        )

    # ------------------------------------------------------------------
    # Negotiated state
    # ------------------------------------------------------------------

    @property
    def connected(self) -> bool:
        return self._alive

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def session_epoch(self) -> int:
        return self._session_epoch

    @property
    def selected_transport(self) -> str:
        return self._selected_transport

    @property
    def capabilities(self) -> SpeculatorPlacementCapabilities:
        if self._capabilities is None:
            raise RemoteDraftError("session is not connected")
        return self._capabilities

    @property
    def feature_schema(self) -> TargetFeatureSchema:
        if self._feature_schema is None:
            raise RemoteDraftError("session is not connected")
        return self._feature_schema

    @property
    def limits(self) -> RemoteServerLimits:
        if self._limits is None:
            raise RemoteDraftError("session is not connected")
        return self._limits

    def sequence_state(self, key: SequenceKey) -> SequenceState | None:
        """Current client-side state of ``key``, or None if unknown/stale."""
        record = self._records.get(key.sequence_id)
        if record is None or record.key != key:
            return None
        return record.state

    @property
    def pending_frames(self) -> int:
        """Buffered inbound data frames not yet consumed (diagnostic)."""
        return len(self._inbound)

    # ------------------------------------------------------------------
    # Handshake
    # ------------------------------------------------------------------

    def connect(self, *, timeout_s: float | None = None) -> None:
        """Connect and run the HELLO handshake; raises on any rejection.

        May be called again after the connection is lost; sequence numbers,
        slots, and batch ids restart with the new connection, and handles
        dispatched on the old one are fenced off at collect time.
        ``timeout_s`` defaults to the configured startup timeout.
        """
        if self._conn is not None:
            raise RemoteDraftError("session is already connected")
        self._reset_connection_state()
        if timeout_s is None:
            timeout_s = self._startup_timeout_s
        deadline = self._clock() + timeout_s
        try:
            self._conn = transport_connect(self._endpoint, timeout=timeout_s)
        except TransportError as e:
            raise RemoteDraftError(
                f"cannot connect to remote draft server at {self._endpoint}: {e}"
            ) from e
        hello = self._hello()
        try:
            self._conn.set_receive_limits(
                max_control_bytes=MAX_HANDSHAKE_BYTES,
                max_data_bytes=0,
                max_buffered_bytes=MAX_HANDSHAKE_BYTES + 5,
            )
            request_id = self._send(hello, deadline=deadline)
            envelope, ack = self._await(
                lambda env, msg: (
                    isinstance(msg, HelloAck) and env.request_id == request_id
                ),
                deadline,
                request_ids=(request_id,),
            )
            if envelope.session_id != ack.session_id:
                raise RemoteDraftError(
                    "session ID in HELLO acknowledgement does not match its envelope"
                )
        except TransportTimeout:
            self._drop_connection()
            raise RemoteDraftError(
                f"HELLO handshake with {self._endpoint} timed out after "
                f"{timeout_s:.1f}s"
            ) from None
        except (TransportError, RemoteDraftError) as e:
            self._drop_connection()
            raise RemoteDraftError(f"HELLO handshake failed: {e}") from e
        reasons = self._handshake_rejections(hello, ack)
        if reasons:
            self._drop_connection()
            raise RemoteDraftError(
                "remote draft placement rejected: " + "; ".join(reasons)
            )
        try:
            rows = self._identity.max_num_seqs or ack.limits.max_batch_size
            slot = (1 << 63) - 1
            result_sizes = (
                check_data_frame_size(
                    slot, "int64", (rows, self._identity.num_speculative_tokens)
                ),
                check_data_frame_size(slot, "int64", (rows,)),
                check_data_frame_size(slot, "int32", (rows,)),
            )
            # One late result can precede the current one. Bound both decoded
            # pending frames and raw frames buffered by duplex progress.
            self._max_pending_bytes = 2 * sum(result_sizes)
            self._max_result_frame_bytes = max(result_sizes)
            self._set_result_receive_budget()
        except TransportError as e:
            self._drop_connection()
            raise RemoteDraftError(f"invalid remote result byte budget: {e}") from e
        self._session_id = ack.session_id
        self._session_epoch = ack.session_epoch
        self._peer_protocol_minor = envelope.protocol_minor
        self._capabilities = ack.capabilities
        self._feature_schema = ack.feature_schema
        self._limits = ack.limits
        self._selected_transport = ack.selected_transport
        self._alive = True
        logger.info(
            "Remote draft session %s (epoch %d) established with server %s "
            "over %s; schema %d with %d feature slots",
            ack.session_id,
            ack.session_epoch,
            ack.server_id,
            ack.selected_transport,
            ack.feature_schema.schema_id,
            len(ack.feature_schema.slots),
        )

    def maybe_reconnect(self, *, deadline: float | None = None) -> bool:
        """Re-establish a lost connection, at most once per backoff period.

        Returns whether the session is connected. Call it between rounds;
        sequences opened on the old connection stay INVALID until they are
        closed or re-opened. Each attempt is bounded by the request timeout
        because it runs on the runner thread. An explicit ``deadline`` also
        caps the remaining connect and handshake time.
        """
        if self._alive:
            return True
        if self._inflight_batch_id is not None or self._clock() < self._reconnect_at:
            return False
        timeout_s = self._request_timeout_s
        if deadline is not None:
            timeout_s = min(timeout_s, deadline - self._clock())
        if timeout_s <= 0:
            return False
        try:
            self.connect(timeout_s=timeout_s)
        except RemoteDraftError as e:
            self._reconnect_at = self._clock() + self._reconnect_backoff_s
            logger.warning(
                "Remote draft server at %s is unavailable; retrying in %.0fs: %s",
                self._endpoint,
                self._reconnect_backoff_s,
                e,
            )
            self._reconnect_backoff_s = min(
                2 * self._reconnect_backoff_s, _MAX_RECONNECT_BACKOFF_S
            )
            return False
        self._reconnect_backoff_s = _RECONNECT_BACKOFF_S
        return True

    def _reset_connection_state(self) -> None:
        """Drop bookkeeping scoped to the previous connection.

        The server numbers proposal responses and reads data slots per
        connection; carrying the old counters across a reconnect would make
        every fresh response look stale and be discarded.
        """
        self._alive = False
        self._session_id = ""
        self._session_epoch = -1
        self._peer_protocol_minor = 0
        self._capabilities = None
        self._feature_schema = None
        self._limits = None
        self._selected_transport = ""
        self._next_request_id = 1
        self._next_slot = 0
        self._last_batch_id = -1
        self._last_sequence_number = -1
        self._inflight_batch_id = None
        self._inbound.clear()
        self._inbound_sizes.clear()
        self._inbound_bytes = 0
        self._max_pending_bytes = 0
        self._max_result_frame_bytes = 0
        self._abandoned_batches.clear()
        self._retiring.clear()

    def _hello(self) -> Hello:
        identity = self._identity
        return Hello(
            verifier_instance_id=identity.verifier_instance_id,
            target_fingerprint=identity.target_fingerprint,
            tokenizer_fingerprint=identity.tokenizer_fingerprint,
            method=identity.method,
            num_speculative_tokens=identity.num_speculative_tokens,
            parallel_drafting=identity.parallel_drafting,
            supported_transports=self._offered_transports,
            draft_checkpoint_fingerprint=identity.draft_checkpoint_fingerprint,
            max_num_seqs=identity.max_num_seqs,
            max_num_batched_tokens=identity.max_num_batched_tokens,
            max_model_len=identity.max_model_len,
        )

    def _handshake_rejections(self, hello: Hello, ack: HelloAck) -> list[str]:
        identity = self._identity
        reasons = limit_shortfalls(hello, ack.limits)
        if ack.selected_transport == INLINE_TRANSPORT:
            reasons.extend(
                inline_feature_frame_rejections(hello, ack.feature_schema, ack.limits)
            )
        if ack.selected_transport not in self._offered_transports:
            reasons.append(
                f"server selected transport {ack.selected_transport!r} which "
                f"the verifier did not offer {self._offered_transports}"
            )
        reasons.extend(
            placement_incompatibilities(
                ack.capabilities,
                parallel_drafting=identity.parallel_drafting,
                draft_sample_method=identity.draft_sample_method,
                provided_features=identity.provided_features,
                uses_multi_module=identity.uses_multi_module,
            )
        )
        provided = set(identity.provided_features)
        for slot in ack.feature_schema.slots:
            if not slot.optional and slot.kind not in provided:
                reasons.append(
                    f"feature schema requires {slot.kind!r} which the verifier "
                    "cannot provide"
                )
        return reasons

    # ------------------------------------------------------------------
    # Sequence lifecycle
    # ------------------------------------------------------------------

    def open_sequence(self, sequence_id: int) -> SequenceKey:
        """Register draft state for a request; returns its SequenceKey.

        Re-opening a desynced or invalidated sequence bumps the generation
        so the server discards anything it still holds for the old one.
        """
        return self.open_sequences((sequence_id,))[0]

    def open_sequences(
        self, sequence_ids: Sequence[int], *, deadline: float | None = None
    ) -> tuple[SequenceKey, ...]:
        """Pipeline request opens within one preparation budget, including reconnect."""
        self._require_no_inflight("open sequences")
        if len(set(sequence_ids)) != len(sequence_ids):
            raise ValueError("duplicate sequence ids in open_sequences")
        for sequence_id in sequence_ids:
            record = self._records.get(sequence_id)
            if record is not None and record.state not in (
                SequenceState.DESYNCED,
                SequenceState.INVALID,
            ):
                raise RemoteDraftError(
                    f"sequence {sequence_id} is already open ({record.state.value})"
                )
        if not sequence_ids:
            return ()
        if deadline is None:
            deadline = self.preparation_deadline()
        try:
            self.maybe_reconnect(deadline=deadline)
            keys = self._open_many(sequence_ids, deadline=deadline)
            if all(
                self.sequence_state(key) is SequenceState.INVALID for key in keys
            ) and self.maybe_reconnect(deadline=deadline):
                # Reconnect succeeded after open found the old connection dead.
                keys = self._open_many(sequence_ids, deadline=deadline)
            return keys
        except MemoryError:
            # No keys were returned to the caller, even if the peer opened them.
            self._invalidate("remote sequence registration allocation failed")
            for sequence_id in sequence_ids:
                record = self._records.pop(sequence_id, None)
                if record is not None:
                    self._retiring.discard(record.key)
            raise

    def _open_many(
        self, sequence_ids: Sequence[int], *, deadline: float
    ) -> tuple[SequenceKey, ...]:
        records: list[_SequenceRecord] = []
        for sequence_id in sequence_ids:
            generation = self._next_generation
            self._next_generation += 1
            key = SequenceKey(
                verifier_instance_id=self._identity.verifier_instance_id,
                sequence_id=sequence_id,
                generation=generation,
            )
            record = _SequenceRecord(key, SequenceState.OPENING)
            self._records[sequence_id] = record
            records.append(record)
        keys = tuple(record.key for record in records)
        if not self._alive:
            for record in records:
                record.state = SequenceState.INVALID
                record.last_status = SpeculatorStatusCode.INTERNAL_ERROR
            # The connection loss and each reconnect attempt already warned.
            self._raise_if_strict(
                "cannot open sequences: session is not connected",
                quiet=True,
            )
            return keys
        replies, deadline = self._sequence_rpcs(
            [(OpenSequence(key=key), key, None) for key in keys], deadline=deadline
        )
        failures: list[str] = []
        for record, (status, detail) in zip(records, replies):
            if status is SpeculatorStatusCode.OK:
                record.state = transition_sequence(
                    record.state, SequenceState.PREFILLING
                )
            else:
                if status is SpeculatorStatusCode.QUEUE_FULL:
                    self._latch_target_only(record, status)
                else:
                    self._apply_status(record, status)
                failures.append(
                    f"sequence {record.key.sequence_id}: {status.name} {detail}"
                )
        self._retire_terminal_sequences(deadline)
        if failures:
            self._raise_if_strict("open_sequences failed for " + "; ".join(failures))
        return keys

    def prefill(
        self, key: SequenceKey, features: FeatureBatch, *, is_final: bool
    ) -> None:
        """Send one prompt chunk; ``is_final`` makes the sequence READY."""
        self.prefill_many([(key, features, is_final)])

    def prefill_many(
        self,
        chunks: Sequence[tuple[SequenceKey, FeatureBatch, bool]],
        *,
        deadline: float | None = None,
    ) -> None:
        """Send one prompt chunk for each of several sequences.

        Each entry is ``(key, features, is_final)``; ``is_final`` makes the
        sequence READY. All chunks are written back to back and acknowledged
        together, so a step that prefills N requests pays one round trip
        instead of N. Chunks are applied in order with client-tracked
        offsets. A sequence that is already latched target-only silently
        ignores prefill so the request keeps running without speculation.
        """
        if deadline is None:
            deadline = self.preparation_deadline()
        self._require_no_inflight("prefill a sequence")
        if len({key for key, _, _ in chunks}) != len(chunks):
            raise ValueError("at most one prefill chunk per sequence per call")
        prepared: list[tuple[_SequenceRecord, PrefillChunk, list[DataFrame]]] = []
        failures: list[str] = []
        for key, features, is_final in chunks:
            record = self._current_record(key)
            if record.state is not SequenceState.PREFILLING:
                if record.state in (
                    SequenceState.TARGET_ONLY,
                    SequenceState.INVALID,
                    SequenceState.DESYNCED,
                ):
                    continue
                raise RemoteDraftError(
                    f"sequence {key.sequence_id} cannot accept prefill in state "
                    f"{record.state.value}"
                )
            num_tokens = self._chunk_length(features)
            self._check_features(features, num_tokens=num_tokens, num_sequences=1)
            limits = self.limits
            if num_tokens > limits.max_feature_tokens:
                self._latch_target_only(record, SpeculatorStatusCode.QUEUE_FULL)
                failures.append(
                    f"sequence {key.sequence_id}: chunk of {num_tokens} tokens "
                    "exceeds the server max_feature_tokens="
                    f"{limits.max_feature_tokens}"
                )
                continue
            if record.prefilled_tokens + num_tokens > limits.max_model_len:
                self._latch_target_only(record, SpeculatorStatusCode.OUT_OF_MEMORY)
                failures.append(
                    f"sequence {key.sequence_id}: prefill would exceed the "
                    f"server max_model_len={limits.max_model_len}"
                )
                continue
            base = self._reserve_slots(len(self.feature_schema.slots))
            frames = self._feature_frames(
                base, features, token_rows=None, sequence_rows=None
            )
            chunk = PrefillChunk(
                key=key,
                offset=record.prefilled_tokens,
                num_tokens=num_tokens,
                is_final=is_final,
                feature_slot=base,
                checksum=_checksum(frames),
            )
            prepared.append((record, chunk, frames))
        replies, deadline = self._sequence_rpcs(
            [(chunk, chunk.key, frames) for _, chunk, frames in prepared],
            deadline=deadline,
        )
        for (record, chunk, _), (status, detail) in zip(prepared, replies):
            if status is SpeculatorStatusCode.OK:
                record.prefilled_tokens += chunk.num_tokens
                if chunk.is_final:
                    record.state = transition_sequence(
                        record.state, SequenceState.READY
                    )
                continue
            self._apply_status(record, status)
            failures.append(
                f"sequence {chunk.key.sequence_id} at offset {chunk.offset}: "
                f"{status.name} {detail}"
            )
        self._retire_terminal_sequences(deadline)
        if failures:
            self._raise_if_strict("prefill failed for " + "; ".join(failures))

    def close_sequences(self, keys: tuple[SequenceKey, ...]) -> None:
        """Release remote state; unknown, stale, and closed keys are no-ops."""
        self._require_no_inflight("close sequences")
        pending: list[SequenceKey] = []
        for key in keys:
            record = self._records.get(key.sequence_id)
            if record is None or record.key != key:
                continue
            if record.state in (SequenceState.CLOSING, SequenceState.CLOSED):
                continue
            if record.state is SequenceState.INVALID:
                del self._records[key.sequence_id]
                self._retiring.discard(key)
                continue
            record.state = transition_sequence(record.state, SequenceState.CLOSING)
            pending.append(key)
        if pending and self._alive:
            self._retiring.difference_update(pending)
            self._close_remote(tuple(pending))
        for key in pending:
            record = self._records.pop(key.sequence_id, None)
            if record is not None and record.state is SequenceState.CLOSING:
                record.state = transition_sequence(record.state, SequenceState.CLOSED)

    def _close_remote(self, keys: tuple[SequenceKey, ...]) -> set[SequenceKey]:
        """Close ``keys`` on the server; returns the acknowledged ones."""
        deadline = self._clock() + self._request_timeout_s
        remaining = set(keys)
        sent_close = False
        try:
            request_id = self._send(CloseSequence(keys=keys), deadline=deadline)
            sent_close = True
            while remaining:
                _, ack = self._await(
                    lambda env, msg: (
                        isinstance(msg, SequenceAck) and env.request_id == request_id
                    ),
                    deadline,
                    request_ids=(request_id,),
                )
                remaining.discard(ack.key)
        except TransportTimeout:
            if not sent_close:
                self._invalidate("close request did not finish sending before deadline")
            else:
                logger.warning(
                    "Remote draft server did not acknowledge closing %d sequences "
                    "before the deadline; releasing them locally",
                    len(remaining),
                )
        except TransportError as e:
            self._invalidate(f"connection lost while closing sequences: {e}")
        except RemoteDraftError as e:
            logger.warning("Remote draft server rejected close: %s", e)
        return set(keys) - remaining

    # ------------------------------------------------------------------
    # Proposal rounds
    # ------------------------------------------------------------------

    def preparation_deadline(self) -> float:
        """Capture the existing request budget before preparing remote features."""
        return self._clock() + self._request_timeout_s

    def fail_sequences(
        self,
        keys: Collection[SequenceKey],
        *,
        status: SpeculatorStatusCode = SpeculatorStatusCode.OUT_OF_MEMORY,
        detail: str,
        deadline: float | None = None,
    ) -> None:
        """Latch local preparation failures until the affected requests close.

        Preparation can advance the verifier's bookkeeping before the peer
        receives its context. Keeping the records terminal prevents a later
        step from resuming with stale own-KV. An outstanding proposal uses
        the same connection teardown as a transport failure. Idle remote state
        is retired without waiting for ACKs, using only the preparation budget
        captured by the caller before the failed work.
        """
        for key in keys:
            record = self._records.get(key.sequence_id)
            if record is not None and record.key == key:
                self._latch_target_only(record, status)
        if self._inflight_batch_id is not None:
            self._invalidate(detail)
        self._retire_terminal_sequences(deadline)
        self._raise_if_strict(detail)

    def _retire_terminal_sequences(self, deadline: float | None) -> None:
        """Reclaim terminal peer state at an idle boundary within its budget."""
        if not self._alive or not self._retiring or self._inflight_batch_id is not None:
            return
        remaining = 0.0 if deadline is None else deadline - self._clock()
        if remaining > 0:
            # A send failure already tears down connection-owned peer KV.
            with contextlib.suppress(TransportError):
                self._send_retirements(
                    self._require_connection(), time.monotonic() + remaining
                )
        elif not any(
            record.state
            in (SequenceState.OPENING, SequenceState.PREFILLING, SequenceState.READY)
            for record in self._records.values()
        ):
            self._invalidate("terminal cleanup exhausted the request budget")
        # A healthy sibling flushes the bounded queue on its next RPC.

    def dispatch(
        self, batch: RemoteProposalBatch, *, deadline: float | None = None
    ) -> RemoteProposalHandle:
        """Send one round to the server without waiting for the result.

        Rows whose sequence is not READY (latched target-only, desynced,
        stale key, unknown) are skipped locally and report
        ``valid_length=0`` at collect time; they generate no wire traffic.

        At most one proposal may be in flight per session: collect the
        previous handle before dispatching the next round.
        """
        if deadline is None:
            deadline = self.preparation_deadline()
        self._require_no_inflight(f"dispatch batch {batch.batch_id}")
        effective_k = batch.num_speculative_tokens
        if effective_k is None:
            effective_k = self._identity.num_speculative_tokens
        if not 0 <= effective_k <= self._identity.num_speculative_tokens:
            raise ValueError("round draft budget must be within the negotiated maximum")
        if batch.batch_id <= self._last_batch_id:
            raise ValueError(
                f"batch_id {batch.batch_id} is not newer than "
                f"{self._last_batch_id}; batch ids must increase per session"
            )
        if len(set(batch.keys)) != len(batch.keys):
            raise ValueError("batch contains duplicate sequence keys")
        batch_size = len(batch.keys)
        if batch.accepted_counts.shape != (batch_size,):
            raise ValueError(
                f"accepted_counts must be [{batch_size}], got "
                f"{tuple(batch.accepted_counts.shape)}"
            )
        if batch.recovery_tokens.shape != (batch_size,):
            raise ValueError(
                f"recovery_tokens must be [{batch_size}], got "
                f"{tuple(batch.recovery_tokens.shape)}"
            )
        query_lens = _query_lens(batch)
        if self._feature_schema is not None:
            self._check_features(
                batch.features, num_tokens=sum(query_lens), num_sequences=batch_size
            )
        self._last_batch_id = batch.batch_id
        ready_rows = [i for i, key in enumerate(batch.keys) if self._is_ready(key)]
        handle = RemoteProposalHandle(
            session_epoch=self._session_epoch,
            session_id=self._session_id,
            batch_id=batch.batch_id,
            keys=batch.keys,
            active_rows=self._admit(batch.keys, ready_rows, query_lens),
            num_speculative_tokens=self._identity.num_speculative_tokens,
            device=batch.recovery_tokens.device,
            deadline=deadline,
        )
        handle.failed_rows.update(
            dict.fromkeys(
                set(ready_rows).difference(handle.active_rows),
                SpeculatorStatusCode.QUEUE_FULL,
            )
        )
        if not handle.active_rows or not self._alive:
            handle.state = transition_handle(handle.state, HandleState.COMPLETED)
            return handle
        rows = token_rows = None
        if len(handle.active_rows) != batch_size:
            rows = torch.tensor(handle.active_rows, dtype=torch.long)
            token_rows = _token_rows(query_lens, handle.active_rows)
        num_inputs = 2 if batch.query_lens is None else 3
        base = self._reserve_slots(num_inputs + len(self.feature_schema.slots))
        frames = [
            tensor_to_frame(base, _select_rows(batch.accepted_counts, rows)),
            tensor_to_frame(base + 1, _select_rows(batch.recovery_tokens, rows)),
        ]
        if batch.query_lens is not None:
            active_lens = [query_lens[i] for i in handle.active_rows]
            frames.append(
                tensor_to_frame(base + 2, torch.tensor(active_lens, dtype=torch.int32))
            )
        frames.extend(
            self._feature_frames(
                base + num_inputs,
                batch.features,
                token_rows=token_rows,
                sequence_rows=rows,
            )
        )
        message = AdvanceAndPropose(
            batch_id=batch.batch_id,
            keys=tuple(batch.keys[i] for i in handle.active_rows),
            accepted_counts_slot=base,
            recovery_tokens_slot=base + 1,
            feature_slot=base + num_inputs,
            query_lens_slot=None if batch.query_lens is None else base + 2,
            num_speculative_tokens=effective_k,
        )
        failure = None
        if self._clock() >= handle.deadline:
            failure = SpeculatorStatusCode.TIMEOUT
        else:
            try:
                handle.request_id = self._send(
                    message, frames=frames, deadline=handle.deadline
                )
            except TransportTimeout:
                failure = SpeculatorStatusCode.TIMEOUT
            except TransportError as e:
                self._invalidate(f"connection lost while dispatching: {e}")
                failure = SpeculatorStatusCode.INTERNAL_ERROR
        if failure is not None:
            # Either nothing reached the server or the connection is gone,
            # so the round is complete without a response to wait for.
            handle.failed_rows.update(self._fail_rows(handle, failure))
            handle.state = transition_handle(handle.state, HandleState.COMPLETED)
            return handle
        for i in handle.active_rows:
            record = self._records[batch.keys[i].sequence_id]
            record.state = transition_sequence(record.state, SequenceState.IN_FLIGHT)
        handle.input_slot = base
        handle.state = transition_handle(handle.state, HandleState.DISPATCHED)
        self._inflight_batch_id = batch.batch_id
        return handle

    def collect(self, handle: RemoteProposalHandle) -> RemoteProposalResult:
        """Wait for the round dispatched as ``handle``; one-shot.

        Timeouts abandon the batch, latch the affected
        sequences target-only, and return an all-zero output; under
        ``failure_policy="error"`` they raise instead (after the same state
        updates).
        """
        if (
            handle.session_id != self._session_id
            or handle.session_epoch != self._session_epoch
        ):
            raise RemoteDraftError(
                f"handle for batch {handle.batch_id} belongs to session "
                f"{handle.session_id!r} (epoch {handle.session_epoch}); the "
                f"current session is {self._session_id!r} "
                f"(epoch {self._session_epoch})"
            )
        if handle.state is HandleState.COMPLETED:
            # Nothing was dispatched: rows were skipped or failed locally.
            handle.state = transition_handle(handle.state, HandleState.COLLECTED)
            return self._finalize(handle, dict(handle.failed_rows), None, None)
        if handle.state is not HandleState.DISPATCHED:
            raise InvalidStateTransition(
                f"cannot collect a {handle.state.value} handle "
                f"(batch {handle.batch_id})"
            )
        tokens = valid = None
        outcome = HandleState.COMPLETED
        try:
            _, response = self._await(
                lambda env, msg: (
                    isinstance(msg, ProposalResponse)
                    and msg.batch_id == handle.batch_id
                ),
                handle.deadline,
                request_ids=(handle.request_id,),
            )
        except TransportTimeout:
            # The deadline is exhausted. Do not delay fallback with an
            # advisory cancellation write; discard the late result instead.
            self._abandoned_batches.add(handle.batch_id)
            outcome = HandleState.TIMED_OUT
            statuses = self._fail_rows(handle, SpeculatorStatusCode.TIMEOUT)
        except TransportError as e:
            self._invalidate(
                f"connection lost while collecting batch {handle.batch_id}: {e}"
            )
            statuses = self._fail_rows(handle, SpeculatorStatusCode.INTERNAL_ERROR)
        except RemoteDraftError as e:
            logger.warning(
                "Batch %d failed with a session-level error: %s", handle.batch_id, e
            )
            statuses = self._fail_rows(handle, SpeculatorStatusCode.INTERNAL_ERROR)
        else:
            try:
                statuses, tokens, valid = self._apply_response(handle, response)
            except TransportError as e:
                self._invalidate(f"invalid buffered result: {e}")
                statuses = self._fail_rows(handle, SpeculatorStatusCode.INTERNAL_ERROR)
        if self._inflight_batch_id == handle.batch_id:
            self._inflight_batch_id = None
        handle.state = transition_handle(handle.state, outcome)
        handle.state = transition_handle(handle.state, HandleState.COLLECTED)
        return self._finalize(handle, {**handle.failed_rows, **statuses}, tokens, valid)

    def _finalize(
        self,
        handle: RemoteProposalHandle,
        statuses: dict[int, SpeculatorStatusCode],
        tokens: torch.Tensor | None,
        valid: torch.Tensor | None,
    ) -> RemoteProposalResult:
        deadline = handle.deadline
        if SpeculatorStatusCode.TIMEOUT in statuses.values():
            # A reported timeout must not start another write even if the
            # injectable clock has not yet observed the transport's deadline.
            deadline = min(deadline, self._clock())
        self._retire_terminal_sequences(deadline)
        result = self._build_result(handle, statuses, tokens, valid)
        # Only rows decided this round: rows latched earlier report their
        # old status every round and must not warn every decode step.
        failed = sorted(
            (i, status.name)
            for i, status in statuses.items()
            if status is not SpeculatorStatusCode.OK
        )
        if failed:
            self._raise_if_strict(f"batch {handle.batch_id} has failed rows: {failed}")
        return result

    def _apply_response(
        self, handle: RemoteProposalHandle, response: ProposalResponse
    ) -> tuple[
        dict[int, SpeculatorStatusCode], torch.Tensor | None, torch.Tensor | None
    ]:
        if response.status is not SpeculatorStatusCode.OK:
            return self._fail_rows(handle, response.status), None, None
        if response.result_slot is None:
            logger.warning("Batch %d succeeded without a result slot", handle.batch_id)
            return (
                self._fail_rows(handle, SpeculatorStatusCode.INTERNAL_ERROR),
                None,
                None,
            )
        frames = self._take_frames(response.result_slot, RESULT_FRAMES)
        token_frame, valid_frame, status_frame = frames
        if token_frame is None or valid_frame is None or status_frame is None:
            logger.warning(
                "Batch %d response references missing result frames at slot %d",
                handle.batch_id,
                response.result_slot,
            )
            return (
                self._fail_rows(handle, SpeculatorStatusCode.INTERNAL_ERROR),
                None,
                None,
            )
        num_active = len(handle.active_rows)
        expected = (num_active, handle.num_speculative_tokens)
        try:
            tokens = frame_to_tensor(token_frame)
            valid = frame_to_tensor(valid_frame)
            status_codes = frame_to_ints(status_frame)
        except TransportError as e:
            logger.warning(
                "Batch %d result frames are malformed: %s", handle.batch_id, e
            )
            return (
                self._fail_rows(handle, SpeculatorStatusCode.INTERNAL_ERROR),
                None,
                None,
            )
        if (
            tuple(tokens.shape) != expected
            or tuple(valid.shape) != (num_active,)
            or len(status_codes) != num_active
            or tokens.is_floating_point()
            or valid.is_floating_point()
        ):
            logger.warning(
                "Batch %d result has unexpected layout: tokens %s, valid %s, "
                "%d statuses for %d rows",
                handle.batch_id,
                tuple(tokens.shape),
                tuple(valid.shape),
                len(status_codes),
                num_active,
            )
            return (
                self._fail_rows(handle, SpeculatorStatusCode.INTERNAL_ERROR),
                None,
                None,
            )
        # Fail rows that propose ids outside the target vocab, which would
        # otherwise index out of bounds on the verifier GPU. Only the first
        # valid_length tokens are consumed; the rest is zeroed later.
        usable = _usable_positions(valid, handle.num_speculative_tokens)
        out_of_vocab = (
            ((tokens < 0) | (tokens >= self._identity.vocab_size)) & usable
        ).any(dim=1)
        statuses: dict[int, SpeculatorStatusCode] = {}
        for row, code, bad in zip(
            handle.active_rows, status_codes, out_of_vocab.tolist()
        ):
            status = _status_from_wire(code)
            if bad and status is SpeculatorStatusCode.OK:
                logger.warning(
                    "Batch %d row %d proposed token ids outside the target "
                    "vocabulary of %d",
                    handle.batch_id,
                    row,
                    self._identity.vocab_size,
                )
                status = SpeculatorStatusCode.INTERNAL_ERROR
            statuses[row] = status
            record = self._records.get(handle.keys[row].sequence_id)
            if record is not None and record.key == handle.keys[row]:
                self._apply_status(record, status)
        return statuses, tokens, valid

    def _build_result(
        self,
        handle: RemoteProposalHandle,
        statuses: dict[int, SpeculatorStatusCode],
        tokens: torch.Tensor | None,
        valid: torch.Tensor | None,
    ) -> RemoteProposalResult:
        batch_size = len(handle.keys)
        num_tokens = handle.num_speculative_tokens
        token_ids = torch.zeros(
            batch_size, num_tokens, dtype=torch.int64, device=handle.device
        )
        valid_lengths = torch.zeros(batch_size, dtype=torch.int32, device=handle.device)
        row_statuses = tuple(
            statuses[i] if i in statuses else self._skip_status(key)
            for i, key in enumerate(handle.keys)
        )
        if tokens is not None and valid is not None:
            rows = torch.tensor(handle.active_rows, dtype=torch.long)
            ok = torch.tensor(
                [statuses[i] is SpeculatorStatusCode.OK for i in handle.active_rows]
            )
            valid = torch.where(ok, valid.to(torch.int32), 0).clamp_(0, num_tokens)
            tokens = torch.where(_usable_positions(valid, num_tokens), tokens, 0)
            token_ids.index_copy_(
                0, rows.to(handle.device), tokens.to(handle.device, torch.int64)
            )
            valid_lengths.index_copy_(
                0, rows.to(handle.device), valid.to(handle.device)
            )
        return RemoteProposalResult(
            output=SpeculatorOutput(token_ids, valid_lengths),
            row_statuses=row_statuses,
        )

    def _fail_rows(
        self, handle: RemoteProposalHandle, status: SpeculatorStatusCode
    ) -> dict[int, SpeculatorStatusCode]:
        for i in handle.active_rows:
            key = handle.keys[i]
            record = self._records.get(key.sequence_id)
            if record is not None and record.key == key:
                self._apply_status(record, status)
        return dict.fromkeys(handle.active_rows, status)

    # ------------------------------------------------------------------
    # Health and teardown
    # ------------------------------------------------------------------

    def ping(self) -> Pong:
        """Round-trip liveness probe."""
        self._require_no_inflight("ping the server")
        nonce = self._next_request_id
        deadline = self._clock() + self._request_timeout_s
        try:
            request_id = self._send(Ping(nonce=nonce), deadline=deadline)
            _, pong = self._await(
                lambda env, msg: isinstance(msg, Pong) and msg.nonce == nonce,
                deadline,
                request_ids=(request_id,),
            )
        except TransportTimeout:
            raise RemoteDraftError("ping timed out") from None
        except TransportError as e:
            self._invalidate(f"connection lost during ping: {e}")
            raise RemoteDraftError(f"ping failed: {e}") from e
        return pong

    def close(self) -> None:
        """Close the connection; every sequence becomes INVALID."""
        self._invalidate("session closed")

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _send(
        self,
        message: Any,
        *,
        deadline: float,
        frames: list[DataFrame] | None = None,
    ) -> int:
        conn = self._require_connection()
        frames = frames or []
        # Translate the injectable session clock to the transport's monotonic
        # clock once, so every frame consumes the same remaining budget.
        budget = max(0.0, deadline - self._clock())
        wire_deadline = time.monotonic() + budget
        self._send_retirements(conn, wire_deadline)
        # Retirement ACKs can still be pending after their write queue emptied.
        request_id = self._next_request_id
        self._next_request_id += 1
        sent_frames = False
        try:
            for frame in frames:
                conn.send_data(frame, deadline=wire_deadline, drain_incoming=True)
                sent_frames = True
            if isinstance(message, (PrefillChunk, AdvanceAndPropose)):
                message.execution_timeout_ms = max(
                    0, int((deadline - self._clock()) * 1000)
                )
            conn.send_control(
                encode_message(
                    message, session_id=self._session_id, request_id=request_id
                ),
                deadline=wire_deadline,
                drain_incoming=True,
            )
        except TransportTimeout:
            # Completed data frames also need their final control message;
            # without it, the peer cannot discard the unfinished request.
            if sent_frames or conn.closed:
                self._invalidate("timed out while sending")
            raise
        return request_id

    def _send_retirements(self, conn: FramedConnection, wire_deadline: float) -> None:
        """Release unused peer KV before later commands, without an ACK wait."""
        if not self._retiring:
            return
        keys = tuple(self._retiring)
        request_id = self._next_request_id
        self._next_request_id += 1
        try:
            conn.send_control(
                encode_message(
                    CloseSequence(keys=keys),
                    session_id=self._session_id,
                    request_id=request_id,
                ),
                deadline=wire_deadline,
                drain_incoming=True,
            )
        except TransportError:
            self._invalidate("terminal sequence cleanup did not finish sending")
            raise
        self._retiring.difference_update(keys)

    def _await(
        self,
        matches: Callable[[MessageEnvelope, Any], bool],
        deadline: float,
        *,
        request_ids: Collection[int],
    ) -> tuple[MessageEnvelope, Any]:
        """Read frames until a control message satisfies ``matches``.

        Data frames are buffered by slot. Stale proposal responses are
        discarded together with their result frames so a later round can
        never read them by accident. Error replies fail the wait only when
        they answer one of ``request_ids`` or are session-level (request 0),
        so a late error for an abandoned request cannot fail a newer one.
        """
        conn = self._require_connection()
        while True:
            remaining = deadline - self._clock()
            if remaining <= 0:
                raise TransportTimeout("deadline elapsed")
            try:
                frame = conn.recv(timeout=remaining)
            except TransportTimeout:
                continue
            if frame.kind == FRAME_DATA:
                if len(self._inbound) >= self._max_pending_frames:
                    raise TransportError("pending result frame count exceeds budget")
                if self._inbound_bytes + len(frame.body) > self._max_pending_bytes:
                    raise TransportError("pending result bytes exceed budget")
                data = decode_data_frame(frame.body)
                if data.slot in self._inbound:
                    raise TransportError("duplicate pending result data slot")
                self._inbound[data.slot] = data
                self._inbound_sizes[data.slot] = len(frame.body)
                self._inbound_bytes += len(frame.body)
                self._set_result_receive_budget()
                continue
            try:
                envelope = decode_envelope(frame.body)
                message = decode_payload(envelope)
            except ProtocolError as e:
                self._invalidate(f"protocol error from server: {e}")
                raise ConnectionClosed(f"protocol error from server: {e}") from e
            if self._alive and envelope.session_id != self._session_id:
                if isinstance(message, ProposalResponse):
                    self._purge_result(message)
                logger.debug("Ignoring reply for another remote draft session")
                continue
            if isinstance(message, ErrorReply):
                if envelope.request_id == 0 or envelope.request_id in request_ids:
                    raise RemoteDraftError(
                        f"server error {message.status.name}: {message.detail}"
                    )
                logger.debug(
                    "Ignoring error reply to earlier request %d: %s",
                    envelope.request_id,
                    message.detail,
                )
                continue
            if envelope.request_id not in request_ids:
                if isinstance(message, ProposalResponse):
                    self._abandoned_batches.discard(message.batch_id)
                    self._purge_result(message)
                logger.debug(
                    "Ignoring reply to another request %d", envelope.request_id
                )
                continue
            if not matches(envelope, message):
                if isinstance(message, ProposalResponse):
                    self._purge_result(message)
                logger.debug(
                    "Ignoring unexpected %s (request %d)",
                    type(message).__name__,
                    envelope.request_id,
                )
                continue
            if isinstance(message, ProposalResponse) and not self._accept_response(
                message
            ):
                continue
            return envelope, message

    def _accept_response(self, response: ProposalResponse) -> bool:
        if response.sequence_number <= self._last_sequence_number:
            self._purge_result(response)
            return False
        self._last_sequence_number = response.sequence_number
        if response.batch_id in self._abandoned_batches:
            self._abandoned_batches.discard(response.batch_id)
            self._purge_result(response)
            return False
        return True

    def _purge_result(self, response: ProposalResponse) -> None:
        if response.result_slot is not None:
            self._take_frames(response.result_slot, RESULT_FRAMES)

    def _take_frames(self, base: int, count: int) -> list[DataFrame | None]:
        frames = []
        for slot in range(base, base + count):
            frames.append(self._inbound.pop(slot, None))
            self._inbound_bytes -= self._inbound_sizes.pop(slot, 0)
        self._set_result_receive_budget()
        return frames

    def _set_result_receive_budget(self) -> None:
        if self._max_result_frame_bytes == 0:
            return
        remaining = self._max_pending_bytes - self._inbound_bytes
        self._require_connection().set_receive_limits(
            max_data_bytes=min(self._max_result_frame_bytes, remaining),
            max_buffered_bytes=(
                MAX_CONTROL_FRAME_BYTES + remaining + 5 * (self._max_pending_frames + 1)
            ),
        )

    def _sequence_rpcs(
        self,
        calls: list[tuple[Any, SequenceKey, list[DataFrame] | None]],
        *,
        deadline: float | None = None,
    ) -> tuple[list[tuple[SpeculatorStatusCode, str]], float]:
        """Return sequence replies and the remaining acknowledgement deadline.

        Writes and acknowledgements consume the same remaining budget.
        """
        results = [
            (SpeculatorStatusCode.TIMEOUT, "not sent before the deadline")
        ] * len(calls)
        pending: dict[int, int] = {}
        acknowledged = 0
        if deadline is None:
            deadline = self._clock() + self._request_timeout_s
        prefill = bool(calls) and all(
            isinstance(message, PrefillChunk) for message, _, _ in calls
        )
        pipeline = not prefill or self._peer_protocol_minor >= 2
        if prefill and pipeline:
            group = self._next_request_id
            for message, _, _ in calls:
                message.execution_group_id = group

        def receive_ack(wait_deadline: float):
            return self._await(
                lambda env, msg: (
                    isinstance(msg, SequenceAck)
                    and env.request_id in pending
                    and msg.key == calls[pending[env.request_id]][1]
                ),
                wait_deadline,
                request_ids=pending,
            )

        lost = ""
        try:
            for index, (message, _, frames) in enumerate(calls):
                request_id = self._send(
                    message,
                    frames=frames,
                    deadline=deadline,
                )
                pending[request_id] = index
                if not pipeline:
                    envelope, ack = receive_ack(deadline)
                    results[pending.pop(envelope.request_id)] = (ack.status, ack.detail)
                    acknowledged += 1
        except TransportTimeout:
            pass
        except TransportError as e:
            self._invalidate(f"connection lost: {e}")
            lost = str(e)
            results[index:] = [(SpeculatorStatusCode.INTERNAL_ERROR, lost)] * (
                len(calls) - index
            )
        except RemoteDraftError as e:
            results[index:] = [(SpeculatorStatusCode.INTERNAL_ERROR, str(e))] * (
                len(calls) - index
            )
            pending.clear()
        unanswered = (
            SpeculatorStatusCode.TIMEOUT,
            "no acknowledgement before deadline",
        )
        while pending:
            if not self._alive:
                unanswered = (
                    SpeculatorStatusCode.INTERNAL_ERROR,
                    lost or "connection lost",
                )
                break
            try:
                envelope, ack = receive_ack(deadline)
            except TransportTimeout:
                break
            except TransportError as e:
                self._invalidate(f"connection lost: {e}")
                lost = str(e)
                continue
            except RemoteDraftError as e:
                unanswered = (SpeculatorStatusCode.INTERNAL_ERROR, str(e))
                break
            results[pending.pop(envelope.request_id)] = (ack.status, ack.detail)
            acknowledged += 1
        for index in pending.values():
            results[index] = unanswered
        if not self._alive:
            # Earlier ACKs cannot restore state invalidated by a later failure.
            for index, (status, _) in enumerate(results):
                if status is SpeculatorStatusCode.OK:
                    results[index] = (
                        SpeculatorStatusCode.INTERNAL_ERROR,
                        "session invalidated after acknowledgement",
                    )
        if any(status is SpeculatorStatusCode.TIMEOUT for status, _ in results):
            # A timeout cannot grant a fresh retirement write budget.
            deadline = min(deadline, self._clock())
        return results, deadline

    def _apply_status(
        self, record: _SequenceRecord, status: SpeculatorStatusCode
    ) -> None:
        record.last_status = status
        if status is SpeculatorStatusCode.OK:
            if record.state is SequenceState.IN_FLIGHT:
                record.state = SequenceState.READY
            return
        if status in _DESYNC_STATUSES and can_transition_sequence(
            record.state, SequenceState.DESYNCED
        ):
            record.state = SequenceState.DESYNCED
            return
        # QUEUE_FULL also drops this round's advancement; a later recovery
        # token cannot restore the skipped verifier context.
        self._latch_target_only(record, status)

    def _admit(
        self, keys: tuple[SequenceKey, ...], rows: list[int], query_lens: list[int]
    ) -> tuple[int, ...]:
        """Keep the rows the server limits allow; latch the rest target-only.

        The handshake already checked the verifier's declared limits, so
        this only triggers when the runner exceeds what it declared.
        """
        limits = self._limits
        if limits is None:
            return tuple(rows)
        admitted: list[int] = []
        num_tokens = 0
        for i in rows:
            if (
                len(admitted) < limits.max_batch_size
                and num_tokens + query_lens[i] <= limits.max_feature_tokens
            ):
                admitted.append(i)
                num_tokens += query_lens[i]
            else:
                self._latch_target_only(
                    self._records[keys[i].sequence_id],
                    SpeculatorStatusCode.QUEUE_FULL,
                )
        if len(admitted) != len(rows):
            logger.warning(
                "Remote draft batch exceeds the server max_batch_size=%d or "
                "max_feature_tokens=%d; %d of %d ready rows decode target-only",
                limits.max_batch_size,
                limits.max_feature_tokens,
                len(rows) - len(admitted),
                len(rows),
            )
        return tuple(admitted)

    def _latch_target_only(
        self, record: _SequenceRecord, status: SpeculatorStatusCode
    ) -> None:
        record.last_status = status
        if can_transition_sequence(record.state, SequenceState.TARGET_ONLY):
            record.state = SequenceState.TARGET_ONLY
            self._retiring.add(record.key)

    def _is_ready(self, key: SequenceKey) -> bool:
        record = self._records.get(key.sequence_id)
        return (
            record is not None
            and record.key == key
            and record.state is SequenceState.READY
        )

    def _skip_status(self, key: SequenceKey) -> SpeculatorStatusCode:
        record = self._records.get(key.sequence_id)
        if record is None:
            return SpeculatorStatusCode.SEQUENCE_DESYNC
        if record.key != key:
            return SpeculatorStatusCode.STALE_GENERATION
        if record.last_status is not SpeculatorStatusCode.OK:
            return record.last_status
        return SpeculatorStatusCode.SEQUENCE_DESYNC

    def _current_record(self, key: SequenceKey) -> _SequenceRecord:
        record = self._records.get(key.sequence_id)
        if record is None:
            raise ValueError(f"sequence {key.sequence_id} is not open")
        if record.key != key:
            raise ValueError(
                f"key generation {key.generation} is not current for sequence "
                f"{key.sequence_id} (current generation {record.key.generation})"
            )
        return record

    def _check_features(
        self, features: FeatureBatch, *, num_tokens: int, num_sequences: int
    ) -> None:
        schema = self.feature_schema
        if features.schema_id != schema.schema_id:
            raise ValueError(
                f"features carry schema {features.schema_id}, session "
                f"negotiated schema {schema.schema_id}"
            )
        if len(features.slots) != len(schema.slots):
            raise ValueError(
                f"features carry {len(features.slots)} slots, schema "
                f"{schema.schema_id} has {len(schema.slots)}"
            )
        for i, (spec, tensor) in enumerate(zip(schema.slots, features.slots)):
            if tensor is None:
                if not spec.optional:
                    raise ValueError(
                        f"required feature slot {i} ({spec.kind}) is missing"
                    )
                continue
            expected = num_sequences if spec.per_sequence else num_tokens
            if tensor.ndim == 0 or tensor.shape[0] != expected:
                raise ValueError(
                    f"feature slot {i} ({spec.kind}) has shape "
                    f"{tuple(tensor.shape)}, expected {expected} rows"
                )

    def _feature_frames(
        self,
        base: int,
        features: FeatureBatch,
        *,
        token_rows: torch.Tensor | None,
        sequence_rows: torch.Tensor | None,
    ) -> list[DataFrame]:
        frames: list[DataFrame] = []
        for i, (spec, tensor) in enumerate(
            zip(self.feature_schema.slots, features.slots)
        ):
            if tensor is not None:
                rows = sequence_rows if spec.per_sequence else token_rows
                frames.append(tensor_to_frame(base + i, _select_rows(tensor, rows)))
        return frames

    def _chunk_length(self, features: FeatureBatch) -> int:
        lengths = {
            tensor.shape[0]
            for spec, tensor in zip(self.feature_schema.slots, features.slots)
            if tensor is not None and tensor.ndim and not spec.per_sequence
        }
        if len(lengths) != 1:
            raise ValueError(
                "prefill token-level features must all share one token "
                f"dimension, got {sorted(lengths)}"
            )
        return lengths.pop()

    def _reserve_slots(self, count: int) -> int:
        base = self._next_slot
        self._next_slot += count
        return base

    def _require_connection(self) -> FramedConnection:
        if self._conn is None or self._conn.closed:
            raise ConnectionClosed("session is not connected")
        return self._conn

    def _require_no_inflight(self, operation: str) -> None:
        if self._inflight_batch_id is not None:
            raise RemoteDraftError(
                f"cannot {operation} while batch {self._inflight_batch_id} is "
                "still in flight; collect it first"
            )

    def _raise_if_strict(self, detail: str, *, quiet: bool = False) -> None:
        if self._failure_policy == "error":
            raise RemoteDraftError(detail)
        log = logger.debug if quiet else logger.warning
        log("Remote draft falling back to target-only: %s", detail)

    def _invalidate(self, reason: str) -> None:
        if not self._alive and self._conn is None:
            return
        if self._alive:
            logger.warning(
                "Remote draft session %s invalidated: %s", self._session_id, reason
            )
        self._alive = False
        for record in self._records.values():
            if can_transition_sequence(record.state, SequenceState.INVALID):
                record.state = SequenceState.INVALID
                record.last_status = SpeculatorStatusCode.INTERNAL_ERROR
        self._inflight_batch_id = None
        self._inbound.clear()
        self._inbound_sizes.clear()
        self._inbound_bytes = 0
        self._drop_connection()
        self._retiring.clear()

    def _drop_connection(self) -> None:
        conn, self._conn = self._conn, None
        if conn is not None:
            try:
                conn.close()
            except Exception:
                logger.exception("Failed to close remote draft connection")


def _usable_positions(valid: torch.Tensor, num_tokens: int) -> torch.Tensor:
    """[rows, num_tokens] mask of the draft positions each row may use."""
    lengths = valid.to(torch.int64).clamp(0, num_tokens).unsqueeze(1)
    return torch.arange(num_tokens) < lengths


def _query_lens(batch: RemoteProposalBatch) -> list[int]:
    batch_size = len(batch.keys)
    if batch.query_lens is None:
        return [1] * batch_size
    if batch.query_lens.shape != (batch_size,) or batch.query_lens.is_floating_point():
        raise ValueError(
            f"query_lens must be an integer [{batch_size}] tensor, got "
            f"{batch.query_lens.dtype} {tuple(batch.query_lens.shape)}"
        )
    query_lens = batch.query_lens.tolist()
    if any(length < 0 for length in query_lens):
        raise ValueError(f"query_lens must be non-negative, got {query_lens}")
    return query_lens


def _token_rows(query_lens: list[int], rows: tuple[int, ...]) -> torch.Tensor:
    """Indices of the token-level feature rows that belong to ``rows``."""
    starts = list(itertools.accumulate(query_lens, initial=0))
    return torch.cat([torch.arange(starts[i], starts[i] + query_lens[i]) for i in rows])


def _select_rows(tensor: torch.Tensor, rows: torch.Tensor | None) -> torch.Tensor:
    if rows is None:
        return tensor
    return tensor.index_select(0, rows.to(tensor.device))


def _checksum(frames: list[DataFrame]) -> int:
    value = 0
    for frame in frames:
        value = zlib.crc32(frame.payload, value)
    return value & 0x7FFFFFFF


def _status_from_wire(code: int) -> SpeculatorStatusCode:
    try:
        return SpeculatorStatusCode(code)
    except ValueError:
        return SpeculatorStatusCode.INTERNAL_ERROR
