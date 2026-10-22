# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Control-plane wire protocol for the standalone speculator server.

The control plane carries small, low-frequency messages (handshake,
sequence lifecycle, health, errors). High-frequency tensor traffic (target
features, draft tokens) travels on a separate data plane and is referenced
from these messages only by slot index.

Versioning rules:
- A different ``protocol_major`` rejects the connection.
- ``protocol_minor`` bumps may only add optional fields; unknown fields are
  ignored on decode.
- The envelope is the single authority for the protocol version; payloads
  never carry their own copy.
"""

import enum
from typing import Annotated

import msgspec

from vllm.v1.spec_decode.remote.capabilities import (
    SpeculatorPlacementCapabilities,
)
from vllm.v1.spec_decode.remote.transport import TransportError, check_data_frame_size

# v2 requires peers to honor the per-round draft budget, including zero.
# A v1 server would silently ignore it and still generate the maximum K.
PROTOCOL_MAJOR = 2
PROTOCOL_MINOR = 1

# Field-level range constraints are enforced by msgspec on decode, so a
# peer cannot deliver negative identifiers or degenerate shapes. Checks
# that need negotiated session state (slot ranges, batch limits, schema
# identity) belong to the session layer, not this module.
NonNegativeInt = Annotated[int, msgspec.Meta(ge=0)]
PositiveInt = Annotated[int, msgspec.Meta(gt=0)]


class ProtocolError(Exception):
    """Malformed or incompatible control-plane message."""


class ProtocolVersionError(ProtocolError):
    """Peer speaks an incompatible protocol major version."""


class SpeculatorStatusCode(enum.IntEnum):
    """Per-batch / per-row status codes.

    Failures are never encoded as token IDs; a failed row reports a status
    here and ``valid_length=0`` so the verifier falls back to target-only
    decoding for that request.
    """

    OK = 0
    QUEUE_FULL = 1
    TIMEOUT = 2
    STALE_GENERATION = 3
    ROUND_MISMATCH = 4
    SEQUENCE_DESYNC = 5
    UNSUPPORTED_FEATURE = 6
    OUT_OF_MEMORY = 7
    INTERNAL_ERROR = 8
    PROTOCOL_ERROR = 9


class SequenceKey(msgspec.Struct, frozen=True):  # type: ignore[call-arg]
    """Identity of one draft sequence's state on the server.

    ``generation`` increments whenever the verifier re-creates state for the
    same request (e.g. after preemption and re-prefill); responses carrying
    an older generation are discarded as stale.
    """

    verifier_instance_id: str
    sequence_id: NonNegativeInt
    generation: NonNegativeInt


class FeatureSlot(msgspec.Struct, frozen=True):  # type: ignore[call-arg]
    """One tensor slot in the negotiated target-feature layout.

    ``kind`` is a TargetFeatureKind value carried as a plain string so that
    unknown kinds from a newer-minor peer stay decodable.
    """

    kind: str
    dtype: str
    trailing_shape: tuple[PositiveInt, ...]
    optional: bool = False
    per_sequence: bool = False
    """One row per sequence (e.g. temperature) instead of one row per
    target token (e.g. hidden states)."""


class TargetFeatureSchema(msgspec.Struct, frozen=True):  # type: ignore[call-arg]
    """Fixed slot ordering negotiated at handshake.

    After the handshake the data plane only ever refers to this layout by
    ``schema_id``; per-round messages never carry string feature maps.
    """

    schema_id: NonNegativeInt
    slots: tuple[FeatureSlot, ...]


class RemoteServerLimits(msgspec.Struct, frozen=True):  # type: ignore[call-arg]
    """Hard limits the server enforces.

    Both peers reject the session at handshake when these cannot hold the
    verifier's declared scheduler limits (see Hello).
    """

    max_batch_size: PositiveInt
    max_feature_tokens: PositiveInt
    max_sequences: PositiveInt
    max_model_len: PositiveInt
    ring_slots: NonNegativeInt = 0


class Hello(msgspec.Struct):
    """Verifier -> server handshake request."""

    verifier_instance_id: str
    target_fingerprint: str
    tokenizer_fingerprint: str
    method: str
    num_speculative_tokens: PositiveInt
    parallel_drafting: bool
    supported_transports: tuple[str, ...]
    draft_checkpoint_fingerprint: str = ""
    max_num_seqs: NonNegativeInt = 0
    """Largest batch the verifier may send; 0 means undeclared."""
    max_num_batched_tokens: NonNegativeInt = 0
    """Largest feature token count per message; 0 means undeclared."""
    max_model_len: NonNegativeInt = 0
    """Longest sequence the verifier may grow; 0 means undeclared."""


class HelloAck(msgspec.Struct):
    """Server -> verifier handshake response."""

    server_id: str
    session_id: str
    session_epoch: NonNegativeInt
    selected_transport: str
    capabilities: SpeculatorPlacementCapabilities
    feature_schema: TargetFeatureSchema
    limits: RemoteServerLimits
    server_vllm_version: str = ""


class OpenSequence(msgspec.Struct):
    """Register a new SequenceKey and reserve draft state for it."""

    key: SequenceKey


class PrefillChunk(msgspec.Struct):
    """Append one prompt chunk to a sequence's draft state.

    Token-level feature slots hold ``num_tokens`` rows and per-sequence
    slots hold one row.

    ``offset`` must equal the number of tokens the server has already
    applied; a gap or checksum mismatch yields SEQUENCE_DESYNC. Duplicate
    chunks with a matching checksum may be ACKed idempotently.
    """

    key: SequenceKey
    offset: NonNegativeInt
    num_tokens: NonNegativeInt
    is_final: bool
    feature_slot: NonNegativeInt
    checksum: NonNegativeInt
    execution_timeout_ms: NonNegativeInt | None = None
    """Remaining relative execution budget when the control frame is sent."""


class AdvanceAndPropose(msgspec.Struct):
    """Apply the last verification outcome and propose the next K tokens.

    Per-sequence feature slots hold one row per key. Token-level slots hold
    ``sum(query_lens)`` rows, the target tokens of each key in key order;
    without ``query_lens_slot`` every key contributes exactly one row.
    """

    batch_id: NonNegativeInt
    keys: tuple[SequenceKey, ...]
    accepted_counts_slot: NonNegativeInt
    recovery_tokens_slot: NonNegativeInt
    feature_slot: NonNegativeInt
    round_ids_slot: NonNegativeInt | None = None
    query_lens_slot: NonNegativeInt | None = None
    """Integer frame [len(keys)] of token-level feature rows per key."""
    num_speculative_tokens: NonNegativeInt | None = None
    """This round's draft budget; None uses the maximum negotiated by Hello.

    Zero still applies verified context but produces no usable drafts. Results
    retain the negotiated maximum width, with valid lengths bounded by this K.
    """
    execution_timeout_ms: NonNegativeInt | None = None
    """Optional relative budget; older minor peers retain their server limit."""


class ProposalResponse(msgspec.Struct):
    """Server -> verifier completion notice for one proposal batch.

    Draft tokens, per-row valid lengths, and per-row status live in the
    data-plane slot referenced by ``result_slot``; a failed batch carries
    no result slot.
    """

    batch_id: NonNegativeInt
    sequence_number: NonNegativeInt
    result_slot: NonNegativeInt | None = None
    status: SpeculatorStatusCode = SpeculatorStatusCode.OK


class SequenceAck(msgspec.Struct):
    """Server -> verifier acknowledgement of a sequence command."""

    key: SequenceKey
    status: SpeculatorStatusCode = SpeculatorStatusCode.OK
    detail: str = ""


class CloseSequence(msgspec.Struct):
    """Release draft KV and state; duplicate closes are idempotent."""

    keys: tuple[SequenceKey, ...]


class CancelBatch(msgspec.Struct):
    """Cancel not-yet-started work after a timeout or request finish."""

    batch_id: NonNegativeInt


class Ping(msgspec.Struct):
    """Liveness probe; either side may send it."""

    nonce: int = 0


class Pong(msgspec.Struct):
    """Liveness / load report."""

    nonce: int = 0
    queue_depth: NonNegativeInt = 0
    active_sequences: NonNegativeInt = 0
    healthy: bool = True


class ErrorReply(msgspec.Struct):
    """Session-level error not tied to a specific sequence."""

    status: SpeculatorStatusCode
    detail: str = ""


class MessageEnvelope(msgspec.Struct):
    """Outer frame of every control-plane message.

    ``request_id`` correlates an RPC with its reply; model state identity is
    carried by SequenceKey inside the payload, never by request_id.
    """

    protocol_major: PositiveInt
    protocol_minor: NonNegativeInt
    message_type: str
    session_id: str
    request_id: NonNegativeInt
    payload: bytes


_MESSAGE_TYPES: dict[str, type[msgspec.Struct]] = {
    "hello": Hello,
    "hello_ack": HelloAck,
    "open_sequence": OpenSequence,
    "prefill_chunk": PrefillChunk,
    "advance_and_propose": AdvanceAndPropose,
    "proposal_response": ProposalResponse,
    "sequence_ack": SequenceAck,
    "close_sequence": CloseSequence,
    "cancel_batch": CancelBatch,
    "ping": Ping,
    "pong": Pong,
    "error": ErrorReply,
}
_TYPE_NAMES = {cls: name for name, cls in _MESSAGE_TYPES.items()}

# Codec instances reuse internal buffers and are not thread-safe; use them
# from a single thread per connection.
_encoder = msgspec.msgpack.Encoder()
_envelope_decoder = msgspec.msgpack.Decoder(MessageEnvelope)
_payload_decoders = {
    name: msgspec.msgpack.Decoder(cls) for name, cls in _MESSAGE_TYPES.items()
}


def encode_message(
    message: msgspec.Struct,
    *,
    session_id: str = "",
    request_id: int = 0,
) -> bytes:
    """Wrap a payload struct in a MessageEnvelope and encode it."""
    type_name = _TYPE_NAMES.get(type(message))
    if type_name is None:
        raise ProtocolError(f"unregistered message type: {type(message).__name__}")
    envelope = MessageEnvelope(
        protocol_major=PROTOCOL_MAJOR,
        protocol_minor=PROTOCOL_MINOR,
        message_type=type_name,
        session_id=session_id,
        request_id=request_id,
        payload=_encoder.encode(message),
    )
    return _encoder.encode(envelope)


def decode_envelope(data: bytes) -> MessageEnvelope:
    """Decode and version-check the outer envelope."""
    try:
        envelope = _envelope_decoder.decode(data)
    except msgspec.DecodeError as e:
        raise ProtocolError(f"malformed envelope: {e}") from e
    if envelope.protocol_major != PROTOCOL_MAJOR:
        raise ProtocolVersionError(
            f"unsupported protocol major {envelope.protocol_major}, "
            f"expected {PROTOCOL_MAJOR}"
        )
    return envelope


def limit_shortfalls(hello: Hello, limits: RemoteServerLimits) -> list[str]:
    """Reasons ``limits`` cannot serve the scheduler limits in ``hello``.

    Undeclared (zero) verifier limits are not checked.
    """
    reasons = []
    for server_name, offered, verifier_name, needed in (
        ("max_batch_size", limits.max_batch_size, "max_num_seqs", hello.max_num_seqs),
        (
            "max_feature_tokens",
            limits.max_feature_tokens,
            "max_num_batched_tokens",
            hello.max_num_batched_tokens,
        ),
        ("max_model_len", limits.max_model_len, "max_model_len", hello.max_model_len),
    ):
        if offered < needed:
            reasons.append(
                f"server {server_name}={offered} cannot hold the verifier's "
                f"{verifier_name}={needed}"
            )
    return reasons


def inline_feature_frame_rejections(
    hello: Hello, schema: TargetFeatureSchema, limits: RemoteServerLimits
) -> list[str]:
    """Check each inline slot against its declared maximum encoded byte size."""
    max_tokens = hello.max_num_batched_tokens or limits.max_feature_tokens
    max_sequences = hello.max_num_seqs or limits.max_batch_size
    reasons = []
    for index, slot in enumerate(schema.slots):
        rows = max_sequences if slot.per_sequence else max_tokens
        try:
            # Slot IDs grow throughout a session; reserve their full wire width.
            check_data_frame_size(
                (1 << 64) - 1, slot.dtype, (rows, *slot.trailing_shape)
            )
        except TransportError as e:
            reasons.append(f"inline feature slot {index} ({slot.kind}): {e}")
    return reasons


def inline_receive_budgets(
    hello: Hello, schema: TargetFeatureSchema, limits: RemoteServerLimits
) -> tuple[int, int]:
    """Maximum encoded data frame and bytes preceding one control message."""
    rows = hello.max_num_seqs or limits.max_batch_size
    tokens = hello.max_num_batched_tokens or limits.max_feature_tokens
    # Accepted counts, recovery tokens, query lengths and round IDs can each
    # use int64. Reserve full-width slot IDs without allocating payloads.
    slot_id = (1 << 64) - 1
    sizes = [check_data_frame_size(slot_id, "int64", (rows,))] * 4
    sizes.extend(
        check_data_frame_size(
            slot_id,
            slot.dtype,
            (rows if slot.per_sequence else tokens, *slot.trailing_shape),
        )
        for slot in schema.slots
    )
    return max(sizes), sum(sizes)


def decode_payload(envelope: MessageEnvelope) -> msgspec.Struct:
    """Decode the payload struct named by the envelope's message_type."""
    decoder = _payload_decoders.get(envelope.message_type)
    if decoder is None:
        raise ProtocolError(f"unknown message type: {envelope.message_type!r}")
    try:
        return decoder.decode(envelope.payload)
    except msgspec.DecodeError as e:
        raise ProtocolError(f"malformed {envelope.message_type} payload: {e}") from e
