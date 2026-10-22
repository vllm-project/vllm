# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Control-plane protocol: envelope versioning, codec roundtrips, and
stale-response identity semantics."""

import msgspec
import pytest

from vllm.v1.spec_decode.remote import transport
from vllm.v1.spec_decode.remote.capabilities import (
    SpeculatorPlacementCapabilities,
    TargetFeatureKind,
)
from vllm.v1.spec_decode.remote.protocol import (
    PROTOCOL_MAJOR,
    AdvanceAndPropose,
    CancelBatch,
    CloseSequence,
    ErrorReply,
    FeatureSlot,
    Hello,
    HelloAck,
    MessageEnvelope,
    OpenSequence,
    Ping,
    Pong,
    PrefillChunk,
    ProposalResponse,
    ProtocolError,
    ProtocolVersionError,
    RemoteServerLimits,
    SequenceAck,
    SequenceKey,
    SpeculatorStatusCode,
    TargetFeatureSchema,
    decode_envelope,
    decode_payload,
    encode_message,
    inline_feature_frame_rejections,
)

KEY = SequenceKey(verifier_instance_id="v0", sequence_id=7, generation=2)

CAPABILITIES = SpeculatorPlacementCapabilities(
    state_dependency="own_kv",
    required_features=(
        TargetFeatureKind.TOKEN_IDS,
        TargetFeatureKind.AUX_HIDDEN_STATES,
    ),
    supports_parallel_drafting=True,
    standalone_weights="materializable",
)

SCHEMA = TargetFeatureSchema(
    schema_id=1,
    slots=(
        FeatureSlot(kind=TargetFeatureKind.TOKEN_IDS, dtype="int32", trailing_shape=()),
        FeatureSlot(
            kind=TargetFeatureKind.AUX_HIDDEN_STATES,
            dtype="bfloat16",
            trailing_shape=(2880,),
        ),
    ),
)


@pytest.mark.parametrize(
    ("dtype", "itemsize"),
    [
        ("int32", 4),
        ("int64", 8),
        ("float16", 2),
        ("bfloat16", 2),
        ("float32", 4),
        ("float64", 8),
    ],
)
def test_inline_negotiation_reserves_encoded_metadata(monkeypatch, dtype, itemsize):
    hello = Hello(
        "v", "t", "tok", "eagle3", 1, False, ("inline",), max_num_batched_tokens=4
    )
    schema = TargetFeatureSchema(1, (FeatureSlot("aux", dtype, (2, 3)),))
    limits = RemoteServerLimits(1, 8, 1, 8)
    frame = transport.DataFrame((1 << 64) - 1, dtype, (4, 2, 3), bytes(24 * itemsize))
    wire_bytes = len(msgspec.msgpack.encode(frame))
    monkeypatch.setattr(transport, "MAX_FRAME_BYTES", wire_bytes)
    assert inline_feature_frame_rejections(hello, schema, limits) == []
    monkeypatch.setattr(transport, "MAX_FRAME_BYTES", wire_bytes - 1)
    reasons = inline_feature_frame_rejections(hello, schema, limits)
    assert len(reasons) == 1
    assert f"frame of {wire_bytes} bytes exceeds {wire_bytes - 1}" in reasons[0]


ALL_MESSAGES = [
    Hello(
        verifier_instance_id="v0",
        target_fingerprint="tf",
        tokenizer_fingerprint="kf",
        method="eagle3",
        num_speculative_tokens=5,
        parallel_drafting=True,
        supported_transports=("cuda_ipc", "pinned_host"),
    ),
    HelloAck(
        server_id="s0",
        session_id="sess",
        session_epoch=3,
        selected_transport="cuda_ipc",
        capabilities=CAPABILITIES,
        feature_schema=SCHEMA,
        limits=RemoteServerLimits(
            max_batch_size=128,
            max_feature_tokens=8192,
            max_sequences=128,
            max_model_len=32768,
            ring_slots=8,
        ),
    ),
    OpenSequence(key=KEY),
    PrefillChunk(
        key=KEY,
        offset=0,
        num_tokens=128,
        is_final=False,
        feature_slot=3,
        checksum=0xABCDEF,
    ),
    AdvanceAndPropose(
        batch_id=9,
        keys=(KEY,),
        round_ids_slot=0,
        accepted_counts_slot=1,
        recovery_tokens_slot=2,
        feature_slot=3,
    ),
    ProposalResponse(batch_id=9, sequence_number=42, result_slot=5),
    SequenceAck(
        key=KEY,
        status=SpeculatorStatusCode.SEQUENCE_DESYNC,
        detail="offset gap",
    ),
    CloseSequence(keys=(KEY,)),
    CancelBatch(batch_id=9),
    Ping(nonce=1),
    Pong(nonce=1, queue_depth=4, active_sequences=2),
    ErrorReply(status=SpeculatorStatusCode.INTERNAL_ERROR, detail="boom"),
]


@pytest.mark.parametrize("message", ALL_MESSAGES, ids=lambda m: type(m).__name__)
def test_roundtrip(message):
    data = encode_message(message, session_id="sess", request_id=11)
    envelope = decode_envelope(data)
    assert envelope.session_id == "sess"
    assert envelope.request_id == 11
    assert decode_payload(envelope) == message


def test_major_version_mismatch_rejected():
    encoder = msgspec.msgpack.Encoder()
    data = encoder.encode(
        MessageEnvelope(
            protocol_major=PROTOCOL_MAJOR + 1,
            protocol_minor=0,
            message_type="ping",
            session_id="",
            request_id=0,
            payload=encoder.encode(Ping()),
        )
    )
    with pytest.raises(ProtocolVersionError):
        decode_envelope(data)


def test_unknown_message_type_rejected():
    envelope = MessageEnvelope(
        protocol_major=PROTOCOL_MAJOR,
        protocol_minor=0,
        message_type="warp_speed",
        session_id="",
        request_id=0,
        payload=b"",
    )
    with pytest.raises(ProtocolError, match="unknown message type"):
        decode_payload(envelope)


def test_minor_version_adds_ignorable_fields():
    # A newer peer may add optional payload fields; they must be ignored.
    payload = msgspec.msgpack.Encoder().encode(
        {"nonce": 5, "field_from_the_future": "later"}
    )
    envelope = MessageEnvelope(
        protocol_major=PROTOCOL_MAJOR,
        protocol_minor=99,
        message_type="ping",
        session_id="",
        request_id=0,
        payload=payload,
    )
    assert decode_payload(envelope) == Ping(nonce=5)


@pytest.mark.parametrize(
    ("message_type", "payload"),
    [
        (
            "open_sequence",
            {
                "key": {
                    "verifier_instance_id": "v0",
                    "sequence_id": -1,
                    "generation": 0,
                }
            },
        ),
        (
            "prefill_chunk",
            {
                "key": {
                    "verifier_instance_id": "v0",
                    "sequence_id": 0,
                    "generation": 0,
                },
                "offset": -4,
                "num_tokens": 1,
                "is_final": True,
                "feature_slot": 0,
                "checksum": 0,
            },
        ),
        ("proposal_response", {"batch_id": 9, "sequence_number": -1}),
        (
            "advance_and_propose",
            {
                "batch_id": 9,
                "keys": [],
                "accepted_counts_slot": -2,
                "recovery_tokens_slot": 0,
                "feature_slot": 0,
            },
        ),
    ],
    ids=["negative-sequence", "negative-offset", "negative-seqno", "negative-slot"],
)
def test_semantic_range_violation_rejected(message_type, payload):
    # Structural decode success is not enough: out-of-range identifiers
    # must be rejected at the codec layer.
    envelope = MessageEnvelope(
        protocol_major=PROTOCOL_MAJOR,
        protocol_minor=0,
        message_type=message_type,
        session_id="",
        request_id=0,
        payload=msgspec.msgpack.Encoder().encode(payload),
    )
    with pytest.raises(ProtocolError, match="malformed"):
        decode_payload(envelope)


def test_degenerate_feature_shape_rejected():
    ack = ALL_MESSAGES[1]
    assert isinstance(ack, HelloAck)
    bad_ack = HelloAck(
        server_id=ack.server_id,
        session_id=ack.session_id,
        session_epoch=ack.session_epoch,
        selected_transport=ack.selected_transport,
        capabilities=ack.capabilities,
        feature_schema=TargetFeatureSchema(
            schema_id=1,
            slots=(FeatureSlot(kind="token_ids", dtype="int32", trailing_shape=(0,)),),
        ),
        limits=ack.limits,
    )
    with pytest.raises(ProtocolError, match="malformed"):
        decode_payload(decode_envelope(encode_message(bad_ack)))


def test_failed_response_carries_no_result_slot():
    response = ProposalResponse(
        batch_id=1,
        sequence_number=2,
        status=SpeculatorStatusCode.QUEUE_FULL,
    )
    decoded = decode_payload(decode_envelope(encode_message(response)))
    assert decoded.result_slot is None


def test_unknown_feature_kind_stays_decodable():
    # A newer-minor peer may advertise feature kinds this build does not
    # know; the handshake must decode so the capability check can reject
    # them by name instead of dying with a codec error.
    ack = ALL_MESSAGES[1]
    assert isinstance(ack, HelloAck)
    future_capabilities = SpeculatorPlacementCapabilities(
        state_dependency="own_kv",
        required_features=(TargetFeatureKind.TOKEN_IDS, "rope_scales"),
        standalone_weights="complete",
    )
    future_ack = HelloAck(
        server_id=ack.server_id,
        session_id=ack.session_id,
        session_epoch=ack.session_epoch,
        selected_transport=ack.selected_transport,
        capabilities=future_capabilities,
        feature_schema=TargetFeatureSchema(
            schema_id=2,
            slots=(
                FeatureSlot(kind="rope_scales", dtype="float32", trailing_shape=()),
            ),
        ),
        limits=ack.limits,
    )
    decoded = decode_payload(decode_envelope(encode_message(future_ack)))
    assert "rope_scales" in decoded.capabilities.required_features
    assert decoded.feature_schema.slots[0].kind == "rope_scales"


def test_malformed_payload_raises_protocol_error():
    envelope = MessageEnvelope(
        protocol_major=PROTOCOL_MAJOR,
        protocol_minor=0,
        message_type="hello",
        session_id="",
        request_id=0,
        payload=b"\xc1garbage",
    )
    with pytest.raises(ProtocolError):
        decode_payload(envelope)


def test_unregistered_message_type_cannot_encode():
    class Rogue(msgspec.Struct):
        x: int = 0

    with pytest.raises(ProtocolError, match="unregistered"):
        encode_message(Rogue())


def test_sequence_key_generation_distinguishes_stale_state():
    # Same request identity, new generation: distinct key, so responses
    # carrying the old generation cannot match current state.
    assert SequenceKey("v0", 7, 3) != KEY
    assert len({KEY, SequenceKey("v0", 7, 2)}) == 1
