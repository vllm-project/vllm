# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for ECCPUConnector wire types and metadata."""

import msgspec
import pytest

from vllm.distributed.ec_transfer.ec_connector.cpu.common import (
    ECCPUConnectorMetadata,
)
from vllm.distributed.ec_transfer.ec_connector.cpu.scheduler.protocol import (
    XferAck,
    XferReq,
    XferStatus,
    compute_ec_compatibility_hash,
)

# ── compute_ec_compatibility_hash ────────────────────────────────────────────


def test_compat_hash_is_deterministic():
    h1 = compute_ec_compatibility_hash("1.0", "llama", "float16", 512)
    h2 = compute_ec_compatibility_hash("1.0", "llama", "float16", 512)
    assert h1 == h2


def test_compat_hash_sensitive_to_each_field():
    base = compute_ec_compatibility_hash("1.0", "llama", "float16", 512)
    assert compute_ec_compatibility_hash("1.1", "llama", "float16", 512) != base
    assert compute_ec_compatibility_hash("1.0", "mistral", "float16", 512) != base
    assert compute_ec_compatibility_hash("1.0", "llama", "bfloat16", 512) != base
    assert compute_ec_compatibility_hash("1.0", "llama", "float16", 1024) != base


def test_compat_hash_no_collision_on_separator_chars():
    """Regression: field values containing the separator must not collide."""
    a = compute_ec_compatibility_hash("1.0", "a|b", "c", 512)
    b = compute_ec_compatibility_hash("1.0", "a", "b|c", 512)
    assert a != b


# ── XferReq / XferAck round-trips ────────────────────────────────────────────

_encoder = msgspec.msgpack.Encoder()
_req_decoder = msgspec.msgpack.Decoder(XferReq)
_ack_decoder = msgspec.msgpack.Decoder(XferAck)


def _make_req(**overrides) -> XferReq:
    defaults = dict(mm_hash="abc123", compatibility_hash="deadbeef")
    defaults.update(overrides)
    return XferReq(**defaults)


def test_xfer_req_roundtrip():
    req = _make_req()
    assert _req_decoder.decode(_encoder.encode(req)) == req


@pytest.mark.parametrize(
    "status",
    [
        XferStatus.OK,
        XferStatus.NACK_MISSING,
        XferStatus.NACK_INCOMPAT,
        XferStatus.NACK_VERSION,
        XferStatus.NACK_INTERNAL,
    ],
)
def test_xfer_ack_roundtrip(status):
    ack = XferAck(
        mm_hash="abc123",
        status=status,
        src_block_indices=[1, 2, 3],
        agent_metadata=b"\xde\xad",
        mem_descriptor=b"\x00\x01",
    )
    decoded = _ack_decoder.decode(_encoder.encode(ack))
    assert decoded == ack


def test_xfer_ack_nack_omits_optional_fields():
    """A NACK carries no grant payload; the optional fields default to empty."""
    ack = XferAck(mm_hash="x", status=XferStatus.NACK_MISSING)
    decoded = _ack_decoder.decode(_encoder.encode(ack))
    assert decoded == ack
    assert decoded.src_block_indices == []
    assert decoded.agent_metadata == b""
    assert decoded.mem_descriptor == b""


def test_tag_discriminator_rejects_wrong_type():
    with pytest.raises((msgspec.DecodeError, msgspec.ValidationError)):
        _req_decoder.decode(_encoder.encode(XferAck(mm_hash="x", status=XferStatus.OK)))
    with pytest.raises((msgspec.DecodeError, msgspec.ValidationError)):
        _ack_decoder.decode(_encoder.encode(_make_req()))


# ── version field handling ───────────────────────────────────────────────────


@pytest.mark.parametrize("version", [0, 99])
def test_xfer_req_explicit_version_preserved(version):
    """Explicit connector_version values must survive encode/decode."""
    req = _make_req(connector_version=version)
    decoded = _req_decoder.decode(_encoder.encode(req))
    assert decoded.connector_version == version


# ── field validation ─────────────────────────────────────────────────────────


def test_xfer_req_rejects_missing_required_field():
    bad = _encoder.encode(
        {
            "type": "req",
            # mm_hash deliberately omitted
            "compatibility_hash": "",
        }
    )
    with pytest.raises((msgspec.DecodeError, msgspec.ValidationError)):
        _req_decoder.decode(bad)


def test_xfer_ack_rejects_non_int_src_block_indices():
    bad = _encoder.encode(
        {
            "type": "ack",
            "mm_hash": "x",
            "status": int(XferStatus.OK),
            "src_block_indices": ["a", "b"],
        }
    )
    with pytest.raises((msgspec.DecodeError, msgspec.ValidationError)):
        _ack_decoder.decode(bad)


def test_xfer_ack_rejects_unknown_status():
    bad = _encoder.encode({"type": "ack", "mm_hash": "x", "status": 999})
    with pytest.raises((msgspec.DecodeError, msgspec.ValidationError)):
        _ack_decoder.decode(bad)


# ── ECCPUConnectorMetadata ───────────────────────────────────────────────────


def test_metadata_defaults_are_empty():
    meta = ECCPUConnectorMetadata()
    assert meta.saves == {}
    assert meta.loads == {}
