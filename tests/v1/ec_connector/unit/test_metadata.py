# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for ECCPUConnector wire types and metadata."""

import msgspec
import pytest

from vllm.distributed.ec_transfer.ec_connector.cpu.metadata import (
    ECCPUConnectorMetadata,
    XferAck,
    XferReq,
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
    defaults = dict(
        mm_hash="abc123",
        dst_block_indices=[0, 1, 2],
        consumer_agent_name="engine-a",
        consumer_nixl_metadata=b"\xde\xad\xbe\xef",
        consumer_mem_descriptor=b"\x00\x01\x02",
        compatibility_hash="deadbeef",
    )
    defaults.update(overrides)
    return XferReq(**defaults)


def test_xfer_req_roundtrip():
    req = _make_req()
    assert _req_decoder.decode(_encoder.encode(req)) == req


@pytest.mark.parametrize("ok", [True, False])
def test_xfer_ack_roundtrip(ok):
    ack = XferAck(mm_hash="abc123", ok=ok)
    decoded = _ack_decoder.decode(_encoder.encode(ack))
    assert decoded == ack


def test_tag_discriminator_rejects_wrong_type():
    with pytest.raises((msgspec.DecodeError, msgspec.ValidationError)):
        _req_decoder.decode(_encoder.encode(XferAck(mm_hash="x", ok=True)))
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


def test_xfer_req_rejects_non_int_block_indices():
    bad = _encoder.encode(
        {
            "type": "req",
            "mm_hash": "x",
            "dst_block_indices": ["a", "b"],
            "consumer_agent_name": "a",
            "consumer_nixl_metadata": b"",
            "consumer_mem_descriptor": b"",
            "compatibility_hash": "",
        }
    )
    with pytest.raises((msgspec.DecodeError, msgspec.ValidationError)):
        _req_decoder.decode(bad)


def test_xfer_req_rejects_missing_required_field():
    bad = _encoder.encode(
        {
            "type": "req",
            # mm_hash deliberately omitted
            "dst_block_indices": [0],
            "consumer_agent_name": "a",
            "consumer_nixl_metadata": b"",
            "consumer_mem_descriptor": b"",
            "compatibility_hash": "",
        }
    )
    with pytest.raises((msgspec.DecodeError, msgspec.ValidationError)):
        _req_decoder.decode(bad)


def test_xfer_ack_rejects_non_bool_ok():
    bad = _encoder.encode({"type": "ack", "mm_hash": "x", "ok": "yes"})
    with pytest.raises((msgspec.DecodeError, msgspec.ValidationError)):
        _ack_decoder.decode(bad)


# ── ECCPUConnectorMetadata ───────────────────────────────────────────────────


def test_metadata_defaults_are_empty():
    meta = ECCPUConnectorMetadata()
    assert meta.saves == {}
    assert meta.loads == {}
