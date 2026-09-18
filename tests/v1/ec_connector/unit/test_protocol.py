# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import msgspec

from vllm.distributed.ec_transfer.ec_connector.cpu.protocol import (
    EC_CONNECTOR_VERSION,
    XferAck,
    XferReq,
    XferStatus,
    compute_ec_compatibility_hash,
)


def test_xferreq_roundtrip():
    req = XferReq(mm_hash="h1", compatibility_hash="c1", session_id="s1")
    data = msgspec.msgpack.encode(req)
    out = msgspec.msgpack.decode(data, type=XferReq)
    assert out.mm_hash == "h1"
    assert out.compatibility_hash == "c1"
    assert out.connector_version == EC_CONNECTOR_VERSION


def test_xferack_roundtrip_ok():
    ack = XferAck(
        mm_hash="h1",
        status=XferStatus.OK,
        src_block_indices=[1, 2],
        agent_metadata=b"meta",
        mem_descriptor=b"desc",
    )
    out = msgspec.msgpack.decode(msgspec.msgpack.encode(ack), type=XferAck)
    assert out.status == XferStatus.OK
    assert out.src_block_indices == [1, 2]


def test_compat_hash_deterministic_and_sensitive():
    a = compute_ec_compatibility_hash("0.1", "m", "float16", 64)
    b = compute_ec_compatibility_hash("0.1", "m", "float16", 64)
    c = compute_ec_compatibility_hash("0.1", "m", "float16", 128)
    assert a == b
    assert a != c
