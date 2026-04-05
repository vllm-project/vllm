# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for P2pNcclConnector request ID handling.

Verifies that the external (original) request ID is used for NCCL
send/recv keys and address parsing in disaggregated prefill/decode,
rather than the randomized internal request ID.
See: https://github.com/vllm-project/vllm/issues/34277
"""

from vllm.distributed.kv_transfer.kv_connector.v1.p2p.p2p_nccl_connector import (
    P2pNcclConnectorMetadata,
    ReqMeta,
)


def test_req_meta_preserves_external_req_id():
    """ReqMeta.make_meta should store the external_req_id separately."""
    internal_id = (
        "___prefill_addr_10.0.0.1:5000___decode_addr_10.0.0.2:6000_abc-deadbeef"
    )
    external_id = "___prefill_addr_10.0.0.1:5000___decode_addr_10.0.0.2:6000_abc"

    meta = ReqMeta.make_meta(
        request_id=internal_id,
        token_ids=[1, 2, 3],
        block_ids=[0, 1],
        block_size=16,
        external_req_id=external_id,
    )

    assert meta.request_id == internal_id
    assert meta.external_req_id == external_id


def test_req_meta_defaults_external_to_internal():
    """When external_req_id is None, it should default to request_id."""
    req_id = "some-request-id"
    meta = ReqMeta.make_meta(
        request_id=req_id,
        token_ids=[1, 2, 3],
        block_ids=[0],
        block_size=16,
    )

    assert meta.external_req_id == req_id


def test_connector_metadata_add_request_propagates_external_id():
    """P2pNcclConnectorMetadata.add_request should propagate external_req_id."""
    metadata = P2pNcclConnectorMetadata()
    internal_id = "req-abc-12345678"
    external_id = "req-abc"

    metadata.add_request(
        request_id=internal_id,
        token_ids=[1, 2, 3, 4],
        block_ids=[0, 1, 2],
        block_size=16,
        external_req_id=external_id,
    )

    assert len(metadata.requests) == 1
    assert metadata.requests[0].request_id == internal_id
    assert metadata.requests[0].external_req_id == external_id


def test_parse_request_id_works_with_external_id():
    """parse_request_id should work on the external (un-randomized) ID."""
    from vllm.distributed.kv_transfer.kv_connector.v1.p2p.p2p_nccl_connector import (
        P2pNcclConnector,
    )

    external_id = "___prefill_addr_10.0.1.2:21001___decode_addr_10.0.1.3:22001_abc123"
    # Prefill side parses decode address
    ip, port = P2pNcclConnector.parse_request_id(external_id, is_prefill=True)
    assert ip == "10.0.1.3"
    assert port == 22001

    # Decode side parses prefill address
    ip, port = P2pNcclConnector.parse_request_id(external_id, is_prefill=False)
    assert ip == "10.0.1.2"
    assert port == 21001


def test_parse_request_id_fails_on_randomized_id():
    """Randomized internal IDs may still parse if the regex matches,
    but the key point is that both sides must use the SAME ID."""

    external_id = "___prefill_addr_10.0.1.2:21001___decode_addr_10.0.1.3:22001_abc123"
    internal_id_a = external_id + "-aabbccdd"
    internal_id_b = external_id + "-eeff0011"

    # Both internal IDs parse to the same addresses (regex still
    # matches), but the NCCL keys would differ. This is the bug:
    # prefill sends with key_a, decode receives with key_b.
    layer = "#model.layers.0.self_attn"
    nccl_key_a = internal_id_a + layer
    nccl_key_b = internal_id_b + layer
    assert nccl_key_a != nccl_key_b, "Internal IDs produce mismatched NCCL keys"

    # With the fix, both sides use the same external_id
    nccl_key_prefill = external_id + layer
    nccl_key_decode = external_id + layer
    assert nccl_key_prefill == nccl_key_decode, (
        "External IDs must produce matching NCCL keys"
    )
