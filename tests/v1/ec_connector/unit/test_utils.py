# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for ECCPUConnector scheduler utilities."""

import msgspec
import pytest

from vllm.distributed.ec_transfer.ec_connector.cpu.utils import (
    build_block_descs,
    deserialize_mem_descriptor,
    serialize_mem_descriptor,
)

# ── build_block_descs ────────────────────────────────────────────────────────


@pytest.mark.parametrize("device_id,expected_dev", [(7, 7), (None, 0)])
def test_build_block_descs(device_id, expected_dev):
    kwargs = dict(base_ptr=1000, num_blocks=4, block_size_bytes=256)
    if device_id is not None:
        kwargs["device_id"] = device_id
    descs = build_block_descs(**kwargs)
    assert len(descs) == 4
    for i, (addr, size, dev) in enumerate(descs):
        assert addr == 1000 + i * 256
        assert size == 256
        assert dev == expected_dev


def test_build_block_descs_zero_blocks_returns_empty():
    assert build_block_descs(base_ptr=100, num_blocks=0, block_size_bytes=64) == []


# ── serialize / deserialize_mem_descriptor ───────────────────────────────────


def test_mem_descriptor_roundtrip():
    descs = [(100, 64, 0), (164, 64, 0), (228, 64, 1)]
    assert deserialize_mem_descriptor(serialize_mem_descriptor(descs)) == descs


@pytest.mark.parametrize(
    "bad_value",
    [
        [(1, 2)],  # 2-tuple instead of 3
        [("a", "b", "c")],  # strings, not ints
    ],
)
def test_mem_descriptor_rejects_malformed_payload(bad_value):
    """Malformed descriptor lists must fail to decode."""
    encoder = msgspec.msgpack.Encoder()
    bad_payload = encoder.encode(bad_value)
    with pytest.raises((msgspec.DecodeError, msgspec.ValidationError)):
        deserialize_mem_descriptor(bad_payload)
