# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the ECZmqConnector wire protocol.

The protocol is the contract between two processes, so what matters is that a
message survives the round trip byte-exact for every dtype/shape the encoder
cache can hold, and that a receiver can tell what it got before decoding.
"""

import pytest
import torch

from vllm.distributed.ec_transfer.ec_connector.zmq.protocol import (
    ECZmqMsgType,
    ECZmqProtocol,
)

pytestmark = pytest.mark.cpu_test


@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.int8]
)
@pytest.mark.parametrize("shape", [(1, 8), (37, 1152), (0, 8)])
def test_embedding_round_trip_is_exact(dtype, shape):
    protocol = ECZmqProtocol()
    embedding = (
        torch.arange(torch.Size(shape).numel(), dtype=torch.int32)
        .reshape(shape)
        .to(dtype)
    )

    decoded = protocol.decode_embedding(protocol.encode_embedding("mm0", embedding))

    assert decoded.mm_hash == "mm0"
    assert decoded.embedding.dtype == dtype
    assert decoded.embedding.shape == embedding.shape
    assert torch.equal(decoded.embedding, embedding)


def test_large_embedding_travels_in_its_own_frame():
    """A tensor must not be copied into the header, so it can be sent as is."""
    protocol = ECZmqProtocol()
    embedding = torch.zeros(1024, 64, dtype=torch.float16)

    frames = protocol.encode_embedding("mm0", embedding)

    assert len(frames) == 3
    assert frames[0] == ECZmqMsgType.EMBEDDING.value
    assert len(bytes(frames[-1])) == embedding.numel() * embedding.element_size()


def test_decoded_embedding_does_not_alias_the_frames():
    """The receive thread releases the frames right after decoding."""
    protocol = ECZmqProtocol()
    embedding = torch.ones(256, 8, dtype=torch.float16)

    frames = protocol.encode_embedding("mm0", embedding)
    decoded = protocol.decode_embedding(frames)
    payload = bytearray(bytes(frames[-1]))
    payload[:] = b"\x00" * len(payload)

    assert torch.equal(decoded.embedding, embedding)


def test_type_is_readable_without_decoding():
    protocol = ECZmqProtocol()
    frames = protocol.encode_embedding("mm0", torch.zeros(4, 4))

    assert ECZmqProtocol.peek_type(frames) is ECZmqMsgType.EMBEDDING


def test_unexpected_message_type_is_rejected():
    protocol = ECZmqProtocol()
    frames = protocol.encode_embedding("mm0", torch.zeros(4, 4))
    frames[0] = ECZmqMsgType.ENCODE_REQUEST.value

    with pytest.raises(ValueError, match="Expected an EMBEDDING"):
        protocol.decode_embedding(frames)


@pytest.mark.parametrize("frames", [[], [b"\xff"]])
def test_undecodable_frames_raise(frames):
    with pytest.raises(ValueError):
        ECZmqProtocol.peek_type(frames)
