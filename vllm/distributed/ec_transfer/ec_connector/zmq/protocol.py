# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Wire protocol for the ECZmqConnector.

A message is a multipart ZMQ frame sequence whose first frame is the message
type, so a receiver can demultiplex without deserializing the payload:

    [msg_type, header, *tensor_buffers]

`header` and the tensor buffers come from `MsgpackEncoder.encode`, which keeps
large tensors in their own frames instead of copying them into the header.
"""

import enum

import msgspec
import torch

from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder, bytestr


class ECZmqMsgType(enum.Enum):
    """Message types, encoded as byte strings so no framing layer is needed."""

    EMBEDDING = b"\x00"
    # Reserved for the encoder-side ingress: a consumer asking a producer to
    # encode an item, and a producer reporting that it could not.
    ENCODE_REQUEST = b"\x01"
    ERROR = b"\x02"


class EmbeddingMsg(
    msgspec.Struct,
    array_like=True,  # type: ignore[call-arg]
    gc=False,
):  # type: ignore[call-arg]
    """One encoder output, keyed by the hash both sides cache it under."""

    mm_hash: str
    embedding: torch.Tensor


class ECZmqProtocol:
    """Encodes and decodes EC messages.

    `MsgpackEncoder` / `MsgpackDecoder` are not thread-safe, so each thread
    that touches the wire owns its own instance.
    """

    def __init__(self) -> None:
        self._encoder = MsgpackEncoder()
        # share_mem=False copies the embedding out of the ZMQ frames (into
        # pinned memory when available), so the frames can be released as soon
        # as the message is decoded, and the tensor is ready for an async H2D.
        self._decoder = MsgpackDecoder(EmbeddingMsg, share_mem=False)

    def encode_embedding(self, mm_hash: str, embedding: torch.Tensor) -> list[bytestr]:
        msg = EmbeddingMsg(mm_hash=mm_hash, embedding=embedding)
        return [ECZmqMsgType.EMBEDDING.value, *self._encoder.encode(msg)]

    def decode_embedding(self, frames: list[bytestr]) -> EmbeddingMsg:
        """Decode frames received for an `EMBEDDING` message.

        Args:
            frames: the multipart frames *including* the type frame.

        Raises:
            ValueError: the frames do not carry an `EMBEDDING` message.
        """
        msg_type = self.peek_type(frames)
        if msg_type is not ECZmqMsgType.EMBEDDING:
            raise ValueError(f"Expected an EMBEDDING message, got {msg_type}")
        return self._decoder.decode(frames[1:])

    @staticmethod
    def peek_type(frames: list[bytestr]) -> ECZmqMsgType:
        """Read the message type without deserializing the payload.

        Raises:
            ValueError: the sequence is empty or the type is unknown.
        """
        if not frames:
            raise ValueError("Received an empty EC message")
        return ECZmqMsgType(bytes(frames[0]))
