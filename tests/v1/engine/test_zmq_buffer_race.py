# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for ZMQ zero-copy buffer race condition fix.

The old process_output_sockets code reused bytearray buffers for zero-copy
ZMQ sends.  Under high concurrency the tracker could report ``done`` before
ZMQ's I/O thread had fully released the memory, allowing the buffer to be
overwritten while still in flight — producing garbled msgpack frames on the
receiving end.

These tests verify:
1. The aliasing mechanism that enables the corruption.
2. That buffer reuse *does* corrupt data for messages that hit ZMQ's
   zero-copy path (>= 65 536 bytes in pyzmq/libzmq 4.x).
3. That allocating a fresh buffer each time prevents the corruption.

See: https://github.com/vllm-project/vllm/issues/24655
"""

import msgspec
import pytest
import zmq

pytestmark = pytest.mark.cpu_test

# pyzmq / libzmq stores messages < 65536 bytes inline in the zmq_msg_t
# struct (effectively copying them), but uses true zero-copy for larger
# messages.  The corruption only manifests on the zero-copy path.
ZMQ_ZERO_COPY_THRESHOLD = 65536


class LargeOutput(
    msgspec.Struct,
    array_like=True,  # type: ignore[call-arg]
    omit_defaults=True,  # type: ignore[call-arg]
):
    """Minimal msgspec type whose encoding exceeds ZMQ_ZERO_COPY_THRESHOLD."""

    request_id: str
    data: list[int] = []


def _make_large_msg(
    tag: str, *, min_size: int = ZMQ_ZERO_COPY_THRESHOLD
) -> LargeOutput:
    """Return a LargeOutput whose msgpack encoding is >= *min_size* bytes."""
    # Each int element encodes to ~1-5 bytes in msgpack.  Overshoot a bit
    # to guarantee we cross the threshold.
    n_items = min_size  # list[int] with small ints → ~1 byte each + overhead
    return LargeOutput(request_id=tag, data=list(range(n_items)))


@pytest.fixture
def zmq_push_pull():
    """Yield a (push, pull) inproc socket pair and clean up afterwards."""
    ctx = zmq.Context()
    push = ctx.socket(zmq.PUSH)
    pull = ctx.socket(zmq.PULL)
    addr = "inproc://test-zmq-buffer-race"
    push.bind(addr)
    pull.connect(addr)
    yield push, pull
    push.close(linger=0)
    pull.close(linger=0)
    ctx.term()


# ── 1. Mechanism proof ──────────────────────────────────────────────


def test_zmq_frame_aliases_mutable_buffer():
    """zmq.Frame(bytearray, copy=False) directly references the buffer.

    Mutating the bytearray after Frame creation changes what the Frame
    reports.  This aliasing is the prerequisite for the race.
    """
    buf = bytearray(b"original message content!")
    frame = zmq.Frame(buf, copy=False)
    assert bytes(frame) == b"original message content!"

    buf[:8] = b"CORRUPT!"
    assert bytes(frame) == b"CORRUPT! message content!"


def test_large_bytearray_is_zero_copied_by_zmq(zmq_push_pull):
    """pyzmq does true zero-copy for raw bytearrays >= 65536 bytes.

    Below this threshold, data is copied into the zmq_msg_t inline
    storage, making the aliasing harmless.  At or above the threshold
    the bytearray's memory is shared — mutations after send are visible
    to the receiver.
    """
    push, pull = zmq_push_pull

    # Below threshold: safe (pyzmq copies internally).
    small = bytearray(b"A" * (ZMQ_ZERO_COPY_THRESHOLD - 1))
    push.send_multipart([small], copy=False)
    small[:4] = b"ZZZZ"
    received = pull.recv_multipart()
    assert received[0][:4] == b"AAAA", "small buffer should NOT be aliased"

    # At threshold: zero-copy (aliased).
    large = bytearray(b"B" * ZMQ_ZERO_COPY_THRESHOLD)
    push.send_multipart([large], copy=False)
    large[:4] = b"ZZZZ"
    received = pull.recv_multipart()
    assert received[0][:4] == b"ZZZZ", "large buffer SHOULD be aliased"


# ── 2. Corruption with buffer reuse ────────────────────────────────


def test_buffer_reuse_corrupts_large_messages(zmq_push_pull):
    """Reusing the encode_into buffer corrupts large in-flight messages.

    This reproduces the mechanism behind the race condition: the encoder
    clears and rewrites the same bytearray, and because the message is
    large enough to hit ZMQ's zero-copy path, the receiver sees the
    *second* encoding instead of the first.
    """
    push, pull = zmq_push_pull
    encoder = msgspec.msgpack.Encoder()
    decoder = msgspec.msgpack.Decoder(LargeOutput)

    msg1 = _make_large_msg("msg1")
    msg2 = _make_large_msg("msg2")

    buffer = bytearray()
    encoder.encode_into(msg1, buffer)
    assert len(buffer) >= ZMQ_ZERO_COPY_THRESHOLD

    snapshot = bytes(buffer)
    push.send_multipart([buffer], copy=False)

    # Reuse the buffer immediately — overwrites in-flight data.
    encoder.encode_into(msg2, buffer)

    frames = pull.recv_multipart()
    received = bytes(frames[0])

    # The received data should NOT match the original msg1 encoding.
    assert received != snapshot, (
        "Expected corruption from buffer reuse, "
        "but the data was intact — the zero-copy aliasing "
        "did not occur on this platform/pyzmq combination."
    )

    # Decoding should give msg2's content or fail entirely.
    try:
        result = decoder.decode(received)
        assert result.request_id != "msg1", (
            "Received msg1 intact despite buffer overwrite"
        )
    except msgspec.DecodeError:
        pass  # garbled data — confirms corruption


# ── 3. Fresh buffers prevent corruption ─────────────────────────────


def test_fresh_buffers_prevent_corruption(zmq_push_pull):
    """Allocating a fresh bytearray per send prevents corruption.

    This verifies the fix: even with large messages that cross the
    zero-copy threshold, each message is encoded into its own buffer,
    so no aliasing can occur.
    """
    push, pull = zmq_push_pull
    encoder = msgspec.msgpack.Encoder()
    decoder = msgspec.msgpack.Decoder(LargeOutput)

    messages = [_make_large_msg(f"msg{i}") for i in range(20)]

    for msg in messages:
        buffer = bytearray()  # fresh buffer — the fix
        encoder.encode_into(msg, buffer)
        push.send_multipart([buffer], copy=False)

    for expected in messages:
        frames = pull.recv_multipart()
        result = decoder.decode(frames[0])
        assert result.request_id == expected.request_id
        assert result.data == expected.data
