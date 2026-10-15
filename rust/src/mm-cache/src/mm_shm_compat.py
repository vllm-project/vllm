#!/usr/bin/env -S uv run
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "msgpack>=1,<2",
# ]
# ///
"""Validate a shm chunk written by the Rust `MmProcessorShmCache`.

Reads the chunk at the given address out of the POSIX shm segment, verifies
the HMAC handle signature against the 32-byte secret at the segment start,
and msgpack-decodes the marker-prefixed body like the Python receiver's
`MsgpackSerde.deserialize` would.
"""

import hashlib
import hmac
import mmap
import struct
import sys

import msgpack


def main() -> None:
    shm_path, address, monotonic_id, buffer_size, key, sig_hex = (
        sys.argv[1],
        int(sys.argv[2]),
        int(sys.argv[3]),
        int(sys.argv[4]),
        sys.argv[5],
        sys.argv[6],
    )

    with open(shm_path, "rb") as f:
        buf = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)

    secret = bytes(buf[:32])

    # signature = HMAC_SHA256(secret, sha256(key) + address_le8 + id_le4)
    def sign(k: str) -> bytes:
        payload = (
            hashlib.sha256(k.encode("utf-8")).digest()
            + address.to_bytes(8, "little")
            + monotonic_id.to_bytes(4, "little")
        )
        return hmac.new(secret, payload, hashlib.sha256).digest()

    signature = bytes.fromhex(sig_hex)
    assert hmac.compare_digest(signature, sign(key)), "handle signature mismatch"
    assert not hmac.compare_digest(signature, sign("other-key")), "wrong key verified"

    offset = 32 + address % buffer_size
    chunk_id, chunk_size = struct.unpack_from("<ii", buf, offset)
    assert chunk_id == monotonic_id, (chunk_id, monotonic_id)
    # `size` includes the 8-byte header.
    assert chunk_size > 8

    payload = buf[offset + 8 : offset + chunk_size]
    (reader_count,) = struct.unpack_from("<i", payload, 0)
    assert reader_count == 0, reader_count
    # Format marker 0x01: the rest is one raw msgpack body.
    assert payload[4] == 0x01, hex(payload[4])
    body = payload[5:]

    def ext_hook(code: int, data: bytes):
        assert code == 3, f"unexpected ext type {code}"
        return ("raw_view", data)

    item = msgpack.unpackb(body, ext_hook=ext_hook, strict_map_key=False)
    assert set(item) == {"offset", "pixel_values", "small"}, sorted(item)

    for key, elem in item.items():
        assert elem["field"] == ["batched", {"keep_on_cpu": False}], (key, elem)

    assert item["offset"]["data"] == 7

    dtype, shape, raw_view = item["pixel_values"]["data"]
    assert dtype == "float32", dtype
    assert shape == [128], shape
    assert raw_view == ("raw_view", struct.pack("<128f", *range(128)))

    dtype, shape, raw_view = item["small"]["data"]
    assert dtype == "float32", dtype
    assert shape == [2], shape
    assert raw_view == ("raw_view", struct.pack("<2f", 1.0, 2.0))

    print("OK")


if __name__ == "__main__":
    main()
