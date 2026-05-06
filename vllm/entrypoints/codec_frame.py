# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Codec: token-native binary transport for vLLM.

Models emit uint32 token IDs internally. Converting to UTF-8 and wrapping
in JSON SSE envelopes wastes ~95% of the wire. Codec skips detokenization
entirely and ships IDs as MessagePack or Protobuf frames.

Bidirectional: clients can also submit prompts as token ID lists in the
same binary framing, eliminating the tokenization round-trip on ingress.

Wire formats
------------
msgpack  (Content-Type: application/x-msgpack):
    Each frame is a msgpack-encoded map.

protobuf (Content-Type: application/x-protobuf):
    Each frame is a 4-byte big-endian length prefix followed by the raw
    protobuf bytes for CodecFrame / CodecRequest (see PROTO_SCHEMA below).

Proto schema (for client-side code generation)
-----------------------------------------------
"""

import struct as _struct
from collections.abc import Sequence

import msgspec as _msgspec

# ── msgspec msgpack ────────────────────────────────────────────────────────────

_mp_encoder = _msgspec.msgpack.Encoder()
_mp_decoder = _msgspec.msgpack.Decoder()


def encode_msgpack(
    ids: Sequence[int],
    done: bool,
    finish_reason: str | None = None,
) -> bytes:
    obj: dict = {"ids": list(ids), "done": done}
    if finish_reason is not None:
        obj["finish_reason"] = finish_reason
    return _mp_encoder.encode(obj)


def decode_msgpack(data: bytes) -> dict:
    return _mp_decoder.decode(data)


# ── protobuf hand-rolled encoder ───────────────────────────────────────────────
# Schema:
#   message CodecFrame  { repeated uint32 ids=1[packed]; bool done=2; optional string finish_reason=3; }
#   message CodecRequest{ repeated uint32 prompt_ids=1[packed]; uint32 max_tokens=2; float temperature=3; repeated string stop=4; string stream_format=5; }
#
# Wire types: 0=varint, 2=len-delimited, 5=32-bit float

def _varint(n: int) -> bytes:
    out: list[int] = []
    while True:
        bits = n & 0x7F
        n >>= 7
        if n:
            out.append(bits | 0x80)
        else:
            out.append(bits)
            break
    return bytes(out)


def _decode_varint(data: bytes, pos: int) -> tuple[int, int]:
    result = shift = 0
    while True:
        b = data[pos]
        pos += 1
        result |= (b & 0x7F) << shift
        if not (b & 0x80):
            return result, pos
        shift += 7


def encode_protobuf_frame(
    ids: Sequence[int],
    done: bool,
    finish_reason: str | None = None,
) -> bytes:
    """Raw protobuf bytes for a CodecFrame (no length prefix)."""
    parts: list[bytes] = []
    if ids:
        packed = b"".join(_varint(i) for i in ids)
        parts += [b"\x0a", _varint(len(packed)), packed]  # field 1, wt=2
    parts += [b"\x10", b"\x01" if done else b"\x00"]       # field 2, wt=0
    if finish_reason is not None:
        enc = finish_reason.encode()
        parts += [b"\x1a", _varint(len(enc)), enc]          # field 3, wt=2
    return b"".join(parts)


def encode_protobuf(
    ids: Sequence[int],
    done: bool,
    finish_reason: str | None = None,
) -> bytes:
    """4-byte big-endian length-prefixed CodecFrame."""
    payload = encode_protobuf_frame(ids, done, finish_reason)
    return _struct.pack(">I", len(payload)) + payload


def decode_protobuf_request(data: bytes) -> dict:
    """Decode a length-prefixed CodecRequest protobuf message to a dict."""
    # strip 4-byte prefix if present
    if len(data) >= 4:
        declared = _struct.unpack_from(">I", data, 0)[0]
        if declared == len(data) - 4:
            data = data[4:]

    result: dict = {}
    pos = 0
    while pos < len(data):
        tag_byte, pos = _decode_varint(data, pos)
        field = tag_byte >> 3
        wt = tag_byte & 0x7
        if wt == 0:                              # varint
            val, pos = _decode_varint(data, pos)
            if field == 2:
                result["max_tokens"] = val
            # other varint fields: consumed and ignored
        elif wt == 1:                            # 64-bit — skip (not used in CodecRequest)
            pos += 8
        elif wt == 2:                            # length-delimited
            length, pos = _decode_varint(data, pos)
            payload = data[pos: pos + length]
            pos += length
            if field == 1:                       # prompt_ids (packed uint32)
                ids: list[int] = []
                p = 0
                while p < len(payload):
                    v, p = _decode_varint(payload, p)
                    ids.append(v)
                result["prompt_ids"] = ids
            elif field == 4:                     # stop (repeated string)
                result.setdefault("stop", []).append(payload.decode())
            elif field == 5:                     # stream_format
                result["stream_format"] = payload.decode()
            # other len-delimited fields: consumed and ignored
        elif wt == 5:                            # 32-bit float
            val = _struct.unpack_from("<f", data, pos)[0]
            pos += 4
            if field == 3:                       # temperature — last value wins (proto semantics)
                result["temperature"] = val
            # other 32-bit fields: consumed and ignored
        else:
            raise ValueError(
                f"Unsupported protobuf wire type {wt} at pos {pos - 1}; "
                "data may be corrupt or from an incompatible schema version."
            )
    return result


# ── shared helpers ─────────────────────────────────────────────────────────────

CONTENT_TYPE: dict[str, str] = {
    "json":     "text/event-stream",
    "msgpack":  "application/x-msgpack",
    "protobuf": "application/x-protobuf",
}

ENCODERS = {
    "msgpack":  encode_msgpack,
    "protobuf": encode_protobuf,
}


def encode_frame(
    fmt: str,
    ids: Sequence[int],
    done: bool,
    finish_reason: str | None = None,
) -> bytes:
    return ENCODERS[fmt](ids, done, finish_reason)


# ── proto schema (for clients) ─────────────────────────────────────────────────

PROTO_SCHEMA = """\
syntax = "proto3";

// Output frame — one per token batch in the binary stream.
message CodecFrame {
  repeated uint32 ids         = 1 [packed = true];
  bool            done        = 2;
  optional string finish_reason = 3;
}

// Binary request body for POST /v1/completions/codec
// Content-Type: application/x-msgpack  →  same keys as JSON dict below
// Content-Type: application/x-protobuf →  CodecRequest message
message CodecRequest {
  repeated uint32 prompt_ids   = 1 [packed = true];
  uint32          max_tokens   = 2;
  float           temperature  = 3;
  repeated string stop         = 4;
  string          stream_format = 5;  // "msgpack" or "protobuf"
}
"""
