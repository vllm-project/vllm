# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Wire-format constants for the LookupKey ZMQ admin channel.

This is the single source of truth shared by `LookupKeyClient` and
`LookupKeyServer` on the scheduler<->worker rank-0 admin channel.

Wire format (DEALER/ROUTER over IPC). The client may have many requests in
flight; the server answers them out of order and the request id pairs each
reply with its request.

    Request: [request_id] [budget_ms] [msg_type: bytes] [payload_frames...]
      request_id: u64 big-endian, 8 bytes; echoed back in the reply
      budget_ms: u32 big-endian, 4 bytes; time the client will wait. The
                 server skips a request still queued once its budget is spent.

    Reply: [request_id] [status] [payload]
      status: RESP_OK, RESP_ERR (the handler failed) or RESP_EXPIRED (the
              budget ran out before the handler started)

      msg_type == LOOKUP_MSG:
          frame 1: num_tokens (u32 big-endian, 4 bytes); the worker derives
                   the aligned lookup length
          frame 2: hash_len (u16 big-endian, 2 bytes) — byte length of each
                   fixed-size block hash (0 when there are no hashes)
          frame 3: raw block hashes concatenated back-to-back (each hash_len
                   bytes); the server splits on hash_len
        Reply payload (RESP_OK): hit_length (u32 big-endian, first 4 bytes),
                  followed by zero or more 8-byte tail-key boundaries. Each
                  entry is group_id (u32), then boundary_tokens (u32).

      msg_type == RESET_MSG:
          (no payload frames)
        Reply payload: empty; status RESP_OK or RESP_ERR

The first frame of every request is a named bytes tag (not a numeric
sentinel that aliases the data field) so the protocol stays
self-describing and extensible: adding new admin commands requires
only a new tag and a new dispatch branch.

Mirrors the named-tag convention used by the NIXL connector (see
`vllm/distributed/kv_transfer/kv_connector/v1/nixl/metadata.py`).
"""

from vllm.distributed.kv_transfer.kv_connector.v1.mooncake_store.data import (
    MooncakeLookupResult,
    TailKeyBoundary,
)

# Request message-type tags. Frame 0 of every request.
LOOKUP_MSG: bytes = b"lookup"
RESET_MSG: bytes = b"reset"

# Single-byte reply status codes.
RESP_OK: bytes = b"\x01"
RESP_ERR: bytes = b"\x00"
RESP_EXPIRED: bytes = b"\x02"

REQUEST_ID_SIZE: int = 8
BUDGET_MS_SIZE: int = 4

# group_id (u32), boundary_tokens (u32).
TAIL_KEY_BOUNDARY_ENTRY_SIZE: int = 8


def encode_lookup_response(result: MooncakeLookupResult) -> bytes:
    hit_length = result.hit_length.to_bytes(4, "big")
    if not result.tail_key_boundaries:
        return hit_length

    payload = bytearray(hit_length)
    for boundary in result.tail_key_boundaries:
        payload.extend(boundary.group_id.to_bytes(4, "big"))
        payload.extend(boundary.num_tokens.to_bytes(4, "big"))
    return bytes(payload)


def decode_lookup_response(payload: bytes) -> MooncakeLookupResult:
    if len(payload) < 4 or (len(payload) - 4) % TAIL_KEY_BOUNDARY_ENTRY_SIZE:
        raise ValueError("Invalid Mooncake lookup response")

    hit_length = int.from_bytes(payload[:4], "big")
    if len(payload) == 4:
        return MooncakeLookupResult(hit_length)

    boundaries = []
    for offset in range(4, len(payload), TAIL_KEY_BOUNDARY_ENTRY_SIZE):
        boundaries.append(
            TailKeyBoundary(
                group_id=int.from_bytes(payload[offset : offset + 4], "big"),
                num_tokens=int.from_bytes(payload[offset + 4 : offset + 8], "big"),
            )
        )
    return MooncakeLookupResult(hit_length, tuple(boundaries))
