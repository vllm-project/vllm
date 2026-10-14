# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Wire-format constants for the LookupKey ZMQ admin channel.

This is the single source of truth shared by ``LookupKeyClient`` and
``LookupKeyServer`` on the scheduler<->worker admin channel. The channel is
addressed per worker rank, so an admin command that concerns every rank is sent
to each of them in turn.

Wire format (REQ/REP over IPC):

    Request: [msg_type: bytes] [payload_frames...]

      msg_type == LOOKUP_MSG:
          frame 1: num_tokens (u32 big-endian, 4 bytes); the worker derives
                   the aligned lookup length
          frame 2: hash_len (u16 big-endian, 2 bytes) — byte length of each
                   fixed-size block hash (0 when there are no hashes)
          frame 3: raw block hashes concatenated back-to-back (each hash_len
                   bytes); the server splits on hash_len
        Response: hit_length (u32 big-endian, first 4 bytes), followed by zero
                  or more 8-byte tail-key boundaries. Each entry is group_id
                  (u32), then boundary_tokens (u32).

      msg_type == RESET_PREPARE_MSG:
          (no payload frames)
        Response: [RESP_OK] or [RESP_ERR]. Waits for the stores one rank has in
        flight, so a store it started before a wipe on another rank cannot land
        in the fresh master afterwards. Sent to every rank before the wipe.

      msg_type == RESET_FORGET_MSG:
          (no payload frames)
        Response: [RESP_OK] or [RESP_ERR]. Drops the residency one rank
        buffered, which describes objects a wipe removed and would otherwise be
        announced as residency of the fresh Store. Sent to every rank after the
        wipe, so a wipe that fails leaves reports of objects the master still
        holds intact.

      msg_type == RESET_MSG:
          (no payload frames)
        Response: [RESP_OK] or [RESP_ERR]. Waits for the stores this rank has in
        flight, wipes the master, and drops the residency it buffered for what
        was removed. Runs on one rank, after ``RESET_PREPARE_MSG`` on every rank.

The first frame of every request is a named bytes tag (not a numeric
sentinel that aliases the data field) so the protocol stays
self-describing and extensible: adding new admin commands requires
only a new tag and a new dispatch branch.

Mirrors the named-tag convention used by the NIXL connector (see
``vllm/distributed/kv_transfer/kv_connector/v1/nixl/metadata.py``).
"""

from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.data import (
    MooncakeLookupResult,
    TailKeyBoundary,
)

# Request message-type tags. Frame 0 of every request.
LOOKUP_MSG: bytes = b"lookup"
RESET_PREPARE_MSG: bytes = b"reset_prepare"
RESET_FORGET_MSG: bytes = b"reset_forget"
RESET_MSG: bytes = b"reset"

# Single-byte response status codes for admin commands.
RESP_OK: bytes = b"\x01"
RESP_ERR: bytes = b"\x00"

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
