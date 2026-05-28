# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Wire-format constants for the LookupKey ZMQ admin channel.

This is the single source of truth shared by ``LookupKeyClient`` and
``LookupKeyServer`` on the scheduler<->worker rank-0 admin channel.

Wire format (REQ/REP over IPC):

    Request: [msg_type: bytes] [payload_frames...]

      msg_type == LOOKUP_MSG:
          frame 1: token_len (u32 big-endian, 4 bytes)
          frame 2..n: msgpack-encoded list[str] of block-hash hex digests
        Response: [hit_count: u32 big-endian, 4 bytes]

      msg_type == RESET_MSG:
          (no payload frames)
        Response: [RESP_OK] or [RESP_ERR]

The first frame of every request is a named bytes tag (not a numeric
sentinel that aliases the data field) so the protocol stays
self-describing and extensible: adding new admin commands requires
only a new tag and a new dispatch branch.

Mirrors the named-tag convention used by the NIXL connector (see
``vllm/distributed/kv_transfer/kv_connector/v1/nixl/metadata.py``).
"""

# Request message-type tags. Frame 0 of every request.
LOOKUP_MSG: bytes = b"lookup"
RESET_MSG: bytes = b"reset"

# Single-byte response status codes for admin commands.
RESP_OK: bytes = b"\x01"
RESP_ERR: bytes = b"\x00"
