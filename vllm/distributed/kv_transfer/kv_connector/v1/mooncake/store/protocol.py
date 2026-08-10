# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Wire-format constants for the LookupKey ZMQ admin channel.

This is the single source of truth shared by ``LookupKeyClient`` and
``LookupKeyServer`` on the scheduler<->worker rank-0 admin channel.

Wire format (REQ/REP over IPC):

    Request: [msg_type: bytes] [payload_frames...]

      msg_type == LOOKUP_MSG:
          frame 1: num_tokens (u32 big-endian, 4 bytes); the worker derives
                   the aligned lookup length
          frame 2: hash_len (u16 big-endian, 2 bytes) — byte length of each
                   fixed-size block hash (0 when there are no hashes)
          frame 3: raw block hashes concatenated back-to-back (each hash_len
                   bytes); the server splits on hash_len
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
