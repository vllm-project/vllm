# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

OK = b"OK"
ERROR = b"ERROR"
EMPTY = b""
SHUTDOWN = b"SHUTDOWN"

POLL_INTERVAL = 1000  # ms
CLEANUP_INTERVAL = 0.1  # s

OPEN_WRITE = b"open_write"
CLOSE_WRITE = b"close_write"
OPEN_READ = b"open_read"
CLOSE_READ = b"close_read"
WAIT_WRITE = b"wait_write"
PIN = b"pin"
UNPIN = b"unpin"
DELETE = b"delete"
GET_INFO = b"get_info"
GET_MANAGER_STATE = b"get_manager_state"
GET_STORAGE_INFO = b"get_storage_info"
DEBUG_CLEAN = b"DEBUG_CLEAN"
