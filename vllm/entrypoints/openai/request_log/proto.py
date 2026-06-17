# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# ZMQ frame-0 tags for the request-log channel.
FRAME_RECORD = b"record"
FRAME_SHUTDOWN = b"shutdown"

# Default high-water-mark for the PUSH side; messages beyond this are dropped.
DEFAULT_SNDHWM = 10000

# How long the writer will wait for new messages on each poll iteration.
POLL_INTERVAL_MS = 500

# How often (records) the writer fsync's the output file.
FSYNC_EVERY_N_RECORDS = 64

# Seconds the parent waits for the writer subprocess to signal "ready".
# The child uses multiprocessing's "spawn" start method, so it has to do a
# cold re-import of vllm before the writer_main function even runs. On a
# typical box that's already 3-5 seconds; with a slow disk or a debug build
# it can be 10+. The default is intentionally generous; override with the
# VLLM_REQUEST_LOG_WRITER_READY_TIMEOUT_S env var if you need tighter
# startup behavior or a longer one for very cold machines.
WRITER_READY_TIMEOUT_S = 30.0

# Seconds the parent waits for the writer subprocess to exit on shutdown
# before falling back to terminate().
WRITER_SHUTDOWN_TIMEOUT_S = 3.0
