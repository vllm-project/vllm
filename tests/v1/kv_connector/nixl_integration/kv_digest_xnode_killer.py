# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Killer side for the cross-node mid-transfer failure test.

Runs on the P host: listens on KILL_PORT (default 9599); on any TCP connect
(from kv_digest_xnode_driver.sh the moment D posts its NIXL READ), kill -9
ONLY the vllm serve on P's port (default 8100) so P dies mid-transfer without
touching unrelated vllm servers on the same host. Multi-trip (loops until
killed).

Usage: PREFILLER_KILL_PORT=9599 PREFILLER_PORT=8100 python3 kv_digest_xnode_killer.py
"""

import os
import socket
import subprocess
import time

kill_port = int(os.environ.get("PREFILLER_KILL_PORT", "9599"))
serve_port = os.environ.get("PREFILLER_PORT", "8100")
srv = socket.socket()
srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
srv.bind(("0.0.0.0", kill_port))
srv.listen(4)
print(f"killer armed on :{kill_port} at {time.strftime('%T')}", flush=True)
while True:
    conn, addr = srv.accept()
    conn.close()
    print(f"trip from {addr} at {time.strftime('%T.%f')}", flush=True)
    # `--` so pkill doesn't misparse the pattern as its own option.
    subprocess.run(["pkill", "-9", "-f", "--", f"--port {serve_port}"])
    print("pkill issued", flush=True)
