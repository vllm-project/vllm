#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Cross-node mid-transfer failure driver: P on a remote host, D + proxy local.
#
# Setup (not done by this script): P with digest on
# (kv_connector_extra_config.enable_kv_digest=true), D with digest on, toy
# proxy on 8192 pointing at P, and kv_digest_xnode_killer.py running on the P
# host (listens on $PREFILLER_KILL_PORT, kills P's --port $PREFILLER_PORT
# server). GB200 note: GPUDirect works there via dmabuf, so
# kv_buffer_device=cuda (no host staging); on Crusoe H200 GPUDirect is
# unavailable, use kv_buffer_device=cpu (host-staged IB) instead.
#
# The script fires one ~25k-token request (~2.9GB KV), trips the killer the
# instant D posts its NIXL READ, and reports the outcome. Expected: the
# request fails (500) via NIXL's reported transport error +
# kv_load_failure_policy - or, if the READ somehow completes with partial
# data, a KV digest mismatch. It must NOT silently serve corrupt output.
set -u
PREFILLER_HOST=${PREFILLER_HOST:?set to P host cluster IP}
PREFILLER_KILL_PORT=${PREFILLER_KILL_PORT:-9599}
D_LOG=${D_LOG:-~/pd1p1d/rdma_d.log}
WORK=$(mktemp -d)

BYTES0=$(wc -c < "$D_LOG")
echo "D log byte offset snapshot: $BYTES0 at $(date +%T.%3N)"

# Unique prompt every run (nonce) so D's prefix cache can't swallow the read.
python3 - "$WORK/prompt.json" <<'PYEOF'
import json, sys, time, uuid
nonce = f"NONCE-{uuid.uuid4().hex}-TS-{time.time()}"
lorem = ("Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do "
         "eiusmod tempor incididunt ut labore et dolore magna aliqua. ")
prompt = nonce + " " + lorem * 1150
req = {"model": "Qwen/Qwen3-0.6B",
       "messages": [{"role": "user", "content": prompt}],
       "max_tokens": 16, "temperature": 0.0, "stream": False}
with open(sys.argv[1], "w") as f:
    json.dump(req, f)
print(f"prompt chars: {len(prompt)}")
PYEOF

curl -sS -o "$WORK/resp.json" -w '%{http_code}' \
    -X POST http://localhost:8192/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d @"$WORK/prompt.json" --max-time 600 > "$WORK/httpcode" &
CURL_PID=$!
echo "TRIGGER $(date +%T.%3N) request sent (curl pid $CURL_PID)"

deadline=$((SECONDS + 120))
hit=""
while [ $SECONDS -lt $deadline ]; do
    if tail -c +$((BYTES0 + 1)) "$D_LOG" 2>/dev/null | grep -m1 -q "_read_blocks on remote"; then
        hit=1
        break
    fi
    sleep 0.1
done
if [ -n "$hit" ]; then
    echo "READ_IN_FLIGHT $(date +%T.%3N)"
    python3 -c "import socket; socket.create_connection(('$PREFILLER_HOST',$PREFILLER_KILL_PORT),2)"
    echo "KILL_TRIPPED $(date +%T.%3N)"
else
    echo "READ_LINE_NOT_SEEN by $(date +%T.%3N) — NOT tripping killer, aborting"
fi

wait $CURL_PID
echo "HTTP_CODE $(cat "$WORK/httpcode") at $(date +%T.%3N)"
echo "--- response body (first 2000B) ---"
head -c 2000 "$WORK/resp.json"; echo
rm -rf "$WORK"
echo "=== mid-transfer kill test done ==="
