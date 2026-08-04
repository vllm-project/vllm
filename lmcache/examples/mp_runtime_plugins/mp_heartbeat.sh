#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
#
# Example MP runtime plugin (Bash) for LMCache multiprocess server.
#
# In MP mode the config JSON contains aggregated sections:
#   mp_config, storage_manager_config, obs_config
#
# This script demonstrates:
#   1. Reading the MP-mode environment variables.
#   2. Extracting fields from the aggregated JSON via jq.
#   3. Running a periodic heartbeat loop.
#
# Requires: jq (optional, falls back to raw echo)

# Graceful shutdown
trap "echo '[mp_heartbeat] Received termination signal, exiting...'; exit 0" SIGTERM SIGINT

config="${LMCACHE_RUNTIME_PLUGIN_CONFIG}"

echo "[mp_heartbeat] Started"

# Dump config as compact single-line JSON.
# The parent captures stdout line-by-line, so avoid multi-line output.
if command -v jq &>/dev/null && [ -n "${config}" ]; then
    echo "[mp_heartbeat] config: $(echo "${config}" | jq -c .)"
else
    echo "[mp_heartbeat] config: ${config}"
fi

# Heartbeat loop
loop_count=0
while true; do
    echo "[mp_heartbeat] heartbeat #${loop_count}"
    loop_count=$((loop_count + 1))
    sleep 30
done
