#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# CPU offloading with LMCache in multi-process (MP) mode.
#
# In MP mode, LMCache runs as a standalone server process (`lmcache server`)
# that owns the KV cache storage. One or more vLLM instances connect to it via
# the `LMCacheMPConnector`. This is the recommended way to run LMCache for
# distributed KV storage and for sharing KV cache across vLLM instances.
#
# vLLM ships a built-in shortcut for this setup: pass `--kv-offloading-backend
# lmcache` together with `--kv-offloading-size <GiB>` and vLLM wires up the
# `LMCacheMPConnector` for you (it defaults to the LMCache server at
# tcp://localhost:5555, matching the `lmcache server` default).
#
# Requires `lmcache` to be installed (`pip install lmcache`).
# Learn more: https://docs.lmcache.ai
set -euo pipefail

MODEL=${MODEL:-meta-llama/Meta-Llama-3.1-8B-Instruct}

# 1. Launch the standalone LMCache server (binds tcp://localhost:5555 by
#    default). `--l1-size-gb` sets the CPU memory budget for the L1 cache.
echo "Starting LMCache server..."
lmcache server --host localhost --port 5555 --l1-size-gb 5 &
LMCACHE_SERVER_PID=$!
trap 'kill $LMCACHE_SERVER_PID 2>/dev/null || true' EXIT

# 2. Launch vLLM and offload KV cache to the LMCache server.
#    The MP connector currently requires the non-hybrid KV cache manager.
echo "Starting vLLM server with LMCache MP offloading..."
vllm serve "$MODEL" \
    --port 8000 \
    --kv-offloading-size 5 \
    --kv-offloading-backend lmcache \
    --disable-hybrid-kv-cache-manager

# Equivalent explicit configuration (instead of the two flags above):
#   --kv-transfer-config \
#   '{"kv_connector":"LMCacheMPConnector","kv_role":"kv_both",
#     "kv_connector_extra_config":{"lmcache.mp.host":"tcp://localhost",
#                                  "lmcache.mp.port":5555}}'
