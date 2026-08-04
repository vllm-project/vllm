#!/bin/bash
# Example plugin for LMCache system
# This plugin runs continuously and exits when parent process terminates

# Handle termination signal
trap "echo 'Received termination signal, exiting...'; exit 0" SIGTERM

role="$LMCACHE_RUNTIME_PLUGIN_ROLE"
worker_id="$LMCACHE_RUNTIME_PLUGIN_WORKER_ID"
worker_count="$LMCACHE_RUNTIME_PLUGIN_WORKER_COUNT"
config="$LMCACHE_RUNTIME_PLUGIN_CONFIG"

echo "All plugin started for role: $role, worker ID: $worker_id, worker count: $worker_count"
echo "All plugin accept LMCache Config: $config"

loop_count=0
while true; do
    echo "All plugin is running for ${role} ${worker_id}...(loop_count: ${loop_count})"
    loop_count=$((loop_count + 1))
    sleep 10
done