#!/opt/venv/bin/python
# SPDX-License-Identifier: Apache-2.0
"""Example plugin for LMCache system
This plugin runs continuously and exits when parent process terminates"""

# Standard
import json
import os
import signal
import time

# First Party
from lmcache.integration.vllm.utils import lmcache_get_or_create_config
from lmcache.v1.config import LMCacheEngineConfig


# Graceful exit handler
def handle_exit(signum, frame):
    print("Received termination signal, exiting...")
    exit(0)


signal.signal(signal.SIGTERM, handle_exit)

role = os.getenv("LMCACHE_RUNTIME_PLUGIN_ROLE")
worker_id = os.getenv("LMCACHE_RUNTIME_PLUGIN_WORKER_ID")
worker_count = os.getenv("LMCACHE_RUNTIME_PLUGIN_WORKER_COUNT")
config_str = os.getenv("LMCACHE_RUNTIME_PLUGIN_CONFIG")
try:
    config = LMCacheEngineConfig.from_json(config_str)
except json.JSONDecodeError as e:
    print(f"Error parsing LMCACHE_RUNTIME_PLUGIN_CONFIG: {e}")
    config = lmcache_get_or_create_config()

print(
    f"Python plugin running with role: {role}, worker_id: {worker_id}, "
    f"worker_count: {worker_count}"
)
print(f"Config: {config}")

# Main loop
loop_count = 0
while True:
    print(f"Scheduler plugin is running... (loop_count: {loop_count})")
    loop_count += 1
    time.sleep(10)
