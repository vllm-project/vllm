#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Example MP runtime plugin for LMCache multiprocess server.

In MP (multiprocess / ZMQ) mode the config passed via
``LMCACHE_RUNTIME_PLUGIN_CONFIG`` is an **aggregated JSON dict**
containing ``mp_config``, ``storage_manager_config``, and
``obs_config`` sections -- NOT a single ``LMCacheEngineConfig``.

Unlike the non-MP (vLLM-integrated) mode, the MP server has
no role/worker concept -- there is only a single server process.
The config is an aggregated dict of dataclass configs rather
than a single ``LMCacheEngineConfig``.

This plugin demonstrates:
  1. How to parse the MP-mode aggregated config.
  2. How to run a periodic background task (e.g. status reporter).
  3. How to handle graceful shutdown via SIGTERM.

Usage:
  Launch the LMCache MP server with:

    python -m lmcache.v1.multiprocess.server \
        --host localhost --port 5555 \
        --l1-size-gb 10 --eviction-policy LRU \
        --runtime-plugin-locations \
            examples/mp_runtime_plugins/
"""

# Standard
import json
import os
import signal
import sys
import time


def handle_exit(signum, frame):
    """Graceful exit on SIGTERM / SIGINT."""
    print("[mp_plugin] Received termination signal, exiting...")
    sys.exit(0)


signal.signal(signal.SIGTERM, handle_exit)
signal.signal(signal.SIGINT, handle_exit)


def parse_mp_config() -> dict:
    """Parse the aggregated MP config from the environment.

    Returns:
        A dict with keys like ``mp_config``,
        ``storage_manager_config``, ``obs_config``, etc.
        Returns an empty dict if the env var is missing or
        cannot be parsed.
    """
    raw = os.getenv("LMCACHE_RUNTIME_PLUGIN_CONFIG", "")
    if not raw:
        print("[mp_plugin] WARNING: LMCACHE_RUNTIME_PLUGIN_CONFIG is empty")
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        print("[mp_plugin] ERROR: failed to parse config: %s" % exc)
        return {}


def dump_parsed_config(config: dict) -> None:
    """Dump the parsed aggregated config as compact JSON.

    The parent process captures stdout line-by-line and logs each
    line separately, so we emit a single-line JSON to keep the
    output in one log entry.
    """
    print("[mp_plugin] config: %s" % json.dumps(config, default=str))


def main() -> None:
    config = parse_mp_config()

    print("[mp_plugin] Started")

    dump_parsed_config(config)

    # --- Periodic status reporter loop ---
    loop_count = 0
    while True:
        print("[mp_plugin] heartbeat #%d" % loop_count)
        loop_count += 1
        time.sleep(30)


if __name__ == "__main__":
    main()
