# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shutdown test utils"""

import time
from dataclasses import dataclass, field

SHUTDOWN_TEST_TIMEOUT_SEC = 120
SHUTDOWN_TEST_THRESHOLD_BYTES = 2 * 2**30

# Standalone script test constants
SERVER_STARTUP_TIMEOUT = 180.0
DRAIN_TIMEOUT = 30


@dataclass
class DrainState:
    """Track state during drain shutdown tests."""

    got_503: bool = False
    requests_after_sigterm: int = 0
    connection_errors: int = 0
    stop_requesting: bool = False
    errors: list[str] = field(default_factory=list)


def wait_for_server_ready(port: int, timeout: float) -> bool:
    """Wait for API server to be ready via health endpoint (sync version)."""
    try:
        import httpx
    except ImportError as e:
        raise ImportError("httpx required for wait_for_server_ready") from e

    start = time.time()
    while time.time() - start < timeout:
        try:
            resp = httpx.get(f"http://localhost:{port}/health", timeout=2)
            if resp.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(1.0)
    return False


async def wait_for_server_ready_async(
    client,
    model: str,
    timeout: float,
) -> bool:
    """Wait for API server to be ready (async version for openai client)."""
    import asyncio

    start = time.time()
    while time.time() - start < timeout:
        try:
            await client.completions.create(model=model, prompt="Hi", max_tokens=1)
            return True
        except Exception:
            await asyncio.sleep(1.0)
    return False


def get_child_pids(parent_pid: int) -> list[int]:
    """Get all child process PIDs recursively."""
    try:
        import psutil

        parent = psutil.Process(parent_pid)
        return [c.pid for c in parent.children(recursive=True)]
    except (ImportError, Exception):
        return []
