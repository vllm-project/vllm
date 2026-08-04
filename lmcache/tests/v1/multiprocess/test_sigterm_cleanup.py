# SPDX-License-Identifier: Apache-2.0
"""Regression coverage for graceful MP server signal handling."""

# Standard
from pathlib import Path
import os
import shutil
import signal
import socket
import subprocess
import sys
import time

# Third Party
import pytest

# First Party
from lmcache import torch_dev, torch_device_type

PROJECT_ROOT = Path(__file__).parents[3]
STARTUP_TIMEOUT_SECONDS = 90.0
SHUTDOWN_TIMEOUT_SECONDS = 60.0
SHM_SIZE_BYTES = 64 << 20

pytestmark = pytest.mark.skipif(
    not sys.platform.startswith("linux")
    or not (torch_dev.is_available() and torch_device_type == "cuda"),
    reason="Mixed-allocator POSIX SHM requires Linux and an accelerator device",
)


def _unused_tcp_port() -> int:
    """Reserve an unused local TCP port for the server subprocess."""
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _run_cli_and_signal(
    tmp_path: Path,
    module: str,
    ready_log: str,
    shutdown_signal: signal.Signals,
    extra_args: list[str] | None = None,
) -> tuple[int, str]:
    """Run an MP CLI, signal it after readiness, and verify SHM cleanup."""
    shm_dir = Path("/dev/shm")
    if shutil.disk_usage(shm_dir).free < 2 * SHM_SIZE_BYTES:
        pytest.skip("insufficient /dev/shm capacity for the regression test")

    log_path = tmp_path / f"{module.rsplit('.', 1)[-1]}-{shutdown_signal.name}.log"
    env = os.environ.copy()
    python_path = env.get("PYTHONPATH")
    env["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(PROJECT_ROOT), python_path) if part
    )
    command = [
        sys.executable,
        "-m",
        module,
        "--host",
        "127.0.0.1",
        "--port",
        str(_unused_tcp_port()),
        "--l1-size-gb",
        "0.0625",
        "--no-l1-use-lazy",
        "--supported-transfer-mode",
        "auto",
        "--eviction-policy",
        "LRU",
        "--disable-observability",
    ]
    if extra_args is not None:
        command.extend(extra_args)

    with log_path.open("w") as log_file:
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )

    shm_path = shm_dir / f"lmcache_l1_pool_{process.pid}"
    try:
        deadline = time.monotonic() + STARTUP_TIMEOUT_SECONDS
        while time.monotonic() < deadline:
            if process.poll() is not None:
                pytest.fail(
                    f"MP server exited before becoming ready:\n"
                    f"{log_path.read_text(errors='replace')}"
                )
            log = log_path.read_text(errors="replace")
            if shm_path.exists() and ready_log in log:
                break
            time.sleep(0.2)
        else:
            pytest.fail(
                f"MP server did not become ready:\n"
                f"{log_path.read_text(errors='replace')}"
            )

        process.send_signal(shutdown_signal)
        return_code = process.wait(timeout=SHUTDOWN_TIMEOUT_SECONDS)

        log = log_path.read_text(errors="replace")
        assert not shm_path.exists(), (
            f"named SHM pool leaked after {shutdown_signal.name}:\n{log}"
        )
        return return_code, log
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=10)
        if shm_path.exists():
            shm_path.unlink()


def test_zmq_cli_sigterm_unlinks_shm_pool(tmp_path: Path) -> None:
    """A ready ZMQ-only MP CLI must unlink its SHM pool after SIGTERM."""
    return_code, log = _run_cli_and_signal(
        tmp_path,
        module="lmcache.v1.multiprocess.server",
        ready_log="LMCache cache server is running...",
        shutdown_signal=signal.SIGTERM,
    )

    assert return_code == 0
    assert "MPCacheServer closed" in log


@pytest.mark.parametrize(
    "shutdown_signal",
    [signal.SIGINT, signal.SIGTERM],
    ids=["sigint", "sigterm"],
)
def test_http_cli_signal_unlinks_shm_pool(
    tmp_path: Path,
    shutdown_signal: signal.Signals,
) -> None:
    """A ready HTTP MP CLI must unlink its SHM pool on graceful signals."""
    _, log = _run_cli_and_signal(
        tmp_path,
        module="lmcache.v1.multiprocess.http_server",
        ready_log="LMCache HTTP server initialized",
        shutdown_signal=shutdown_signal,
        extra_args=[
            "--http-host",
            "127.0.0.1",
            "--http-port",
            str(_unused_tcp_port()),
        ],
    )

    assert "MPCacheServer closed" in log
    assert "LMCache HTTP server stopped" in log
