# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import contextlib
import os
import signal
import subprocess
import time
from types import TracebackType

import requests
from typing_extensions import Self


class ServerProcess:
    def __init__(
        self,
        server_cmd: list[str],
        after_bench_cmd: list[str],
        *,
        show_stdout: bool,
    ) -> None:
        super().__init__()

        self.server_cmd = server_cmd
        self.after_bench_cmd = after_bench_cmd
        self.show_stdout = show_stdout

    def __enter__(self) -> Self:
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        exc_traceback: TracebackType | None,
    ) -> None:
        self.stop()

    def start(self):
        # Create new process for clean termination
        self._server_process = subprocess.Popen(
            self.server_cmd,
            start_new_session=True,
            stdout=None if self.show_stdout else subprocess.DEVNULL,
            # Need `VLLM_SERVER_DEV_MODE=1` for `_reset_caches`
            env=os.environ | {"VLLM_SERVER_DEV_MODE": "1"},
        )

    def stop(self):
        server_process = self._server_process

        if server_process.poll() is None:
            # In case only some processes have been terminated
            with contextlib.suppress(ProcessLookupError):
                # We need to kill both API Server and Engine processes
                os.killpg(os.getpgid(server_process.pid), signal.SIGKILL)

    def run_subcommand(self, cmd: list[str]):
        return subprocess.run(
            cmd,
            stdout=None if self.show_stdout else subprocess.DEVNULL,
            check=True,
        )

    def after_bench(self) -> None:
        if not self.after_bench_cmd:
            self.reset_caches()
            return

        self.run_subcommand(self.after_bench_cmd)

    def _get_vllm_server_address(self) -> str:
        server_cmd = self.server_cmd

        for host_key in ("--host",):
            if host_key in server_cmd:
                host = server_cmd[server_cmd.index(host_key) + 1]
                break
        else:
            host = "localhost"

        for port_key in ("-p", "--port"):
            if port_key in server_cmd:
                port = int(server_cmd[server_cmd.index(port_key) + 1])
                break
        else:
            port = 8000  # The default value in vllm serve

        return f"http://{host}:{port}"

    def is_server_ready(self) -> bool:
        server_address = self._get_vllm_server_address()
        try:
            response = requests.get(f"{server_address}/health")
            return response.status_code == 200
        except requests.RequestException:
            return False

    def wait_until_ready(self, timeout: int) -> None:
        start_time = time.monotonic()
        while not self.is_server_ready():
            # Check if server process has crashed
            if self._server_process.poll() is not None:
                returncode = self._server_process.returncode
                raise RuntimeError(
                    f"Server process crashed with return code {returncode}"
                )
            if time.monotonic() - start_time > timeout:
                raise TimeoutError(
                    f"Server failed to become ready within {timeout} seconds."
                )
            time.sleep(1)

    def reset_caches(self) -> None:
        server_cmd = self.server_cmd

        # Use `.endswith()` to match `/bin/...`
        if server_cmd[0].endswith("vllm"):
            server_address = self._get_vllm_server_address()
            print(f"Resetting caches at {server_address}")

            res = requests.post(f"{server_address}/reset_prefix_cache")
            res.raise_for_status()

            res = requests.post(f"{server_address}/reset_mm_cache")
            res.raise_for_status()
        elif server_cmd[0].endswith("infinity_emb"):
            if "--vector-disk-cache" in server_cmd:
                raise NotImplementedError(
                    "Infinity server uses caching but does not expose a method "
                    "to reset the cache"
                )
        else:
            raise NotImplementedError(
                f"No implementation of `reset_caches` for `{server_cmd[0]}` server. "
                "Please specify a custom command via `--after-bench-cmd`."
            )
