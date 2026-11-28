# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import contextlib
import os
import signal
import subprocess
import time
from argparse import ArgumentError, ArgumentParser
from pathlib import Path
from types import TracebackType
from typing import BinaryIO, NoReturn, Self
from urllib.parse import urlsplit

import requests

from vllm.benchmarks.lib.utils import parse_request_headers
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.utils.network_utils import join_host_port


class _BenchmarkArgsParser(FlexibleArgumentParser):
    def error(self, message: str) -> NoReturn:
        raise ArgumentError(None, message)


class ServerProcess:
    VLLM_RESET_CACHE_ENDPOINTS = [
        "/reset_prefix_cache",
        "/reset_mm_cache",
        "/reset_encoder_cache",
    ]

    def __init__(
        self,
        server_cmd: list[str],
        after_bench_cmd: list[str],
        *,
        show_stdout: bool,
        session: requests.Session | None = None,
    ) -> None:
        super().__init__()

        self.server_cmd = server_cmd
        self.after_bench_cmd = after_bench_cmd
        self.show_stdout = show_stdout
        self.http = session or requests
        self._stopped = True

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

    def start(
        self, *, log: BinaryIO | None = None, enable_cache_reset: bool = True
    ) -> None:
        stdout: BinaryIO | int | None = log
        if log is None and not self.show_stdout:
            stdout = subprocess.DEVNULL

        server_cmd = self.server_cmd
        if enable_cache_reset and server_cmd[0].endswith("vllm"):
            server_cmd = [*server_cmd, "--enable-cache-reset"]

        # Create new process for clean termination
        self.launch_time = time.monotonic()
        self._server_process = subprocess.Popen(
            server_cmd,
            start_new_session=True,
            stdout=stdout,
            stderr=subprocess.STDOUT if log is not None else None,
        )
        self._stopped = False

    def stop(self, *, grace_period: float | None = None) -> bool:
        """Return True if this call had to force shutdown."""
        if self._stopped:
            return False
        if grace_period is None:
            self._kill_group()
            return True

        try:
            with contextlib.suppress(OSError):
                self._server_process.terminate()
            self._server_process.wait(timeout=grace_period)
        except subprocess.TimeoutExpired:
            return True
        finally:
            self._kill_group(timeout=10)
        return False

    def _kill_group(self, timeout: float | None = None) -> None:
        server_process = self._server_process
        # start_new_session makes the server PID the PGID, even after it exits.
        with contextlib.suppress(OSError):
            os.killpg(server_process.pid, signal.SIGKILL)
        server_process.wait(timeout=timeout)
        self._stopped = True

    def run_subcommand(self, cmd: list[str]):
        return subprocess.run(
            cmd,
            stdout=None if self.show_stdout else subprocess.DEVNULL,
            check=True,
        )

    def after_bench(self, bench_cmd: list[str] | None = None) -> None:
        if not self.after_bench_cmd:
            self.reset_caches(bench_cmd)
            return

        self.run_subcommand(self.after_bench_cmd)

    def _get_vllm_server_address(self) -> str:
        parser = ArgumentParser(add_help=False, allow_abbrev=False, exit_on_error=False)
        parser.add_argument("--host", default="127.0.0.1")
        parser.add_argument("-p", "--port", type=int, default=8000)
        try:
            args, _ = parser.parse_known_args(self.server_cmd)
        except ArgumentError as exc:
            raise ValueError(str(exc)) from exc
        # Connect to loopback when the server binds every local interface.
        host = {"0.0.0.0": "127.0.0.1", "::": "::1"}.get(args.host, args.host)
        return "http://" + join_host_port(host, args.port)

    @property
    def server_address(self) -> str:
        return self._get_vllm_server_address()

    def is_server_ready(self, timeout: float | None = None) -> bool:
        try:
            with self.http.get(
                f"{self.server_address}/health", timeout=timeout, stream=True
            ) as response:
                return response.status_code == 200
        except requests.RequestException:
            return False

    def wait_until_ready(
        self,
        timeout: float,
        *,
        poll_interval: float = 1.0,
        request_timeout: float | None = None,
    ) -> None:
        start_time = time.monotonic()
        while not self.is_server_ready(timeout=request_timeout):
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
            time.sleep(poll_interval)

    @staticmethod
    def _get_reset_authorization(
        bench_cmd: list[str] | None, server_address: str
    ) -> str | None:
        if (
            not bench_cmd
            or Path(bench_cmd[0]).name != "vllm"
            or bench_cmd[1:3] != ["bench", "serve"]
        ):
            return None

        from vllm.benchmarks.serve import add_cli_args

        parser = _BenchmarkArgsParser(add_help=False, exit_on_error=False)
        add_cli_args(parser)
        try:
            args = parser.parse_args(bench_cmd[3:])
            headers = parse_request_headers(args.header)
            if headers is None or "Authorization" not in headers:
                return None

            server_url = urlsplit(server_address)
            base_url = args.base_url
            if base_url is None:
                base_url = f"http://{join_host_port(args.host, args.port)}"
            bench_url = urlsplit(f"{base_url}{args.endpoint}")
            bench_port = 80 if bench_url.port is None else bench_url.port
            if (
                bench_url.scheme,
                bench_url.hostname,
                bench_port,
            ) != (server_url.scheme, server_url.hostname, server_url.port):
                return None
        except (ArgumentError, ValueError):
            return None

        return headers["Authorization"]

    def reset_caches(self, bench_cmd: list[str] | None = None) -> None:
        server_cmd = self.server_cmd

        # Use `.endswith()` to match `/bin/...`
        if server_cmd[0].endswith("vllm"):
            server_address = self.server_address
            print(f"Resetting caches at {server_address}")
            headers = {}
            authorization = self._get_reset_authorization(bench_cmd, server_address)
            if authorization is not None:
                headers["Authorization"] = authorization
            elif api_key := os.environ.get("OPENAI_API_KEY"):
                headers["Authorization"] = f"Bearer {api_key}"

            for endpoint in self.VLLM_RESET_CACHE_ENDPOINTS:
                res = self.http.post(server_address + endpoint, headers=headers)
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
