# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run fixed-SHA prefix-routing E2E checks against two real vLLM servers."""

import argparse
import http.client
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

IDENTITY_MIDDLEWARE = (
    "tests.distributed.prefix_routing_e2e_middleware.PrefixRoutingE2EIdentityMiddleware"
)


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True, encoding="utf-8").strip()


def _vllm_cli() -> str:
    executable = Path(sys.executable).with_name("vllm")
    if not executable.is_file():
        raise RuntimeError(f"vLLM CLI not found next to Python: {executable}")
    return str(executable)


def _free_ports(count: int) -> list[int]:
    sockets: list[socket.socket] = []
    try:
        for _ in range(count):
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.bind(("127.0.0.1", 0))
            sockets.append(sock)
        return [int(sock.getsockname()[1]) for sock in sockets]
    finally:
        for sock in sockets:
            sock.close()


def _tail(path: Path, lines: int = 80) -> str:
    if not path.exists():
        return ""
    return "\n".join(
        path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:]
    )


def _wait_for_log(
    path: Path,
    marker: str,
    offset: int,
    timeout: float,
) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists():
            with path.open("rb") as log_file:
                log_file.seek(offset)
                if marker.encode("utf-8") in log_file.read():
                    return
        time.sleep(0.25)
    raise TimeoutError(f"log marker not found: {marker!r}\n{_tail(path)}")


@dataclass
class _ServerProcess:
    name: str
    command: list[str]
    environment: dict[str, str]
    log_path: Path
    process: subprocess.Popen | None = None
    log_file: Any = None

    def start(self) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_path.open("w", encoding="utf-8")
        kwargs: dict[str, Any] = {}
        if os.name == "nt":
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            kwargs["start_new_session"] = True
        self.process = subprocess.Popen(
            self.command,
            env=self.environment,
            stdout=self.log_file,
            stderr=subprocess.STDOUT,
            **kwargs,
        )

    def stop(self, timeout: float = 30.0) -> None:
        if self.process is None:
            return
        if self.process.poll() is None:
            if os.name == "nt":
                self.process.terminate()
            else:
                os.killpg(self.process.pid, signal.SIGTERM)
            try:
                self.process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                if os.name == "nt":
                    self.process.kill()
                else:
                    os.killpg(self.process.pid, signal.SIGKILL)
                self.process.wait(timeout=10)
        self._close_log()

    def kill(self) -> None:
        if self.process is None:
            return
        if self.process.poll() is None:
            if os.name == "nt":
                self.process.kill()
            else:
                os.killpg(self.process.pid, signal.SIGKILL)
            self.process.wait(timeout=10)
        self._close_log()

    def _close_log(self) -> None:
        if self.log_file is not None:
            self.log_file.close()
            self.log_file = None

    def assert_running(self) -> None:
        if self.process is not None and self.process.poll() is not None:
            raise RuntimeError(
                f"{self.name} exited with code {self.process.returncode}\n"
                f"{_tail(self.log_path)}"
            )


def _wait_for_health(server: _ServerProcess, url: str, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        server.assert_running()
        try:
            with urllib.request.urlopen(f"{url}/health", timeout=2) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, TimeoutError):
            pass
        time.sleep(1)
    raise TimeoutError(
        f"{server.name} did not become healthy within {timeout}s\n"
        f"{_tail(server.log_path)}"
    )


def _completion(
    url: str,
    model: str,
    prompt: str,
    *,
    headers: dict[str, str] | None = None,
    max_tokens: int = 1,
    timeout: float = 120.0,
) -> tuple[int, dict[str, str], bytes]:
    body = json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0,
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        f"{url}/v1/completions",
        data=body,
        headers={"content-type": "application/json", **(headers or {})},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return (
            response.status,
            {key.lower(): value for key, value in response.headers.items()},
            response.read(),
        )


def _wait_for_worker_route(
    master_url: str,
    model: str,
    prompt: str,
    timeout: float,
) -> dict[str, str]:
    deadline = time.monotonic() + timeout
    last_headers: dict[str, str] = {}
    while time.monotonic() < deadline:
        status, headers, _ = _completion(
            master_url,
            model,
            prompt,
            headers={"x-data-parallel-rank": "99"},
        )
        if status == 200 and headers.get("x-prefix-routing-e2e-node") == "node1":
            return headers
        last_headers = headers
        time.sleep(1)
    raise TimeoutError(
        "master did not route the warmed prefix to node1; "
        f"last response headers: {last_headers}"
    )


def _stream_then_stop_worker(
    master_port: int,
    model: str,
    prompt: str,
    worker: _ServerProcess,
    max_tokens: int,
) -> None:
    body = json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0,
            "ignore_eos": True,
            "stream": True,
        }
    ).encode("utf-8")
    connection = http.client.HTTPConnection("127.0.0.1", master_port, timeout=120)
    connection.request(
        "POST",
        "/v1/completions",
        body=body,
        headers={
            "content-type": "application/json",
            "content-length": str(len(body)),
        },
    )
    response = connection.getresponse()
    if response.status != 200:
        raise RuntimeError(f"streaming request returned HTTP {response.status}")
    if response.getheader("x-prefix-routing-e2e-node") != "node1":
        raise RuntimeError("streaming request was not routed to node1")
    if not response.read(1):
        raise RuntimeError("streaming response ended before its first body byte")

    worker.kill()
    interrupted = False
    try:
        response.read()
    except (http.client.HTTPException, OSError):
        interrupted = True
    finally:
        connection.close()
    if not interrupted:
        raise RuntimeError("stream completed after node1 was terminated")


def _make_prompt(label: str, repeats: int) -> str:
    unit = (
        "Prefix routing reuses the immutable system prompt, retrieved context, "
        "tool schema, and conversation history across related requests. "
    )
    return f"{label}\n" + unit * repeats


def _server_command(
    model_path: str,
    served_model_name: str,
    port: int,
    max_model_len: int,
    kv_events_config: dict[str, Any],
    extra_server_args: list[str],
) -> list[str]:
    return [
        _vllm_cli(),
        "serve",
        model_path,
        "--served-model-name",
        served_model_name,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--max-model-len",
        str(max_model_len),
        "--enable-prefix-caching",
        "--kv-events-config",
        json.dumps(kv_events_config, separators=(",", ":")),
        *extra_server_args,
    ]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-sha", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--served-model-name", default="prefix-e2e-model")
    parser.add_argument("--device-env-var", default="ASCEND_RT_VISIBLE_DEVICES")
    parser.add_argument("--master-device", default="0")
    parser.add_argument("--worker-device", default="1")
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--startup-timeout", type=float, default=600.0)
    parser.add_argument("--sync-timeout", type=float, default=60.0)
    parser.add_argument("--sync-settle-seconds", type=float, default=5.0)
    parser.add_argument("--stream-max-tokens", type=int, default=512)
    parser.add_argument("--result-dir", type=Path)
    parser.add_argument("--extra-server-args-json", default="[]")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    actual_sha = _git("rev-parse", "HEAD")
    if len(args.expected_sha) != 40 or actual_sha != args.expected_sha:
        raise SystemExit(
            f"expected exact Git SHA {args.expected_sha!r}, found {actual_sha}"
        )
    if _git("status", "--porcelain", "--untracked-files=normal"):
        raise SystemExit("fixed-SHA E2E requires a clean worktree")

    extra_server_args = json.loads(args.extra_server_args_json)
    if not isinstance(extra_server_args, list) or not all(
        isinstance(value, str) for value in extra_server_args
    ):
        raise SystemExit("--extra-server-args-json must be a JSON string list")
    if args.sync_settle_seconds <= 0:
        raise SystemExit("--sync-settle-seconds must be positive")
    if args.stream_max_tokens <= 0:
        raise SystemExit("--stream-max-tokens must be positive")

    result_dir = args.result_dir or Path("/tmp") / f"prefix-routing-e2e-{actual_sha}"
    result_dir.mkdir(parents=True, exist_ok=True)
    result_path = result_dir / "result.json"
    (
        master_port,
        worker_port,
        master_event_port,
        master_replay_port,
        worker_event_port,
        worker_replay_port,
    ) = _free_ports(6)
    master_url = f"http://127.0.0.1:{master_port}"
    worker_url = f"http://127.0.0.1:{worker_port}"
    routing_token = "prefix-routing-e2e-secret"

    common_env = os.environ.copy()
    common_env["PYTHONHASHSEED"] = "0"
    worker_env = common_env.copy()
    worker_env[args.device_env_var] = args.worker_device
    worker_env["PREFIX_ROUTING_E2E_NODE_ID"] = "node1"
    master_env = common_env.copy()
    master_env[args.device_env_var] = args.master_device

    worker_kv_config = {
        "enable_kv_cache_events": True,
        "publisher": "zmq",
        "endpoint": f"tcp://*:{worker_event_port}",
        "replay_endpoint": f"tcp://*:{worker_replay_port}",
    }
    master_kv_config = {
        "enable_kv_cache_events": True,
        "publisher": "zmq",
        "endpoint": f"tcp://*:{master_event_port}",
        "replay_endpoint": f"tcp://*:{master_replay_port}",
    }
    routing_config = {
        "nodes": [
            {
                "id": "node0",
                "url": "local",
                "local": True,
                "event_endpoint": f"tcp://127.0.0.1:{master_event_port}",
                "replay_endpoint": f"tcp://127.0.0.1:{master_replay_port}",
                "data_parallel_rank": 0,
            },
            {
                "id": "node1",
                "url": worker_url,
                "event_endpoint": f"tcp://127.0.0.1:{worker_event_port}",
                "replay_endpoint": f"tcp://127.0.0.1:{worker_replay_port}",
                "data_parallel_rank": 0,
                "routing_token": routing_token,
            },
        ],
        "routing_token": routing_token,
        "event_sync_interval": 1.0,
        "event_replay_timeout": 2.0,
        "request_timeout": 120.0,
    }

    worker_command = _server_command(
        args.model,
        args.served_model_name,
        worker_port,
        args.max_model_len,
        worker_kv_config,
        ["--middleware", IDENTITY_MIDDLEWARE, *extra_server_args],
    )
    master_command = _server_command(
        args.model,
        args.served_model_name,
        master_port,
        args.max_model_len,
        master_kv_config,
        [
            "--enable-prefix-routing",
            "--prefix-routing-config",
            json.dumps(routing_config, separators=(",", ":")),
            *extra_server_args,
        ],
    )
    worker = _ServerProcess(
        "node1", worker_command, worker_env, result_dir / "node1.log"
    )
    master = _ServerProcess(
        "node0", master_command, master_env, result_dir / "node0.log"
    )
    result: dict[str, Any] = {
        "git_sha": actual_sha,
        "pythonhashseed": "0",
        "model": args.model,
        "served_model_name": args.served_model_name,
        "device_env_var": args.device_env_var,
        "master_device": args.master_device,
        "worker_device": args.worker_device,
        "max_model_len": args.max_model_len,
        "stream_max_tokens": args.stream_max_tokens,
        "extra_server_args": extra_server_args,
        "started_at_unix": time.time(),
        "checks": {},
        "success": False,
    }

    try:
        worker.start()
        _wait_for_health(worker, worker_url, args.startup_timeout)
        master.start()
        _wait_for_health(master, master_url, args.startup_timeout)

        local_prompt = _make_prompt("local-only-prefix", 48)
        status, headers, _ = _completion(
            master_url,
            args.served_model_name,
            local_prompt + " local warmup",
            headers={"x-vllm-prefix-routing": routing_token},
        )
        if status != 200:
            raise RuntimeError("local warmup failed")
        status, headers, _ = _completion(
            master_url,
            args.served_model_name,
            local_prompt + " local routed request",
        )
        if status != 200 or "x-prefix-routing-e2e-node" in headers:
            raise RuntimeError("local prefix was not served by node0")
        result["checks"]["local_route"] = "passed"

        remote_prompt = _make_prompt("remote-long-prefix", 96)
        _completion(
            worker_url,
            args.served_model_name,
            remote_prompt + " worker warmup",
        )
        headers = _wait_for_worker_route(
            master_url,
            args.served_model_name,
            remote_prompt + " routed request",
            args.sync_timeout,
        )
        if headers.get("x-prefix-routing-e2e-rank") != "0":
            raise RuntimeError("forged DP rank was not replaced by rank 0")
        result["checks"]["multi_node_route_and_rank"] = "passed"

        failure_prompt = _make_prompt("upstream-failure-prefix", 64)
        _completion(
            worker_url,
            args.served_model_name,
            failure_prompt + " worker warmup",
        )
        time.sleep(args.sync_settle_seconds)
        master_log_offset = master.log_path.stat().st_size
        worker.stop()
        status, headers, _ = _completion(
            master_url,
            args.served_model_name,
            failure_prompt + " upstream failure",
        )
        if status != 200 or "x-prefix-routing-e2e-node" in headers:
            raise RuntimeError(
                "pre-response upstream failure did not fall back locally"
            )
        _wait_for_log(
            master.log_path,
            "Prefix routing upstream node1 failed before responding",
            master_log_offset,
            10.0,
        )
        result["checks"]["upstream_failure"] = "passed"

        worker = _ServerProcess(
            "node1-restarted",
            worker_command,
            worker_env,
            result_dir / "node1-restarted.log",
        )
        worker.start()
        _wait_for_health(worker, worker_url, args.startup_timeout)
        stream_prompt = _make_prompt("stream-failure-prefix", 48)
        _completion(
            worker_url,
            args.served_model_name,
            stream_prompt + " worker warmup",
        )
        time.sleep(args.sync_settle_seconds)
        _stream_then_stop_worker(
            master_port,
            args.served_model_name,
            stream_prompt + " interrupted stream",
            worker,
            args.stream_max_tokens,
        )
        _wait_for_health(master, master_url, 30.0)
        result["checks"]["streaming_failure"] = "passed"
        result["success"] = True
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        worker.stop()
        master.stop()
        result["finished_at_unix"] = time.time()
        result["duration_seconds"] = (
            result["finished_at_unix"] - result["started_at_unix"]
        )
        result_path.write_text(
            json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
        )
        print(f"prefix routing E2E result: {result_path}")


if __name__ == "__main__":
    main()
