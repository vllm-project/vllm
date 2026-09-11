# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""`--grpc-services` against a real `vllm-rs serve` process.

`vllm serve` does not forward `--grpc-port` to the Rust frontend, so the server
is launched through the `vllm-rs` binary directly. Every request below is an
empty message, so the calls use raw bytes and need no generated stubs.
"""

import contextlib
import os
import signal
import subprocess
import time
from collections.abc import Generator
from dataclasses import dataclass
from pathlib import Path

import grpc
import pytest
import requests

import vllm
from vllm.utils.network_utils import get_open_port

MODEL_NAME = "Qwen/Qwen3-0.6B"
STARTUP_TIMEOUT_S = 480
SHUTDOWN_TIMEOUT_S = 60

# `GrpcService` values from rust/proto/control.proto.
INFERENCE = 1
CONTROL = 2
KV_TRANSFER = 3
RL_CONTROL = 4

SERVICE_NAMES = ("vllm.Inference", "vllm.Control", "vllm.KvTransfer", "vllm.RlControl")


def _vllm_rs() -> str:
    path = os.environ.get("VLLM_RUST_FRONTEND_PATH", "auto")
    if path.lower() in ("auto", "1", "true"):
        path = str(Path(vllm.__file__).parent / "vllm-rs")
    if not os.access(path, os.X_OK):
        pytest.skip(f"vllm-rs binary not found at {path}")
    return path


@dataclass
class VllmRs:
    proc: subprocess.Popen
    http_port: int
    grpc_port: int
    log: Path

    def log_tail(self, lines: int = 80) -> str:
        return "\n".join(self.log.read_text(errors="replace").splitlines()[-lines:])


def _launch(extra_args: list[str], log: Path) -> VllmRs:
    http_port, grpc_port = get_open_port(), get_open_port()
    cmd = [
        _vllm_rs(),
        "serve",
        MODEL_NAME,
        "--host",
        "127.0.0.1",
        "--port",
        str(http_port),
        "--grpc-port",
        str(grpc_port),
        "--dtype",
        "bfloat16",
        "--max-model-len",
        "2048",
        "--enforce-eager",
        *extra_args,
    ]
    with log.open("wb") as out:
        proc = subprocess.Popen(
            cmd, stdout=out, stderr=subprocess.STDOUT, start_new_session=True
        )
    return VllmRs(proc, http_port, grpc_port, log)


def _stop(server: VllmRs) -> None:
    if server.proc.poll() is not None:
        return
    with contextlib.suppress(ProcessLookupError):
        os.killpg(server.proc.pid, signal.SIGTERM)
    try:
        server.proc.wait(timeout=SHUTDOWN_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(server.proc.pid, signal.SIGKILL)
        server.proc.wait()


@contextlib.contextmanager
def serve(extra_args: list[str], log: Path) -> Generator[VllmRs]:
    server = _launch(extra_args, log)
    try:
        deadline = time.monotonic() + STARTUP_TIMEOUT_S
        while True:
            if server.proc.poll() is not None:
                pytest.fail(
                    f"vllm-rs exited with {server.proc.returncode} during startup:\n"
                    f"{server.log_tail()}"
                )
            with contextlib.suppress(requests.ConnectionError):
                if requests.get(
                    f"http://127.0.0.1:{server.http_port}/health", timeout=5
                ).ok:
                    break
            if time.monotonic() > deadline:
                pytest.fail(f"vllm-rs did not become healthy:\n{server.log_tail()}")
            time.sleep(1)
        yield server
    finally:
        _stop(server)


@contextlib.contextmanager
def channel(server: VllmRs) -> Generator[grpc.Channel]:
    with grpc.insecure_channel(f"127.0.0.1:{server.grpc_port}") as chan:
        grpc.channel_ready_future(chan).result(timeout=30)
        yield chan


def call(chan: grpc.Channel, path: str, request: bytes = b"") -> bytes:
    return chan.unary_unary(path)(request, timeout=30)


def call_error(chan: grpc.Channel, path: str, request: bytes = b"") -> grpc.RpcError:
    with pytest.raises(grpc.RpcError) as info:
        call(chan, path, request)
    return info.value


def _varint(data: bytes, pos: int) -> tuple[int, int]:
    value = shift = 0
    while True:
        byte = data[pos]
        pos += 1
        value |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return value, pos
        shift += 7


def _packed_enum_field(data: bytes, field_number: int) -> list[int]:
    """Decode one packed repeated enum field from a serialized message."""
    values: list[int] = []
    pos = 0
    while pos < len(data):
        key, pos = _varint(data, pos)
        number, wire_type = key >> 3, key & 0x7
        if wire_type == 0:
            value, pos = _varint(data, pos)
            if number == field_number:
                values.append(value)
        elif wire_type == 2:
            length, pos = _varint(data, pos)
            end = pos + length
            if number == field_number:
                while pos < end:
                    value, pos = _varint(data, pos)
                    values.append(value)
            pos = end
        elif wire_type == 1:
            pos += 8
        elif wire_type == 5:
            pos += 4
        else:
            raise ValueError(f"unexpected wire type {wire_type}")
    return values


def mounted_services(chan: grpc.Channel) -> list[int]:
    # ServerInfo.services is field 12.
    return _packed_enum_field(call(chan, "/vllm.Control/GetServerInfo"), 12)


def _health_request(service: str) -> bytes:
    # HealthCheckRequest { string service = 1; }
    name = service.encode()
    return b"\x0a" + bytes([len(name)]) + name


HEALTH_CHECK = "/grpc.health.v1.Health/Check"


SERVING = b"\x08\x01"


def test_default_mounts_every_service(tmp_path: Path) -> None:
    with serve([], tmp_path / "vllm-rs.log") as server, channel(server) as chan:
        assert mounted_services(chan) == [INFERENCE, CONTROL, KV_TRANSFER, RL_CONTROL]
        for service in SERVICE_NAMES:
            status = call(chan, HEALTH_CHECK, _health_request(service))
            assert status == SERVING, service

        assert call(chan, "/vllm.RlControl/IsPaused") == call(
            chan, "/vllm.Control/IsPaused"
        )
        assert call(chan, "/vllm.KvTransfer/GetKvEventSources") == call(
            chan, "/vllm.Control/GetKvEventSources"
        )


def test_explicit_list_unmounts_the_rest(tmp_path: Path) -> None:
    with (
        serve(
            ["--grpc-services", "inference,control"], tmp_path / "vllm-rs.log"
        ) as server,
        channel(server) as chan,
    ):
        assert mounted_services(chan) == [INFERENCE, CONTROL]
        call(chan, "/vllm.Control/GetModelInfo")

        for path in (
            "/vllm.RlControl/IsPaused",
            "/vllm.RlControl/PauseGeneration",
            "/vllm.KvTransfer/GetKvEventSources",
        ):
            assert call_error(chan, path).code() == grpc.StatusCode.UNIMPLEMENTED, path

        for path in ("/vllm.Control/IsPaused", "/vllm.Control/GetKvEventSources"):
            error = call_error(chan, path)
            assert error.code() == grpc.StatusCode.UNIMPLEMENTED, path
            assert "--grpc-services" in (error.details() or ""), path

        assert call(chan, HEALTH_CHECK, _health_request("vllm.Control")) == SERVING
        for service in ("vllm.KvTransfer", "vllm.RlControl"):
            error = call_error(chan, HEALTH_CHECK, _health_request(service))
            assert error.code() == grpc.StatusCode.NOT_FOUND, service


def test_configured_follows_engine_config(tmp_path: Path) -> None:
    args = ["--grpc-services", "configured", "--enable-sleep-mode"]
    with serve(args, tmp_path / "vllm-rs.log") as server, channel(server) as chan:
        assert mounted_services(chan) == [INFERENCE, CONTROL, RL_CONTROL]
        call(chan, "/vllm.RlControl/IsSleeping")
        error = call_error(chan, "/vllm.KvTransfer/GetKvEventSources")
        assert error.code() == grpc.StatusCode.UNIMPLEMENTED


def test_unconfigured_explicit_service_fails_startup(tmp_path: Path) -> None:
    server = _launch(["--grpc-services", "rl-control"], tmp_path / "vllm-rs.log")
    try:
        returncode = server.proc.wait(timeout=STARTUP_TIMEOUT_S)
    finally:
        _stop(server)
    assert returncode != 0, server.log_tail()
    assert "requested rl-control" in server.log.read_text(errors="replace"), (
        server.log_tail()
    )
