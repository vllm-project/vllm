# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import contextlib
import signal
import socket
import time

import psutil
import requests

from tests.v1.distributed.test_internal_lb_dp import APIOnlyServerManager
from vllm.platforms import current_platform
from vllm.utils.network_utils import get_open_ports_list


def test_python_api_only_children_exit_when_launcher_is_killed(monkeypatch):
    """Python API launcher death releases its endpoint and preserves remote engines."""
    monkeypatch.setenv("VLLM_USE_RUST_FRONTEND", "0")
    model_name = "hmellor/tiny-random-LlamaForCausalLM"
    server_args = [
        "--host",
        "127.0.0.1",
        "--dtype",
        "float32" if current_platform.is_cpu() else "bfloat16",
        "--enforce-eager",
        "--max-model-len",
        "128",
        "--max-num-seqs",
        "2",
        "--max-num-batched-tokens",
        "128",
        "--gpu-memory-utilization",
        "0.05",
        "--kv-cache-memory-bytes",
        str(64 * 1024 * 1024),
    ]

    def is_alive(process: psutil.Process) -> bool:
        try:
            # A zombie leader may precede its other threads releasing sockets.
            return process.is_running() and (
                process.status() != psutil.STATUS_ZOMBIE or process.num_threads() > 1
            )
        except psutil.NoSuchProcess:
            return False

    api_port, rpc_port = get_open_ports_list(2)
    with APIOnlyServerManager(
        model_name,
        dp_size=2,
        api_server_count=2,
        base_server_args=server_args,
        tp_size=1,
        api_port=api_port,
        rpc_port=rpc_port,
    ) as server_list:
        api_server = server_list[0][0]
        engines_server = server_list[1][0]
        api_children = psutil.Process(api_server.proc.pid).children(recursive=True)
        engines_launcher = psutil.Process(engines_server.proc.pid)
        engine_children = engines_launcher.children(recursive=True)
        engine_processes = [engines_launcher, *engine_children]
        try:
            with requests.get(api_server.url_for("v1/models"), timeout=10) as response:
                response.raise_for_status()
                assert [model["id"] for model in response.json()["data"]] == [
                    model_name
                ]
            assert len(api_children) >= 2
            assert len(engine_children) >= 2
            assert all(
                is_alive(process) and process.status() != psutil.STATUS_ZOMBIE
                for process in api_children + engine_processes
            )

            # Kill only the launcher; fixture teardown would hide orphaned children.
            deadline = time.monotonic() + 15
            api_server.proc.kill()
            assert api_server.proc.wait(timeout=5) == -signal.SIGKILL
            remaining = api_children
            while remaining:
                remaining = [process for process in remaining if is_alive(process)]
                if not remaining or time.monotonic() >= deadline:
                    break
                time.sleep(0.05)
            assert not remaining, (
                f"API-only children outlived their launcher: "
                f"{[process.pid for process in remaining]}"
            )
            with socket.socket() as released_listener:
                released_listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                released_listener.bind((api_server.host, api_server.port))
            assert all(
                is_alive(process) and process.status() != psutil.STATUS_ZOMBIE
                for process in engine_processes
            ), "Separate headless launcher or engines exited with the API launcher"
        finally:
            # Reaped launchers cannot enumerate orphaned children during teardown.
            # These handles retain process identity and belong only to this test.
            for process in api_children:
                with contextlib.suppress(psutil.NoSuchProcess):
                    process.kill()
            psutil.wait_procs(api_children, timeout=5)
