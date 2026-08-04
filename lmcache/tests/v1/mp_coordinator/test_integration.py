# SPDX-License-Identifier: Apache-2.0
"""End-to-end test: a real uvicorn-served coordinator driven over HTTP.

Exercises the REST API against a live server (real lifespan + sockets) the way
an mp server will: register, heartbeat, deregister, with health-check eviction.
"""

# Standard
import socket as _socket
import threading
import time

# Third Party
import requests
import uvicorn

# First Party
from lmcache.v1.mp_coordinator.app import create_app
from lmcache.v1.mp_coordinator.config import MPCoordinatorConfig


def _free_port() -> int:
    s = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _wait_until_up(base_url: str, timeout: float = 5.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if requests.get(f"{base_url}/healthz", timeout=0.5).status_code == 200:
                return
        except requests.RequestException:
            time.sleep(0.05)
    raise RuntimeError("coordinator did not come up")


def _wait_until_instances_empty(base_url: str, timeout: float = 3.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            instances = requests.get(f"{base_url}/instances", timeout=0.5).json()[
                "instances"
            ]
            if not instances:
                return True
        except requests.RequestException:
            # The server is shutting down test requests quickly; keep polling.
            pass
        time.sleep(0.1)
    return False


def _serve(config: MPCoordinatorConfig):
    """Start the coordinator in a background thread; return (server, thread)."""
    server = uvicorn.Server(
        uvicorn.Config(
            create_app(config), host=config.host, port=config.port, log_level="warning"
        )
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    return server, thread


def test_register_heartbeat_deregister_over_real_http():
    port = _free_port()
    base = f"http://127.0.0.1:{port}"
    config = MPCoordinatorConfig(host="127.0.0.1", port=port, health_check_interval=0.0)
    server, thread = _serve(config)
    try:
        _wait_until_up(base)
        body = {"instance_id": "i1", "ip": "127.0.0.1", "http_port": 9999}
        assert (
            requests.post(f"{base}/instances", json=body, timeout=2).status_code == 200
        )

        listed = requests.get(f"{base}/instances", timeout=2).json()["instances"]
        assert [i["instance_id"] for i in listed] == ["i1"]

        assert (
            requests.put(f"{base}/instances/i1/heartbeat", timeout=2).status_code == 200
        )
        assert requests.delete(f"{base}/instances/i1", timeout=2).status_code == 204
        assert requests.get(f"{base}/instances", timeout=2).json()["instances"] == []
    finally:
        server.should_exit = True
        thread.join(timeout=5.0)


def test_health_loop_evicts_stale_instance():
    port = _free_port()
    base = f"http://127.0.0.1:{port}"
    # Fast loop + short timeout so a non-heartbeating instance is evicted quickly.
    config = MPCoordinatorConfig(
        host="127.0.0.1",
        port=port,
        instance_timeout=0.6,
        health_check_interval=0.2,
    )
    server, thread = _serve(config)
    try:
        _wait_until_up(base)
        body = {"instance_id": "ghost", "ip": "127.0.0.1", "http_port": 9999}
        requests.post(f"{base}/instances", json=body, timeout=2)
        assert requests.get(f"{base}/instances", timeout=2).json()["instances"]

        # Never heartbeat -> the health loop evicts it within a couple seconds.
        assert _wait_until_instances_empty(base, timeout=3.0)
    finally:
        server.should_exit = True
        thread.join(timeout=5.0)
