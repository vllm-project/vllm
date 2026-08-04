# SPDX-License-Identifier: Apache-2.0
"""Tests for the coordinator REST API via FastAPI TestClient."""

# Third Party
from fastapi.testclient import TestClient

# First Party
from lmcache.v1.mp_coordinator.app import create_app
from lmcache.v1.mp_coordinator.config import MPCoordinatorConfig


def _client() -> TestClient:
    # health_check_interval=0 disables the background loop for deterministic tests.
    config = MPCoordinatorConfig(health_check_interval=0.0)
    return TestClient(create_app(config))


def test_register_lists_then_deregister():
    with _client() as client:
        resp = client.post(
            "/instances",
            json={"instance_id": "i1", "ip": "127.0.0.1", "http_port": 8080},
        )
        assert resp.status_code == 200
        assert resp.json() == {"instance_id": "i1", "re_registered": False}

        listed = client.get("/instances").json()["instances"]
        assert [i["instance_id"] for i in listed] == ["i1"]

        assert client.delete("/instances/i1").status_code == 204
        assert client.get("/instances").json()["instances"] == []


def test_register_round_trips_p2p_and_mq_fields():
    with _client() as client:
        client.post(
            "/instances",
            json={
                "instance_id": "i1",
                "ip": "10.0.0.1",
                "http_port": 8080,
                "p2p_advertised_url": "10.0.0.1:7600",
                "mq_port": 5555,
            },
        )
        listed = client.get("/instances").json()["instances"]
        assert listed[0]["ip"] == "10.0.0.1"
        assert listed[0]["p2p_advertised_url"] == "10.0.0.1:7600"
        assert listed[0]["mq_port"] == 5555


def test_register_defaults_mq_port_when_absent():
    with _client() as client:
        client.post(
            "/instances",
            json={"instance_id": "i1", "ip": "127.0.0.1", "http_port": 8080},
        )
        listed = client.get("/instances").json()["instances"]
        assert listed[0]["mq_port"] == 0


def test_re_register_reports_true():
    with _client() as client:
        client.post(
            "/instances",
            json={"instance_id": "i1", "ip": "127.0.0.1", "http_port": 8080},
        )
        resp = client.post(
            "/instances",
            json={"instance_id": "i1", "ip": "127.0.0.1", "http_port": 9090},
        )
        assert resp.json()["re_registered"] is True


def test_heartbeat_known_and_unknown():
    with _client() as client:
        client.post(
            "/instances",
            json={"instance_id": "i1", "ip": "127.0.0.1", "http_port": 8080},
        )
        assert client.put("/instances/i1/heartbeat").status_code == 200
        # Unknown instance -> 404 so the client knows to re-register.
        assert client.put("/instances/ghost/heartbeat").status_code == 404


def test_deregister_is_idempotent():
    with _client() as client:
        assert client.delete("/instances/never-registered").status_code == 204


def test_register_rejects_bad_body():
    # The Pydantic model validates the body; FastAPI returns 422 on bad input.
    with _client() as client:
        # Missing ip / http_port.
        assert client.post("/instances", json={"instance_id": "x"}).status_code == 422
        # http_port not coercible to int.
        bad = {"instance_id": "x", "ip": "127.0.0.1", "http_port": "nope"}
        assert client.post("/instances", json=bad).status_code == 422


def test_register_generates_id_when_empty():
    with _client() as client:
        # Empty / omitted instance_id -> coordinator assigns one.
        resp = client.post("/instances", json={"ip": "127.0.0.1", "http_port": 8080})
        assert resp.status_code == 200
        assigned = resp.json()["instance_id"]
        assert assigned
        listed = client.get("/instances").json()["instances"]
        assert [i["instance_id"] for i in listed] == [assigned]


def test_register_rejects_blank_ip():
    with _client() as client:
        # Whitespace-only ip is stripped to empty and rejected.
        bad = {"instance_id": "x", "ip": "   ", "http_port": 8080}
        assert client.post("/instances", json=bad).status_code == 422


def test_whitespace_instance_id_is_generated():
    with _client() as client:
        # Whitespace-only id strips to empty -> coordinator assigns one.
        resp = client.post(
            "/instances",
            json={"instance_id": "  ", "ip": "127.0.0.1", "http_port": 8080},
        )
        assert resp.status_code == 200
        assert resp.json()["instance_id"].strip()


def test_p2p_advertised_url_round_trips():
    with _client() as client:
        client.post(
            "/instances",
            json={
                "instance_id": "i1",
                "ip": "127.0.0.1",
                "http_port": 8080,
                "p2p_advertised_url": "http://10.0.0.1:9100",
            },
        )
        listed = client.get("/instances").json()["instances"]
        assert listed[0]["p2p_advertised_url"] == "http://10.0.0.1:9100"


def test_p2p_advertised_url_defaults_to_empty():
    with _client() as client:
        # Omitting p2p_advertised_url leaves it empty (instance is not in P2P).
        client.post(
            "/instances",
            json={"instance_id": "i1", "ip": "127.0.0.1", "http_port": 8080},
        )
        listed = client.get("/instances").json()["instances"]
        assert listed[0]["p2p_advertised_url"] == ""


def test_healthz():
    with _client() as client:
        assert client.get("/healthz").json() == {"status": "healthy"}
