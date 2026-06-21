# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Tests for the structured JSON response bodies of /sleep and /wake_up.

Before this change /sleep and /wake_up returned a bare ``Response(200)`` with
no body, inconsistent with /pause (which returns ``{"status": "paused"}``).
RL orchestrators had no machine-readable confirmation of what transition
happened or how long it took.

These tests assert the response contract:
  POST /sleep   -> {"status": "sleeping", "level": int, "elapsed_ms": float}
  POST /wake_up -> {"status": "awake"|"sleeping", "tags_woken": list|None,
                    "elapsed_ms": float}

The ``status`` of /wake_up reflects partial wakes: waking only ``weights``
leaves the engine sleeping (kv_cache still unmapped), so status is "sleeping"
until the remaining tags are woken.

RFC: https://github.com/vllm-project/vllm/issues/45585
"""

import requests

from .conftest import gen, health, ok, server


def _sleep_resp(url, level=1, mode="abort"):
    return requests.post(
        f"{url}/sleep", params={"level": level, "mode": mode}, timeout=20
    )


def _wake_resp(url, tags=None):
    params = {"tags": tags} if tags else {}
    return requests.post(f"{url}/wake_up", params=params, timeout=30)


_PORT_BASE = 8800


class TestSleepResponseSchema:
    """POST /sleep returns a structured JSON body."""

    def test_sleep_response_has_status_level_elapsed(self):
        with server(port=_PORT_BASE, dummy_weights=True) as url:
            r = _sleep_resp(url, level=1)
            assert r.status_code == 200
            body = r.json()
            assert body["status"] == "sleeping", body
            assert body["level"] == 1, body
            assert "elapsed_ms" in body
            assert isinstance(body["elapsed_ms"], (int, float))
            assert body["elapsed_ms"] >= 0.0

            # restore so the server shuts down cleanly
            _wake_resp(url)

    def test_sleep_level2_echoed_in_response(self):
        with server(port=_PORT_BASE + 1, dummy_weights=True) as url:
            r = _sleep_resp(url, level=2)
            assert r.status_code == 200
            body = r.json()
            assert body["status"] == "sleeping"
            assert body["level"] == 2, (
                f"response must echo the requested level, got {body}"
            )
            _wake_resp(url)


class TestWakeResponseSchema:
    """POST /wake_up returns a structured JSON body reflecting wake state."""

    def test_full_wake_reports_awake(self):
        with server(port=_PORT_BASE + 2, dummy_weights=True) as url:
            assert _sleep_resp(url, level=1).status_code == 200

            r = _wake_resp(url)  # no tags = wake all
            assert r.status_code == 200
            body = r.json()
            assert body["status"] == "awake", (
                f"full wake should report awake, got {body}"
            )
            assert body["tags_woken"] is None, body
            assert isinstance(body["elapsed_ms"], (int, float))
            assert body["elapsed_ms"] >= 0.0
            assert health(url) == 200

    def test_partial_wake_reports_still_sleeping(self):
        """Waking only ['weights'] must report status='sleeping'.

        kv_cache is still unmapped after a weights-only wake, so the engine
        remains sleeping. This lets orchestrators sequence staged wakes.
        """
        with server(port=_PORT_BASE + 3, dummy_weights=True) as url:
            assert _sleep_resp(url, level=1).status_code == 200

            r = _wake_resp(url, tags=["weights"])
            assert r.status_code == 200
            body = r.json()
            assert body["status"] == "sleeping", (
                f"partial (weights-only) wake should still report sleeping, got {body}"
            )
            assert body["tags_woken"] == ["weights"], body

            # finish the wake
            r2 = _wake_resp(url, tags=["kv_cache"])
            assert r2.json()["status"] == "awake", r2.json()
            assert health(url) == 200

    def test_response_schema_round_trip_generates(self):
        """After a structured sleep/wake round trip the engine still serves."""
        with server(port=_PORT_BASE + 4, dummy_weights=True) as url:
            assert _sleep_resp(url, level=1).json()["status"] == "sleeping"
            assert _wake_resp(url).json()["status"] == "awake"
            assert health(url) == 200
            # dummy weights produce garbage tokens but the request must succeed
            assert ok(gen(url)), "generate failed after structured sleep/wake"
