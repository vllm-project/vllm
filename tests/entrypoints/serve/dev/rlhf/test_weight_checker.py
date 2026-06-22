# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Tests for POST /weight_checker — SHA-256 per-tensor weight snapshot/compare.

Endpoint actions:
  snapshot  → take fresh digests, store them in app.state.weight_checker
  compare   → diff current weights against stored snapshot
  checksum  → return per-tensor SHA-256 without storing (idempotent)
  reset     → clear stored snapshot

RFC: https://github.com/vllm-project/vllm/issues/45585
"""

import requests

from .conftest import gen, health, ok, server, sleep, wake

_PORT_BASE = 8830


def _wc(url: str, action: str, **kwargs) -> requests.Response:
    return requests.post(f"{url}/weight_checker", json={"action": action, **kwargs}, timeout=30)


class TestWeightCheckerSnapshot:
    """snapshot action returns structured response."""

    def test_snapshot_returns_n_tensors(self):
        with server(port=_PORT_BASE, dummy_weights=True) as url:
            r = _wc(url, "snapshot")
            assert r.status_code == 200, r.text
            body = r.json()
            assert body["status"] == "snapshotted"
            assert isinstance(body["n_tensors"], int)
            assert body["n_tensors"] > 0, f"expected > 0 tensors, got {body}"

    def test_snapshot_twice_overwrites(self):
        """Calling snapshot twice should succeed; second call updates the stored digest."""
        with server(port=_PORT_BASE + 1, dummy_weights=True) as url:
            r1 = _wc(url, "snapshot")
            assert r1.status_code == 200
            r2 = _wc(url, "snapshot")
            assert r2.status_code == 200
            # Both calls should report the same tensor count.
            assert r1.json()["n_tensors"] == r2.json()["n_tensors"]


class TestWeightCheckerCompare:
    """compare returns match=True when weights are unchanged."""

    def test_compare_without_snapshot_returns_400(self):
        """Comparing without a prior snapshot must return 400."""
        with server(port=_PORT_BASE + 2, dummy_weights=True) as url:
            r = _wc(url, "compare")
            assert r.status_code == 400, (
                f"expected 400 for compare without snapshot, got {r.status_code}: {r.text}"
            )
            assert health(url) == 200

    def test_compare_immediately_after_snapshot_matches(self):
        """Weights unchanged → compare.match must be True."""
        with server(port=_PORT_BASE + 3, dummy_weights=True) as url:
            assert _wc(url, "snapshot").status_code == 200

            r = _wc(url, "compare")
            assert r.status_code == 200, r.text
            body = r.json()
            assert body["match"] is True, f"expected match=True, got {body}"
            assert body["mismatches"] == [], f"unexpected mismatches: {body}"

    def test_compare_after_sleep_wake_still_matches(self):
        """sleep → wake does NOT change weights; compare must still pass."""
        with server(port=_PORT_BASE + 4, dummy_weights=True) as url:
            assert _wc(url, "snapshot").status_code == 200

            # conftest sleep/wake return the HTTP status code (int), not Response
            assert sleep(url) == 200
            assert wake(url) == 200

            r = _wc(url, "compare")
            assert r.status_code == 200, r.text
            assert r.json()["match"] is True, f"sleep/wake changed weights: {r.json()}"

    def test_compare_after_generate_still_matches(self):
        """Inference should not mutate weights."""
        with server(port=_PORT_BASE + 5, dummy_weights=True) as url:
            assert _wc(url, "snapshot").status_code == 200
            assert ok(gen(url))

            r = _wc(url, "compare")
            assert r.status_code == 200
            assert r.json()["match"] is True, f"generate changed weights: {r.json()}"


class TestWeightCheckerChecksum:
    """checksum returns per-tensor SHA-256 digests."""

    def test_checksum_returns_dict_of_hex_strings(self):
        with server(port=_PORT_BASE + 6, dummy_weights=True) as url:
            r = _wc(url, "checksum")
            assert r.status_code == 200, r.text
            body = r.json()
            assert "checksums" in body
            assert isinstance(body["checksums"], dict)
            assert len(body["checksums"]) > 0
            # Every value must be a 64-char hex string (SHA-256)
            for name, digest in body["checksums"].items():
                assert isinstance(digest, str), f"{name}: digest is not str"
                assert len(digest) == 64, f"{name}: digest len {len(digest)} != 64"

    def test_checksum_stable_across_calls(self):
        """Two consecutive checksum calls must return identical digests."""
        with server(port=_PORT_BASE + 7, dummy_weights=True) as url:
            r1 = _wc(url, "checksum").json()["checksums"]
            r2 = _wc(url, "checksum").json()["checksums"]
            assert r1 == r2, "checksum not stable: same weights, different digests"

    def test_checksum_does_not_affect_snapshot(self):
        """checksum must not overwrite an existing snapshot."""
        with server(port=_PORT_BASE + 8, dummy_weights=True) as url:
            snap = _wc(url, "snapshot")
            assert snap.status_code == 200

            # Call checksum — should NOT change stored snapshot
            _wc(url, "checksum")

            # compare should still match original snapshot
            r = _wc(url, "compare")
            assert r.json()["match"] is True, f"checksum clobbered snapshot: {r.json()}"


class TestWeightCheckerReset:
    """reset clears the stored snapshot."""

    def test_reset_clears_snapshot(self):
        with server(port=_PORT_BASE + 9, dummy_weights=True) as url:
            assert _wc(url, "snapshot").status_code == 200

            r = _wc(url, "reset")
            assert r.status_code == 200
            assert r.json()["status"] == "reset"

            # compare after reset must return 400 (no snapshot)
            r2 = _wc(url, "compare")
            assert r2.status_code == 400, (
                f"expected 400 after reset, got {r2.status_code}: {r2.text}"
            )

    def test_reset_is_idempotent(self):
        """Resetting when there is no snapshot must succeed silently."""
        with server(port=_PORT_BASE + 10, dummy_weights=True) as url:
            r = _wc(url, "reset")
            assert r.status_code == 200
            r2 = _wc(url, "reset")
            assert r2.status_code == 200


class TestWeightCheckerErrors:
    """Invalid requests are rejected cleanly."""

    def test_missing_action_returns_400(self):
        with server(port=_PORT_BASE + 11, dummy_weights=True) as url:
            r = requests.post(f"{url}/weight_checker", json={}, timeout=10)
            assert r.status_code in (400, 422), r.text
            assert health(url) == 200

    def test_unknown_action_returns_400(self):
        with server(port=_PORT_BASE + 12, dummy_weights=True) as url:
            r = _wc(url, "frobnicate")
            assert r.status_code in (400, 422), r.text
            assert health(url) == 200
