# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Tests for /weight_info and /update_weight_label endpoints.

weight_gen is a monotonic integer that auto-increments on every
/finish_weight_update call. weight_label is a client-set string for
human identification (e.g. "step-1500").

Design improvements over sglang's weight_version:
  - weight_gen is INTERNAL (auto-incremented), not client-supplied → can't be
    forgotten, can't go backwards, enables staleness arithmetic
  - weight_label is EXTERNAL (client-set), decoupled from ordering
  - both update AFTER engine confirms finish_weight_update, no TOCTOU window

These tests exercise the HTTP endpoints WITHOUT requiring init_weight_transfer_engine
(NCCL), because weight_gen/label state lives on the API router, not in the engine.

RFC: https://github.com/vllm-project/vllm/issues/45585
"""

import requests

from .conftest import gen, health, ok, server

_PORT_BASE = 8810


class TestWeightInfoEndpoint:
    """/weight_info returns current weight version state."""

    def test_initial_weight_gen_is_zero(self):
        """Fresh server starts at weight_gen=0, weight_label=''."""
        with server(port=_PORT_BASE, dummy_weights=True) as url:
            r = requests.get(f"{url}/weight_info", timeout=5)
            assert r.status_code == 200
            body = r.json()
            assert body["weight_gen"] == 0, f"initial gen must be 0, got {body}"
            assert body["weight_label"] == "", f"initial label must be empty, got {body}"

    def test_weight_info_schema(self):
        """Response must contain exactly weight_gen (int) and weight_label (str)."""
        with server(port=_PORT_BASE + 1, dummy_weights=True) as url:
            body = requests.get(f"{url}/weight_info", timeout=5).json()
            assert isinstance(body["weight_gen"], int)
            assert isinstance(body["weight_label"], str)
            assert set(body.keys()) == {"weight_gen", "weight_label"}


class TestUpdateWeightLabel:
    """/update_weight_label sets the label without bumping gen."""

    def test_set_label_preserves_gen(self):
        """Setting a label must not change weight_gen."""
        with server(port=_PORT_BASE + 2, dummy_weights=True) as url:
            before = requests.get(f"{url}/weight_info", timeout=5).json()
            assert before["weight_gen"] == 0

            r = requests.post(
                f"{url}/update_weight_label",
                json={"weight_label": "step-0"},
                timeout=5,
            )
            assert r.status_code == 200
            body = r.json()
            assert body["weight_gen"] == 0, "label update must not bump gen"
            assert body["weight_label"] == "step-0"

            # Confirm /weight_info reflects the change
            info = requests.get(f"{url}/weight_info", timeout=5).json()
            assert info["weight_label"] == "step-0"
            assert info["weight_gen"] == 0

    def test_label_can_be_overwritten(self):
        """Label can be set multiple times without changing gen."""
        with server(port=_PORT_BASE + 3, dummy_weights=True) as url:
            requests.post(
                f"{url}/update_weight_label",
                json={"weight_label": "first"},
                timeout=5,
            )
            requests.post(
                f"{url}/update_weight_label",
                json={"weight_label": "second"},
                timeout=5,
            )
            info = requests.get(f"{url}/weight_info", timeout=5).json()
            assert info["weight_label"] == "second"
            assert info["weight_gen"] == 0

    def test_missing_label_returns_400(self):
        """Omitting weight_label must return 400, not crash."""
        with server(port=_PORT_BASE + 4, dummy_weights=True) as url:
            r = requests.post(
                f"{url}/update_weight_label",
                json={},
                timeout=5,
            )
            assert r.status_code in (400, 422), (
                f"missing label must return 400/422, got {r.status_code}"
            )
            assert health(url) == 200


class TestWeightGenAutoIncrement:
    """weight_gen increments on finish_weight_update, not on label change.

    Since init_weight_transfer_engine (NCCL) is not available in single-process
    test containers, we cannot call start/finish_weight_update through the
    engine. We test the increment indirectly:

    1. Verify gen starts at 0
    2. Verify /update_weight_label does NOT bump gen
    3. The actual gen increment on finish_weight_update is tested in Phase 2.5
       when we have a real trainer (via vime container).

    For now, this class documents the contract without triggering NCCL.
    """

    def test_gen_starts_at_zero_and_label_does_not_bump(self):
        with server(port=_PORT_BASE + 5, dummy_weights=True) as url:
            info = requests.get(f"{url}/weight_info", timeout=5).json()
            assert info["weight_gen"] == 0

            # Set label 3 times — gen must stay 0
            for i in range(3):
                requests.post(
                    f"{url}/update_weight_label",
                    json={"weight_label": f"label-{i}"},
                    timeout=5,
                )

            info = requests.get(f"{url}/weight_info", timeout=5).json()
            assert info["weight_gen"] == 0, (
                "weight_gen must not increment on label changes"
            )
            assert info["weight_label"] == "label-2"

    def test_gen_is_monotonic_contract(self):
        """Document: weight_gen is an int, starts at 0, only increases.

        The full increment test requires finish_weight_update (NCCL path).
        Here we just verify the type invariant and initial value.
        """
        with server(port=_PORT_BASE + 6, dummy_weights=True) as url:
            info = requests.get(f"{url}/weight_info", timeout=5).json()
            assert isinstance(info["weight_gen"], int)
            assert info["weight_gen"] >= 0


class TestWeightInfoWithGeneration:
    """weight_info endpoint must not interfere with generation."""

    def test_weight_info_and_generate_coexist(self):
        """Calling /weight_info before/after generation must not affect output."""
        with server(port=_PORT_BASE + 7, dummy_weights=True) as url:
            info_before = requests.get(f"{url}/weight_info", timeout=5).json()
            resp = gen(url)
            assert ok(resp), "generate failed with weight_info active"
            info_after = requests.get(f"{url}/weight_info", timeout=5).json()
            assert info_before == info_after, (
                "weight_info changed after generation without weight update"
            )
