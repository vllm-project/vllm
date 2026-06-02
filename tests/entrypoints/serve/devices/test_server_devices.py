# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for GET /v1/server/devices.

These tests do not spin up a real vLLM server or load model weights.
They exercise:
  - _build_devices_response() directly (pure function, fast, no I/O)
  - The FastAPI route via TestClient (HTTP stack, no engine)
  - The Pydantic protocol models (schema correctness)
"""

from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from vllm.entrypoints.serve.devices.api_router import (
    _build_devices_response,
    attach_router,
    router,
)
from vllm.entrypoints.serve.devices.protocol import (
    ComputeCapability,
    DeviceInfo,
    DevicesResponse,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_A100_ENTRY = {
    "rank": 0,
    "name": "A100-PCIE-40GB",
    "total_memory_bytes": 42_949_672_960,
    "compute_capability": {"major": 8, "minor": 0},
    "num_compute_units": 108,
}

_H100_ENTRY = {
    "rank": 1,
    "name": "H100-SXM5-80GB",
    "total_memory_bytes": 85_899_345_920,
    "compute_capability": {"major": 9, "minor": 0},
    "num_compute_units": 132,
}

_NON_CUDA_ENTRY = {
    "rank": 0,
    "name": "SomeCPUOrNonCUDADevice",
    "total_memory_bytes": 8_589_934_592,
    "compute_capability": None,
    "num_compute_units": 64,
}


def _make_app(devices: list[dict] | None) -> FastAPI:
    """Build a minimal FastAPI app with the devices router and fixed state."""
    app = FastAPI()
    app.state.devices = devices
    attach_router(app)
    return app


def _make_client(devices: list[dict] | None) -> TestClient:
    """Return a synchronous TestClient wrapping a minimal devices app."""
    return TestClient(_make_app(devices))


# ---------------------------------------------------------------------------
# Protocol model tests
# ---------------------------------------------------------------------------


class TestComputeCapability:
    def test_basic(self):
        cap = ComputeCapability(major=8, minor=0)
        assert cap.major == 8
        assert cap.minor == 0

    def test_roundtrip(self):
        cap = ComputeCapability(major=9, minor=0)
        data = cap.model_dump()
        assert data == {"major": 9, "minor": 0}
        restored = ComputeCapability.model_validate(data)
        assert restored == cap

    @pytest.mark.parametrize(
        "major,minor",
        [
            (7, 0),
            (8, 0),
            (8, 6),
            (9, 0),
        ],
    )
    def test_major_minor_values(self, major, minor):
        cap = ComputeCapability(major=major, minor=minor)
        assert cap.major == major
        assert cap.minor == minor


class TestDeviceInfo:
    def test_with_compute_capability(self):
        device = DeviceInfo(
            rank=0,
            name="A100-PCIE-40GB",
            total_memory_bytes=42_949_672_960,
            compute_capability=ComputeCapability(major=8, minor=0),
            num_compute_units=108,
        )
        assert device.rank == 0
        assert device.name == "A100-PCIE-40GB"
        assert device.total_memory_bytes == 42_949_672_960
        assert device.compute_capability is not None
        assert device.compute_capability.major == 8
        assert device.compute_capability.minor == 0
        assert device.num_compute_units == 108

    def test_compute_capability_none(self):
        """Non-CUDA platforms return null compute_capability."""
        device = DeviceInfo(
            rank=0,
            name="SomeNonCUDADevice",
            total_memory_bytes=8_589_934_592,
            compute_capability=None,
            num_compute_units=64,
        )
        assert device.compute_capability is None

    def test_compute_capability_defaults_to_none(self):
        """compute_capability field defaults to None when omitted."""
        device = DeviceInfo(
            rank=1,
            name="TestDevice",
            total_memory_bytes=1024,
            num_compute_units=4,
        )
        assert device.compute_capability is None

    def test_roundtrip(self):
        original = DeviceInfo(
            rank=2,
            name="H100-SXM5-80GB",
            total_memory_bytes=85_899_345_920,
            compute_capability=ComputeCapability(major=9, minor=0),
            num_compute_units=132,
        )
        data = original.model_dump()
        restored = DeviceInfo.model_validate(data)
        assert restored == original

    def test_serializes_nested_capability(self):
        device = DeviceInfo(
            rank=0,
            name="GPU",
            total_memory_bytes=4096,
            compute_capability=ComputeCapability(major=7, minor=5),
            num_compute_units=72,
        )
        data = device.model_dump()
        assert data["compute_capability"] == {"major": 7, "minor": 5}

    def test_serializes_null_capability(self):
        device = DeviceInfo(
            rank=0,
            name="GPU",
            total_memory_bytes=4096,
            compute_capability=None,
            num_compute_units=8,
        )
        data = device.model_dump()
        assert data["compute_capability"] is None

    @pytest.mark.parametrize("rank", [0, 1, 255])
    def test_rank_values(self, rank):
        device = DeviceInfo(
            rank=rank,
            name="GPU",
            total_memory_bytes=1024,
            num_compute_units=1,
        )
        assert device.rank == rank


class TestDevicesResponse:
    def test_empty(self):
        response = DevicesResponse(devices=[])
        assert response.devices == []

    def test_single_device(self):
        device = DeviceInfo(
            rank=0,
            name="A100-PCIE-40GB",
            total_memory_bytes=42_949_672_960,
            compute_capability=ComputeCapability(major=8, minor=0),
            num_compute_units=108,
        )
        response = DevicesResponse(devices=[device])
        assert len(response.devices) == 1
        assert response.devices[0] == device

    def test_multi_rank(self):
        """Two-rank topology: both devices fully populated."""
        devices = [
            DeviceInfo(
                rank=0,
                name="A100-PCIE-40GB",
                total_memory_bytes=42_949_672_960,
                compute_capability=ComputeCapability(major=8, minor=0),
                num_compute_units=108,
            ),
            DeviceInfo(
                rank=1,
                name="A100-PCIE-40GB",
                total_memory_bytes=42_949_672_960,
                compute_capability=ComputeCapability(major=8, minor=0),
                num_compute_units=108,
            ),
        ]
        response = DevicesResponse(devices=devices)
        assert len(response.devices) == 2
        assert response.devices[0].rank == 0
        assert response.devices[1].rank == 1

    def test_mixed_capability(self):
        """Heterogeneous cluster: one CUDA device, one non-CUDA device."""
        devices = [
            DeviceInfo(
                rank=0,
                name="A100",
                total_memory_bytes=42_949_672_960,
                compute_capability=ComputeCapability(major=8, minor=0),
                num_compute_units=108,
            ),
            DeviceInfo(
                rank=1,
                name="NonCUDADevice",
                total_memory_bytes=8_589_934_592,
                compute_capability=None,
                num_compute_units=64,
            ),
        ]
        response = DevicesResponse(devices=devices)
        assert response.devices[0].compute_capability is not None
        assert response.devices[1].compute_capability is None

    def test_json_schema_shape(self):
        """Serialized shape matches the design doc schema."""
        response = DevicesResponse(
            devices=[
                DeviceInfo(
                    rank=0,
                    name="A100-PCIE-40GB",
                    total_memory_bytes=42_949_672_960,
                    compute_capability=ComputeCapability(major=8, minor=0),
                    num_compute_units=108,
                )
            ]
        )
        data = response.model_dump()
        assert "devices" in data
        assert isinstance(data["devices"], list)
        entry = data["devices"][0]
        assert set(entry.keys()) == {
            "rank",
            "name",
            "total_memory_bytes",
            "compute_capability",
            "num_compute_units",
        }
        assert entry["compute_capability"] == {"major": 8, "minor": 0}

    def test_roundtrip(self):
        original = DevicesResponse(
            devices=[
                DeviceInfo(
                    rank=0,
                    name="H100",
                    total_memory_bytes=85_899_345_920,
                    compute_capability=ComputeCapability(major=9, minor=0),
                    num_compute_units=132,
                ),
                DeviceInfo(
                    rank=1,
                    name="H100",
                    total_memory_bytes=85_899_345_920,
                    compute_capability=ComputeCapability(major=9, minor=0),
                    num_compute_units=132,
                ),
            ]
        )
        data = original.model_dump()
        restored = DevicesResponse.model_validate(data)
        assert restored == original


# ---------------------------------------------------------------------------
# _build_devices_response unit tests
# ---------------------------------------------------------------------------


class TestBuildDevicesResponse:
    def test_single(self):
        result = _build_devices_response([_A100_ENTRY])
        assert isinstance(result, DevicesResponse)
        assert len(result.devices) == 1
        d = result.devices[0]
        assert d.rank == 0
        assert d.name == "A100-PCIE-40GB"
        assert d.total_memory_bytes == 42_949_672_960
        assert d.compute_capability == ComputeCapability(major=8, minor=0)
        assert d.num_compute_units == 108

    def test_multi(self):
        result = _build_devices_response([_A100_ENTRY, _H100_ENTRY])
        assert len(result.devices) == 2
        assert result.devices[0].rank == 0
        assert result.devices[1].rank == 1
        assert result.devices[1].name == "H100-SXM5-80GB"
        assert result.devices[1].compute_capability == ComputeCapability(
            major=9, minor=0
        )

    def test_null_capability(self):
        result = _build_devices_response([_NON_CUDA_ENTRY])
        assert len(result.devices) == 1
        assert result.devices[0].compute_capability is None

    def test_empty_list(self):
        result = _build_devices_response([])
        assert result.devices == []

    def test_preserves_input_order(self):
        entries = [
            {**_A100_ENTRY, "rank": 3},
            {**_A100_ENTRY, "rank": 1},
            {**_A100_ENTRY, "rank": 0},
        ]
        result = _build_devices_response(entries)
        assert [d.rank for d in result.devices] == [3, 1, 0]

    def test_mixed_capability(self):
        result = _build_devices_response([_A100_ENTRY, _NON_CUDA_ENTRY])
        assert result.devices[0].compute_capability is not None
        assert result.devices[1].compute_capability is None

    @pytest.mark.parametrize(
        "entry,expected_name,expected_major",
        [
            (_A100_ENTRY, "A100-PCIE-40GB", 8),
            (_H100_ENTRY, "H100-SXM5-80GB", 9),
        ],
    )
    def test_gpu_name_and_compute_capability(
        self, entry, expected_name, expected_major
    ):
        result = _build_devices_response([entry])
        d = result.devices[0]
        assert d.name == expected_name
        assert d.compute_capability is not None
        assert d.compute_capability.major == expected_major


# ---------------------------------------------------------------------------
# HTTP endpoint tests via TestClient
# ---------------------------------------------------------------------------


class TestGetDevicesEndpoint:
    def test_status_200(self):
        with _make_client([_A100_ENTRY, _H100_ENTRY]) as client:
            response = client.get("/v1/server/devices")
        assert response.status_code == 200

    def test_response_shape(self):
        with _make_client([_A100_ENTRY, _H100_ENTRY]) as client:
            data = client.get("/v1/server/devices").json()
        assert "devices" in data
        assert isinstance(data["devices"], list)
        assert len(data["devices"]) == 2

    def test_content_type_is_json(self):
        with _make_client([_A100_ENTRY]) as client:
            response = client.get("/v1/server/devices")
        assert "application/json" in response.headers["content-type"]

    @pytest.mark.parametrize(
        "entry,expected",
        [
            (
                _A100_ENTRY,
                {
                    "rank": 0,
                    "name": "A100-PCIE-40GB",
                    "total_memory_bytes": 42_949_672_960,
                    "compute_capability": {"major": 8, "minor": 0},
                    "num_compute_units": 108,
                },
            ),
            (
                _H100_ENTRY,
                {
                    "rank": 1,
                    "name": "H100-SXM5-80GB",
                    "total_memory_bytes": 85_899_345_920,
                    "compute_capability": {"major": 9, "minor": 0},
                    "num_compute_units": 132,
                },
            ),
        ],
    )
    def test_entry_fields(self, entry, expected):
        with _make_client([entry]) as client:
            data = client.get("/v1/server/devices").json()
        assert data["devices"][0] == expected

    def test_non_cuda_null_capability(self):
        with _make_client([_NON_CUDA_ENTRY]) as client:
            data = client.get("/v1/server/devices").json()
        assert data["devices"][0]["compute_capability"] is None

    def test_none_state_returns_empty(self):
        """Render-only servers set state.devices = None; endpoint returns []."""
        with _make_client(None) as client:
            data = client.get("/v1/server/devices").json()
        assert data == {"devices": []}

    def test_empty_list_state(self):
        """state.devices = [] (no workers) returns an empty devices list."""
        with _make_client([]) as client:
            response = client.get("/v1/server/devices")
        assert response.status_code == 200
        assert response.json() == {"devices": []}

    def test_eight_rank_tp(self):
        """8-rank tensor-parallel topology: one entry per rank."""
        entries = [
            {
                "rank": i,
                "name": "A100-PCIE-40GB",
                "total_memory_bytes": 42_949_672_960,
                "compute_capability": {"major": 8, "minor": 0},
                "num_compute_units": 108,
            }
            for i in range(8)
        ]
        with _make_client(entries) as client:
            response = client.get("/v1/server/devices")
        assert response.status_code == 200
        devices = response.json()["devices"]
        assert len(devices) == 8
        for i, device in enumerate(devices):
            assert device["rank"] == i

    def test_post_not_allowed(self):
        """POST to a GET-only endpoint must return 405."""
        with _make_client([]) as client:
            response = client.post("/v1/server/devices", json={})
        assert response.status_code == 405


# ---------------------------------------------------------------------------
# Router registration
# ---------------------------------------------------------------------------


class TestRouterRegistration:
    def test_attach_router_registers_route(self):
        app = FastAPI()
        app.state.devices = []
        attach_router(app)
        routes = {r.path for r in app.routes}  # type: ignore[attr-defined]
        assert "/v1/server/devices" in routes

    def test_router_has_get_method(self):
        assert any(
            r.path == "/v1/server/devices"  # type: ignore[attr-defined]
            and "GET" in r.methods  # type: ignore[attr-defined]
            for r in router.routes
        )


# ---------------------------------------------------------------------------
# Handler isolation tests (mock request)
# ---------------------------------------------------------------------------


class TestHandlerIsolation:
    @pytest.mark.asyncio
    async def test_with_device_state(self):
        """Call the handler function directly using a fully mocked request."""
        from vllm.entrypoints.serve.devices.api_router import get_devices

        mock_request = MagicMock()
        mock_request.app.state.devices = [_A100_ENTRY]

        result = await get_devices(mock_request)

        assert isinstance(result, DevicesResponse)
        assert len(result.devices) == 1
        assert result.devices[0].rank == 0

    @pytest.mark.asyncio
    async def test_with_none_state(self):
        """Handler returns empty DevicesResponse when state.devices is None."""
        from vllm.entrypoints.serve.devices.api_router import get_devices

        mock_request = MagicMock()
        mock_request.app.state.devices = None

        result = await get_devices(mock_request)

        assert isinstance(result, DevicesResponse)
        assert result.devices == []
