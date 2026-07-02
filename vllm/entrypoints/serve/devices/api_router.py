# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from fastapi import APIRouter, FastAPI, Request

from vllm.entrypoints.serve.devices.protocol import (
    ComputeCapability,
    DeviceInfo,
    DevicesResponse,
)

router = APIRouter()


def _build_devices_response(devices: list[dict]) -> DevicesResponse:
    device_list: list[DeviceInfo] = []
    for d in devices:
        cap = d.get("compute_capability")
        device_list.append(
            DeviceInfo(
                rank=d["rank"],
                name=d["name"],
                total_memory_bytes=d["total_memory_bytes"],
                compute_capability=(
                    ComputeCapability(major=cap["major"], minor=cap["minor"])
                    if cap is not None
                    else None
                ),
                num_compute_units=d["num_compute_units"],
            )
        )
    return DevicesResponse(devices=device_list)


@router.get("/v1/server/devices")
async def get_devices(raw_request: Request) -> DevicesResponse:
    devices: list[dict] | None = raw_request.app.state.devices
    if devices is None:
        return DevicesResponse(devices=[])
    return _build_devices_response(devices)


def attach_router(app: FastAPI) -> None:
    app.include_router(router)
