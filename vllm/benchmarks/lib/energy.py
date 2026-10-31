# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import time
from collections.abc import Iterator
from contextlib import contextmanager

from vllm.logger import init_logger
from vllm.utils.import_utils import import_pynvml

logger = init_logger(__name__)


def _resolve_physical_device_id(device_id: str) -> int:
    from vllm.platforms.cuda import NvmlCudaPlatform

    try:
        visible_device_id = int(device_id)
    except ValueError:
        return NvmlCudaPlatform.device_control_id_to_physical_device_id(device_id)
    if visible_device_id < 0:
        raise ValueError("Energy GPU ordinals must be non-negative")
    return NvmlCudaPlatform.visible_device_id_to_physical_device_id(visible_device_id)


class GpuEnergyMeter:
    """Measure whole-device energy on explicitly selected local NVML GPUs."""

    def __init__(self, device_ids: list[str]) -> None:
        if not device_ids or any(not device_id for device_id in device_ids):
            raise ValueError("Energy GPU IDs must be non-empty")
        if len(set(device_ids)) != len(device_ids):
            raise ValueError("Energy GPU IDs must be distinct")
        for device_id in device_ids:
            try:
                visible_device_id = int(device_id)
            except ValueError:
                continue
            if visible_device_id < 0:
                raise ValueError("Energy GPU ordinals must be non-negative")
        self.device_ids = list(device_ids)
        self.physical_device_ids: list[int] | None = None
        self.energy_j: float | None = None
        self.duration_s = 0.0

    @contextmanager
    def measure(self) -> Iterator[None]:
        self.energy_j = None
        self.duration_s = 0.0
        nvml = import_pynvml()
        initialized = False
        try:
            try:
                nvml.nvmlInit()
                initialized = True
                self.physical_device_ids = [
                    _resolve_physical_device_id(device_id)
                    for device_id in self.device_ids
                ]
                if len(set(self.physical_device_ids)) != len(self.physical_device_ids):
                    raise ValueError(
                        "Energy GPU IDs must resolve to distinct physical devices"
                    )
                handles = [
                    nvml.nvmlDeviceGetHandleByIndex(i) for i in self.physical_device_ids
                ]
                initial = [nvml.nvmlDeviceGetTotalEnergyConsumption(h) for h in handles]
            except nvml.NVMLError as exc:
                logger.warning("GPU energy measurement unavailable: %s", exc)
                yield
                return

            start = time.perf_counter()
            try:
                yield
            finally:
                self.duration_s = time.perf_counter() - start
                try:
                    final = [
                        nvml.nvmlDeviceGetTotalEnergyConsumption(h) for h in handles
                    ]
                    deltas = [end - begin for begin, end in zip(initial, final)]
                    if any(delta < 0 for delta in deltas):
                        logger.warning(
                            "GPU energy counter reset; omitting energy metrics"
                        )
                    elif self.duration_s > 0:
                        self.energy_j = sum(deltas) / 1000.0
                except nvml.NVMLError as exc:
                    logger.warning("GPU energy measurement unavailable: %s", exc)
        finally:
            if initialized:
                try:
                    nvml.nvmlShutdown()
                except nvml.NVMLError as exc:
                    logger.warning(
                        "Could not shut down GPU energy measurement: %s", exc
                    )

    def get_results(self, output_tokens: int) -> dict[str, float | list[int] | None]:
        if self.energy_j is None:
            return {}
        assert self.physical_device_ids is not None
        return {
            "energy_gpu_ids": self.physical_device_ids,
            "gpu_energy_j": self.energy_j,
            "gpu_energy_duration_s": self.duration_s,
            "gpu_avg_power_w": self.energy_j / self.duration_s,
            "gpu_energy_per_output_token_j": (
                self.energy_j / output_tokens if output_tokens > 0 else None
            ),
        }
