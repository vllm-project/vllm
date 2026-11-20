# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import time
from collections.abc import Iterator
from contextlib import contextmanager

from vllm.logger import init_logger
from vllm.utils.import_utils import import_pynvml

logger = init_logger(__name__)


def _resolve_physical_device_id(device_id: str) -> int:
    """Map a CUDA-visible ordinal or device UUID to its physical NVML index.

    Avoid importing the CUDA platform (and its compiled extension): energy
    counter tests must also run against a CPU-only vLLM installation.
    NVML is initialized by the caller when UUID resolution is needed.
    """
    try:
        visible_ordinal = int(device_id)
    except ValueError:
        control_id = device_id
    else:
        if visible_ordinal < 0:
            raise ValueError("Energy GPU ordinals must be non-negative")
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if not visible:
            return visible_ordinal
        entries = visible.split(",")
        if visible_ordinal >= len(entries):
            raise ValueError(
                f"Energy GPU ordinal {visible_ordinal} is out of range for "
                f"CUDA_VISIBLE_DEVICES={visible}"
            )
        control_id = entries[visible_ordinal]

    try:
        return int(control_id)
    except ValueError:
        nvml = import_pynvml()
        handle = nvml.nvmlDeviceGetHandleByUUID(control_id)
        return int(nvml.nvmlDeviceGetIndex(handle))


class GpuEnergyMeter:
    """Measure whole-device energy on explicitly selected local NVML GPUs."""

    def __init__(self, device_ids: list[str]) -> None:
        if not device_ids or any(not device_id for device_id in device_ids):
            raise ValueError("Energy GPU IDs must be non-empty")
        if len(set(device_ids)) != len(device_ids):
            raise ValueError("Energy GPU IDs must be distinct")
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        visible_count = len(visible.split(",")) if visible else None
        for device_id in device_ids:
            try:
                visible_device_id = int(device_id)
            except ValueError:
                continue
            if visible_device_id < 0:
                raise ValueError("Energy GPU ordinals must be non-negative")
            if visible_count is not None and visible_device_id >= visible_count:
                raise ValueError(
                    f"Energy GPU ordinal {visible_device_id} is out of range for "
                    f"CUDA_VISIBLE_DEVICES={visible}"
                )
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
