# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import time
from collections.abc import Iterator
from contextlib import contextmanager

from vllm.logger import init_logger
from vllm.utils.import_utils import import_pynvml

logger = init_logger(__name__)


class GpuEnergyMeter:
    """Measure whole-device energy on explicitly selected local NVML GPUs."""

    def __init__(self, device_ids: list[int]) -> None:
        if (
            not device_ids
            or any(device_id < 0 for device_id in device_ids)
            or len(set(device_ids)) != len(device_ids)
        ):
            raise ValueError("Energy GPU IDs must be distinct non-negative indices")
        self.device_ids = list(device_ids)
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
                handles = [nvml.nvmlDeviceGetHandleByIndex(i) for i in self.device_ids]
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
        return {
            "energy_gpu_ids": self.device_ids,
            "gpu_energy_j": self.energy_j,
            "gpu_energy_duration_s": self.duration_s,
            "gpu_avg_power_w": self.energy_j / self.duration_s,
            "gpu_energy_per_output_token_j": (
                self.energy_j / output_tokens if output_tokens > 0 else None
            ),
        }
