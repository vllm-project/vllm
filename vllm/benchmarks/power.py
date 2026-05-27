# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Power recording utilities for benchmarking."""

import os
import threading
from abc import ABC, abstractmethod

import numpy as np
import torch


class PowerRecorderMixin(ABC):
    """Abstract base class for power recorders."""

    def __init__(self, interval: float = 5.0):
        self.interval = interval
        self.accelerator_samples: list[float] = []
        self.cpu_samples: list[float] = []
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._accelerator_handles = None
        self._cpu_handles = None
        self.init_accelerator()

    @abstractmethod
    def init_accelerator(self) -> None:
        """Initialize accelerator handles into self._accelerator_handles."""

    @abstractmethod
    def _record_accelerator(self) -> float:
        """Sample and return total accelerator power in watts."""

    @abstractmethod
    def _record_cpu(self) -> float:
        """Sample and return total CPU power in watts."""

    def start(self) -> None:
        self._stop_event.clear()
        self.accelerator_samples = []
        self.cpu_samples = []
        self._thread = threading.Thread(target=self._record_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._thread is None:
            return
        self._stop_event.set()
        self._thread.join()
        self._thread = None

    def _record_loop(self) -> None:
        while not self._stop_event.is_set():
            if self._accelerator_handles:
                self.accelerator_samples.append(self._record_accelerator())
            if self._cpu_handles:
                self.cpu_samples.append(self._record_cpu())
            self._stop_event.wait(timeout=self.interval)

    def get_stats(self) -> dict:
        """Return a dict of power statistics collected since start()."""
        assert self._accelerator_handles is not None
        assert self._cpu_handles is not None

        if not self.accelerator_samples:
            raise RuntimeError("No accelerator power samples collected.")

        arr = np.array(self.accelerator_samples)
        num_devices = len(self._accelerator_handles)
        stats = {
            "p25_power_w": float(np.percentile(arr, 25)),
            "mean_power_w": float(np.mean(arr)),
            "median_power_w": float(np.median(arr)),
            "median_power_per_accelerator_w": float(np.median(arr) / num_devices),
            "p75_power_w": float(np.percentile(arr, 75)),
            "num_samples": len(self.accelerator_samples),
        }

        if self.cpu_samples and len(self._cpu_handles) > 0:
            cpu_arr = np.array(self.cpu_samples)
            num_sockets = len(self._cpu_handles)
            stats.update(
                {
                    "cpu_p25_power_w": float(np.percentile(cpu_arr, 25)),
                    "cpu_mean_power_w": float(np.mean(cpu_arr)),
                    "cpu_median_power_w": float(np.median(cpu_arr)),
                    "cpu_median_power_per_socket_w": float(
                        np.median(cpu_arr) / num_sockets
                    ),
                    "cpu_p75_power_w": float(np.percentile(cpu_arr, 75)),
                    "cpu_num_samples": len(self.cpu_samples),
                }
            )
        else:
            stats.update(
                {
                    "cpu_p25_power_w": None,
                    "cpu_mean_power_w": None,
                    "cpu_median_power_w": None,
                    "cpu_median_power_per_socket_w": None,
                    "cpu_p75_power_w": None,
                    "cpu_num_samples": None,
                }
            )
        return stats


class PowerRecorderAMD(PowerRecorderMixin):
    """Records GPU and CPU power usage in a background thread using amdsmi."""

    def init_accelerator(self) -> None:
        try:
            import amdsmi

            amdsmi.amdsmi_init(amdsmi.AmdSmiInitFlags.INIT_ALL_PROCESSORS)
        except Exception:
            amdsmi.amdsmi_init(amdsmi.AmdSmiInitFlags.INIT_AMD_GPUS)

        self.amdsmi = amdsmi

        all_handles = amdsmi.amdsmi_get_processor_handles()

        self.all_handles = all_handles

        # CUDA_VISIBLE_DEVICES order is not the same as amdsmi handles order,
        # even when setting `CUDA_DEVICE_ORDER=PCI_BUS_ID`,
        # thus we rely on the bus IDs to monitor the correct devices.
        cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if cuda_visible is None:
            raise ValueError(
                "The power measurement utility requires the environment variable "
                "`CUDA_VISIBLE_DEVICES` to be set in the "
                "`vllm bench serve` call to the same devices ids than the server."
            )
        assert torch.accelerator.device_count() > 0

        bus_id_to_handle = {}
        for handle in all_handles:
            # amdsmi_get_gpu_device_bdf returns e.g. "0000:e5:00.0";
            # the bus field is hex.
            bdf = amdsmi.amdsmi_get_gpu_device_bdf(handle)
            bus_id = int(bdf.split(":")[1], 16)
            bus_id_to_handle[bus_id] = handle

        # For each visible CUDA device, look up its PCI bus ID via torch,
        # and match it to amdsmi correct handles.
        self._accelerator_handles = []

        bus_ids = []
        for idx in range(torch.accelerator.device_count()):
            pci_bus_id = torch.cuda.get_device_properties(idx).pci_bus_id
            bus_ids.append(pci_bus_id)
            if pci_bus_id not in bus_id_to_handle:
                raise ValueError(
                    f"Could not find amdsmi handle for CUDA device {idx} "
                    f"(PCI bus ID {pci_bus_id}). "
                    f"Available bus IDs: {list(bus_id_to_handle.keys())}"
                )
            self._accelerator_handles.append(bus_id_to_handle[pci_bus_id])

        # Will be an empty list if amdsmi CPU dependencies are not met.
        # NOTE: This may also be empty if `amdsmi.amdsmi_shut_down()`
        # was called elsewhere earlier.
        self._cpu_handles = amdsmi.amdsmi_get_cpusocket_handles()

        if len(self._cpu_handles) == 0:
            print(
                "WARNING: could not find CPU handles for power recording, "
                "amdsmi dependencies are likely not met. Recording GPU power only."
            )

        print(
            f"Power recorder: monitoring {len(self._accelerator_handles)} GPU(s) "
            f"(CUDA_VISIBLE_DEVICES={cuda_visible!r}, bus IDs: {bus_ids}), and "
            f"{len(self._cpu_handles)} CPU sockets."
        )

    def _record_accelerator(self) -> float:
        """Sample total GPU power across all devices. Returns watts."""
        total_power = 0.0
        for handle in self._accelerator_handles:
            power_info = self.amdsmi.amdsmi_get_power_info(handle)
            total_power += power_info["socket_power"]
        return total_power

    def _record_cpu(self) -> float:
        """Sample total CPU socket power across all sockets. Returns watts."""
        total_power_mw = 0.0
        for handle in self._cpu_handles:
            # amdsmi_get_cpu_socket_power returns a string like "121650 mW"
            power_str = self.amdsmi.amdsmi_get_cpu_socket_power(handle)
            mw_value = float(power_str.split()[0])
            total_power_mw += mw_value
        return total_power_mw / 1000.0


def fill_and_print_power_summary(
    power_stats: dict, total_input: int, total_output: int, benchmark_duration: float
) -> dict:
    def _fmt(val) -> str:
        return f"{val:<10.2f}" if val is not None else "None"

    print("{s:{c}^{n}}".format(s="GPU Power (all devices)", n=50, c="-"))
    print(
        "{:<40} {:<10.2f}".format(
            "P25 Total Power (total W):", power_stats["p25_power_w"]
        )
    )
    print(
        "{:<40} {:<10.2f}".format(
            "Mean Total Power (total W):", power_stats["mean_power_w"]
        )
    )
    print(
        "{:<40} {:<10.2f}".format(
            "Median Total Power (total W):", power_stats["median_power_w"]
        )
    )
    print(
        "{:<40} {:<10.2f}".format(
            "Median Power (per-GPU W):",
            power_stats["median_power_per_accelerator_w"],
        )
    )
    print(
        "{:<40} {:<10.2f}".format(
            "P75 Total Power (total W):", power_stats["p75_power_w"]
        )
    )
    print("{:<40} {:<10}".format("Power samples:", power_stats["num_samples"]))

    print("{s:{c}^{n}}".format(s="CPU Power (all sockets)", n=50, c="-"))
    print(
        "{:<40} {}".format(
            "CPU P25 Total Power (W):", _fmt(power_stats["cpu_p25_power_w"])
        )
    )
    print(
        "{:<40} {}".format(
            "CPU Mean Total Power (W):", _fmt(power_stats["cpu_mean_power_w"])
        )
    )
    print(
        "{:<40} {}".format(
            "CPU Median Total Power (W):", _fmt(power_stats["cpu_median_power_w"])
        )
    )
    print(
        "{:<40} {}".format(
            "CPU Median Power (per-socket W):",
            _fmt(power_stats["cpu_median_power_per_socket_w"]),
        )
    )
    print(
        "{:<40} {}".format(
            "CPU P75 Total Power (W):", _fmt(power_stats["cpu_p75_power_w"])
        )
    )
    print(
        "{:<40} {}".format(
            "CPU Power samples:",
            (
                power_stats["cpu_num_samples"]
                if power_stats["cpu_num_samples"] is not None
                else "None"
            ),
        )
    )

    # Compute tokens/joule = (tokens/s) / watts = throughput / power
    median_power = power_stats["median_power_w"]

    input_throughput = total_input / benchmark_duration
    output_throughput = total_output / benchmark_duration
    total_throughput = (total_input + total_output) / benchmark_duration

    input_toks_per_joule = input_throughput / median_power
    output_toks_per_joule = output_throughput / median_power
    total_toks_per_joule = total_throughput / median_power
    power_stats["input_toks_per_joule"] = input_toks_per_joule
    power_stats["output_toks_per_joule"] = output_toks_per_joule
    power_stats["total_toks_per_joule"] = total_toks_per_joule

    if power_stats["cpu_median_power_w"] is not None:
        total_median_power = median_power + power_stats["cpu_median_power_w"]

        power_stats["input_toks_per_total_joule"] = (
            input_throughput / total_median_power
        )
        power_stats["output_toks_per_total_joule"] = (
            output_throughput / total_median_power
        )
        power_stats["total_toks_per_total_joule"] = (
            total_throughput / total_median_power
        )

    print("{s:{c}^{n}}".format(s="Power summary", n=50, c="-"))
    print(
        "{:<40} {:<10.4f}".format(
            "Input toks/joule (GPU):",
            power_stats["input_toks_per_joule"],
        )
    )
    print(
        "{:<40} {:<10.4f}".format(
            "Output toks/joule (GPU):",
            power_stats["output_toks_per_joule"],
        )
    )
    print(
        "{:<40} {:<10.4f}".format(
            "Total toks/joule (GPU):",
            power_stats["total_toks_per_joule"],
        )
    )

    if power_stats["cpu_median_power_w"] is not None:
        print(
            "{:<40} {:<10.4f}".format(
                "Input toks/joule (CPU + GPU):",
                power_stats["input_toks_per_total_joule"],
            )
        )
        print(
            "{:<40} {:<10.4f}".format(
                "Output toks/joule (CPU + GPU):",
                power_stats["output_toks_per_total_joule"],
            )
        )
        print(
            "{:<40} {:<10.4f}".format(
                "Total toks/joule (CPU + GPU):",
                power_stats["total_toks_per_total_joule"],
            )
        )

    return power_stats


def get_power_recorder(interval: float = 5.0) -> PowerRecorderMixin:
    """Return the appropriate PowerRecorder for the current hardware."""
    from vllm.platforms import current_platform

    if current_platform.is_rocm():
        return PowerRecorderAMD(interval=interval)
    raise NotImplementedError(
        "Power recording is only implemented on AMD GPUs."
    )
