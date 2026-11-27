# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""XPU hardware discovery and runtime policies for vLLM Recipes.

Discovery records process-visible facts; runtime policies are separate functions.
Import torch lazily so plain recipe conversion does not require an XPU runtime.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

from core.parallelism import GPUFallbackPolicy, select_gpu_fallback
from core.tuning import Policy, TuningResult, WorkloadHints, record_override
from core.validation import RecipeSkipError

from hardware.gpu import GPUDeviceInfo, GPUHardwareInfo, GPULayoutPolicy


@dataclass(frozen=True, slots=True)
class XPUHardwareInfo(GPUHardwareInfo):
    ze_affinity_mask: str | None


GPU_LAYOUT_POLICY = GPULayoutPolicy(
    device_label="XPU", title="Intel XPU", memory_label="B70"
)


def detect_xpu_hardware() -> XPUHardwareInfo:
    """Report visible XPU count, names and device memory through PyTorch.

    Indices are process-relative; respect Level Zero/PyTorch visibility rather
    than guessing physical GPU count from PCI inventory or xpu-smi.
    """
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "XPU detection needs PyTorch with Intel XPU support"
        ) from exc

    if not hasattr(torch, "xpu") or not torch.xpu.is_available():
        raise RuntimeError(
            "Intel XPU unavailable in this process/container; verify drivers, "
            "device passthrough and the XPU-enabled PyTorch installation"
        )

    devices = []
    for index in range(torch.xpu.device_count()):
        properties = torch.xpu.get_device_properties(index)
        # mem_get_info returns (free_bytes, total_bytes) for the selected device.
        try:
            free, total = torch.xpu.mem_get_info(index)
        except Exception as exc:
            raise RuntimeError(
                f"Cannot query XPU {index} memory; refusing to guess VRAM"
            ) from exc
        free, total = int(free), int(total)
        if total <= 0 or free < 0 or free > total:
            raise RuntimeError(
                f"XPU {index} reported invalid memory: free={free}, total={total}"
            )
        devices.append(
            GPUDeviceInfo(
                index=index,
                name=properties.name,
                total_memory_bytes=total,
                free_memory_bytes=free,
            )
        )

    if not devices:
        raise RuntimeError("No Intel XPU devices visible to this process")

    return XPUHardwareInfo(
        devices=tuple(devices),
        ze_affinity_mask=os.environ.get("ZE_AFFINITY_MASK"),
    )


def _validate_xpu_tensor_parallel_size(
    config: dict[str, Any],
    hardware: XPUHardwareInfo | None,
    workload: WorkloadHints,
    result: TuningResult,
) -> None:
    del workload
    if hardware is None:
        return
    tp = config.get("tensor-parallel-size", 1)
    pp = config.get("pipeline-parallel-size", 1)
    dp = config.get("data-parallel-size", 1)
    for name, value in (("TP", tp), ("PP", pp), ("DP", dp)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"Invalid {name} value in recipe: {value!r}")
    required_devices = tp * pp * dp
    if required_devices > hardware.gpu_count:
        raise RecipeSkipError(
            "INSUFFICIENT_XPU_DEVICES",
            f"B70 recipe requires at least {required_devices} visible XPU "
            f"devices (TP={tp}, PP={pp}, DP={dp}); detected "
            f"{hardware.gpu_count}. Refusing to change TP without a model "
            "memory-fit estimate.",
            required_devices=required_devices,
            available_devices=hardware.gpu_count,
            parallelism={"tp": tp, "pp": pp, "dp": dp},
        )
    result.notes.append(
        f"XPU devices={hardware.gpu_count}; retained recipe TP={tp}, PP={pp}, "
        f"DP={dp}. Device count alone does not establish memory fit or optimum TP"
    )


def _resolve_xpu_gpu_memory_utilization(
    config: dict[str, Any],
    hardware: XPUHardwareInfo | None,
    workload: WorkloadHints,
    result: TuningResult,
) -> None:
    del workload
    if hardware is None:
        return
    # A conservative initial capacity guard, not a model-fit guarantee.
    reserve_fraction = 0.05
    min_free_fraction = min(
        device.free_memory_bytes / device.total_memory_bytes
        for device in hardware.devices
    )
    limit = int((min_free_fraction - reserve_fraction) * 100) / 100
    if limit <= 0:
        raise ValueError("Insufficient free XPU memory after 5% reserve")

    recipe_value = config.get("gpu-memory-utilization", 0.90)
    if (
        isinstance(recipe_value, bool)
        or not isinstance(recipe_value, int | float)
        or not 0 < recipe_value <= 1
    ):
        raise ValueError(f"Invalid gpu-memory-utilization: {recipe_value!r}")
    candidate = min(float(recipe_value), limit)
    if candidate < float(recipe_value):
        record_override(
            result,
            "gpu-memory-utilization",
            round(candidate, 2),
            "reduced using minimum free/total VRAM across visible XPUs "
            "and a 5% reserve; model/KV-cache fit remains unverified",
        )
    else:
        result.notes.append(
            f"gpu-memory-utilization: retained recipe/default {recipe_value}; "
            "XPU free-memory check did not require a reduction"
        )


DEFAULT_POLICIES: tuple[Policy, ...] = (
    _validate_xpu_tensor_parallel_size,
    _resolve_xpu_gpu_memory_utilization,
)


# B70-specific limits and memory reservations remain explicit and opt-in.
FALLBACK_POLICY = GPUFallbackPolicy(
    device_label="XPU",
    supported_tp=GPU_LAYOUT_POLICY.supported_tp,
    supports_device=lambda name: "b70" in name.lower(),
    unsupported_device_message="Auto fit requires all visible GPUs to be Arc Pro B70",
    reserve_fraction=0.05,
    weight_multiplier=1.25,
    runtime_reserve_bytes=3 * 1024**3,
)


def fit_parallelism(
    config: dict[str, Any], hardware: XPUHardwareInfo
) -> dict[str, Any]:
    return select_gpu_fallback(config, hardware, policy=FALLBACK_POLICY)


def sweep_kwargs(hardware: XPUHardwareInfo) -> dict[str, Any]:
    return {"gpu_count": hardware.gpu_count, "gpu_layout_policy": GPU_LAYOUT_POLICY}


# A consistent detector name is used by the backend registry.
detect_hardware = detect_xpu_hardware


if __name__ == "__main__":
    print(json.dumps(detect_xpu_hardware().to_dict(), indent=2))
