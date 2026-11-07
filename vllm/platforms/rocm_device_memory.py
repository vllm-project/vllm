# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Free device memory corrected by amdgpu's exact VRAM count.

hipMemGetInfo counts each allocation as at least 2 MB, so small allocations
hide free memory; amdgpu's sysfs mem_info_vram_used counts exact sizes.
"""

import os
import time
from dataclasses import dataclass

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

KFD_TOPOLOGY_NODES = "/sys/class/kfd/kfd/topology/nodes"
DRM_SYSFS = "/sys/class/drm"
# sysfs VRAM total may exceed hipMemGetInfo's total by at most this share.
VRAM_TOTAL_TOLERANCE = 0.01


def _kfd_gpu_nodes(nodes_dir: str) -> list[dict[str, str]]:
    """Properties of the KFD GPU nodes; CPU and malformed nodes are skipped."""
    nodes = []
    for node in sorted(os.listdir(nodes_dir)):
        try:
            # One "name value" pair per line.
            with open(os.path.join(nodes_dir, node, "properties")) as f:
                props = dict(line.split() for line in f if line.strip())
            if int(props.get("simd_count", 0)) > 0 and "drm_render_minor" in props:
                int(props["location_id"])
                int(props.get("unique_id", -1))
                nodes.append(props)
        except (OSError, KeyError, ValueError):
            continue
    return nodes


def _pci(props: dict[str, str]) -> tuple[int, int, int]:
    # location_id is (bus << 8) | (dev << 3) | fn; torch reports no function.
    loc = int(props["location_id"])
    return int(props.get("domain", 0)), loc >> 8, (loc >> 3) & 0x1F


def _unique_id(uuid: str) -> int | None:
    # torch's ROCm UUID is the KFD unique_id written as ASCII hex digits.
    try:
        return int(bytes.fromhex(uuid.replace("-", "")).decode(), 16)
    except ValueError:
        return None


def find_vram_used_file(
    uuid: str,
    pci: tuple[int, int, int],
    total_memory: int,
    *,
    nodes_dir: str = KFD_TOPOLOGY_NODES,
    drm_dir: str = DRM_SYSFS,
) -> tuple[str | None, str]:
    """The mem_info_vram_used file of the GPU with this torch UUID and PCI
    address, and hipMemGetInfo's total_memory; or None and why not."""
    try:
        nodes = _kfd_gpu_nodes(nodes_dir)
    except OSError as e:
        return None, f"no KFD topology ({e})"
    unique_id = _unique_id(uuid)
    by_uid = [n for n in nodes if int(n.get("unique_id", -1)) == unique_id]
    by_pci = [n for n in nodes if _pci(n) == pci]
    if len(by_pci) > 1:
        # Partitions share one device's VRAM counter.
        return None, "partitioned GPU"
    if len(by_pci) != 1 or (unique_id is not None and by_uid != by_pci):
        return None, "no single KFD node matches the GPU"
    device_dir = os.path.join(
        drm_dir, f"renderD{by_pci[0]['drm_render_minor']}", "device"
    )
    try:
        with open(os.path.join(device_dir, "mem_info_vram_total")) as f:
            vram_total = int(f.read())
        with open(os.path.join(device_dir, "mem_info_vram_used")) as f:
            int(f.read())
    except (OSError, ValueError) as e:
        return None, f"no amdgpu VRAM counters ({e})"
    # A total far from HIP's means memory HIP sees is not this VRAM (e.g. an
    # APU's shared memory). Below HIP's would let the cap exceed real VRAM.
    if not total_memory <= vram_total <= total_memory * (1 + VRAM_TOTAL_TOLERANCE):
        return None, (
            f"VRAM total {vram_total} bytes does not match HIP's {total_memory}"
        )
    return os.path.join(device_dir, "mem_info_vram_used"), ""


@dataclass(frozen=True)
class DeviceMemoryReading:
    """The exact VRAM count read together with free memory as HIP reports it,
    and this process's torch reservation."""

    used: int
    free: int
    total: int
    torch_reserved: int

    @property
    def gap(self) -> int:
        """Exact used bytes less the used bytes free memory implies."""
        return self.used - (self.total - self.free)


def free_memory_correction(
    baseline: DeviceMemoryReading,
    current: DeviceMemoryReading,
    untracked_bytes: int,
) -> int:
    """Free memory that current.free leaves out by the exact count, since
    baseline: the drop in free memory less the growth of the exact count.
    Other processes' allocations change both alike.

    Never negative, and never more than the total less the exact count.
    """
    # The exact count must have grown by at least what this process
    # allocated; if not, memory left VRAM or was freed meanwhile and the
    # difference cannot be trusted.
    own_growth = current.torch_reserved - baseline.torch_reserved + untracked_bytes
    if current.used - baseline.used < own_growth:
        logger.info(
            "Free memory not corrected: device memory in use grew less than "
            "this process allocated."
        )
        return 0
    return max(min(baseline.gap - current.gap, -current.gap), 0)


# A reading is settled once its gap has not fallen for this long. Freed memory
# stays in the exact count until the driver has erased it (about 10 ms per GB
# on an R9700) while free memory shows it free at once, which raises the gap.
SETTLE_SECONDS = 0.2
MAX_SAMPLING_SECONDS = 2.0
SAMPLE_INTERVAL_SECONDS = 0.002

# mem_info_vram_used per device index, or None where there is none.
_vram_used_files: dict[int, str | None] = {}
# Per device index; a later engine in the process replaces it.
_baselines: dict[int, DeviceMemoryReading] = {}


def _device_index(device: torch.device) -> int:
    return torch.cuda.current_device() if device.index is None else device.index


def _vram_used_file(index: int) -> str | None:
    """Only unpartitioned discrete GPUs whose VRAM total matches HIP's."""
    if index not in _vram_used_files:
        props = torch.cuda.get_device_properties(index)
        if props.is_integrated:
            path, reason = None, "integrated GPU"
        else:
            path, reason = find_vram_used_file(
                str(props.uuid),
                (props.pci_domain_id, props.pci_bus_id, props.pci_device_id),
                torch.cuda.mem_get_info(index)[1],
            )
        if path is None:
            logger.debug(
                "Using free device memory as HIP reports it, which counts each "
                "allocation as at least 2 MB: %s.",
                reason,
            )
        _vram_used_files[index] = path
    return _vram_used_files[index]


def _read_used(path: str) -> int | None:
    try:
        with open(path) as f:
            return int(f.read())
    except (OSError, ValueError):
        return None


def _settled_reading(index: int) -> DeviceMemoryReading | None:
    """The reading with the lowest gap, sampled until it stops falling; an
    erase still pending afterwards only makes the correction smaller."""
    path = _vram_used_file(index)
    if path is None:
        return None
    torch.cuda.synchronize(index)
    start = last_drop = time.monotonic()
    lowest = None
    while True:
        used = _read_used(path)
        free, total = torch.cuda.mem_get_info(index)
        used_after = _read_used(path)
        if used is None or used_after is None:
            return None
        now = time.monotonic()
        # Skip a sample when another process allocated between the two
        # counts: free memory would already show it and the gap be too low.
        if used_after <= used:
            reserved = torch.cuda.memory_reserved(index)
            reading = DeviceMemoryReading(used, free, total, reserved)
            if lowest is None or reading.gap < lowest.gap:
                lowest, last_drop = reading, now
        if now - last_drop >= SETTLE_SECONDS or now - start >= MAX_SAMPLING_SECONDS:
            return lowest
        time.sleep(SAMPLE_INTERVAL_SECONDS)


def record_memory_baseline(device: torch.device) -> None:
    index = _device_index(device)
    reading = _settled_reading(index)
    if reading is None:
        _baselines.pop(index, None)
    else:
        _baselines[index] = reading


def corrected_free_memory(
    device: torch.device, untracked_bytes: int
) -> tuple[int, int]:
    index = _device_index(device)
    baseline = _baselines.get(index)
    current = _settled_reading(index) if baseline is not None else None
    if baseline is None or current is None:
        from vllm.utils.mem_utils import device_memory_info

        return device_memory_info(device)[0], 0
    correction = free_memory_correction(baseline, current, untracked_bytes)
    if correction > 0:
        logger.info(
            "Free device memory: %.2f GiB reported, %.2f GiB more by the exact "
            "VRAM count.",
            current.free / (1 << 30),
            correction / (1 << 30),
        )
    return current.free + correction, correction
