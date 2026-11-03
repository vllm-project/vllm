# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Where amdgpu reports the exact VRAM used on a GPU.

hipMemGetInfo counts each allocation as at least 2 MB unless the GPU's
memory is kept in system memory (kernel apu_prefer_gtt). amdgpu's sysfs
mem_info_vram_used counts exact sizes for the whole device, plus page tables,
queue save areas, reserved ranges and graphics buffers. Plain file parsing,
so it is importable and testable without a GPU.
"""

import os

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
