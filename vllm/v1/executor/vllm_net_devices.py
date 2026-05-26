# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GPU-to-NIC net-device mapping for RDMA transports (UCX, NVSHMEM, ...).

Shared by both UniProcExecutor (TP=1) and MultiprocExecutor (TP>1).
All transport-specific env vars and sysfs lookups live here so executor
files only need to call ``set_worker_net_device(local_rank, vllm_config)``.

Requires two env vars set together:
- ``VLLM_GPU_NIC_PCIE_MAPPING`` -- comma-separated GPU_BDF=NIC_BDF pairs.
- ``VLLM_NIC_SELECTION_VARS`` -- comma-separated list of env vars to set,
  each optionally suffixed (e.g. ``UCX_NET_DEVICES:1,NCCL_IB_HCA:1``).
"""

import os
from pathlib import Path

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.platforms import current_platform

logger = init_logger(__name__)


def normalize_pci(addr: str) -> tuple[int, int, int, int]:
    """Parse PCI BDF/domain-bus-device-function into comparable ints (all hex).

    Supported shapes:
    - ``domain:bus:dev.fn`` -- domain width varies (e.g. ``00000001:00:00.0``,
      ``0001:00:00.0``, ``0000:3f:00.0``).
    - ``bus:dev.fn`` -- domain **0** (e.g. ``01:00.0``, ``40:00.0``).

    Function suffix is hex (typically ``0``--``7``). Raises ``ValueError`` if malformed.
    """
    s = addr.strip().lower().replace(" ", "")
    if s.startswith("0x"):
        s = s[2:]
    if "." not in s:
        raise ValueError(f"invalid PCI BDF (missing function suffix): {addr!r}")
    body, fn_s = s.rsplit(".", 1)
    if not fn_s or any(c not in "0123456789abcdef" for c in fn_s):
        raise ValueError(f"invalid PCI function in BDF: {addr!r}")
    fn = int(fn_s, 16)
    if fn > 0xFF:
        raise ValueError(f"PCI function out of range: {addr!r}")

    parts = body.split(":")
    if len(parts) == 2:
        domain = 0
        bus = int(parts[0], 16)
        device = int(parts[1], 16)
    elif len(parts) == 3:
        domain = int(parts[0], 16)
        bus = int(parts[1], 16)
        device = int(parts[2], 16)
    else:
        raise ValueError(
            f"invalid PCI BDF (want domain:bus:dev.fn or bus:dev.fn): {addr!r}"
        )

    if bus > 0xFF or device > 0x1F:
        raise ValueError(f"PCI bus or device out of range: {addr!r}")
    return (domain, bus, device, fn)


def parse_gpu_nic_mapping(
    raw: str,
) -> dict[tuple[int, int, int, int], tuple[int, int, int, int]]:
    out: dict[tuple[int, int, int, int], tuple[int, int, int, int]] = {}
    for segment in raw.split(","):
        segment = segment.strip()
        if not segment:
            continue
        if "=" not in segment:
            raise ValueError(
                "VLLM_GPU_NIC_PCIE_MAPPING: expected comma-separated gpu_bdf=nic_bdf pairs; "
                f"ambiguous segment: {segment!r}"
            )
        gpu_s, nic_s = segment.split("=", 1)
        gpu_key = normalize_pci(gpu_s.strip())
        nic_val = normalize_pci(nic_s.strip())
        out[gpu_key] = nic_val
    return out




def rdma_name_for_nic_pci(nic_pci: tuple[int, int, int, int]) -> str:
    """Map NIC PCI BDF to sysfs RDMA name (mlx5_*, ibp*, ...).

    Under ``/sys/class/infiniband/<name>/``, ``device`` is a **symlink** to the PCI
    device directory (e.g. ``.../0101:00:00.0``). We take ``Path(...).resolve().name``
    as the BDF string.

    ``VLLM_GPU_NIC_PCIE_MAPPING`` NIC keys must **normalize** (via ``normalize_pci``)
    to the same tuple as this basename.
    """
    ib = Path("/sys/class/infiniband")
    if not ib.is_dir():
        raise RuntimeError("/sys/class/infiniband not found or not a directory")
    names = sorted(p.name for p in ib.iterdir() if p.is_dir())
    for name in names:
        dev_link = ib / name / "device"
        if not dev_link.exists():
            continue
        try:
            resolved = dev_link.resolve()
        except OSError:
            continue
        pci_name = resolved.name
        try:
            if normalize_pci(pci_name) == nic_pci:
                return name
        except ValueError:
            continue
    raise RuntimeError(
        f"No /sys/class/infiniband device for NIC PCI {nic_pci}; "
        f"have entries: {names}"
    )


def parse_nic_selection_vars(raw: str) -> list[tuple[str, str]]:
    """Parse ``VLLM_NIC_SELECTION_VARS`` into ``(env_var_name, suffix)`` pairs.

    Each entry is ``VAR_NAME`` or ``VAR_NAME:<suffix>``.  The colon and
    everything after it is appended verbatim to the RDMA device name.
    """
    result: list[tuple[str, str]] = []
    for entry in raw.split(","):
        entry = entry.strip()
        if not entry:
            continue
        if ":" in entry:
            var_name, suffix = entry.split(":", 1)
            result.append((var_name, ":" + suffix))
        else:
            result.append((entry, ""))
    return result


def set_worker_gpu_nic_mapping(local_rank: int) -> None:
    """Set NIC selection env vars from VLLM_GPU_NIC_PCIE_MAPPING for a worker.

    Which env vars are set is controlled by ``VLLM_NIC_SELECTION_VARS``.
    """
    raw = os.environ.get("VLLM_GPU_NIC_PCIE_MAPPING", "").strip()
    if not raw:
        return
    selection_raw = os.environ.get("VLLM_NIC_SELECTION_VARS", "").strip()
    selection_vars = parse_nic_selection_vars(selection_raw)
    mapping = parse_gpu_nic_mapping(raw)
    pci_by_index = current_platform.get_all_gpu_pci_bus_ids()
    # Translate CUDA-relative local_rank to the physical device index,
    # which accounts for CUDA_VISIBLE_DEVICES narrowing (e.g. DP sharding).
    physical_id = current_platform.device_id_to_physical_device_id(local_rank)
    if physical_id not in pci_by_index:
        raise RuntimeError(
            f"No GPU PCI for physical device index {physical_id} "
            f"(local_rank={local_rank}) in map "
            f"(have indices {sorted(pci_by_index.keys())})"
        )
    gpu_bdf = pci_by_index[physical_id]
    gpu_key = normalize_pci(gpu_bdf)
    if gpu_key not in mapping:
        keys_fmt = ", ".join(
            f"{d:04x}:{b:02x}:{dev:02x}.{fn}"
            for d, b, dev, fn in sorted(mapping.keys())
        )
        raise RuntimeError(
            f"No VLLM_GPU_NIC_PCIE_MAPPING entry for GPU PCI {gpu_bdf} "
            f"(worker local_rank={local_rank}); mapped GPUs: {keys_fmt}"
        )
    nic_pci = mapping[gpu_key]
    rdma_dev = rdma_name_for_nic_pci(nic_pci)

    set_vars: list[str] = []
    for var_name, suffix in selection_vars:
        value = f"{rdma_dev}{suffix}"
        existing = os.environ.get(var_name, "").strip()
        if existing:
            value = f"{value},{existing}"
        os.environ[var_name] = value
        set_vars.append(f"{var_name}={value}")

    nic_fmt = f"{nic_pci[0]:04x}:{nic_pci[1]:02x}:{nic_pci[2]:02x}.{nic_pci[3]}"
    logger.info(
        "GPU rank %s (PCIe addr %s) mapped to NIC %s (PCIe addr %s) via env vars: %s",
        local_rank,
        gpu_bdf,
        rdma_dev,
        nic_fmt,
        ", ".join(set_vars),
    )


def _dp_adjusted_local_rank(tp_local_rank: int, vllm_config: VllmConfig) -> int:
    """Compute the node-wide GPU index accounting for data parallelism.

    On CUDA-alike platforms without env-var device isolation (the common
    MP-backend path), the worker sees *all* GPUs on the node and selects
    its device via ``torch.accelerator.set_device_index()`` using::

        dp_local_rank * tp_pp_world_size + tp_local_rank

    This mirrors the adjustment in ``Worker.init_device()`` so we resolve
    the correct GPU PCI address *before* the CUDA device is initialised.
    """
    pc = vllm_config.parallel_config
    if (
        pc.distributed_executor_backend not in ("ray", "external_launcher")
        and pc.data_parallel_backend != "ray"
        and pc.nnodes_within_dp == 1
    ):
        dp_local_rank = pc.data_parallel_rank_local
        if dp_local_rank is None:
            dp_local_rank = pc.data_parallel_index
        tp_pp_world_size = pc.pipeline_parallel_size * pc.tensor_parallel_size
        return dp_local_rank * tp_pp_world_size + tp_local_rank
    return tp_local_rank


def set_worker_net_device(local_rank: int, vllm_config: VllmConfig) -> None:
    """Top-level entry point for both UniProcExecutor and MultiprocExecutor.

    Sets NIC selection env vars from ``VLLM_GPU_NIC_PCIE_MAPPING`` and
    ``VLLM_NIC_SELECTION_VARS`` if present; no-op otherwise.
    """
    has_pcie_mapping = bool(
        os.environ.get("VLLM_GPU_NIC_PCIE_MAPPING", "").strip())
    has_selection_vars = bool(
        os.environ.get("VLLM_NIC_SELECTION_VARS", "").strip())
    if has_pcie_mapping and not has_selection_vars:
        raise RuntimeError(
            "VLLM_GPU_NIC_PCIE_MAPPING is set but VLLM_NIC_SELECTION_VARS "
            "is not; both must be set together."
        )
    if has_selection_vars and not has_pcie_mapping:
        raise RuntimeError(
            "VLLM_NIC_SELECTION_VARS is set but VLLM_GPU_NIC_PCIE_MAPPING "
            "is not; both must be set together."
        )
    # No-op when neither env var is present.
    if not has_pcie_mapping and not has_selection_vars:
        return
    # Both env vars are present, so set the NIC selection env vars.
    adjusted_rank = _dp_adjusted_local_rank(local_rank, vllm_config)
    set_worker_gpu_nic_mapping(adjusted_rank)
