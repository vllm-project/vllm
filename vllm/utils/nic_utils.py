# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NIC binding utilities for vLLM worker processes.

Sets ``NCCL_IB_HCA`` and ``UCX_NET_DEVICES`` per worker so each process
uses the RDMA NIC(s) closest to its GPU (or CPU NUMA node).  The user
provides device specs in ``NCCL_IB_HCA`` syntax (prefix match, ``=``
exact match, ``^`` exclude, ``:port``).  The value is passed through to
``NCCL_IB_HCA`` unchanged; for ``UCX_NET_DEVICES`` vLLM expands the
pattern against the IB devices visible in sysfs.
"""

import logging
import os
from contextlib import contextmanager
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING

import regex as re

from vllm.utils.numa_utils import _get_gpu_index

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = logging.getLogger(__name__)

# Regex for a single NCCL_IB_HCA token (without the comma separator).
# Optional ^ (exclude) and/or = (exact) prefixes, then a device name
# composed of word characters (alphanumeric + underscore), optionally
# followed by :<port>.
_NCCL_HCA_TOKEN_RE = re.compile(
    r"^(?P<exclude>\^)?(?P<exact>=)?(?P<name>[\w]+)(?::(?P<port>\d+))?$"
)

# ---------------------------------------------------------------------------
# NCCL_IB_HCA syntax validation
# ---------------------------------------------------------------------------


def validate_nccl_hca_syntax(pattern: str) -> None:
    """Validate ``NCCL_IB_HCA`` pattern syntax.

    Raises :class:`ValueError` on malformed input.  Does **not** require
    sysfs access — this is a pure syntax check suitable for config-time
    validation.
    """
    if not pattern or not pattern.strip():
        raise ValueError("NIC device pattern must not be empty.")

    for token in pattern.split(","):
        token = token.strip()
        if not token:
            raise ValueError(
                f"NIC device pattern has empty token (double comma?): {pattern!r}"
            )
        if not _NCCL_HCA_TOKEN_RE.fullmatch(token):
            raise ValueError(
                f"Invalid NCCL_IB_HCA token {token!r} in pattern {pattern!r}. "
                "Expected format: [^][=]<name>[:<port>]"
            )


# ---------------------------------------------------------------------------
# Sysfs IB device enumeration
# ---------------------------------------------------------------------------

_SYSFS_IB_PATH = Path("/sys/class/infiniband")


@cache
def enumerate_ib_devices() -> dict[str, list[int]] | None:
    """Enumerate InfiniBand devices and their active ports from sysfs.

    Returns a mapping ``{"mlx5_0": [1], "mlx5_1": [1, 2], ...}`` or
    *None* if ``/sys/class/infiniband`` does not exist (no RDMA stack).
    """
    if not _SYSFS_IB_PATH.is_dir():
        return None

    devices: dict[str, list[int]] = {}
    for dev_dir in sorted(_SYSFS_IB_PATH.iterdir()):
        if not dev_dir.is_dir():
            continue
        ports_dir = dev_dir / "ports"
        if not ports_dir.is_dir():
            continue
        ports: list[int] = []
        for port_dir in sorted(ports_dir.iterdir()):
            if port_dir.is_dir() and port_dir.name.isdigit():
                ports.append(int(port_dir.name))
        if ports:
            devices[dev_dir.name] = ports
    return devices if devices else None


def get_ib_device_numa_node(device: str) -> int | None:
    """Return the NUMA node for an IB device, or *None* if unknown."""
    numa_path = _SYSFS_IB_PATH / device / "device" / "numa_node"
    try:
        value = numa_path.read_text().strip()
        node = int(value)
        # -1 means "not available" on some kernels
        return node if node >= 0 else None
    except (OSError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Pattern expansion (NCCL_IB_HCA → explicit device:port list for UCX)
# ---------------------------------------------------------------------------


def expand_nccl_hca_pattern(
    pattern: str,
    available: dict[str, list[int]],
) -> list[str]:
    """Expand an ``NCCL_IB_HCA`` pattern into explicit ``device:port`` pairs.

    *available* is the dict returned by :func:`enumerate_ib_devices`.
    The expansion follows NCCL semantics:

    * bare name → prefix match (``mlx5`` matches ``mlx5_0``, ``mlx5_1``, …)
    * ``=`` prefix → exact match
    * ``^`` prefix → exclude (prefix by default, ``^=`` for exact)
    * ``:port`` suffix → filter to that port only

    Returns a sorted list of ``"device:port"`` strings.
    Raises :class:`RuntimeError` if the pattern matches zero devices.
    """
    include_set: set[str] = set()
    exclude_set: set[str] = set()

    for token in pattern.split(","):
        token = token.strip()
        m = _NCCL_HCA_TOKEN_RE.fullmatch(token)
        if m is None:
            raise RuntimeError(f"Malformed NCCL_IB_HCA token: {token!r}")

        is_exclude = m.group("exclude") is not None
        is_exact = m.group("exact") is not None
        name = m.group("name")
        port_filter = int(m.group("port")) if m.group("port") else None

        matched = _match_devices(name, is_exact, port_filter, available)
        if is_exclude:
            exclude_set.update(matched)
        else:
            include_set.update(matched)

    # If no explicit include tokens, start with all devices
    has_include_token = any(
        _NCCL_HCA_TOKEN_RE.fullmatch(t.strip()).group("exclude") is None  # type: ignore[union-attr]
        for t in pattern.split(",")
    )
    if not has_include_token:
        # All tokens are exclusions → start with everything
        for dev, ports in available.items():
            for port in ports:
                include_set.add(f"{dev}:{port}")

    result = sorted(include_set - exclude_set)
    if not result:
        raise RuntimeError(
            f"NCCL_IB_HCA pattern {pattern!r} matched no devices among "
            f"{list(available.keys())}."
        )
    return result


def _match_devices(
    name: str,
    exact: bool,
    port_filter: int | None,
    available: dict[str, list[int]],
) -> set[str]:
    """Return ``device:port`` pairs matching a single token."""
    matched: set[str] = set()
    for dev, ports in available.items():
        if exact:
            if dev != name:
                continue
        else:
            if not dev.startswith(name):
                continue

        target_ports = [port_filter] if port_filter is not None else ports
        for port in target_ports:
            if port in ports:
                matched.add(f"{dev}:{port}")
    return matched


# ---------------------------------------------------------------------------
# Auto-detection: GPU/worker → NIC affinity via NUMA topology
# ---------------------------------------------------------------------------


@cache
def get_auto_nic_devices() -> list[str] | None:
    """Auto-detect per-GPU NIC device specs via NUMA topology.

    Returns a list of ``NCCL_IB_HCA``-style patterns (one per visible
    GPU), using exact-match syntax so both NCCL and UCX get unambiguous
    values.  Returns *None* when detection is not possible.
    """
    from vllm.platforms import current_platform

    if not hasattr(current_platform, "get_all_device_numa_nodes"):
        logger.warning(
            "Platform %s does not support GPU NUMA detection; "
            "skipping automatic NIC binding.",
            type(current_platform).__name__,
        )
        return None

    gpu_numa_nodes = current_platform.get_all_device_numa_nodes()
    if gpu_numa_nodes is None:
        return None

    available = enumerate_ib_devices()
    if available is None:
        logger.warning(
            "No InfiniBand devices found in sysfs; skipping automatic NIC binding."
        )
        return None

    # Build NIC NUMA map: numa_node → ["mlx5_0:1", ...]
    nic_by_numa: dict[int, list[str]] = {}
    for dev, ports in available.items():
        nic_numa = get_ib_device_numa_node(dev)
        if nic_numa is None:
            logger.debug(
                "Cannot determine NUMA node for IB device %s; "
                "excluding from auto-detection.",
                dev,
            )
            continue
        for port in ports:
            nic_by_numa.setdefault(nic_numa, []).append(f"{dev}:{port}")

    if not nic_by_numa:
        logger.warning(
            "Could not determine NUMA nodes for any IB device; "
            "skipping automatic NIC binding."
        )
        return None

    # Match each GPU to the NICs on its NUMA node
    result: list[str] = []
    for gpu_idx, gpu_numa in enumerate(gpu_numa_nodes):
        nics = nic_by_numa.get(gpu_numa)
        if nics is None:
            logger.warning(
                "GPU %d is on NUMA node %d but no IB device was found "
                "on that node; disabling automatic NIC binding.",
                gpu_idx,
                gpu_numa,
            )
            return None
        # Use exact-match syntax for each device
        result.append(",".join(f"={nic}" for nic in sorted(nics)))

    logger.info("Auto-detected NIC binding for GPUs: %s", result)
    return result


# ---------------------------------------------------------------------------
# Per-worker env var computation
# ---------------------------------------------------------------------------


def _get_nic_device_spec(parallel_config, gpu_index: int) -> str | None:
    """Return the raw NCCL_IB_HCA pattern for a single worker."""
    devices = parallel_config.nic_bind_devices
    if devices is None:
        devices = get_auto_nic_devices()
        if devices is None:
            raise RuntimeError(
                "NIC binding was requested but vLLM could not detect the "
                "GPU-to-NIC topology automatically. Pass --nic-bind-devices "
                "explicitly or disable --nic-bind."
            )
        parallel_config.nic_bind_devices = devices

    if gpu_index >= len(devices):
        raise ValueError(
            f"GPU index {gpu_index} exceeds nic_bind_devices size "
            f"{len(devices)}. Ensure the binding list covers every "
            "visible GPU."
        )
    return devices[gpu_index]


def _expand_for_ucx(pattern: str) -> str:
    """Expand a NCCL_IB_HCA pattern into a UCX_NET_DEVICES value.

    Enumerates IB devices from sysfs, applies the NCCL matching rules,
    and returns a comma-separated list of explicit ``device:port`` pairs.
    """
    available = enumerate_ib_devices()
    if available is None:
        logger.warning(
            "No InfiniBand devices in sysfs; cannot expand pattern "
            "for UCX_NET_DEVICES (pattern: %s).",
            pattern,
        )
        return pattern  # best effort: pass through as-is
    return ",".join(expand_nccl_hca_pattern(pattern, available))


def get_nic_env_vars(
    vllm_config: "VllmConfig",
    local_rank: int,
    dp_local_rank: int | None = None,
) -> dict[str, str] | None:
    """Compute ``NCCL_IB_HCA`` and ``UCX_NET_DEVICES`` for a worker.

    Returns *None* if NIC binding is disabled.
    """
    parallel_config = vllm_config.parallel_config
    if not parallel_config.nic_bind:
        return None

    gpu_index = _get_gpu_index(parallel_config, local_rank, dp_local_rank)
    pattern = _get_nic_device_spec(parallel_config, gpu_index)
    if pattern is None:
        return None

    ucx_value = _expand_for_ucx(pattern)
    logger.info(
        "NIC binding (gpu_index=%s): NCCL_IB_HCA=%s, UCX_NET_DEVICES=%s",
        gpu_index,
        pattern,
        ucx_value,
    )
    return {
        "NCCL_IB_HCA": pattern,
        "UCX_NET_DEVICES": ucx_value,
    }


# ---------------------------------------------------------------------------
# Context manager for multiproc subprocess spawning
# ---------------------------------------------------------------------------


@contextmanager
def configure_subprocess(
    vllm_config: "VllmConfig",
    local_rank: int,
    dp_local_rank: int | None = None,
    process_kind: str = "worker",
):
    """Temporarily set NIC binding env vars for a child process."""
    env_vars = get_nic_env_vars(vllm_config, local_rank, dp_local_rank)
    if env_vars is None:
        yield
        return

    old_values = {k: os.environ.get(k) for k in env_vars}
    for k, v in env_vars.items():
        os.environ[k] = v
    try:
        yield
    finally:
        for k, old_v in old_values.items():
            if old_v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = old_v


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def log_nic_binding(label: str) -> None:
    """Log the effective NIC binding env vars for a worker."""
    hca = os.environ.get("NCCL_IB_HCA", "<unset>")
    ucx = os.environ.get("UCX_NET_DEVICES", "<unset>")
    logger.info("%s NIC binding: NCCL_IB_HCA=%s, UCX_NET_DEVICES=%s", label, hca, ucx)
