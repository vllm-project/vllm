# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""XPU GPU topology detection and automatic CCL/OFI environment
variable configuration."""

import glob
import os

from vllm.logger import init_logger

logger = init_logger(__name__)

# Topology type constants
XELINK = "xelink"
PCIE = "pcie"


def detect_xpu_interconnect() -> str:
    """Detect whether Intel GPUs are connected via XeLink or PCIe-only.

    Detection order:
      1. User override via ``VLLM_XPU_INTERCONNECT`` env var.
      2. Sysfs fabric port / IAF device detection.
      3. Device-name heuristic via ``torch.xpu``.
      4. Fall back to PCIe (safest default).
    """

    # 1. User override
    user_override = os.environ.get("VLLM_XPU_INTERCONNECT", "").lower()
    if user_override in (XELINK, PCIE):
        logger.info(
            "XPU interconnect overridden by VLLM_XPU_INTERCONNECT=%s",
            user_override,
        )
        return user_override

    # 2. Sysfs detection – look for XeLink fabric indicators
    try:
        iaf_paths = glob.glob("/sys/class/drm/card*/device/iaf.*")
        fabric_paths = glob.glob("/sys/class/drm/card*/gt/gt*/fabric_ports")
        if iaf_paths or fabric_paths:
            logger.info(
                "Detected XeLink interconnect via sysfs (iaf=%d, fabric_ports=%d)",
                len(iaf_paths),
                len(fabric_paths),
            )
            return XELINK
    except OSError:
        # sysfs may not be available in all environments
        pass

    # 3. Device-name heuristic
    try:
        import torch

        name = torch.xpu.get_device_properties(0).name.lower()
        if "data center gpu max" in name or "pvc" in name:
            logger.info("Detected XeLink-capable GPU via device name: %s", name)
            return XELINK
    except Exception:
        logger.warning(
            "Failed to query XPU device properties for interconnect "
            "detection; defaulting to PCIe topology."
        )
        return PCIE

    # 4. Default – PCIe
    logger.info("No XeLink indicators found; assuming PCIe topology.")
    return PCIE


def configure_ccl_env(interconnect: str, world_size: int) -> dict[str, str]:
    """Set CCL/OFI environment variables for the given topology.

    User-set variables are never overridden.  Returns a dict of all
    variables that were actually set by this function.
    """

    env_vars: dict[str, str] = {}

    def _set(key: str, value: str) -> None:
        existing = os.environ.get(key)
        if existing is not None:
            logger.info(
                "CCL env %s already set by user (%s); not overriding",
                key,
                existing,
            )
            return
        os.environ[key] = value
        env_vars[key] = value
        logger.info("Set CCL env %s=%s", key, value)

    if interconnect == PCIE:
        _set("CCL_ATL_TRANSPORT", "ofi")
        _set("FI_PROVIDER", "shm")
        _set("CCL_TOPO_FABRIC_VERTEX_CONNECTION_CHECK", "0")
        _set("CCL_ZE_IPC_EXCHANGE", "sockets")
    elif interconnect == XELINK:
        _set("CCL_ATL_TRANSPORT", "ofi")
    else:
        logger.warning(
            "Unknown XPU interconnect type %r; skipping CCL env setup",
            interconnect,
        )

    # Avoid OFI endpoint exhaustion on large clusters
    if world_size >= 8:
        _set("CCL_WORKER_COUNT", "1")

    return env_vars


def auto_configure_xpu_distributed(world_size: int) -> None:
    """Convenience wrapper: detect topology and configure CCL env."""
    interconnect = detect_xpu_interconnect()
    configure_ccl_env(interconnect, world_size)
