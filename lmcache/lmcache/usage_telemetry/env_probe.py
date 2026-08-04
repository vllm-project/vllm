# SPDX-License-Identifier: Apache-2.0
"""Hardware and platform environment probing for usage telemetry."""

# Future
from __future__ import annotations

# Standard
from pathlib import Path
import importlib.metadata
import os
import platform
import subprocess

# Third Party
import cpuinfo
import psutil

# First Party
from lmcache import torch_dev
from lmcache.usage_telemetry.messages import EnvMessage


def _get_provider() -> str:
    """Detect the cloud provider from DMI files and environment variables."""
    vendor_files = [
        "/sys/class/dmi/id/product_version",
        "/sys/class/dmi/id/bios_vendor",
        "/sys/class/dmi/id/product_name",
        "/sys/class/dmi/id/chassis_asset_tag",
        "/sys/class/dmi/id/sys_vendor",
    ]
    # Mapping of identifiable strings to cloud providers
    cloud_identifiers = {
        "amazon": "AWS",
        "microsoft corporation": "AZURE",
        "google": "GCP",
        "oraclecloud": "OCI",
    }

    for vendor_file in vendor_files:
        path = Path(vendor_file)
        if path.is_file():
            file_content = path.read_text().lower()
            for identifier, provider in cloud_identifiers.items():
                if identifier in file_content:
                    return provider

    # Try detecting through environment variables
    env_to_cloud_provider = {
        "RUNPOD_DC_ID": "RUNPOD",
    }
    for env_var, provider in env_to_cloud_provider.items():
        if os.environ.get(env_var):
            return provider

    return "UNKNOWN"


def _get_cpu_info() -> tuple[int, str, str]:
    """Return (cpu count, cpu brand, family/model/stepping); 0 = unknown."""
    info = cpuinfo.get_cpu_info()
    num_cpu = int(info.get("count") or 0)
    cpu_type = str(info.get("brand_raw", ""))
    cpu_family_model_stepping = ",".join(
        [
            str(info.get("family", "")),
            str(info.get("model", "")),
            str(info.get("stepping", "")),
        ]
    )
    return num_cpu, cpu_type, cpu_family_model_stepping


def _get_gpu_info() -> tuple[int, str, int]:
    """Return (gpu count, gpu type, memory per device in bytes).

    On accelerator-less machines this falls back to the physical CPU count,
    the processor name, and total host memory.
    """
    if torch_dev.is_available():
        device_property = torch_dev.get_device_properties(0)
        return (
            torch_dev.device_count(),
            device_property.name,
            device_property.total_memory,
        )
    return (
        int(psutil.cpu_count(logical=False) or 0),
        platform.processor(),
        psutil.virtual_memory().total,
    )


def _get_source() -> str:
    """Detect the install source (DOCKER, PIP, CONDA, or UNKNOWN)."""
    path = "/proc/1/cgroup"
    if os.path.exists(path):
        with open(path, "r") as f:
            for line in f:
                if "docker" in line:
                    return "DOCKER"
    try:
        _ = importlib.metadata.distribution("LMCache")
        return "PIP"
    except importlib.metadata.PackageNotFoundError:
        pass
    try:
        result = subprocess.run(
            ["conda", "list", "LMCache"], capture_output=True, text=True
        )
        if "LMCache" in result.stdout:
            return "CONDA"
    except FileNotFoundError:
        pass

    return "UNKNOWN"


def collect_env_message() -> EnvMessage:
    """Collect the hardware/platform environment for one-shot reporting.

    Returns:
        The populated environment message.
    """
    num_cpu, cpu_type, cpu_family_model_stepping = _get_cpu_info()
    gpu_count, gpu_type, gpu_memory_per_device = _get_gpu_info()
    return EnvMessage(
        provider=_get_provider(),
        num_cpu=num_cpu,
        cpu_type=cpu_type,
        cpu_family_model_stepping=cpu_family_model_stepping,
        total_memory=psutil.virtual_memory().total,
        architecture=platform.architecture(),
        platforms=platform.platform(),
        gpu_count=gpu_count,
        gpu_type=gpu_type,
        gpu_memory_per_device=gpu_memory_per_device,
        source=_get_source(),
    )
