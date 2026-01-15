# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import platform
from dataclasses import field
from typing import Any, Literal

import cpuinfo
import psutil
import torch
from pydantic import ConfigDict, SkipValidation
from pydantic.dataclasses import dataclass

from vllm.config.utils import config
from vllm.utils.hashing import safe_hash
from vllm.utils.platform_utils import cuda_get_device_properties

Device = Literal["auto", "cuda", "cpu", "tpu", "xpu"]


@dataclass
class CudaInfo:
    """CUDA device information including GPU and CPU/platform details."""

    # GPU properties
    gpu_name: str | None = None
    """GPU name (e.g., 'NVIDIA A100-SXM4-40GB'). """
    gpu_memory: str | None = None
    """Total GPU memory (e.g., '39.59 GiB'). """
    gpu_compute_capability: str | None = None
    """GPU compute capability as a string (e.g., '8.6'). """
    gpu_multi_processor_count: int | None = None
    """Number of streaming multiprocessors (SMs). """
    gpu_device_index: int | None = None
    """GPU device index. """
    gpu_pci_bus_id: int | None = None
    """GPU PCI bus ID. """
    gpu_pci_device_id: int | None = None
    """GPU PCI device ID. """
    gpu_pci_domain_id: int | None = None
    """GPU PCI domain ID. """
    gpu_l2_cache_size: str | None = None
    """GPU L2 cache size (e.g., '6.00 MiB'). """

    # CPU and platform information
    cpu_count: int | None = None
    """Number of CPU cores."""
    cpu_type: str | None = None
    """CPU type/brand."""
    cpu_family_model_stepping: str | None = None
    """CPU family, model, and stepping as comma-separated string."""
    architecture: str | None = None
    """System architecture (e.g., 'x86_64')."""
    platform_info: str | None = None
    """Platform information string."""
    total_memory: str | None = None
    """Total system memory (e.g., '15.91 GiB'). """

    @classmethod
    def collect(
        cls,
        device_type: str,
        device: torch.device | None,
        current_platform: Any,
    ) -> "CudaInfo":
        """Collect CUDA information. Only collects data if device_type is 'cuda'."""
        cuda_info = cls()

        if device_type != "cuda" or not current_platform.is_cuda_alike():
            return cuda_info

        device_index = (
            device.index if device is not None and device.index is not None else 0
        )

        # Get all GPU properties
        property_names = [
            "name",
            "total_memory",
            "major",
            "minor",
            "multi_processor_count",
            "pci_bus_id",
            "pci_device_id",
            "pci_domain_id",
            "L2_cache_size",
        ]
        (
            cuda_info.gpu_name,
            gpu_memory,
            major,
            minor,
            cuda_info.gpu_multi_processor_count,
            cuda_info.gpu_pci_bus_id,
            cuda_info.gpu_pci_device_id,
            cuda_info.gpu_pci_domain_id,
            l2_cache_size,
        ) = cuda_get_device_properties(device_index, property_names)

        if gpu_memory is not None:
            cuda_info.gpu_memory = f"{gpu_memory / 1024 / 1024 / 1024:.2f} GiB"
        if l2_cache_size is not None:
            cuda_info.gpu_l2_cache_size = f"{l2_cache_size / 1024 / 1024:.2f} MiB"
        cuda_info.gpu_compute_capability = (
            f"{major}.{minor}" if major is not None and minor is not None else None
        )
        cuda_info.gpu_device_index = device_index

        # Collect CPU and platform information
        info = cpuinfo.get_cpu_info()
        cuda_info.cpu_count = info.get("count", None)
        cuda_info.cpu_type = info.get("brand_raw", "")
        cpu_family = str(info.get("family", ""))
        cpu_model = str(info.get("model", ""))
        cpu_stepping = str(info.get("stepping", ""))
        cuda_info.cpu_family_model_stepping = ",".join(
            [cpu_family, cpu_model, cpu_stepping]
        )

        # Platform information
        cuda_info.architecture = platform.machine()
        cuda_info.platform_info = platform.platform()
        total_memory = psutil.virtual_memory().total
        cuda_info.total_memory = (
            f"{total_memory / 1024 / 1024 / 1024:.2f} GiB"
            if total_memory is not None
            else None
        )

        return cuda_info


@config
@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class DeviceConfig:
    """Configuration for the device to use for vLLM execution."""

    device: SkipValidation[Device | torch.device | None] = "auto"
    """Device type for vLLM execution.
    This parameter is deprecated and will be
    removed in a future release.
    It will now be set automatically based
    on the current platform."""
    device_type: str = field(init=False)
    """Device type from the current platform. This is set in
    `__post_init__`."""
    cuda_info: CudaInfo = field(init=False)
    """CUDA device information including GPU and CPU/platform details."""

    def compute_hash(self) -> str:
        """
        WARNING: Whenever a new field is added to this config,
        ensure that it is included in the factors list if
        it affects the computation graph.

        Provide a hash that uniquely identifies all the configs
        that affect the structure of the computation
        graph from input ids/embeddings to the final hidden states,
        excluding anything before input ids/embeddings and after
        the final hidden states.
        """
        # no factors to consider.
        # the device/platform information will be summarized
        # by torch/vllm automatically.
        factors: list[Any] = []
        hash_str = safe_hash(str(factors).encode(), usedforsecurity=False).hexdigest()
        return hash_str

    def __post_init__(self):
        from vllm.platforms import current_platform

        if self.device == "auto":
            # Automated device type detection
            self.device_type = current_platform.device_type
            if not self.device_type:
                raise RuntimeError(
                    "Failed to infer device type, please set "
                    "the environment variable `VLLM_LOGGING_LEVEL=DEBUG` "
                    "to turn on verbose logging to help debug the issue."
                )
        else:
            # Device type is assigned explicitly
            if isinstance(self.device, str):
                self.device_type = self.device
            elif isinstance(self.device, torch.device):
                self.device_type = self.device.type

        # Some device types require processing inputs on CPU
        if self.device_type in ["tpu"]:
            self.device = None
        else:
            # Set device with device type
            self.device = torch.device(self.device_type)

        # Collect CUDA information (only populated for CUDA devices)
        self.cuda_info = CudaInfo.collect(
            self.device_type, self.device, current_platform
        )

    def metrics_info(self):
        result = {}
        for key, value in self.__dict__.items():
            if key == "cuda_info":
                # Flatten cuda_info fields
                for cuda_key, cuda_value in value.__dict__.items():
                    result[f"{cuda_key}"] = str(cuda_value)
            else:
                result[key] = str(value)
        return result
