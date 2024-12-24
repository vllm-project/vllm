from dataclasses import dataclass, field
from typing import Callable, Dict, Optional

from vllm import utils
from vllm.platforms import Platform

from .interface import UnspecifiedPlatform

# The list of supported in-tree platforms. Update this list when adding/removing
# platforms.
_VLLM_PLATFORMS = {
    "cpu": "vllm.platforms.cpu.CpuPlatform",
    "cuda": "vllm.platforms.cuda.CudaPlatform",
    "hpu": "vllm.platforms.hpu.HpuPlatform",
    "neuron": "vllm.platforms.neuron.NeuronPlatform",
    "openvino": "vllm.platforms.openvino.OpenVinoPlatform",
    "rocm": "vllm.platforms.rocm.RocmPlatform",
    "tpu": "vllm.platforms.tpu.TpuPlatform",
    "xpu": "vllm.platforms.xpu.XPUPlatform",
}


@dataclass
class _PlatformRegistry:
    # The mapping from device name to platform class string.
    platforms: Dict[str, str] = field(default_factory=dict)
    # The current platform name.
    current_platform: Optional[str] = None

    def _load_platform_cls(self, device_name: str) -> Callable:
        """Load a platform object by device name."""
        if device_name not in self.platforms:
            raise ValueError(
                f"Platform {device_name} not registered. "
                f"Available platforms: {list(self.platforms.keys())}")
        platform_cls_str = self.platforms[device_name]
        return utils.resolve_obj_by_qualname(platform_cls_str)

    def register_platform(self, device_name: str, platform: str):
        """Register a platform by device name. This function is called by the
        platform plugin."""
        if device_name in self.platforms:
            raise ValueError(f"Platform {device_name} already registered.")
        self.platforms[device_name] = platform

    def set_current_platform(self, device_name: str):
        """Set the current platform by device name."""
        if device_name not in self.platforms:
            raise ValueError(
                f"Platform {device_name} not registered. "
                f"Available platforms: {list(self.platforms.keys())}")
        self.current_platform = device_name

    def get_current_platform_cls(self) -> Callable:
        """Get the current platform object."""
        if self.current_platform is None:
            raise ValueError("No current platform set.")
        return self._load_platform_cls(self.current_platform)


PlatformRegistry = _PlatformRegistry({
    device_name: platform
    for device_name, platform in _VLLM_PLATFORMS.items()
})


def detect_current_platform() -> Platform:
    """Detect the current platform by checking the installed packages."""
    CurrentPlatform: Optional[type[Platform]] = None
    # NOTE: we don't use `torch.version.cuda` / `torch.version.hip` because
    # they only indicate the build configuration, not the runtime environment.
    # For example, people can install a cuda build of pytorch but run on tpu.

    # Load TPU Platform
    try:
        # While it's technically possible to install libtpu on a non-TPU
        # machine, this is a very uncommon scenario. Therefore, we assume that
        # libtpu is installed if and only if the machine has TPUs.
        import libtpu  # noqa: F401

        from .tpu import TpuPlatform as CurrentPlatform
    except Exception:
        pass

    # Load CUDA Platform
    if not CurrentPlatform:
        try:
            import pynvml
            pynvml.nvmlInit()
            try:
                if pynvml.nvmlDeviceGetCount() > 0:
                    from .cuda import CudaPlatform as CurrentPlatform
            finally:
                pynvml.nvmlShutdown()
        except Exception:
            # CUDA is supported on Jetson, but NVML may not be.
            import os

            def cuda_is_jetson() -> bool:
                return os.path.isfile("/etc/nv_tegra_release") \
                    or os.path.exists("/sys/class/tegra-firmware")

            if cuda_is_jetson():
                from .cuda import CudaPlatform as CurrentPlatform

    # Load ROCm Platform
    if not CurrentPlatform:
        try:
            import amdsmi
            amdsmi.amdsmi_init()
            try:
                if len(amdsmi.amdsmi_get_processor_handles()) > 0:
                    from .rocm import RocmPlatform as CurrentPlatform
            finally:
                amdsmi.amdsmi_shut_down()
        except Exception:
            pass

    # Load HPU Platform
    if not CurrentPlatform:
        try:
            from importlib import util
            assert util.find_spec('habana_frameworks') is not None
            from .hpu import HpuPlatform as CurrentPlatform
        except Exception:
            pass

    # Load XPU Platform
    if not CurrentPlatform:
        try:
            # installed IPEX if the machine has XPUs.
            import intel_extension_for_pytorch  # noqa: F401
            import oneccl_bindings_for_pytorch  # noqa: F401
            import torch
            if hasattr(torch, 'xpu') and torch.xpu.is_available():
                from .xpu import XPUPlatform as CurrentPlatform
        except Exception:
            pass

    # Load CPU Platform
    if not CurrentPlatform:
        try:
            from importlib.metadata import version
            assert "cpu" in version("vllm")
            from .cpu import CpuPlatform as CurrentPlatform
        except Exception:
            pass

    # Load Neuron Platform
    if not CurrentPlatform:
        try:
            import transformers_neuronx  # noqa: F401

            from .neuron import NeuronPlatform as CurrentPlatform
        except ImportError:
            pass

    # Load OpenVINO Platform
    if not CurrentPlatform:
        try:
            from importlib.metadata import version
            assert "openvino" in version("vllm")
            from .openvino import OpenVinoPlatform as CurrentPlatform
        except Exception:
            pass

    if CurrentPlatform:
        PlatformRegistry.set_current_platform(CurrentPlatform.device_name)
        return CurrentPlatform()

    return UnspecifiedPlatform()
