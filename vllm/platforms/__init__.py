import logging
import traceback
from itertools import chain
from typing import TYPE_CHECKING, Tuple

from vllm.plugins import load_plugins_by_group
from vllm.utils import resolve_obj_by_qualname

from .interface import _Backend  # noqa: F401
from .interface import CpuArchEnum, Platform, PlatformEnum

logger = logging.getLogger(__name__)


def tpu_platform_plugin() -> Tuple[bool, str]:
    is_tpu = False
    try:
        # While it's technically possible to install libtpu on a
        # non-TPU machine, this is a very uncommon scenario. Therefore,
        # we assume that libtpu is installed if and only if the machine
        # has TPUs.
        import libtpu  # noqa: F401
        is_tpu = True
    except Exception:
        pass

    return is_tpu, "vllm.platforms.tpu.TpuPlatform"


def cuda_platform_plugin() -> Tuple[bool, str]:
    is_cuda = False

    try:
        import pynvml
        pynvml.nvmlInit()
        try:
            if pynvml.nvmlDeviceGetCount() > 0:
                is_cuda = True
        finally:
            pynvml.nvmlShutdown()
    except Exception:
        # CUDA is supported on Jetson, but NVML may not be.
        import os

        def cuda_is_jetson() -> bool:
            return os.path.isfile("/etc/nv_tegra_release") \
                or os.path.exists("/sys/class/tegra-firmware")

        if cuda_is_jetson():
            is_cuda = True

    return is_cuda, "vllm.platforms.cuda.CudaPlatform"


def rocm_platform_plugin() -> Tuple[bool, str]:
    is_rocm = False

    try:
        import amdsmi
        amdsmi.amdsmi_init()
        try:
            if len(amdsmi.amdsmi_get_processor_handles()) > 0:
                is_rocm = True
        finally:
            amdsmi.amdsmi_shut_down()
    except Exception:
        pass

    return is_rocm, "vllm.platforms.rocm.RocmPlatform"


def hpu_platform_plugin() -> Tuple[bool, str]:
    is_hpu = False
    try:
        from importlib import util
        is_hpu = util.find_spec('habana_frameworks') is not None
    except Exception:
        pass

    return is_hpu, "vllm.platforms.hpu.HpuPlatform"


def xpu_platform_plugin() -> Tuple[bool, str]:
    is_xpu = False

    try:
        # installed IPEX if the machine has XPUs.
        import intel_extension_for_pytorch  # noqa: F401
        import oneccl_bindings_for_pytorch  # noqa: F401
        import torch
        if hasattr(torch, 'xpu') and torch.xpu.is_available():
            is_xpu = True
    except Exception:
        pass

    return is_xpu, "vllm.platforms.xpu.XPUPlatform"


def cpu_platform_plugin() -> Tuple[bool, str]:
    is_cpu = False
    try:
        from importlib.metadata import version
        is_cpu = "cpu" in version("vllm")
    except Exception:
        pass

    return is_cpu, "vllm.platforms.cpu.CpuPlatform"


def neuron_platform_plugin() -> Tuple[bool, str]:
    is_neuron = False
    try:
        import transformers_neuronx  # noqa: F401
        is_neuron = True
    except ImportError:
        pass

    return is_neuron, "vllm.platforms.neuron.NeuronPlatform"


def openvino_platform_plugin() -> Tuple[bool, str]:
    is_openvino = False
    try:
        from importlib.metadata import version
        is_openvino = "openvino" in version("vllm")
    except Exception:
        pass

    return is_openvino, "vllm.platforms.openvino.OpenVinoPlatform"


builtin_platform_plugins = {
    'tpu': tpu_platform_plugin,
    'cuda': cuda_platform_plugin,
    'rocm': rocm_platform_plugin,
    'hpu': hpu_platform_plugin,
    'xpu': xpu_platform_plugin,
    'cpu': cpu_platform_plugin,
    'neuron': neuron_platform_plugin,
    'openvino': openvino_platform_plugin,
}


def resolve_current_platform_cls_qualname() -> str:
    platform_plugins = load_plugins_by_group('vllm.platform_plugins')

    activated_plugins = []

    for name, func in chain(builtin_platform_plugins.items(),
                            platform_plugins.items()):
        try:
            assert callable(func)
            is_platform, platform_cls_qualname = func()
            if is_platform:
                activated_plugins.append(name)
        except Exception:
            pass

    activated_builtin_plugins = list(
        set(activated_plugins) & set(builtin_platform_plugins.keys()))
    activated_oot_plugins = list(
        set(activated_plugins) & set(platform_plugins.keys()))

    if len(activated_oot_plugins) >= 2:
        raise RuntimeError(
            "Only one platform plugin can be activated, but got: "
            f"{activated_oot_plugins}")
    elif len(activated_oot_plugins) == 1:
        platform_cls_qualname = platform_plugins[activated_oot_plugins[0]]()[1]
        logger.info("Platform plugin %s is activated",
                    activated_oot_plugins[0])
    elif len(activated_builtin_plugins) >= 2:
        raise RuntimeError(
            "Only one platform plugin can be activated, but got: "
            f"{activated_builtin_plugins}")
    elif len(activated_builtin_plugins) == 1:
        platform_cls_qualname = builtin_platform_plugins[
            activated_builtin_plugins[0]]()[1]
        logger.info("Automatically detected platform %s.",
                    activated_builtin_plugins[0])
    else:
        platform_cls_qualname = "vllm.interface.UnspecifiedPlatform"
        logger.info(
            "No platform detected, vLLM is running on UnspecifiedPlatform")
    return platform_cls_qualname


_current_platform = None
_init_trace: str = ''

if TYPE_CHECKING:
    current_platform: Platform


def __getattr__(name: str):
    if name == 'current_platform':
        # lazy init current_platform so that plugins can import vllm.platforms
        # to inherit Platform without circular imports
        global _current_platform
        if _current_platform is None:
            platform_cls_qualname = resolve_current_platform_cls_qualname()
            _current_platform = resolve_obj_by_qualname(
                platform_cls_qualname)()
            global _init_trace
            _init_trace = "".join(traceback.format_stack())
        return _current_platform
    else:
        return globals()[name]


__all__ = [
    'Platform', 'PlatformEnum', 'current_platform', 'CpuArchEnum',
    "_init_trace"
]
