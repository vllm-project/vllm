import logging
import traceback
from itertools import chain
from typing import TYPE_CHECKING, Optional

from vllm.plugins import load_plugins_by_group
from vllm.utils import resolve_obj_by_qualname

from .interface import _Backend  # noqa: F401
from .interface import CpuArchEnum, Platform, PlatformEnum

logger = logging.getLogger(__name__)


def tpu_platform_plugin() -> Optional[str]:
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

    return "vllm.platforms.tpu.TpuPlatform" if is_tpu else None


def cuda_platform_plugin() -> Optional[str]:
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

    return "vllm.platforms.cuda.CudaPlatform" if is_cuda else None


def rocm_platform_plugin() -> Optional[str]:
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

    return "vllm.platforms.rocm.RocmPlatform" if is_rocm else None


def hpu_platform_plugin() -> Optional[str]:
    is_hpu = False
    try:
        from importlib import util
        is_hpu = util.find_spec('habana_frameworks') is not None
    except Exception:
        pass

    return "vllm.platforms.hpu.HpuPlatform" if is_hpu else None


def xpu_platform_plugin() -> Optional[str]:
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

    return "vllm.platforms.xpu.XPUPlatform" if is_xpu else None


def cpu_platform_plugin() -> Optional[str]:
    is_cpu = False
    try:
        from importlib.metadata import version
        is_cpu = "cpu" in version("vllm")
    except Exception:
        pass

    return "vllm.platforms.cpu.CpuPlatform" if is_cpu else None


def neuron_platform_plugin() -> Optional[str]:
    is_neuron = False
    try:
        import transformers_neuronx  # noqa: F401
        is_neuron = True
    except ImportError:
        pass

    return "vllm.platforms.neuron.NeuronPlatform" if is_neuron else None


def openvino_platform_plugin() -> Optional[str]:
    is_openvino = False
    try:
        from importlib.metadata import version
        is_openvino = "openvino" in version("vllm")
    except Exception:
        pass

    return "vllm.platforms.openvino.OpenVinoPlatform" if is_openvino else None


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
            platform_cls_qualname = func()
            if platform_cls_qualname is not None:
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
        platform_cls_qualname = platform_plugins[activated_oot_plugins[0]]()
        logger.info("Platform plugin %s is activated",
                    activated_oot_plugins[0])
    elif len(activated_builtin_plugins) >= 2:
        raise RuntimeError(
            "Only one platform plugin can be activated, but got: "
            f"{activated_builtin_plugins}")
    elif len(activated_builtin_plugins) == 1:
        platform_cls_qualname = builtin_platform_plugins[
            activated_builtin_plugins[0]]()
        logger.info("Automatically detected platform %s.",
                    activated_builtin_plugins[0])
    else:
        platform_cls_qualname = "vllm.platforms.interface.UnspecifiedPlatform"
        logger.info(
            "No platform detected, vLLM is running on UnspecifiedPlatform")
    return platform_cls_qualname


_current_platform = None
_init_trace: str = ''

if TYPE_CHECKING:
    current_platform: Platform


def __getattr__(name: str):
    if name == 'current_platform':
        # lazy init current_platform.
        # 1. out-of-tree platform plugins need `from vllm.platforms import
        #    Platform` so that they can inherit `Platform` class. Therefore,
        #    we cannot resolve `current_platform` during the import of
        #    `vllm.platforms`.
        # 2. when users use out-of-tree platform plugins, they might run
        #    `import vllm`, some vllm internal code might access
        #    `current_platform` during the import, and we need to make sure
        #    `current_platform` is only resolved after the plugins are loaded
        #    (we have tests for this, if any developer violate this, they will
        #    see the test failures).
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
