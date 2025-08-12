# SPDX-License-Identifier: Apache-2.0

import logging
import traceback
from contextlib import suppress
from itertools import chain
from typing import TYPE_CHECKING, Optional

from vllm.plugins import load_plugins_by_group
from vllm.utils import resolve_obj_by_qualname

from .interface import _Backend  # noqa: F401
from .interface import CpuArchEnum, Platform, PlatformEnum

logger = logging.getLogger(__name__)


def vllm_version_matches_substr(substr: str) -> bool:
    """
    Check to see if the vLLM version matches a substring.
    """
    from importlib.metadata import PackageNotFoundError, version
    try:
        vllm_version = version("vllm")
    except PackageNotFoundError as e:
        logger.warning(
            "The vLLM package was not found, so its version could not be "
            "inspected. This may cause platform detection to fail.")
        raise e
    return substr in vllm_version


def tpu_platform_plugin() -> Optional[str]:
    is_tpu = False
    logger.debug("Checking if TPU platform is available.")
    try:
        # While it's technically possible to install libtpu on a
        # non-TPU machine, this is a very uncommon scenario. Therefore,
        # we assume that libtpu is installed if and only if the machine
        # has TPUs.
        import libtpu  # noqa: F401
        is_tpu = True
        logger.debug("Confirmed TPU platform is available.")
    except Exception as e:
        logger.debug("TPU platform is not available because: %s", str(e))
        pass

    return "vllm.platforms.tpu.TpuPlatform" if is_tpu else None


def cuda_platform_plugin() -> Optional[str]:
    is_cuda = False
    logger.debug("Checking if CUDA platform is available.")
    try:
        from vllm.utils import import_pynvml
        pynvml = import_pynvml()
        pynvml.nvmlInit()
        try:
            # NOTE: Edge case: vllm cpu build on a GPU machine.
            # Third-party pynvml can be imported in cpu build,
            # we need to check if vllm is built with cpu too.
            # Otherwise, vllm will always activate cuda plugin
            # on a GPU machine, even if in a cpu build.
            is_cuda = (pynvml.nvmlDeviceGetCount() > 0
                       and not vllm_version_matches_substr("cpu"))
            if pynvml.nvmlDeviceGetCount() <= 0:
                logger.debug(
                    "CUDA platform is not available because no GPU is found.")
            if vllm_version_matches_substr("cpu"):
                logger.debug("CUDA platform is not available because"
                             " vLLM is built with CPU.")
            if is_cuda:
                logger.debug("Confirmed CUDA platform is available.")
        finally:
            pynvml.nvmlShutdown()
    except Exception as e:
        logger.debug("Exception happens when checking CUDA platform: %s",
                     str(e))
        if "nvml" not in e.__class__.__name__.lower():
            # If the error is not related to NVML, re-raise it.
            raise e

        # CUDA is supported on Jetson, but NVML may not be.
        import os

        def cuda_is_jetson() -> bool:
            return os.path.isfile("/etc/nv_tegra_release") \
                or os.path.exists("/sys/class/tegra-firmware")

        if cuda_is_jetson():
            logger.debug("Confirmed CUDA platform is available on Jetson.")
            is_cuda = True
        else:
            logger.debug("CUDA platform is not available because: %s", str(e))

    return "vllm.platforms.cuda.CudaPlatform" if is_cuda else None


def rocm_platform_plugin() -> Optional[str]:
    is_rocm = False
    logger.debug("Checking if ROCm platform is available.")
    try:
        import amdsmi
        amdsmi.amdsmi_init()
        try:
            if len(amdsmi.amdsmi_get_processor_handles()) > 0:
                is_rocm = True
                logger.debug("Confirmed ROCm platform is available.")
            else:
                logger.debug("ROCm platform is not available because"
                             " no GPU is found.")
        finally:
            amdsmi.amdsmi_shut_down()
    except Exception as e:
        logger.debug("ROCm platform is not available because: %s", str(e))
        pass

    return "vllm.platforms.rocm.RocmPlatform" if is_rocm else None


def hpu_platform_plugin() -> Optional[str]:
    is_hpu = False
    logger.debug("Checking if HPU platform is available.")
    try:
        from importlib import util
        is_hpu = util.find_spec('habana_frameworks') is not None
        if is_hpu:
            logger.debug("Confirmed HPU platform is available.")
        else:
            logger.debug("HPU platform is not available because "
                         "habana_frameworks is not found.")
    except Exception as e:
        logger.debug("HPU platform is not available because: %s", str(e))
        pass

    return "vllm.platforms.hpu.HpuPlatform" if is_hpu else None


def xpu_platform_plugin() -> Optional[str]:
    is_xpu = False
    logger.debug("Checking if XPU platform is available.")
    try:
        # installed IPEX if the machine has XPUs.
        import intel_extension_for_pytorch  # noqa: F401
        import oneccl_bindings_for_pytorch  # noqa: F401
        import torch
        if hasattr(torch, 'xpu') and torch.xpu.is_available():
            is_xpu = True
            logger.debug("Confirmed XPU platform is available.")
    except Exception as e:
        logger.debug("XPU platform is not available because: %s", str(e))
        pass

    return "vllm.platforms.xpu.XPUPlatform" if is_xpu else None


def cpu_platform_plugin() -> Optional[str]:
    is_cpu = False
    logger.debug("Checking if CPU platform is available.")
    try:
        is_cpu = vllm_version_matches_substr("cpu")
        if is_cpu:
            logger.debug("Confirmed CPU platform is available because"
                         " vLLM is built with CPU.")
        if not is_cpu:
            import sys
            is_cpu = sys.platform.startswith("darwin")
            if is_cpu:
                logger.debug("Confirmed CPU platform is available"
                             " because the machine is MacOS.")

    except Exception as e:
        logger.debug("CPU platform is not available because: %s", str(e))
        pass

    return "vllm.platforms.cpu.CpuPlatform" if is_cpu else None


def neuron_platform_plugin() -> Optional[str]:
    is_neuron = False
    logger.debug("Checking if Neuron platform is available.")
    try:
        import transformers_neuronx  # noqa: F401
        is_neuron = True
        logger.debug("Confirmed Neuron platform is available because"
                     " transformers_neuronx is found.")
    except ImportError as e:
        logger.debug("Neuron platform is not available because: %s", str(e))
        pass

    return "vllm.platforms.neuron.NeuronPlatform" if is_neuron else None


def openvino_platform_plugin() -> Optional[str]:
    is_openvino = False
    logger.debug("Checking if OpenVINO platform is available.")
    with suppress(Exception):
        is_openvino = vllm_version_matches_substr("openvino")
        if is_openvino:
            logger.debug("Confirmed OpenVINO platform is available"
                         " because vLLM is built with OpenVINO.")
    if not is_openvino:
        logger.debug("OpenVINO platform is not available because"
                     " vLLM is not built with OpenVINO.")

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
    elif name in globals():
        return globals()[name]
    else:
        raise AttributeError(
            f"No attribute named '{name}' exists in {__name__}.")


__all__ = [
    'Platform', 'PlatformEnum', 'current_platform', 'CpuArchEnum',
    "_init_trace"
]
