"""Code inside this file can safely assume cuda platform, e.g. importing
pynvml. However, it should not initialize cuda context.
"""

import os
from functools import lru_cache, wraps
from typing import Callable, List, Tuple, TypeVar

import pynvml
from typing_extensions import ParamSpec

from vllm.logger import init_logger

from .interface import DeviceCapability, Platform, PlatformEnum

logger = init_logger(__name__)

_P = ParamSpec("_P")
_R = TypeVar("_R")

if pynvml.__file__.endswith("__init__.py"):
    logger.warning(
        "You are using a deprecated `pynvml` package. Please install"
        " `nvidia-ml-py` instead, and make sure to uninstall `pynvml`."
        " When both of them are installed, `pynvml` will take precedence"
        " and cause errors. See https://pypi.org/project/pynvml "
        "for more information.")

# NVML utils
# Note that NVML is not affected by `CUDA_VISIBLE_DEVICES`,
# all the related functions work on real physical device ids.
# the major benefit of using NVML is that it will not initialize CUDA


def with_nvml_context(fn: Callable[_P, _R]) -> Callable[_P, _R]:

    @wraps(fn)
    def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        pynvml.nvmlInit()
        try:
            return fn(*args, **kwargs)
        finally:
            pynvml.nvmlShutdown()

    return wrapper


@lru_cache(maxsize=8)
@with_nvml_context
def get_physical_device_capability(device_id: int = 0) -> Tuple[int, int]:
    handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
    return pynvml.nvmlDeviceGetCudaComputeCapability(handle)


@lru_cache(maxsize=8)
@with_nvml_context
def get_physical_device_name(device_id: int = 0) -> str:
    handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
    return pynvml.nvmlDeviceGetName(handle)


@lru_cache(maxsize=8)
@with_nvml_context
def get_physical_device_total_memory(device_id: int = 0) -> int:
    handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
    return int(pynvml.nvmlDeviceGetMemoryInfo(handle).total)


@with_nvml_context
def warn_if_different_devices():
    device_ids: int = pynvml.nvmlDeviceGetCount()
    if device_ids > 1:
        device_names = [get_physical_device_name(i) for i in range(device_ids)]
        if len(set(device_names)) > 1 and os.environ.get(
                "CUDA_DEVICE_ORDER") != "PCI_BUS_ID":
            logger.warning(
                "Detected different devices in the system: \n%s\nPlease"
                " make sure to set `CUDA_DEVICE_ORDER=PCI_BUS_ID` to "
                "avoid unexpected behavior.", "\n".join(device_names))


try:
    from sphinx.ext.autodoc.mock import _MockModule

    if not isinstance(pynvml, _MockModule):
        warn_if_different_devices()
except ModuleNotFoundError:
    warn_if_different_devices()


def device_id_to_physical_device_id(device_id: int) -> int:
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        device_ids = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
        if device_ids == [""]:
            raise RuntimeError("CUDA_VISIBLE_DEVICES is set to empty string,"
                               " which means GPU support is disabled.")
        physical_device_id = device_ids[device_id]
        return int(physical_device_id)
    else:
        return device_id


class CudaPlatform(Platform):
    _enum = PlatformEnum.CUDA

    @classmethod
    def get_device_capability(cls, device_id: int = 0) -> DeviceCapability:
        physical_device_id = device_id_to_physical_device_id(device_id)
        major, minor = get_physical_device_capability(physical_device_id)
        return DeviceCapability(major=major, minor=minor)

    @classmethod
    def get_device_name(cls, device_id: int = 0) -> str:
        physical_device_id = device_id_to_physical_device_id(device_id)
        return get_physical_device_name(physical_device_id)

    @classmethod
    def get_device_total_memory(cls, device_id: int = 0) -> int:
        physical_device_id = device_id_to_physical_device_id(device_id)
        return get_physical_device_total_memory(physical_device_id)

    @classmethod
    @with_nvml_context
    def is_full_nvlink(cls, physical_device_ids: List[int]) -> bool:
        """
        query if the set of gpus are fully connected by nvlink (1 hop)
        """
        handles = [
            pynvml.nvmlDeviceGetHandleByIndex(i) for i in physical_device_ids
        ]
        for i, handle in enumerate(handles):
            for j, peer_handle in enumerate(handles):
                if i < j:
                    try:
                        p2p_status = pynvml.nvmlDeviceGetP2PStatus(
                            handle, peer_handle,
                            pynvml.NVML_P2P_CAPS_INDEX_NVLINK)
                        if p2p_status != pynvml.NVML_P2P_STATUS_OK:
                            return False
                    except pynvml.NVMLError as error:
                        logger.error(
                            "NVLink detection failed. This is normal if your"
                            " machine has no NVLink equipped.",
                            exc_info=error)
                        return False
        return True
