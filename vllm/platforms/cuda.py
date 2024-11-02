"""Code inside this file can safely assume cuda platform, e.g. importing
pynvml. However, it should not initialize cuda context.
"""

import os
from collections.abc import Iterator
from contextlib import contextmanager
from functools import lru_cache
from typing import List, Optional, Tuple, TypeVar

import pynvml
import torch
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

# pytorch 2.5 uses cudnn sdpa by default, which will cause crash on some models
# see https://github.com/huggingface/diffusers/issues/9704 for details
torch.backends.cuda.enable_cudnn_sdp(False)

# NVML utils
# Note that NVML is not affected by `CUDA_VISIBLE_DEVICES`,
# all the related functions work on real physical device ids.
# the major benefit of using NVML is that it will not initialize CUDA


class NVMLContext:

    @classmethod
    @lru_cache(maxsize=8)
    def get_physical_device_capability(cls,
                                       device_id: int = 0) -> Tuple[int, int]:
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
        return pynvml.nvmlDeviceGetCudaComputeCapability(handle)

    @classmethod
    @lru_cache(maxsize=8)
    def get_physical_device_name(cls, device_id: int = 0) -> str:
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
        return pynvml.nvmlDeviceGetName(handle)

    @classmethod
    @lru_cache(maxsize=8)
    def get_physical_device_total_memory(cls, device_id: int = 0) -> int:
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
        return int(pynvml.nvmlDeviceGetMemoryInfo(handle).total)

    @classmethod
    def warn_if_different_devices(cls):
        device_ids: int = pynvml.nvmlDeviceGetCount()
        if device_ids > 1:
            device_names = [
                cls.get_physical_device_name(i) for i in range(device_ids)
            ]
            if len(set(device_names)) > 1 and os.environ.get(
                    "CUDA_DEVICE_ORDER") != "PCI_BUS_ID":
                logger.warning(
                    "Detected different devices in the system: \n%s\nPlease"
                    " make sure to set `CUDA_DEVICE_ORDER=PCI_BUS_ID` to "
                    "avoid unexpected behavior.", "\n".join(device_names))


@contextmanager
def nvml_context() -> Iterator[Optional[NVMLContext]]:
    nvml_init_ok = False
    try:
        try:
            pynvml.nvmlInit()
            nvml_init_ok = True
            yield NVMLContext()
        except Exception:
            # On Jetson, NVML is not supported.
            yield None
    finally:
        if nvml_init_ok:
            pynvml.nvmlShutdown()


try:
    from sphinx.ext.autodoc.mock import _MockModule

    if not isinstance(pynvml, _MockModule):
        with nvml_context() as nvml:
            if nvml:
                nvml.warn_if_different_devices()
except ModuleNotFoundError:
    with nvml_context() as nvml:
        if nvml:
            nvml.warn_if_different_devices()


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
        with nvml_context() as nvml:
            if nvml:
                physical_device_id = device_id_to_physical_device_id(device_id)
                major, minor = nvml.get_physical_device_capability(
                    physical_device_id)
            else:
                major, minor = torch.cuda.get_device_capability(device_id)
        return DeviceCapability(major=major, minor=minor)

    @classmethod
    def get_device_name(cls, device_id: int = 0) -> str:
        with nvml_context() as nvml:
            if nvml:
                physical_device_id = device_id_to_physical_device_id(device_id)
                return nvml.get_physical_device_name(physical_device_id)
            else:
                return torch.cuda.get_device_name(device_id)

    @classmethod
    def get_device_total_memory(cls, device_id: int = 0) -> int:
        with nvml_context() as nvml:
            if nvml:
                physical_device_id = device_id_to_physical_device_id(device_id)
                return nvml.get_physical_device_total_memory(
                    physical_device_id)
            else:
                device_props = torch.cuda.get_device_properties(device_id)
                return device_props.total_memory

    @classmethod
    def is_full_nvlink(cls, physical_device_ids: List[int]) -> bool:
        """
        query if the set of gpus are fully connected by nvlink (1 hop)
        """
        with nvml_context() as nvml:
            if nvml:
                handles = [
                    pynvml.nvmlDeviceGetHandleByIndex(i)
                    for i in physical_device_ids
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
                            except pynvml.NVMLError:
                                logger.exception(
                                    "NVLink detection failed. This is normal if"
                                    " your machine has no NVLink equipped.")
                                return False
                return True
            else:
                logger.exception(
                    "NVLink detection not possible, as NVML support was"
                    " not found. Assuming no NVLink available.")
                return False
