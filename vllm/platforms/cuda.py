"""Code inside this file can safely assume cuda platform, e.g. importing
pynvml. However, it should not initialize cuda context.
"""

import os
from functools import lru_cache, wraps
from typing import Callable, List, Tuple, TypeVar

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


class CudaPlatformBase(Platform):
    _enum = PlatformEnum.CUDA

    @classmethod
    def get_device_capability(cls, device_id: int = 0) -> DeviceCapability:
        raise NotImplementedError

    @classmethod
    def get_device_name(cls, device_id: int = 0) -> str:
        raise NotImplementedError

    @classmethod
    def get_device_total_memory(cls, device_id: int = 0) -> int:
        raise NotImplementedError

    @classmethod
    def is_full_nvlink(cls, device_ids: List[int]) -> bool:
        raise NotImplementedError

    @classmethod
    def log_warnings(cls):
        pass


# NVML utils
# Note that NVML is not affected by `CUDA_VISIBLE_DEVICES`,
# all the related functions work on real physical device ids.
# the major benefit of using NVML is that it will not initialize CUDA
class NvmlCudaPlatform(CudaPlatformBase):

    @staticmethod
    def with_nvml_context(fn: Callable[_P, _R]) -> Callable[_P, _R]:

        @wraps(fn)
        def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
            pynvml.nvmlInit()
            try:
                return fn(*args, **kwargs)
            finally:
                pynvml.nvmlShutdown()

        return wrapper

    @classmethod
    def get_device_capability(cls, device_id: int = 0) -> DeviceCapability:
        physical_device_id = device_id_to_physical_device_id(device_id)
        major, minor = cls._get_physical_device_capability(physical_device_id)
        return DeviceCapability(major=major, minor=minor)

    @classmethod
    def get_device_name(cls, device_id: int = 0) -> str:
        physical_device_id = device_id_to_physical_device_id(device_id)
        return cls._get_physical_device_name(physical_device_id)

    @classmethod
    def get_device_total_memory(cls, device_id: int = 0) -> int:
        physical_device_id = device_id_to_physical_device_id(device_id)
        return cls._get_physical_device_total_memory(physical_device_id)

    @classmethod
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
                            handle,
                            peer_handle,
                            pynvml.NVML_P2P_CAPS_INDEX_NVLINK,
                        )
                        if p2p_status != pynvml.NVML_P2P_STATUS_OK:
                            return False
                    except pynvml.NVMLError:
                        logger.exception(
                            "NVLink detection failed. This is normal if"
                            " your machine has no NVLink equipped.")
                        return False
        return True

    @classmethod
    @lru_cache(maxsize=8)
    @with_nvml_context
    def _get_physical_device_capability(cls,
                                        device_id: int = 0) -> Tuple[int, int]:
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
        return pynvml.nvmlDeviceGetCudaComputeCapability(handle)

    @classmethod
    @lru_cache(maxsize=8)
    @with_nvml_context
    def _get_physical_device_name(cls, device_id: int = 0) -> str:
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
        return pynvml.nvmlDeviceGetName(handle)

    @classmethod
    @lru_cache(maxsize=8)
    @with_nvml_context
    def _get_physical_device_total_memory(cls, device_id: int = 0) -> int:
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
        return int(pynvml.nvmlDeviceGetMemoryInfo(handle).total)

    @classmethod
    @with_nvml_context
    def log_warnings(cls):
        device_ids: int = pynvml.nvmlDeviceGetCount()
        if device_ids > 1:
            device_names = [
                cls._get_physical_device_name(i) for i in range(device_ids)
            ]
            if (len(set(device_names)) > 1
                    and os.environ.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID"):
                logger.warning(
                    "Detected different devices in the system: \n%s\nPlease"
                    " make sure to set `CUDA_DEVICE_ORDER=PCI_BUS_ID` to "
                    "avoid unexpected behavior.",
                    "\n".join(device_names),
                )


class NonNvmlCudaPlatform(CudaPlatformBase):

    @classmethod
    def get_device_capability(cls, device_id: int = 0) -> DeviceCapability:
        major, minor = torch.cuda.get_device_capability(device_id)
        return DeviceCapability(major=major, minor=minor)

    @classmethod
    def get_device_name(cls, device_id: int = 0) -> str:
        return torch.cuda.get_device_name(device_id)

    @classmethod
    def get_device_total_memory(cls, device_id: int = 0) -> int:
        device_props = torch.cuda.get_device_properties(device_id)
        return device_props.total_memory

    @classmethod
    def is_full_nvlink(cls, physical_device_ids: List[int]) -> bool:
        logger.exception(
            "NVLink detection not possible, as context support was"
            " not found. Assuming no NVLink available.")
        return False


# Autodetect either NVML-enabled or non-NVML platform
# based on whether NVML is available.
nvml_available = False
try:
    try:
        pynvml.nvmlInit()
        nvml_available = True
    except Exception:
        # On Jetson, NVML is not supported.
        nvml_available = False
finally:
    if nvml_available:
        pynvml.nvmlShutdown()

CudaPlatform = NvmlCudaPlatform if nvml_available else NonNvmlCudaPlatform

try:
    from sphinx.ext.autodoc.mock import _MockModule

    if not isinstance(pynvml, _MockModule):
        CudaPlatform.log_warnings()
except ModuleNotFoundError:
    CudaPlatform.log_warnings()
