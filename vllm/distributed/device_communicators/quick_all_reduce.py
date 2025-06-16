# SPDX-License-Identifier: Apache-2.0
import logging
from enum import Enum
from typing import Union

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

import vllm.envs as envs
from vllm import _custom_ops as ops
from vllm.platforms import current_platform

logger = logging.getLogger(__name__)

try:
    ops.qr_max_size()
    ops_available = True
except Exception:
    # For CPUs
    ops_available = False


class QuickReduceRegime(Enum):
    FP = 0
    INT8 = 1
    INT4 = 2
    NONE = 3


class QuickAllReduce:
    _SUPPORTED_WORLD_SIZES = [2, 4, 8]
    _SUPPORTED_DTYPES = [torch.float16, torch.bfloat16]

    def __init__(self, group: ProcessGroup,
                 device: Union[int, str, torch.device]) -> None:
        self.disabled = True
        if not ops_available:
            # disable because of missing quick reduce library
            # e.g. in a non-cuda environment
            logger.info("Custom quick allreduce is disabled because "
                        "of missing custom quick allreduce library")
            return

        self.max_size = ops.qr_max_size()
        self.group = group
        regime_str = envs.VLLM_ROCM_CA_QUANT_REGIME
        assert regime_str in QuickReduceRegime.__members__, (
            f"Invalid quantization level: {regime_str}. "
            "Supported levels: "
            f"{list(QuickReduceRegime.__members__.keys())}")
        if regime_str == "NONE":
            logger.debug("Custom quickreduce is disabled based on "
                         "env variable VLLM_ROCM_CA_QUANT_REGIME")
            return
        self.quant_level = QuickReduceRegime[regime_str]
        # On RocM bfloat16 kernels are slower than fp16
        # due to slower match operations
        # If environment is not set to 1 we convert input to fp16
        self.use_fp16_kernels = envs.VLLM_ROCM_CA_CAST_BF16_TO_FP16
        assert dist.get_backend(group) != dist.Backend.NCCL, (
            "QuickReduce should be attached to a non-NCCL group.")
        rank = dist.get_rank(group=self.group)
        world_size = dist.get_world_size(group=self.group)
        if world_size == 1:
            # No need to initialize QuickReduce for single GPU case.
            return

        if world_size not in QuickAllReduce._SUPPORTED_WORLD_SIZES:
            logger.warning(
                "QuickReduce is disabled due to an unsupported world"
                " size: %d. Supported world sizes: %s."
                " To disable this warning set VLLM_ROCM_CA_BACKEND"
                " to None", world_size,
                str(QuickAllReduce._SUPPORTED_WORLD_SIZES))
            return

        assert current_platform.is_rocm(), (
            "QuickReduce is only supported on ROCm platform.")
        if isinstance(device, int):
            device = torch.device(f"cuda:{device}")
        elif isinstance(device, str):
            device = torch.device(device)
        # now `device` is a `torch.device` object
        assert isinstance(device, torch.device)
        self.device = device
        torch.cuda.set_device(self.device)

        self._ptr = ops.init_custom_qr(rank, world_size)
        self.create_shared_buffer()
        self.disabled = False

    def create_shared_buffer(self):
        """
        Creates a shared buffer for quickreduce. 
        Has to be called after qr_init_device_collectives
        """
        handle = ops.qr_get_handle(self._ptr)
        world_size = dist.get_world_size(group=self.group)
        handles = [None] * world_size
        dist.all_gather_object(handles, handle, group=self.group)
        ops.qr_open_handles(self._ptr, handles)

    def all_reduce(self, inp: torch.Tensor, *, out: torch.Tensor = None):
        """
        Performs an out-of-place all reduce.       
        """
        inp_size = inp.numel() * inp.element_size()
        if inp_size >= self.max_size:
            return None

        inp_dtype = inp.dtype
        if inp_dtype == torch.bfloat16 and self.use_fp16_kernels:
            inp = inp.to(torch.float16)
        if out is None:
            out = torch.empty_like(inp)

        ops.qr_all_reduce(self._ptr, inp, out, self.quant_level.value)
        return out.to(inp_dtype)

    def close(self):
        if not self.disabled and getattr(self, "_ptr", None):
            ops.qr_destroy(self._ptr)
            self._ptr = 0

    def __del__(self):
        self.close()

    def should_quick_allreduce(self, inp: torch.Tensor):
        if self.disabled:
            return False
        inp_size = inp.numel() * inp.element_size()
        # QuickReduce requires input byte size to be multiples of 16
        if inp_size % 16 != 0:
            return False
        return inp.dtype in QuickAllReduce._SUPPORTED_DTYPES and \
            inp_size < self.max_size
