# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import contextmanager
from enum import Enum
from typing import Optional, Union

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

import vllm.envs as envs
from vllm import _custom_ops as ops
from vllm.config import get_current_vllm_config
from vllm.distributed.device_communicators.custom_all_reduce_utils import (
    gpu_p2p_access_check)
from vllm.distributed.parallel_state import in_the_same_node_as
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils import cuda_device_count_stateless

try:
    ops.meta_size()
    custom_ar = True
except Exception:
    # For CPUs
    custom_ar = False
try:
    ops.qr_max_size()
    quick_ar = True
except Exception:
    # For CPUs
    quick_ar = False

logger = init_logger(__name__)


class QuickReduceRegime(Enum):
    FP = 0
    INT8 = 1
    INT6 = 2
    INT4 = 3
    NONE = 4


def _can_p2p(rank: int, world_size: int) -> bool:
    for i in range(world_size):
        if i == rank:
            continue
        if envs.VLLM_SKIP_P2P_CHECK:
            logger.info(
                "Skipping P2P check and trusting the driver's P2P report.")
            return torch.cuda.can_device_access_peer(rank, i)
        if not gpu_p2p_access_check(rank, i):
            return False
    return True


def is_weak_contiguous(inp: torch.Tensor):
    return inp.is_contiguous() or (inp.storage().nbytes() -
                                   inp.storage_offset() * inp.element_size()
                                   == inp.numel() * inp.element_size())


class CustomAllreduce:

    _SUPPORTED_WORLD_SIZES = [2, 4, 6, 8]
    _QR_SUPPORTED_WORLD_SIZES = [2, 4, 8]
    _QR_SUPPORTED_DTYPES = [torch.float16, torch.bfloat16]

    # TODO: We should set a reasonable range for FP.
    MB = 1024 * 1024
    _QR_MIN_SIZE = {
        (torch.float16, 2): [16 * MB, 2 * MB, 2 * MB, 1 * MB],
        (torch.float16, 4): [16 * MB, 64 * MB, 4 * MB, 2 * MB],
        (torch.float16, 8): [16 * MB, 4 * MB, 4 * MB, 2 * MB],
        (torch.bfloat16, 2): [16 * MB, 8 * MB, 8 * MB, 8 * MB],
        (torch.bfloat16, 4): [16 * MB, 128 * MB, 128 * MB, 16 * MB],
        (torch.bfloat16, 8): [16 * MB, 2048 * MB, 2048 * MB, 2048 * MB],
    }

    # max_size: max supported allreduce size
    def __init__(self,
                 group: ProcessGroup,
                 device: Union[int, str, torch.device],
                 cr_max_size=8192 * 1024) -> None:
        """
        Custom allredcue (cr) is non-destructive acceleration, which is
        available for cuda and rocm MI300 series.
        Custom quick allreduce (qr) is accelerated by quantization, 
        currently supports fp16, Q8, Q6, Q4 quantization. 
        We view qr as complementary to cr, the condition for qr is 
        even more demanding; qr is initialized, then cr must also 
        be initialized. If the conditions of cr are not met, qr is 
        naturally not initialized.
        Due to instruction set limitations, only rocm MI300 series
        is supported for the time being.
        Args:
            group: the process group to work on. If None, it will use the
                default process group.
            device: the device to bind the CustomAllreduce to. If None,
                it will be bind to f"cuda:{local_rank}".
            cr_max_size: max supported size of cr.
        It is the caller's responsibility to make sure each communicator
        is bind to a unique device, and all communicators in this group
        are in the same node.
        """
        self._QR_SHOULD_INIT = True
        self._IS_CAPTURING = False
        self.disabled = True
        self.qr_disabled = True
        self.cr_max_size = cr_max_size
        self.qr_max_size = ops.qr_max_size()

        if not custom_ar:
            # disable because of missing custom allreduce library
            # e.g. in a non-GPU environment
            logger.info("Custom allreduce is disabled because "
                        "of missing custom allreduce library")
        if not quick_ar:
            logger.info("Custom quick allreduce is disabled because "
                        "of missing quick allreduce library")
            self._QR_SHOULD_INIT = False

        if not quick_ar and not custom_ar:
            return
        self.group = group

        assert dist.get_backend(group) != dist.Backend.NCCL, (
            "CustomAllreduce should be attached to a non-NCCL group.")

        if not all(in_the_same_node_as(group, source_rank=0)):
            # No need to initialize custom allreduce for multi-node case.
            logger.warning(
                "Custom allreduce is disabled because this process group"
                " spans across nodes.")
            return

        rank = dist.get_rank(group=self.group)
        world_size = dist.get_world_size(group=self.group)
        self.rank = rank
        self.world_size = world_size
        if world_size == 1:
            # No need to initialize custom allreduce or custom quick
            # allreduce for single GPU case.
            return

        if world_size not in CustomAllreduce._SUPPORTED_WORLD_SIZES:
            logger.warning(
                "Custom allreduce is disabled due to an unsupported world"
                " size: %d. Supported world sizes: %s. To silence this "
                "warning, specify disable_custom_all_reduce=True explicitly.",
                world_size, str(CustomAllreduce._SUPPORTED_WORLD_SIZES))
            return

        if self._QR_SHOULD_INIT and \
            world_size not in CustomAllreduce._QR_SUPPORTED_WORLD_SIZES:
            self._QR_SHOULD_INIT = False
            logger.warning(
                "Custom quick allreduce is disabled due to an unsupported "
                "world size: %d.", world_size)

        if isinstance(device, int):
            device = torch.device(f"cuda:{device}")
        elif isinstance(device, str):
            device = torch.device(device)
        # now `device` is a `torch.device` object
        assert isinstance(device, torch.device)
        self.device = device

        cuda_visible_devices = envs.CUDA_VISIBLE_DEVICES
        if cuda_visible_devices:
            device_ids = list(map(int, cuda_visible_devices.split(",")))
        else:
            device_ids = list(range(cuda_device_count_stateless()))

        physical_device_id = device_ids[device.index]
        tensor = torch.tensor([physical_device_id],
                              dtype=torch.int,
                              device="cpu")
        gather_list = [
            torch.tensor([0], dtype=torch.int, device="cpu")
            for _ in range(world_size)
        ]
        dist.all_gather(gather_list, tensor, group=self.group)
        physical_device_ids = [t.item() for t in gather_list]

        # test nvlink first, this will filter out most of the cases
        # where custom allreduce is not supported
        # this checks hardware and driver support for NVLink
        assert current_platform.is_cuda_alike()
        self.fully_connected = current_platform.is_fully_connected(
            physical_device_ids)
        if world_size > 2 and not self.fully_connected:
            logger.warning(
                "Custom allreduce is disabled because it's not supported on"
                " more than two PCIe-only GPUs. To silence this warning, "
                "specify disable_custom_all_reduce=True explicitly.")
            return
        if not current_platform.is_rocm():
            # First, we only enable quickreduce for MI300 series,
            # If it's rocm then it must be MI300 series because cr is only
            # available on the mi300 series, qr must be available.
            self._QR_SHOULD_INIT = False

            # test P2P capability, this checks software/cudaruntime support
            # this is expensive to compute at the first time
            # then we cache the result
            # On AMD GPU, p2p is always enabled between XGMI connected GPUs
            if not _can_p2p(rank, world_size):
                logger.warning(
                    "Custom allreduce is disabled because your platform lacks "
                    "GPU P2P capability or P2P test failed. To silence this "
                    "warning, specify disable_custom_all_reduce=True "
                    "explicitly.")
                return

        self.init_custom_allreduce()
        # self.disabled is used to indicate cr, if the condition
        # of cr is not satisfied, qr must not be satisfied,
        # This boolean serves as a uniform identifier for external.
        self.disabled = False
        self.init_custom_quick_allreduce()

    def init_custom_allreduce(self):
        """
        Initialize custom allreduce
        """
        # Buffers memory are owned by this Python class and passed to C++.
        # Meta data composes of two parts: meta data for synchronization and a
        # temporary buffer for storing intermediate allreduce results.
        self.meta_ptrs = self.create_shared_buffer(ops.meta_size() +
                                                   self.cr_max_size,
                                                   group=self.group,
                                                   uncached=True)
        # This is a pre-registered IPC buffer. In eager mode, input tensors
        # are first copied into this buffer before allreduce is performed
        self.buffer_ptrs = self.create_shared_buffer(self.cr_max_size,
                                                     group=self.group)
        # This is a buffer for storing the tuples of pointers pointing to
        # IPC buffers from all ranks. Each registered tuple has size of
        # 8*world_size bytes where world_size is at most 8. Allocating 8MB
        # is enough for 131072 such tuples. The largest model I've seen only
        # needs less than 10000 of registered tuples.
        self.rank_data = torch.empty(8 * 1024 * 1024,
                                     dtype=torch.uint8,
                                     device=self.device)
        self.cr_max_size = self.cr_max_size

        self._cr_ptr = ops.init_custom_ar(self.meta_ptrs, self.rank_data,
                                          self.rank, self.fully_connected)
        ops.register_buffer(self._cr_ptr, self.buffer_ptrs)

    def init_custom_quick_allreduce(self):
        """
        Initialize a custom quick allreduce implementation for AMD
        based on quick reduce (https://github.com/mk1-project/quickreduce).
        """
        vllm_config = get_current_vllm_config()
        dtype = vllm_config.model_config.dtype
        if dtype not in [torch.float16, torch.bfloat16]:
            self._QR_SHOULD_INIT = False
        # On RocM bfloat16 kernels are slower than fp16
        # due to slower match operations
        # If environment is not set to 1 we convert input to fp16
        self.use_fp16_kernels: bool = envs.VLLM_ROCM_QR_CAST_BF16_TO_FP16
        regime_str = envs.VLLM_ROCM_QR_QUANT_REGIME
        if self._QR_SHOULD_INIT:
            if regime_str not in QuickReduceRegime.__members__:
                logger.warning(
                    "Custom quick allreduce:",
                    f"Invalid quantization level: {regime_str}. "
                    "Supported levels: "
                    f"{list(QuickReduceRegime.__members__.keys())}")
                return

            if regime_str == "NONE":
                logger.debug("Custom quick allreduce is disabled based "
                             "on env variable VLLM_ROCM_QR_QUANT_REGIME")
                return

            self.qr_quant_level = QuickReduceRegime[regime_str]
            self._qr_ptr = ops.init_custom_qr(self.rank, self.world_size)
            self.create_qr_shared_buffer()
            if dtype == torch.bfloat16 and not self.use_fp16_kernels:
                logger.info(
                    "Custom quick allreduce: converting bf16 to fp16 "
                    "can speed up qr, "
                    "set envs.VLLM_ROCM_QR_CAST_BF16_TO_FP16=1 to turn on.")
            self.qr_disabled = False

    @contextmanager
    def capture(self):
        """
        The main responsibility of this context manager is the 
        `register_graph_buffers` call at the end of the context.
        It records all the buffer addresses used in the CUDA graph.
        """
        try:
            self._IS_CAPTURING = True
            yield
        finally:
            self._IS_CAPTURING = False
            if not self.disabled:
                self.register_graph_buffers()

    def register_graph_buffers(self):
        handle, offset = ops.get_graph_buffer_ipc_meta(self._cr_ptr)
        logger.info("Registering %d cuda graph addresses", len(offset))
        # We cannot directly use `dist.all_gather_object` here
        # because it is incompatible with `gloo` backend under inference mode.
        # see https://github.com/pytorch/pytorch/issues/126032 for details.
        all_data = [[None, None]
                    for _ in range(dist.get_world_size(group=self.group))]
        all_data[self.rank] = [handle, offset]
        ranks = sorted(dist.get_process_group_ranks(group=self.group))
        for i, rank in enumerate(ranks):
            dist.broadcast_object_list(all_data[i],
                                       src=rank,
                                       group=self.group,
                                       device="cpu")
        # Unpack list of tuples to tuple of lists.
        handles = [d[0] for d in all_data]  # type: ignore
        offsets = [d[1] for d in all_data]  # type: ignore
        ops.register_graph_buffers(self._cr_ptr, handles, offsets)

    def should_quick_allreduce(self, inp: torch.Tensor):
        """
        Check if quickreduce is available
        """
        if self.qr_disabled:
            return False
        inp_size = inp.numel() * inp.element_size()
        # custom quick allreduce requires input byte size to be
        # multiples of 16
        if inp_size % 16 != 0:
            return False
        if not is_weak_contiguous(inp):
            return False
        # custom quick allreduce requires input byte size to be multiples of 16
        dtype = inp.dtype
        if self.use_fp16_kernels:
            dtype = torch.float16
        return inp_size <= self.qr_max_size and \
            inp_size > self._QR_MIN_SIZE[(dtype, self.world_size)]\
                [self.qr_quant_level.value]

    def should_custom_allreduce(self, inp: torch.Tensor):
        if self.disabled:
            return False
        inp_size = inp.numel() * inp.element_size()
        # custom allreduce requires input byte size to be multiples of 16
        if inp_size % 16 != 0:
            return False
        if not is_weak_contiguous(inp):
            return False
        # for 4 or more non NVLink-capable GPUs, custom allreduce provides
        # little performance improvement over NCCL.
        if self.world_size == 2 or self.fully_connected:
            return inp_size < self.cr_max_size
        return False

    def should_custom_ar(self, inp: torch.Tensor):
        # Determine whether to use qr, or cr or quit
        return self.should_quick_allreduce(
            inp) or self.should_custom_allreduce(inp)

    def cr_all_reduce(self,
                      inp: torch.Tensor,
                      *,
                      out: torch.Tensor = None,
                      registered: bool = False):
        """Performs an out-of-place custom all reduce.
        
        If registered is True, this assumes inp's pointer is already
        IPC-registered. Otherwise, inp is first copied into a pre-registered
        buffer.
        """
        if out is None:
            out = torch.empty_like(inp)
        if registered:
            ops.all_reduce(self._cr_ptr, inp, out, 0, 0)
        else:
            ops.all_reduce(self._cr_ptr, inp, out, self.buffer_ptrs[self.rank],
                           self.cr_max_size)
        return out

    def qr_all_reduce(self, inp: torch.Tensor, *, out: torch.Tensor = None):
        """Performs an out-of-place custom quick all reduce."""
        inp_dtype = inp.dtype
        if inp_dtype == torch.bfloat16 and self.use_fp16_kernels:
            inp = inp.to(torch.float16)
        if out is None:
            out = torch.empty_like(inp)
        ops.qr_all_reduce(self._qr_ptr, inp, out, self.qr_quant_level.value)
        return out.to(inp_dtype)

    def custom_all_reduce(self, input: torch.Tensor) -> Optional[torch.Tensor]:
        """The main allreduce API that provides support for cuda graph."""
        # try custom quick allreduce first, then custom allreduce
        if self.should_quick_allreduce(input):
            # We don't need the context of quick allreduce to do graph capture
            # because the ipc access is already collected in init() and
            # we can capture the quick allreduce directly.
            return self.qr_all_reduce(input)

        if self.should_custom_allreduce(input):
            if self._IS_CAPTURING:
                if torch.cuda.is_current_stream_capturing():
                    return self.cr_all_reduce(input, registered=True)
                else:
                    # If warm up, mimic the allocation pattern since custom
                    # allreduce is out-of-place.
                    return torch.empty_like(input)
            else:
                # Note: outside of cuda graph context, custom allreduce
                # incurs a cost of cudaMemcpy, which should be small
                # (<=1% of overall latency) compared to the performance
                # gain of using custom kernels
                return self.cr_all_reduce(input, registered=False)

        return None

    def close(self):
        if not self.disabled and self._cr_ptr:
            if ops is not None:
                ops.dispose(self._cr_ptr)
            self._cr_ptr = 0
            self.free_shared_buffer(self.meta_ptrs, rank=self.rank)
            self.free_shared_buffer(self.buffer_ptrs, rank=self.rank)
            self.disabled = True
        if not self.qr_disabled and self._qr_ptr:
            if ops is not None:
                ops.qr_destroy(self._qr_ptr)
            self._qr_ptr = 0
            self.qr_disabled = True

    def __del__(self):
        self.close()

    @staticmethod
    def create_shared_buffer(size_in_bytes: int,
                             group: Optional[ProcessGroup] = None,
                             uncached: Optional[bool] = False) -> list[int]:
        pointer, handle = ops.allocate_shared_buffer_and_handle(size_in_bytes)

        world_size = dist.get_world_size(group=group)
        rank = dist.get_rank(group=group)
        handles = [None] * world_size
        dist.all_gather_object(handles, handle, group=group)

        pointers: list[int] = []
        for i, h in enumerate(handles):
            if i == rank:
                pointers.append(pointer)  # type: ignore
            else:
                pointers.append(ops.open_mem_handle(h))
        return pointers

    def create_qr_shared_buffer(self):
        """
        Creates a shared buffer for quickreduce. 
        Has to be called after qr_init_device_collectives
        """
        handle = ops.qr_get_handle(self._qr_ptr)
        world_size = dist.get_world_size(group=self.group)
        handles = [None] * world_size
        dist.all_gather_object(handles, handle, group=self.group)
        ops.qr_open_handles(self._qr_ptr, handles)

    @staticmethod
    def free_shared_buffer(pointers: list[int],
                           group: Optional[ProcessGroup] = None,
                           rank: Optional[int] = 0) -> None:
        if rank is None:
            rank = dist.get_rank(group=group)
        if ops is not None:
            ops.free_shared_buffer(pointers[rank])
