from contextlib import contextmanager
import pynvml
import torch
import torch.distributed as dist
from typing import Optional

from vllm._C import fast_ar
from vllm.logger import init_logger
from vllm.model_executor.parallel_utils.parallel_state import (
    get_tensor_model_parallel_world_size, get_tensor_model_parallel_rank)

logger = init_logger(__name__)

_FA_HANDLE = None
_IS_CAPTURING = False


def init_fast_ar() -> None:
    global _FA_HANDLE
    world_size = get_tensor_model_parallel_world_size()
    if world_size > 1:
        _FA_HANDLE = FastAllreduce(get_tensor_model_parallel_rank(),
                                   world_size)


def begin_capture() -> None:
    global _IS_CAPTURING
    _IS_CAPTURING = True


def end_capture() -> None:
    global _IS_CAPTURING
    _IS_CAPTURING = False


def is_capturing() -> bool:
    return _IS_CAPTURING and _FA_HANDLE is not None


def get_handle() -> Optional["FastAllreduce"]:
    return _FA_HANDLE


@contextmanager
def capture(enable: bool):
    if enable:
        init_fast_ar()
    try:
        begin_capture()
        yield
    finally:
        end_capture()
        if enable:
            get_handle().register_graph_buffers()


# query if the set of gpus are fully connected by nvlink (1 hop)
def _is_full_nvlink(rank, world_size):
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(rank)
    for i in range(world_size):
        if i != rank:
            try:
                link_state = pynvml.nvmlDeviceGetNvLinkState(handle, i)
                if not link_state:
                    return False
            except pynvml.NVMLError as error:
                logger.info(
                    f"NVLink detection failed with message \"{str(error)}\". "
                    "This is normal if your machine has no NVLink equipped")
                return False
    pynvml.nvmlShutdown()
    return True


class FastAllreduce:

    # max_size: max supported allreduce size
    def __init__(self, rank, world_size, max_size=8192 * 1024) -> None:
        # buffers memory are owned by this Python class and passed to C++
        self.meta = torch.zeros(fast_ar.meta_size() + max_size,
                                dtype=torch.uint8,
                                device="cuda")
        self.rank_data = torch.empty(16 * 1024 * 1024,
                                     dtype=torch.uint8,
                                     device="cuda")
        self.max_size = max_size
        self.world_size = world_size
        handles, offsets = self._get_ipc_meta(self.meta)
        self.full_nvlink = _is_full_nvlink(rank, world_size)
        self._ptr = fast_ar.init_fast_ar(self.meta, self.rank_data, handles,
                                         offsets, rank, self.full_nvlink)
        self.fast_cond = self.full_nvlink or world_size <= 2

    def _get_ipc_meta(self, inp: torch.Tensor):
        data = inp.storage()._share_cuda_()
        shard_data = (
            data[1],  # ipc handle to base ptr
            data[3],  # offset of base ptr
        )
        return self._gather_ipc_meta(shard_data)

    def _gather_ipc_meta(self, shard_data):
        all_data = [None] * self.world_size
        dist.all_gather_object(all_data, shard_data)

        handles = []
        offsets = []
        for i in range(len(all_data)):
            handles.append(all_data[i][0])
            offsets.append(all_data[i][1])
        return handles, offsets

    def register_buffer(self, inp: torch.Tensor):
        handles, offsets = self._get_ipc_meta(inp)
        fast_ar.register_buffer(self._ptr, inp, handles, offsets)

    def register_graph_buffers(self):
        handle, offset = fast_ar.get_graph_buffer_ipc_meta(self._ptr)
        handles, offsets = self._gather_ipc_meta((bytes(handle), offset))
        logger.info("Registering %d cuda graph addresses", len(offset))
        fast_ar.register_graph_buffers(self._ptr, handles, offsets)

    def should_fast_ar(self, inp: torch.Tensor):
        inp_size = inp.numel() * torch.finfo(inp.dtype).bits // 8
        if self.fast_cond:
            return inp_size <= self.max_size
        # 4 pcie gpus use 2 stage AR, and is only faster than NCCL
        # when size <= 512k
        return self.world_size <= 4 and inp_size <= 512 * 1024

    def all_reduce(self, inp: torch.Tensor, out: torch.Tensor = None):
        if out is None:
            out = torch.empty_like(inp)
        fast_ar.allreduce(self._ptr, inp, out)
        return out

    def close(self):
        if self._ptr:
            fast_ar.dispose(self._ptr)
            self._ptr = 0

    def __del__(self):
        self.close()
