from contextlib import contextmanager
from typing import Any, List, Optional

import torch
import torch.distributed as dist

import vllm.envs as envs
from vllm.logger import init_logger

try:
    import pynvml

    from vllm._C import custom_ar
except ImportError:
    # For AMD GPUs
    custom_ar = None
    pynvml = None

logger = init_logger(__name__)

_CA_HANDLE: Optional["CustomAllreduce"] = None
_IS_CAPTURING = False
_SUPPORTED_WORLD_SIZES = [2, 4, 6, 8]


def init_custom_ar() -> None:
    from vllm.distributed import (get_tensor_model_parallel_rank,
                                  get_tensor_model_parallel_world_size)

    global _CA_HANDLE
    if _CA_HANDLE is not None:
        return
    rank = get_tensor_model_parallel_rank()
    world_size = get_tensor_model_parallel_world_size()
    if world_size == 1:
        # No need to initialize custom allreduce for single GPU case.
        return

    if world_size not in _SUPPORTED_WORLD_SIZES:
        logger.warning(
            "Custom allreduce is disabled due to an unsupported world size: "
            "%d. Supported world sizes: %s. To silence this warning, specify"
            " disable_custom_all_reduce=True explicitly.", world_size,
            str(_SUPPORTED_WORLD_SIZES))
        return
    num_dev = torch.cuda.device_count()
    # note: num dev can be larger than world_size if we're only using
    # first few GPUs
    if num_dev < world_size:
        logger.warning(
            "Cannot test GPU P2P because not all GPUs are visible to the "
            "current process. This might be the case if 'CUDA_VISIBLE_DEVICES'"
            " is set.")
        return
    # test nvlink first, this will filter out most of the cases
    # where custom allreduce is not supported
    cuda_visible_devices = envs.CUDA_VISIBLE_DEVICES
    if cuda_visible_devices:
        device_ids = list(map(int, cuda_visible_devices.split(",")))
    else:
        device_ids = list(range(num_dev))
    # this checks hardware and driver support for NVLink
    full_nvlink = _is_full_nvlink(device_ids)
    if world_size > 2 and not full_nvlink:
        logger.warning(
            "Custom allreduce is disabled because it's not supported on more"
            " than two PCIe-only GPUs. To silence this warning, specify"
            " disable_custom_all_reduce=True explicitly.")
        return
    # test P2P capability, this checks software/cudaruntime support
    # this is expensive to compute at the first time
    # then we cache the result
    if not _can_p2p(rank, world_size):
        logger.warning(
            "Custom allreduce is disabled because your platform lacks GPU P2P"
            " capability or P2P test failed. To silence this warning, specify"
            " disable_custom_all_reduce=True explicitly.")
        return
    _CA_HANDLE = CustomAllreduce(rank, world_size, full_nvlink)


def begin_capture() -> None:
    global _IS_CAPTURING
    _IS_CAPTURING = True


def end_capture() -> None:
    global _IS_CAPTURING
    _IS_CAPTURING = False


def is_capturing() -> bool:
    return _IS_CAPTURING and _CA_HANDLE is not None


def get_handle() -> Optional["CustomAllreduce"]:
    return _CA_HANDLE


def is_initialized() -> bool:
    return _CA_HANDLE is not None


@contextmanager
def capture():
    try:
        begin_capture()
        yield
    finally:
        end_capture()
        handle = get_handle()
        if handle is not None:
            handle.register_graph_buffers()


def custom_all_reduce(input: torch.Tensor) -> Optional[torch.Tensor]:
    ca_handle = get_handle()
    # when custom allreduce is disabled, this will be None
    if ca_handle is None:
        return None
    if is_capturing():
        if torch.cuda.is_current_stream_capturing():
            if ca_handle.should_custom_ar(input):
                return ca_handle.all_reduce_reg(input)
        else:
            if ca_handle.should_custom_ar(input):
                # if warm up, mimic the allocation pattern
                # since custom allreduce is out-of-place
                return torch.empty_like(input)
    else:
        # note: outside of cuda graph context,
        # custom allreduce incurs a cost of cudaMemcpy, which should
        # be small(<=1% of overall latency) compared to the performance
        # gains of using custom kernels
        if ca_handle.should_custom_ar(input):
            return ca_handle.all_reduce_unreg(input)

    return None


@contextmanager
def _nvml():
    try:
        pynvml.nvmlInit()
        yield
    finally:
        pynvml.nvmlShutdown()


@_nvml()
def _is_full_nvlink(device_ids: List[int]) -> bool:
    """
    query if the set of gpus are fully connected by nvlink (1 hop)
    Note that `pynvml` is not affected by `CUDA_VISIBLE_DEVICES`,
    so it works on real physical device ids.
    """
    handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in device_ids]
    for i, handle in enumerate(handles):
        for j, peer_handle in enumerate(handles):
            if i < j:
                try:
                    p2p_status = pynvml.nvmlDeviceGetP2PStatus(
                        handle, peer_handle, pynvml.NVML_P2P_CAPS_INDEX_NVLINK)
                    if p2p_status != pynvml.NVML_P2P_STATUS_OK:
                        return False
                except pynvml.NVMLError as error:
                    logger.error(
                        "NVLink detection failed. This is normal if your"
                        " machine has no NVLink equipped.",
                        exc_info=error)
                    return False
    return True


def _can_p2p(rank: int, world_size: int) -> bool:
    from vllm.distributed.utils import gpu_p2p_access_check
    for i in range(world_size):
        if i == rank:
            continue
        if not gpu_p2p_access_check(rank, i):
            return False
    return True


class CustomAllreduce:

    # max_size: max supported allreduce size
    def __init__(self,
                 rank,
                 world_size,
                 full_nvlink,
                 max_size=8192 * 1024) -> None:
        # buffers memory are owned by this Python class and passed to C++
        # meta data composes of two parts: meta data for synchronization
        # (256 bytes) and a temporary buffer for storing intermediate
        # allreduce results.
        self.meta = torch.zeros(custom_ar.meta_size() + max_size,
                                dtype=torch.uint8,
                                device="cuda")
        # This is a pre-registered IPC buffer. In eager mode, input tensors
        # are first copied into this buffer before allreduce is performed
        self.buffer = torch.empty(max_size, dtype=torch.uint8, device="cuda")
        # This is a buffer for storing the tuples of pointers pointing to
        # IPC buffers from all ranks. Each registered tuple has size of
        # 8*world_size bytes where world_size is at most 8. Allocating 8MB
        # is enough for 131072 such tuples. The largest model I've seen only
        # needs less than 10000 of registered tuples.
        self.rank_data = torch.empty(8 * 1024 * 1024,
                                     dtype=torch.uint8,
                                     device="cuda")
        self.max_size = max_size
        self.world_size = world_size
        handles, offsets = self._get_ipc_meta(self.meta)
        self.full_nvlink = full_nvlink
        self._ptr = custom_ar.init_custom_ar(self.meta, self.rank_data,
                                             handles, offsets, rank,
                                             self.full_nvlink)
        self.register_buffer(self.buffer)

    def _get_ipc_meta(self, inp: torch.Tensor):
        data = inp.untyped_storage()._share_cuda_()
        shard_data = (
            data[1],  # ipc handle to base ptr
            data[3],  # offset of base ptr
        )
        return self._gather_ipc_meta(shard_data)

    def _gather_ipc_meta(self, shard_data):
        all_data: List[Optional[Any]] = [None] * self.world_size
        dist.all_gather_object(all_data, shard_data)

        handles = []
        offsets = []
        for i in range(len(all_data)):
            handles.append(all_data[i][0])  # type: ignore
            offsets.append(all_data[i][1])  # type: ignore
        return handles, offsets

    def register_buffer(self, inp: torch.Tensor):
        handles, offsets = self._get_ipc_meta(inp)
        custom_ar.register_buffer(self._ptr, inp, handles, offsets)

    def register_graph_buffers(self):
        handle, offset = custom_ar.get_graph_buffer_ipc_meta(self._ptr)
        handles, offsets = self._gather_ipc_meta((bytes(handle), offset))
        logger.info("Registering %d cuda graph addresses", len(offset))
        custom_ar.register_graph_buffers(self._ptr, handles, offsets)

    def should_custom_ar(self, inp: torch.Tensor):
        return custom_ar.should_custom_ar(inp, self.max_size, self.world_size,
                                          self.full_nvlink)

    # all reduce, assuming inp tensor is IPC registered with register_buffer,
    # or, in the context of cuda graphs, register_graph_buffers
    def all_reduce_reg(self, inp: torch.Tensor, out: torch.Tensor = None):
        if out is None:
            out = torch.empty_like(inp)
        custom_ar.all_reduce_reg(self._ptr, inp, out)
        return out

    # all reduce, assuming inp tensor is NOT IPC registered
    def all_reduce_unreg(self, inp: torch.Tensor, out: torch.Tensor = None):
        if out is None:
            out = torch.empty_like(inp)
        custom_ar.all_reduce_unreg(self._ptr, inp, self.buffer, out)
        return out

    def close(self):
        if self._ptr:
            custom_ar.dispose(self._ptr)
            self._ptr = 0

    def __del__(self):
        self.close()
