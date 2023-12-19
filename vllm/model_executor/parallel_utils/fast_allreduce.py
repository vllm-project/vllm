import torch
import torch.distributed as dist
import pynvml
from vllm.logger import init_logger
from vllm._C import fast_ar

logger = init_logger(__name__)


# query if the set of gpus are fully connected by nvlink (1 hop)
def full_nvlink(rank, world_size):
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

    def __init__(self, rank, world_size, max_size=8192 * 1024) -> None:
        self.meta = torch.zeros(fast_ar.meta_size() + max_size,
                                dtype=torch.uint8,
                                device=rank)
        self.max_size = max_size
        self.world_size = world_size
        handles, offsets = self._get_ipc_meta(self.meta)
        self.full_nvlink = full_nvlink(rank, world_size)
        self._ptr = fast_ar.prepare_buffer(self.meta.data_ptr(), handles,
                                           offsets, rank, self.full_nvlink)
        self.fast_cond = self.full_nvlink or world_size <= 2
        self.is_capturing = False

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
