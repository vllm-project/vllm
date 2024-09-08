
from vllm.distributed.group_coordinator import GroupCoordinator
from vllm.distributed.kv_transfer.kv_pipe.base import KVPipeBase
from torch.distributed import Backend, ProcessGroup
import torch
from typing import Any, Dict, List, Optional, Tuple, Union
import threading
from concurrent.futures import ThreadPoolExecutor
import time
import threading
from collections import namedtuple
from typing import Dict, Any, Tuple, List
import pickle

from vllm.logger import init_logger


logger = init_logger(__name__)


# if the tensor is only one-element and only contains this number
# this means that the sended object is None.
NONE_INT = -150886311
FLOAT16_INT = -543205003776624
INT64_INT = -375623078607432
BOOL_INT = -28035262008646
BFLOAT16_INT = -452084912267662
FLOAT32_INT = -1049557997456592
FLOAT64_INT = -452201007054137

DTYPE2INT = {
    torch.float16: FLOAT16_INT,
    torch.int64: INT64_INT,
    torch.bool: BOOL_INT,
    torch.bfloat16: BFLOAT16_INT,
    torch.float32: FLOAT32_INT,
    torch.float64: FLOAT64_INT,
}

INT2DTYPE = {
    FLOAT16_INT: torch.float16,
    INT64_INT: torch.int64,
    BOOL_INT: torch.bool,
    BFLOAT16_INT: torch.bfloat16,
    FLOAT32_INT: torch.float32,
    FLOAT64_INT: torch.float64,
}


class BrokenPipeException(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)

class TorchDistributedPipe(KVPipeBase):
    
    def __init__(
        self,
        group_ranks: List[List[int]],
        local_rank: int,
        torch_distributed_backend: Union[str, Backend]
    ):

        self.rank = torch.distributed.get_rank()
        self.local_rank = local_rank
        self.device_group = None
        self.cpu_group = None

        for ranks in group_ranks:
            device_group = torch.distributed.new_group(
                ranks, backend=torch_distributed_backend)
            # a group with `gloo` backend, to allow direct coordination between
            # processes through the CPU.
            cpu_group = torch.distributed.new_group(ranks, backend="gloo")
            if self.rank in ranks:
                self.ranks = ranks
                self.world_size = len(ranks)
                self.rank_in_group = ranks.index(self.rank)
                self.device_group = device_group
                self.cpu_group = cpu_group

        assert self.cpu_group is not None
        assert self.device_group is not None
        assert self.rank_in_group <= 1

        if torch.cuda.is_available():
            self.device = torch.device(f"cuda:{local_rank}")
        else:
            self.device = torch.device("cpu")

        # if turned on, will use CPU-based communication to perform a series of sanity check.
        # but it adds ~5ms delay, so please turn it off in performance-demanding usecases (e.g. disaggregated prefill)
        self.target_rank_for_send = self.ranks[(self.rank_in_group + 1) %
                                               self.world_size]
        self.target_rank_for_recv = self.ranks[(self.rank_in_group - 1) %
                                               self.world_size]
        torch.set_default_device(self.device)

        self.kv_sending_thread = None
        self.buffer_size = 0
        self.buffer_size_lock = threading.Lock()

        self.none_tensor = torch.tensor([NONE_INT]).to(self.device)
        self.broken = False

        
    def quick_send(self, tensor):

        group = self.device_group

        # NCCL is NOT fully duplex
        # so CPU communication is ALWAYS necessary
        torch.distributed.send_object_list(
            [tensor.dtype, tensor.shape, str(tensor.device)],
            dst=self.target_rank_for_send,
            group=self.cpu_group
        )

        torch.distributed.send(
            tensor,
            dst=self.target_rank_for_send,
            group=self.device_group
        )


    def quick_recv(self):

        # NCCL is NOT fully duplex
        # so CPU communication is necessary
        metadata = [None, None, None]
        torch.distributed.recv_object_list(
            metadata,
            src=self.target_rank_for_recv,
            group=self.cpu_group
        )
        
        dtype, shape, device = metadata
        if 'cuda' in device:
            device = self.device
        else:
            device = 'cpu'
        buffer = torch.zeros(shape, dtype=dtype).to(device, non_blocking=True)
        
        torch.distributed.recv(
            buffer,
            src=self.target_rank_for_recv,
            group=self.device_group
        )
        return buffer
        

        
    def send_tensor_wrapper(self, tensor) -> None:

        try:
            tensor_size = tensor.element_size() * tensor.numel()
            self.quick_send(tensor)
            
            with self.buffer_size_lock:
                self.buffer_size = self.buffer_size - tensor_size
        except Exception as e:
            logger.error("Encountering exception in KV sending thread")
            logger.error("%s", e)
        
    def block_if_full(self):
        
        while self.buffer_size > 1e9:
            logger.debug("KV cache transfer pipe is full. Waiting...")
            time.sleep(0.05)

    def send_tensor(self,
                   tensor: Optional[torch.Tensor]) -> None:
        """
        Sends a tensor to the destination rank in a non-blocking way.
        Flow: send tensor dim -- send tensor shape -- send tensor data
        """
        
        if self.kv_sending_thread is None:
            self.kv_sending_thread = ThreadPoolExecutor(max_workers=1)

        if tensor is None:
            tensor = self.none_tensor
            tensor_size = 0
        else:
            tensor_size = tensor.element_size() * tensor.numel()

        assert 0 < len(tensor.shape) <  100, "Send tensor does not support tensor with 0 dim or >=100 dim. Got %d" % len(tensor.shape)

        self.block_if_full()

        with self.buffer_size_lock:
            self.buffer_size = self.buffer_size + tensor_size
            
        # prepare the metadata before sending the tensor.
        self.kv_sending_thread.submit(
            self.send_tensor_wrapper, 
            tensor)
    
    def recv_tensor(self) -> Optional[torch.Tensor]:
        """Receives a tensor from the src rank. Blocking."""
        
        tensor = self.quick_recv()
        if tensor.numel() == 1 and tensor.item() == NONE_INT:
            return None
        else:
            return tensor
    

    
