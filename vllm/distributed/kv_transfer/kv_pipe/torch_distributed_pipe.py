
from vllm.distributed.group_coordinator import GroupCoordinator
from vllm.distributed.kv_transfer.kv_pipe.base import KVPipeBase
from torch.distributed import Backend, ProcessGroup
import torch
from typing import List, Union, Optional
import threading
from concurrent.futures import ThreadPoolExecutor
import time
import threading


# if the tensor is only one-element and only contains this number
# this means that the sended object is None.
NONE_INT = -150886311


class BrokenPipeException(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)

class TorchDistributedPipe(KVPipeBase, GroupCoordinator):
    
    def __init__(
        self,
        group_ranks: List[List[int]],
        local_rank: int,
        torch_distributed_backend: Union[str, Backend],
        # DO NOT use pynccl here
        # Pynccl send is non-blocking
        # and it's possible that the memory is freed before the data being sent
        # which may happen at high qps
        use_pynccl: bool = False,
        use_custom_allreduce: bool = False,
        use_tpu_communicator: bool = True,
        use_message_queue_broadcaster: bool = False,
    ):

        super().__init__(
            group_ranks,
            local_rank,
            torch_distributed_backend,
            use_pynccl,
            use_custom_allreduce,
            use_tpu_communicator,
            use_message_queue_broadcaster,
        )

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
        
        
    def send_tensor_wrapper(self, tensor: torch.Tensor) -> None:
        print('Sending ', tensor)
        """Wrapper for send_tensor_dict"""
        tensor_size = tensor['tensor'].element_size() * tensor['tensor'].numel()
        self.send_tensor_dict({'tensor': tensor}, self.target_rank_for_send)
        
        with self.buffer_size_lock:
            self.buffer_size = self.buffer_size - tensor_size
        
    def block_if_full(self):
        
        while self.buffer_size > 1e9:
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

        self.block_if_full()

        with self.buffer_size_lock:
            self.buffer_size = self.buffer_size + tensor_size
            
        self.kv_sending_thread.submit(self.send_tensor_wrapper, tensor)

        

    def recv_tensor(self) -> Optional[torch.Tensor]:
        """Receives a tensor from the src rank. Blocking."""
        
        tensor = self.recv_tensor_dict(self.target_rank_for_recv)['tensor']
        if tensor.numel() == 1 and tensor.item() == 150886311:
            return None
        else:
            return tensor
    