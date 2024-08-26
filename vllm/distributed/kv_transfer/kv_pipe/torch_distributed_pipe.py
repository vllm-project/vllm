
from vllm.distributed.group_coordinator import GroupCoordinator
from vllm.distributed.kv_transfer.kv_pipe.base import KVPipeBase
from torch.distributed import Backend, ProcessGroup
import torch
from typing import List, Union, Optional

class TorchDistributedPipe(KVPipeBase, GroupCoordinator):
    
class DistributedKVCoordinator(GroupCoordinator):
    """
    A class designated for distributed KV transfer
    
    Target use cases:
        1. Disaggregated prefill
        2. Remote KV cache storage
        
    """

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
        blocking_send_recv: bool = False,
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
        self.blocking_send_recv = blocking_send_recv
        self.target_rank_for_send = self.ranks[(self.rank_in_group + 1) %
                                               self.world_size]
        self.target_rank_for_recv = self.ranks[(self.rank_in_group - 1) %
                                               self.world_size]
        torch.set_default_device(self.device)

    def send_tensor(self,
                   tensor: torch.Tensor) -> None:
        """
        Sends a tensor to the destination rank in a non-blocking way.
        Flow: send tensor dim -- send tensor shape -- send tensor data
        """
        
        dim_tensor = torch.tensor([len(tensor.shape)], dtype=torch.int).to(self.device, non_blocking=True)
        shape_tensor = torch.tensor(tensor.shape, dtype=torch.int).to(self.device, non_blocking=True)
        
        torch.distributed.isend(dim_tensor, self.target_rank_for_send, self.device_group)
        torch.distributed.isend(shape_tensor, self.target_rank_for_send, self.device_group)
        torch.distributed.isend(tensor, self.target_rank_for_send, self.device_group)

    def recv_tensor(self) -> torch.Tensor:
        """Receives a tensor from the src rank. Blocking."""
        
        # FIXME(Kuntai): this incurs frequent data moving between CPU and GPU
        # can be optimized by pre-allocating tensors on GPU.
        dim_tensor = torch.tensor([0], dtype=torch.int).to(self.device)
        torch.distributed.irecv(dim_tensor, self.target_rank_for_recv, self.device_group)
        dim = dim_tensor.item()
        shape_tensor = torch.zeros(dim, dtype=torch.int).to(self.device)
        torch.distributed.irecv(shape_tensor, self.target_rank_for_recv, self.device_group)
        return_tensor = torch.zeros(shape_tensor, dtype=torch.float32).to(self.device)
        torch.distributed.irecv(return_tensor, self.target_rank_for_recv, self.device_group)

        result = self.recv_tensor_dict(src)
        tensor = result["tensor"]
        assert torch.allclose(result["mean"], tensor.float().mean())
        assert result["shape"] == tensor.shape
        assert result[
            "shape"] == size, f"The shape sent by sender is {result['shape']} but trying to receive {size}"
        return tensor
