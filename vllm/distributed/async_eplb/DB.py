from abc import ABC, abstractmethod
import torch
import torch_npu
import torch.distributed as dist
from torch.distributed import (P2POp, ProcessGroup, 
                               batch_isend_irecv, barrier)

class DeviceBackend(ABC):
    """硬件后端抽象基类，定义统一接口"""
    
    @abstractmethod
    def synchronize(self) -> None:
        """同步当前设备的所有操作"""
        pass
    
    @abstractmethod
    def create_buffer_like(self, tensor: torch.Tensor) -> torch.Tensor:
        """创建与输入张量相同类型和设备的缓冲区"""
        pass
    
    @abstractmethod
    def all_gather(self, output_tensor_list: list[torch.Tensor], 
                  input_tensor: torch.Tensor, group=None) -> None:
        """执行all_gather集体通信操作"""
        pass
    
    @abstractmethod
    def batch_isend_irecv(self, p2p_ops: list[P2POp]) -> list[dist.Work]:
        """执行批量异步发送和接收操作"""
        pass
    
    @abstractmethod
    def barrier(self, group=None) -> None:
        """执行屏障同步"""
        pass


class CUDABackend(DeviceBackend):
    """CUDA/NVIDIA GPU后端实现"""
    
    def synchronize(self) -> None:
        torch.cuda.synchronize()
    
    def create_buffer_like(self, tensor: torch.Tensor) -> torch.Tensor:
        return torch.empty_like(tensor, device='cuda')
    
    def all_gather(self, output_tensor_list: list[torch.Tensor], 
                  input_tensor: torch.Tensor, group=None) -> None:
        dist.all_gather(output_tensor_list, input_tensor, group=group)
    
    def batch_isend_irecv(self, p2p_ops: list[P2POp]) -> list[dist.Work]:
        return dist.batch_isend_irecv(p2p_ops)
    
    def barrier(self, group=None) -> None:
        dist.barrier(group=group)


class NPUBackend(DeviceBackend):
    """NPU后端实现"""
    
    def synchronize(self) -> None:
        torch_npu.synchronize()
        pass
    
    def create_buffer_like(self, tensor: torch.Tensor) -> torch.Tensor:
        return torch_npu.empty_like(tensor, device='npu')
    
    def all_gather(self, output_tensor_list: list[torch.Tensor], 
                  input_tensor: torch.Tensor, group=None) -> None:
        dist.all_gather(output_tensor_list, input_tensor, group=group)
    
    def batch_isend_irecv(self, p2p_ops: list[P2POp]) -> list[dist.Work]:
        return dist.batch_isend_irecv(p2p_ops)
    
    def barrier(self, group=None) -> None:
        dist.barrier(group=group)


# 根据可用硬件创建适当的后端
def create_device_backend(device) -> DeviceBackend:
    if device=='gpu' or device=='cuda':
        return CUDABackend()
    elif device=='npu' :
        return NPUBackend()