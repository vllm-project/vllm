from typing import Optional, Union, Any
from torch.distributed import Backend, ProcessGroup
import torch

from vllm.distributed.parallel_state import GroupCoordinator, TensorMetadata
from vllm.distributed.parallel_state import _get_unique_name, _register_group, _split_tensor_dict
from vllm.distributed.device_communicators.cuda_communicator import CudaCommunicator
from vllm.distributed.utils import (
    StatelessProcessGroup, stateless_init_torch_distributed_process_group,
    stateless_destroy_torch_distributed_process_group)
from vllm.logger import init_logger
from vllm.utils import resolve_obj_by_qualname

logger = init_logger(__name__)


class StatelessGroupCoordinator(GroupCoordinator):

    def __init__(
        self,
        group_ranks: list[list[int]],
        local_rank: int,
        torch_distributed_backend: Union[str, Backend],
        use_device_communicator: bool,
        use_message_queue_broadcaster: bool = False,
        group_name: Optional[str] = None,
        host: str = "127.0.0.1",
        group_ports: list[list[int]] = None,
        global_rank: int = 0,
        global_world_size: int = 1,
    ):
        group_name = group_name or "anonymous"
        self.unique_name = _get_unique_name(group_name)
        _register_group(self)

        self.rank = global_rank
        self.local_rank = local_rank

        self_device_group = None
        self_cpu_group = None
        self_tcp_store_group = None

        from vllm.platforms import current_platform

        backend = str(torch_distributed_backend)
        self.backend = backend

        for idx, ranks in enumerate(group_ranks):
            if self.rank in ranks:
                self.ranks = ranks
                self.world_size = len(ranks)
                self.rank_in_group = ranks.index(self.rank)

                ports = group_ports[idx]
                device_port = ports[0]
                cpu_port = ports[1]
                tcp_store_port = ports[2]

                device_group = stateless_init_torch_distributed_process_group(
                    host=host,
                    port=device_port,
                    rank=self.rank_in_group,
                    world_size=self.world_size,
                    backend=backend,
                    group_name=f"{self.unique_name}_device"
                )
                cpu_group = stateless_init_torch_distributed_process_group(
                    host=host,
                    port=cpu_port,
                    rank=self.rank_in_group,
                    world_size=self.world_size,
                    backend="gloo",
                    group_name=f"{self.unique_name}_cpu"
                )
                tcp_store_group = StatelessProcessGroup.create(
                    host=host,
                    port=tcp_store_port,
                    rank=self.rank_in_group,
                    world_size=self.world_size,
                )

                self_device_group = device_group
                self_cpu_group = cpu_group
                self_tcp_store_group = tcp_store_group

        assert self_cpu_group is not None
        assert self_device_group is not None
        assert self_tcp_store_group is not None

        self.cpu_group = self_cpu_group
        self.device_group = self_device_group
        self.tcp_store_group = self_tcp_store_group

        if current_platform.is_cuda_alike():
            self.device = torch.device(f"cuda:{local_rank}")
        elif current_platform.is_xpu():
            self.device = torch.device(f"xpu:{local_rank}")
        elif current_platform.is_out_of_tree():
            self.device = torch.device(
                f"{current_platform.device_name}:{local_rank}")
        else:
            self.device = torch.device("cpu")

        self.use_device_communicator = use_device_communicator
        self.device_communicator = None
        if use_device_communicator and self.world_size > 1:
            device_comm_cls = resolve_obj_by_qualname(
                current_platform.get_device_communicator_cls())
            assert device_comm_cls == CudaCommunicator
            self.device_communicator = CudaCommunicator(
                cpu_group=self.cpu_group,
                device=self.device,
                device_group=self.device_group,
                unique_name=self.unique_name,
                global_ranks=self.ranks,
                global_world_size=global_world_size,
                tcp_store_group=self.tcp_store_group
            )

        self.mq_broadcaster = None

        self.use_custom_op_call = (current_platform.is_cuda_alike()
                                   or current_platform.is_tpu())
        self.use_cpu_custom_send_recv = False

    def destroy(self):
        if self.device_communicator:
            self.device_communicator.destroy()
        if self.device_group:
            stateless_destroy_torch_distributed_process_group(self.device_group)
        if self.cpu_group:
            stateless_destroy_torch_distributed_process_group(self.cpu_group)
        self.tcp_store_group = None

    def broadcast(self, input_: torch.Tensor, src: int = 0):
        if self.world_size == 1:
            return input_

        if self.device_communicator and input_.is_cuda:
            return self.device_communicator.broadcast(input_, src)
        else:
            return self.tcp_store_group.broadcast(input_, src)

    def broadcast_object(self, obj=None, src: int = 0):
        if self.world_size == 1:
            return obj
        return self.tcp_store_group.broadcast_obj(obj, src)

    def broadcast_object_list(self,
                              obj_list: list[Any],
                              src: int = 0,
                              group: Optional[ProcessGroup] = None):
        assert src < self.world_size

        if self.world_size == 1:
            return obj_list

        if self.rank_in_group == src:
            for obj in obj_list:
                self.tcp_store_group.broadcast_obj(obj, src)
        else:
            for i in range(len(obj_list)):
                obj_list[i] = self.tcp_store_group.broadcast_obj(None, src)

        return obj_list

    def broadcast_tensor_dict(
        self,
        tensor_dict: Optional[dict[str, Union[torch.Tensor, any]]] = None,
        src: int = 0,
        group: Optional[ProcessGroup] = None,
        metadata_group: Optional[ProcessGroup] = None
    ) -> Optional[dict[str, Union[torch.Tensor, any]]]:
        if self.world_size == 1:
            return tensor_dict

        if self.rank_in_group == src:
            metadata_list, tensor_list = _split_tensor_dict(tensor_dict)
        else:
            metadata_list = None
            tensor_list = []

        metadata_list = self.tcp_store_group.broadcast_obj(metadata_list, src)

        if self.rank_in_group != src:
            tensor_dict = {}
            for key, value in metadata_list:
                if isinstance(value, TensorMetadata):
                    tensor = torch.empty(value.size,
                                        dtype=value.dtype,
                                        device=value.device)
                    tensor_list.append(tensor)
                    tensor_dict[key] = tensor
                else:
                    tensor_dict[key] = value

        for tensor in tensor_list:
            if tensor.numel() == 0:
                continue
            if self.device_communicator and tensor.is_cuda:
                self.device_communicator.broadcast(tensor, src)
            else:
                self.tcp_store_group.broadcast(tensor, src)

        return tensor_dict

    def send_object(self, obj, dst: int) -> None:
        assert dst < self.world_size
        assert dst != self.rank_in_group
        self.tcp_store_group.send_obj(obj, dst)

    def recv_object(self, src: int):
        assert src < self.world_size
        assert src != self.rank_in_group
        return self.tcp_store_group.recv_obj(src)

    def send_tensor_dict(
        self,
        tensor_dict: dict[str, Union[torch.Tensor, any]],
        dst: Optional[int] = None,
        all_gather_group: Optional["GroupCoordinator"] = None,
        all_gather_tensors: Optional[dict[str, bool]] = None,
    ) -> Optional[dict[str, Union[torch.Tensor, any]]]:
        if self.world_size == 1:
            return tensor_dict

        if dst is None:
            dst = (self.rank_in_group + 1) % self.world_size
        assert dst < self.world_size

        metadata_list, tensor_list = _split_tensor_dict(tensor_dict)
        self.tcp_store_group.send_obj(metadata_list, dst)

        for tensor in tensor_list:
            if tensor.numel() == 0:
                continue
            if self.device_communicator and tensor.is_cuda:
                self.device_communicator.send(tensor, dst)
            else:
                self.tcp_store_group.send(tensor, dst)

        return None

    def recv_tensor_dict(
        self,
        src: Optional[int] = None,
        all_gather_group: Optional["GroupCoordinator"] = None,
        all_gather_tensors: Optional[dict[str, bool]] = None,
    ) -> Optional[dict[str, Union[torch.Tensor, any]]]:
        if self.world_size == 1:
            return None

        if src is None:
            src = (self.rank_in_group - 1) % self.world_size
        assert src < self.world_size

        recv_metadata_list = self.tcp_store_group.recv_obj(src)
        tensor_dict = {}
        for key, value in recv_metadata_list:
            if isinstance(value, TensorMetadata):
                tensor = torch.empty(value.size,
                                   dtype=value.dtype,
                                   device=value.device)
                if tensor.numel() > 0:
                    if self.device_communicator and tensor.is_cuda:
                        tensor = self.device_communicator.recv(tensor.size(), tensor.dtype, src)
                    else:
                        tensor = self.tcp_store_group.recv(tensor, src)
                tensor_dict[key] = tensor
            else:
                tensor_dict[key] = value
        return tensor_dict

    def barrier(self):
        self.tcp_store_group.barrier()

    def gather(self,
               input_: torch.Tensor,
               dst: int = 0,
               dim: int = -1) -> Optional[torch.Tensor]:
        if self.world_size == 1:
            return input_

        if self.device_communicator is None:
            raise ValueError("No device communicator found")

        if self.rank_in_group == dst:
            gathered_list = [torch.empty_like(input_) for _ in range(self.world_size)]
            gathered_list[self.rank_in_group] = input_
            for src_rank in range(self.world_size):
                if src_rank != self.rank_in_group:
                    gathered_list[src_rank] = self.device_communicator.recv(input_.size(), input_.dtype, src_rank)
            return torch.cat(gathered_list, dim=dim)
        else:
            self.device_communicator.send(input_, dst)
            return None