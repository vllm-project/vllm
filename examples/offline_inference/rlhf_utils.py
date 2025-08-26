# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os

import torch

from vllm.platforms import current_platform


def stateless_init_process_group(master_address, master_port, rank, world_size, device):
    """
    vLLM provides `StatelessProcessGroup` to create a process group
    without considering the global process group in torch.distributed.
    It is recommended to create `StatelessProcessGroup`, and then initialize
    the data-plane communication (NCCL) between external (train processes)
    and vLLM workers.
    """
    if current_platform.is_xpu():
        from vllm.distributed.device_communicators.xpu_communicator import (
            XpuCommunicator,
        )

        os.environ.setdefault("CCL_ATL_TRANSPORT", "ofi")
        os.environ.setdefault("LOCAL_WORLD_SIZE", str(world_size))
        os.environ["LOCAL_RANK"] = str(rank)
        from vllm.utils import get_distributed_init_method

        distributed_init_method = get_distributed_init_method(
            master_address, master_port
        )

        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(
                backend="ccl",
                init_method=distributed_init_method,
                world_size=world_size,
                rank=rank,
            )

        ranks = list(range(torch.distributed.get_world_size()))
        pg = torch.distributed.new_group(ranks, backend="ccl")
        ccl = XpuCommunicator(pg, device=device)
        return ccl
    else:
        from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
        from vllm.distributed.utils import StatelessProcessGroup

        pg = StatelessProcessGroup.create(
            host=master_address, port=master_port, rank=rank, world_size=world_size
        )
        pynccl = PyNcclCommunicator(pg, device=device)
        return pynccl


class WorkerExtension:
    """
    The class for vLLM's worker to inherit from.
    By defining an extension class, the code can work no matter what is
    the underlying worker class. This way, the code can be compatible
    with both vLLM V0 and V1.
    NOTE: we define this class in a separate module, and the main module
    should pass the full qualified name as `worker_extension_cls` argument.
    """

    def init_weight_update_group(
        self, master_address, master_port, rank_offset, world_size
    ):
        from vllm.distributed.parallel_state import get_world_group

        rank = get_world_group().rank + rank_offset
        self.model_update_group = stateless_init_process_group(
            master_address,
            master_port,
            rank,
            world_size,
            self.device,
        )

    def update_weight(self, name, dtype_name, shape):
        dtype = getattr(torch, dtype_name)
        weight = torch.empty(shape, dtype=dtype, device=current_platform.device_type)

        if current_platform.is_xpu():
            self.model_update_group.broadcast(weight, src=0)
        else:
            self.model_update_group.broadcast(
                weight, src=0, stream=torch.cuda.current_stream()
            )

        self.model_runner.model.load_weights(weights=[(name, weight)])

        del weight

    def check_weights_changed(self):
        """
        Check if the weights are updated to 0.
        """
        weights_updated = True
        for name, p in self.model_runner.model.named_parameters():
            weights_updated = weights_updated and torch.allclose(p, torch.zeros_like(p))
        return weights_updated


class ColocateWorkerExtension:
    """
    The class for vLLM's worker to inherit from, in the colocate setting.
    By defining an extension class, the code can work no matter what is
    the underlying worker class. This way, the code can be compatible
    with both vLLM V0 and V1.
    NOTE: we define this class in a separate module, and the main module
    should pass the full qualified name as `worker_extension_cls` argument.
    """

    def report_device_id(self) -> str:
        from vllm.platforms import current_platform

        self.device_uuid = current_platform.get_device_uuid(self.device.index)
        return self.device_uuid

    def update_weights_from_ipc_handles(self, ipc_handles):
        handles = ipc_handles[self.device_uuid]
        device_id = self.device.index
        weights = []
        for name, handle in handles.items():
            func, args = handle
            list_args = list(args)
            # the key is to change device id to the current device id
            # in case two processes have different CUDA_VISIBLE_DEVICES
            if current_platform.is_xpu():
                tensor = func(*list_args)
                tensor = tensor.to(current_platform.device_type + ":" + str(device_id))
            else:
                list_args[6] = device_id
                tensor = func(*list_args)
            weights.append((name, tensor))
        self.model_runner.model.load_weights(weights=weights)
        if current_platform.is_xpu():
            torch.xpu.synchronize()
        else:
            torch.cuda.synchronize()

    def check_weights_changed(self):
        """
        Check if the weights are updated to 0.
        """
        weights_updated = True
        for name, p in self.model_runner.model.named_parameters():
            weights_updated = weights_updated and torch.allclose(p, torch.zeros_like(p))
        return weights_updated
