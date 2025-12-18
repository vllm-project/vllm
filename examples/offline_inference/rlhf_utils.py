# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import gc
from collections.abc import Callable
from typing import TypedDict

import torch
import zmq


def stateless_init_process_group(master_address, master_port, rank, world_size, device):
    """
    vLLM provides `StatelessProcessGroup` to create a process group
    without considering the global process group in torch.distributed.
    It is recommended to create `StatelessProcessGroup`, and then initialize
    the data-plane communication (NCCL) between external (train processes)
    and vLLM workers.
    """
    from vllm.distributed.utils import StatelessProcessGroup

    pg = StatelessProcessGroup.create(
        host=master_address, port=master_port, rank=rank, world_size=world_size
    )

    if device.type == "xpu":
        from vllm.distributed.device_communicators.xpu_communicator import XpuCommunicator
        pynccl = XpuCommunicator(pg, device=device)
    else:
        from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
        pynccl = PyNcclCommunicator(pg, device=device)
    return pynccl


class WorkerExtension:
    """
    The class for vLLM's worker to inherit from.
    By defining an extension class, the code can work no matter what is
    the underlying worker class.

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
        weight = torch.empty(shape, dtype=dtype, device="cuda")
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


def rebuild_ipc(
    handle: tuple[Callable, tuple], device_id: int | None = None
) -> torch.Tensor:
    func, args = handle
    list_args = list(args)
    if device_id is not None:
        # the key is to change device id to the current device id
        # in case two processes have different CUDA_VISIBLE_DEVICES
        list_args[6] = device_id
    buffer = func(*list_args)
    return buffer


class FlattenedTensorMetadata(TypedDict):
    name: str
    shape: torch.Size
    dtype: torch.dtype
    # specify the start offset of this tensor in shared ipc_buffer tensor
    offset: int


class ColocateWorkerExtension:
    """
    The class for vLLM's worker to inherit from, in the colocate setting.
    By defining an extension class, the code can work no matter what is
    the underlying worker class.

    NOTE: we define this class in a separate module, and the main module
    should pass the full qualified name as `worker_extension_cls` argument.
    """

    def update_weights_from_ipc(self, zmq_handles: dict[str, str]):
        from vllm.model_executor.model_loader.utils import process_weights_after_loading

        assert self.device is not None
        if not hasattr(self, "_zmq_ctx") or self._zmq_ctx is None:
            self._zmq_ctx = zmq.Context()
        socket = self._zmq_ctx.socket(zmq.REP)
        socket.connect(zmq_handles[self.report_device_id()])
        buffer: torch.Tensor | None = None
        while True:
            payload: tuple[Callable, tuple] | list[FlattenedTensorMetadata] | None = (
                socket.recv_pyobj()
            )
            if payload is None:
                # means the update is done
                process_weights_after_loading(
                    self.model_runner.model, self.model_config, self.device
                )
                torch.cuda.synchronize()
                socket.send(b"")
                break
            if isinstance(payload, tuple):
                # an ipc handle that vLLM can use `func, args = handle`
                # and `func(*args)` to rebuild GPU tensor.
                buffer = rebuild_ipc(payload, self.device.index)
                assert buffer.dtype == torch.uint8
                socket.send(b"")
                continue
            assert isinstance(payload, list)
            assert buffer is not None
            weights = []
            for item in payload:
                shape = item["shape"]
                if isinstance(shape, (list, tuple)):
                    shape = torch.Size(shape)
                assert isinstance(shape, torch.Size)
                dtype, offset = item["dtype"], item["offset"]
                size = dtype.itemsize * shape.numel()
                tensor = buffer[offset : offset + size].view(dtype=dtype).view(shape)
                weights.append((item["name"], tensor))
            self.model_runner.model.load_weights(weights=weights)
            del weights
            torch.cuda.synchronize()
            socket.send(b"")

        socket.close()
        del buffer
        gc.collect()
        torch.cuda.empty_cache()

    def report_device_id(self) -> str:
        from vllm.platforms import current_platform

        self.device_uuid = current_platform.get_device_uuid(self.device.index)
        return self.device_uuid

    def check_weights_changed(self):
        """
        Check if the weights are updated to 0.
        """
        weights_updated = True
        for name, p in self.model_runner.model.named_parameters():
            weights_updated = weights_updated and torch.allclose(p, torch.zeros_like(p))
        return weights_updated
