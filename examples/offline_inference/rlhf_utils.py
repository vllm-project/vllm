# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import gc
from ast import Dict, Tuple
from collections.abc import Callable
from enum import Enum
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


class PayloadType(Enum):
    """Enumerates possible payload types in IPC protocol."""

    HANDLES = "handles"
    BUFFER_UPDATE = "buffer_update"
    DONE = "done"
    UNKNOWN = "unknown"


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

    def update_weights_from_ipc_async(self, zmq_handles: dict[str, str]):
        assert self.device is not None
        if not hasattr(self, "_zmq_ctx") or self._zmq_ctx is None:
            self._zmq_ctx = zmq.Context()
        socket = self._zmq_ctx.socket(zmq.ROUTER)
        socket.bind(zmq_handles[self.report_device_id()])
        poller = zmq.Poller()
        poller.register(socket, zmq.POLLIN)

        buffers: Dict[int, torch.Tensor] = {}
        while True:
            events = dict(poller.poll(timeout=100))
            if socket in events and (events[socket] & zmq.POLLIN):
                # Router identity
                identity = socket.recv()

                payload: (
                    list[tuple[Callable, tuple]]
                    | tuple[int, list[FlattenedTensorMetadata]]
                    | None
                ) = socket.recv_pyobj()

                payload_type = self._identify_payload_type(payload)

                # === HANDLE LIST OF SHARED MEMORY HANDLES ===
                if payload_type == PayloadType.HANDLES:
                    handles: list[tuple[Callable, tuple]] = payload
                    for i, h in enumerate(handles):
                        buffers[i] = rebuild_ipc(h, self.device.index)
                    socket.send_multipart([identity, b"ACK_HANDLES"])
                    continue

                # === HANDLE BUFFERED MODEL UPDATES ===
                if payload_type == PayloadType.BUFFER_UPDATE:
                    buf_id, items = payload
                    buffer = buffers.get(buf_id)
                    if buffer is None:
                        continue

                    weights: list[Tuple[str, torch.Tensor]] = []
                    for item in items:
                        assert isinstance(item, dict)
                        shape = torch.Size(item["shape"])
                        dtype, offset = item["dtype"], item["offset"]
                        size = dtype.itemsize * shape.numel()
                        tensor = (
                            buffer[offset : offset + size].view(dtype=dtype).view(shape)
                        )
                        weights.append((item["name"], tensor))

                    self.model_runner.model.load_weights(weights=weights)
                    torch.cuda.synchronize()
                    socket.send_multipart([identity, str(buf_id).encode()])

                # === DONE SIGNAL ===
                elif payload_type == PayloadType.DONE:
                    socket.send_multipart([identity, b"DONE"])
                    break
                else:
                    continue

        socket.close()
        gc.collect()
        torch.cuda.empty_cache()

    def report_device_id(self) -> str:
        from vllm.platforms import current_platform

        self.device_uuid = current_platform.get_device_uuid(self.device.index)
        return self.device_uuid

    def _identify_payload_type(self, payload) -> PayloadType:
        if isinstance(payload, list):
            return PayloadType.HANDLES
        elif isinstance(payload, tuple):
            buf_id, _ = payload
            if buf_id is None:
                return PayloadType.DONE
            return PayloadType.BUFFER_UPDATE
        return PayloadType.UNKNOWN

    def check_weights_changed(self):
        """
        Check if the weights are updated to 0.
        """
        weights_updated = True
        for name, p in self.model_runner.model.named_parameters():
            weights_updated = weights_updated and torch.allclose(p, torch.zeros_like(p))
        return weights_updated
