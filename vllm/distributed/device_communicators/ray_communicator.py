# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any, Optional

import ray
import torch
from ray.exceptions import RayChannelError
from ray.experimental.channel import (AcceleratorContext, Communicator,
                                      TorchTensorAllocator)
from torch.distributed import ReduceOp

from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
from vllm.distributed.device_communicators.pynccl_wrapper import NCCLLibrary
from vllm.distributed.utils import StatelessProcessGroup
from vllm.logger import init_logger

logger = init_logger(__name__)


class RayStatelessProcessGroup(StatelessProcessGroup):
    """
    StatelessProcessGroup for Ray.
    
    This only holds information about the group, and does not do any
    communication.
    """

    def __init__(self, rank: int, world_size: int):
        super().__init__(rank, world_size, None, None)

    def get_rank(self) -> int:
        return self.rank


class RayCudaCommunicator(Communicator):
    """
    Communicator for a group of Ray Compiled Graph actors on NVIDIA GPU.
    This is based on the PyNCCL communicator.

    The Ray Compiled Graph execution uses this communicator to support
    communication between actors in the group.

    This class is not thread-safe.
    """

    _nccl = NCCLLibrary()

    def __init__(
        self,
        world_size: int,
        comm_id: Any,
        rank: Optional[int],
        actor_handles: list["ray.actor.ActorHandle"],
        cuda_stream: Optional[torch.cuda.Stream],
        use_communication_streams: bool = False,
    ):
        """
        Initialize a RayCudaCommunicator that can be used to communicate with
        other GPU actors.

        This method blocks until the same call has been made on all other
        actors in the group, with the same arguments for world_size and
        comm_id.

        NOTE: A concurrent RayCudaCommunicator can coexist with this one
        but using the two groups concurrently on different CUDA streams
        may cause deadlock.
        See
        https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/
        communicators.html#using-multiple-nccl-communicators-concurrently.

        If the user can guarantee that all involved actors execute the same ops
        in the same order, then the other RayCudaCommunicator should use the
        given `cuda_stream`, and there will not be a concurrency issue.
        Otherwise, the other stream needs to synchronize with the given
        `cuda_stream` before and after it launches NCCL ops, e.g., at the
        beginning and end of a DAG task.

        Args:
            world_size: The number of participating actors/devices.
            comm_id: A unique communicator ID returned by ncclGetUniqueId().
            rank: The rank of this actor. If None, then the caller is not a
                participant of the RayCudaCommunicator group.
            actor_handles: A list of actor handles, in rank order.
            cuda_stream: A CUDA stream to dispatch NCCL ops to. If rank is
                specified, then this must be specified too.
            use_communication_streams: Whether to use dedicated send and recv
                streams for communication. If True, communication and
                computation can be overlapped to improve performance.
        """
        self._world_size = world_size
        self._rank: Optional[int] = rank
        self._actor_handles = actor_handles
        self._use_communication_streams = use_communication_streams

        device = None
        if rank is not None:
            assert ray.get_gpu_ids(
            ), "RayCudaCommunicator has no GPUs assigned"
            assert cuda_stream is not None, (
                "RayCudaCommunicator must specify cuda_stream")

            expected_rank = self.get_rank(
                ray.get_runtime_context().current_actor)
            assert (
                rank == expected_rank), f"RayCudaCommunicator's rank {rank} "
            f"does not match expected rank {expected_rank}"

            pg = RayStatelessProcessGroup(rank, world_size)
            device = AcceleratorContext.get().get_accelerator_devices()[0]
            self._pynccl: Optional[PyNcclCommunicator] = PyNcclCommunicator(
                pg, device=device, unique_id=comm_id)
        else:
            # Driver does not have a rank.
            self._pynccl: Optional[PyNcclCommunicator] = None

        self._cuda_stream: Optional[torch.cuda.Stream] = None
        self._send_stream: Optional[torch.cuda.Stream] = None
        self._recv_stream: Optional[torch.cuda.Stream] = None
        if cuda_stream is not None:
            assert rank is not None, "Actor has no rank assigned"
            self._cuda_stream = cuda_stream

            if use_communication_streams:
                assert device is not None, "Device should have been set"
                self._send_stream = torch.cuda.Stream(device=device)
                self._recv_stream = torch.cuda.Stream(device=device)
            else:
                self._send_stream = self._cuda_stream
                self._recv_stream = self._cuda_stream

        self._closed = False

    def initialize(self, rank: int) -> None:
        # No additional initialization is needed.
        pass

    def get_actor_handles(self) -> list["ray.actor.ActorHandle"]:
        return self._actor_handles

    def get_rank(self, actor: ray.actor.ActorHandle) -> int:
        """
        Return the given actor's rank in the communicator.

        Args:
            actor: The actor handle to look up.
        """
        actor_ids = [a._ray_actor_id for a in self._actor_handles]
        try:
            rank = actor_ids.index(actor._ray_actor_id)
        except ValueError as e:
            raise ValueError(
                "Actor is not in the RayCudaCommunicator group.") from e
        return rank

    def get_self_rank(self) -> Optional[int]:
        """
        Return this actor's rank.
        """
        return self._rank

    def get_world_size(self) -> int:
        """
        Return the number of ranks in the RayCudaCommunicator group.
        """
        return self._world_size

    def send(self, buf: "torch.Tensor", peer_rank: int) -> None:
        """
        Send a torch.Tensor to a peer.

        This returns when the send kernel has been queued, but the kernel may
        not have completed. Therefore, the caller should ensure that there are
        no concurrent writes to the sent `buf` until the send has finished.
        That is, either all writes should be submitted on the current stream
        (self._cuda_stream) or, if on a different stream, that stream should
        synchronize with the current stream.

        Args:
            buf: The torch.Tensor to send. It should already be on this
                actor's default device.
            peer_rank: The rank of the actor to send to.
        """
        if self._closed:
            raise RayChannelError("RayCudaCommunicator has been destroyed.")

        if self._use_communication_streams:
            assert self._send_stream is not None
            # We observed that if all recv/compute/send operations run on GPU,
            # since there is no synchronization, the CPU execution loop may be
            # far ahead of the GPU operations and lead to runtime failures.
            # To avoid that, we synchronize on the send stream.
            # TODO(rui): find a better approach
            self._send_stream.synchronize()

        self._pynccl.send(buf, peer_rank, stream=self._send_stream)

    def recv(
        self,
        shape: tuple[int],
        dtype: "torch.dtype",
        peer_rank: int,
        allocator: TorchTensorAllocator,
    ) -> "torch.Tensor":
        """
        Receive a torch.Tensor from a peer and synchronize the current stream.

        After this call returns, the receive buffer is safe to read from from
        any stream. An RayChannelError will be raised if an error occurred
        (e.g., remote actor died), and the buffer is not safe to read.

        Args:
            shape: The shape of the tensor to receive.
            dtype: The dtype of the tensor to receive.
            peer_rank: The rank of the actor to receive from.
            allocator: The allocator to use to create the received tensor.
        """
        if self._closed:
            raise RayChannelError("RayCudaCommunicator has been destroyed.")
        assert allocator is not None, (
            "RayCudaCommunicator requires a tensor allocator")
        buf = allocator(shape, dtype)

        if self._use_communication_streams:
            assert self._recv_stream is not None
            # We observed that if all recv/compute/send operations run on GPU,
            # since there is no synchronization, the CPU execution loop may be
            # far ahead of the GPU operations and lead to runtime failures.
            # To avoid that, we synchronize on the recv stream.
            # TODO(rui): find a better approach
            self._recv_stream.synchronize()

            self._pynccl.recv(buf, peer_rank, stream=self._recv_stream)
        else:
            self._pynccl.recv(buf, peer_rank, stream=self._recv_stream)

            assert self._cuda_stream is not None
            # Buffer values are undefined if NCCL ops are aborted. Therefore, we
            # need to synchronize here and check that the channel is still
            # open to ensure that the receive buffer is valid.
            # TODO(swang): Avoid CUDA synchronization.
            self._cuda_stream.synchronize()

        if self._closed:
            raise RayChannelError("RayCudaCommunicator has been destroyed.")
        return buf

    def allgather(
        self,
        send_buf: "torch.Tensor",
        recv_buf: "torch.Tensor",
    ):
        # TODO(rui): in vLLM, Compiled Graph does not use collectives
        # probably want to just raise NotImplementedError
        self._pynccl.all_gather(recv_buf, send_buf, stream=self._cuda_stream)

    def allreduce(
        self,
        send_buf: "torch.Tensor",
        recv_buf: "torch.Tensor",
        op: ReduceOp = ReduceOp.SUM,
    ):
        out_tensor = self._pynccl.all_reduce(send_buf,
                                             op=op,
                                             stream=self._cuda_stream)
        if out_tensor is not None:
            recv_buf.copy_(out_tensor)

    def reducescatter(
        self,
        send_buf: "torch.Tensor",
        recv_buf: "torch.Tensor",
        op: ReduceOp = ReduceOp.SUM,
    ):
        self._pynccl.reduce_scatter(recv_buf,
                                    send_buf,
                                    op=op,
                                    stream=self._cuda_stream)

    @property
    def recv_stream(self):
        return torch.cuda.StreamContext(self._recv_stream)

    @property
    def send_stream(self):
        return torch.cuda.StreamContext(self._send_stream)

    def destroy(self) -> None:
        """
        Destroy the RayCudaCommunicator.
        """
        if self._closed:
            return
        self._closed = True

        if self._pynccl is not None:
            logger.info("Destructing RayCudaCommunicator on actor: %s",
                        ray.get_runtime_context().current_actor)
            self._pynccl = None

    def get_transport_name(self) -> str:
        return "nccl"

    @classmethod
    def generate_communicator_id(cls) -> Any:
        return cls._nccl.ncclGetUniqueId()
