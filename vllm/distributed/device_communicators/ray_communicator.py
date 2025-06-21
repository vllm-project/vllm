# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import uuid
from typing import Optional, list, tuple

import ray
import torch
from ray.exceptions import RayChannelError
from ray.experimental.channel.accelerator_context import AcceleratorContext
from ray.experimental.channel.communicator import (Communicator,
                                                   TorchTensorAllocator)
from ray.experimental.util.types import ReduceOp

from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
from vllm.distributed.utils import StatelessProcessGroup
from vllm.logger import init_logger

logger = init_logger(__name__)


class RayCudaCommunicator(Communicator):
    """
    Communicator for a group of Compiled Graph actors on NVIDIA GPU.

    The Compiled Graph execution leverages this internally to support
    communication between actors in the group.
    """

    def __init__(
        self,
        world_size: int,
        comm_id: tuple,
        rank: Optional[int],
        actor_handles: list["ray.actor.ActorHandle"],
        cuda_stream: Optional["torch.cuda.Stream"],
        use_communication_streams: bool = False,
    ):
        self._world_size = world_size
        self._rank: Optional[int] = rank
        self._actor_handles = actor_handles
        self._use_communication_streams = use_communication_streams

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

            # FIXME(rui): fix host and port, or revamp CPU group integration
            pg = StatelessProcessGroup.create(
                host="localhost",
                port=12345,
                world_size=world_size,
                rank=rank,
                data_expiration_seconds=3600,
                store_timeout=300,
            )
            device = AcceleratorContext.get().get_accelerator_devices()[0]
            self._pynccl = PyNcclCommunicator(pg, device=device)
        else:
            # Driver does not have a rank.
            self._pynccl = None

        self._cuda_stream: Optional[torch.cuda.Stream] = None
        self._send_stream: Optional[torch.cuda.Stream] = None
        self._recv_stream: Optional[torch.cuda.Stream] = None
        if cuda_stream is not None:
            assert rank is not None, "NCCL actor has no rank assigned"
            self._cuda_stream = cuda_stream

            if use_communication_streams:
                import torch

                device = AcceleratorContext.get().get_accelerator_devices()[0]

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
        Return the given actor's rank in the NCCL communicator.

        Args:
            actor: The actor handle to look up.
        """
        actor_ids = [a._ray_actor_id for a in self._actor_handles]
        try:
            rank = actor_ids.index(actor._ray_actor_id)
        except ValueError:
            raise ValueError("Actor is not in the NCCL group.")
        return rank

    def get_self_rank(self) -> Optional[int]:
        """
        Return this actor's rank.
        """
        return self._rank

    def get_world_size(self) -> int:
        """
        Return the number of ranks in the RayCudaCommunicator.
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
            # We observed that if all recv/compute/send operations run on GPU,
            # since there is no synchronization, the CPU execution loop may be
            # far ahead of the GPU operations and lead to runtime failures.
            # To avoid that, we synchronize on the send stream.
            # TODO(rui): find a better approach
            self._send_stream.synchronize()

        self._pynccl.send(buf, peer_rank, stream=self._send_stream)

    def recv(
        self,
        shape: Tuple[int],
        dtype: "torch.dtype",
        peer_rank: int,
        allocator=Optional[TorchTensorAllocator],
    ) -> "torch.Tensor":
        """
        Receive a torch.Tensor from a peer and synchronize the current stream.

        After this call returns, the receive buffer is safe to read from from
        any stream. An RayChannelError will be raised if an error occurred
        (e.g., remote actor died), and the buffer is not safe to read.

        Args:
            buf: The torch.Tensor to receive into. This buffer is safe to read
            peer_rank: The rank of the actor to receive from.
        """
        if self._closed:
            raise RayChannelError("RayCudaCommunicator has been destroyed.")
        assert allocator is not None, (
            "RayCudaCommunicator requires a tensor allocator")
        buf = allocator(shape, dtype)

        if self._use_communication_streams:
            # We observed that if all recv/compute/send operations run on GPU,
            # since there is no synchronization, the CPU execution loop may be
            # far ahead of the GPU operations and lead to runtime failures.
            # To avoid that, we synchronize on the recv stream.
            # TODO(rui): find a better approach
            self._recv_stream.synchronize()

            self._pynccl.recv(buf, peer_rank, stream=self._recv_stream)
        else:
            self._pynccl.recv(buf, peer_rank, stream=self._recv_stream)

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
        # TODO(rui): optimize the integration
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
        import torch

        return torch.cuda.StreamContext(self._recv_stream)

    @property
    def send_stream(self):
        import torch

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
            # TODO(rui): need to destroy the pynccl communicator?

    def get_transport_name(self) -> str:
        return "nccl"

    @classmethod
    def generate_communicator_id(cls) -> str:
        return str(uuid.uuid4())
