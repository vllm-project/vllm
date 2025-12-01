# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import uuid
from typing import Any

import ray
import torch
from ray.exceptions import RayChannelError
from ray.experimental.channel.communicator import Communicator, TorchTensorAllocator
from torch.distributed import ReduceOp

from vllm.distributed.device_communicators.base_device_communicator import (
    DeviceCommunicatorBase,
)
from vllm.distributed.parallel_state import get_pp_group
from vllm.logger import init_logger
from vllm.utils.torch_utils import current_stream

logger = init_logger(__name__)


class RayPPCommunicator(Communicator):
    """
    Communicator to be used for pipeline parallelism in Ray Compiled Graph.
    This is wraps around the vLLM _PP GroupCoordinator.

    This class is not thread-safe.
    """

    _comm: DeviceCommunicatorBase | None

    def __init__(
        self,
        world_size: int,
        comm_id: Any,
        rank: int | None,
        actor_handles: list["ray.actor.ActorHandle"],
        cuda_stream: torch.cuda.Stream | None,
        use_communication_streams: bool = False,
    ):
        """
        Initialize a RayPPCommunicator that can be used to communicate with
        other Ray Compiled Graph actors for pipeline parallelism.

        Args:
            world_size: The number of participating actors.
            comm_id: A unique communicator ID. This is just to conform with
                the Ray Communicator API and is not used.
            rank: The rank of this actor. If None, then the caller is not a
                participant of the RayPPCommunicator group (e.g., the Ray
                driver).
            actor_handles: A list of actor handles.
            cuda_stream: A CUDA stream to dispatch communication ops to. This
                is not supported.
            use_communication_streams: Whether to use communication streams.
                This is not supported.
        """
        self._world_size = world_size
        self._rank: int | None = None
        self._actor_handles = actor_handles
        if use_communication_streams:
            raise NotImplementedError("use_communication_streams is not supported")
        if cuda_stream is not None and cuda_stream != current_stream():
            raise ValueError(
                "cuda_stream other than the current stream is not supported"
            )

        if rank is not None:
            # Rank is not None, this is Ray worker
            assert ray.get_gpu_ids(), "RayPPCommunicator has no GPUs assigned"

            self._comm = get_pp_group().device_communicator
            assert self._comm is not None

            # Since we wrap around the vLLM _PP communicator, we use
            # the rank from the vLLM communicator, and ignore the rank
            # passed in from Ray.
            # TODO(rui): refactor the Ray Communicator API so that
            # it also supports no rank passed in.
            self._rank = self._comm.rank_in_group

            self._build_actor_rank_mapping()
        else:
            # Rank is None, this is Ray driver
            self._comm = None

        self._closed = False

    def _build_actor_rank_mapping(self):
        """
        Use collective communication to build a mapping from actor IDs to ranks.
        This should be called once during initialization.
        """
        if self._comm is None:
            return {}

        current_actor = ray.get_runtime_context().current_actor
        actor_id_str = current_actor._actor_id.hex()

        # Ray actor IDs are 32-character hex strings (128 bits)
        ACTOR_ID_LEN = 32
        actor_id_bytes = bytearray(actor_id_str.encode("utf-8"))
        assert len(actor_id_bytes) == ACTOR_ID_LEN, (
            f"Unexpected actor ID length: {len(actor_id_bytes)}"
        )

        actor_id_tensor = torch.frombuffer(actor_id_bytes, dtype=torch.uint8).to(
            self._comm.device
        )

        # All-gather full actor IDs from all actors
        gathered_ids = self._comm.all_gather(actor_id_tensor, dim=0)

        # Build mapping: actor_id -> device_comm_rank
        self._actor_id_to_rank = {}
        for rank in range(self._world_size):
            start_idx = rank * ACTOR_ID_LEN
            end_idx = (rank + 1) * ACTOR_ID_LEN
            actor_bytes = gathered_ids[start_idx:end_idx].cpu().numpy().tobytes()
            actor_id = actor_bytes.decode("utf-8")
            self._actor_id_to_rank[actor_id] = rank

    def initialize(self, rank: int) -> None:
        # No additional initialization is needed.
        pass

    def get_actor_handles(self) -> list["ray.actor.ActorHandle"]:
        return self._actor_handles

    def get_rank(self, actor: ray.actor.ActorHandle) -> int:
        """
        Return the given actor's rank using device communicator collective ops.
        """
        assert hasattr(self, "_actor_id_to_rank"), (
            "Actor rank mapping not built. "
            "This should have been done during initialization."
        )

        actor_id_str = actor._actor_id.hex()

        if actor_id_str in self._actor_id_to_rank:
            return self._actor_id_to_rank[actor_id_str]  # type: ignore
        else:
            raise ValueError(f"Actor {actor} not found in communicator group")

    def get_self_rank(self) -> int | None:
        """
        Return this actor's rank.
        """
        return self._rank

    def get_world_size(self) -> int:
        """
        Return the number of ranks in the RayPPCommunicator group.
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
            raise RayChannelError("RayPPCommunicator has been destroyed.")

        assert self._comm is not None
        self._comm.send(buf, peer_rank)

    def recv(
        self,
        shape: tuple[int, ...],
        dtype: "torch.dtype",
        peer_rank: int,
        allocator: TorchTensorAllocator,
    ) -> "torch.Tensor":
        """
        Receive a torch.Tensor from a peer and synchronize the current stream.

        After this call returns, the receive buffer is safe to read from
        any stream. An RayChannelError will be raised if an error occurred
        (e.g., remote actor died), and the buffer is not safe to read.

        Args:
            shape: The shape of the tensor to receive.
            dtype: The dtype of the tensor to receive.
            peer_rank: The rank of the actor to receive from.
            allocator: The allocator to use to create the received tensor.
                This is ignored for this implementation.
        """
        if self._closed:
            raise RayChannelError("RayPPCommunicator has been destroyed.")

        assert self._comm is not None
        size = torch.Size(shape)
        buf = self._comm.recv(size, dtype, src=peer_rank)

        # Buffer values are undefined if NCCL ops are aborted. Therefore, we
        # need to synchronize here and check that the channel is still
        # open to ensure that the receive buffer is valid.
        # TODO(swang): Avoid CUDA synchronization.
        current_stream().synchronize()

        if self._closed:
            raise RayChannelError("RayPPCommunicator has been destroyed.")
        return buf

    def allgather(
        self,
        send_buf: "torch.Tensor",
        recv_buf: "torch.Tensor",
    ):
        raise NotImplementedError("allgather is not supported")

    def allreduce(
        self,
        send_buf: "torch.Tensor",
        recv_buf: "torch.Tensor",
        op: ReduceOp = ReduceOp.SUM,
    ):
        raise NotImplementedError("allreduce is not supported")

    def reducescatter(
        self,
        send_buf: "torch.Tensor",
        recv_buf: "torch.Tensor",
        op: ReduceOp = ReduceOp.SUM,
    ):
        raise NotImplementedError("reducescatter is not supported")

    @property
    def recv_stream(self):
        return torch.cuda.StreamContext(current_stream())

    @property
    def send_stream(self):
        return torch.cuda.StreamContext(current_stream())

    def destroy(self) -> None:
        # Just sets a flag, vLLM manages the lifecycle of the underlying
        # _PP GroupCoordinator.
        self._closed = True

    def get_transport_name(self) -> str:
        return "nccl"

    @classmethod
    def generate_communicator_id(cls) -> Any:
        return uuid.uuid4()
