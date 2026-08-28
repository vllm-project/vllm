# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import threading
from weakref import WeakValueDictionary

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from vllm.logger import init_logger

logger = init_logger(__name__)


class Cache:
    def __init__(self):
        self._cache: WeakValueDictionary = WeakValueDictionary()
        self._lock = threading.RLock()  # Reentrant lock for thread safety

    def get_or_create(self, kwargs, func):
        # Create a hashable key from the kwargs
        key = tuple(sorted((k, v) for k, v in kwargs.items()))

        with self._lock:
            instance = self._cache.get(key)
            if instance is None:
                instance = func(**kwargs)
                self._cache[key] = instance
            return instance


class All2AllManagerBase:
    rank: int
    world_size: int

    def __init__(self, cpu_group, tcp_store_group=None):
        self.cpu_group = cpu_group
        self.tcp_store_group = tcp_store_group

        # compute some common properties
        from vllm.distributed.parallel_state import (
            get_dp_group,
            get_tp_group,
            in_the_same_node_as,
        )

        # all2all lives in ep group, which is merged from dp and tp group
        self.dp_group = get_dp_group()
        self.tp_group = get_tp_group()

        # no self.ep_group since self.ep_group is still in construction
        # when we create this object
        self.dp_rank = self.dp_group.rank_in_group
        self.dp_world_size = self.dp_group.world_size
        self.rank = cpu_group.rank()
        self.world_size = cpu_group.size()

        # all2all communication often has separate implementations for
        # intra-node and inter-node communication
        if tcp_store_group is None:
            self.internode = not all(in_the_same_node_as(cpu_group, source_rank=0))
        else:
            self.internode = not all(
                in_the_same_node_as(tcp_store_group, source_rank=0)
            )

        self.support_fault_tolerance = False

    def get_handle(self, kwargs):
        # get a handle for the all2all communication,
        # based on the kwargs.
        # different layers can have different configs,
        # e.g. one layer has hidden size 1024, another has 2048.
        # usually the underlying implementation caches the handle
        # and reuse it for the same config.
        raise NotImplementedError

    def dispatch_router_logits(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        is_sequence_parallel: bool = False,
        extra_tensors: list[torch.Tensor] | None = None,
    ) -> (
        tuple[torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]
    ):
        # Subclasses should either:
        # - implement handling for extra_tensors, or
        # - raise a clear error if extra_tensors is not supported.
        raise NotImplementedError

    def dispatch(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        is_sequence_parallel: bool = False,
        extra_tensors: list[torch.Tensor] | None = None,
    ) -> (
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[torch.Tensor]]
    ):
        # Subclasses should either:
        # - implement handling for extra_tensors, or
        # - raise a clear error if extra_tensors is not supported.
        raise NotImplementedError

    def query_active_mask(self) -> torch.Tensor:
        """Return the all2all liveness mask for the EP ranks.

        Returns:
            An int32 device tensor where 0 marks a live rank and 1 marks a
            masked (dead/unreachable) rank.
        """
        raise NotImplementedError

    def query_fault(self) -> torch.Tensor:
        """Return a scalar bool tensor, True if a new fault appeared.

        Compares the current mask against the baseline recorded at the last
        recovery point.
        """
        raise NotImplementedError

    def clean_buffers(self) -> None:
        """Reset this rank's RDMA buffers and all2all mask state (rank-local).

        Post-fault cleanup: a dispatch/combine that hit a dead peer or timed
        out can leave partially-written or stale tokens in the RDMA receive
        buffer, so it is zeroed to stop the next forward from reading that
        contaminated data.
        """
        raise NotImplementedError

    def set_num_sms(self, num_sms: int):
        pass

    def max_sms_used(self) -> int | None:
        return None  # None means it could use the whole GPU

    def checkpoint_prepare(self) -> None:
        logger.warning_once(
            "%s.checkpoint_prepare is not implemented; skipping.",
            type(self).__name__,
        )

    def checkpoint_restore(self) -> None:
        logger.warning_once(
            "%s.checkpoint_restore is not implemented; skipping.",
            type(self).__name__,
        )

    def combine(self, hidden_states: torch.Tensor, is_sequence_parallel: bool = False):
        raise NotImplementedError

    def destroy(self):
        pass


class DeviceCommunicatorBase:
    """
    Base class for device-specific communicator.
    It can use the `cpu_group` to initialize the communicator.
    If the device has PyTorch integration (PyTorch can recognize its
    communication backend), the `device_group` will also be given.
    """

    def __init__(
        self,
        cpu_group: ProcessGroup,
        device: torch.device | None = None,
        device_group: ProcessGroup | None = None,
        unique_name: str = "",
        global_ranks: list[int] | None = None,
        global_world_size: int | None = None,
        use_all2all: bool = False,
    ):
        self.device = device or torch.device("cpu")
        self.cpu_group = cpu_group
        self.device_group = device_group
        self.unique_name = unique_name

        # Check if this is a stateless process group
        from torch.distributed.distributed_c10d import _world

        is_stateless = _world.pg_map.get(cpu_group, None) is None

        if is_stateless:
            # For stateless groups, we can't use torch.distributed methods
            self.rank = cpu_group.rank()
            self.world_size = cpu_group.size()
            assert global_ranks is not None
            assert global_world_size is not None
            self.ranks = global_ranks
            self.global_rank = self.ranks[self.rank]
            self.global_world_size = global_world_size
            self.rank_in_group = self.rank
        else:
            self.rank = dist.get_rank(cpu_group)
            self.world_size = dist.get_world_size(cpu_group)
            self.ranks = dist.get_process_group_ranks(cpu_group)
            self.global_rank = dist.get_rank()
            self.global_world_size = dist.get_world_size()
            self.rank_in_group = dist.get_group_rank(self.cpu_group, self.global_rank)

        all2all_backend = None
        from vllm.config import get_current_vllm_config_or_none

        config = get_current_vllm_config_or_none()
        if config is not None:
            all2all_backend = config.parallel_config.all2all_backend

        self.is_ep_communicator = unique_name.split(":")[0] == "ep"
        self.use_all2all = self.is_ep_communicator and use_all2all
        self.all2all_backend = all2all_backend
        self.all2all_manager: All2AllManagerBase | None = None

    # NOTE(fallback-collectives): Gloo is a CPU-only distributed backend.
    # The fallback collective methods below (all_reduce, all_gather, etc.)
    # stage tensors through host memory (.cpu() / .copy_()). For simplicity
    # and minimal memory footprint in this fallback path, standard pageable
    # host memory is used. A future optimization can introduce a pinned memory
    # staging buffer pool for asynchronous D2H/H2D transfers to improve throughput.
    def _is_gloo(self) -> bool:
        return (
            self.device_group is not None
            and dist.get_backend(self.device_group) == dist.Backend.GLOO
        )

    def all_reduce(self, input_: torch.Tensor) -> torch.Tensor:
        if self._is_gloo():
            cpu_input = input_.cpu()
            dist.all_reduce(cpu_input, group=self.device_group)
            input_.copy_(cpu_input)
            return input_
        dist.all_reduce(input_, group=self.device_group)
        return input_

    def checkpoint_prepare(self) -> None:
        """Prepare reclaimable communicator state for checkpoint (default: no-op)."""

    def checkpoint_restore(self) -> None:
        """Restore communicator state after checkpoint (default: no-op)."""

    def all_gather(self, input_: torch.Tensor, dim: int = -1) -> torch.Tensor:
        if dim < 0:
            # Convert negative dim to positive.
            dim += input_.dim()
        input_size = input_.size()
        # NOTE: we have to use concat-style all-gather here,
        # stack-style all-gather has compatibility issues with
        # torch.compile . see https://github.com/pytorch/pytorch/issues/138795
        output_size = (input_size[0] * self.world_size,) + input_size[1:]
        # Allocate output tensor.
        output_tensor = torch.empty(
            output_size, dtype=input_.dtype, device=input_.device
        )
        if self._is_gloo():
            cpu_input = input_.contiguous().cpu()
            cpu_output = torch.empty(output_size, dtype=input_.dtype, device="cpu")
            dist.all_gather_into_tensor(cpu_output, cpu_input, group=self.device_group)
            output_tensor.copy_(cpu_output)
        else:
            # All-gather.
            dist.all_gather_into_tensor(output_tensor, input_, group=self.device_group)
        # Reshape
        output_tensor = output_tensor.reshape((self.world_size,) + input_size)
        output_tensor = output_tensor.movedim(0, dim)
        return output_tensor.reshape(
            input_size[:dim]
            + (self.world_size * input_size[dim],)
            + input_size[dim + 1 :]
        )

    def all_gatherv(
        self,
        input_: torch.Tensor | list[torch.Tensor],
        dim: int = 0,
        sizes: list[int] | None = None,
    ) -> torch.Tensor | list[torch.Tensor]:
        if dim != 0:
            raise NotImplementedError("only dim 0 all-gatherv is supported")
        world_size = self.world_size
        if world_size == 1:
            if isinstance(input_, torch.Tensor):
                return input_
            return list(input_)

        if sizes is not None and all(s == sizes[0] for s in sizes):
            sizes = None

        def _all_gather_single(
            inp: torch.Tensor, sizes: list[int] | None = None
        ) -> torch.Tensor:
            input_size = inp.size()
            if sizes is not None:
                assert len(sizes) == world_size
                assert inp.shape[dim] == sizes[self.rank_in_group], (
                    f"{inp.shape[dim]} != {sizes[self.rank_in_group]}"
                )
                output_size = (sum(sizes),) + input_size[1:]
            else:
                output_size = (input_size[0] * world_size,) + input_size[1:]

            output_tensor = torch.empty(output_size, dtype=inp.dtype, device=inp.device)
            if self._is_gloo():
                cpu_input = inp.contiguous().cpu()
                if sizes is not None:
                    max_s = max(sizes)
                    padded_input = torch.zeros(
                        (max_s,) + input_size[1:],
                        dtype=inp.dtype,
                        device="cpu",
                    )
                    padded_input[: inp.shape[0]].copy_(cpu_input)
                    padded_gather_list = [
                        torch.empty(
                            (max_s,) + input_size[1:],
                            dtype=inp.dtype,
                            device="cpu",
                        )
                        for _ in range(world_size)
                    ]
                    dist.all_gather(
                        padded_gather_list, padded_input, group=self.device_group
                    )
                    cpu_gather_list = [
                        padded_gather_list[i][: sizes[i]] for i in range(world_size)
                    ]
                    cpu_output = torch.cat(cpu_gather_list, dim=0)
                else:
                    cpu_output = torch.empty(output_size, dtype=inp.dtype, device="cpu")
                    dist.all_gather_into_tensor(
                        cpu_output, cpu_input, group=self.device_group
                    )
                output_tensor.copy_(cpu_output)
            else:
                # Native collective fallback for non-CUDA, non-GLOO platforms.
                if sizes is not None:
                    gather_list = [
                        torch.empty(
                            (s,) + input_size[1:],
                            dtype=inp.dtype,
                            device=inp.device,
                        )
                        for s in sizes
                    ]
                    dist.all_gather(
                        gather_list, inp.contiguous(), group=self.device_group
                    )
                    torch.cat(gather_list, dim=0, out=output_tensor)
                else:
                    dist.all_gather_into_tensor(
                        output_tensor,
                        inp.contiguous(),
                        group=self.device_group,
                    )
            return output_tensor

        if isinstance(input_, torch.Tensor):
            return _all_gather_single(input_, sizes)
        return [_all_gather_single(inp, sizes) for inp in input_]

    def reduce_scatter(self, input_: torch.Tensor, dim: int = -1) -> torch.Tensor:
        world_size = self.world_size
        # Bypass the function if we are using only 1 GPU.
        if world_size == 1:
            return input_
        assert -input_.dim() <= dim < input_.dim(), (
            f"Invalid dim ({dim}) for input tensor with shape {input_.size()}"
        )

        if dim < 0:
            # Convert negative dim to positive.
            dim += input_.dim()

        # Note: This will produce an incorrect answer if we don't make
        # the input_tensor contiguous. Possible bug in reduce_scatter_tensor?
        input_tensor = input_.movedim(0, dim).contiguous()

        assert input_tensor.shape[0] % world_size == 0
        chunk_size = input_tensor.shape[0] // world_size
        output_shape = (chunk_size,) + input_tensor.shape[1:]

        output_tensor = torch.empty(
            output_shape, dtype=input_tensor.dtype, device=input_tensor.device
        )

        if self._is_gloo():
            cpu_input = input_tensor.cpu()
            cpu_output = torch.empty(
                output_shape, dtype=input_tensor.dtype, device="cpu"
            )
            dist.reduce_scatter_tensor(cpu_output, cpu_input, group=self.device_group)
            output_tensor.copy_(cpu_output)
        else:
            # Perform reduce-scatter operation
            torch.distributed.reduce_scatter_tensor(
                output_tensor, input_tensor, group=self.device_group
            )

        # Reshape before returning
        return output_tensor.movedim(0, dim).contiguous()

    def reduce_scatterv(
        self, input_: torch.Tensor, dim: int = -1, sizes: list[int] | None = None
    ) -> torch.Tensor:
        world_size = self.world_size
        if world_size == 1:
            return input_
        assert -input_.dim() <= dim < input_.dim(), (
            f"Invalid dim ({dim}) for input tensor with shape {input_.size()}"
        )
        if dim < 0:
            dim += input_.dim()

        input_tensor = input_.movedim(0, dim).contiguous()

        if sizes is not None:
            assert len(sizes) == world_size, f"{len(sizes)} == {world_size}"
            assert input_tensor.shape[0] == sum(sizes)
            chunk_size = sizes[self.rank_in_group]
        else:
            assert input_tensor.shape[0] % world_size == 0
            chunk_size = input_tensor.shape[0] // world_size

        output_shape = (chunk_size,) + input_tensor.shape[1:]
        output = torch.empty(
            output_shape, dtype=input_tensor.dtype, device=input_tensor.device
        )

        if self._is_gloo():
            cpu_input = input_tensor.cpu()
            cpu_output = torch.empty(
                output_shape, dtype=input_tensor.dtype, device="cpu"
            )
            if sizes is not None and sizes.count(sizes[0]) != len(sizes):
                # PyTorch Gloo does not support uneven list-of-splits in
                # reduce_scatter, so perform all_reduce then slice locally.
                dist.all_reduce(cpu_input, group=self.device_group)
                offset = sum(sizes[: self.rank_in_group])
                cpu_output = cpu_input[offset : offset + chunk_size]
            else:
                dist.reduce_scatter_tensor(
                    cpu_output, cpu_input, group=self.device_group
                )
            output.copy_(cpu_output)
        else:
            # Native collective fallback for non-CUDA, non-GLOO platforms.
            if sizes is not None and sizes.count(sizes[0]) != len(sizes):
                input_splits = list(input_tensor.split(sizes, dim=0))
                dist.reduce_scatter(output, input_splits, group=self.device_group)
            else:
                dist.reduce_scatter_tensor(
                    output, input_tensor, group=self.device_group
                )

        return output.movedim(0, dim).contiguous()

    def gather(
        self, input_: torch.Tensor, dst: int = 0, dim: int = -1
    ) -> torch.Tensor | None:
        """
        NOTE: We assume that the input tensor is on the same device across
        all the ranks.
        NOTE: `dst` is the local rank of the destination rank.
        """
        world_size = self.world_size
        assert -input_.dim() <= dim < input_.dim(), (
            f"Invalid dim ({dim}) for input tensor with shape {input_.size()}"
        )
        if dim < 0:
            # Convert negative dim to positive.
            dim += input_.dim()

        if self._is_gloo():
            cpu_input = input_.cpu()
            if self.rank_in_group == dst:
                cpu_gather_list = [
                    torch.empty_like(cpu_input) for _ in range(world_size)
                ]
            else:
                cpu_gather_list = None
            torch.distributed.gather(
                cpu_input,
                cpu_gather_list,
                dst=self.ranks[dst],
                group=self.device_group,
            )
            if self.rank_in_group == dst:
                assert cpu_gather_list is not None
                cpu_output = torch.cat(cpu_gather_list, dim=dim)
                return cpu_output.to(input_.device)
            return None
        else:
            # Allocate output tensor.
            if self.rank_in_group == dst:
                gather_list = [torch.empty_like(input_) for _ in range(world_size)]
            else:
                gather_list = None
            # Gather.
            torch.distributed.gather(
                input_, gather_list, dst=self.ranks[dst], group=self.device_group
            )
            if self.rank_in_group == dst:
                assert gather_list is not None
                output_tensor = torch.cat(gather_list, dim=dim)
            else:
                output_tensor = None
            return output_tensor

    def send(self, tensor: torch.Tensor, dst: int | None = None) -> None:
        """Sends a tensor to the destination rank in a blocking way"""
        """NOTE: `dst` is the local rank of the destination rank."""
        if dst is None:
            dst = (self.rank_in_group + 1) % self.world_size
        if self._is_gloo():
            cpu_tensor = tensor.cpu()
            torch.distributed.send(cpu_tensor, self.ranks[dst], self.device_group)
        else:
            torch.distributed.send(tensor, self.ranks[dst], self.device_group)

    def recv(
        self, size: torch.Size, dtype: torch.dtype, src: int | None = None
    ) -> torch.Tensor:
        """Receives a tensor from the source rank."""
        """NOTE: `src` is the local rank of the source rank."""
        if src is None:
            src = (self.rank_in_group - 1) % self.world_size

        if self._is_gloo():
            cpu_tensor = torch.empty(size, dtype=dtype, device="cpu")
            torch.distributed.recv(cpu_tensor, self.ranks[src], self.device_group)
            return cpu_tensor.to(self.device)
        else:
            tensor = torch.empty(size, dtype=dtype, device=self.device)
            torch.distributed.recv(tensor, self.ranks[src], self.device_group)
            return tensor

    def broadcast(self, tensor: torch.Tensor, src: int = 0) -> torch.Tensor:
        """Broadcast a tensor from source rank to all ranks."""
        if self.world_size == 1:
            return tensor
        if self._is_gloo():
            cpu_tensor = tensor.cpu()
            torch.distributed.broadcast(cpu_tensor, self.ranks[src], self.device_group)
            tensor.copy_(cpu_tensor)
        else:
            torch.distributed.broadcast(tensor, self.ranks[src], self.device_group)
        return tensor

    def destroy(self):
        pass

    def dispatch_router_logits(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        is_sequence_parallel: bool = False,
        extra_tensors: list[torch.Tensor] | None = None,
    ) -> (
        tuple[torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]
    ):
        """
        Dispatch the hidden states and router logits to the appropriate device.
        This is a no-op in the base class.
        """
        if extra_tensors is not None:
            return hidden_states, router_logits, extra_tensors
        return hidden_states, router_logits

    def dispatch(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        is_sequence_parallel: bool = False,
        extra_tensors: list[torch.Tensor] | None = None,
    ) -> (
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[torch.Tensor]]
    ):
        """
        Dispatch the hidden states and topk weights/ids to the appropriate device.
        This is a no-op in the base class.
        """
        if extra_tensors is not None:
            return hidden_states, topk_weights, topk_ids, extra_tensors
        return hidden_states, topk_weights, topk_ids

    def combine(
        self, hidden_states: torch.Tensor, is_sequence_parallel: bool = False
    ) -> torch.Tensor:
        """
        Combine the hidden states and router logits from the appropriate device.
        This is a no-op in the base class.
        """
        return hidden_states

    def batch_isend_irecv(self, p2p_ops: list):
        raise NotImplementedError
