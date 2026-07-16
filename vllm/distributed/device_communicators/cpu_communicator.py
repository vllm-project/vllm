# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from typing import Any

import torch
from torch.distributed import ProcessGroup

from vllm.distributed.utils import pickle
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.platforms.interface import CpuArchEnum

from .base_device_communicator import DeviceCommunicatorBase

logger = init_logger(__name__)


class CpuCommunicator(DeviceCommunicatorBase):
    _CPUSHM_GROUP_KINDS = {"tp", "pp", "dp", "ep"}

    def __init__(
        self,
        cpu_group: ProcessGroup,
        device: torch.device | None = None,
        device_group: ProcessGroup | None = None,
        unique_name: str = "",
    ):
        super().__init__(cpu_group, device, device_group, unique_name)
        self.dist_module = torch.distributed
        use_cpushm_group = self._is_cpushm_group_name(unique_name)

        if (
            (
                current_platform.get_cpu_architecture() == CpuArchEnum.X86
                or current_platform.get_cpu_architecture() == CpuArchEnum.ARM
                or current_platform.get_cpu_architecture() == CpuArchEnum.POWERPC
            )
            and hasattr(torch.ops._C, "init_shm_manager")
            and use_cpushm_group
            and self._all_group_ranks_share_shm_group_name()
        ):
            self.dist_module = _CPUSHMDistributed(self)
        elif use_cpushm_group:
            logger.info(
                "CPU SHM communicator disabled for group %s: ranks do not share "
                "the same SHM group name, falling back to torch.distributed.",
                unique_name,
            )

        # send/recv tensor_dict is only supported through the SHM communicator backend
        self.supports_tensor_dict = isinstance(self.dist_module, _CPUSHMDistributed)
        # Ragged SHM all_gatherv needs temporary padded input and a uniform
        # receive buffer; cache them by tensor signature to avoid steady-state
        # reallocations.
        self._ragged_pad_buffers: dict[
            tuple[torch.dtype, torch.device, tuple[int, ...]], torch.Tensor
        ] = {}
        self._ragged_shm_gather_buffers: dict[
            tuple[torch.dtype, torch.device, tuple[int, ...]], torch.Tensor
        ] = {}

        if self.use_all2all:
            if self.all2all_backend not in (
                "naive",
                "allgather_reducescatter",
            ):  # type: ignore[has-type]
                logger.warning(
                    "`%s` all2all manager is not supported on CPU. "
                    "Falling back to `allgather_reducescatter` manager.",
                    self.all2all_backend,  # type: ignore[has-type]
                )
            from .all2all import AgRsAll2AllManager

            self.all2all_manager = AgRsAll2AllManager(self.cpu_group)
            logger.info("Using allgather_reducescatter all2all manager.")
            self._register_ep_custom_ops()

    @classmethod
    def _is_cpushm_group_name(cls, unique_name: str) -> bool:
        group_kind = unique_name.split(":", maxsplit=1)[0]
        return group_kind in cls._CPUSHM_GROUP_KINDS

    def _all_group_ranks_share_shm_group_name(self) -> bool:
        """
        CPUSHM requires all ranks in this group to agree on one SHM group name.
        This is a lightweight consistency check for VLLM_DIST_IDENT/name inputs.
        """
        local_name = _CPUSHMDistributed.make_group_name(self)
        names: list[str] = [""] * self.world_size
        torch.distributed.all_gather_object(
            names,
            local_name,
            group=self.device_group,
        )
        shared_name = len(set(names)) == 1
        if not shared_name:
            logger.debug(
                "CPU SHM group-name mismatch for group %s: %s",
                self.unique_name,
                names,
            )
        return shared_name

    def all_reduce(self, input_):
        self.dist_module.all_reduce(input_, group=self.device_group)
        return input_

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

        # Allocate output tensor.
        if self.rank_in_group == dst:
            gather_list = [torch.empty_like(input_) for _ in range(world_size)]
        else:
            gather_list = None

        # Gather.
        self.dist_module.gather(
            input_, gather_list, dst=self.ranks[dst], group=self.device_group
        )

        if self.rank_in_group == dst:
            output_tensor = torch.cat(gather_list, dim=dim)
        else:
            output_tensor = None
        return output_tensor

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
        # All-gather.
        self.dist_module.all_gather_into_tensor(
            output_tensor, input_, group=self.device_group
        )

        # Reshape
        output_tensor = output_tensor.reshape((self.world_size,) + input_size)
        output_tensor = output_tensor.movedim(0, dim)
        output_tensor = output_tensor.reshape(
            input_size[:dim]
            + (self.world_size * input_size[dim],)
            + input_size[dim + 1 :]
        )
        return output_tensor

    @staticmethod
    def _ragged_buffer_key(
        tensor: torch.Tensor,
    ) -> tuple[torch.dtype, torch.device, tuple[int, ...]]:
        return (tensor.dtype, tensor.device, tuple(tensor.shape[1:]))

    @staticmethod
    def _sizes_are_uniform(sizes: list[int]) -> bool:
        return all(size == sizes[0] for size in sizes[1:])

    def _get_ragged_pad_buffer(
        self,
        tensor: torch.Tensor,
        padded_rows: int,
    ) -> torch.Tensor:
        key = self._ragged_buffer_key(tensor)
        buffer = self._ragged_pad_buffers.get(key)
        if buffer is None or buffer.shape[0] < padded_rows:
            buffer = torch.empty(
                (padded_rows,) + tensor.shape[1:],
                dtype=tensor.dtype,
                device=tensor.device,
            )
            self._ragged_pad_buffers[key] = buffer
        return buffer[:padded_rows]

    def _get_ragged_shm_gather_buffer(
        self,
        tensor: torch.Tensor,
        padded_rows: int,
    ) -> torch.Tensor:
        key = self._ragged_buffer_key(tensor)
        needed_rows = self.world_size * padded_rows
        buffer = self._ragged_shm_gather_buffers.get(key)
        if buffer is None or buffer.shape[0] < needed_rows:
            buffer = torch.empty(
                (needed_rows,) + tensor.shape[1:],
                dtype=tensor.dtype,
                device=tensor.device,
            )
            self._ragged_shm_gather_buffers[key] = buffer
        return buffer[:needed_rows]

    def _all_gatherv_shm_uniform(
        self,
        tensor: torch.Tensor,
        rows_per_rank: int,
    ) -> torch.Tensor:
        output = torch.empty(
            (self.world_size * rows_per_rank,) + tensor.shape[1:],
            dtype=tensor.dtype,
            device=tensor.device,
        )
        self.dist_module.all_gather_into_tensor(
            output,
            tensor.contiguous(),
            group=self.device_group,
        )
        return output

    def _trim_ragged_rows(
        self,
        gathered: torch.Tensor,
        sizes: list[int],
        padded_rows: int,
    ) -> torch.Tensor:
        """Un-pad a (world_size * padded_rows, ...) buffer into
        (sum(sizes), ...) via bulk narrow+copy_, avoiding a Python-list
        select/cat."""
        total_rows = sum(sizes)
        output = torch.empty(
            (total_rows,) + gathered.shape[1:],
            dtype=gathered.dtype,
            device=gathered.device,
        )
        offset = 0
        for rank, n in enumerate(sizes):
            if n:
                output.narrow(0, offset, n).copy_(
                    gathered.narrow(0, rank * padded_rows, n)
                )
            offset += n
        return output

    def _all_gatherv_shm_ragged(
        self,
        tensor: torch.Tensor,
        sizes: list[int],
    ) -> torch.Tensor:
        padded_rows = max(sizes)
        padded_input = self._get_ragged_pad_buffer(tensor, padded_rows)
        padded_input[: tensor.shape[0]].copy_(tensor.contiguous())

        gathered = self._get_ragged_shm_gather_buffer(tensor, padded_rows)
        self.dist_module.all_gather_into_tensor(
            gathered,
            padded_input,
            group=self.device_group,
        )

        return self._trim_ragged_rows(gathered, sizes, padded_rows)

    def _pad_rows_for_gatherv(
        self,
        tensor: torch.Tensor,
        padded_rows: int,
    ) -> torch.Tensor:
        pad_rows = padded_rows - tensor.shape[0]
        assert pad_rows >= 0, f"{pad_rows} < 0"
        if pad_rows == 0:
            return tensor.contiguous()
        padding = torch.zeros(
            (pad_rows,) + tensor.shape[1:],
            dtype=tensor.dtype,
            device=tensor.device,
        )
        return torch.cat([tensor.contiguous(), padding], dim=0)

    def _gather_single_gatherv(
        self,
        t: torch.Tensor,
        sizes: list[int] | None,
    ) -> torch.Tensor:
        if sizes is not None:
            assert t.shape[0] == sizes[self.rank_in_group], (
                f"{t.shape[0]} != {sizes[self.rank_in_group]}"
            )
            if not any(sizes):
                return t.new_empty((0,) + t.shape[1:])

        if isinstance(self.dist_module, _CPUSHMDistributed):
            if sizes is None:
                return self._all_gatherv_shm_uniform(t, t.shape[0])
            if self._sizes_are_uniform(sizes):
                return self._all_gatherv_shm_uniform(t, sizes[0])
            return self._all_gatherv_shm_ragged(t, sizes)
        else:
            # gloo path (multi-node or non-SHM groups).
            if sizes is not None:
                max_size = max(sizes)
                t_padded = self._pad_rows_for_gatherv(t, max_size)
                recv_list = [
                    torch.empty(
                        (max_size,) + t.shape[1:], dtype=t.dtype, device=t.device
                    )
                    for _ in range(self.world_size)
                ]
                torch.distributed.all_gather(
                    recv_list, t_padded, group=self.device_group
                )
                gathered = torch.cat(recv_list, dim=0)
                return self._trim_ragged_rows(gathered, sizes, max_size)
            else:
                recv_list = [torch.empty_like(t) for _ in range(self.world_size)]
                torch.distributed.all_gather(
                    recv_list, t.contiguous(), group=self.device_group
                )
                return torch.cat(recv_list, dim=0)

    def all_gatherv(
        self,
        input_: torch.Tensor | list[torch.Tensor],
        dim: int = 0,
        sizes: list[int] | None = None,
    ) -> torch.Tensor | list[torch.Tensor]:
        """Variable-length all-gather over dim 0."""
        if dim != 0:
            raise NotImplementedError("CpuCommunicator.all_gatherv only supports dim=0")

        if sizes is not None:
            assert len(sizes) == self.world_size, f"{len(sizes)} != {self.world_size}"

        if isinstance(input_, torch.Tensor):
            return self._gather_single_gatherv(input_, sizes)
        return [self._gather_single_gatherv(t, sizes) for t in input_]

    def reduce_scatterv(
        self,
        input_: torch.Tensor,
        dim: int = 0,
        sizes: list[int] | None = None,
    ) -> torch.Tensor:
        """Reduce-scatter with variable output sizes via all_reduce + local slice."""
        if dim < 0:
            dim += input_.dim()

        if sizes is not None:
            assert len(sizes) == self.world_size, f"{len(sizes)} != {self.world_size}"
            assert input_.shape[dim] == sum(sizes), (
                f"{input_.shape[dim]} != {sum(sizes)}"
            )
        else:
            assert input_.shape[dim] % self.world_size == 0, (
                "Implicit reduce_scatterv requires the scatter dimension "
                f"{input_.shape[dim]} to be divisible by world_size "
                f"{self.world_size}."
            )

        out = input_.contiguous().clone()

        self.dist_module.all_reduce(out, group=self.device_group)

        if sizes is not None:
            start = sum(sizes[: self.rank_in_group])
            end = start + sizes[self.rank_in_group]
        else:
            chunk = out.shape[dim] // self.world_size
            start = self.rank_in_group * chunk
            end = start + chunk

        return out.narrow(dim, start, end - start).contiguous()

    def send_tensor_dict(
        self,
        tensor_dict: dict[str, torch.Tensor | Any],
        dst: int,
    ) -> None:
        if not self.supports_tensor_dict:
            raise NotImplementedError(
                "CpuCommunicator does not support tensor dict fastpath with "
                "torch.distributed backend."
            )
        return self.dist_module.send_tensor_dict(tensor_dict, dst)

    def recv_tensor_dict(
        self,
        src: int,
    ) -> dict[str, torch.Tensor | Any]:
        if not self.supports_tensor_dict:
            raise NotImplementedError(
                "CpuCommunicator does not support tensor dict fastpath with "
                "torch.distributed backend."
            )
        return self.dist_module.recv_tensor_dict(src)

    def _register_ep_custom_ops(self) -> None:
        """Register dispatch/combine as custom ops opaque to torch.compile."""
        assert self.all2all_manager is not None
        mgr = self.all2all_manager
        safe_name = (
            self.unique_name.replace("-", "_").replace("/", "_").replace(":", "_")
        )

        def _with_non_sp_dp_sizes(fn):
            dp_metadata = get_forward_context().dp_metadata
            assert dp_metadata is not None
            if dp_metadata.local_sizes is not None:
                return fn()
            with dp_metadata.sp_local_sizes(sequence_parallel_size=1):
                return fn()

        def _with_sp_sizes(fn):
            dp_metadata = get_forward_context().dp_metadata
            assert dp_metadata is not None
            dp_size = len(dp_metadata.num_tokens_across_dp_cpu)
            assert self.world_size % dp_size == 0
            sp_size = self.world_size // dp_size
            with dp_metadata.sp_local_sizes(sp_size):
                return fn()

        @torch.library.custom_op(
            f"vllm::cpu_ep_dispatch_rl_{safe_name}", mutates_args=()
        )
        def _dispatch_rl(
            hidden_states: torch.Tensor,
            router_logits: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            return _with_non_sp_dp_sizes(
                lambda: mgr.dispatch_router_logits(
                    hidden_states,
                    router_logits,
                )
            )

        @_dispatch_rl.register_fake
        def _(
            hidden_states: torch.Tensor, router_logits: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            # Ragged dispatch: the real op all-gathers each rank's actual row
            # count and returns sum(per_rank_sizes) rows. That total is
            # data-dependent and unknown at trace time, so emit an unbacked
            # symint instead of the uniform shape[0] * world_size (GPU parity).
            ctx = torch.library.get_ctx()
            n = ctx.new_dynamic_size()
            return (
                hidden_states.new_empty((n,) + hidden_states.shape[1:]),
                router_logits.new_empty((n,) + router_logits.shape[1:]),
            )

        @torch.library.custom_op(
            f"vllm::cpu_ep_dispatch_rl_sp_{safe_name}", mutates_args=()
        )
        def _dispatch_rl_sp(
            hidden_states: torch.Tensor,
            router_logits: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            return _with_sp_sizes(
                lambda: mgr.dispatch_router_logits(
                    hidden_states,
                    router_logits,
                    is_sequence_parallel=True,
                )
            )

        @_dispatch_rl_sp.register_fake
        def _(
            hidden_states: torch.Tensor,
            router_logits: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            ctx = torch.library.get_ctx()
            n = ctx.new_dynamic_size()
            return (
                hidden_states.new_empty((n,) + hidden_states.shape[1:]),
                router_logits.new_empty((n,) + router_logits.shape[1:]),
            )

        @torch.library.custom_op(f"vllm::cpu_ep_dispatch_{safe_name}", mutates_args=())
        def _dispatch(
            hidden_states: torch.Tensor,
            topk_weights: torch.Tensor,
            topk_ids: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            result = _with_non_sp_dp_sizes(
                lambda: mgr.dispatch(hidden_states, topk_weights, topk_ids)
            )
            return result[0], result[1], result[2]

        @_dispatch.register_fake
        def _(
            hidden_states: torch.Tensor,
            topk_weights: torch.Tensor,
            topk_ids: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            # Ragged dispatch: the real op all-gathers each rank's actual row
            # count and returns sum(per_rank_sizes) rows. That total is
            # data-dependent and unknown at trace time, so emit an unbacked
            # symint instead of the uniform shape[0] * world_size (GPU parity).
            # All three outputs share the same n so downstream sees one row dim.
            ctx = torch.library.get_ctx()
            n = ctx.new_dynamic_size()
            return (
                hidden_states.new_empty((n,) + hidden_states.shape[1:]),
                topk_weights.new_empty((n,) + topk_weights.shape[1:]),
                topk_ids.new_empty((n,) + topk_ids.shape[1:]),
            )

        @torch.library.custom_op(
            f"vllm::cpu_ep_dispatch_sp_{safe_name}", mutates_args=()
        )
        def _dispatch_sp(
            hidden_states: torch.Tensor,
            topk_weights: torch.Tensor,
            topk_ids: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            result = _with_sp_sizes(
                lambda: mgr.dispatch(
                    hidden_states,
                    topk_weights,
                    topk_ids,
                    is_sequence_parallel=True,
                )
            )
            return result[0], result[1], result[2]

        @_dispatch_sp.register_fake
        def _(
            hidden_states: torch.Tensor,
            topk_weights: torch.Tensor,
            topk_ids: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            ctx = torch.library.get_ctx()
            n = ctx.new_dynamic_size()
            return (
                hidden_states.new_empty((n,) + hidden_states.shape[1:]),
                topk_weights.new_empty((n,) + topk_weights.shape[1:]),
                topk_ids.new_empty((n,) + topk_ids.shape[1:]),
            )

        @torch.library.custom_op(f"vllm::cpu_ep_combine_{safe_name}", mutates_args=())
        def _combine(hidden_states: torch.Tensor) -> torch.Tensor:
            return _with_non_sp_dp_sizes(lambda: mgr.combine(hidden_states))

        @_combine.register_fake
        def _(hidden_states: torch.Tensor) -> torch.Tensor:
            ctx = torch.library.get_ctx()
            local_n = ctx.new_dynamic_size()
            return hidden_states.new_empty((local_n,) + hidden_states.shape[1:])

        @torch.library.custom_op(
            f"vllm::cpu_ep_combine_sp_{safe_name}", mutates_args=()
        )
        def _combine_sp(hidden_states: torch.Tensor) -> torch.Tensor:
            return _with_sp_sizes(
                lambda: mgr.combine(hidden_states, is_sequence_parallel=True)
            )

        @_combine_sp.register_fake
        def _(hidden_states: torch.Tensor) -> torch.Tensor:
            ctx = torch.library.get_ctx()
            local_n = ctx.new_dynamic_size()
            return hidden_states.new_empty((local_n,) + hidden_states.shape[1:])

        self._ep_dispatch_rl_op = _dispatch_rl
        self._ep_dispatch_rl_sp_op = _dispatch_rl_sp
        self._ep_dispatch_op = _dispatch
        self._ep_dispatch_sp_op = _dispatch_sp
        self._ep_combine_op = _combine
        self._ep_combine_sp_op = _combine_sp

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
        assert self.all2all_manager is not None
        if (
            extra_tensors is None
            and is_sequence_parallel
            and hasattr(self, "_ep_dispatch_rl_sp_op")
        ):
            return self._ep_dispatch_rl_sp_op(hidden_states, router_logits)
        if (
            extra_tensors is None
            and not is_sequence_parallel
            and hasattr(self, "_ep_dispatch_rl_op")
        ):
            return self._ep_dispatch_rl_op(hidden_states, router_logits)
        return self.all2all_manager.dispatch_router_logits(
            hidden_states, router_logits, is_sequence_parallel, extra_tensors
        )

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
        assert self.all2all_manager is not None
        if (
            extra_tensors is None
            and is_sequence_parallel
            and hasattr(self, "_ep_dispatch_sp_op")
        ):
            return self._ep_dispatch_sp_op(hidden_states, topk_weights, topk_ids)
        if (
            extra_tensors is None
            and not is_sequence_parallel
            and hasattr(self, "_ep_dispatch_op")
        ):
            return self._ep_dispatch_op(hidden_states, topk_weights, topk_ids)
        return self.all2all_manager.dispatch(
            hidden_states,
            topk_weights,
            topk_ids,
            is_sequence_parallel,
            extra_tensors=extra_tensors,
        )

    def combine(
        self, hidden_states: torch.Tensor, is_sequence_parallel: bool = False
    ) -> torch.Tensor:
        assert self.all2all_manager is not None
        if is_sequence_parallel and hasattr(self, "_ep_combine_sp_op"):
            return self._ep_combine_sp_op(hidden_states)
        if not is_sequence_parallel and hasattr(self, "_ep_combine_op"):
            return self._ep_combine_op(hidden_states)
        return self.all2all_manager.combine(hidden_states, is_sequence_parallel)


class _CPUSHMDistributed:
    def __init__(self, communicator: CpuCommunicator):
        self.communicator = communicator

        self.group_name = self.make_group_name(communicator)

        self.handle = self._init_cpu_shm()

    @staticmethod
    def make_group_name(communicator: CpuCommunicator) -> str:
        instance_identifier = os.environ["VLLM_DIST_IDENT"]
        unique_name = communicator.unique_name
        instance_identifier = f"{instance_identifier}-{unique_name}"
        group_ranks = [str(rank) for rank in communicator.ranks]
        shm_group_identifier = f"[{'-'.join(group_ranks)}]"
        return f"{instance_identifier}-{shm_group_identifier}-cpushm"

    def _init_cpu_shm(self) -> int:
        thread_num_tensor = torch.tensor(
            [torch.get_num_threads()],
            dtype=torch.int64,
        )
        torch.distributed.all_reduce(
            thread_num_tensor,
            op=torch.distributed.ReduceOp.MIN,
            group=self.communicator.device_group,
        )
        thread_num = thread_num_tensor.item()

        handle = torch.ops._C.init_shm_manager(
            self.group_name,
            self.communicator.world_size,
            self.communicator.rank,
            thread_num,
        )
        torch.distributed.barrier(self.communicator.device_group)
        torch.ops._C.join_shm_manager(
            handle,
            self.group_name,
        )
        torch.distributed.barrier(self.communicator.device_group)

        return handle

    def all_reduce(
        self, input: torch.Tensor, group: ProcessGroup | None = None
    ) -> None:
        torch.ops._C.shm_allreduce(self.handle, input)

    def gather(
        self,
        input: torch.Tensor,
        gather_list: list[torch.Tensor] | None,
        dst: int = -1,
        group: ProcessGroup | None = None,
    ) -> None:
        # Note: different from the torch gather, here we use local dst rank.
        torch.ops._C.shm_gather(
            self.handle,
            input,
            gather_list,
            torch.distributed.get_group_rank(group, dst),
        )

    def all_gather_into_tensor(
        self,
        output: torch.Tensor,
        input: torch.Tensor,
        group: ProcessGroup | None = None,
    ) -> None:
        torch.ops._C.shm_all_gather(self.handle, input, output)

    def send_tensor_dict(
        self,
        tensor_dict: dict[str, torch.Tensor | Any],
        dst: int,
    ) -> None:
        key_list = list(tensor_dict.keys())
        value_list = list(tensor_dict.values())
        size_list = []
        for v in value_list:
            if not isinstance(v, torch.Tensor):
                raise RuntimeError("CpuCommunicator only supports sending tensors.")
            size_list.append(v.size())
        key_size_tensor = torch.frombuffer(
            pickle.dumps([key_list, size_list]), dtype=torch.uint8
        )
        value_list.append(key_size_tensor)

        torch.ops._C.shm_send_tensor_list(self.handle, value_list, dst)

        return None

    def recv_tensor_dict(
        self,
        src: int,
    ) -> dict[str, torch.Tensor | Any]:
        tensor_list = torch.ops._C.shm_recv_tensor_list(self.handle, src)

        value_list: list[torch.Tensor] = tensor_list[:-1]
        key_size_tensor = tensor_list[-1]

        key_size = pickle.loads(key_size_tensor.numpy().tobytes())
        key_list = key_size[0]
        size_list = key_size[1]
        assert len(key_list) == len(size_list)
        assert len(key_list) == len(value_list)

        tensor_dict: dict[str, torch.Tensor] = {}
        for key, size, t in zip(key_list, size_list, value_list):
            tensor_dict[key] = t.view(size)
        return tensor_dict
