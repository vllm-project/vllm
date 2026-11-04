# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from typing import Any, cast

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
    _CPUSHM_GROUP_KINDS = {"tp", "pp"}

    def __init__(
        self,
        cpu_group: ProcessGroup,
        device: torch.device | None = None,
        device_group: ProcessGroup | None = None,
        unique_name: str = "",
        use_all2all: bool = False,
    ):
        super().__init__(
            cpu_group, device, device_group, unique_name, use_all2all=use_all2all
        )
        self.dist_module = torch.distributed
        use_cpushm_group = self._is_cpushm_group_name(unique_name)

        if (
            (
                current_platform.get_cpu_architecture() == CpuArchEnum.X86
                or current_platform.get_cpu_architecture() == CpuArchEnum.ARM
                or current_platform.get_cpu_architecture() == CpuArchEnum.POWERPC
                or current_platform.get_cpu_architecture() == CpuArchEnum.S390X
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
        if unique_name.split(":", maxsplit=1)[0] in {"dp", "ep"}:
            logger.info(
                "CPU group %s uses %s: ranks=%s, rank_in_group=%d, "
                "GLOO_SOCKET_IFNAME=%s.",
                unique_name,
                torch.distributed.get_backend(self.cpu_group),
                self.ranks,
                self.rank_in_group,
                os.environ.get("GLOO_SOCKET_IFNAME", "auto"),
            )

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
        """CPUSHM requires all ranks in this group to agree on one SHM group name.
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
        """NOTE: We assume that the input tensor is on the same device across
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

    def _validate_sizes(self, sizes: list[int]) -> None:
        if len(sizes) != self.world_size or any(
            not isinstance(size, int) or isinstance(size, bool) or size < 0
            for size in sizes
        ):
            raise ValueError(
                "sizes must contain one nonnegative integer per subgroup rank "
                f"(world_size={self.world_size}), got {sizes}."
            )

    @staticmethod
    def _validate_cpu_tensor(tensor: torch.Tensor) -> None:
        if tensor.device.type != "cpu" or tensor.layout != torch.strided:
            raise ValueError("CPU collectives require dense CPU tensors.")
        if tensor.ndim == 0:
            raise ValueError(
                "CPU variable-size collectives require a tensor dimension."
            )

    def _all_gatherv_uniform(
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
        """Copy rank contributions out of the padded gather buffer."""
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

    def _all_gatherv_ragged(
        self,
        tensor: torch.Tensor,
        sizes: list[int],
    ) -> torch.Tensor:
        padded_rows = max(sizes)
        padded_input = tensor.new_zeros((padded_rows,) + tensor.shape[1:])
        padded_input[: tensor.shape[0]].copy_(tensor.contiguous())

        gathered = tensor.new_empty((self.world_size * padded_rows,) + tensor.shape[1:])
        self.dist_module.all_gather_into_tensor(
            gathered,
            padded_input,
            group=self.device_group,
        )

        return self._trim_ragged_rows(gathered, sizes, padded_rows)

    def _gather_single_gatherv(
        self,
        t: torch.Tensor,
        sizes: list[int] | None,
    ) -> torch.Tensor:
        total_rows = sum(sizes) if sizes is not None else t.shape[0] * self.world_size
        if total_rows == 0 or 0 in t.shape[1:]:
            return t.new_empty((total_rows,) + t.shape[1:])
        if sizes is None or all(size == sizes[0] for size in sizes):
            return self._all_gatherv_uniform(t, t.shape[0])
        return self._all_gatherv_ragged(t, sizes)

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
            self._validate_sizes(sizes)
        tensors = [input_] if isinstance(input_, torch.Tensor) else input_
        for tensor in tensors:
            self._validate_cpu_tensor(tensor)
            if sizes is not None and tensor.shape[0] != sizes[self.rank_in_group]:
                raise ValueError(
                    f"Local rows {tensor.shape[0]} do not match "
                    f"sizes[{self.rank_in_group}]={sizes[self.rank_in_group]}."
                )
        gathered = [self._gather_single_gatherv(tensor, sizes) for tensor in tensors]
        return gathered[0] if isinstance(input_, torch.Tensor) else gathered

    def reduce_scatterv(
        self,
        input_: torch.Tensor,
        dim: int = -1,
        sizes: list[int] | None = None,
    ) -> torch.Tensor:
        """Reduce-scatter with variable output sizes via all_reduce + local slice."""
        self._validate_cpu_tensor(input_)
        if not -input_.ndim <= dim < input_.ndim:
            raise ValueError(f"Invalid dim {dim} for tensor with shape {input_.shape}.")
        dim %= input_.ndim

        if sizes is not None:
            self._validate_sizes(sizes)
            if input_.shape[dim] != sum(sizes):
                raise ValueError(
                    f"Scatter dimension {input_.shape[dim]} does not match "
                    f"sum(sizes)={sum(sizes)}."
                )
        else:
            if input_.shape[dim] % self.world_size:
                raise ValueError(
                    "Implicit reduce_scatterv requires the scatter dimension "
                    f"{input_.shape[dim]} to be divisible by world_size "
                    f"{self.world_size}."
                )
            sizes = [input_.shape[dim] // self.world_size] * self.world_size

        reduced = input_.clone(memory_format=torch.contiguous_format)
        if reduced.numel():
            self.dist_module.all_reduce(reduced, group=self.device_group)
        start = sum(sizes[: self.rank_in_group])
        return reduced.narrow(dim, start, sizes[self.rank_in_group]).clone(
            memory_format=torch.contiguous_format
        )

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
        """Keep CPU collectives opaque to torch.compile."""
        from .all2all import AgRsAll2AllManager

        assert self.all2all_manager is not None
        mgr = cast(AgRsAll2AllManager, self.all2all_manager)
        safe_name = (
            self.unique_name.replace("-", "_").replace("/", "_").replace(":", "_")
        )
        # Fake shapes bind local rows to this rank; Inductor caches are shared.
        safe_name += f"_rank{self.rank_in_group}"

        def _dispatch_rows(
            hidden_states: torch.Tensor,
            is_sequence_parallel: bool,
            sizes: list[int],
        ) -> int:
            group = mgr._get_comm_group(is_sequence_parallel)
            if not sizes:
                return hidden_states.shape[0] * group.world_size
            torch._check(len(sizes) == group.world_size)
            for size in sizes:
                torch._check_is_size(size)
            torch._check(sizes[group.rank_in_group] == hidden_states.shape[0])
            return sum(sizes)

        @torch.library.custom_op(
            f"vllm::cpu_ep_dispatch_rl_{safe_name}", mutates_args=()
        )
        def _dispatch_rl(
            hidden_states: torch.Tensor,
            router_logits: torch.Tensor,
            is_sequence_parallel: bool,
            sizes: list[int],
            extra_tensors: list[torch.Tensor],
        ) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
            return cast(
                tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]],
                mgr.dispatch_router_logits(
                    hidden_states,
                    router_logits,
                    is_sequence_parallel,
                    extra_tensors,
                    sizes_override=sizes or None,
                ),
            )

        @_dispatch_rl.register_fake
        def _(
            hidden_states: torch.Tensor,
            router_logits: torch.Tensor,
            is_sequence_parallel: bool,
            sizes: list[int],
            extra_tensors: list[torch.Tensor],
        ) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
            n = _dispatch_rows(hidden_states, is_sequence_parallel, sizes)
            return (
                hidden_states.new_empty((n,) + hidden_states.shape[1:]),
                router_logits.new_empty((n,) + router_logits.shape[1:]),
                [t.new_empty((n,) + t.shape[1:]) for t in extra_tensors],
            )

        @torch.library.custom_op(f"vllm::cpu_ep_dispatch_{safe_name}", mutates_args=())
        def _dispatch(
            hidden_states: torch.Tensor,
            topk_weights: torch.Tensor,
            topk_ids: torch.Tensor,
            is_sequence_parallel: bool,
            sizes: list[int],
            extra_tensors: list[torch.Tensor],
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[torch.Tensor]]:
            return cast(
                tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[torch.Tensor]],
                mgr.dispatch(
                    hidden_states,
                    topk_weights,
                    topk_ids,
                    is_sequence_parallel,
                    extra_tensors,
                    sizes_override=sizes or None,
                ),
            )

        @_dispatch.register_fake
        def _(
            hidden_states: torch.Tensor,
            topk_weights: torch.Tensor,
            topk_ids: torch.Tensor,
            is_sequence_parallel: bool,
            sizes: list[int],
            extra_tensors: list[torch.Tensor],
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[torch.Tensor]]:
            n = _dispatch_rows(hidden_states, is_sequence_parallel, sizes)
            return (
                hidden_states.new_empty((n,) + hidden_states.shape[1:]),
                topk_weights.new_empty((n,) + topk_weights.shape[1:]),
                topk_ids.new_empty((n,) + topk_ids.shape[1:]),
                [t.new_empty((n,) + t.shape[1:]) for t in extra_tensors],
            )

        @torch.library.custom_op(f"vllm::cpu_ep_combine_{safe_name}", mutates_args=())
        def _combine(
            hidden_states: torch.Tensor, is_sequence_parallel: bool, sizes: list[int]
        ) -> torch.Tensor:
            return mgr.combine(
                hidden_states, is_sequence_parallel, sizes_override=sizes or None
            )

        @_combine.register_fake
        def _(
            hidden_states: torch.Tensor, is_sequence_parallel: bool, sizes: list[int]
        ) -> torch.Tensor:
            group = mgr._get_comm_group(is_sequence_parallel)
            if sizes:
                torch._check(len(sizes) == group.world_size)
                for size in sizes:
                    torch._check_is_size(size)
                torch._check(hidden_states.shape[0] == sum(sizes))
                rows = sizes[group.rank_in_group]
            else:
                torch._check(hidden_states.shape[0] % group.world_size == 0)
                rows = hidden_states.shape[0] // group.world_size
            return hidden_states.new_empty((rows,) + hidden_states.shape[1:])

        self._ep_dispatch_rl_op = _dispatch_rl
        self._ep_dispatch_op = _dispatch
        self._ep_combine_op = _combine

    def _get_ep_sizes(self) -> list[int]:
        assert self.all2all_manager is not None
        if self.all2all_manager.dp_world_size == 1:
            return []
        dp_metadata = get_forward_context().dp_metadata
        assert dp_metadata is not None
        sizes = dp_metadata.get_chunk_sizes_across_dp_rank()
        assert sizes is not None
        return sizes

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
        hidden, logits, extras = self._ep_dispatch_rl_op(
            hidden_states,
            router_logits,
            is_sequence_parallel,
            self._get_ep_sizes(),
            [] if extra_tensors is None else extra_tensors,
        )
        if extra_tensors is None:
            return hidden, logits
        return hidden, logits, extras

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
        hidden, weights, ids, extras = self._ep_dispatch_op(
            hidden_states,
            topk_weights,
            topk_ids,
            is_sequence_parallel,
            self._get_ep_sizes(),
            [] if extra_tensors is None else extra_tensors,
        )
        if extra_tensors is None:
            return hidden, weights, ids
        return hidden, weights, ids, extras

    def combine(
        self, hidden_states: torch.Tensor, is_sequence_parallel: bool = False
    ) -> torch.Tensor:
        assert self.all2all_manager is not None
        return self._ep_combine_op(
            hidden_states, is_sequence_parallel, self._get_ep_sizes()
        )


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
