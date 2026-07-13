# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from collections import defaultdict
from typing import Any

import torch
from torch.distributed import ProcessGroup

from vllm.distributed.utils import pickle
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.platforms.interface import CpuArchEnum

from .base_device_communicator import DeviceCommunicatorBase

logger = init_logger(__name__)


class CpuCommunicator(DeviceCommunicatorBase):
    def __init__(
        self,
        cpu_group: ProcessGroup,
        device: torch.device | None = None,
        device_group: ProcessGroup | None = None,
        unique_name: str = "",
    ):
        super().__init__(cpu_group, device, device_group, unique_name)
        self.dist_module = torch.distributed

        # Detect multi-node topology
        from vllm.config import get_current_vllm_config_or_none
        config = get_current_vllm_config_or_none()
        nnodes = 1
        if config is not None:
            nnodes = config.parallel_config.nnodes
        total_world_size = torch.distributed.get_world_size()
        local_world_size = total_world_size // max(nnodes, 1)
        self._is_internode = False
        if nnodes > 1 and self.world_size > 1:
            node_ids = set(r // local_world_size for r in self.ranks)
            self._is_internode = len(node_ids) > 1

        can_use_shm = (
            current_platform.get_cpu_architecture()
            in (CpuArchEnum.X86, CpuArchEnum.ARM, CpuArchEnum.POWERPC)
            and hasattr(torch.ops._C, "init_shm_manager")
            and (unique_name.startswith("tp") or unique_name.startswith("pp"))
        )

        if (
            can_use_shm
            and not self._is_internode
            and self._all_group_ranks_share_shm_group_name()
        ):
            self.dist_module = _CPUSHMDistributed(self)
        elif (
            (unique_name.startswith("tp") or unique_name.startswith("pp"))
            and not self._is_internode
        ):
            logger.info(
                "CPU SHM communicator disabled for group %s: ranks do not share "
                "the same SHM group name, falling back to torch.distributed.",
                unique_name,
            )

        # send/recv tensor_dict is only supported through the SHM communicator backend
        self.supports_tensor_dict = isinstance(self.dist_module, _CPUSHMDistributed)

        # Hierarchical all-reduce: SHM (intra-node) + RDMA IBV (cross-node)
        # Enabled automatically when built with VLLM_CPU_RDMA_HAR=ON
        self._use_hierarchical_ar = False
        if (hasattr(torch.ops._C, 'ibv_ar_create')
                and self._is_internode and can_use_shm
                and self.world_size > 1):
            self._setup_hierarchical_ar(local_world_size)

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

    def _setup_hierarchical_ar(self, local_world_size: int) -> None:
        """Set up hierarchical all-reduce for inter-node TP groups.

        Uses SHM for intra-node reduction and IBVerbs RDMA for cross-node.
        Only available when built with VLLM_CPU_RDMA_HAR=ON.
        """
        node_to_ranks: dict[int, list[int]] = defaultdict(list)
        for r in self.ranks:
            node_to_ranks[r // local_world_size].append(r)

        my_node = self.global_rank // local_world_size
        my_node_ranks = sorted(node_to_ranks[my_node])

        # Create intra-node Gloo subgroups (collective: all ranks must call)
        local_gloo_group = None
        for nid in sorted(node_to_ranks.keys()):
            g = torch.distributed.new_group(sorted(node_to_ranks[nid]))
            if nid == my_node:
                local_gloo_group = g

        self._is_leader = (self.global_rank == min(my_node_ranks))
        leader_ranks = sorted(
            min(ranks) for ranks in node_to_ranks.values())

        # Initialize SHM for intra-node all-reduce.
        self._local_shm_handle = None
        if len(my_node_ranks) > 1 and local_gloo_group is not None:
            _sub_comm = CpuCommunicator(
                cpu_group=local_gloo_group,
                device=self.device,
                device_group=local_gloo_group,
                unique_name=f"{self.unique_name}_hier",
            )
            if isinstance(_sub_comm.dist_module, _CPUSHMDistributed):
                self._local_shm_handle = _sub_comm.dist_module.handle
                logger.info(
                    "RDMA HAR: SHM active for local subgroup "
                    "rank=%d local_ranks=%s",
                    self.global_rank, my_node_ranks,
                )
            else:
                logger.warning(
                    "RDMA HAR: SHM NOT active for local subgroup "
                    "rank=%d, performance may be degraded",
                    self.global_rank,
                )

        # Set up IBVerbs RDMA for cross-node
        ibv_handle = self._setup_ibv_all_workers(local_world_size)
        if ibv_handle < 0:
            logger.warning(
                "RDMA HAR: IBV setup failed, rank=%d. "
                "Falling back to standard Gloo all-reduce.",
                self.global_rank,
            )
            return

        # Register with C++ hierarchical AR manager
        shm_h = (self._local_shm_handle
                 if self._local_shm_handle is not None else -1)
        self._hier_ar_handle = torch.ops._C.init_hier_ar(
            shm_h, "", self._is_leader, ibv_handle)
        logger.info(
            "RDMA HAR: rank=%d node=%d "
            "local_ranks=%s is_leader=%s leaders=%s",
            self.global_rank, my_node, my_node_ranks,
            self._is_leader, leader_ranks,
        )

        self._use_hierarchical_ar = True

    def _get_ibv_config(self, local_world_size: int = 2) -> tuple[str, int, int]:
        """Get IBV device name, port, and GID index for this rank.

        Supports per-NUMA NIC selection via VLLM_IBV_DEVICE_{local_rank}
        and VLLM_IBV_GID_INDEX_{local_rank} env vars.
        Falls back to VLLM_IBV_DEVICE, then GLOO_SOCKET_IFNAME (if it
        looks like an IB device), then mlx5_0.
        """
        local_rank = self.global_rank % local_world_size
        # Default IBV device: prefer VLLM_IBV_DEVICE, then check if
        # GLOO_SOCKET_IFNAME is an IB device (mlx5_*), then mlx5_0
        gloo_ifname = os.environ.get("GLOO_SOCKET_IFNAME", "")
        default_dev = os.environ.get("VLLM_IBV_DEVICE", "")
        if not default_dev:
            default_dev = gloo_ifname if gloo_ifname.startswith("mlx5") else "mlx5_0"
        dev_name = os.environ.get(
            f"VLLM_IBV_DEVICE_{local_rank}",
            default_dev,
        )
        gid_index = int(os.environ.get(
            f"VLLM_IBV_GID_INDEX_{local_rank}",
            os.environ.get("TORCH_GLOO_IBV_INDEX", "0"),
        ))
        port = 1
        return dev_name, port, gid_index

    def _setup_ibv_all_workers(self, local_world_size: int) -> int:
        """Set up IBVerbs RDMA for ALL workers (4-way direct exchange).

        Each worker pairs with its cross-node partner (same local_rank
        on the other node) and creates a direct RDMA connection.
        """
        dev_name, port, gid_index = self._get_ibv_config(local_world_size)

        # Compute buffer size from model config.
        # Two tensor categories go through hier_allreduce (IBV):
        #   1. Row-parallel outputs (attention/MLP): (batch_tokens, hidden_size)
        #   2. Logits allreduce (_gather_logits_allreduce): (num_reqs, vocab_full)
        # num_reqs = min(max_num_batched_tokens, max_num_seqs) since logits
        # are computed for one token per request (the last position).
        buf_size_bytes = 0  # 0 = use C++ default (256 MB)
        from vllm.config import get_current_vllm_config_or_none
        cfg = get_current_vllm_config_or_none()
        if cfg is not None:
            hf_cfg = cfg.model_config.hf_config
            hidden = getattr(hf_cfg, 'hidden_size', 0)
            vocab = getattr(hf_cfg, 'vocab_size', 0)
            tp = cfg.parallel_config.tensor_parallel_size
            max_tokens = getattr(cfg.scheduler_config,
                                 'max_num_batched_tokens', 2048)
            max_num_seqs = getattr(cfg.scheduler_config,
                                   'max_num_seqs', 128)
            dtype_bytes = 2  # bf16
            # Vocab is padded to (padding_size * tp) alignment
            vocab_padded = (
                ((vocab + 63) // 64 * 64 + tp - 1) // tp * tp
            )
            # Row-parallel allreduce: max_tokens * hidden_size
            buf_rowparallel = max_tokens * hidden * dtype_bytes
            # Logits allreduce: num_reqs * vocab_padded (full, not per-TP)
            max_logit_tokens = min(max_tokens, max_num_seqs)
            buf_logits = max_logit_tokens * vocab_padded * dtype_bytes
            buf_size_bytes = max(buf_rowparallel, buf_logits)
            if buf_size_bytes > 0:
                logger.info(
                    "IBV buf_size auto: rowparallel=%d MB "
                    "(max_tokens=%d hidden=%d), logits=%d MB "
                    "(max_logit_tokens=%d vocab_padded=%d) -> %d MB",
                    buf_rowparallel // (1024 * 1024),
                    max_tokens, hidden,
                    buf_logits // (1024 * 1024),
                    max_logit_tokens, vocab_padded,
                    buf_size_bytes // (1024 * 1024),
                )

        try:
            ibv_handle = torch.ops._C.ibv_ar_create(
                dev_name, port, gid_index, buf_size_bytes)
            local_info = torch.ops._C.ibv_ar_get_local_info(
                ibv_handle, port, gid_index)

            # Exchange connection info across ALL workers in TP group
            all_info: list[str] = [""] * self.world_size
            torch.distributed.all_gather_object(
                all_info, local_info, group=self.device_group)

            # Partner: same local_rank on other node
            # For 2 nodes: partner = (rank + local_world_size) % world_size
            partner_rank = (
                (self.global_rank + local_world_size) % self.world_size
            )
            # Map from global rank to index in self.ranks
            rank_to_idx = {r: i for i, r in enumerate(self.ranks)}
            partner_idx = rank_to_idx[partner_rank]
            remote_info = all_info[partner_idx]

            torch.ops._C.ibv_ar_connect(
                ibv_handle, remote_info, port, gid_index)

            logger.info(
                "4-way IBV RDMA: rank=%d partner=%d dev=%s "
                "port=%d gid_index=%d",
                self.global_rank, partner_rank, dev_name, port, gid_index,
            )
            return ibv_handle
        except Exception as e:
            logger.warning(
                "IBVerbs setup failed (rank=%d): %s. "
                "Falling back to no cross-node exchange.",
                self.global_rank, e,
            )
            return -1

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
        return len(set(names)) == 1

    def all_reduce(self, input_):
        if self._use_hierarchical_ar:
            torch.ops._C.hier_allreduce(
                self._hier_ar_handle, input_)
            return input_
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

        assert self.all2all_manager is not None
        return self.all2all_manager.dispatch_router_logits(
            hidden_states,
            router_logits,
            is_sequence_parallel,
            extra_tensors,
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
        """
        Dispatch the hidden states and topk weights/ids to the appropriate device.
        This is a no-op in the base class.
        """
        assert self.all2all_manager is not None
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
        """
        Combine the hidden states and router logits from the appropriate device.
        This is a no-op in the base class.
        """
        assert self.all2all_manager is not None
        return self.all2all_manager.combine(
            hidden_states,
            is_sequence_parallel,
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
