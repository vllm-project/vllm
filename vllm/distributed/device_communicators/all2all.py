# SPDX-License-Identifier: Apache-2.0
import importlib.util
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist

from vllm.forward_context import get_forward_context
from vllm.logger import init_logger

from .base_device_communicator import All2AllManagerBase, Cache

logger = init_logger(__name__)

if TYPE_CHECKING:
    from vllm.model_executor.layers.fused_moe.layer import FusedMoE
else:
    FusedMoE = None


class NaiveAll2AllManager(All2AllManagerBase):
    """
    A naive implementation of all2all communication.
    It uses all-reduce under the hood, which is not
    efficient at all. The main purpose is for testing and
    debugging.
    """

    def __init__(self, cpu_group):
        super().__init__(cpu_group)

    def naive_multicast(self, x: torch.Tensor,
                        cu_tokens_across_dp_cpu: torch.Tensor):
        assert (len(x.shape) == 2)
        buffer = torch.empty((cu_tokens_across_dp_cpu[-1], x.size(1)),
                             device=x.device,
                             dtype=x.dtype)

        start = 0 if self.dp_rank == 0 else cu_tokens_across_dp_cpu[
            self.dp_rank - 1]
        end = cu_tokens_across_dp_cpu[self.dp_rank]
        buffer[start:end, :].copy_(x)
        for idx in range(self.dp_world_size):
            start = 0 if idx == 0 else cu_tokens_across_dp_cpu[idx - 1]
            end = cu_tokens_across_dp_cpu[idx]
            self.dp_group.broadcast(buffer[start:end, :], idx)

        return buffer

    def dispatch(self, hidden_states: torch.Tensor,
                 router_logits: torch.Tensor):
        cu_tokens_across_dp_cpu = get_forward_context(
        ).dp_metadata.cu_tokens_across_dp_cpu

        hidden_states = self.naive_multicast(hidden_states,
                                             cu_tokens_across_dp_cpu)
        router_logits = self.naive_multicast(router_logits,
                                             cu_tokens_across_dp_cpu)
        return hidden_states, router_logits

    def combine(self, hidden_states: torch.Tensor) -> torch.Tensor:
        cu_tokens_across_dp_cpu = get_forward_context(
        ).dp_metadata.cu_tokens_across_dp_cpu
        start = 0 if self.dp_rank == 0 else cu_tokens_across_dp_cpu[
            self.dp_rank - 1]
        end = cu_tokens_across_dp_cpu[self.dp_rank]

        all_hidden_states = self.dp_group.all_reduce(hidden_states)
        hidden_states = all_hidden_states[start:end, :]
        return hidden_states

    def destroy(self):
        pass


class PPLXAll2AllManager(All2AllManagerBase):
    """
    All2All communication based on PPLX kernels.
    """

    def __init__(self, cpu_group):
        has_pplx = importlib.util.find_spec("pplx_kernels") is not None
        assert has_pplx, "pplx_kernels not found. Please follow https://github.com/vllm-project/vllm/blob/main/tools/ep_kernels/README.md to install pplx_kernels."  # noqa
        super().__init__(cpu_group)

        if self.internode:
            # inter-node communication needs nvshmem,
            # intra-node communication uses p2p mapping directly
            from pplx_kernels.nvshmem import (nvshmem_alloc_empty_unique_id,
                                              nvshmem_get_unique_id,
                                              nvshmem_init)
            logger.debug(
                "Initialize NVSHMEM for pplx_kernels: "
                "rank=%d, world size=%d", self.rank, self.world_size)
            uid = nvshmem_get_unique_id(
            ) if self.rank == 0 else nvshmem_alloc_empty_unique_id()
            dist.broadcast(uid,
                           src=dist.get_process_group_ranks(self.cpu_group)[0],
                           group=self.cpu_group)
            logger.debug("PPLX NVSHMEM UID = %s", uid)
            nvshmem_init(uid, self.rank, self.world_size)

        self.handle_cache = Cache()

    def get_handle(self, kwargs):
        import pplx_kernels as pplx
        return self.handle_cache.get_or_create(
            kwargs, pplx.AllToAll.internode
            if self.internode else pplx.AllToAll.intranode)

    def dispatch(self, hidden_states: torch.Tensor,
                 router_logits: torch.Tensor):
        raise NotImplementedError

    def combine(self, hidden_states: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def destroy(self):
        with self.handle_cache._lock:
            for _, handle in self.handle_cache._cache.items():
                handle.destroy()

        if self.internode:
            from pplx_kernels.nvshmem import nvshmem_finalize
            logger.debug("PPLX NVSHMEM finalize")
            nvshmem_finalize()
