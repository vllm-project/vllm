# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any

import torch
import torch.distributed as dist

from vllm.distributed import get_dp_group
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.utils import has_deep_ep, has_pplx

from .base_device_communicator import All2AllManagerBase, Cache

logger = init_logger(__name__)


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


class AgRsAll2AllManager(All2AllManagerBase):
    """
    An implementation of all2all communication based on
    all-gather (dispatch) and reduce-scatter (combine).
    """

    def __init__(self, cpu_group):
        super().__init__(cpu_group)

    def dispatch(self, hidden_states: torch.Tensor,
                 router_logits: torch.Tensor):
        """
        Gather hidden_states and router_logits from all dp ranks.
        """
        sizes = get_forward_context(
        ).dp_metadata.get_chunk_sizes_across_dp_rank()
        hidden_states, router_logits = get_dp_group().all_gatherv(
            [hidden_states, router_logits],
            dim=0,
            sizes=sizes,
        )
        return hidden_states, router_logits

    def combine(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Reduce-scatter hidden_states across all dp ranks.
        """
        sizes = get_forward_context(
        ).dp_metadata.get_chunk_sizes_across_dp_rank()
        hidden_states = get_dp_group().reduce_scatterv(hidden_states,
                                                       dim=0,
                                                       sizes=sizes)
        return hidden_states

    def destroy(self):
        pass


class PPLXAll2AllManager(All2AllManagerBase):
    """
    All2All communication based on PPLX kernels.
    """

    def __init__(self, cpu_group):
        assert has_pplx(
        ), "pplx_kernels not found. Please follow https://github.com/vllm-project/vllm/blob/main/tools/ep_kernels/README.md to install pplx_kernels."  # noqa
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


class DeepEPAll2AllManagerBase(All2AllManagerBase):
    """
    All2All communication based on DeepEP High-Throughput kernels.
    """

    def __init__(self, cpu_group):
        assert has_deep_ep(
        ), "DeepEP kernels not found. Please follow https://github.com/vllm-project/vllm/blob/main/tools/ep_kernels/README.md to install DeepEP kernels."  # noqa
        super().__init__(cpu_group)
        self.handle_cache = Cache()

        # This is the DeepEP default. Stick to it till we can establish
        # reasonable defaults based on profiling.
        self.num_sms = 20

    def get_handle(self, kwargs):
        raise NotImplementedError

    def dispatch(self, hidden_states: torch.Tensor,
                 router_logits: torch.Tensor):
        raise NotImplementedError

    def combine(self, hidden_states: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def destroy(self):
        pass


class DeepEPHTAll2AllManager(DeepEPAll2AllManagerBase):
    """
    All2All communication based on DeepEP High-Throughput kernels.
    """

    def __init__(self, cpu_group):
        super().__init__(cpu_group)

    def _make_all2all_kwargs(self) -> dict[Any, Any]:
        # Defaults for internode and intranode are taken from DeepEP tests.
        num_nvl_bytes = 1024 * 1024 * 1024
        num_rdma_bytes = None
        num_qps_per_rank = None

        if self.internode:
            num_rdma_bytes = 1024 * 1024 * 1024
            num_qps_per_rank = self.num_sms // 2
        else:
            num_rdma_bytes = 0
            num_qps_per_rank = 1

        assert num_rdma_bytes is not None
        assert num_qps_per_rank is not None
        return dict(group=self.cpu_group,
                    num_nvl_bytes=num_nvl_bytes,
                    num_rdma_bytes=num_rdma_bytes,
                    low_latency_mode=False,
                    num_qps_per_rank=num_qps_per_rank)

    def get_handle(self, kwargs):

        assert len(kwargs) == 0, (
            "DeepEPHTAll2AllManager expects no arguments. All the required "
            "args are computed in the Manager itself.")

        import deep_ep
        buffer_kwargs = self._make_all2all_kwargs()
        logger.debug("DeepEP all2all args %s", buffer_kwargs)
        handle: deep_ep.Buffer = self.handle_cache.get_or_create(
            buffer_kwargs, deep_ep.Buffer)
        # It is dangerous to set num sms outside this function. num_sms is not
        # a part of the hash-key that identifies this object. If we are in a
        # situation where we make objects with different num_sms, the hash key
        # in get_or_create must be updated.
        handle.set_num_sms(self.num_sms)
        return handle


class DeepEPLLAll2AllManager(DeepEPAll2AllManagerBase):
    """
    All2All communication based on DeepEP Low-Latency kernels.
    """

    def __init__(self, cpu_group):
        super().__init__(cpu_group)

    def _make_all2all_kwargs(
        self,
        max_num_tokens_per_dp_rank: int,
        token_hidden_size: int,
        num_ep_ranks: int,
        num_global_experts: int,
        num_local_experts: int,
    ) -> dict[Any, Any]:
        """
        max_num_tokens_per_dp_rank : the maximum number of tokens a DP rank
          can dispatch all the ranks must hold the same value.
        token_hidden_size: the hidden dimension of each token.
        num_ep_ranks: the number of EP group ranks.
        num_global_experts: Number of experts in the model.
        num_local_experts: Number of experts in an EP rank.
        """
        import deep_ep

        # Defaults for internode and intranode are taken from DeepEP tests.
        num_nvl_bytes = 1024 * 1024 * 1024
        num_qps_per_rank = num_local_experts
        num_rdma_bytes = deep_ep.Buffer.get_low_latency_rdma_size_hint(
            num_max_dispatch_tokens_per_rank=max_num_tokens_per_dp_rank,
            hidden=token_hidden_size,
            num_ranks=num_ep_ranks,
            num_experts=num_global_experts)

        assert num_rdma_bytes is not None
        return dict(group=self.cpu_group,
                    num_nvl_bytes=num_nvl_bytes,
                    num_rdma_bytes=num_rdma_bytes,
                    low_latency_mode=True,
                    num_qps_per_rank=num_qps_per_rank)

    def get_handle(self, kwargs):
        """
        The kwargs for DeepEPLLAll2AllManager is dictated by
        _make_all2all_kwargs.
        """
        import deep_ep
        buffer_kwargs = self._make_all2all_kwargs(**kwargs)
        logger.debug("DeepEP all2all args %s", buffer_kwargs)
        handle: deep_ep.Buffer = self.handle_cache.get_or_create(
            buffer_kwargs, deep_ep.Buffer)
        return handle
