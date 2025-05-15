# SPDX-License-Identifier: Apache-2.0
import importlib.util
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist

from vllm.forward_context import get_forward_context
from vllm.logger import init_logger

logger = init_logger(__name__)

if TYPE_CHECKING:
    from vllm.model_executor.layers.fused_moe.layer import FusedMoE
else:
    FusedMoE = None


class All2AllBase:

    def __init__(self, cpu_group, model: torch.nn.Module):
        self.cpu_group = cpu_group

        # compute some common properties
        from vllm.distributed.parallel_state import (get_dp_group,
                                                     get_ep_group,
                                                     get_tp_group,
                                                     in_the_same_node_as)

        # all2all lives in ep group, which is merged from dp and tp group
        self.dp_group = get_dp_group()
        self.tp_group = get_tp_group()
        self.ep_group = get_ep_group()
        self.dp_rank = self.dp_group.rank_in_group
        self.dp_world_size = self.dp_group.world_size
        self.rank = self.ep_group.rank_in_group
        self.world_size = self.ep_group.world_size

        # all2all communication often has separate implementations for
        # intra-node and inter-node communication
        self.intranode = in_the_same_node_as(cpu_group, source_rank=0)
        self.internode = not self.intranode

    def dispatch(self, hidden_states: torch.Tensor,
                 router_logits: torch.Tensor):
        raise NotImplementedError

    def combine(self, hidden_states: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def destroy(self):
        pass


class NaiveAll2All(All2AllBase):
    """
    A naive implementation of all2all communication.
    It uses all-reduce under the hood, which is not
    efficient at all. The main purpose is for testing and
    debugging.
    """

    def __init__(self, cpu_group, model: torch.nn.Module):
        super().__init__(cpu_group, model)

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


class PPLXAll2All(All2AllBase):
    """
    All2All communication based on PPLX kernels.
    """

    def __init__(self, cpu_group, model: torch.nn.Module):
        has_pplx = importlib.util.find_spec("pplx_kernels") is not None
        assert has_pplx, "pplx_kernels not found. Please follow https://github.com/vllm-project/vllm/blob/main/tools/ep_kernels/README.md to install pplx_kernels."  # noqa
        import pplx_kernels as pplx
        super().__init__(cpu_group, model)
        moe_layer: FusedMoE = None
        for module in model.modules():
            if module.__class__.__name__ == "FusedMoE":
                moe_layer = module
                break
        # assume all MoE layers have the same config
        moe = moe_layer.moe_config
        MOE_DP_CHUNK_SIZE = 256
        max_num_tokens = MOE_DP_CHUNK_SIZE

        all_to_all_args = dict(
            max_num_tokens=max_num_tokens,
            num_experts=moe.num_experts,
            experts_per_token=moe.experts_per_token,  # topk
            rank=self.rank,
            world_size=self.world_size,
            dp_size=self.tp_group.
            world_size,  # dp_size actually means tp_size, bug in pplx kernels
            hidden_dim=moe.hidden_dim,
            hidden_dim_bytes=moe.hidden_dim * moe.in_dtype.itemsize,
            # For blocked per token: set to
            #   ceil_div(hidden_dim, block_size) * sizeof(float32)
            # For per-token: set to sizeof(float32)
            hidden_dim_scale_bytes=(0 if moe.in_dtype.itemsize != 1 else
                                    ((moe.hidden_dim + moe.block_size - 1) //
                                     moe.block_size * torch.float32.itemsize)),
            group_name=self.cpu_group.group_name,
        )

        if self.internode:
            from pplx_kernels.nvshmem import (nvshmem_alloc_empty_unique_id,
                                              nvshmem_get_unique_id,
                                              nvshmem_init)
            logger.debug(
                "Initialize NVSHMEM for pplx_kernels: "
                "rank=%d, world size=%d", self.rank, self.world_size)
            uid = nvshmem_get_unique_id(
            ) if self.rank == 0 else nvshmem_alloc_empty_unique_id()
            dist.broadcast(uid,
                           src=self.ep_group.ranks[0],
                           group=self.cpu_group)
            logger.debug("PPLX NVSHMEM UID = %s", uid)
            nvshmem_init(uid, self.rank, self.world_size)
            self.pplx_handle = pplx.AllToAll.internode(**all_to_all_args)
        else:
            self.pplx_handle = pplx.AllToAll.intranode(**all_to_all_args)

        # TODO: refactor the initialization logic
        for module in model.modules():
            if module.__class__.__name__ == "FusedMoE":
                module.quant_method.fused_experts.prepare_finalize.a2a \
                    = self.pplx_handle

    def dispatch(self, hidden_states: torch.Tensor,
                 router_logits: torch.Tensor):
        raise NotImplementedError

    def combine(self, hidden_states: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def destroy(self):
        self.pplx_handle.destroy()
        from pplx_kernels.nvshmem import nvshmem_finalize
        logger.debug("PPLX NVSHMEM finalize")
        nvshmem_finalize()
