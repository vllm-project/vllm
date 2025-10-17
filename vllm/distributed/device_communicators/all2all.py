# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any

import torch
import torch.distributed as dist

import vllm.envs as envs
from vllm.distributed import get_dp_group, get_ep_group
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.utils import has_deep_ep, has_pplx
from vllm.utils.flashinfer import has_flashinfer_all2all

from .base_device_communicator import All2AllManagerBase, Cache

if has_flashinfer_all2all():
    from flashinfer.comm import Mapping  # type: ignore[import-not-found]
    from flashinfer.comm.mnnvl import MnnvlConfig  # type: ignore[import-not-found]
    from flashinfer.comm.trtllm_alltoall import (
        MnnvlMoe,  # type: ignore[import-not-found]
    )

logger = init_logger(__name__)


class NaiveAll2AllManager(All2AllManagerBase):
    """
    A naive implementation of all2all communication.
    It uses all-reduce under the hood, which is not
    efficient at all. The main purpose is for testing and
    debugging.
    """

    def __init__(self, cpu_group, tcp_store_group=None):
        super().__init__(cpu_group, tcp_store_group)

    def naive_multicast(
        self,
        x: torch.Tensor,
        cu_tokens_across_sp_cpu: torch.Tensor,
        is_sequence_parallel: bool,
    ) -> torch.Tensor:
        assert len(x.shape) == 2
        buffer = torch.empty(
            (cu_tokens_across_sp_cpu[-1], x.size(1)), device=x.device, dtype=x.dtype
        )

        rank = self.rank if is_sequence_parallel else self.dp_rank
        world_size = self.world_size if is_sequence_parallel else self.dp_world_size

        start = 0 if rank == 0 else cu_tokens_across_sp_cpu[rank - 1]
        end = cu_tokens_across_sp_cpu[rank]
        buffer[start:end, :].copy_(x)
        for idx in range(world_size):
            start = 0 if idx == 0 else cu_tokens_across_sp_cpu[idx - 1]
            end = cu_tokens_across_sp_cpu[idx]
            get_ep_group().broadcast(buffer[start:end, :], idx)

        return buffer

    def dispatch(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        is_sequence_parallel: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sp_size = self.tp_group.world_size if is_sequence_parallel else 1
        dp_metadata = get_forward_context().dp_metadata
        assert dp_metadata is not None
        cu_tokens_across_sp_cpu = dp_metadata.cu_tokens_across_sp(sp_size)

        hidden_states = self.naive_multicast(
            hidden_states, cu_tokens_across_sp_cpu, is_sequence_parallel
        )
        router_logits = self.naive_multicast(
            router_logits, cu_tokens_across_sp_cpu, is_sequence_parallel
        )
        return hidden_states, router_logits

    def combine(
        self, hidden_states: torch.Tensor, is_sequence_parallel: bool = False
    ) -> torch.Tensor:
        ep_rank = self.rank if is_sequence_parallel else self.dp_rank

        dp_metadata = get_forward_context().dp_metadata
        assert dp_metadata is not None
        sp_size = self.tp_group.world_size if is_sequence_parallel else 1
        cu_tokens_across_sp_cpu = dp_metadata.cu_tokens_across_sp(sp_size)

        start = 0 if ep_rank == 0 else cu_tokens_across_sp_cpu[ep_rank - 1]
        end = cu_tokens_across_sp_cpu[ep_rank]

        all_hidden_states = get_ep_group().all_reduce(hidden_states)
        hidden_states = all_hidden_states[start:end, :]
        return hidden_states

    def destroy(self):
        pass


class AgRsAll2AllManager(All2AllManagerBase):
    """
    An implementation of all2all communication based on
    all-gather (dispatch) and reduce-scatter (combine).
    """

    def __init__(self, cpu_group, tcp_store_group=None):
        super().__init__(cpu_group, tcp_store_group)

    def dispatch(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        is_sequence_parallel: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Gather hidden_states and router_logits from all dp ranks.
        """
        dp_metadata = get_forward_context().dp_metadata
        assert dp_metadata is not None
        sizes = dp_metadata.get_chunk_sizes_across_dp_rank()
        assert sizes is not None

        dist_group = get_ep_group() if is_sequence_parallel else get_dp_group()
        assert sizes[dist_group.rank_in_group] == hidden_states.shape[0]
        hidden_states, router_logits = dist_group.all_gatherv(
            [hidden_states, router_logits],
            dim=0,
            sizes=sizes,
        )
        return hidden_states, router_logits

    def combine(
        self, hidden_states: torch.Tensor, is_sequence_parallel: bool = False
    ) -> torch.Tensor:
        """
        Reduce-scatter hidden_states across all dp ranks.
        """
        dp_metadata = get_forward_context().dp_metadata
        assert dp_metadata is not None
        sizes = dp_metadata.get_chunk_sizes_across_dp_rank()
        assert sizes is not None

        dist_group = get_ep_group() if is_sequence_parallel else get_dp_group()
        hidden_states = dist_group.reduce_scatterv(hidden_states, dim=0, sizes=sizes)
        return hidden_states

    def destroy(self):
        pass


class PPLXAll2AllManager(All2AllManagerBase):
    """
    All2All communication based on PPLX kernels.
    """

    def __init__(self, cpu_group, tcp_store_group=None):
        assert has_pplx(
        ), "pplx_kernels not found. Please follow https://github.com/vllm-project/vllm/blob/main/tools/ep_kernels/README.md to install pplx_kernels."  # noqa
        super().__init__(cpu_group, tcp_store_group)

        self.nvshmem_initialized = False
        self.handle_cache = Cache()

    def get_handle(self, kwargs):
        if self.internode and not self.nvshmem_initialized:
            from pplx_kernels.nvshmem import (nvshmem_alloc_empty_unique_id,
                                              nvshmem_get_unique_id,
                                              nvshmem_init)
            logger.debug(
                "Initialize NVSHMEM for pplx_kernels: "
                "rank=%d, world size=%d", self.rank, self.world_size)
            uid = nvshmem_get_unique_id(
            ) if self.rank == 0 else nvshmem_alloc_empty_unique_id()

            if self.tcp_store_group is not None:
                uid = self.tcp_store_group.broadcast_obj(uid, src=0)
            else:
                dist.broadcast(uid,
                               src=dist.get_process_group_ranks(self.cpu_group)[0],
                               group=self.cpu_group)

            logger.debug("PPLX NVSHMEM UID = %s", uid)
            nvshmem_init(uid, self.rank, self.world_size)
            self.nvshmem_initialized = True

        import pplx_kernels as pplx
        return self.handle_cache.get_or_create(
            kwargs,
            pplx.AllToAll.internode if self.internode else pplx.AllToAll.intranode,
        )

    def dispatch(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        is_sequence_parallel: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def combine(
        self, hidden_states: torch.Tensor, is_sequence_parallel: bool = False
    ) -> torch.Tensor:
        raise NotImplementedError

    def destroy(self):
        with self.handle_cache._lock:
            for _, handle in self.handle_cache._cache.items():
                handle.destroy()

        if self.internode:
            from pplx_kernels.nvshmem import (
                nvshmem_finalize,  # type: ignore[import-not-found]
            )

            logger.debug("PPLX NVSHMEM finalize")
            nvshmem_finalize()


class DeepEPAll2AllManagerBase(All2AllManagerBase):
    """
    All2All communication based on DeepEP High-Throughput kernels.
    """

    def __init__(self, cpu_group, tcp_store_group=None):
        assert has_deep_ep(
        ), "DeepEP kernels not found. Please follow https://github.com/vllm-project/vllm/blob/main/tools/ep_kernels/README.md to install DeepEP kernels."  # noqa
        super().__init__(cpu_group, tcp_store_group)
        self.handle_cache = Cache()

        # This is the DeepEP default. Stick to it till we can establish
        # reasonable defaults based on profiling.
        self.num_sms = 20

    def get_handle(self, kwargs):
        raise NotImplementedError

    def dispatch(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        is_sequence_parallel: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def combine(
        self, hidden_states: torch.Tensor, is_sequence_parallel: bool = False
    ) -> torch.Tensor:
        raise NotImplementedError

    def destroy(self):
        pass


class DeepEPHTAll2AllManager(DeepEPAll2AllManagerBase):
    """
    All2All communication based on DeepEP High-Throughput kernels.
    """

    def __init__(self, cpu_group, tcp_store_group=None):
        super().__init__(cpu_group, tcp_store_group)

    def _make_all2all_kwargs(self) -> dict[Any, Any]:
        # Defaults for internode and intranode are taken from DeepEP tests.
        num_nvl_bytes = envs.VLLM_DEEPEP_BUFFER_SIZE_MB * 1024 * 1024
        num_rdma_bytes = None
        num_qps_per_rank = None

        if self.internode:
            num_rdma_bytes = envs.VLLM_DEEPEP_BUFFER_SIZE_MB * 1024 * 1024
            num_qps_per_rank = self.num_sms // 2
        else:
            num_rdma_bytes = 0
            num_qps_per_rank = 1

        assert num_rdma_bytes is not None
        assert num_qps_per_rank is not None
        return dict(
            group=self.cpu_group,
            num_nvl_bytes=num_nvl_bytes,
            num_rdma_bytes=num_rdma_bytes,
            low_latency_mode=False,
            num_qps_per_rank=num_qps_per_rank,
        )

    def get_handle(self, kwargs):
        assert len(kwargs) == 0, (
            "DeepEPHTAll2AllManager expects no arguments. All the required "
            "args are computed in the Manager itself."
        )

        import deep_ep  # type: ignore[import-not-found]

        buffer_kwargs = self._make_all2all_kwargs()
        logger.debug("DeepEP all2all args %s", buffer_kwargs)
        handle: deep_ep.Buffer = self.handle_cache.get_or_create(
            buffer_kwargs, deep_ep.Buffer
        )
        return handle

    def set_num_sms(self, num_sms: int):
        import deep_ep  # type: ignore[import-not-found]

        # Right now the buffers are sized for only what the kernels were
        # created with. So we can only reduce the number of SMS used
        # but not increase it.
        if num_sms > self.num_sms:
            num_sms = self.num_sms
        deep_ep.Buffer.set_num_sms(num_sms)


class DeepEPLLAll2AllManager(DeepEPAll2AllManagerBase):
    """
    All2All communication based on DeepEP Low-Latency kernels.
    """

    def __init__(self, cpu_group, tcp_store_group=None):
        super().__init__(cpu_group, tcp_store_group)

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
        import deep_ep  # type: ignore[import-not-found]

        # Defaults for internode and intranode are taken from DeepEP tests.
        # num_nvl_bytes = 1024 * 1024 * 1024
        num_nvl_bytes = 0
        num_qps_per_rank = num_local_experts
        num_rdma_bytes = deep_ep.Buffer.get_low_latency_rdma_size_hint(
            num_max_dispatch_tokens_per_rank=max_num_tokens_per_dp_rank,
            hidden=token_hidden_size,
            num_ranks=num_ep_ranks,
            num_experts=num_global_experts,
        )

        assert num_rdma_bytes is not None
        return dict(group=self.cpu_group,
                    num_nvl_bytes=num_nvl_bytes,
                    num_rdma_bytes=num_rdma_bytes,
                    low_latency_mode=True,
                    num_qps_per_rank=num_qps_per_rank,
                    allow_mnnvl=True)

    def get_handle(self, kwargs):
        """
        The kwargs for DeepEPLLAll2AllManager is dictated by
        _make_all2all_kwargs.
        """
        import deep_ep  # type: ignore[import-not-found]

        buffer_kwargs = self._make_all2all_kwargs(**kwargs)
        logger.debug("DeepEP all2all args %s", buffer_kwargs)
        handle: deep_ep.Buffer = self.handle_cache.get_or_create(
            buffer_kwargs, deep_ep.Buffer
        )
        return handle


class NIXLDeepEPLLAll2AllManager(All2AllManagerBase):
    """
    All2All communication based on NIXL DeepEP Low-Latency kernels.
    """
    _persistent_buffer = None
    _buffer_kwargs = None
    _current_ep_size = -1
    _max_num_ep_ranks = -1
    # NOTE(yongji): set in prepare_communication_buffer_for_model
    _ep_group_changed = False

    def __init__(self, cpu_group, tcp_store_group=None):
        super().__init__(cpu_group, tcp_store_group)
        import os
        max_num_ep_ranks = int(os.environ.get('NIXL_DEEPEP_MAX_NUM_RANKS', -1))
        if max_num_ep_ranks == -1:
            raise RuntimeError("NIXL_DEEPEP_MAX_NUM_RANKS is not set")
        if NIXLDeepEPLLAll2AllManager._max_num_ep_ranks == -1:
            NIXLDeepEPLLAll2AllManager._max_num_ep_ranks = max_num_ep_ranks
        else:
            assert NIXLDeepEPLLAll2AllManager._max_num_ep_ranks == max_num_ep_ranks
        assert 'NIXL_ETCD_ENDPOINTS' in os.environ, "NIXL_ETCD_ENDPOINTS is not set"
        assert 'NIXL_UCX_IB_DEVICES' in os.environ, "NIXL_UCX_IB_DEVICES is not set"
        assert 'NIXL_UCX_TCP_DEVICES' in os.environ, "NIXL_UCX_TCP_DEVICES is not set"
        ucx_ib_nics = os.environ['NIXL_UCX_IB_DEVICES'].split(',')

        from vllm.distributed.parallel_state import get_tp_group, get_pp_group
        from vllm import envs
        # NOTE(yongji): # envs.LOCAL_RANK may not be set
        # an ugly way to get current worker's device index under DPEngineCoreActor
        cuda_visible_devices = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
        assert get_pp_group().world_size == 1
        local_device_index = int(cuda_visible_devices[get_tp_group().rank_in_group])
        pxb_ib_nic = ucx_ib_nics[local_device_index]
        os.environ['UCX_NET_DEVICES'] = f'cuda0-{pxb_ib_nic}' + ',' + os.environ['NIXL_UCX_TCP_DEVICES']

    def _init_buffer(
        self,
        max_num_tokens_per_dp_rank: int,
        token_hidden_size: int,
        num_experts_per_rank: int,
    ):
        import deep_ep

        max_num_ep_ranks = NIXLDeepEPLLAll2AllManager._max_num_ep_ranks
        max_num_global_experts = max_num_ep_ranks * num_experts_per_rank
        num_nvl_bytes = 0
        num_rdma_bytes = deep_ep.Buffer.get_low_latency_rdma_size_hint(
            num_max_dispatch_tokens_per_rank=max_num_tokens_per_dp_rank,
            hidden=token_hidden_size,
            num_ranks=max_num_ep_ranks,
            num_experts=max_num_global_experts)

        assert NIXLDeepEPLLAll2AllManager._persistent_buffer is None, "NIXL EP buffer already initialized"
        buffer = deep_ep.Buffer.nixl_buffer(
            rank=self.rank,
            low_latency_mode=True,
            explicitly_destroy=True,
            allow_nvlink_for_low_latency_mode=True,
            allow_mnnvl=True,
            explicitly_destroy=True
        )
        buffer.update_memory_buffers(
            num_ranks=max_num_ep_ranks,
            num_experts_per_rank=num_experts_per_rank,
            num_nvl_bytes=num_nvl_bytes,
            num_rdma_bytes=num_rdma_bytes,
        )
        NIXLDeepEPLLAll2AllManager._persistent_buffer = buffer
        ranks_to_connect = list(range(self.cpu_group.size()))
        buffer.connect_ranks(ranks_to_connect)
        NIXLDeepEPLLAll2AllManager._current_ep_size = self.cpu_group.size()
        NIXLDeepEPLLAll2AllManager._ep_group_changed = False

    def _update_buffer(self):
        buffer = NIXLDeepEPLLAll2AllManager._persistent_buffer
        assert buffer is not None
        current_ranks = list(range(NIXLDeepEPLLAll2AllManager._current_ep_size))
        new_ep_size = self.cpu_group.size()
        if new_ep_size > len(current_ranks):
            ranks_to_connect = list(range(len(current_ranks), new_ep_size))
            buffer.connect_ranks(ranks_to_connect)
        else:
            ranks_to_remove = current_ranks[new_ep_size:]
            buffer.remove_ranks(ranks_to_remove)
        NIXLDeepEPLLAll2AllManager._current_ep_size = new_ep_size
        NIXLDeepEPLLAll2AllManager._ep_group_changed = False

    def get_handle(self, kwargs):
        if not NIXLDeepEPLLAll2AllManager._ep_group_changed:
            assert NIXLDeepEPLLAll2AllManager._persistent_buffer is not None
            return NIXLDeepEPLLAll2AllManager._persistent_buffer
        
        # NOTE(yongji): kwargs passed by FusedMoEMethodBase is the same as DeepEPLL, which contains:
        #   max_num_tokens_per_dp_rank, token_hidden_size, 
        #   num_ep_ranks, num_global_experts, num_local_experts
        num_experts_per_rank = kwargs['num_global_experts'] // kwargs['num_local_experts']
        nixl_kwargs = dict(
            max_num_tokens_per_dp_rank=kwargs['max_num_tokens_per_dp_rank'],
            token_hidden_size=kwargs['token_hidden_size'],
            num_experts_per_rank=num_experts_per_rank,
        )
        # kwargs = nixl_kwargs
        
        buffer_kwargs = sorted((k, v) for k, v in nixl_kwargs.items())
        if NIXLDeepEPLLAll2AllManager._persistent_buffer is None:
            self._init_buffer(**nixl_kwargs)
            NIXLDeepEPLLAll2AllManager._buffer_kwargs = buffer_kwargs
        else:
            assert NIXLDeepEPLLAll2AllManager._buffer_kwargs == buffer_kwargs, "NIXL EP buffer kwargs changed"
            self._update_buffer()
        handle = NIXLDeepEPLLAll2AllManager._persistent_buffer
        return handle

    def dispatch(self, hidden_states: torch.Tensor,
                 router_logits: torch.Tensor):
        raise NotImplementedError

    def combine(self, hidden_states: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def destroy(self):
        # NOTE(yongji): NIXLDeepEPLLAll2AllManager instance is recreated during scale-up/down,
        # so we cannot destroy the persistent buffer here.
        pass
