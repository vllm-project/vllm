# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any

import torch
import torch.distributed as dist

import vllm.envs as envs
from vllm.distributed import get_dp_group, get_ep_group
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.utils.flashinfer import has_flashinfer_all2all
from vllm.utils.import_utils import has_deep_ep, has_mori, has_pplx

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

    def __init__(self, cpu_group):
        super().__init__(cpu_group)

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

    def dispatch_router_logits(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        is_sequence_parallel: bool = False,
        extra_tensors: list[torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if extra_tensors is not None:
            raise NotImplementedError(
                "extra_tensors is not supported for NaiveAll2AllManager"
            )
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

    def dispatch(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        is_sequence_parallel: bool = False,
        extra_tensors: list[torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if extra_tensors is not None:
            raise NotImplementedError(
                "extra_tensors is not supported for NaiveAll2AllManager"
            )
        sp_size = self.tp_group.world_size if is_sequence_parallel else 1
        dp_metadata = get_forward_context().dp_metadata
        assert dp_metadata is not None
        cu_tokens_across_sp_cpu = dp_metadata.cu_tokens_across_sp(sp_size)

        hidden_states = self.naive_multicast(
            hidden_states, cu_tokens_across_sp_cpu, is_sequence_parallel
        )
        topk_weights = self.naive_multicast(
            topk_weights, cu_tokens_across_sp_cpu, is_sequence_parallel
        )
        topk_ids = self.naive_multicast(
            topk_ids, cu_tokens_across_sp_cpu, is_sequence_parallel
        )
        return hidden_states, topk_weights, topk_ids

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

    def __init__(self, cpu_group):
        super().__init__(cpu_group)

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
        Gather hidden_states and router_logits from all dp ranks.
        """
        dp_metadata = get_forward_context().dp_metadata
        assert dp_metadata is not None
        sizes = dp_metadata.get_chunk_sizes_across_dp_rank()
        assert sizes is not None
        dist_group = get_ep_group() if is_sequence_parallel else get_dp_group()
        assert sizes[dist_group.rank_in_group] == hidden_states.shape[0]

        tensors_to_gather = [hidden_states, router_logits]
        if extra_tensors is not None:
            tensors_to_gather.extend(extra_tensors)

        gathered_tensors = dist_group.all_gatherv(
            tensors_to_gather,
            dim=0,
            sizes=sizes,
        )

        if extra_tensors is not None:
            return (gathered_tensors[0], gathered_tensors[1], gathered_tensors[2:])
        return gathered_tensors[0], gathered_tensors[1]

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
        Gather hidden_states and router_logits from all dp ranks.
        """
        dp_metadata = get_forward_context().dp_metadata
        assert dp_metadata is not None
        sizes = dp_metadata.get_chunk_sizes_across_dp_rank()
        assert sizes is not None
        dist_group = get_ep_group() if is_sequence_parallel else get_dp_group()
        assert sizes[dist_group.rank_in_group] == hidden_states.shape[0]

        tensors_to_gather = [hidden_states, topk_weights, topk_ids]
        if extra_tensors is not None:
            tensors_to_gather.extend(extra_tensors)

        gathered_tensors = dist_group.all_gatherv(
            tensors_to_gather,
            dim=0,
            sizes=sizes,
        )

        hidden_states = gathered_tensors[0]
        topk_weights = gathered_tensors[1]
        topk_ids = gathered_tensors[2]

        if extra_tensors is None:
            return hidden_states, topk_weights, topk_ids

        return hidden_states, topk_weights, topk_ids, gathered_tensors[3:]

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

    def __init__(self, cpu_group):
        assert has_pplx(), (
            "pplx_kernels not found. Please follow https://github.com/vllm-project/vllm/blob/main/tools/ep_kernels/README.md"
            " to install pplx_kernels."
        )
        super().__init__(cpu_group)

        if self.internode:
            # inter-node communication needs nvshmem,
            # intra-node communication uses p2p mapping directly
            from pplx_kernels.nvshmem import (  # type: ignore[import-not-found]
                nvshmem_alloc_empty_unique_id,
                nvshmem_get_unique_id,
                nvshmem_init,
            )

            logger.debug(
                "Initialize NVSHMEM for pplx_kernels: rank=%d, world size=%d",
                self.rank,
                self.world_size,
            )
            uid = (
                nvshmem_get_unique_id()
                if self.rank == 0
                else nvshmem_alloc_empty_unique_id()
            )
            dist.broadcast(
                uid,
                src=dist.get_process_group_ranks(self.cpu_group)[0],
                group=self.cpu_group,
            )
            logger.debug("PPLX NVSHMEM UID = %s", uid)
            nvshmem_init(uid, self.rank, self.world_size)

        self.handle_cache = Cache()

    def get_handle(self, kwargs):
        import pplx_kernels as pplx  # type: ignore[import-not-found]

        return self.handle_cache.get_or_create(
            kwargs,
            pplx.AllToAll.internode if self.internode else pplx.AllToAll.intranode,
        )

    def dispatch_router_logits(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        is_sequence_parallel: bool = False,
        extra_tensors: list[torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
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

    def __init__(self, cpu_group):
        assert has_deep_ep(), (
            "DeepEP kernels not found. Please follow https://github.com/vllm-project/vllm/blob/main/tools/ep_kernels/README.md"
            " to install DeepEP kernels."
        )  # noqa
        super().__init__(cpu_group)
        self.handle_cache = Cache()

        # This is the DeepEP default. Stick to it till we can establish
        # reasonable defaults based on profiling.
        self.num_sms = 20

    def get_handle(self, kwargs):
        raise NotImplementedError

    def dispatch_router_logits(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        is_sequence_parallel: bool = False,
        extra_tensors: list[torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
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

    def __init__(self, cpu_group):
        super().__init__(cpu_group)

    def _make_all2all_kwargs(self) -> dict[Any, Any]:
        # Defaults for internode and intranode are taken from DeepEP tests.
        num_nvl_bytes = envs.VLLM_DEEPEP_BUFFER_SIZE_MB * 1024 * 1024
        num_rdma_bytes = None
        num_qps_per_rank = None

        if self.internode and not envs.VLLM_DEEPEP_HIGH_THROUGHPUT_FORCE_INTRA_NODE:
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
        import deep_ep  # type: ignore[import-not-found]

        # Defaults for internode and intranode are taken from DeepEP tests.
        num_nvl_bytes = envs.VLLM_DEEPEP_BUFFER_SIZE_MB * 1024 * 1024
        num_qps_per_rank = num_local_experts
        num_rdma_bytes = deep_ep.Buffer.get_low_latency_rdma_size_hint(
            num_max_dispatch_tokens_per_rank=max_num_tokens_per_dp_rank,
            hidden=token_hidden_size,
            num_ranks=num_ep_ranks,
            num_experts=num_global_experts,
        )

        assert num_rdma_bytes is not None
        return dict(
            group=self.cpu_group,
            num_nvl_bytes=num_nvl_bytes,
            num_rdma_bytes=num_rdma_bytes,
            low_latency_mode=True,
            num_qps_per_rank=num_qps_per_rank,
            allow_nvlink_for_low_latency_mode=True,
            allow_mnnvl=envs.VLLM_DEEPEP_LOW_LATENCY_USE_MNNVL,
        )

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

    # DeepEP LL uses RDMA so no SMs are used for communication
    def max_sms_used(self) -> int | None:
        return 0


class FlashInferAllToAllManager(All2AllManagerBase):
    """
    All2All communication based on flashinfer kernels.
    """

    # This type lint could be removed after all of the work in
    # https://github.com/vllm-project/vllm/issues/26533 done.
    rank: int
    world_size: int

    def __init__(self, cpu_group):
        assert has_flashinfer_all2all(), (
            "flashinfer all2all module not found. Please install/check flashinfer"
        )  # noqa
        super().__init__(cpu_group)
        logger.debug(
            "Initialize for flashinfer All2All rank=%d, world size=%d",
            self.rank,
            self.world_size,
        )
        self.initialized = False
        self.alltoall_info = None

    def initialize(
        self,
        world_size: int,
        rank: int,
        gpus_per_node: int,
    ):
        """Initialize workspace"""
        if self.initialized:
            return

        self.cleanup()
        logger.debug("making map: rank=%d, world size=%d", rank, world_size)
        self.mapping = Mapping(
            world_size,
            rank,
            gpus_per_node,
            tp_size=world_size,
        )

        from vllm.distributed.device_communicators.mnnvl_compat import (
            CustomCommunicator,
        )

        dp_config = MnnvlConfig(
            comm_backend=CustomCommunicator(get_dp_group().cpu_group),
            fabric_page_size=1 << 29,  # 512MB
            allocation_granularity=0,  # Auto-detect
        )

        self.workspace_tensor = MnnvlMoe.get_moe_workspaces(self.mapping, dp_config)
        self.prepare_workspace_tensor = MnnvlMoe.get_moe_prepare_workspace(
            self.mapping, dp_config
        )

        self.world_size = world_size
        self.rank = rank
        self.gpus_per_node = gpus_per_node
        self.initialized = True

        logger.info(
            "FlashInfer All2All initialized for rank %s, size %s", rank, world_size
        )

    def ensure_alltoall_workspace_initialized(self):
        """Ensure workspace is initialized"""
        if not has_flashinfer_all2all():
            return False

        if self.world_size <= 1:
            return False

        if not self.initialized:
            self.initialize(
                world_size=self.world_size,
                rank=self.rank,
                gpus_per_node=torch.cuda.device_count,
            )
        return self.initialized

    def get_handle(self, kwargs):
        return self

    def cleanup(self):
        """Clean up workspace"""
        if (
            self.initialized
            and self.workspace_tensor is not None
            and self.prepare_workspace_tensor is not None
        ):
            try:
                del self.workspace_tensor
                del self.prepare_workspace_tensor
            except Exception as e:
                logger.warning("Failed to cleanup FlashInfer workspace: %s", e)
            finally:
                self.workspace_tensor = None
                self.prepare_workspace_tensor = None
                self.mapping = None
                self.initialized = False


class MoriAll2AllManager(All2AllManagerBase):
    def __init__(self, cpu_group):
        assert has_mori(), (
            "MoRI kernels not found. Please follow https://github.com/ROCm/mori/blob/main/README.md"
            " to install MoRI kernels."
        )  # noqa
        import mori

        super().__init__(cpu_group)
        self.handle_cache = Cache()

        torch._C._distributed_c10d._register_process_group("mori", cpu_group)
        mori.shmem.shmem_torch_process_group_init("mori")

    def _make_all2all_kwargs(
        self,
        rank: int,
        num_ep_ranks: int,
        input_dtype: torch.dtype,
        quant_dtype: torch.dtype,
        token_hidden_size: int,
        scale_dim: int,
        scale_type_size: int,
        max_num_tokens_per_dp_rank: int,
        num_local_experts: int,
        num_experts_per_token: int,
    ):
        import mori  # type: ignore[import-not-found]

        from vllm.platforms.rocm import on_gfx942, on_gfx950

        assert on_gfx942() or on_gfx950(), (
            "mori currently only support arch gfx942 and gfx950"
        )

        if not self.internode:
            # single node
            kernel_type = mori.ops.EpDispatchCombineKernelType.IntraNode
            rdma_block_num = 0
            warp_num_per_block = 16
            block_num = 80
        else:
            # multi node
            kernel_type = mori.ops.EpDispatchCombineKernelType.InterNodeV1
            if on_gfx942():
                warp_num_per_block = 16
                block_num = 32
                rdma_block_num = 16
            elif on_gfx950():
                warp_num_per_block = 8
                block_num = 64
                rdma_block_num = 32
            else:
                raise NotImplementedError(
                    "mori currently only support arch gfx942 and gfx950"
                )

        return dict(
            rank=rank,
            world_size=num_ep_ranks,
            data_type=quant_dtype,
            hidden_dim=token_hidden_size,
            scale_dim=scale_dim,
            scale_type_size=scale_type_size,
            max_token_type_size=input_dtype.itemsize,
            max_num_inp_token_per_rank=max_num_tokens_per_dp_rank,
            num_experts_per_rank=num_local_experts,
            num_experts_per_token=num_experts_per_token,
            warp_num_per_block=warp_num_per_block,
            block_num=block_num,
            kernel_type=kernel_type,
            rdma_block_num=rdma_block_num,
            gpu_per_node=min(8, num_ep_ranks),
        )

    def _make_handle(self, **kwargs):
        import mori  # type: ignore[import-not-found]

        mori_config = mori.ops.EpDispatchCombineConfig(**kwargs)
        handle = mori.ops.EpDispatchCombineOp(mori_config)
        return handle

    def get_handle(self, kwargs):
        import mori  # type: ignore[import-not-found]

        mori_kwargs = self._make_all2all_kwargs(**kwargs)
        logger.debug("MoRI all2all args %s", mori_kwargs)
        handle: mori.ops.EpDispatchCombineOp = self.handle_cache.get_or_create(
            mori_kwargs, self._make_handle
        )
        return handle
