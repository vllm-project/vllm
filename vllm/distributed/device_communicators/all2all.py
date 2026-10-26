# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

import torch
import torch.distributed as dist

import vllm.envs as envs
from vllm.config import get_current_vllm_config
from vllm.distributed import get_dp_group, get_ep_group, get_pcp_group
from vllm.distributed.utils import StatelessProcessGroup
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.utils.flashinfer import (
    has_flashinfer_nvlink_one_sided,
    has_flashinfer_nvlink_two_sided,
)
from vllm.utils.func_utils import supports_kw
from vllm.utils.import_utils import (
    check_moonep_system_support,
    deep_ep_v2_unavailable_reason,
    has_deep_ep,
    has_deep_ep_v2,
    has_mori,
)

from .base_device_communicator import All2AllManagerBase, Cache

if has_flashinfer_nvlink_two_sided():
    from flashinfer.comm import Mapping  # type: ignore[import-not-found]
    from flashinfer.comm.mnnvl import MnnvlConfig  # type: ignore[import-not-found]
    from flashinfer.comm.trtllm_alltoall import (
        MnnvlMoe,  # type: ignore[import-not-found]
    )

if has_flashinfer_nvlink_one_sided():
    from flashinfer.comm import Mapping  # type: ignore[import-not-found]
    from flashinfer.comm.mnnvl import MnnvlConfig  # type: ignore[import-not-found]
    from flashinfer.comm.trtllm_moe_alltoall import (
        MoeAlltoAll,  # type: ignore[import-not-found]
        moe_a2a_get_workspace_size_per_rank,
    )


logger = init_logger(__name__)


class PassThroughAll2AllManager(All2AllManagerBase):
    """Placeholder for ``all2all_backend="passthrough"``.

    The MoE backend dispatches and combines itself, so there is no all2all to
    manage.
    """

    def get_handle(self, kwargs):
        raise RuntimeError(
            "passthrough has no all2all handle: dispatch and combine run "
            "inside the MoE backend."
        )


class AgRsAll2AllManager(All2AllManagerBase):
    """An implementation of all2all communication based on
    all-gather (dispatch) and reduce-scatter (combine).
    """

    def __init__(self, cpu_group, tcp_store_group=None):
        super().__init__(cpu_group, tcp_store_group)
        self.use_ep = get_current_vllm_config().parallel_config.enable_expert_parallel

    def _get_comm_group(self, is_sequence_parallel: bool) -> Any:
        if is_sequence_parallel:
            return get_ep_group()
        if self.dp_world_size > 1:
            if self.use_ep and get_pcp_group().world_size > 1:
                assert self.tp_group.world_size == 1, (
                    "DP+PCP with TP>1 requires sequence-parallel MoE inputs"
                )
                return get_ep_group()
            return get_dp_group()
        return get_pcp_group()

    def _get_sizes(self, num_local_tokens: int, comm_group: Any) -> list[int]:
        if self.dp_world_size == 1:
            return [num_local_tokens] * comm_group.world_size

        dp_metadata = get_forward_context().dp_metadata
        assert dp_metadata is not None
        sizes = dp_metadata.get_chunk_sizes_across_dp_rank()
        assert sizes is not None
        assert len(sizes) == comm_group.world_size
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
        """Gather hidden_states and router_logits from all dp ranks."""
        dist_group = self._get_comm_group(is_sequence_parallel)
        sizes = self._get_sizes(hidden_states.shape[0], dist_group)
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
        """Gather hidden_states and router_logits from all dp ranks."""
        dist_group = self._get_comm_group(is_sequence_parallel)
        sizes = self._get_sizes(hidden_states.shape[0], dist_group)
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
        """Reduce-scatter hidden_states across all dp ranks."""
        dist_group = self._get_comm_group(is_sequence_parallel)
        sizes = self._get_sizes(
            hidden_states.shape[0] // dist_group.world_size,
            dist_group,
        )
        hidden_states = dist_group.reduce_scatterv(hidden_states, dim=0, sizes=sizes)
        return hidden_states

    def allocate_combine_input(
        self,
        shape: tuple[int, ...],
        dtype: torch.dtype,
        device: torch.device,
        is_sequence_parallel: bool = False,
    ) -> torch.Tensor | None:
        dist_group = self._get_comm_group(is_sequence_parallel)
        sizes = self._get_sizes(shape[0] // dist_group.world_size, dist_group)
        if sum(sizes) != shape[0] or any(size != sizes[0] for size in sizes):
            return None
        device_communicator = dist_group.device_communicator
        if device_communicator is None:
            return None
        return device_communicator.get_symmetric_memory_buffer(
            "moe_ag_rs_combine", shape, dtype, device
        )

    def combine_into_output(
        self,
        hidden_states: torch.Tensor,
        output: torch.Tensor,
        is_sequence_parallel: bool = False,
    ) -> torch.Tensor:
        dist_group = self._get_comm_group(is_sequence_parallel)
        sizes = self._get_sizes(
            hidden_states.shape[0] // dist_group.world_size,
            dist_group,
        )
        return dist_group.reduce_scatterv_into_output(
            hidden_states,
            output,
            dim=0,
            sizes=sizes,
        )

    def destroy(self):
        pass


class DeepEPAll2AllManagerBase(All2AllManagerBase):
    """All2All communication based on DeepEP High-Throughput kernels."""

    def __init__(self, cpu_group, tcp_store_group=None):
        assert has_deep_ep(), (
            "DeepEP kernels not found. Please follow https://github.com/vllm-project/vllm/blob/main/tools/ep_kernels/README.md"
            " to install DeepEP kernels."
        )  # noqa
        super().__init__(cpu_group, tcp_store_group)
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
        with self.handle_cache._lock:
            for _, handle in self.handle_cache._cache.items():
                handle.destroy()
            self.handle_cache._cache.clear()


class DeepEPHTAll2AllManager(DeepEPAll2AllManagerBase):
    """All2All communication based on DeepEP High-Throughput kernels."""

    def __init__(self, cpu_group, tcp_store_group=None):
        super().__init__(cpu_group, tcp_store_group)

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
        # TODO: remove platform-specific logic
        # once ROCm DeepEP is updated with the latest APIs.
        kwargs = dict(
            group=self.cpu_group,
            num_nvl_bytes=num_nvl_bytes,
            num_rdma_bytes=num_rdma_bytes,
            low_latency_mode=False,
            num_qps_per_rank=num_qps_per_rank,
            explicitly_destroy=True,
        )
        return kwargs

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
    """All2All communication based on DeepEP Low-Latency kernels."""

    _buffer: Any = None
    _mask: torch.Tensor | None = None
    _last_mask: torch.Tensor | None = None

    def __init__(self, cpu_group, tcp_store_group=None):
        super().__init__(cpu_group, tcp_store_group)
        self.support_fault_tolerance = (
            get_current_vllm_config().parallel_config.enable_fault_tolerance
        )

    def _make_all2all_kwargs(
        self,
        max_num_tokens_per_dp_rank: int,
        token_hidden_size: int,
        num_ep_ranks: int,
        num_global_experts: int,
        num_local_experts: int,
    ) -> dict[Any, Any]:
        """max_num_tokens_per_dp_rank : the maximum number of tokens a DP rank
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
        # TODO: remove platform-specific logic
        # once ROCm DeepEP is updated with the latest APIs.
        kwargs = dict(
            group=self.cpu_group,
            num_nvl_bytes=num_nvl_bytes,
            num_rdma_bytes=num_rdma_bytes,
            low_latency_mode=True,
            num_qps_per_rank=num_qps_per_rank,
            allow_nvlink_for_low_latency_mode=True,
            allow_mnnvl=envs.VLLM_DEEPEP_LOW_LATENCY_USE_MNNVL,
            explicitly_destroy=True,
            enable_shrink=self.support_fault_tolerance,
        )
        return kwargs

    def get_handle(self, kwargs):
        """The kwargs for DeepEPLLAll2AllManager is dictated by
        _make_all2all_kwargs.
        """
        import deep_ep  # type: ignore[import-not-found]

        buffer_kwargs = self._make_all2all_kwargs(**kwargs)
        logger.debug("DeepEP all2all args %s", buffer_kwargs)
        handle: deep_ep.Buffer = self.handle_cache.get_or_create(
            buffer_kwargs, deep_ep.Buffer
        )
        DeepEPLLAll2AllManager._buffer = handle
        return handle

    # DeepEP LL uses RDMA so no SMs are used for communication
    def max_sms_used(self) -> int | None:
        return 0

    def query_active_mask(self) -> torch.Tensor:
        buf = DeepEPLLAll2AllManager._buffer
        assert buf is not None
        if DeepEPLLAll2AllManager._mask is None:
            DeepEPLLAll2AllManager._mask = torch.zeros(
                self.world_size, device="cuda", dtype=torch.int32
            )
        buf.low_latency_query_mask_buffer(DeepEPLLAll2AllManager._mask)
        return DeepEPLLAll2AllManager._mask

    def query_fault(self) -> torch.Tensor:
        current = self.query_active_mask()
        if DeepEPLLAll2AllManager._last_mask is None:
            DeepEPLLAll2AllManager._last_mask = torch.zeros_like(current)
        has_fault = (current != DeepEPLLAll2AllManager._last_mask).any()
        return has_fault

    def clean_buffers(self) -> None:
        buf = DeepEPLLAll2AllManager._buffer
        if buf is None:
            return
        buf.get_local_buffer_tensor(dtype=torch.int8, use_rdma_buffer=True).zero_()
        torch.accelerator.synchronize()
        buf.low_latency_clean_mask_buffer()
        torch.accelerator.synchronize()
        DeepEPLLAll2AllManager._last_mask = None


@dataclass
class _NixlEPBufferState:
    buffer: Any
    max_num_ep_ranks: int
    connected_ep_size: int
    active_ep_size: int
    # Cached init args so the buffer can be rebuilt after destroy without a
    # live forward pass to supply them (used by flash_epscale sleep/wake).
    init_args: dict[str, int] | None = None


class NixlEPAll2AllManager(All2AllManagerBase):
    """All2All communication based on NIXL EP kernels.
    This backend supports elastic EP with dynamic rank connection/disconnection.
    """

    _buffer: _NixlEPBufferState | None = None
    _mask: torch.Tensor | None = None
    _last_mask: torch.Tensor | None = None
    _lock = threading.RLock()
    # Cached init args (survives destroy_buffer so ensure_buffer can rebuild).
    _last_init_args: dict[str, int] | None = None
    # Monotonic counter incremented on every collective ensure_buffer call so
    # each barrier round uses a fresh TCP-store key namespace.
    _ensure_generation: int = 0

    def __init__(self, cpu_group, tcp_store_group=None):
        if tcp_store_group is None:
            tcp_store_group = StatelessProcessGroup(
                rank=cpu_group.rank(),
                world_size=cpu_group.size(),
                store=dist.PrefixStore("nixl_ep", cpu_group.get_group_store()),
            )
        super().__init__(cpu_group, tcp_store_group)
        self.support_fault_tolerance = True

    @property
    def max_num_ep_ranks(self) -> int:
        state = NixlEPAll2AllManager._buffer
        if state is not None:
            return state.max_num_ep_ranks
        return self._max_num_ep_ranks

    def _init_buffer(
        self,
        max_num_tokens_per_dp_rank: int,
        token_hidden_size: int,
        num_experts_per_rank: int,
    ) -> None:
        from nixl_ep import Buffer  # type: ignore[import-not-found]

        parallel_config = get_current_vllm_config().parallel_config
        max_num_ep_ranks = (
            self.world_size
            * parallel_config.elastic_ep_max_dp_size
            // parallel_config.data_parallel_size
            if parallel_config.enable_elastic_ep
            else self.world_size
        )
        max_num_global_experts = max_num_ep_ranks * num_experts_per_rank
        self._max_num_ep_ranks = max_num_ep_ranks
        num_rdma_bytes = Buffer.get_rdma_size_hint(
            num_max_dispatch_tokens_per_rank=max_num_tokens_per_dp_rank,
            hidden=token_hidden_size,
            num_ranks=max_num_ep_ranks,
            num_experts=max_num_global_experts,
        )
        logger.info(
            "NIXL EP RDMA buffer: %.3f GiB (max_num_ranks=%d, "
            "experts_per_rank=%d, max_tokens=%d, hidden=%d). This is "
            "non-torch memory sized for the max EP world (elastic EP), not "
            "the current world_size=%d.",
            num_rdma_bytes / (1024**3),
            max_num_ep_ranks,
            num_experts_per_rank,
            max_num_tokens_per_dp_rank,
            token_hidden_size,
            self.world_size,
        )
        assert NixlEPAll2AllManager._buffer is None, (
            "NIXL EP buffer already initialized"
        )
        buffer = Buffer(
            rank=self.rank,
            tcp_store_group=self.tcp_store_group.store,
            explicitly_destroy=True,
        )
        free_before, _ = torch.cuda.mem_get_info()
        buffer.update_memory_buffers(
            num_ranks=max_num_ep_ranks,
            num_experts_per_rank=num_experts_per_rank,
            num_rdma_bytes=num_rdma_bytes,
        )
        ranks_to_connect = list(range(self.world_size))
        buffer.connect_ranks(ranks_to_connect)
        free_after, _ = torch.cuda.mem_get_info()
        logger.info(
            "NIXL EP buffer non-torch memory (measured): %.1f MiB allocated "
            "(get_rdma_size_hint=%.1f MiB)",
            (free_before - free_after) / (1024**2),
            num_rdma_bytes / (1024**2),
        )
        init_args = {
            "max_num_tokens_per_dp_rank": max_num_tokens_per_dp_rank,
            "token_hidden_size": token_hidden_size,
            "num_experts_per_rank": num_experts_per_rank,
        }
        NixlEPAll2AllManager._buffer = _NixlEPBufferState(
            buffer=buffer,
            max_num_ep_ranks=max_num_ep_ranks,
            connected_ep_size=self.world_size,
            active_ep_size=self.world_size,
            init_args=init_args,
        )
        NixlEPAll2AllManager._last_init_args = init_args

    def disconnect_from_sleeping(self, sleeping_ep_ranks: list[int]) -> int:
        """On a LIVE rank, drop the connection to each rank that is about
        to be destroyed. This calls nixl's ``invalidateRemoteMD`` so that
        the live rank's nixl agent forgets the sleeping rank's UCX
        endpoint. Without this, a later ``connect_ranks`` on the sleeping
        rank hangs on the two-way handshake (the live side never picks up
        the new endpoint). Returns the number of peers dropped.
        """
        peers = [r for r in sleeping_ep_ranks if r != self.rank]
        if not peers:
            return 0
        with NixlEPAll2AllManager._lock:
            state = NixlEPAll2AllManager._buffer
            if state is None:
                return 0
            # disconnect_ranks removes them from remote_ranks and invalidates
            # both directions of the nixl agent view.
            state.buffer.disconnect_ranks(peers)
            state.connected_ep_size = max(
                0, state.connected_ep_size - len(peers)
            )
        logger.info(
            "NIXL EP live rank %d disconnected sleeping peers %s",
            self.rank,
            peers,
        )
        return len(peers)

    def destroy_buffer(self) -> int:
        """Release the RDMA buffer and drop all NIXL connections.

        Used by flash_epscale to reclaim the ~2 GiB NIXL non-torch memory on
        sleeping ranks. Must be called from a stop-the-world (paused) window;
        all in-flight dispatch/combine must have completed. Returns the number
        of MiB actually freed. Idempotent -- returns 0 if already destroyed.
        """
        with NixlEPAll2AllManager._lock:
            state = NixlEPAll2AllManager._buffer
            if state is None:
                return 0
            free_before, _ = torch.cuda.mem_get_info()
            # Buffer.destroy() releases the C++ runtime, which internally
            # tears down every connection. Calling disconnect_ranks first is
            # unnecessary and fragile (its assertion is sensitive to whether
            # self-rank was ever registered as "remote").
            state.buffer.destroy()
            NixlEPAll2AllManager._buffer = None
            free_after, _ = torch.cuda.mem_get_info()
            freed_mib = (free_after - free_before) / (1024**2)
            logger.info("NIXL EP buffer destroyed (freed %.1f MiB)", freed_mib)
            return int(freed_mib)

    def ensure_buffer(self, waking_ep_ranks: list[int] | None = None) -> str:
        """Restore all2all state so waking ranks can rejoin the EP group.

        This is a collective operation: every rank in the EP group must call
        it, because ``connect_ranks`` under the hood does a TCP-store
        rendezvous that requires both endpoints. Behavior per rank:

        * Rank in ``waking_ep_ranks`` with ``_buffer is None``: rebuilds
          the buffer from cached init args, then handshakes back to every
          peer. Returns "rebuilt".
        * Rank in ``waking_ep_ranks`` with a live buffer (nixl teardown
          was skipped during scale-down): the alive peers already dropped
          us from their remote_ranks via disconnect_from_sleeping, so a
          symmetric ``connect_ranks`` is still required to re-establish
          the two-way handshake. Returns "reconnected-waking".
        * Rank not in ``waking_ep_ranks`` (alive side) with waking peers:
          waits for each waking peer to publish its ready flag, then
          reconnects to those peers. Returns "reconnected".
        * Otherwise: no-op.

        The self-vs-waking membership -- not the local buffer state -- is
        what decides which side of the barrier we sit on. This matters
        when nixl teardown is skipped: buffer is still alive but the
        rank must still act as the waking side because alive peers have
        already ``disconnect_ranks``'d it.

        Synchronization: rebuild/rejoin (waking) side and alive side must
        enter ``connect_ranks`` in the same TCP-store rendezvous window,
        otherwise the alive side arrives first and hangs on the two-way
        handshake while the waking side is still allocating. The waking
        side publishes a readiness flag via the shared TCP store after
        its local setup is complete; the alive side waits on those flags
        before calling ``connect_ranks``. A generation counter guards
        against key reuse across successive waves.
        """
        waking = list(waking_ep_ranks or [])
        is_waking_side = self.rank in waking
        with NixlEPAll2AllManager._lock:
            gen = NixlEPAll2AllManager._ensure_generation
            NixlEPAll2AllManager._ensure_generation += 1
            state = NixlEPAll2AllManager._buffer
            need_rebuild = is_waking_side and state is None
        # Key namespace shared by waking-side publishers and alive-side
        # waiters. Generation guards against reuse across successive waves.
        ready_key = lambda r: f"vllm_nixl_ep_ensure_gen{gen}_rebuilt_{r}"
        store = self.tcp_store_group.store
        barrier_timeout = timedelta(seconds=60)

        peers_excluding_self = [
            r for r in range(self.world_size) if r != self.rank
        ]

        if need_rebuild:
            args = NixlEPAll2AllManager._last_init_args
            if args is None:
                raise RuntimeError(
                    "NIXL EP buffer has never been initialized; cannot "
                    "ensure (call get_handle from a MoE forward first)"
                )
            from nixl_ep import Buffer  # type: ignore[import-not-found]

            num_experts_per_rank = args["num_experts_per_rank"]
            max_num_global_experts = (
                self.max_num_ep_ranks * num_experts_per_rank
            )
            num_rdma_bytes = Buffer.get_rdma_size_hint(
                num_max_dispatch_tokens_per_rank=args[
                    "max_num_tokens_per_dp_rank"
                ],
                hidden=args["token_hidden_size"],
                num_ranks=self.max_num_ep_ranks,
                num_experts=max_num_global_experts,
            )
            buffer = Buffer(
                rank=self.rank,
                tcp_store_group=store,
                explicitly_destroy=True,
            )
            free_before, _ = torch.cuda.mem_get_info()
            buffer.update_memory_buffers(
                num_ranks=self.max_num_ep_ranks,
                num_experts_per_rank=num_experts_per_rank,
                num_rdma_bytes=num_rdma_bytes,
            )
            free_after, _ = torch.cuda.mem_get_info()
            logger.info(
                "NIXL EP buffer rebuilt: %.1f MiB allocated (about to "
                "connect_ranks)",
                (free_before - free_after) / (1024**2),
            )
            # Announce readiness so alive peers can enter connect_ranks in
            # the same rendezvous window as us. Must happen after the local
            # buffer is fully constructed -- alive-side connect_ranks will
            # publish/fetch metadata via the same TCP store and only
            # completes once our side is ready to fetch theirs.
            store.set(ready_key(self.rank), "1")
            # Symmetric handshake: connect to every peer in the current
            # world (excluding self -- nixl's connect_ranks rejects
            # self-connection). Alive peers are simultaneously connecting
            # back to us via their reconnect branch below.
            buffer.connect_ranks(peers_excluding_self)
            with NixlEPAll2AllManager._lock:
                NixlEPAll2AllManager._buffer = _NixlEPBufferState(
                    buffer=buffer,
                    max_num_ep_ranks=self.max_num_ep_ranks,
                    connected_ep_size=self.world_size,
                    active_ep_size=self.world_size,
                    init_args=args,
                )
            logger.info("NIXL EP buffer rebuild finished (connect_ranks OK)")
            with NixlEPAll2AllManager._lock:
                cur = NixlEPAll2AllManager._buffer
                if cur is not None:
                    cur.active_ep_size = self.world_size
            self._dump_mask_state(tag="after-rebuilt-connect")
            return "rebuilt"

        if is_waking_side:
            # Buffer is still live because nixl teardown was skipped this
            # wave (e.g. CuMem-only sleep). Alive peers already ran
            # disconnect_from_sleeping on us during scale-down, so their
            # remote_ranks no longer contain us -- but on our side, every
            # alive peer is still registered as remote (we did nothing).
            # nixl's connect_ranks is idempotent w.r.t. already-known
            # peers: if rank X is already in remote_ranks it returns
            # without publishing new metadata to the TCP store, and the
            # alive side's connect_ranks(us) then hangs waiting for
            # metadata that will never arrive. To force a real symmetric
            # handshake we first disconnect our known alive peers, then
            # reconnect. This mirrors the destroy/rebuild path (where a
            # brand-new buffer starts with an empty remote_ranks) without
            # actually reallocating the RDMA buffer.
            assert state is not None
            key = ready_key(self.rank)
            logger.info(
                "NIXL EP waking rank %d clearing stale remote_ranks %s "
                "before symmetric reconnect (gen=%d)",
                self.rank, peers_excluding_self, gen,
            )
            try:
                state.buffer.disconnect_ranks(peers_excluding_self)
            except Exception:
                logger.warning(
                    "NIXL EP waking rank %d: disconnect_ranks(%s) raised; "
                    "continuing anyway",
                    self.rank, peers_excluding_self, exc_info=True,
                )
            logger.info(
                "NIXL EP waking rank %d publishing ready key %r (gen=%d)",
                self.rank, key, gen,
            )
            store.set(key, "1")
            logger.info(
                "NIXL EP waking rank %d ready key set; calling connect_ranks(%s)",
                self.rank, peers_excluding_self,
            )
            state.buffer.connect_ranks(peers_excluding_self)
            with NixlEPAll2AllManager._lock:
                cur = NixlEPAll2AllManager._buffer
                if cur is not None:
                    cur.connected_ep_size = self.world_size
                    cur.active_ep_size = self.world_size
            logger.info(
                "NIXL EP waking rank %d re-handshaked (buffer was kept "
                "across sleep)",
                self.rank,
            )
            # See rebuilt branch: connect_ranks(activate=True) already
            # took care of unmasking and active_rank_bound refresh.
            return "reconnected-waking"

        # Alive side. Because scale-down invoked disconnect_from_sleeping
        # on us, the waking peers are no longer in our remote_ranks --
        # nixl's connect_ranks will genuinely handshake with them instead
        # of skipping. Run the symmetric rendezvous in parallel with the
        # waking side so the TCP-store windows overlap.
        if not waking:
            return "noop"
        peers_to_reconnect = [r for r in waking if r != self.rank]
        if not peers_to_reconnect:
            return "noop"
        # Wait for every waking peer to publish its ready flag. Without
        # this we would enter connect_ranks first and hang inside nixl's
        # two-way handshake while the waking side is still allocating.
        wait_keys = [ready_key(r) for r in peers_to_reconnect]
        logger.info(
            "NIXL EP live rank %d waiting for waking peers %s to finish "
            "buffer rebuild (gen=%d, keys=%s)",
            self.rank,
            peers_to_reconnect,
            gen,
            wait_keys,
        )
        store.wait(wait_keys, barrier_timeout)
        logger.info(
            "NIXL EP live rank %d ready keys received; calling connect_ranks(%s)",
            self.rank, peers_to_reconnect,
        )
        with NixlEPAll2AllManager._lock:
            state = NixlEPAll2AllManager._buffer
            assert state is not None
            buffer = state.buffer
        buffer.connect_ranks(peers_to_reconnect)
        logger.info(
            "NIXL EP live rank %d connect_ranks returned",
            self.rank,
        )
        with NixlEPAll2AllManager._lock:
            state = NixlEPAll2AllManager._buffer
            if state is not None:
                state.connected_ep_size = self.world_size
                state.active_ep_size = self.world_size
        logger.info(
            "NIXL EP live rank %d reconnected to waking peers %s",
            self.rank,
            peers_to_reconnect,
        )
        # See rebuilt branch: connect_ranks(activate=True) already
        # took care of unmasking and active_rank_bound refresh.
        self._dump_mask_state(tag="after-reconnected")
        return "reconnected"

    def _dump_mask_state(self, *, tag: str) -> None:
        """Query the current nixl mask buffer and log which ranks are masked
        vs. unmasked. Also derive active_rank_bound the same way nixl
        does (largest-active-rank + 1) so we can tell if dispatch- and
        combine-time bounds agree.
        """
        with NixlEPAll2AllManager._lock:
            state = NixlEPAll2AllManager._buffer
            if state is None:
                logger.info(
                    "NIXL EP rank %d mask-dump[%s]: buffer=None",
                    self.rank, tag,
                )
                return
            try:
                mask_status = torch.zeros(
                    self.max_num_ep_ranks,
                    dtype=torch.int32,
                    device="cuda",
                )
                state.buffer.query_mask_buffer(mask_status)
                torch.cuda.current_stream().synchronize()
                m = mask_status.tolist()
                # active_rank_bound = max_id_where_unmasked + 1
                bound = 0
                for r_id in range(self.max_num_ep_ranks - 1, -1, -1):
                    if m[r_id] == 0:  # 0 = unmasked = active
                        bound = r_id + 1
                        break
            except Exception:
                logger.warning(
                    "NIXL EP rank %d mask-dump[%s]: query FAILED",
                    self.rank, tag, exc_info=True,
                )
                return
        logger.info(
            "NIXL EP rank %d mask-dump[%s]: mask=%s => active_rank_bound=%d",
            self.rank, tag, m, bound,
        )

    def set_masked_ranks(self, masked_ranks: list[int]) -> None:
        """Update the shared NIXL shrink mask without changing process groups.

        This is a control-plane operation for logical sleep / fast elasticity:
        all ranks remain connected, but dispatch/combine skip the masked peers.
        """
        with NixlEPAll2AllManager._lock:
            if NixlEPAll2AllManager._buffer is None:
                raise RuntimeError("NIXL EP buffer is not initialized")

            state = NixlEPAll2AllManager._buffer
            masked = {int(rank) for rank in masked_ranks}
            world_size = self.world_size
            invalid = sorted(rank for rank in masked if rank < 0 or rank >= world_size)
            if invalid:
                raise ValueError(
                    f"masked_ranks must be in [0, {world_size}), got {invalid}"
                )

            logger.info(
                "NIXL EP rank %d set_masked_ranks ENTER: masked=%s",
                self.rank, sorted(masked),
            )
            for rank in range(world_size):
                should_mask = rank in masked
                # nixl asserts (rank_to_mask != self or !mask): a rank cannot
                # mask itself. This is a collective call with the same masked
                # list on every rank, so ranks that appear in the mask are
                # the ones being put to sleep -- they will next call
                # destroy_buffer and stop participating anyway. Skip the
                # self-mask entry on those ranks; alive ranks still mask
                # them, which is what dispatch/combine actually reads.
                if should_mask and rank == self.rank:
                    continue
                state.buffer.update_mask_buffer(rank, should_mask)
            state.active_ep_size = world_size - len(masked)

            # Control-plane operation: make the updated peer mask visible before
            # the sleep / wake API returns and new requests enter the model path.
            torch.cuda.current_stream().synchronize()
            logger.info(
                "NIXL EP rank %d set_masked_ranks EXIT: masked=%s",
                self.rank, sorted(masked),
            )
        self._dump_mask_state(tag=f"after-set_masked({sorted(masked)})")

    def _connect_to_ep_size(self, ep_size: int, *, make_active: bool) -> None:
        assert NixlEPAll2AllManager._buffer is not None
        state = NixlEPAll2AllManager._buffer
        if ep_size <= state.connected_ep_size:
            return

        state.buffer.set_tcp_store_group(self.tcp_store_group.store)
        ranks_to_connect = list(range(state.connected_ep_size, ep_size))
        state.buffer.connect_ranks(ranks_to_connect, activate=make_active)
        state.connected_ep_size = ep_size
        if make_active:
            state.active_ep_size = ep_size

    def _disconnect_to_ep_size(self, ep_size: int) -> None:
        assert NixlEPAll2AllManager._buffer is not None
        state = NixlEPAll2AllManager._buffer
        if ep_size >= state.connected_ep_size:
            return

        state.buffer.set_tcp_store_group(self.tcp_store_group.store)
        ranks_to_disconnect = list(range(ep_size, state.connected_ep_size))
        state.buffer.disconnect_ranks(ranks_to_disconnect)
        state.connected_ep_size = ep_size
        state.active_ep_size = min(state.active_ep_size, ep_size)

    def _unmask_connected_ranks(self, target_ep_size: int) -> None:
        assert NixlEPAll2AllManager._buffer is not None
        state = NixlEPAll2AllManager._buffer
        state.buffer.set_tcp_store_group(self.tcp_store_group.store)
        if target_ep_size <= state.active_ep_size:
            return
        assert state.connected_ep_size >= target_ep_size

        for rank in range(state.active_ep_size, target_ep_size):
            state.buffer.update_mask_buffer(rank, mask=False)
        torch.accelerator.synchronize()
        state.active_ep_size = target_ep_size

    def _stage_ep_size(self) -> None:
        assert NixlEPAll2AllManager._buffer is not None
        state = NixlEPAll2AllManager._buffer
        target_ep_size = self.world_size

        # Scale-up can safely connect standby ranks while leaving them masked.
        # Scale-down must not disconnect active ranks until commit.
        if target_ep_size > state.connected_ep_size:
            self._connect_to_ep_size(target_ep_size, make_active=False)

    def stage_ep_size(self) -> None:
        with NixlEPAll2AllManager._lock:
            self._stage_ep_size()

    def commit_ep_size(self) -> None:
        """Commit the staged EP size to the active communication set."""
        with NixlEPAll2AllManager._lock:
            assert NixlEPAll2AllManager._buffer is not None
            state = NixlEPAll2AllManager._buffer
            target_ep_size = self.world_size

            if target_ep_size < state.connected_ep_size:
                self._disconnect_to_ep_size(target_ep_size)
            elif target_ep_size > state.connected_ep_size:
                self._connect_to_ep_size(target_ep_size, make_active=True)

            self._unmask_connected_ranks(target_ep_size)

    @contextmanager
    def mask_remote_ranks(self) -> Iterator[None]:
        peers = [rank for rank in range(self.world_size) if rank != self.rank]
        assert NixlEPAll2AllManager._buffer is not None
        buffer = NixlEPAll2AllManager._buffer.buffer
        for rank in peers:
            buffer.update_mask_buffer(rank, mask=True)
        torch.accelerator.synchronize()
        try:
            yield
        finally:
            for rank in peers:
                buffer.update_mask_buffer(rank, mask=False)
            torch.accelerator.synchronize()

    def get_handle(self, kwargs):
        with NixlEPAll2AllManager._lock:
            state = NixlEPAll2AllManager._buffer
            if state is None:
                max_num_tokens_per_dp_rank = kwargs["max_num_tokens_per_dp_rank"]
                self._init_buffer(
                    max_num_tokens_per_dp_rank=max_num_tokens_per_dp_rank,
                    token_hidden_size=kwargs["token_hidden_size"],
                    num_experts_per_rank=kwargs["num_local_experts"],
                )

            assert NixlEPAll2AllManager._buffer is not None
            handle = NixlEPAll2AllManager._buffer.buffer
            return handle

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
        # NOTE(yongji): NIXLEPAll2AllManager instance is recreated during
        # scale-up/down, so we cannot destroy the persistent buffer here.
        assert NixlEPAll2AllManager._buffer is not None
        buffer = NixlEPAll2AllManager._buffer.buffer
        buffer.set_tcp_store_group(None)

    # NIXL EP uses RDMA so no SMs are used for communication
    def max_sms_used(self) -> int | None:
        return 0

    def query_active_mask(self) -> torch.Tensor:
        state = NixlEPAll2AllManager._buffer
        assert state is not None
        if NixlEPAll2AllManager._mask is None:
            NixlEPAll2AllManager._mask = torch.zeros(
                state.max_num_ep_ranks, device="cuda", dtype=torch.int32
            )
        state.buffer.query_mask_buffer(NixlEPAll2AllManager._mask)
        return NixlEPAll2AllManager._mask[: state.active_ep_size]

    def query_fault(self) -> torch.Tensor:
        current = self.query_active_mask()
        last = NixlEPAll2AllManager._last_mask
        if last is None or last.shape != current.shape:
            NixlEPAll2AllManager._last_mask = torch.zeros_like(current)
            last = NixlEPAll2AllManager._last_mask
        has_fault = (current != last).any()
        return has_fault

    def clean_buffers(self) -> None:
        if NixlEPAll2AllManager._buffer is None:
            return
        state = NixlEPAll2AllManager._buffer
        state.buffer.get_local_buffer_tensor(
            dtype=torch.int8, use_rdma_buffer=True
        ).zero_()
        torch.accelerator.synchronize()
        state.buffer.clean_mask_buffer()
        torch.accelerator.synchronize()
        NixlEPAll2AllManager._last_mask = None


class FlashInferNVLinkTwoSidedManager(All2AllManagerBase):
    """All2All communication based on flashinfer all2allv/two-sided NVLink kernels."""

    # This type lint could be removed after all of the work in
    # https://github.com/vllm-project/vllm/issues/26533 done.
    rank: int
    world_size: int

    def __init__(self, cpu_group, tcp_store_group=None):
        assert has_flashinfer_nvlink_two_sided(), (
            "flashinfer all2all module not found. Please install/check flashinfer"
        )  # noqa
        super().__init__(cpu_group, tcp_store_group)
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
        """Initialize workspace."""
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

        # MNNVL workspace is allocated per rank in the comm_backend's group; the
        # flashinfer kernel asserts workspace.size(0) == moe_ep_size, so the backend
        # must span the EP group (= DP*PCP*TP), not the DP group.
        ep_config = MnnvlConfig(
            comm_backend=CustomCommunicator(self.cpu_group),
            fabric_page_size=1 << 29,  # 512MB
            allocation_granularity=0,  # Auto-detect
        )

        self.workspace_tensor = MnnvlMoe.get_moe_workspaces(self.mapping, ep_config)
        self.prepare_workspace_tensor = MnnvlMoe.get_moe_prepare_workspace(
            self.mapping, ep_config
        )

        self.world_size = world_size
        self.rank = rank
        self.gpus_per_node = gpus_per_node
        self.initialized = True

        logger.info(
            "FlashInfer All2All initialized for rank %s, size %s", rank, world_size
        )

    def ensure_alltoall_workspace_initialized(self):
        """Ensure workspace is initialized."""
        if not has_flashinfer_nvlink_two_sided():
            return False

        if self.world_size <= 1:
            return False

        if not self.initialized:
            self.initialize(
                world_size=self.world_size,
                rank=self.rank,
                gpus_per_node=torch.accelerator.device_count,
            )
        return self.initialized

    def get_handle(self, kwargs):
        return self

    def cleanup(self):
        """Clean up workspace."""
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


class FlashInferNVLinkOneSidedManager(All2AllManagerBase):
    """All2All communication based on FlashInfer's MoeAlltoAll/One-sided NVLink kernel.
    This is a newer kernel from trtllm that should perform better than the kernel
    used by flashinfer_nvlink_two_sided.
    """

    rank: int
    world_size: int
    low_precision_combine: bool = False

    def __init__(self, cpu_group):
        assert has_flashinfer_nvlink_one_sided(), (
            "flashinfer trtllm_moe_alltoall module not found. "
            "Please install/check flashinfer"
        )
        super().__init__(cpu_group)
        logger.debug(
            "Initialize FlashInfer One-sided NVLink rank=%d, world size=%d",
            self.rank,
            self.world_size,
        )
        self.initialized = False
        self.moe_alltoall: MoeAlltoAll | None = None
        self.mapping = None
        self.workspace_size = 0
        self.max_num_tokens = 0
        self.top_k = 0
        self.num_experts = 0
        self._combine_supports_output = False
        self.low_precision_combine = self._resolve_low_precision_combine()

    def _resolve_low_precision_combine(self) -> bool:
        """Whether to use the low-precision combine: requested via the env var
        and supported by the installed FlashInfer.
        """
        if not envs.VLLM_FLASHINFER_MOE_A2A_LOW_PRECISION_COMBINE:
            return False
        try:
            supported = supports_kw(
                MoeAlltoAll.combine, "use_low_precision", allow_var_kwargs=False
            )
        except (TypeError, ValueError):
            supported = False
        if not supported:
            logger.warning_once(
                "VLLM_FLASHINFER_MOE_A2A_LOW_PRECISION_COMBINE is set, but the "
                "installed FlashInfer MoeAlltoAll.combine() does not accept "
                "`use_low_precision`. Falling back to a BF16 combine."
            )
        return supported

    def initialize(
        self,
        max_num_tokens: int,
        top_k: int,
        num_experts: int,
        hidden_size: int,
        x_bytes_per_token: int,
        x_sf_bytes_per_token: int,
    ):
        """Initialize (or grow) the MoeAlltoAll workspace."""
        total_dispatch_payload_size_per_token = (
            x_bytes_per_token
            + x_sf_bytes_per_token
            + top_k * 4  # int32 topks ids
            + top_k * 4  # float32 topk weights
        )
        # Sized from the bf16 payload passed to combine(), which is what the
        # kernel checks this region against. Low-precision transport quantizes
        # on write, so it does not shrink the requirement.
        combine_payload_size_per_token = hidden_size * 2
        needed_workspace_size = moe_a2a_get_workspace_size_per_rank(
            ep_size=self.world_size,
            max_num_tokens=max_num_tokens,
            total_dispatch_payload_size_per_token=total_dispatch_payload_size_per_token,
            combine_payload_size_per_token=combine_payload_size_per_token,
        )
        # workspace_size and max_num_tokens are kernel-side max-bounds, so
        # heterogeneous MoE layers (e.g. NVFP4 base + bf16 MTP head) only
        # need the shared workspace grown to the union. top_k and num_experts
        # must match across layers: top_k is a strict-equality assert at
        # dispatch (FlashInfer csrc/trtllm_moe_alltoall.cu), and num_experts
        # feeds the expert-to-rank routing math, so any mismatch would crash
        # or silently corrupt routing. All ranks see the same MoE layers in
        # the same order with identical shapes, so the skip / rebuild
        # branches are taken consistently across ranks.
        if self.initialized:
            assert top_k == self.top_k, (
                "FlashInfer one-sided MoeAlltoAll does not support "
                f"heterogeneous top_k across MoE layers (got {top_k}, "
                f"was built with {self.top_k})"
            )
            assert num_experts == self.num_experts, (
                "FlashInfer one-sided MoeAlltoAll does not support "
                f"heterogeneous num_experts across MoE layers (got "
                f"{num_experts}, was built with {self.num_experts})"
            )
            if (
                needed_workspace_size <= self.workspace_size
                and max_num_tokens <= self.max_num_tokens
            ):
                return

        self.workspace_size = max(self.workspace_size, needed_workspace_size)
        self.max_num_tokens = max(self.max_num_tokens, max_num_tokens)
        self.top_k = top_k
        self.num_experts = num_experts

        self.cleanup()
        from vllm.platforms.interface import get_assigned_physical_gpu_ids

        assigned_physical_gpu_ids = get_assigned_physical_gpu_ids()
        gpus_per_node = (
            len(assigned_physical_gpu_ids)
            if assigned_physical_gpu_ids is not None
            else torch.accelerator.device_count()
        )
        logger.debug(
            "Making One-sided NVLink mapping: rank=%d, world size=%d",
            self.rank,
            self.world_size,
        )
        self.mapping = Mapping(
            self.world_size,
            self.rank,
            gpus_per_node,
            tp_size=self.world_size,
            moe_ep_size=self.world_size,
        )

        from vllm.distributed.device_communicators.mnnvl_compat import (
            CustomCommunicator,
        )

        # MNNVL workspace is allocated per rank in the comm_backend's group; the
        # flashinfer kernel asserts workspace.size(0) == moe_ep_size, so the backend
        # must span the EP group (= DP*PCP*TP), not the DP group.
        ep_config = MnnvlConfig(
            comm_backend=CustomCommunicator(self.cpu_group),
        )

        # Release cached allocator blocks before FlashInfer reserves the
        # symmetric MNNVL fabric workspace.
        torch.accelerator.empty_cache()

        self.moe_alltoall = MoeAlltoAll(
            mapping=self.mapping,
            max_num_tokens=self.max_num_tokens,
            top_k=self.top_k,
            num_experts=self.num_experts,
            workspace_size_per_rank=self.workspace_size,
            mnnvl_config=ep_config,
        )
        try:
            self._combine_supports_output = supports_kw(
                self.moe_alltoall.combine, "output", allow_var_kwargs=False
            )
        except (TypeError, ValueError):
            self._combine_supports_output = False

        self.gpus_per_node = gpus_per_node
        self.initialized = True

        logger.info(
            "FlashInfer One-sided NVLink initialized for rank %s, size %s",
            self.rank,
            self.world_size,
        )
        # Scope barrier to the EP group: with PP, different EP groups can
        # rebuild a different number of times if their MoE layers have
        # different shape sequences, so a world-level barrier would deadlock.
        dist.barrier(group=self.cpu_group)

    def combine_into(
        self,
        payload: torch.Tensor,
        runtime_max_tokens_per_rank: int,
        output: torch.Tensor,
    ) -> None:
        """Combine into ``output``, with a fallback for older FlashInfer."""
        assert self.moe_alltoall is not None
        if self._combine_supports_output:
            self.moe_alltoall.combine(
                payload=payload,
                runtime_max_tokens_per_rank=runtime_max_tokens_per_rank,
                output=output,
                use_low_precision=self.low_precision_combine,
            )
        else:
            # FlashInfer < 0.6.16 has neither `output` nor `use_low_precision`.
            combined_output = self.moe_alltoall.combine(
                payload=payload,
                runtime_max_tokens_per_rank=runtime_max_tokens_per_rank,
            )
            output.copy_(combined_output)

    def get_handle(self, kwargs):
        return self

    def cleanup(self):
        """Clean up resources."""
        if self.initialized and self.moe_alltoall is not None:
            try:
                del self.moe_alltoall
            except Exception as e:
                logger.warning(
                    "Failed to cleanup FlashInfer One-sided NVLink workspace: %s", e
                )
            finally:
                self.moe_alltoall = None
                self.mapping = None
                self.initialized = False

    def checkpoint_prepare(self) -> None:
        if self.initialized:
            assert self.moe_alltoall is not None
            self.moe_alltoall.checkpoint_prepare()

    def checkpoint_restore(self) -> None:
        if self.initialized:
            assert self.moe_alltoall is not None
            from vllm.distributed.device_communicators.mnnvl_compat import (
                CustomCommunicator,
            )

            self.moe_alltoall.checkpoint_restore(CustomCommunicator(self.cpu_group))


class MoriAll2AllManager(All2AllManagerBase):
    def __init__(self, cpu_group, all2all_backend: str):
        assert has_mori(), (
            "MoRI kernels not found. Please follow https://github.com/ROCm/mori/blob/main/README.md"
            " to install MoRI kernels."
        )  # noqa
        assert all2all_backend in (
            "mori_high_throughput",
            "mori_low_latency",
        ), f"unsupported MoRI all2all backend: {all2all_backend!r}"
        import mori

        super().__init__(cpu_group)
        self._all2all_backend = all2all_backend
        self.handle_cache = Cache()

        torch._C._distributed_c10d._register_process_group("mori", cpu_group)
        if get_current_vllm_config().kernel_config.moe_backend == "aiter_mega_moe":
            # MegaMoEV2 places its dispatch/combine workspaces on the MoRI
            # symmetric heap, which defaults to 2 GB. MegaMoEV2 requires > 4GB
            heap_size = os.environ.setdefault("MORI_SHMEM_HEAP_SIZE", "8G")
            logger.info_once("AITER MegaMoE: MORI_SHMEM_HEAP_SIZE=%s", heap_size)
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
            # Multi-node: kernel follows --all2all-backend (mirrors deepep_* split).
            # mori_low_latency → InterNodeV1LL; mori_high_throughput → V1.
            if self._all2all_backend == "mori_low_latency":
                kernel_type = mori.ops.EpDispatchCombineKernelType.InterNodeV1LL
            else:
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


class DeepEPV2All2AllManager(All2AllManagerBase):
    """All2All communication based on DeepEP v2 ElasticBuffer (unified API).
    Uses NCCL Gin backend with analytical SM calculation.
    """

    def __init__(self, cpu_group, tcp_store_group=None, device_group=None):
        assert has_deep_ep_v2(), deep_ep_v2_unavailable_reason()
        super().__init__(cpu_group, tcp_store_group)
        self._device_group = device_group
        self.handle_cache = Cache()
        self._num_sms: int | None = None
        self._gin_checked = False

    def _make_all2all_kwargs(
        self,
        num_max_tokens_per_rank: int,
        hidden: int,
        num_topk: int,
        use_fp8_dispatch: bool,
    ) -> dict:
        return dict(
            group=self._device_group
            if self._device_group is not None
            else self.cpu_group,
            num_max_tokens_per_rank=num_max_tokens_per_rank,
            hidden=hidden,
            num_topk=num_topk,
            use_fp8_dispatch=use_fp8_dispatch,
            allow_hybrid_mode=envs.VLLM_DEEPEP_V2_ALLOW_HYBRID_MODE,
            prefer_overlap_with_compute=envs.VLLM_DEEPEP_V2_PREFER_OVERLAP,
            allow_multiple_reduction=(envs.VLLM_DEEPEP_V2_ALLOW_MULTIPLE_REDUCTION),
            explicitly_destroy=True,
        )

    def _check_gin_support(self, group) -> None:
        from vllm.utils.nccl import query_nccl_gin_type

        # ProcessGroupNCCL creates communicators lazily. Initialize this exact
        # group before querying so a null comm pointer is not mistaken for
        # missing GIN support.
        probe = torch.zeros(1, device="cuda")
        torch.distributed.all_reduce(probe, group=group)
        # DeepEPv2 respects EP_DISABLE_GIN, so skip the GIN requirement check.
        if os.environ.get("EP_DISABLE_GIN", "0") != "0":
            return

        gin_type = query_nccl_gin_type(
            group, railed=envs.VLLM_DEEPEP_V2_ALLOW_HYBRID_MODE
        )
        if gin_type is None:
            raise RuntimeError(
                "DeepEPv2 communicator properties query failed; "
                "networking capability could not be determined."
            )
        if gin_type == 0:
            raise RuntimeError(
                "DeepEPv2 requires NCCL GIN (GPU-Initiated Networking). "
                "This usually means IBGDA-capable InfiniBand NICs or drivers "
                "are not available. See tools/ep_kernels/README.md for "
                "requirements."
            )

    def get_handle(self, kwargs):
        import deep_ep  # type: ignore[import-not-found]

        num_experts = kwargs.pop("num_experts", 256)
        buffer_kwargs = self._make_all2all_kwargs(**kwargs)
        if not self._gin_checked:
            self._check_gin_support(buffer_kwargs["group"])
            self._gin_checked = True
        logger.debug("DeepEP v2 all2all args %s", buffer_kwargs)
        handle: deep_ep.ElasticBuffer = self.handle_cache.get_or_create(
            buffer_kwargs, deep_ep.ElasticBuffer
        )
        if self._num_sms is None:
            self._num_sms = handle.get_theoretical_num_sms(
                num_experts=num_experts,
                num_topk=kwargs["num_topk"],
            )
        return handle

    def max_sms_used(self) -> int | None:
        return self._num_sms

    def destroy(self):
        with self.handle_cache._lock:
            for _, handle in self.handle_cache._cache.items():
                handle.destroy()
            self.handle_cache._cache.clear()


class MoonEPAll2AllManager(All2AllManagerBase):
    """All2All communication based on MoonEP
    (https://github.com/MoonshotAI/MoonEP).

    MoonEP keeps token loads perfectly balanced across EP ranks by planning
    a small number of dynamically redundant experts online and prefetching
    their weights before expert compute. Every rank receives exactly
    S x K token slots regardless of router skew, so all communication and
    compute shapes are static.

    Requires NVLink symmetric-memory / multicast capable topologies
    (single node NVSwitch, e.g. H100/H200/B200/GB300 class).
    """

    def __init__(self, cpu_group, tcp_store_group=None, device_group=None):
        check_moonep_system_support()
        super().__init__(cpu_group, tcp_store_group)
        self._device_group = device_group
        self.handle_cache = Cache()

    def _make_buffer_kwargs(
        self,
        max_num_tokens_per_dp_rank: int,
        token_hidden_size: int,
        num_topk: int,
        num_global_experts: int,
        num_prefetch_slots: int,
        token_padding: int,
        num_sms: int,
    ) -> dict:
        return dict(
            S=max_num_tokens_per_dp_rank,
            H=token_hidden_size,
            K=num_topk,
            E=num_global_experts,
            num_ep_ranks=self.world_size,
            num_sms=num_sms,
            token_padding=token_padding,
            B=num_prefetch_slots,
            group=self._device_group
            if self._device_group is not None
            else self.cpu_group,
            explicitly_destroy=True,
        )

    def get_handle(self, kwargs):
        from vllm.model_executor.layers.fused_moe.prepare_finalize.moonep import (
            MoonEPBufferPool,
        )

        buffer_kwargs = self._make_buffer_kwargs(**kwargs)
        logger.debug("MoonEP all2all args %s", buffer_kwargs)

        def make_pool(**kw):
            return MoonEPBufferPool(kw, max_tokens_per_rank=kw["S"])

        handle: MoonEPBufferPool = self.handle_cache.get_or_create(
            buffer_kwargs, make_pool
        )
        return handle

    def destroy(self):
        with self.handle_cache._lock:
            for _, handle in self.handle_cache._cache.items():
                handle.destroy()
            self.handle_cache._cache.clear()
