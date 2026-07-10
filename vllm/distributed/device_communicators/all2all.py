# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import threading
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

import torch
import torch.distributed as dist

import vllm.envs as envs
from vllm.distributed import get_dp_group, get_ep_group
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.utils.flashinfer import (
    has_flashinfer_nvlink_one_sided,
    has_flashinfer_nvlink_two_sided,
)
from vllm.utils.import_utils import has_deep_ep, has_deep_ep_v2, has_mori

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


class AgRsAll2AllManager(All2AllManagerBase):
    """
    An implementation of all2all communication based on
    all-gather (dispatch) and reduce-scatter (combine).
    """

    def __init__(self, cpu_group, tcp_store_group=None):
        super().__init__(cpu_group, tcp_store_group)

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


class DeepEPAll2AllManagerBase(All2AllManagerBase):
    """
    All2All communication based on DeepEP High-Throughput kernels.
    """

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
        )
        return kwargs

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


@dataclass
class _NixlEPBufferState:
    buffer: Any
    connected_ep_size: int
    active_ep_size: int
    # Cached init args so the buffer can be rebuilt after destroy without a
    # live forward pass to supply them (used by flash_epscale sleep/wake).
    init_args: dict[str, int] | None = None


class NixlEPAll2AllManager(All2AllManagerBase):
    """
    All2All communication based on NIXL EP kernels.
    This backend supports elastic EP with dynamic rank connection/disconnection.
    """

    _buffer: _NixlEPBufferState | None = None
    _lock = threading.RLock()
    # Cached init args (survives destroy_buffer so ensure_buffer can rebuild).
    _last_init_args: dict[str, int] | None = None
    # Monotonic counter incremented on every collective ensure_buffer call so
    # each barrier round uses a fresh TCP-store key namespace.
    _ensure_generation: int = 0

    def __init__(self, cpu_group, tcp_store_group=None):
        assert tcp_store_group is not None
        super().__init__(cpu_group, tcp_store_group)

        self.max_num_ep_ranks = envs.VLLM_NIXL_EP_MAX_NUM_RANKS

    def _init_buffer(
        self,
        max_num_tokens_per_dp_rank: int,
        token_hidden_size: int,
        num_experts_per_rank: int,
    ) -> None:
        from nixl_ep import Buffer  # type: ignore[import-not-found]

        max_num_global_experts = self.max_num_ep_ranks * num_experts_per_rank
        num_rdma_bytes = Buffer.get_rdma_size_hint(
            num_max_dispatch_tokens_per_rank=max_num_tokens_per_dp_rank,
            hidden=token_hidden_size,
            num_ranks=self.max_num_ep_ranks,
            num_experts=max_num_global_experts,
        )
        logger.info(
            "NIXL EP RDMA buffer: %.3f GiB (max_num_ranks=%d, "
            "experts_per_rank=%d, max_tokens=%d, hidden=%d). This is "
            "non-torch memory sized for the max EP world (elastic EP), not "
            "the current world_size=%d.",
            num_rdma_bytes / (1024**3),
            self.max_num_ep_ranks,
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
            num_ranks=self.max_num_ep_ranks,
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

    def _clear_all_peer_masks(self, *, reason: str) -> None:
        """Force every peer's mask bit to False after a wake-time
        reconnect. During scale-down alive ranks set the mask for
        sleeping peers via set_masked_ranks; on wake the mask must be
        cleared for the wake path to actually let dispatch/combine
        talk to those peers again. Without this, connect_ranks
        succeeds but the kernel still skips the peer, showing up as
        NIXL-EP dispatch-receive timeouts on every local expert.

        This is a defensive belt-and-suspenders: the API-router path
        also calls resize_sleep_ep_ranks([]) after wake which reaches
        set_masked_ranks and would clear the mask too, but doing it
        here ties the mask reset to the same collective that just
        rebuilt the connection so the two states cannot diverge.
        """
        with NixlEPAll2AllManager._lock:
            state = NixlEPAll2AllManager._buffer
            if state is None:
                return
            cleared = []
            for rank in range(self.world_size):
                if rank == self.rank:
                    continue
                try:
                    state.buffer.update_mask_buffer(rank, False)
                    cleared.append(rank)
                except Exception:
                    logger.warning(
                        "NIXL EP rank %d: update_mask_buffer(%d, False) "
                        "raised while clearing masks after %s",
                        self.rank, rank, reason, exc_info=True,
                    )
            state.active_ep_size = self.world_size
            torch.cuda.current_stream().synchronize()
        logger.info(
            "NIXL EP rank %d cleared peer masks %s after %s",
            self.rank, cleared, reason,
        )
        # Barrier across all active ranks to verify RDMA control-plane is
        # actually functional after the reconnect. This is a nixl-level
        # collective that requires every peer to reach it -- if the wake
        # path only re-established the metadata but not the underlying
        # QPs / registered-memory bindings, this call will hang or fail
        # before dispatch/combine ever gets used, giving us a clear
        # signal instead of "64 experts all timeout on first inference".
        try:
            with NixlEPAll2AllManager._lock:
                state = NixlEPAll2AllManager._buffer
            if state is not None:
                logger.info(
                    "NIXL EP rank %d entering post-reconnect barrier (reason=%s)",
                    self.rank, reason,
                )
                state.buffer.barrier()
                logger.info(
                    "NIXL EP rank %d post-reconnect barrier OK (reason=%s)",
                    self.rank, reason,
                )
        except Exception:
            logger.warning(
                "NIXL EP rank %d post-reconnect barrier FAILED (reason=%s); "
                "dispatch will likely timeout",
                self.rank, reason, exc_info=True,
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
        state.active_ep_size = target_ep_size

    def _stage_ep_size(self) -> None:
        assert NixlEPAll2AllManager._buffer is not None
        state = NixlEPAll2AllManager._buffer
        target_ep_size = self.world_size

        # Scale-up can safely connect standby ranks while leaving them masked.
        # Scale-down must not disconnect active ranks until commit.
        if target_ep_size > state.connected_ep_size:
            self._connect_to_ep_size(target_ep_size, make_active=False)

    def commit_staged_state(self) -> None:
        """Commit staged NIXL EP state to the active communication set."""
        with NixlEPAll2AllManager._lock:
            assert NixlEPAll2AllManager._buffer is not None
            state = NixlEPAll2AllManager._buffer
            target_ep_size = self.world_size

            if target_ep_size < state.connected_ep_size:
                self._disconnect_to_ep_size(target_ep_size)
            elif target_ep_size > state.connected_ep_size:
                self._connect_to_ep_size(target_ep_size, make_active=True)

            self._unmask_connected_ranks(target_ep_size)

    def _ensure_ep_size(self, *, stage: bool) -> None:
        if stage:
            self._stage_ep_size()
        else:
            self.commit_staged_state()

    def get_handle(self, kwargs):
        with NixlEPAll2AllManager._lock:
            stage = bool(kwargs.get("stage", False))
            state = NixlEPAll2AllManager._buffer
            if state is None:
                assert not stage, (
                    "NIXL EP staged initialization requires an existing buffer"
                )
                max_num_tokens_per_dp_rank = kwargs["max_num_tokens_per_dp_rank"]
                num_experts_per_rank = (
                    kwargs["num_global_experts"] // kwargs["num_ep_ranks"]
                )
                self._init_buffer(
                    max_num_tokens_per_dp_rank=max_num_tokens_per_dp_rank,
                    token_hidden_size=kwargs["token_hidden_size"],
                    num_experts_per_rank=num_experts_per_rank,
                )
            else:
                self._ensure_ep_size(stage=stage)

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


class FlashInferNVLinkTwoSidedManager(All2AllManagerBase):
    """
    All2All communication based on flashinfer all2allv/two-sided NVLink kernels.
    """

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
        """Ensure workspace is initialized"""
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


class FlashInferNVLinkOneSidedManager(All2AllManagerBase):
    """
    All2All communication based on FlashInfer's MoeAlltoAll/One-sided NVLink kernel.
    This is a newer kernel from trtllm that should perform better than the kernel
    used by flashinfer_nvlink_two_sided.
    """

    rank: int
    world_size: int

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

    def initialize(
        self,
        max_num_tokens: int,
        top_k: int,
        num_experts: int,
        hidden_size: int,
        dispatch_dtype_bytes_per_elem: int = 0,
        dispatch_scale_bytes_per_token: int = 0,
    ):
        """Initialize (or grow) the MoeAlltoAll workspace."""
        if dispatch_dtype_bytes_per_elem == 0:
            hidden_bytes = hidden_size // 2
        else:
            hidden_bytes = hidden_size * dispatch_dtype_bytes_per_elem
        total_dispatch_payload_size_per_token = (
            hidden_bytes
            + dispatch_scale_bytes_per_token
            + top_k * 4  # int32 topks ids
            + top_k * 4  # float32 topk weights
        )
        combine_payload_size_per_token = hidden_size * 2  # bf16 hidden states
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
        gpus_per_node = torch.accelerator.device_count()
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

        self.moe_alltoall = MoeAlltoAll(
            mapping=self.mapping,
            max_num_tokens=self.max_num_tokens,
            top_k=self.top_k,
            num_experts=self.num_experts,
            workspace_size_per_rank=self.workspace_size,
            mnnvl_config=ep_config,
        )

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
    """
    All2All communication based on DeepEP v2 ElasticBuffer (unified API).
    Uses NCCL Gin backend with analytical SM calculation.
    """

    def __init__(self, cpu_group, tcp_store_group=None, device_group=None):
        assert has_deep_ep_v2(), (
            "DeepEP v2 (ElasticBuffer) not available. Requires DeepEP >= 2.0 "
            "(https://github.com/deepseek-ai/DeepEP) and NCCL >= 2.30.4."
        )
        super().__init__(cpu_group, tcp_store_group)
        self._device_group = device_group
        self.handle_cache = Cache()
        self._num_sms: int | None = None

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

    def get_handle(self, kwargs):
        import deep_ep  # type: ignore[import-not-found]

        num_experts = kwargs.pop("num_experts", 256)
        buffer_kwargs = self._make_all2all_kwargs(**kwargs)
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
