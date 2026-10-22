# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""EPLB communicator implementations and factory."""

import contextlib
import os
import time
import uuid
from abc import ABC, abstractmethod
from collections.abc import Sequence
from datetime import timedelta
from typing import Any

import numpy as np
import torch
from torch.distributed import (
    P2POp,
    ProcessGroup,
    batch_isend_irecv,
)

import vllm.distributed.nixl_utils as nixl_utils
from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
from vllm.distributed.device_communicators.pynccl_wrapper import (
    ncclDataTypeEnum,
)
from vllm.distributed.eplb.eplb_utils import device_stream
from vllm.distributed.parallel_state import (
    GroupCoordinator,
    get_pp_group,
    is_local_first_rank,
)
from vllm.distributed.stateless_coordinator import StatelessGroupCoordinator
from vllm.distributed.utils import is_weak_contiguous
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils.gpu_sync_debug import gpu_sync_allowed
from vllm.utils.torch_utils import PIN_MEMORY

logger = init_logger(__name__)


def has_nixl() -> bool:
    """Whether the optional NIXL package is available."""
    return nixl_utils.NixlWrapper is not None


class _ArenaAgent:
    """The one NIXL agent in this process that has an expert arena registered.

    A registration belongs to an agent, so reusing the registration means
    reusing the agent. The expert arena is fixed for the life of the worker,
    since a reconfiguration only re-points views at the same preallocated
    pages, so the agent and its registration are too.

    Building a second agent over the same arena is what used to break a scale
    up: two agents held the identical pages, and whichever was collected first
    deregistered memory the other was still reading from, which surfaced as a
    fault in the middle of the layer 0 expert reads.

    Peers are keyed by their metadata rather than removed and re-added, because
    the agent name is now stable for the life of the worker and a
    reconfiguration gathers the very same metadata back.
    """

    def __init__(self, agent: Any) -> None:
        self.agent = agent
        self.registered_descs: list[object] = []
        # Address ranges this agent has registered, so a reconfiguration can
        # tell what is already covered. The expert weights always are; the
        # receive buffers are checked rather than assumed.
        self.registered_ranges: list[tuple[int, int]] = []
        self.remote_names: dict[bytes, str] = {}
        self.users = 0

    def covers(self, tensor: torch.Tensor) -> bool:
        lo = tensor.data_ptr()
        hi = lo + tensor.numel() * tensor.element_size()
        return any(lo >= r_lo and hi <= r_hi for r_lo, r_hi in self.registered_ranges)


# (device, arena start, arena end) -> the agent that owns that registration.
_ARENA_AGENTS: dict[tuple[int, int, int], _ArenaAgent] = {}


class EplbCommunicator(ABC):
    """Abstract EPLB communicator for expert weight transfers."""

    @abstractmethod
    def add_send(
        self,
        tensors: list[torch.Tensor],
        dst_rank: int,
        expert_id: int,
    ) -> None:
        pass

    @abstractmethod
    def add_recv(
        self,
        tensors: list[torch.Tensor],
        src_rank: int,
        expert_id: int,
    ) -> None:
        pass

    @abstractmethod
    def execute(self) -> None:
        """Complete all enqueued transfers.

        Some backends perform communication here; others (e.g. NIXL)
        issue transfers eagerly in add_recv and only wait here.
        On return, all data is available in the destination buffers.
        """

    def set_transfer_context(  # noqa: B027
        self, old_indices: np.ndarray, layer_idx: int
    ) -> None:
        """Pre-set layer context before add_recv calls.

        Default is a no-op; overridden by backends (e.g. NIXL) that need
        layer-level context to issue transfers inside add_recv.
        """

    @property
    def needs_profile_buffer_reservation(self) -> bool:
        """Whether the profile path must run a dummy collective operation to reserve
        communication buffers."""
        return True

    def set_stream(self, stream: torch.Stream | None) -> None:
        self._stream = stream

    def _log_initialized(self) -> None:
        if is_local_first_rank():
            logger.info("Initialized EPLB communicator: %s.", self.__class__.__name__)


class TorchDistNcclEplbCommunicator(EplbCommunicator):
    """EPLB communicator backed by torch.distributed isend/irecv."""

    def __init__(
        self,
        ep_group: ProcessGroup,
        stream: torch.Stream | None = None,
    ) -> None:
        self._ep_group = ep_group
        self._stream = stream
        self._p2p_ops: list[P2POp] = []
        self._log_initialized()

    def add_send(
        self,
        tensors: list[torch.Tensor],
        dst_rank: int,
        expert_id: int,  # unused by this backend
    ) -> None:
        for tensor in tensors:
            self._p2p_ops.append(
                P2POp(
                    torch.distributed.isend,
                    tensor,
                    group=self._ep_group,
                    group_peer=dst_rank,
                )
            )

    def add_recv(
        self,
        tensors: list[torch.Tensor],
        src_rank: int,
        expert_id: int,  # unused by this backend
    ) -> None:
        for tensor in tensors:
            self._p2p_ops.append(
                P2POp(
                    torch.distributed.irecv,
                    tensor,
                    group=self._ep_group,
                    group_peer=src_rank,
                )
            )

    def execute(self) -> None:
        if not self._p2p_ops:
            return
        try:
            with device_stream(self._stream):
                reqs = batch_isend_irecv(self._p2p_ops)
                for req in reqs:
                    req.wait()
        finally:
            self._p2p_ops.clear()


class TorchDistGlooStagedEplbCommunicator(EplbCommunicator):
    """EPLB communicator using gloo P2P with CPU staging."""

    def __init__(
        self,
        cpu_group: ProcessGroup,
        stream: torch.Stream | None = None,
    ) -> None:
        self._cpu_group = cpu_group
        self._stream = stream
        self._ops: list[tuple[str, torch.Tensor, int]] = []
        self._log_initialized()

    def add_send(
        self,
        tensors: list[torch.Tensor],
        dst_rank: int,
        expert_id: int,  # unused by this backend
    ) -> None:
        for tensor in tensors:
            self._ops.append(("send", tensor, dst_rank))

    def add_recv(
        self,
        tensors: list[torch.Tensor],
        src_rank: int,
        expert_id: int,  # unused by this backend
    ) -> None:
        for tensor in tensors:
            self._ops.append(("recv", tensor, src_rank))

    def execute(self) -> None:
        if not self._ops:
            return

        p2p_ops: list[P2POp] = []
        recv_staging: list[tuple[torch.Tensor, torch.Tensor]] = []

        def build_ops() -> None:
            for op, tensor, peer_rank in self._ops:
                if op == "send":
                    cpu_tensor = tensor.to(device="cpu", non_blocking=True)
                    p2p_ops.append(
                        P2POp(
                            torch.distributed.isend,
                            cpu_tensor,
                            group=self._cpu_group,
                            group_peer=peer_rank,
                        )
                    )
                    continue
                cpu_tensor = torch.empty_like(
                    tensor, device="cpu", pin_memory=PIN_MEMORY
                )
                p2p_ops.append(
                    P2POp(
                        torch.distributed.irecv,
                        cpu_tensor,
                        group=self._cpu_group,
                        group_peer=peer_rank,
                    )
                )
                recv_staging.append((tensor, cpu_tensor))

        try:
            with device_stream(self._stream):
                build_ops()
        finally:
            self._ops.clear()

        # Wait for all D2H copies to finish
        # before issuing gloo batch_isend_irecv operations.
        with gpu_sync_allowed():
            if self._stream is not None:
                self._stream.synchronize()
            else:
                torch.accelerator.current_stream().synchronize()

        reqs = batch_isend_irecv(p2p_ops)
        for req in reqs:
            req.wait()

        if not recv_staging:
            return
        with device_stream(self._stream):
            for dst_tensor, cpu_tensor in recv_staging:
                dst_tensor.copy_(cpu_tensor, non_blocking=True)


class TorchDistXCCLStagedEplbCommunicator(EplbCommunicator):
    """EPLB communicator using XCCL device-to-device P2P on XPU."""

    def __init__(
        self,
        ep_group: ProcessGroup,
        stream: torch.Stream | None = None,
    ) -> None:
        self._ep_group = ep_group
        self._stream = stream
        self._ops: list[tuple[str, torch.Tensor, int]] = []
        self._log_initialized()

    def add_send(
        self,
        tensors: list[torch.Tensor],
        dst_rank: int,
        expert_id: int,  # unused by this backend
    ) -> None:
        for tensor in tensors:
            self._ops.append(("send", tensor, dst_rank))

    def add_recv(
        self,
        tensors: list[torch.Tensor],
        src_rank: int,
        expert_id: int,  # unused by this backend
    ) -> None:
        for tensor in tensors:
            self._ops.append(("recv", tensor, src_rank))

    def execute(self) -> None:
        if not self._ops:
            return

        p2p_ops: list[P2POp] = []
        try:
            for op, tensor, peer_rank in self._ops:
                send_or_recv = (
                    torch.distributed.isend if op == "send" else torch.distributed.irecv
                )
                p2p_ops.append(
                    P2POp(
                        send_or_recv,
                        tensor,
                        peer_rank,
                        self._ep_group,
                    )
                )
        finally:
            self._ops.clear()

        with device_stream(self._stream):
            reqs = batch_isend_irecv(p2p_ops)
            for req in reqs:
                req.wait()


class NixlEplbCommunicator(EplbCommunicator):
    """EPLB communicator backed by NIXL READ transfers."""

    def __init__(
        self,
        cpu_group: ProcessGroup,
        all_expert_weights: Sequence[Sequence[torch.Tensor]],
        expert_buffer: Sequence[torch.Tensor],
    ) -> None:
        """Create a NIXL-backed EPLB communicator.

        Args:
            cpu_group: CPU process group for metadata exchange.
            all_expert_weights: Expert weight tensors for all MoE layers.
            expert_buffer: Pre-allocated receive buffer tensors.

        """
        assert all_expert_weights, (
            "NixlEplbCommunicator requires non-empty all_expert_weights."
        )
        assert expert_buffer, "NixlEplbCommunicator requires non-empty expert_buffer."
        nixl_wrapper_cls = nixl_utils.NixlWrapper
        if nixl_wrapper_cls is None:
            raise RuntimeError("NIXL is unavailable.")

        self._cpu_group = cpu_group
        self._world_size = cpu_group.size()
        self._rank = cpu_group.rank()

        self._all_expert_weights = all_expert_weights
        self._expert_buffer = expert_buffer
        self._num_local_experts: int = all_expert_weights[0][0].shape[0]
        self._device = all_expert_weights[0][0].device

        for layer_tensors in all_expert_weights:
            for tensor in layer_tensors:
                assert is_weak_contiguous(tensor), (
                    "Expert weight tensors must be contiguous in memory"
                )
                assert tensor.device == self._device, (
                    "All local EPLB tensors are expected to be on the same "
                    f"device: expected={self._device}, got={tensor.device}"
                )
        for tensor in expert_buffer:
            assert is_weak_contiguous(tensor), (
                "expert_buffer tensors must be contiguous in memory"
            )

        # (local_dlist, remote_dlist, xfer_handle) for in-flight READs;
        # accumulated by add_recv, drained by execute.
        self._xfer_entries: list[tuple[int, int, int]] = []
        # Per-rank expert_id -> physical row; set by set_transfer_context.
        self._expert_to_src_row: list[dict[int, int]] | None = None
        self._layer_idx: int | None = None
        self.set_stream(None)

        nixl_agent_config = nixl_utils.nixl_agent_config
        config = (
            nixl_agent_config(capture_telemetry=False)
            if nixl_agent_config is not None
            else None
        )
        self._cuda_device_id = int(self._device.index or 0)
        # One agent per expert arena per process, see `_ArenaAgent`. A
        # reconfiguration lands on the arena that is already registered, so it
        # reuses that agent instead of building a second one over the same
        # pages.
        self._arena_key = self._expert_arena_key()
        arena_agent = _ARENA_AGENTS.get(self._arena_key)
        if arena_agent is None:
            arena_agent = _ArenaAgent(nixl_wrapper_cls(self._make_agent_name(), config))
            _ARENA_AGENTS[self._arena_key] = arena_agent
        arena_agent.users += 1
        self._arena_agent = arena_agent
        self._nixl_wrapper = arena_agent.agent
        self._nixl_memory_type = "VRAM"
        self._remote_agents: dict[int, str] = {}
        # peer -> (layer, tensor) -> (base_ptr, bytes_per_expert, dev_id).
        self._remote_send_meta: dict[
            int, dict[tuple[int, int], tuple[int, int, int]]
        ] = {}

        self._init_step("buffers", self._init_registered_buffers)
        self._init_remote_state()
        self._log_initialized()

    def _init_remote_state(self) -> None:
        """Exchange NIXL agent metadata and RDMA pointer info with all peers.

        This is a collective operation (uses ``all_gather_object`` twice).
        """
        self._init_step("agents", self._init_remote_agents)
        self._init_step("send meta", self._exchange_remote_send_meta)

    @property
    def needs_profile_buffer_reservation(self) -> bool:
        return False

    @staticmethod
    def _init_step(name: str, fn: object, *args: object, **kwargs: object) -> None:
        try:
            fn(*args, **kwargs)  # type: ignore[operator]
        except Exception as exc:
            raise RuntimeError(f"NIXL EPLB init failed: {name}") from exc

    def _make_agent_name(self) -> str:
        """Build a deployment-unique nixl agent name."""
        pp_size = get_pp_group().world_size
        pp_suffix = f"-pp{get_pp_group().rank_in_group}" if pp_size > 1 else ""
        uid = uuid.uuid4().hex[:8]
        return f"eplb-{self._rank}{pp_suffix}-{uid}"

    def set_stream(self, stream: torch.Stream | None) -> None:
        self._stream = stream

    def add_send(
        self,
        tensors: list[torch.Tensor],
        dst_rank: int,
        expert_id: int,
    ) -> None:
        # No-op: NIXL READ is receiver-initiated. The sender's expert
        # weights are pre-registered and always readable in-place.
        pass

    def set_transfer_context(self, old_indices: np.ndarray, layer_idx: int) -> None:
        assert not self._xfer_entries, (
            f"set_transfer_context() called with {len(self._xfer_entries)} "
            f"pending transfers from layer {self._layer_idx}; "
            f"execute() was not called after previous add_recv() calls"
        )
        # NIXL READs bypass device streams; wait for pending copies out of
        # the receive buffer before add_recv overwrites it.
        if current_platform.is_rocm():
            with gpu_sync_allowed():
                torch.accelerator.current_stream().synchronize()
                if self._stream is not None:
                    self._stream.synchronize()
        self._layer_idx = layer_idx
        n = self._num_local_experts
        rank_experts = old_indices[: self._world_size * n].reshape(self._world_size, n)
        self._expert_to_src_row = [
            {int(eid): i for i, eid in enumerate(row) if eid != -1}
            for row in rank_experts
        ]

    def add_recv(
        self,
        tensors: list[torch.Tensor],
        src_rank: int,
        expert_id: int,
    ) -> None:
        # Build NIXL descriptors and issue the RDMA READ immediately,
        # overlapping the transfer with the remaining Python loop in
        # move_to_buffer.
        assert self._expert_to_src_row is not None and self._layer_idx is not None, (
            "set_transfer_context() must be called before add_recv()"
        )
        src_row = self._expert_to_src_row[src_rank][expert_id]
        layer_idx = self._layer_idx

        local_descs: list[tuple[int, int, int]] = []
        remote_descs: list[tuple[int, int, int]] = []
        for t_idx, t in enumerate(tensors):
            send_base, send_stride, remote_dev = self._remote_send_meta[src_rank][
                (layer_idx, t_idx)
            ]
            assert t.nbytes == send_stride, (
                f"tensor {t_idx} size {t.nbytes} != remote stride {send_stride}"
            )
            local_descs.append(
                (
                    t.data_ptr(),
                    t.nbytes,
                    self._cuda_device_id,
                )
            )
            remote_descs.append(
                (
                    send_base + src_row * send_stride,
                    send_stride,
                    remote_dev,
                )
            )

        local_h, remote_h, xfer_h = self._create_peer_xfer(
            src_rank, local_descs, remote_descs
        )
        self._nixl_wrapper.transfer(xfer_h)
        self._xfer_entries.append((local_h, remote_h, xfer_h))

    def _init_remote_agents(self) -> None:
        local_metadata = self._nixl_wrapper.get_agent_metadata()
        gathered_metadata: list[bytes | None] = [None] * self._world_size
        torch.distributed.all_gather_object(
            gathered_metadata, local_metadata, group=self._cpu_group
        )
        for peer in range(self._world_size):
            if peer == self._rank:
                continue
            peer_metadata = gathered_metadata[peer]
            assert peer_metadata is not None
            # The peer agent is now stable for the life of its worker, so a
            # reconfiguration gathers metadata this agent already knows. Adding
            # it twice would mean two entries over one peer, and removing
            # either would pull the rkeys out from under the live one.
            known = self._arena_agent.remote_names.get(peer_metadata)
            if known is None:
                known = self._nixl_wrapper.add_remote_agent(peer_metadata)
                self._arena_agent.remote_names[peer_metadata] = known
            self._remote_agents[peer] = known

    def _expert_arena_key(self) -> tuple[int, int, int]:
        """The device and address span the peers read expert weights from.

        A span rather than the exact tensor list, because a reconfiguration
        re-points the views and only has to land on the same arena, not on the
        same tensor objects.
        """
        lo = min(
            tensor.data_ptr() for layer in self._all_expert_weights for tensor in layer
        )
        hi = max(
            tensor.data_ptr() + tensor.numel() * tensor.element_size()
            for layer in self._all_expert_weights
            for tensor in layer
        )
        return (self._cuda_device_id, lo, hi)

    def _init_registered_buffers(self) -> None:
        all_tensors: list[torch.Tensor] = []
        for layer_tensors in self._all_expert_weights:
            all_tensors.extend(layer_tensors)
        all_tensors.extend(self._expert_buffer)

        # Register only what this agent does not already hold. On a first build
        # that is everything; on a reconfiguration it is normally nothing, and
        # at most the receive buffers if they were reallocated.
        pending = [t for t in all_tensors if not self._arena_agent.covers(t)]
        if not pending:
            return

        descs = self._nixl_wrapper.get_reg_descs(pending)
        self._nixl_wrapper.register_memory(descs)
        self._arena_agent.registered_descs.append(descs)
        for tensor in pending:
            lo = tensor.data_ptr()
            self._arena_agent.registered_ranges.append(
                (lo, lo + tensor.numel() * tensor.element_size())
            )

    def _exchange_remote_send_meta(self) -> None:
        """Exchange per-layer per-tensor metadata so receivers can compute
        remote RDMA addresses at transfer time."""
        local_meta: dict[tuple[int, int], tuple[int, int, int]] = {}
        for layer_idx, layer_tensors in enumerate(self._all_expert_weights):
            for t_idx, t in enumerate(layer_tensors):
                nbytes_per_expert = t.nbytes // self._num_local_experts
                local_meta[(layer_idx, t_idx)] = (
                    t.data_ptr(),
                    nbytes_per_expert,
                    self._cuda_device_id,
                )

        # Per-rank map: (layer_idx, tensor_idx) -> (base_ptr, bytes_per_expert, dev_id).
        # add_recv uses base_ptr + src_row * bytes_per_expert to compute
        # the remote RDMA address for each expert.
        gathered_meta: list[dict[tuple[int, int], tuple[int, int, int]] | None] = [
            None
        ] * self._world_size
        torch.distributed.all_gather_object(
            gathered_meta, local_meta, group=self._cpu_group
        )

        local_keys = set(local_meta.keys())
        for peer in self._remote_agents:
            peer_meta = gathered_meta[peer]
            assert peer_meta is not None
            peer_keys = set(peer_meta.keys())
            if peer_keys != local_keys:
                raise RuntimeError(
                    f"NIXL EPLB metadata key mismatch with rank {peer}: "
                    f"local={sorted(local_keys)}, peer={sorted(peer_keys)}"
                )
            for key in local_keys:
                _, local_stride, _ = local_meta[key]
                _, peer_stride, _ = peer_meta[key]
                if local_stride != peer_stride:
                    raise RuntimeError(
                        f"NIXL EPLB nbytes_per_expert mismatch for {key} "
                        f"with rank {peer}: "
                        f"local={local_stride}, peer={peer_stride}"
                    )
            self._remote_send_meta[peer] = peer_meta

    def _wait_for_all_transfers(self, handles: list[int]) -> None:
        """Wait for every READ issued for this layer to report DONE.

        The wait is bounded. A transfer only leaves this loop by reaching DONE
        or by reporting a state that is neither DONE nor PROC, and a peer whose
        GPU faults never does either: its handle stays in PROC for good, so the
        survivors spin here until the test itself is killed. On MI355 two ranks
        faulted with `Memory access fault by GPU node-N` during the expert
        reshuffle and the rest of the ranks sat in this loop for the whole
        780 s budget without printing a thing.
        """
        timeout = float(os.getenv("VLLM_EPLB_NIXL_XFER_TIMEOUT_S") or 300.0)
        started = time.perf_counter()
        total = len(handles)
        pending = set(handles)
        while pending:
            completed: list[int] = []
            for handle in pending:
                state = self._nixl_wrapper.check_xfer_state(handle)
                if state == "DONE":
                    completed.append(handle)
                    continue
                if state != "PROC":
                    raise RuntimeError(f"NIXL transfer failed with state={state}")
            for handle in completed:
                pending.remove(handle)
            if pending:
                elapsed = time.perf_counter() - started
                if timeout > 0 and elapsed > timeout:
                    raise RuntimeError(
                        f"NIXL EPLB transfers for layer {self._layer_idx} did "
                        f"not finish within {timeout:.0f} s: "
                        f"{len(pending)} of {total} still in PROC. A peer whose "
                        f"GPU faulted leaves its READ in PROC forever, so this "
                        f"rank cannot tell a slow transfer from a dead peer."
                    )
                time.sleep(0.0005)

    def _create_peer_xfer(
        self,
        src: int,
        local_descs: list[tuple[int, int, int]],
        remote_descs: list[tuple[int, int, int]],
    ) -> tuple[int, int, int]:
        """Create a batched xfer for multiple descriptors from one peer.

        Each element in *local_descs* / *remote_descs* is an
        ``(address, size, device_id)`` tuple.

        Returns ``(local_dlist, remote_dlist, xfer_handle)``.
        """
        local_desc = self._nixl_wrapper.get_xfer_descs(
            local_descs, self._nixl_memory_type
        )
        local_handle = self._nixl_wrapper.prep_xfer_dlist(
            "NIXL_INIT_AGENT",
            local_desc,
        )

        remote_desc = self._nixl_wrapper.get_xfer_descs(
            remote_descs, self._nixl_memory_type
        )
        remote_handle = self._nixl_wrapper.prep_xfer_dlist(
            self._remote_agents[src],
            remote_desc,
        )

        indices = np.arange(len(local_descs), dtype=np.int32)
        xfer_handle = self._nixl_wrapper.make_prepped_xfer(
            "READ",
            local_handle,
            indices,
            remote_handle,
            indices,
        )
        return (local_handle, remote_handle, xfer_handle)

    def _post_read_barrier(self) -> None:
        """Correctness fence: prevents overwrite-while-remote-read race.

        We avoid ``torch.distributed.monitored_barrier`` because it
        calls ``get_backend(group)`` which fails for stateless groups
        (elastic EP).  An async ``all_reduce`` + ``wait(timeout)``
        works with both regular and stateless groups and provides
        equivalent timeout detection.
        """
        _dummy = torch.zeros(1, dtype=torch.int32)
        work = torch.distributed.all_reduce(
            _dummy, group=self._cpu_group, async_op=True
        )
        work.wait(timeout=timedelta(minutes=5))

    def execute(self) -> None:
        assert self._layer_idx is not None or not self._xfer_entries, (
            "set_transfer_context() must be called before execute() "
            "if any add_recv() calls were made"
        )
        try:
            self._wait_for_all_transfers([x[2] for x in self._xfer_entries])
            self._post_read_barrier()
        finally:
            for local_h, remote_h, xfer_h in self._xfer_entries:
                with contextlib.suppress(Exception):
                    self._nixl_wrapper.release_xfer_handle(xfer_h)
                with contextlib.suppress(Exception):
                    self._nixl_wrapper.release_dlist_handle(local_h)
                with contextlib.suppress(Exception):
                    self._nixl_wrapper.release_dlist_handle(remote_h)
            self._xfer_entries.clear()
            self._expert_to_src_row = None
            self._layer_idx = None

    def __del__(self) -> None:
        # At interpreter shutdown the module globals this body needs, down to
        # `contextlib` itself, may already be None, and every worker exit then
        # prints a traceback per communicator. Nothing here is worth that, so
        # the whole teardown is guarded without touching a module global.
        try:  # noqa: SIM105 `contextlib` is exactly what may be gone here
            self._teardown()
        except BaseException:  # noqa: BLE001
            pass

    def _teardown(self) -> None:
        with contextlib.suppress(Exception):
            for local_h, remote_h, xfer_h in self._xfer_entries:
                with contextlib.suppress(Exception):
                    self._nixl_wrapper.release_xfer_handle(xfer_h)
                with contextlib.suppress(Exception):
                    self._nixl_wrapper.release_dlist_handle(local_h)
                with contextlib.suppress(Exception):
                    self._nixl_wrapper.release_dlist_handle(remote_h)
        self._remote_agents.clear()
        with contextlib.suppress(Exception):
            self._release_arena_agent()

    def _release_arena_agent(self) -> None:
        """Give up this communicator's share of the agent that owns the arena.

        A reconfiguration builds the replacement before dropping this object,
        so the count goes to two and back to one and the registration is never
        touched. It reaches zero only when expert parallelism itself is going
        away, and then nothing local or remote is reading any more, so that is
        the one point where deregistering is both safe and necessary.
        """
        agent = self._arena_agent
        agent.users -= 1
        if agent.users > 0:
            return
        _ARENA_AGENTS.pop(self._arena_key, None)
        for name in agent.remote_names.values():
            with contextlib.suppress(Exception):
                agent.agent.remove_remote_agent(name)
        agent.remote_names.clear()
        for descs in agent.registered_descs:
            with contextlib.suppress(Exception):
                agent.agent.deregister_memory(descs)
        agent.registered_descs.clear()
        agent.registered_ranges.clear()


class PyNcclEplbCommunicator(EplbCommunicator):
    """EPLB communicator backed by PyNcclCommunicator using ncclSend/ncclRecv."""

    def __init__(
        self,
        pynccl_comm: PyNcclCommunicator,
        stream: torch.Stream | None = None,
    ) -> None:
        self._pynccl_comm = pynccl_comm
        self._stream = stream
        self._group_started = False
        self._log_initialized()

    def _ensure_group_started(self) -> None:
        if not self._group_started:
            self._pynccl_comm.group_start()
            self._group_started = True

    def add_send(
        self,
        tensors: list[torch.Tensor],
        dst_rank: int,
        expert_id: int,  # unused by this backend
    ) -> None:
        self._ensure_group_started()
        for tensor in tensors:
            self._pynccl_comm.send(tensor, dst_rank, stream=self._stream)

    def add_recv(
        self,
        tensors: list[torch.Tensor],
        src_rank: int,
        expert_id: int,  # unused by this backend
    ) -> None:
        self._ensure_group_started()
        for tensor in tensors:
            self._pynccl_comm.recv(tensor, src_rank, stream=self._stream)

    def execute(self) -> None:
        if self._group_started:
            self._pynccl_comm.group_end()
            self._group_started = False


def create_eplb_communicator(
    group_coordinator: GroupCoordinator,
    backend: str,
    expert_weights: Sequence[Sequence[torch.Tensor]],
    expert_buffer: Sequence[torch.Tensor],
) -> EplbCommunicator:
    """Create an EPLB communicator for the given backend.

    Args:
        group_coordinator: Process-group coordinator that provides the
            device and CPU communication groups.
        backend: Communicator backend name (``"torch_nccl"``,
            ``"torch_gloo"``, ``"pynccl"``, or ``"nixl"``).
            Falls back to ``"torch_nccl"`` when *None*.
            Stateless (elastic EP) groups support ``"torch_nccl"``,
            ``"pynccl"``, and ``"nixl"``; ``"torch_nccl"`` is silently
            promoted to ``"pynccl"``.  When tensors reside on CPU,
            ``"torch_gloo"`` or
            ``"torch_nccl"`` are used via the CPU process group.
        expert_weights: Expert weight tensors for *all* MoE layers.
            Shape ``(num_layers)(num_tensors_per_layer)``.
            NixlEplbCommunicator registers all layers with NIXL for
            zero-copy RDMA reads.
        expert_buffer: Pre-allocated receive buffer tensors (one per
            weight tensor in a single layer).

    """
    first_layer = expert_weights[0] if expert_weights else []
    tensor_device_type = first_layer[0].device.type if first_layer else "cpu"
    torch_group = (
        group_coordinator.cpu_group
        if tensor_device_type == "cpu"
        else group_coordinator.device_group
    )

    def _create_pynccl() -> EplbCommunicator:
        if tensor_device_type == "cpu":
            raise RuntimeError(
                "EPLB communicator 'pynccl' supports only cuda-like devices "
                f"(got {tensor_device_type})."
            )
        unsupported_dtypes = sorted(
            {
                tensor.dtype
                for tensor in first_layer
                if not ncclDataTypeEnum.supports_torch_dtype(tensor.dtype)
            },
            key=str,
        )
        if unsupported_dtypes:
            raise RuntimeError(
                "EPLB communicator 'pynccl' requested but expert weights contain "
                "unsupported dtypes: "
                f"({', '.join(str(dtype) for dtype in unsupported_dtypes)})."
            )

        device_comm = group_coordinator.device_communicator
        pynccl_comm = (
            getattr(device_comm, "pynccl_comm", None)
            if device_comm is not None
            else None
        )
        if pynccl_comm is None or pynccl_comm.disabled or not pynccl_comm.available:
            raise RuntimeError("EPLB communicator 'pynccl' requested but unavailable.")
        try:
            return PyNcclEplbCommunicator(pynccl_comm=pynccl_comm)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to initialize PyNcclEplbCommunicator ({exc})."
            ) from exc

    is_stateless = isinstance(group_coordinator, StatelessGroupCoordinator)
    if is_stateless and backend != "nixl":
        if backend not in ("torch_nccl", "pynccl"):
            raise ValueError(
                f"Elastic EP requires 'torch_nccl', 'pynccl', or 'nixl' "
                f"EPLB communicator (got '{backend}')."
            )
        if backend == "torch_nccl":
            logger.warning(
                "Stateless elastic EP requires PyNCCL backend. "
                "Forcing EPLB communicator to 'pynccl'."
            )
            backend = "pynccl"
        return _create_pynccl()

    if backend == "nixl":
        if not has_nixl():
            raise RuntimeError(
                "EPLB communicator 'nixl' requested but NIXL is unavailable."
            )
        if not (current_platform.is_cuda_alike() and tensor_device_type != "cpu"):
            raise RuntimeError(
                "EPLB communicator 'nixl' supports only cuda-like devices "
                f"(got {tensor_device_type})."
            )
        try:
            return NixlEplbCommunicator(
                cpu_group=group_coordinator.cpu_group,
                all_expert_weights=expert_weights,
                expert_buffer=expert_buffer,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to initialize NixlEplbCommunicator ({exc})."
            ) from exc
    elif backend == "torch_gloo":
        return TorchDistGlooStagedEplbCommunicator(
            cpu_group=group_coordinator.cpu_group,
        )
    elif backend == "torch_xccl":
        return TorchDistXCCLStagedEplbCommunicator(ep_group=torch_group)
    elif backend == "torch_nccl":
        return TorchDistNcclEplbCommunicator(ep_group=torch_group)
    elif backend == "pynccl":
        return _create_pynccl()
    raise ValueError(f"Unknown EPLB communicator backend: {backend}")
