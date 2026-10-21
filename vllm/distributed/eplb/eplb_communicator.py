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


# Expert arena -> how many live NIXL registrations in this process cover it,
# and the agents that must stay alive for those registrations to hold. See
# `_arena_registration_is_last`.
_ARENA_USERS: dict[tuple[int, tuple[tuple[int, int], ...]], int] = {}
_PARKED_AGENTS: list[object] = []


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
        # (layer, peer) -> local and remote address ranges, diagnostic only.
        self._desc_ranges: dict[tuple[int, int], list[int]] = {}
        # (layer, tensor) -> (base_ptr, bytes) as published to the peers.
        self._published_ptrs: dict[tuple[int, int], tuple[int, int]] = {}
        # Device backed address ranges, to place a fault address. Diagnostic.
        self._mappings: dict[int, tuple[int, str]] = {}
        self.set_stream(None)

        nixl_agent_config = nixl_utils.nixl_agent_config
        config = (
            nixl_agent_config(capture_telemetry=False)
            if nixl_agent_config is not None
            else None
        )
        self._nixl_wrapper = nixl_wrapper_cls(self._make_agent_name(), config)
        self._nixl_memory_type = "VRAM"
        # NIXL registration handles; deregistered in __del__.
        self._registered_descs: list[object] = []
        # Identifies the expert arena this agent registered, so a teardown can
        # tell whether another communicator still covers the same memory.
        self._arena_key: tuple[int, tuple[tuple[int, int], ...]] | None = None
        self._remote_agents: dict[int, str] = {}
        # peer -> (layer, tensor) ->
        # (base_ptr, bytes_per_expert, dev_id, total_bytes, num_rows).
        self._remote_send_meta: dict[
            int, dict[tuple[int, int], tuple[int, int, int, int, int]]
        ] = {}

        self._cuda_device_id = int(self._device.index or 0)
        self._init_step("buffers", self._init_registered_buffers)
        self._init_remote_state()
        self._log_initialized()

    def _init_remote_state(self) -> None:
        """Exchange NIXL agent metadata and RDMA pointer info with all peers.

        This is a collective operation (uses ``all_gather_object`` twice).
        """
        self._init_step("agents", self._init_remote_agents)
        self._init_step("send meta", self._exchange_remote_send_meta)
        self._init_step("warm transfers", self._warm_peer_transfers)
        if os.getenv("VLLM_EEP_DEBUG_MAPPINGS") == "1":
            from vllm.distributed.elastic_ep import rocm_debug

            # The baseline, taken before a single transfer is issued, so
            # anything UCX maps for a peer later shows up as new.
            self._mappings = rocm_debug.device_mappings()
            logger.warning(
                "[EEP-DEBUG] %d device mapping(s) before any transfer",
                len(self._mappings),
            )

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
        self._check_published_ptrs(layer_idx)
        n = self._num_local_experts
        rank_experts = old_indices[: self._world_size * n].reshape(self._world_size, n)
        self._expert_to_src_row = [
            {int(eid): i for i, eid in enumerate(row) if eid != -1}
            for row in rank_experts
        ]

    def _check_published_ptrs(self, layer_idx: int) -> None:
        """Confirm this rank's expert weights still sit where it said they do.

        Every peer reads this layer straight out of the addresses handed out
        once, at init. Nothing re-publishes them, so a reallocation of these
        tensors afterwards leaves the peers reading an address this rank no
        longer owns, and a READ served from a freed or reused mapping is a
        fault on this device, not on theirs. The rank that moved its tensors
        is the only one that can see it happen, which is why the check lives on
        the sending side.
        """
        for t_idx, t in enumerate(self._all_expert_weights[layer_idx]):
            published = self._published_ptrs.get((layer_idx, t_idx))
            if published is None:
                continue
            base, nbytes = published
            if t.data_ptr() == base and t.nbytes == nbytes:
                continue
            raise RuntimeError(
                f"NIXL EPLB expert weights moved after their addresses were "
                f"published: layer {layer_idx} tensor {t_idx} was "
                f"{nbytes} byte(s) at 0x{base:x} and is now {t.nbytes} "
                f"byte(s) at 0x{t.data_ptr():x}. Every peer still reads the "
                f"old address, so their READs land on memory this rank has "
                f"released."
            )

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
        # What the peer registered, as opposed to what this layer reads.
        valid_bounds: list[tuple[int, int]] = []
        for t_idx, t in enumerate(tensors):
            (
                send_base,
                send_stride,
                remote_dev,
                send_total,
                send_rows,
            ) = self._remote_send_meta[src_rank][(layer_idx, t_idx)]
            assert t.nbytes == send_stride, (
                f"tensor {t_idx} size {t.nbytes} != remote stride {send_stride}"
            )
            # A READ past the end of the peer's tensor is not refused: the
            # address is still a plausible device address, so the peer's GPU
            # takes the access and faults, which kills that rank with nothing
            # but an address to show for it and leaves every other rank
            # spinning on a transfer that will never report DONE. The bound is
            # known here, before the READ is issued, so check it here.
            if not 0 <= src_row < send_rows:
                raise RuntimeError(
                    f"NIXL EPLB would read outside rank {src_rank}'s expert "
                    f"weights: layer {layer_idx} tensor {t_idx} expert "
                    f"{expert_id} resolved to row {src_row}, but that peer "
                    f"holds {send_rows} row(s). The read would start at "
                    f"0x{send_base + src_row * send_stride:x}, which is "
                    f"{(src_row - send_rows + 1) * send_stride} byte(s) past "
                    f"the end of a {send_total} byte region based at "
                    f"0x{send_base:x}."
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
            valid_bounds.append((send_base, send_base + send_total))

        if os.getenv("VLLM_EEP_DEBUG_XFER_DESCS") == "1":
            self._record_desc_range(
                layer_idx, src_rank, local_descs, remote_descs, valid_bounds
            )

        local_h, remote_h, xfer_h = self._create_peer_xfer(
            src_rank, local_descs, remote_descs
        )
        self._nixl_wrapper.transfer(xfer_h)
        self._xfer_entries.append((local_h, remote_h, xfer_h))

    def _record_desc_range(
        self,
        layer_idx: int,
        src_rank: int,
        local_descs: list[tuple[int, int, int]],
        remote_descs: list[tuple[int, int, int]],
        valid_bounds: list[tuple[int, int]],
    ) -> None:
        """Keep the address span this layer reads from each peer.

        A fault during the reshuffle names the device and the address but
        nothing else, and the two readings it can have need opposite fixes. If
        the address falls inside what the peer registered, the mapping was
        revoked under a descriptor that was still correct, which is a runtime
        question. If it falls outside, send_base plus src_row times send_stride
        walked off the registration, which is arithmetic this file owns. The
        registered bound is recorded next to the span so one line answers that
        without cross-referencing a second one. One span per layer and peer is
        cheap, where a line per expert is thousands of lines per
        reconfiguration.

        Diagnostic only.
        """
        key = (layer_idx, src_rank)
        span = self._desc_ranges.get(key)
        local_lo = min(addr for addr, _, _ in local_descs)
        local_hi = max(addr + size for addr, size, _ in local_descs)
        remote_lo = min(addr for addr, _, _ in remote_descs)
        remote_hi = max(addr + size for addr, size, _ in remote_descs)
        valid_lo = min(lo for lo, _ in valid_bounds)
        valid_hi = max(hi for _, hi in valid_bounds)
        remote_dev = remote_descs[0][2]
        if span is None:
            self._desc_ranges[key] = [
                local_lo,
                local_hi,
                remote_lo,
                remote_hi,
                remote_dev,
                1,
                valid_lo,
                valid_hi,
            ]
            return
        span[0] = min(span[0], local_lo)
        span[1] = max(span[1], local_hi)
        span[2] = min(span[2], remote_lo)
        span[3] = max(span[3], remote_hi)
        span[5] += 1
        span[6] = min(span[6], valid_lo)
        span[7] = max(span[7], valid_hi)

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
            self._remote_agents[peer] = self._nixl_wrapper.add_remote_agent(
                peer_metadata
            )

    def _init_registered_buffers(self) -> None:
        all_tensors: list[torch.Tensor] = []
        for layer_tensors in self._all_expert_weights:
            all_tensors.extend(layer_tensors)
        all_tensors.extend(self._expert_buffer)

        descs = self._nixl_wrapper.get_reg_descs(all_tensors)
        self._nixl_wrapper.register_memory(descs)
        self._registered_descs.append(descs)

        if os.getenv("VLLM_EEP_SHARED_ARENA_REGISTRATION") == "1":
            # The span of the expert weights, which is what peers read from.
            # A span rather than the exact descriptor list, because a
            # reconfiguration re-points the views and only has to land on the
            # same arena, not on the same tensor objects.
            lo = min(
                tensor.data_ptr()
                for layer in self._all_expert_weights
                for tensor in layer
            )
            hi = max(
                tensor.data_ptr() + tensor.numel() * tensor.element_size()
                for layer in self._all_expert_weights
                for tensor in layer
            )
            self._arena_key = (self._cuda_device_id, ((lo, hi),))
            _ARENA_USERS[self._arena_key] = _ARENA_USERS.get(self._arena_key, 0) + 1

    def _arena_registration_is_last(self) -> bool:
        """Whether this teardown may release the registration over the arena.

        A reconfiguration builds the new communicator before the switch, and
        the expert arena never moves, so for a while two NIXL agents in this
        process hold a registration over the very same pages. Whichever is
        collected first would otherwise deregister memory the other is reading
        from, which is the climb failing in the middle of the layer 0 reads.

        Releasing is therefore allowed only for the last registration over the
        arena. The earlier ones keep both their descriptors and their agent,
        which is what `_PARKED_AGENTS` is for: dropping the agent would take
        its registration down with it.
        """
        key = self._arena_key
        if key is None:
            return True
        remaining = _ARENA_USERS.get(key, 1) - 1
        if remaining > 0:
            _ARENA_USERS[key] = remaining
            return False
        _ARENA_USERS.pop(key, None)
        return True

    def _report_allocation_extents(self) -> None:
        """Report how much memory each published tensor really exports.

        An IPC export covers the whole allocation the pointer falls inside,
        not the tensor handed to it. Under the torch caching allocator these
        tensors are slices of large pool segments, so exporting one of them
        hands a peer the whole segment, including unrelated tensors this rank
        keeps writing to while the peer holds it mapped read only. That would
        fault the owning rank, on a write, at an address inside no registered
        region, which is exactly what the logs show and what disappears when
        the caching allocator is switched off. Nothing in the existing output
        says whether the segments are shared, so measure it.

        Diagnostic only.
        """
        if os.getenv("VLLM_EEP_DEBUG_ALLOC_EXTENTS") != "1":
            return
        if not current_platform.is_rocm():
            return
        from vllm.distributed.elastic_ep import rocm_debug

        # The receive buffers are registered alongside the weights and are
        # small enough to be slices of a shared pool segment, so report them
        # under a negative layer index rather than leaving a hole.
        spans = dict(sorted(self._published_ptrs.items()))
        for b_idx, buf in enumerate(self._expert_buffer):
            spans[(-1, b_idx)] = (buf.data_ptr(), buf.nbytes)

        for (layer_idx, t_idx), (ptr, nbytes) in spans.items():
            extent = rocm_debug.allocation_extent(ptr)
            if extent is None:
                logger.warning(
                    "[EEP-DEBUG] layer %s tensor %s at 0x%x: HIP will not "
                    "report an allocation range",
                    layer_idx,
                    t_idx,
                    ptr,
                )
                continue
            base, size = extent
            logger.warning(
                "[EEP-DEBUG] layer %s tensor %s exports allocation "
                "0x%x-0x%x (%d B) for a %d B tensor at 0x%x: offset %d, "
                "%d B before it and %d B after it go out with it",
                layer_idx,
                t_idx,
                base,
                base + size,
                size,
                nbytes,
                ptr,
                ptr - base,
                ptr - base,
                max(base + size - (ptr + nbytes), 0),
            )

    def _exchange_remote_send_meta(self) -> None:
        """Exchange per-layer per-tensor metadata so receivers can compute
        remote RDMA addresses at transfer time."""
        local_meta: dict[tuple[int, int], tuple[int, int, int, int, int]] = {}
        for layer_idx, layer_tensors in enumerate(self._all_expert_weights):
            for t_idx, t in enumerate(layer_tensors):
                nbytes_per_expert = t.nbytes // self._num_local_experts
                self._published_ptrs[(layer_idx, t_idx)] = (t.data_ptr(), t.nbytes)
                local_meta[(layer_idx, t_idx)] = (
                    t.data_ptr(),
                    nbytes_per_expert,
                    self._cuda_device_id,
                    t.nbytes,
                    t.shape[0],
                )
        self._report_allocation_extents()

        # Per-rank map: (layer_idx, tensor_idx) ->
        # (base_ptr, bytes_per_expert, dev_id, total_bytes, num_rows).
        # add_recv uses base_ptr + src_row * bytes_per_expert to compute
        # the remote RDMA address for each expert, and total_bytes to confirm
        # that address is still inside what the peer registered.
        gathered_meta: list[
            dict[tuple[int, int], tuple[int, int, int, int, int]] | None
        ] = [None] * self._world_size
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
                _, local_stride, _, _, local_rows = local_meta[key]
                _, peer_stride, _, _, peer_rows = peer_meta[key]
                if local_stride != peer_stride:
                    raise RuntimeError(
                        f"NIXL EPLB nbytes_per_expert mismatch for {key} "
                        f"with rank {peer}: "
                        f"local={local_stride}, peer={peer_stride}"
                    )
                if local_rows != peer_rows:
                    raise RuntimeError(
                        f"NIXL EPLB expert row count mismatch for {key} "
                        f"with rank {peer}: "
                        f"local={local_rows}, peer={peer_rows}. Rows index the "
                        f"remote read, so a peer holding fewer rows than this "
                        f"rank expects is read past the end of its tensor."
                    )
            self._remote_send_meta[peer] = peer_meta

    def _warm_peer_transfers(self) -> None:
        """Read one small block from every peer before the real resharding.

        Every stall on this platform happens on layer 0 and nowhere else: all
        286 stall reports in a full run name that layer, and no rank ever logs
        a read for layer 1. So what breaks is the first batch of transfers a
        freshly built communicator issues, which is the same shape as the RCCL
        communicators that need a deferred warm up here.

        This issues one READ per peer, of one expert's worth of bytes into the
        scratch buffer, and waits for it. If first use is the problem, the
        warm up takes the hit on a single descriptor that no result depends on,
        and the resharding behind it starts on a connection that has already
        carried traffic.

        A failure here is logged and swallowed. The warm up exists to make the
        real transfers work, so it must not be the thing that breaks them, and
        it reads into scratch so a half finished read costs nothing.
        """
        if os.getenv("VLLM_EEP_WARM_NIXL_TRANSFERS") != "1":
            return
        if not self._remote_agents or not self._expert_buffer:
            return

        timeout = float(os.getenv("VLLM_EEP_WARM_NIXL_TIMEOUT_S") or 60.0)
        dst = self._expert_buffer[0]
        started = time.perf_counter()
        entries: list[tuple[int, tuple[int, int, int]]] = []

        for peer in sorted(self._remote_agents):
            base, stride, dev, _, _ = self._remote_send_meta[peer][(0, 0)]
            size = min(stride, dst.nbytes)
            try:
                handles = self._create_peer_xfer(
                    peer,
                    [(dst.data_ptr(), size, self._cuda_device_id)],
                    [(base, size, dev)],
                )
                self._nixl_wrapper.transfer(handles[2])
            except Exception as exc:
                logger.warning(
                    "[EEP-DEBUG] warm up READ from rank %s could not be issued: %s: %s",
                    peer,
                    type(exc).__name__,
                    exc,
                )
                continue
            entries.append((peer, handles))

        pending = {handles[2]: peer for peer, handles in entries}
        while pending:
            for handle, peer in list(pending.items()):
                state = self._nixl_wrapper.check_xfer_state(handle)
                if state == "DONE":
                    del pending[handle]
                elif state != "PROC":
                    logger.warning(
                        "[EEP-DEBUG] warm up READ from rank %s reported state=%s",
                        peer,
                        state,
                    )
                    del pending[handle]
            if pending and time.perf_counter() - started > timeout:
                logger.warning(
                    "[EEP-DEBUG] warm up READ still in PROC after %.0f s for "
                    "rank(s) %s, going ahead anyway",
                    timeout,
                    sorted(set(pending.values())),
                )
                break
            if pending:
                time.sleep(0.0005)

        for _, handles in entries:
            local_h, remote_h, xfer_h = handles
            with contextlib.suppress(Exception):
                self._nixl_wrapper.release_xfer_handle(xfer_h)
            with contextlib.suppress(Exception):
                self._nixl_wrapper.release_dlist_handle(local_h)
            with contextlib.suppress(Exception):
                self._nixl_wrapper.release_dlist_handle(remote_h)

        logger.warning(
            "[EEP-DEBUG] warmed %d peer transfer(s) in %.3f s",
            len(entries),
            time.perf_counter() - started,
        )

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
        report_every = float(os.getenv("VLLM_EPLB_NIXL_XFER_REPORT_S") or 0)
        next_report = report_every
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
                if report_every > 0 and elapsed > next_report:
                    # A reshuffle that normally costs under 5 s was measured at
                    # 260 s with every rank sitting in this loop, and a single
                    # duration at the end cannot say whether one layer stalled
                    # or every layer crawled.
                    next_report += report_every
                    logger.warning(
                        "[EEP-DEBUG] layer %s still waiting after %.0f s: "
                        "%d of %d transfer(s) in PROC",
                        self._layer_idx,
                        elapsed,
                        len(pending),
                        total,
                    )
                    if os.getenv("VLLM_EEP_DEBUG_MAPPINGS") == "1":
                        # The snapshot taken before this wait misses anything
                        # UCX maps while it is performing the read, and that is
                        # the window the fault falls in: not one of the six
                        # fault addresses of the last run was inside any range
                        # known at issue time. Re-reading here puts those
                        # mappings on the record while the stall is live.
                        from vllm.distributed.elastic_ep import rocm_debug

                        self._mappings = rocm_debug.log_new_device_mappings(
                            f"layer {self._layer_idx} stalled", self._mappings
                        )
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
        trace = os.getenv("VLLM_EEP_DEBUG_TRANSFER_TRACE") == "1"
        if trace:
            # Diagnostic only. A rank dies somewhere in here with no Python
            # frames at all, and the survivors only report
            # NIXL_ERR_REMOTE_DISCONNECT, which is the victim's view. The last
            # line the dying rank printed is the only thing that says how far
            # it got, and whether it was the wait or the barrier.
            logger.warning(
                "[EEP-DEBUG] layer %s: waiting for %d expert transfer(s)",
                self._layer_idx,
                len(self._xfer_entries),
            )
        # Before the wait, never after: the fault ends the process.
        for (layer, peer), span in self._desc_ranges.items():
            (
                local_lo,
                local_hi,
                remote_lo,
                remote_hi,
                remote_dev,
                reads,
                valid_lo,
                valid_hi,
            ) = span
            logger.warning(
                "[EEP-DEBUG] layer %s reads %d expert(s) from rank %s on "
                "device %s: remote 0x%x-0x%x of registered 0x%x-0x%x, "
                "local 0x%x-0x%x",
                layer,
                reads,
                peer,
                remote_dev,
                remote_lo,
                remote_hi,
                valid_lo,
                valid_hi,
                local_lo,
                local_hi,
            )
        if os.getenv("VLLM_EEP_DEBUG_MAPPINGS") == "1":
            from vllm.distributed.elastic_ep import rocm_debug

            self._mappings = rocm_debug.log_new_device_mappings(
                f"layer {self._layer_idx}", self._mappings
            )
        try:
            self._wait_for_all_transfers([x[2] for x in self._xfer_entries])
            if trace:
                logger.warning(
                    "[EEP-DEBUG] layer %s: transfers done, entering barrier",
                    self._layer_idx,
                )
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
            self._desc_ranges.clear()
            self._expert_to_src_row = None
            self._layer_idx = None

    def _report_teardown(self) -> None:
        """Say when this communicator is collected, and what it deregisters.

        A reconfiguration builds the new communicator before the switch and
        registers the same expert weight tensors the old one still has
        registered, so two NIXL registrations cover the same memory until the
        old object is collected. `__del__` then deregisters that memory, and
        nothing decides when: a stray reference delays it, which can put the
        deregistration inside the window where peers are reading. Timestamps
        are the only way to tell whether that is what happens, since the fault
        and the teardown are in different processes.

        Diagnostic only.
        """
        # At interpreter shutdown the module globals are already gone, so even
        # reading the flag can raise. A teardown report is never worth a
        # traceback on the way out.
        with contextlib.suppress(Exception):
            if os.getenv("VLLM_EEP_DEBUG_COMM_TEARDOWN") != "1":
                return
            logger.warning(
                "[EEP-DEBUG] eplb communicator teardown: %d registration(s), "
                "%d remote agent(s), %d transfer handle(s)",
                len(self._registered_descs),
                len(self._remote_agents),
                len(self._xfer_entries),
            )

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
            self._report_teardown()
        with contextlib.suppress(Exception):
            for local_h, remote_h, xfer_h in self._xfer_entries:
                with contextlib.suppress(Exception):
                    self._nixl_wrapper.release_xfer_handle(xfer_h)
                with contextlib.suppress(Exception):
                    self._nixl_wrapper.release_dlist_handle(local_h)
                with contextlib.suppress(Exception):
                    self._nixl_wrapper.release_dlist_handle(remote_h)
        release_arena = True
        with contextlib.suppress(Exception):
            release_arena = self._arena_registration_is_last()
        if not release_arena:
            # Park the agent untouched. Removing its peers here would run the
            # same teardown paths over the same rkeys that the live agent is
            # using, which is the defect this guard exists for.
            with contextlib.suppress(Exception):
                _PARKED_AGENTS.append(self._nixl_wrapper)
                logger.warning(
                    "[EEP-DEBUG] keeping an eplb registration alive because "
                    "another communicator still covers the same expert arena, "
                    "%d agent(s) parked so far",
                    len(_PARKED_AGENTS),
                )
            return
        with contextlib.suppress(Exception):
            for descs in self._registered_descs:
                with contextlib.suppress(Exception):
                    self._nixl_wrapper.deregister_memory(descs)
            self._registered_descs.clear()
        with contextlib.suppress(Exception):
            for agent_name in self._remote_agents.values():
                with contextlib.suppress(Exception):
                    self._nixl_wrapper.remove_remote_agent(agent_name)
            self._remote_agents.clear()


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
