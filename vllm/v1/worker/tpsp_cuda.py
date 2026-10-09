# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""BF16 CUDA TPSP projection with NCCL and optional P2P transport."""

from __future__ import annotations

import logging
import math
import socket
from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as torch_symm_mem
from torch.distributed import distributed_c10d as c10d

from vllm.v1.worker.tpsp_profile import ChunkConfig, TPSPBackend

_LOG = logging.getLogger(__name__)


@dataclass
class CudaTPSPContext:
    group_name: str
    device: torch.device
    tp_size: int
    rank: int
    comm_address: int
    modes: frozenset[str]
    workspace: torch.Tensor | None = None
    peer_workspaces: tuple[torch.Tensor, ...] = ()
    signal_one: torch.Tensor | None = None
    workspace_handle: object | None = None
    closed: bool = False


class CudaTPSPOps:
    tpsp_chunk_granularity = 64

    @classmethod
    def open(
        cls,
        group_name: str,
        device: torch.device,
        max_batched_tokens: int,
        hidden_size: int,
    ) -> CudaTPSPContext | None:
        if device.type != "cuda" or max_batched_tokens <= 0 or hidden_size <= 0:
            _LOG.warning("CUDA TPSP open requires valid device and dimensions")
            return None
        if (
            device.index is not None
            and device.index >= torch.accelerator.device_count()
        ):
            _LOG.warning("CUDA TPSP device index is unavailable: %s", device.index)
            return None
        try:
            group = c10d._resolve_process_group(group_name)
        except (RuntimeError, ValueError) as exc:
            _LOG.warning("CUDA TPSP process group is unavailable: %s", exc)
            return None
        tp_size = dist.get_world_size(group)
        if tp_size < 2 or hidden_size % tp_size:
            _LOG.warning("CUDA TPSP open requires TP >= 2 and divisible hidden size")
            return None
        if not hasattr(
            torch.ops._C, "tpsp_fused_matmul_reduce_scatter_norm_all_gather"
        ):
            _LOG.warning("CUDA TPSP native operator is not built")
            return None
        from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
        from vllm.distributed.parallel_state import get_tp_group

        tp_group = get_tp_group()
        if tp_group.device_group is not group:
            _LOG.warning("CUDA TPSP group is not the active TP group")
            return None
        comm = getattr(tp_group.device_communicator, "pynccl_comm", None)
        nccl_ready = (
            isinstance(comm, PyNcclCommunicator)
            and comm.available
            and not comm.disabled
            and comm.comm.value is not None
        )
        if nccl_ready:
            assert isinstance(comm, PyNcclCommunicator)
            comm_address = comm.comm.value
            assert comm_address is not None
        else:
            comm_address = 0
        rank = dist.get_rank(group)
        context = CudaTPSPContext(
            group_name, device, tp_size, rank, comm_address, frozenset()
        )
        cls._initialize_p2p(context, group, max_batched_tokens, hidden_size)
        nccl_available = torch.tensor(int(nccl_ready), device=device)
        dist.all_reduce(nccl_available, op=dist.ReduceOp.MIN, group=group)
        modes = set()
        if nccl_available.item():
            modes.add("nccl")
        if context.workspace is not None:
            modes.add("p2p")
        if not modes:
            _LOG.warning("TPSP unavailable: neither NCCL nor P2P is usable")
            return None
        context.modes = frozenset(modes)
        return context

    @staticmethod
    def _initialize_p2p(
        context: CudaTPSPContext,
        group: dist.ProcessGroup,
        max_batched_tokens: int,
        hidden_size: int,
    ) -> None:
        tp_size = context.tp_size
        rank = context.rank
        device = context.device
        rows = math.ceil(max_batched_tokens / tp_size)
        slot_bytes = 2 * rows * hidden_size
        flag_bytes = 3 * tp_size * 4
        workspace_bytes = 2 * tp_size * slot_bytes + flag_bytes
        local_uuid = str(getattr(torch.cuda.get_device_properties(device), "uuid", ""))
        devices: list[tuple[str, str] | None] = [None] * tp_size
        dist.all_gather_object(devices, (socket.gethostname(), local_uuid), group=group)
        if any(peer is None for peer in devices):
            raise RuntimeError("TPSP P2P device discovery returned an incomplete group")
        peers = [peer for peer in devices if peer is not None]
        visible = {
            str(getattr(torch.cuda.get_device_properties(index), "uuid", "")): index
            for index in range(torch.accelerator.device_count())
        }
        accessible = (
            device.index is not None
            and bool(local_uuid)
            and all(
                host == peers[rank][0]
                and bool(uuid)
                and uuid in visible
                and (
                    source == rank
                    or torch.cuda.can_device_access_peer(device.index, visible[uuid])
                )
                for source, (host, uuid) in enumerate(peers)
            )
            and len({uuid for _, uuid in peers}) == tp_size
        )
        supported = torch.tensor(int(accessible), device=device)
        dist.all_reduce(supported, op=dist.ReduceOp.MIN, group=group)
        if not supported.item():
            _LOG.info("TPSP P2P topology unavailable; profiling NCCL only")
            return
        try:
            workspace = torch_symm_mem.empty(
                workspace_bytes, dtype=torch.uint8, device=device
            )
            handle = torch_symm_mem.rendezvous(workspace, group)
            peer_workspaces = tuple(
                workspace
                if source == rank
                else handle.get_buffer(source, (workspace_bytes,), torch.uint8)
                for source in range(tp_size)
            )
        except RuntimeError as exc:
            _LOG.warning("TPSP P2P workspace unavailable: %s", exc)
            workspace = None
            handle = None
            peer_workspaces = ()
        ready = torch.tensor(int(handle is not None), device=device)
        dist.all_reduce(ready, op=dist.ReduceOp.MIN, group=group)
        if not ready.item():
            return
        assert workspace is not None and handle is not None
        workspace.zero_()
        torch.accelerator.synchronize(device)
        dist.barrier(group=group)
        signal_one = torch.ones(1, dtype=torch.int32, device=device)
        try:
            for dest in range(tp_size):
                if dest != rank:
                    peer_workspaces[dest][
                        -flag_bytes + 4 * rank : -flag_bytes + 4 * (rank + 1)
                    ].copy_(signal_one.view(torch.uint8))
            torch.accelerator.synchronize(device)
            copy_works = True
        except RuntimeError as exc:
            _LOG.warning("TPSP P2P copy probe failed: %s", exc)
            copy_works = False
        ready = torch.tensor(int(copy_works), device=device)
        dist.all_reduce(ready, op=dist.ReduceOp.MIN, group=group)
        if not ready.item():
            return
        try:
            copy_works = all(
                torch.equal(
                    workspace[
                        -flag_bytes + 4 * source : -flag_bytes + 4 * (source + 1)
                    ],
                    signal_one.view(torch.uint8),
                )
                for source in range(tp_size)
                if source != rank
            )
        except RuntimeError as exc:
            _LOG.warning("TPSP P2P copy probe failed: %s", exc)
            copy_works = False
        ready = torch.tensor(int(copy_works), device=device)
        dist.all_reduce(ready, op=dist.ReduceOp.MIN, group=group)
        if not ready.item():
            return
        workspace[-flag_bytes:].zero_()
        torch.accelerator.synchronize(device)
        dist.barrier(group=group)
        context.workspace = workspace
        context.peer_workspaces = peer_workspaces
        context.signal_one = signal_one
        context.workspace_handle = handle

    @staticmethod
    def fused_matmul_reduce_scatter_norm_all_gather(
        context: CudaTPSPContext,
        a: torch.Tensor,
        b: torch.Tensor,
        weight: torch.Tensor,
        _unused: None,
        *,
        eps: float,
        norm_type: str,
        residual: torch.Tensor,
        microchunk_tokens: int,
        projection_bias: torch.Tensor | None = None,
        norm_bias: torch.Tensor | None = None,
        comm_mode: str = "nccl",
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if context.closed:
            raise RuntimeError("CUDA TPSP backend is closed")
        if (
            norm_type not in ("rms_norm", "layer_norm")
            or comm_mode not in ("nccl", "p2p")
            or microchunk_tokens <= 0
            or eps <= 0
            or (norm_bias is not None and norm_type != "layer_norm")
        ):
            raise ValueError("Invalid CUDA TPSP normalization or chunk configuration")
        if (
            any(
                t.dtype != torch.bfloat16 or t.device != context.device
                for t in (a, b, weight, residual)
            )
            or a.ndim != 2
            or b.ndim != 2
            or weight.ndim != 1
            or residual.ndim != 2
            or a.shape[1] != b.shape[0]
            or b.shape[1] != weight.numel()
        ):
            raise ValueError("CUDA TPSP requires compatible CUDA BF16 inputs")
        for bias in (projection_bias, norm_bias):
            if bias is not None and (
                bias.dtype != torch.bfloat16
                or bias.device != context.device
                or bias.shape != weight.shape
                or not bias.is_contiguous()
            ):
                raise ValueError("CUDA TPSP bias must match the BF16 norm weight")
        if comm_mode not in context.modes:
            raise RuntimeError(f"CUDA TPSP {comm_mode} transport is unavailable")
        use_p2p = comm_mode == "p2p"

        return torch.ops._C.tpsp_fused_matmul_reduce_scatter_norm_all_gather(
            a,
            b,
            weight,
            residual,
            projection_bias,
            norm_bias,
            eps,
            1 if norm_type == "layer_norm" else 0,
            microchunk_tokens,
            context.comm_address,
            context.tp_size,
            context.workspace if use_p2p else None,
            [peer.data_ptr() for peer in context.peer_workspaces] if use_p2p else [],
            context.signal_one if use_p2p else None,
            context.rank if use_p2p else -1,
        )

    @staticmethod
    def close_tpsp(context: CudaTPSPContext) -> None:
        if context.closed:
            return
        if context.workspace is not None:
            torch.accelerator.synchronize(context.device)
        context.workspace_handle = None
        context.peer_workspaces = ()
        context.workspace = None
        context.signal_one = None
        context.closed = True


class CudaTPSPBackend(TPSPBackend):
    requires_projection_context = True
    profiles_transport_modes = True
    synchronize_after_fused = False
    supports_projection_bias = True

    def __init__(self, ops: type[CudaTPSPOps], group_name: str, device: torch.device):
        super().__init__(ops, group_name, device)
        self._open_context_ids: set[int] = set()

    @classmethod
    def create(cls, group_name: str, device: torch.device) -> CudaTPSPBackend:
        return cls(CudaTPSPOps, group_name, device)

    def open(
        self,
        *,
        dtype: torch.dtype,
        tp_size: int,
        hidden_size: int,
        max_batched_tokens: int,
        group_name: str,
        device: torch.device,
    ) -> CudaTPSPContext | None:
        if self._closed:
            raise RuntimeError("TPSP backend is closed")
        if (
            device != self.device
            or group_name != self.group_name
            or device.type != "cuda"
            or not self._valid_open(
                dtype=dtype,
                tp_size=tp_size,
                hidden_size=hidden_size,
                max_batched_tokens=max_batched_tokens,
                group_name=group_name,
                device=device,
            )
        ):
            return None
        context = self.ops.open(group_name, device, max_batched_tokens, hidden_size)
        if context is not None:
            self._open_context_ids.add(id(context))
        return context

    def _profile_context(self, context: CudaTPSPContext | None) -> CudaTPSPContext:
        if context is None:
            raise ValueError("CUDA TPSP requires a projection context")
        if (
            not isinstance(context, CudaTPSPContext)
            or id(context) not in self._open_context_ids
            or context.group_name != self.group_name
            or context.device != self.device
            or context.closed
        ):
            raise ValueError("CUDA TPSP context belongs to another backend")
        return context

    def run(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        weight: torch.Tensor,
        residual: torch.Tensor,
        eps: float,
        config: ChunkConfig,
        *,
        norm_type: str = "rms_norm",
        projection_bias: torch.Tensor | None = None,
        norm_bias: torch.Tensor | None = None,
        context: CudaTPSPContext | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        context = self._profile_context(context)
        return self.ops.fused_matmul_reduce_scatter_norm_all_gather(
            context,
            a,
            b,
            weight,
            None,
            eps=eps,
            norm_type=norm_type,
            residual=residual,
            microchunk_tokens=config.microchunk_tokens,
            projection_bias=projection_bias,
            norm_bias=norm_bias,
            comm_mode=config.comm_mode,
        )

    def close(self, context: CudaTPSPContext | None = None) -> None:
        if context is not None:
            self._profile_context(context)
            self.ops.close_tpsp(context)
            self._open_context_ids.remove(id(context))
        else:
            super().close()
