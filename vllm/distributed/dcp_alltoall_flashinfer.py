# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Workspace manager for FlashInfer's DCP all-to-all kernel.

This module wraps FlashInfer's ``decode_cp_a2a_*`` API (added in
flashinfer-ai/flashinfer#2951) so that vLLM's DCP A2A path can use it as
an alternative to ``dist.all_to_all_single``.

Lifecycle (per CP group, done once):
  1. Allocate the MNNVL-backed workspace.
  2. ``decode_cp_a2a_init_workspace`` to reset the FIFO.
  3. CPU barrier across the CP group so every rank's init has completed
     before any rank issues the first all-to-all (otherwise a peer can
     start writing into a not-yet-initialized FIFO).

Per-call:
  ``run(partial_o, softmax_stats)`` invokes ``decode_cp_a2a_alltoall``.
  The kernel is a single fused LL128 exchange — no extra Python
  overhead beyond the call.

Workspaces are cached by ``(cp_rank, cp_size)`` so the allocate+init
cost is paid exactly once per process.

Note on ``enable_pdl=False``
============================
We pass ``enable_pdl=False`` to match TensorRT-LLM's binding
(``cpp/tensorrt_llm/thop/alltoallOp.cpp``), which uses the 3-arg
``launchHelixAllToAll`` overload (no PDL). FlashInfer 0.6.9's Python
binding defaults to PDL on SM90+, but TRT-LLM has shipped without it
for production helix CP — so we follow that conservative choice.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import torch
import torch.distributed as dist

if TYPE_CHECKING:
    from torch.distributed import ProcessGroup

logger = logging.getLogger(__name__)


def _to_torch(t: Any) -> torch.Tensor:
    """Convert a tvm-ffi tensor (or any DLPack object) to ``torch.Tensor``."""
    if isinstance(t, torch.Tensor):
        return t
    return torch.from_dlpack(t)


class DCPAllToAllFlashInfer:
    """Manages FlashInfer DCP workspace and executes the all-to-all kernel.

    Usage::

        mgr = DCPAllToAllFlashInfer.get(
            cp_rank=cp_group.rank_in_group,
            cp_size=cp_group.world_size,
            cp_cpu_group=cp_group.cpu_group,
        )
        recv_o, recv_stats = mgr.run(partial_o, softmax_stats)
    """

    _cache: dict[tuple[int, int], DCPAllToAllFlashInfer] = {}

    def __init__(
        self,
        cp_rank: int,
        cp_size: int,
        workspace: torch.Tensor,
        *,
        use_mnnvl: bool,
        cp_cpu_group: ProcessGroup | None = None,
    ) -> None:
        self.cp_rank = cp_rank
        self.cp_size = cp_size
        self.workspace = workspace
        self._use_mnnvl = use_mnnvl
        self._cp_cpu_group = cp_cpu_group

    @staticmethod
    def get(
        cp_rank: int,
        cp_size: int,
        cp_cpu_group: ProcessGroup | None = None,
    ) -> DCPAllToAllFlashInfer:
        """Return a ready-to-use manager for the given ``(cp_rank, cp_size)``.

        Allocates and initializes the workspace on first call; subsequent
        calls return the cached instance.

        Args:
            cp_rank: This rank's position in the CP group.
            cp_size: CP group size.
            cp_cpu_group: CPU ``ProcessGroup`` for the CP ranks. Required
                for the post-init barrier and (when present) drives the
                MNNVL communicator. Pass the ``cpu_group`` from the DCP
                ``GroupCoordinator``.
        """
        key = (cp_rank, cp_size)
        cached = DCPAllToAllFlashInfer._cache.get(key)
        if cached is not None:
            return cached

        workspace, used_mnnvl = DCPAllToAllFlashInfer._allocate(
            cp_rank, cp_size, cp_cpu_group
        )

        from flashinfer.comm import decode_cp_a2a_init_workspace

        decode_cp_a2a_init_workspace(workspace, cp_rank, cp_size)

        # Cross-rank barrier so every rank's FIFO init is visible to peers
        # before any of them starts writing to a peer's slot.
        if cp_cpu_group is not None:
            dist.barrier(group=cp_cpu_group)
        else:
            torch.accelerator.synchronize()

        mgr = DCPAllToAllFlashInfer(
            cp_rank,
            cp_size,
            workspace,
            use_mnnvl=used_mnnvl,
            cp_cpu_group=cp_cpu_group,
        )
        DCPAllToAllFlashInfer._cache[key] = mgr
        logger.info(
            "Rank %d: FlashInfer DCP A2A workspace ready "
            "(cp_size=%d, mnnvl=%s, shape=%s)",
            cp_rank,
            cp_size,
            used_mnnvl,
            list(workspace.shape),
        )
        return mgr

    @staticmethod
    def _allocate(
        cp_rank: int,
        cp_size: int,
        cp_cpu_group: ProcessGroup | None,
    ) -> tuple[torch.Tensor, bool]:
        """Allocate the MNNVL workspace; non-MNNVL is unsupported.

        FlashInfer's ``decode_cp_a2a_alltoall`` kernel addresses peer FIFOs
        through a single ``params.workspace + peer_rank * stride`` base
        pointer. That only resolves correctly when the workspace is a
        unified VA backed by MNNVL fabric memory. The plain ``torch.zeros``
        fallback in PR #2951 hangs the kernel — see
        ``project_dcp_a2a_h200_unsupported.md`` for a full diagnosis.
        """
        if cp_cpu_group is None:
            raise RuntimeError(
                "DCPAllToAllFlashInfer requires a Gloo CPU group for MNNVL "
                "communicator setup; got cp_cpu_group=None."
            )
        workspace = DCPAllToAllFlashInfer._allocate_mnnvl(
            cp_rank, cp_size, cp_cpu_group
        )
        return workspace, True

    @staticmethod
    def _allocate_mnnvl(
        cp_rank: int,
        cp_size: int,
        cp_cpu_group: ProcessGroup | None,
    ) -> torch.Tensor:
        """Allocate an MNNVL-backed workspace visible across nodes.

        We bypass FlashInfer's ``set_comm_from_config`` because its split
        formula (``pp_rank * cp_size + cp_rank``) is meant for MoE A2A,
        which groups TP peers together. For DCP A2A we want CP peers
        grouped, which is exactly what ``cp_cpu_group`` already
        represents — so we set ``MnnvlMemory.comm`` directly and skip
        the split.
        """
        from flashinfer.comm import (
            Mapping,
            decode_cp_a2a_allocate_mnnvl_workspace,
        )
        from flashinfer.comm.mnnvl import MnnvlMemory, TorchDistBackend

        MnnvlMemory.initialize()
        MnnvlMemory.comm = TorchDistBackend(group=cp_cpu_group)

        # Mapping is consumed by ``MnnvlMemory.__init__`` for segment
        # layout; the comm is already correct so ``mnnvl_config`` is
        # intentionally not passed (which suppresses the broken split).
        mapping = Mapping(
            world_size=cp_size,
            rank=cp_rank,
            cp_size=cp_size,
            tp_size=1,
            pp_size=1,
        )
        # New FlashInfer signature (post-#3210) takes mapping only;
        # cp_size and cp_rank come from the mapping fields above.
        return decode_cp_a2a_allocate_mnnvl_workspace(mapping)

    def run(
        self,
        partial_o: torch.Tensor,
        softmax_stats: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the FlashInfer all-to-all and return ``(recv_o, recv_stats)``.

        Args:
            partial_o: ``[..., cp_size, D]``, half/bfloat16, contiguous.
            softmax_stats: ``[..., cp_size, S]``, float32, contiguous,
                ``S >= 2`` and even.

        Returns:
            ``(recv_o, recv_stats)`` with the same shapes/dtypes as the
            inputs. Transpose property along the cp dimension:
            ``recv_o[rank][..., peer, :] == send_o[peer][..., rank, :]``.
        """
        from flashinfer.comm import decode_cp_a2a_alltoall

        # DEBUG (Phase 3 multinode validation): write one-shot per-rank
        # marker the first time run() is invoked, to confirm FlashInfer
        # is actually being routed to (not silently NCCL fallback).
        # Cheap: one open()+write per process for life of the process.
        if not getattr(self, "_first_call_logged", False):
            try:
                with open(f"/tmp/a2a_first_call.{self.cp_rank}", "w") as f:
                    f.write(
                        f"cp_rank={self.cp_rank} cp_size={self.cp_size} "
                        f"po_shape={tuple(partial_o.shape)} "
                        f"po_dtype={partial_o.dtype}\n"
                    )
            except Exception:
                pass
            self._first_call_logged = True

        recv_o, recv_stats = decode_cp_a2a_alltoall(
            partial_o,
            softmax_stats,
            self.workspace,
            self.cp_rank,
            self.cp_size,
            enable_pdl=False,
        )
        return _to_torch(recv_o), _to_torch(recv_stats)

    @staticmethod
    def clear_cache() -> None:
        """Drop all cached managers. For tests / shutdown only."""
        DCPAllToAllFlashInfer._cache.clear()

    def __repr__(self) -> str:
        return (
            f"DCPAllToAllFlashInfer(cp_rank={self.cp_rank}, "
            f"cp_size={self.cp_size}, mnnvl={self._use_mnnvl}, "
            f"workspace={tuple(self.workspace.shape)})"
        )
