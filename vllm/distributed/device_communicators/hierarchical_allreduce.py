# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Hierarchical AllReduce combining:
  - vLLM's custom intra-node all-reduce (NVLink)
  - UCCL-EP inter-node all-reduce (RDMA via CPU proxy)

Optimized for SMALL messages where latency dominates.

Architecture:
  Phase 1: Intra-node reduce   (vLLM custom AR, NVLink direct ptr)
  Phase 2: Inter-node reduce   (UCCL-EP FIFO → CPU proxy → RDMA)
  Phase 3: Intra-node broadcast (NVLink direct ptr read)
"""

from contextlib import contextmanager
from typing import Optional

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from vllm import _custom_ops as ops
from vllm.distributed.device_communicators.custom_all_reduce import (
    CustomAllreduce,
    is_weak_contiguous,
)
from vllm.logger import init_logger

logger = init_logger(__name__)


class HierarchicalAllreduce:
    """
    Hierarchical all-reduce for multi-node vLLM inference.

    On single-node, this disables itself and the caller falls through
    to the regular CustomAllreduce path.
    """

    _SUPPORTED_WORLD_SIZES = [2, 4, 6, 8]

    def __init__(
        self,
        # Process groups
        local_group: ProcessGroup,
        global_group: ProcessGroup,
        gateway_group: Optional[ProcessGroup],
        # Device info
        device: torch.device,
        local_rank: int,
        global_rank: int,
        node_id: int,
        num_nodes: int,
        local_world_size: int,
        # Configuration
        max_size: int = 8192 * 1024,  # 8 MB max message
        gateway_local_rank: int = 0,
        num_proxy_threads: int = 4,
    ):
        self.local_group = local_group
        self.global_group = global_group
        self.gateway_group = gateway_group
        self.device = device
        self.local_rank = local_rank
        self.global_rank = global_rank
        self.node_id = node_id
        self.num_nodes = num_nodes
        self.local_world_size = local_world_size
        self.max_size = max_size
        self.gateway_local_rank = gateway_local_rank
        self.is_gateway = (local_rank == gateway_local_rank)
        self.disabled = False
        self._IS_CAPTURING = False
        self._ptr = 0

        if num_nodes <= 1:
            logger.info("Single node — hierarchical AR not needed")
            self.disabled = True
            return

        if local_world_size not in self._SUPPORTED_WORLD_SIZES:
            logger.warning(
                "local_world_size %d not in %s; hierarchical AR disabled",
                local_world_size, self._SUPPORTED_WORLD_SIZES)
            self.disabled = True
            return

        # ────────────────────────────────────────────
        # Step 1: Initialize intra-node custom AR
        # ────────────────────────────────────────────
        self.intra_ar = CustomAllreduce(
            group=local_group,
            device=device,
            max_size=max_size,
        )
        if self.intra_ar.disabled:
            logger.warning("Intra-node custom AR disabled; "
                           "hierarchical AR disabled too")
            self.disabled = True
            return

        # ────────────────────────────────────────────
        # Step 2: Create broadcast buffer (IPC shared within node)
        # ────────────────────────────────────────────
        self.broadcast_ptrs = CustomAllreduce.create_shared_buffer(
            max_size, group=local_group
        )

        # ────────────────────────────────────────────
        # Step 3: Create hierarchical signal buffer (IPC shared)
        # ────────────────────────────────────────────
        self.hier_signal_ptrs = CustomAllreduce.create_shared_buffer(
            ops.hier_signal_size(), group=local_group
        )

        # ────────────────────────────────────────────
        # Step 4: Initialize C++ HierarchicalAllreduce
        # ────────────────────────────────────────────
        self._ptr = ops.init_hierarchical_ar(
            self.intra_ar._ptr,
            self.broadcast_ptrs,
            self.hier_signal_ptrs,
            local_rank,
            local_world_size,
            node_id,
            num_nodes,
            gateway_local_rank,
            max_size,
            num_proxy_threads,
        )

        logger.info(
            "HierarchicalAllreduce initialized: node %d/%d, "
            "local_rank %d/%d, gateway=%s",
            node_id, num_nodes, local_rank, local_world_size,
            self.is_gateway
        )

    def should_custom_ar(self, inp: torch.Tensor) -> bool:
        """Check if hierarchical allreduce should be used."""
        if self.disabled:
            return False
        inp_size = inp.numel() * inp.element_size()
        if inp_size % 16 != 0:
            return False
        if not is_weak_contiguous(inp):
            return False
        if inp_size > self.max_size:
            return False
        return True

    def all_reduce(
        self, inp: torch.Tensor, *, out: torch.Tensor = None
    ) -> torch.Tensor:
        """Perform hierarchical all-reduce."""
        if out is None:
            out = torch.empty_like(inp)
        ops.hierarchical_all_reduce(self._ptr, inp, out)
        return out

    def custom_all_reduce(
        self, input: torch.Tensor
    ) -> Optional[torch.Tensor]:
        """Main API matching CustomAllreduce interface."""
        if self.disabled or not self.should_custom_ar(input):
            return None

        if self._IS_CAPTURING:
            if torch.cuda.is_current_stream_capturing():
                return self.all_reduce(input)
            else:
                return torch.empty_like(input)
        else:
            return self.all_reduce(input)

    @contextmanager
    def capture(self):
        """CUDA graph capture context."""
        try:
            self._IS_CAPTURING = True
            if not self.disabled:
                self.intra_ar._IS_CAPTURING = True
            yield
        finally:
            self._IS_CAPTURING = False
            if not self.disabled:
                self.intra_ar._IS_CAPTURING = False
                self.intra_ar.register_graph_buffers()

    def close(self):
        if not self.disabled and self._ptr:
            ops.dispose_hierarchical_ar(self._ptr)
            self._ptr = 0
            self.intra_ar.close()
            CustomAllreduce.free_shared_buffer(
                self.broadcast_ptrs, rank=self.local_rank)
            CustomAllreduce.free_shared_buffer(
                self.hier_signal_ptrs, rank=self.local_rank)

    def __del__(self):
        self.close()
