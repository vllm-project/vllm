# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Two-level all-reduce for a TP group spanning two XGMI islands (ROCm).

Targets nodes whose 8 GPUs form two fully connected 4-GPU XGMI islands joined
only by PCIe (e.g. 8x MI210). There the flat custom all-reduce is unavailable
(the group is not fully connected) and a flat RCCL ring crosses PCIe twice per
step. Instead:

* large tensors: RCCL reduce-scatter inside the island, all-reduce
  of the quarter-size shard across the pairs, all-gather inside the island;
* smaller tensors: vLLM's custom all-reduce inside each
  island, then across the GPU pair at the same island position.

The islands are found from the link topology, so the TP ranks need not be
ordered by island.

The custom kernels always copy the input into the pre-registered IPC buffer,
also inside CUDA graphs: registered graph buffers return wrong sums on MI210
with ROCm 7.
"""

from contextlib import ExitStack, contextmanager

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.platforms import current_platform

logger = init_logger(__name__)

_ISLAND_WORLD_SIZE = 8


def find_two_islands(
    physical_device_ids: list[int],
) -> tuple[list[int], list[int]] | None:
    """Split the GPUs into two fully connected halves.

    Returns the positions in ``physical_device_ids`` of each half, in
    ascending order, or None if the GPUs are fully connected or do not split
    into two equal fully connected groups.
    """
    ids = physical_device_ids
    if len(ids) != _ISLAND_WORLD_SIZE or current_platform.is_fully_connected(ids):
        return None
    first = [0] + [
        i
        for i in range(1, len(ids))
        if current_platform.is_fully_connected([ids[0], ids[i]])
    ]
    second = [i for i in range(len(ids)) if i not in first]
    if len(first) != len(second):
        return None
    for island in (first, second):
        if not current_platform.is_fully_connected([ids[i] for i in island]):
            return None
    return first, second


class IslandAllReduce:
    def __init__(
        self,
        cpu_group: ProcessGroup,
        device: torch.device,
        global_ranks: list[int],
    ):
        from vllm.distributed.device_communicators.custom_all_reduce import (
            CustomAllreduce,
        )
        from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator

        self.disabled = True
        self.ca_island: CustomAllreduce | None = None
        self.ca_pair: CustomAllreduce | None = None
        world_size = dist.get_world_size(cpu_group)
        # dist.new_group below must be called by every process of the job.
        if (
            not current_platform.is_rocm()
            or world_size != _ISLAND_WORLD_SIZE
            or world_size != dist.get_world_size()
        ):
            return

        assert device.index is not None
        phys = torch.tensor(
            [current_platform.visible_device_id_to_physical_device_id(device.index)],
            dtype=torch.int,
        )
        gathered = [torch.zeros_like(phys) for _ in range(world_size)]
        dist.all_gather(gathered, phys, group=cpu_group)
        ids = [int(t.item()) for t in gathered]
        positions = find_two_islands(ids)
        if positions is None:
            return

        # A group ranks its members by ascending global rank, so the k-th GPU
        # of one island pairs with the k-th GPU of the other: the hierarchical
        # path needs both ends of a pair to hold the same shard.
        members = [sorted(global_ranks[i] for i in island) for island in positions]
        islands = [dist.new_group(ranks, backend="gloo") for ranks in members]
        pairs = [
            dist.new_group([a, b], backend="gloo")
            for a, b in zip(members[0], members[1])
        ]
        my_rank = global_ranks[dist.get_rank(cpu_group)]
        side = 0 if my_rank in members[0] else 1
        island, pair = islands[side], pairs[members[side].index(my_rank)]

        self.max_custom_bytes = envs.VLLM_ROCM_ISLAND_AR_CUSTOM_MAX_BYTES
        self.min_hier_bytes = envs.VLLM_ROCM_ISLAND_AR_HIER_MIN_BYTES
        # A zero limit turns the custom path off; CustomAllreduce cannot
        # allocate an empty shared buffer.
        if self.max_custom_bytes > 0:
            ca_island, ca_pair = (
                CustomAllreduce(
                    group,
                    device,
                    max_size=self.max_custom_bytes,
                    max_all_gather_size=self.max_custom_bytes,
                    max_reduce_scatter_size=self.max_custom_bytes,
                )
                for group in (island, pair)
            )
            if not (ca_island.disabled or ca_pair.disabled):
                self.ca_island, self.ca_pair = ca_island, ca_pair
        # PyNccl rather than torch NCCL process groups: their watchdog queries
        # events, which HIP forbids during graph capture.
        self.nccl_island = PyNcclCommunicator(island, device)
        self.nccl_pair = PyNcclCommunicator(pair, device)
        self.hierarchical = not (self.nccl_island.disabled or self.nccl_pair.disabled)
        if self.ca_island is None and not self.hierarchical:
            return
        self.disabled = False
        logger.info_once(
            "Island all-reduce enabled for GPU islands %s: two-level custom < %d B "
            "(%s), hierarchical RCCL >= %d B (%s).",
            str([[ids[i] for i in island] for island in positions]),
            self.max_custom_bytes,
            "on" if self.ca_island is not None else "off",
            self.min_hier_bytes,
            "on" if self.hierarchical else "off",
            scope="global",
        )

    def all_reduce(self, input_: torch.Tensor) -> torch.Tensor | None:
        """Returns None when the caller should use its default path."""
        if self.disabled:
            return None
        nbytes = input_.numel() * input_.element_size()
        if (
            self.hierarchical
            and nbytes >= self.min_hier_bytes
            and input_.numel() % 4 == 0
            and input_.is_contiguous()
        ):
            flat = input_.view(-1)
            shard = torch.empty(
                flat.numel() // 4, dtype=input_.dtype, device=input_.device
            )
            self.nccl_island.reduce_scatter(shard, flat)
            self.nccl_pair.all_reduce(shard, shard)
            out = torch.empty_like(input_)
            self.nccl_island.all_gather(out.view(-1), shard)
            return out
        if (
            self.ca_island is not None
            and self.ca_pair is not None
            and nbytes < self.max_custom_bytes
            and self.ca_island.should_custom_ar(input_)
        ):
            out = self.ca_island.all_reduce(input_, registered=False)
            return self.ca_pair.all_reduce(out, registered=False)
        return None

    @contextmanager
    def capture(self):
        """Exchange the custom kernels' graph buffer addresses after capture."""
        with ExitStack() as stack:
            if self.ca_island is not None and self.ca_pair is not None:
                stack.enter_context(self.ca_island.capture())
                stack.enter_context(self.ca_pair.capture())
            yield

    def destroy(self):
        if self.disabled:
            return
        self.nccl_island.destroy()
        self.nccl_pair.destroy()
        self.ca_island = self.ca_pair = None
        self.disabled = True
