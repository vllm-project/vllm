# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""All-reduce for TP groups on ROCm GPUs joined by XGMI islands (e.g. MI210).

ROCm enables vLLM's custom all-reduce only on MI300 and newer, so on MI200 every
TP all-reduce goes through RCCL, which uses a ring. Inside a fully connected
XGMI island the custom one-shot kernel reads every peer's buffer directly in
one step and is faster for the smaller tensors:

* one island (a TP group of 2 or 4 fully connected GPUs): custom all-reduce
  for the smaller tensors, RCCL for the rest;
* two islands (a TP group of 8 whose halves are fully connected and joined by
  PCIe): custom all-reduce inside each island, then across the GPU pair at the
  same island position, for the smaller tensors; for large tensors, RCCL
  reduce-scatter inside the island, all-reduce of the quarter-size shard
  across the pairs, all-gather inside the island.

The islands are found from the link topology, so the TP ranks need not be
ordered by island. The custom kernels always copy the input into the
pre-registered IPC buffer, also inside CUDA graphs: registered graph buffers
return wrong sums on MI210 with ROCm 7.
"""

from contextlib import ExitStack, contextmanager

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

import vllm.envs as envs
from vllm.config import get_current_vllm_config_or_none
from vllm.logger import init_logger
from vllm.platforms import current_platform

logger = init_logger(__name__)

_ISLAND_WORLD_SIZE = 8
_ONE_ISLAND_WORLD_SIZES = (2, 4)
# Speculative methods that run no draft model, so capture no drafter graphs.
_NO_DRAFT_MODEL_METHODS = ("ngram", "ngram_gpu", "suffix")


def _drafter_replays_graphs_in_capture() -> bool:
    """Whether a drafter graph can be replayed before graph capture ends.

    The custom kernels' graph buffer addresses are filled in when
    graph_capture() exits. The V1 model runner runs the drafter inside each
    target capture, so a drafter graph recorded for one size is replayed while
    a later size is captured, before its addresses exist.
    """
    config = get_current_vllm_config_or_none()
    spec = config.speculative_config if config is not None else None
    return (
        config is not None
        and spec is not None
        and spec.method not in _NO_DRAFT_MODEL_METHODS
        and not config.use_v2_model_runner
    )


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
        # One custom all-reduce per level: island, then pair. A single island
        # has the first only. Empty when the custom path is off.
        self.ca_stages: list[CustomAllreduce] = []
        self.nccl_island: PyNcclCommunicator | None = None
        self.nccl_pair: PyNcclCommunicator | None = None
        self.hierarchical = False
        world_size = dist.get_world_size(cpu_group)
        if not current_platform.is_rocm() or world_size < 2:
            return

        assert device.index is not None
        phys = torch.tensor(
            [current_platform.visible_device_id_to_physical_device_id(device.index)],
            dtype=torch.int,
        )
        gathered = [torch.zeros_like(phys) for _ in range(world_size)]
        dist.all_gather(gathered, phys, group=cpu_group)
        ids = [int(t.item()) for t in gathered]

        if current_platform.is_fully_connected(ids):
            # Platforms with the flat custom all-reduce use that instead.
            if (
                current_platform.use_custom_allreduce()
                or world_size not in _ONE_ISLAND_WORLD_SIZES
            ):
                return
            island_ids = [ids]
            groups = [cpu_group]
        else:
            # dist.new_group below must be called by every process of the job.
            if world_size != dist.get_world_size():
                return
            positions = find_two_islands(ids)
            if positions is None:
                return
            island_ids = [[ids[i] for i in island] for island in positions]

            # A group ranks its members by ascending global rank, so the k-th
            # GPU of one island pairs with the k-th GPU of the other: the
            # hierarchical path needs both ends of a pair to hold the same shard.
            members = [sorted(global_ranks[i] for i in island) for island in positions]
            islands = [dist.new_group(ranks, backend="gloo") for ranks in members]
            pairs = [
                dist.new_group([a, b], backend="gloo")
                for a, b in zip(members[0], members[1])
            ]
            my_rank = global_ranks[dist.get_rank(cpu_group)]
            side = 0 if my_rank in members[0] else 1
            groups = [islands[side], pairs[members[side].index(my_rank)]]
            # PyNccl rather than torch NCCL process groups: their watchdog
            # queries events, which HIP forbids during graph capture.
            self.nccl_island = PyNcclCommunicator(groups[0], device)
            self.nccl_pair = PyNcclCommunicator(groups[1], device)
            self.hierarchical = not (
                self.nccl_island.disabled or self.nccl_pair.disabled
            )

        self.max_custom_bytes = envs.VLLM_ROCM_ISLAND_AR_CUSTOM_MAX_BYTES
        self.min_hier_bytes = envs.VLLM_ROCM_ISLAND_AR_HIER_MIN_BYTES
        # A zero limit turns the custom path off; CustomAllreduce cannot
        # allocate an empty shared buffer.
        if self.max_custom_bytes > 0 and _drafter_replays_graphs_in_capture():
            logger.info_once(
                "Island all-reduce: custom path off with a speculative draft "
                "model on the V1 model runner.",
                scope="global",
            )
        elif self.max_custom_bytes > 0:
            stages = [
                CustomAllreduce(
                    group,
                    device,
                    max_size=self.max_custom_bytes,
                    max_all_gather_size=self.max_custom_bytes,
                    max_reduce_scatter_size=self.max_custom_bytes,
                )
                for group in groups
            ]
            if not any(ca.disabled for ca in stages):
                self.ca_stages = stages
        if not self.ca_stages and not self.hierarchical:
            return
        self.disabled = False
        logger.info_once(
            "Island all-reduce enabled for GPU islands %s: custom < %d B (%s), "
            "hierarchical RCCL >= %d B (%s).",
            str(island_ids),
            self.max_custom_bytes,
            "on" if self.ca_stages else "off",
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
            self.nccl_island is not None
            and self.nccl_pair is not None
            and self.hierarchical
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
            self.ca_stages
            and nbytes < self.max_custom_bytes
            and self.ca_stages[0].should_custom_ar(input_)
        ):
            out = input_
            for ca in self.ca_stages:
                out = ca.all_reduce(out, registered=False)
            return out
        return None

    @contextmanager
    def capture(self):
        """Exchange the custom kernels' graph buffer addresses after capture."""
        with ExitStack() as stack:
            for ca in self.ca_stages:
                stack.enter_context(ca.capture())
            yield

    def destroy(self):
        if self.disabled:
            return
        for nccl in (self.nccl_island, self.nccl_pair):
            if nccl is not None:
                nccl.destroy()
        self.ca_stages = []
        self.disabled = True
