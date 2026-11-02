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
also inside CUDA graphs: registered graph buffers give wrong results on MI210.
"""

from contextlib import ExitStack, contextmanager

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from vllm.config import get_current_vllm_config_or_none
from vllm.logger import init_logger
from vllm.platforms import current_platform

logger = init_logger(__name__)

# Tensors below this size take the custom path, the rest the hierarchical one.
# This is the end-to-end break-even on 8x MI210, above the ~1 MiB crossover of
# the two paths in isolation. Other hardware may break even at a different size,
# so tune it here (per device if needed).
_SWITCH_BYTES = 2304 * 1024
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
    if (
        len(ids) not in IslandAllReduce.SUPPORTED_WORLD_SIZES
        or current_platform.is_fully_connected(ids)
    ):
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
    SUPPORTED_WORLD_SIZES = (8,)

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
        self.nccl_island: PyNcclCommunicator | None = None
        self.nccl_pair: PyNcclCommunicator | None = None
        self.groups: list[ProcessGroup] = []
        world_size = dist.get_world_size(cpu_group)
        # dist.new_group below must be called by every process of the job.
        if world_size != dist.get_world_size():
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

        self.island_size = len(positions[0])
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
        self.groups = [island, pair]

        if _drafter_replays_graphs_in_capture():
            logger.info_once(
                "Island all-reduce: custom path off with a speculative draft "
                "model on the V1 model runner.",
                scope="global",
            )
        else:
            ca_island, ca_pair = (
                CustomAllreduce(
                    group,
                    device,
                    max_size=_SWITCH_BYTES,
                    max_all_gather_size=_SWITCH_BYTES,
                    max_reduce_scatter_size=_SWITCH_BYTES,
                )
                for group in (island, pair)
            )
            if not (ca_island.disabled or ca_pair.disabled):
                self.ca_island, self.ca_pair = ca_island, ca_pair
        # This path running next to RDMA out of NIC-registered GPU memory reset our
        # 8x MI210 nodes (Broadcom bnxt_re NICs), possibly only on that setup. Large
        # tensors then take the default path, which did not reset them.
        config = get_current_vllm_config_or_none()
        kv_config = config.kv_transfer_config if config is not None else None
        if kv_config is not None and kv_config.registers_kv_cache_memory():
            logger.info_once(
                "Island all-reduce: hierarchical path off because the KV connector "
                "registers GPU memory for RDMA.",
                scope="global",
            )
        else:
            # PyNccl rather than torch NCCL process groups: their watchdog queries
            # events, which HIP forbids during graph capture.
            nccl_island = PyNcclCommunicator(island, device)
            nccl_pair = PyNcclCommunicator(pair, device)
            if not (nccl_island.disabled or nccl_pair.disabled):
                self.nccl_island, self.nccl_pair = nccl_island, nccl_pair
        if self.ca_island is None and self.nccl_island is None:
            return
        self.disabled = False
        logger.info_once(
            "Island all-reduce enabled for GPU islands %s: two-level custom < %d B "
            "(%s), hierarchical RCCL at or above it (%s).",
            str([[ids[i] for i in island] for island in positions]),
            _SWITCH_BYTES,
            "on" if self.ca_island is not None else "off",
            "on" if self.nccl_island is not None else "off",
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
            and nbytes >= _SWITCH_BYTES
            and input_.numel() % self.island_size == 0
            and input_.is_contiguous()
        ):
            flat = input_.view(-1)
            shard = torch.empty(
                flat.numel() // self.island_size,
                dtype=input_.dtype,
                device=input_.device,
            )
            self.nccl_island.reduce_scatter(shard, flat)
            self.nccl_pair.all_reduce(shard, shard)
            out = torch.empty_like(input_)
            self.nccl_island.all_gather(out.view(-1), shard)
            return out
        if (
            self.ca_island is not None
            and self.ca_pair is not None
            and nbytes < _SWITCH_BYTES
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
        if self.nccl_island is not None and self.nccl_pair is not None:
            self.nccl_island.destroy()
            self.nccl_pair.destroy()
        self.nccl_island = self.nccl_pair = None
        self.ca_island = self.ca_pair = None
        for group in self.groups:
            dist.destroy_process_group(group)
        self.groups = []
        self.disabled = True
