# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests that the ranks of each node prefetch what the node will read."""

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.model_loader import weight_utils
from vllm.model_executor.model_loader.ep_weight_filter import (
    compute_local_expert_ids,
    should_skip_weight,
)

NUM_EXPERTS = 8
# One MoE layer per file: dense weights, expert weights and expert scales.
TENSORS = {
    f"model-{i:05d}-of-00011.safetensors": [
        f"layers.{i}.self_attn.qkv_proj.weight",
        *(f"layers.{i}.mlp.experts.{e}.down_proj.weight" for e in range(NUM_EXPERTS)),
        *(f"layers.{i}.mlp.experts.{e}.down_proj.weight_scale" for e in range(4)),
    ]
    for i in range(11)
}
FILES = list(TENSORS)


class _InlineThread:
    def __init__(self, target, daemon):
        self.target = target

    def start(self):
        self.target()


@pytest.mark.parametrize("ep", [False, True])
@pytest.mark.parametrize("num_nodes,ranks_per_node", [(1, 1), (1, 4), (2, 4), (4, 2)])
def test_each_node_prefetches_what_it_reads(monkeypatch, num_nodes, ranks_per_node, ep):
    world_size = num_nodes * ranks_per_node
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: world_size > 1)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: world_size)
    monkeypatch.setattr(
        weight_utils, "get_node_count", lambda: num_nodes, raising=False
    )
    monkeypatch.setattr(
        weight_utils, "threading", SimpleNamespace(Thread=_InlineThread)
    )
    prefetched: set[str] = set()

    def fake_prefetch(path, block_size, keep=None):
        prefetched.update(n for n in TENSORS[path] if keep is None or keep(n))

    monkeypatch.setattr(weight_utils, "_prefetch_checkpoint", fake_prefetch)
    for node in range(num_nodes):
        prefetched.clear()
        needed: set[str] = set()
        for local_rank in range(ranks_per_node):
            rank = node * ranks_per_node + local_rank
            ids = (
                compute_local_expert_ids(NUM_EXPERTS, world_size, rank) if ep else None
            )
            world = SimpleNamespace(
                world_size=world_size, rank=rank, local_rank=local_rank
            )
            monkeypatch.setattr(torch.distributed, "get_rank", lambda r=rank: r)
            monkeypatch.setattr(weight_utils, "get_world_group", lambda w=world: w)
            weight_utils._prefetch_all_checkpoints(FILES, 1, local_expert_ids=ids)
            needed.update(
                n
                for ns in TENSORS.values()
                for n in ns
                if not should_skip_weight(n, ids)
            )
        # The page cache is per node: each node must prefetch exactly what its
        # ranks read, no more and no less.
        assert prefetched == needed, f"node {node}"
