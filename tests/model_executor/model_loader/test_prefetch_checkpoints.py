# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for splitting checkpoint prefetch among the ranks of each node."""

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.model_loader import weight_utils

FILES = [f"model-{i:05d}-of-00011.safetensors" for i in range(11)]


class _InlineThread:
    def __init__(self, target, daemon):
        self.target = target

    def start(self):
        self.target()


def _prefetched_files(monkeypatch, files):
    prefetched = []
    monkeypatch.setattr(
        weight_utils, "threading", SimpleNamespace(Thread=_InlineThread)
    )
    monkeypatch.setattr(
        weight_utils, "_prefetch_checkpoint", lambda p, _: prefetched.append(p)
    )
    weight_utils._prefetch_all_checkpoints(files, num_prefetch_threads=1)
    return prefetched


@pytest.mark.parametrize("num_nodes,ranks_per_node", [(1, 1), (1, 4), (2, 4), (4, 2)])
def test_each_node_prefetches_all_files(monkeypatch, num_nodes, ranks_per_node):
    world_size = num_nodes * ranks_per_node
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: world_size > 1)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: world_size)
    monkeypatch.setattr(
        weight_utils, "get_node_count", lambda: num_nodes, raising=False
    )
    for node in range(num_nodes):
        node_files = []
        for local_rank in range(ranks_per_node):
            rank = node * ranks_per_node + local_rank
            world = SimpleNamespace(
                world_size=world_size, rank=rank, local_rank=local_rank
            )
            monkeypatch.setattr(torch.distributed, "get_rank", lambda r=rank: r)
            monkeypatch.setattr(weight_utils, "get_world_group", lambda w=world: w)
            node_files += _prefetched_files(monkeypatch, FILES)
        # The page cache is per node: each node must read the whole checkpoint.
        assert sorted(node_files) == FILES, f"node {node}"
