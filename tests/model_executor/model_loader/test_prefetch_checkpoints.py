# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests that the ranks of each node prefetch what the node will read."""

from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

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


class _InlineThread:
    def __init__(self, target, daemon):
        self.target = target

    def start(self):
        self.target()


@pytest.mark.parametrize("ep", [False, True])
@pytest.mark.parametrize("num_nodes,ranks_per_node", [(1, 1), (1, 4), (2, 4), (4, 2)])
def test_each_node_prefetches_what_it_reads(
    monkeypatch, tmp_path, num_nodes, ranks_per_node, ep
):
    world_size = num_nodes * ranks_per_node
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: world_size > 1)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: world_size)
    monkeypatch.setattr(
        weight_utils, "get_node_count", lambda: num_nodes, raising=False
    )
    monkeypatch.setattr(
        weight_utils, "threading", SimpleNamespace(Thread=_InlineThread)
    )
    files = [str(tmp_path / f) for f in TENSORS]
    for i, (path, names) in enumerate(zip(files, TENSORS.values())):
        save_file({n: torch.zeros(i + 1) for n in names}, path)

    def tensor_bytes(path, ids=None):
        with open(path, "rb") as f:
            offsets = weight_utils._tensor_offsets(f).items()
        return {
            (path, b)
            for n, (start, end) in offsets
            if not should_skip_weight(n, ids)
            for b in range(start, end)
        }

    prefetched: set[tuple[str, int]] = set()

    def fake_prefetch(path, block_size, ranges=None):
        if ranges is None:
            prefetched.update(tensor_bytes(path))
        else:
            prefetched.update((path, b) for o, n in ranges for b in range(o, o + n))

    monkeypatch.setattr(weight_utils, "_prefetch_checkpoint", fake_prefetch)
    for node in range(num_nodes):
        prefetched.clear()
        needed: set[tuple[str, int]] = set()
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
            ranges = None
            if ids is not None:
                ranges, node_bytes = weight_utils._node_prefetch_ranges(files, ids)
            weight_utils._prefetch_all_checkpoints(files, 1, ranges=ranges)
            needed.update(b for p in files for b in tensor_bytes(p, ids))
        # The page cache is per node: each node must prefetch exactly the bytes
        # its ranks read, no more and no less, and estimate them exactly.
        assert prefetched == needed, f"node {node}"
        if ids is not None:
            assert node_bytes == len(needed), f"node {node}"


def test_prefetch_checkpoint_sends_the_ranges(tmp_path, monkeypatch):
    path = tmp_path / "model.safetensors"
    path.write_bytes(bytes(5000))
    sent: list[tuple[int, int]] = []

    def fake_sendfile(out_fd, in_fd, offset, count):
        sent.append((offset, count))
        return count

    monkeypatch.setattr(weight_utils.os, "sendfile", fake_sendfile)
    weight_utils._prefetch_checkpoint(str(path), ranges=[[100, 2003], [3000, 1004]])
    assert sent == [(100, 2003), (3000, 1004)]
    sent.clear()
    weight_utils._prefetch_checkpoint(str(path))
    assert sent == [(0, 5000)]
