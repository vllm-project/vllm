# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests that each node prefetches what it reads, and how ranks read it."""

from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file, save_file

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
            header = weight_utils._read_header(f).items()
        return {
            (path, b)
            for n, t in header
            if not should_skip_weight(n, ids)
            for b in range(*t["data_offsets"])
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


def test_read_matches_safe_open(tmp_path):
    tensors = {
        "embed.weight": torch.randn(7, 5, dtype=torch.bfloat16),
        "layers.0.mlp.experts.0.down_proj.weight": torch.randint(-9, 9, (4, 3)).char(),
        "layers.0.mlp.experts.1.down_proj.weight": torch.randn(4, 3),
        "layers.0.mlp.experts.1.down_proj.weight_scale": torch.tensor(2.0),
        "layers.0.empty": torch.empty(0, 3),
    }
    path = str(tmp_path / "model.safetensors")
    save_file(tensors, path)
    loaded = dict(weight_utils.safetensors_weights_iterator([path], False, "lazy", {0}))
    expected = {
        k: v for k, v in load_file(path).items() if not should_skip_weight(k, {0})
    }
    assert loaded.keys() == expected.keys()
    for name, tensor in loaded.items():
        assert tensor.dtype == expected[name].dtype
        assert torch.equal(tensor, expected[name]), name


@pytest.mark.parametrize(
    "strategy,whole,reads",
    [("prefetch", True, True), ("prefetch", False, False), ("lazy", False, True)],
)
def test_read_whole_tensors_when_the_rank_uses_them(
    monkeypatch, tmp_path, strategy, whole, reads
):
    path = str(tmp_path / "model.safetensors")
    save_file({"a": torch.randn(64, 8)}, path)
    monkeypatch.setattr(weight_utils, "_rank_reads_whole_tensors", lambda: whole)
    monkeypatch.setattr(weight_utils, "_prefetch_all_checkpoints", lambda *a, **k: None)
    calls = []
    read = weight_utils._read_safetensors_file
    monkeypatch.setattr(
        weight_utils, "_read_safetensors_file", lambda *a: calls.append(a) or read(*a)
    )
    dict(weight_utils.safetensors_weights_iterator([path], False, strategy))
    # Under TP/PP with prefetch, the loader copies its slices from the cached
    # mapping; without prefetch, reading whole tensors beats cold faults.
    assert bool(calls) == reads
