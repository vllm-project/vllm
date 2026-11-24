# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests that each node prefetches what it reads, and how ranks read it."""

import time
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
                ranges, node_bytes, _ = weight_utils._node_prefetch_ranges(files, ids)
            weight_utils._prefetch_all_checkpoints(files, 1, ranges=ranges)
            needed.update(b for p in files for b in tensor_bytes(p, ids))
        # The page cache is per node: each node must prefetch exactly the bytes
        # its ranks read, no more and no less, and estimate them exactly.
        assert prefetched == needed, f"node {node}"
        if ids is not None:
            assert sum(node_bytes) == len(needed), f"node {node}"


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

    def record(*args):
        calls.append(args)
        return read(*args)

    monkeypatch.setattr(weight_utils, "_read_safetensors_file", record)
    dict(weight_utils.safetensors_weights_iterator([path], False, strategy))
    # Under TP/PP with prefetch, the loader copies its slices from the cached
    # mapping; without prefetch, reading whole tensors beats cold faults.
    assert bool(calls) == reads


@pytest.mark.parametrize(
    "contents,expected", [(["600000", "900000"], 600.0), (["300s\n"], 300.0)]
)
def test_lustre_lock_max_age(monkeypatch, tmp_path, contents, expected):
    paths = []
    for i, text in enumerate(contents):
        paths.append(tmp_path / f"lru_max_age{i}")
        paths[-1].write_text(text)
    monkeypatch.setattr(
        weight_utils.glob,
        "glob",
        lambda p: [str(x) for x in paths] if "sys" in p else [],
    )
    assert weight_utils._lustre_lock_max_age_s() == expected


@pytest.mark.parametrize(
    "env,fs_type,max_age,expected",
    [
        (None, "lustre", 600.0, 150.0),
        (None, "lustre", None, 150.0),
        (None, "nfs4", 600.0, 0.0),
        ("90", "nfs4", 600.0, 90.0),
        ("0", "lustre", 600.0, 0.0),
    ],
)
def test_prefetch_lead(monkeypatch, env, fs_type, max_age, expected):
    if env is None:
        monkeypatch.delenv("VLLM_SAFETENSORS_PREFETCH_LEAD_S", raising=False)
    else:
        monkeypatch.setenv("VLLM_SAFETENSORS_PREFETCH_LEAD_S", env)
    monkeypatch.setattr(weight_utils, "_lustre_lock_max_age_s", lambda: max_age)
    assert weight_utils._prefetch_lead_s(fs_type) == expected


def test_pacer_allows_files_within_the_lead(monkeypatch):
    now = [0.0]
    monkeypatch.setattr(weight_utils.time, "monotonic", lambda: now[0])
    # Ranks load half of each file, the node prefetches all of it.
    pacer = weight_utils._PrefetchPacer([100] * 10, [50] * 10, lead_s=2)
    # Unpaced until the loader's rate is known.
    assert pacer.unpaced_s == 0.4 and pacer.allowed(9)
    now[0] = 0.5
    assert pacer.allowed(0) and not pacer.allowed(1)
    now[0] = 1.0
    pacer.start_file(1)
    pacer.add(25)  # halfway through file 1: 150 bytes in 1 s
    pacer.sync()
    assert pacer.position() == 1.5
    # 300 bytes of lead: files starting up to 450
    assert [pacer.allowed(i) for i in range(6)] == [True] * 5 + [False]
    now[0] = 3.0  # the loader stalls: 150 bytes in 3 s, 100 bytes of lead
    assert pacer.allowed(2) and not pacer.allowed(3)
    pacer.min_lead_bytes = 300
    assert pacer.allowed(4) and not pacer.allowed(5)
    assert pacer.passed(0) and not pacer.passed(1)


class _FakeStore:
    def __init__(self):
        self.values: dict[str, int] = {}

    def add(self, key, value):
        self.values[key] = self.values.get(key, 0) + value
        return self.values[key]


def test_pacer_follows_the_node_fastest_loader(monkeypatch):
    now = [0.0]
    monkeypatch.setattr(weight_utils.time, "monotonic", lambda: now[0])
    store = _FakeStore()
    pacers = []
    for rank in range(2):
        pacer = weight_utils._PrefetchPacer([100] * 10, [100] * 10, lead_s=1)
        pacer.store, pacer.keys = store, ["r0", "r1"]
        pacer.key = pacer.keys[rank]
        pacers.append(pacer)
    fast, slow = pacers
    now[0] = 10.0
    fast.start_file(5)
    slow.start_file(1)
    for pacer in (fast, slow, fast):
        pacer.sync()
    # The slow rank prefetches for the fast one (50 bytes/s of lead) and keeps
    # the files it has not read itself.
    assert slow.allowed(5) and not slow.allowed(6)
    assert slow.passed(0) and not slow.passed(1) and not fast.passed(1)
    slow.finish()
    slow.sync()
    fast.sync()
    assert fast.passed(4) and not fast.passed(5)


def test_paced_prefetch_follows_the_loader(monkeypatch, tmp_path):
    files = [str(tmp_path / f"model-{i}.safetensors") for i in range(4)]
    prefetched: list[str] = []
    monkeypatch.setattr(
        weight_utils, "_prefetch_checkpoint", lambda p, *a: prefetched.append(p)
    )
    pacer = weight_utils._PrefetchPacer([1] * 4, [1] * 4, lead_s=1e-9)
    weight_utils._prefetch_all_checkpoints(files, 2, pacer=pacer)

    def wait_for(n):
        deadline = time.monotonic() + 10
        while len(prefetched) < n and time.monotonic() < deadline:
            time.sleep(0.01)
        time.sleep(0.5)  # nothing more arrives
        return prefetched

    assert wait_for(1) == files[:1]
    for i in range(1, 4):
        pacer.start_file(i)
        assert wait_for(i + 1) == files[: i + 1]
    pacer.finish()


def test_iterator_reports_progress_and_finishes(monkeypatch, tmp_path):
    monkeypatch.setenv("VLLM_SAFETENSORS_PREFETCH_LEAD_S", "60")
    pacers = []
    monkeypatch.setattr(
        weight_utils,
        "_prefetch_all_checkpoints",
        lambda *a, pacer=None, **k: pacers.append(pacer),
    )
    monkeypatch.setattr(weight_utils, "_rank_reads_whole_tensors", lambda: True)
    files = []
    for i in range(3):
        files.append(str(tmp_path / f"model-{i}.safetensors"))
        save_file({f"w{i}.a": torch.zeros(64), f"w{i}.b": torch.zeros(64)}, files[-1])
    it = weight_utils.safetensors_weights_iterator(files, False, "prefetch")
    next(it)
    next(it)
    next(it)
    (pacer,) = pacers
    assert pacer.loaded == (1, 256) and not pacer.done
    # A loader that stops early still releases the prefetch.
    it.close()
    assert pacer.done and pacer.position() == 3
