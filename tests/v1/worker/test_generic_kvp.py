# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Generic KVP: layers see every rank's pages; only new writes persist."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

import vllm.v1.worker.gpu.generic_kvp as kvp
from vllm.v1.kv_cache_interface import KVCacheGroupSpec, MLAAttentionSpec

SIZE = 2
RANK = 1
NUM_BLOCKS = 8
PAGE_TOKENS = 4


class _Layer(SimpleNamespace):
    def bind_kv_cache(self, kv_cache: torch.Tensor) -> None:
        self.kv_cache = kv_cache


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_layers_see_all_pages_and_persist_new_writes(monkeypatch):
    names = ["model.layers.0.self_attn.attn", "model.layers.1.self_attn.attn"]
    # Like the real KV cache: every layer's cache is a view into one allocation.
    backing = torch.zeros(NUM_BLOCKS, len(names), 4, 8, dtype=torch.uint8).cuda()
    kv_caches = {name: backing[:, index] for index, name in enumerate(names)}
    for index, name in enumerate(names):
        kv_caches[name][:, 0, 1] = index
        kv_caches[name][:, 0, 2] = torch.arange(NUM_BLOCKS, device="cuda")
    # Peers hold copies of this rank's allocation, tagged by their rank.
    peers = [backing if rank == RANK else backing.clone() for rank in range(SIZE)]
    for rank, peer in enumerate(peers):
        peer[:, :, 0, 0] = rank
    handle = SimpleNamespace(
        buffer_ptrs=[peer.data_ptr() for peer in peers],
        buffer_size=backing.nbytes,
        rank=RANK,
        barrier=lambda: None,
    )
    monkeypatch.setattr(kvp, "_size", SIZE)
    monkeypatch.setattr(kvp.torch.distributed, "get_rank", lambda: RANK)
    monkeypatch.setattr(kvp.torch.distributed, "new_group", lambda *a, **k: None)
    monkeypatch.setattr(kvp.symm_mem, "rendezvous", lambda *a: handle)
    layers = {name: _Layer() for name in names}
    spec = MLAAttentionSpec(
        block_size=PAGE_TOKENS, num_kv_heads=1, head_size=8, dtype=torch.uint8
    )
    config = SimpleNamespace(
        kv_cache_groups=[KVCacheGroupSpec(names, spec)], num_blocks=NUM_BLOCKS
    )
    # Requests 2 and 0 share block 3 (a full prefix block) and append to blocks
    # 5 and 6; the block tables hold each block's kernel blocks b * SIZE + r.
    req_blocks = {2: [3, 5], 0: [3, 6]}
    tables = torch.zeros(4, 8, dtype=torch.int32, device="cuda")
    num_blocks = np.zeros((1, 4), dtype=np.int32)
    for req, blocks in req_blocks.items():
        kernel_blocks = [b * SIZE + r for b in blocks for r in range(SIZE)]
        tables[req, : len(kernel_blocks)] = torch.tensor(kernel_blocks)
        num_blocks[0, req] = len(kernel_blocks)
    block_tables = SimpleNamespace(
        block_tables=[SimpleNamespace(gpu=tables)],
        num_blocks=SimpleNamespace(np=num_blocks),
        block_sizes=[PAGE_TOKENS * SIZE],
    )
    runtime = kvp.GenericKVP(config, kv_caches, layers, block_tables)

    num_computed = np.zeros(4, dtype=np.int32)
    num_computed[[2, 0]] = PAGE_TOKENS * SIZE
    batch = SimpleNamespace(
        idx_mapping_np=np.array([2, 0]), num_scheduled_tokens=np.array([3, 2])
    )
    runtime.prepare(batch, num_computed)
    scratch = layers[names[0]].kv_cache
    assert scratch is layers[names[1]].kv_cache
    assert scratch.shape == (NUM_BLOCKS * SIZE, 4, 8)
    for index, name in enumerate(names):
        runtime.acquire(name)
        for block in (3, 5, 6):
            for rank in range(SIZE):
                page = scratch[block * SIZE + rank, 0]
                assert page[:3].tolist() == [rank, index, block]
            # The layer writes every page; only new blocks' own pages persist.
            scratch[block * SIZE : (block + 1) * SIZE, 1] = 7
        runtime.release(name)
        # Releasing a layer starts pulling the next; the last ends the forward.
        assert runtime.index == (1 if index == 0 else None)
        assert kvp.get_generic_kvp() is (runtime if index == 0 else None)
    torch.accelerator.synchronize()

    for name in names:
        persistent = kv_caches[name]
        assert (persistent[[5, 6], 1] == 7).all()
        assert (persistent[3, 1] == 0).all()
        assert (persistent[:, 0, 0] == RANK).all()
