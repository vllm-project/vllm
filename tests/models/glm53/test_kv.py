# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM-5.3 decode-segment KV hygiene (vllm/models/glm53/nvidia/kv.py)."""

from __future__ import annotations

import threading

import pytest
import torch

from vllm.models.glm53.nvidia import kv as glm53_kv
from vllm.models.glm53.nvidia.kv import Glm53KvHygiene, PendingRows, Rows

BLOCK = 64
BLOCKS = 6
WORLD = 8


def make_caches(layers: int, seed: int) -> dict[str, torch.Tensor]:
    """One block-outer buffer holding every layer's MLA and index pages, as vLLM's
    views.
    """
    page_mla, page_index = BLOCK * 576, BLOCK * 132
    stride = layers * (page_mla + page_index)
    generator = torch.Generator().manual_seed(seed)
    buffer = torch.randint(
        0, 256, (BLOCKS * stride,), dtype=torch.uint8, generator=generator
    )
    caches = {}
    for layer in range(layers):
        base = layer * (page_mla + page_index)
        caches[f"model.layers.{layer}.self_attn.attn"] = torch.as_strided(
            buffer, (BLOCKS, BLOCK, 576), (stride, 576, 1), base
        )
        caches[f"model.layers.{layer}.self_attn.indexer.k_cache"] = torch.as_strided(
            buffer, (BLOCKS, BLOCK, 132), (stride, 132, 1), base + page_mla
        )
    return caches


class Group:
    """One rank's handle on a simulated TP group (threads, one per rank)."""

    def __init__(self, shared: dict, rank: int) -> None:
        self.shared, self.rank_in_group, self.world_size = shared, rank, WORLD

    def broadcast(self, input_: torch.Tensor, src: int = 0) -> torch.Tensor:
        if self.rank_in_group == src:
            self.shared["data"] = input_.clone()
        self.shared["barrier"].wait()
        input_.copy_(self.shared["data"])
        self.shared["barrier"].wait()
        return input_


def run_ranks(work) -> list:
    """Run ``work(rank, group)`` on eight threads; return the per-rank results."""
    shared = {"barrier": threading.Barrier(WORLD)}
    results: list = [None] * WORLD
    errors: list[BaseException] = []

    def body(rank: int) -> None:
        try:
            results[rank] = work(rank, Group(shared, rank))
        except BaseException as error:  # noqa: BLE001 - reported below
            errors.append(error)
            shared["barrier"].abort()

    threads = [threading.Thread(target=body, args=(rank,)) for rank in range(WORLD)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if errors:
        raise errors[0]
    return results


def test_planes_and_default_writers(geometry) -> None:
    caches = make_caches(2, 0)
    caches["model.layers.78.self_attn.indexer.k_cache"] = caches.pop(
        "model.layers.1.self_attn.indexer.k_cache"
    )
    hygiene = Glm53KvHygiene.from_caches(caches, geometry, rank=4, block_size=BLOCK)
    assert sorted(hygiene.planes) == [
        (0, "index_key"),
        (0, "index_scale"),
        (0, "latent"),
        (0, "rope"),
        (1, "latent"),
        (1, "rope"),
        (78, "index_key"),
        (78, "index_scale"),
    ]
    # Layer 78 is the drafter's: written on every rank.
    assert hygiene.transfers == ((0, ((0, "index_key"), (0, "index_scale"))),)
    # The scale plane sits after the block's 64 keys.
    rows = Rows.of_positions([3], 5, 6, block_size=BLOCK, device="cpu")
    cache = caches["model.layers.0.self_attn.indexer.k_cache"]
    raw = torch.as_strided(cache, (BLOCKS, BLOCK * 132), (cache.stride(0), 1))
    assert torch.equal(
        rows.gather(hygiene.planes[(0, "index_scale")])[0],
        raw[3, BLOCK * 128 + 20 : BLOCK * 128 + 24],
    )
    assert torch.equal(
        rows.gather(hygiene.planes[(0, "index_key")])[0], raw[3, 5 * 128 : 6 * 128]
    )


def test_mla_rows_are_576_bytes(geometry) -> None:
    caches = make_caches(1, 2)
    mla = caches["model.layers.0.self_attn.attn"]
    planes = glm53_kv.mla_planes(mla, geometry, BLOCK)
    rows = Rows.of_positions([2], 7, 8, block_size=BLOCK, device="cpu")
    assert torch.equal(rows.gather(planes["rope"])[0], mla[2, 7, 512:576])
    # Not a layer-outer pool: the kernels address consecutive rows.
    with pytest.raises(ValueError):
        glm53_kv.check_mla_pool(mla, geometry, BLOCK)
    glm53_kv.check_mla_pool(mla.contiguous(), geometry, BLOCK)
    # The removed fp8_ds_mla rows (656 bytes) are refused.
    with pytest.raises(ValueError):
        glm53_kv.mla_planes(
            torch.zeros((2, BLOCK, 656), dtype=torch.uint8), geometry, BLOCK
        )
    with pytest.raises(ValueError):
        glm53_kv.check_mla_pool(
            torch.zeros((2, BLOCK, 656), dtype=torch.uint8), geometry, BLOCK
        )


@pytest.mark.parametrize("table", ["list", "device row"])
def test_release_completes_the_index_keys(table: str, geometry) -> None:
    blocks = [4, 1, 3]
    ids = (
        blocks if table == "list" else torch.tensor([*blocks, 0, 0], dtype=torch.int32)
    )

    def work(rank: int, group: Group) -> dict:
        caches = make_caches(2, 7)  # identical on every rank ...
        hygiene = Glm53KvHygiene.from_caches(
            caches, geometry, rank=rank, block_size=BLOCK
        )
        if rank in (
            0,
            1,
            2,
        ):  # ... except the selector ranks' index keys of positions [70, 140)
            rows = Rows.of_positions(blocks, 70, 140, block_size=BLOCK, device="cpu")
            for key in ((0, "index_key"), (1, "index_scale")):
                plane = hygiene.planes[key]
                rows.scatter(plane, torch.full((70, plane.width), 9, dtype=torch.uint8))
        snapshot = {name: cache.clone() for name, cache in caches.items()}
        hygiene.pending.decoded("r", 60, 150)
        hygiene.release("r", ids, group)
        return {name: (snapshot[name], cache) for name, cache in caches.items()}

    results = run_ranks(work)
    for rank, caches in enumerate(results):
        for name, (old, new) in caches.items():
            if rank < 3 or name.endswith(".attn"):
                assert torch.equal(old, new), (rank, name)
            else:  # the index keys differed from rank 0's before the release
                assert not torch.equal(old, results[0][name][0]), (rank, name)
            assert torch.equal(new, results[0][name][1]), (rank, name)


def test_rows_of_a_device_row_match_the_list() -> None:
    row = torch.tensor([5, 2, 4, 0], dtype=torch.int32)
    for start, stop in ((0, 1), (63, 130), (100, 192)):
        a = Rows.of_positions([5, 2, 4, 0], start, stop, block_size=BLOCK, device="cpu")
        b = Rows.of_positions(row, start, stop, block_size=BLOCK, device="cpu")
        assert torch.equal(a.blocks, b.blocks) and torch.equal(a.rows, b.rows)
    with pytest.raises(ValueError):
        Rows.of_positions(row, 0, 4 * BLOCK + 1, block_size=BLOCK, device="cpu")


def test_pending_rows() -> None:
    pending = PendingRows()
    pending.decoded("b", -8, 4)
    pending.decoded("a", 10, 14)
    pending.decoded("a", 6, 12)
    pending.decoded("c", 5, 5)
    assert pending.take("a") == [("a", 6, 14)]
    pending.decoded("a", 20, 24)
    assert pending.take() == [("a", 20, 24), ("b", 0, 4)]
    assert not pending


def test_complete_clamps_to_the_blocks(geometry) -> None:
    def work(rank: int, group: Group) -> int:
        caches = make_caches(1, 3)
        hygiene = Glm53KvHygiene.from_caches(
            caches, geometry, rank=rank, block_size=BLOCK
        )
        hygiene.complete([2], 0, 10 * BLOCK, group)  # one block: positions [0, 64)
        hygiene.complete([2], 70, 80, group)  # past the blocks: nothing, no collective
        return 0

    run_ranks(work)
    with pytest.raises(ValueError):
        Rows.of_positions([1], 0, BLOCK + 1, block_size=BLOCK, device="cpu")
