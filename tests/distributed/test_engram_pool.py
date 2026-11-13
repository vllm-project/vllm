# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Engram pool fetches across 4 GPUs, eager and replayed from CUDA graphs."""

import pytest
import torch
import torch.multiprocessing as mp

from vllm.distributed.engram_pool import EngramPool, PeerGroup, verbs
from vllm.platforms import current_platform
from vllm.utils.vmm_driver import imex_channel_available

WORLD_SIZE, ROWS, WORDS, TABLES, K = 4, 1001, 9, 2, 3


def _expected(ids: torch.Tensor) -> torch.Tensor:
    """Word w of row r of table t holds (t << 40) | (r << 8) | w; dead ids are 0."""
    table = torch.arange(TABLES, device=ids.device).view(1, TABLES, 1, 1)
    words = torch.arange(WORDS, device=ids.device)
    rows = (table << 40) | (ids[..., None] << 8) | words
    return rows.masked_fill(((ids < 0) | (ids >= ROWS))[..., None], 0)


def _worker(rank: int, store: str, transport: str, host: bool) -> None:
    torch.accelerator.set_device_index(rank)
    group = PeerGroup(WORLD_SIZE, store)
    pool = EngramPool(group, transport, host, 64 * TABLES * K, WORDS * 8)
    tables = [pool.table(ROWS, WORDS * 8, f"t{t}") for t in range(TABLES)]
    for t, table in enumerate(tables):
        rows = torch.arange(ROWS)[:, None]
        table.load(0, (t << 40) | (rows << 8) | torch.arange(WORDS), 0)
    pool.ready()
    generator = torch.Generator(device="cuda").manual_seed(rank)

    def ids(num_tokens: int) -> torch.Tensor:
        shape = (num_tokens, TABLES, K)
        return torch.randint(-2, ROWS + 2, shape, device="cuda", generator=generator)

    for num_tokens in (1, 64):
        batch = ids(num_tokens)
        rows = pool.lookup(tables, batch).view(torch.int64)
        assert torch.equal(rows.view(*batch.shape, WORDS), _expected(batch))
    # Replays carry no host state: each one fetches the ids now in `static`.
    static = ids(64)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        pool.lookup(tables, static)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        rows = pool.prefetch(tables, static).wait().view(torch.int64)
    for _ in range(3):
        static.copy_(ids(64))
        graph.replay()
        assert torch.equal(rows.view(*static.shape, WORDS), _expected(static))
    group.barrier()
    pool.close()


@pytest.mark.parametrize(
    "transport,host", [("nvlink", False), ("rdma", False), ("rdma", True)]
)
def test_engram_pool_fetch(transport, host, tmp_path):
    if not current_platform.is_cuda():
        pytest.skip("The Engram pool is CUDA-only")
    if torch.accelerator.device_count() < WORLD_SIZE:
        pytest.skip(f"Need {WORLD_SIZE} GPUs")
    if transport == "nvlink" and not imex_channel_available():
        pytest.skip("NVLink transport needs an IMEX channel")
    if transport == "rdma" and not verbs.active_devices():
        pytest.skip("RDMA transport needs an active RDMA device")
    store = str(tmp_path / "store")
    mp.spawn(_worker, args=(store, transport, host), nprocs=WORLD_SIZE)
