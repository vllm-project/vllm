# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NVFP4 indexer decode CP at TP4: each rank's top-k must equal an exact fp32
top-15 over the full-context scores, across repeated runs (sentinel reset)."""

import os

import pytest
import torch
import torch.multiprocessing as mp

from vllm.platforms import current_platform

if not current_platform.is_cuda():
    pytest.skip("MiniMax M3 NVFP4 indexer requires CUDA.", allow_module_level=True)

PAGE = 128
SLOTS = 8
H = 4


def _cache(num_pages: int, gen: torch.Generator) -> torch.Tensor:
    kv = torch.randint(
        0, 256, (num_pages, PAGE, 72), dtype=torch.uint8, device="cuda", generator=gen
    )
    # Positive normal E4M3 block scales keep every score finite.
    s = torch.rand(num_pages, PAGE * 8, device="cuda", generator=gen) * 1.75 + 0.25
    kv.view(num_pages, -1)[:, 8192:] = s.to(torch.float8_e4m3fn).view(torch.uint8)
    return kv


def _reference(scores, seq_lens, query_len, head) -> torch.Tensor:
    num_reqs = seq_lens.shape[0]
    out = torch.full((num_reqs * query_len, 16), -1, dtype=torch.int32)
    for b in range(num_reqs):
        for i in range(query_len):
            t = b * query_len + i
            own = (int(seq_lens[b]) - query_len + i) // PAGE
            if own + 1 <= 16:
                out[t, : own + 1] = torch.arange(own + 1)
                continue
            row = scores[b, (SLOTS - query_len + i) * H + head, :own]
            out[t, :15] = torch.sort(-row, stable=True).indices[:15].to(torch.int32)
            out[t, 15] = own
    return out


def _run(rank: int, port: int, query_len: int, seq_lens_list: list[int]) -> None:
    from vllm.distributed import (
        init_distributed_environment,
        initialize_model_parallel,
    )
    from vllm.models.minimax_m3.nvidia.indexer_nvfp4_decode_cp import (
        Nvfp4IndexerDecodeCP,
    )

    torch.accelerator.set_device_index(rank)
    init_distributed_environment(
        world_size=4,
        rank=rank,
        local_rank=rank,
        distributed_init_method=f"tcp://127.0.0.1:{port}",
    )
    initialize_model_parallel(tensor_model_parallel_size=4)

    gen = torch.Generator(device="cuda").manual_seed(0)  # same on every rank
    seq_lens = torch.tensor(seq_lens_list, dtype=torch.int32, device="cuda")
    num_reqs = seq_lens.shape[0]
    pages = (int(seq_lens.max()) + PAGE - 1) // PAGE
    kv = _cache(num_reqs * pages + 8, gen)
    # Sized to the longest context, as vLLM's block table is: the longest
    # request then owns the last page of its rank's compact table.
    block_table = (
        torch.randperm(num_reqs * pages, device="cuda", generator=gen)
        .to(torch.int32)
        .view(num_reqs, pages)
    )
    rows = num_reqs * query_len

    cp = Nvfp4IndexerDecodeCP(max_reqs=num_reqs, device=torch.device("cuda", rank))
    layers = [cp.new_layer() for _ in range(2)]
    plan = cp.plan(block_table, seq_lens)

    from vllm.third_party.fmha_sm100.sparse import BatchDecodeIndexerQ8KV4Wrapper

    full = BatchDecodeIndexerQ8KV4Wrapper(num_heads=H)
    full.plan(block_table, seq_lens)
    for _ in range(3):
        for layer in layers:
            q = (torch.randn(rows, H, 128, device="cuda", generator=gen) * 0.5).to(
                torch.float8_e4m3fn
            )
            out = torch.full((rows, 1, 16), -7, dtype=torch.int32, device="cuda")
            cp.forward(
                layer, plan, q[:, rank : rank + 1].contiguous(), kv, out, query_len
            )
            q_tile = torch.zeros(
                num_reqs, SLOTS, H, 128, dtype=torch.float8_e4m3fn, device="cuda"
            )
            q_tile[:, SLOTS - query_len :] = q.view(num_reqs, query_len, H, 128)
            # NaN in the columns the kernel leaves alone, so a selection
            # that reads one cannot match the reference.
            full._scores.fill_(float("nan"))
            scores = full._run_scores(q_tile.view(-1, H, 128), kv)
            expected = _reference(scores.cpu(), seq_lens.cpu(), query_len, rank)
            torch.testing.assert_close(out[:, 0].cpu(), expected, rtol=0, atol=0)


def _worker(rank: int, port: int, query_len: int, seq_lens_list: list[int]) -> None:
    from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config

    config = VllmConfig(parallel_config=ParallelConfig(tensor_parallel_size=4))
    with set_current_vllm_config(config):
        _run(rank, port, query_len, seq_lens_list)


@pytest.mark.skipif(
    torch.accelerator.device_count() < 4
    or not (
        current_platform.is_device_capability(100)
        or current_platform.is_device_capability(103)
    ),
    reason="Needs 4 SM100/SM103 GPUs.",
)
@pytest.mark.parametrize("query_len", [1, 4, 8])
def test_nvfp4_indexer_decode_cp_matches_exact_topk(query_len: int):
    # Trivial rows (<= 16 pages), page boundaries, and long contexts.
    seq_lens = [9, 2049, 2048 + query_len, 20000, 131072, 300001]
    port = 29500 + (os.getpid() % 1000)
    mp.spawn(_worker, args=(port, query_len, seq_lens), nprocs=4, join=True)
