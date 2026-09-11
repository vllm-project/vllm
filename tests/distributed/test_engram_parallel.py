# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Collective coverage of Engram's head-to-token redistribution."""

from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from vllm.models.deepseek_v4_1.common.engram_parallel import exchange_heads_for_tokens


def _exchange_worker(rank: int, world_size: int, rendezvous: str):
    dist.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=30),
    )
    try:
        for tokens, heads in [(0, 23), (1, 1), (3, 7), (64, 24), (65, 23)]:
            dim = 5
            local_heads = (heads + world_size - 1) // world_size
            chunk = (tokens + world_size - 1) // world_size
            full = torch.arange(tokens * heads * dim, dtype=torch.float32).reshape(
                tokens, heads, dim
            )
            padded = torch.nn.functional.pad(
                full, (0, 0, 0, local_heads * world_size - heads)
            )
            local = padded[:, rank * local_heads : (rank + 1) * local_heads]

            def exchange(rows):
                received = torch.empty_like(rows)
                dist.all_to_all_single(received, rows.contiguous())
                return received

            actual = exchange_heads_for_tokens(local, world_size, heads, exchange)
            expected = torch.nn.functional.pad(
                full, (0, 0, 0, 0, 0, chunk * world_size - tokens)
            )[rank * chunk : (rank + 1) * chunk]
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("world_size", [2, 4])
def test_all_to_all_preserves_token_and_head_order(world_size, tmp_path):
    mp.spawn(
        _exchange_worker,
        args=(world_size, (tmp_path / "rendezvous").as_uri()),
        nprocs=world_size,
        join=True,
    )


def _cuda_exchange_worker(rank, tp_size, port):
    from tests.utils import init_test_distributed_environment
    from vllm.distributed import cleanup_dist_env_and_memory, get_tp_group
    from vllm.models.deepseek_v4_1.common.engram import Engram

    torch.accelerator.set_device_index(rank)
    init_test_distributed_environment(tp_size, 4 // tp_size, rank, str(port), rank)
    group = get_tp_group()
    local_rank = group.rank_in_group
    group_offset = group.ranks[0] * 7

    def full_rows(tokens, heads, dtype, offset=0):
        values = torch.arange(tokens * heads * 256, device="cuda")
        return (
            ((values % 97) + group_offset + offset).to(dtype).view(tokens, heads, 256)
        )

    def module_for(full, strided):
        tokens, heads, dim = full.shape
        local_heads = (heads + tp_size - 1) // tp_size
        padded = torch.nn.functional.pad(full, (0, 0, 0, -heads % tp_size))
        local = padded[:, local_rank * local_heads : (local_rank + 1) * local_heads]
        rows = (
            torch.empty(
                tokens,
                local_heads,
                dim * (2 if strided else 1),
                dtype=full.dtype,
                device="cuda",
            )[..., ::2]
            if strided
            else torch.empty_like(local, memory_format=torch.contiguous_format)
        )
        rows.copy_(local)
        module = Engram.__new__(Engram)
        torch.nn.Module.__init__(module)
        module.embed_tokens = SimpleNamespace(tp_size=tp_size, n_hash_cols=heads)
        module.use_sequence_parallel = True
        module.staged_rows = rows
        return module, torch.empty(tokens, heads, dtype=torch.int32, device="cuda")

    def expected(full):
        chunk = (len(full) + tp_size - 1) // tp_size
        return torch.nn.functional.pad(full, (0, 0, 0, 0, 0, -len(full) % tp_size))[
            local_rank * chunk : (local_rank + 1) * chunk
        ]

    try:
        for tokens in sorted(
            {
                0,
                1,
                tp_size - 1,
                tp_size,
                tp_size + 1,
                63,
                64,
                65,
                127,
                128,
                129,
                511,
                512,
                513,
            }
        ):
            for heads in (23, 24):
                for dtype in (torch.bfloat16, torch.float32):
                    for strided in (False, True):
                        full = full_rows(tokens, heads, dtype)
                        module, ids = module_for(full, strided)
                        torch.testing.assert_close(
                            module.embed(ids), expected(full), rtol=0, atol=0
                        )

        # Two captures own separate buffers; ordered streams must not reuse
        # each other's storage when fresh input is supplied on every replay.
        captures = []
        for offset in (0, 13):
            full = full_rows(65, 23, torch.bfloat16, offset)
            module, ids = module_for(full, True)
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    module.embed(ids)
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                output = module.embed(ids)
            captures.append((module, graph, output, stream, offset))
        for iteration in range(100):
            for module, graph, output, stream, offset in captures:
                full = full_rows(65, 23, torch.bfloat16, offset + iteration % 17)
                fresh, _ = module_for(full, True)
                module.staged_rows.copy_(fresh.staged_rows)
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    graph.replay()
                torch.cuda.current_stream().wait_stream(stream)
                torch.testing.assert_close(output, expected(full), rtol=0, atol=0)
        torch.accelerator.synchronize()
        del captures, graph, output
    finally:
        cleanup_dist_env_and_memory()


@pytest.mark.distributed(num_gpus=4)
@pytest.mark.skipif(
    torch.accelerator.device_count() < 4, reason="Requires four CUDA GPUs"
)
@pytest.mark.parametrize("tp_size", [1, 2, 4])
def test_engram_nondefault_groups_and_changing_graph_inputs(tp_size):
    """The production embed path preserves rows on every TP group and replay."""
    from vllm.utils.network_utils import get_open_port

    mp.spawn(_cuda_exchange_worker, args=(tp_size, get_open_port()), nprocs=4)
