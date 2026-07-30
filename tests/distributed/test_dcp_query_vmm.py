# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Correctness and CUDA-graph tests for DCP FP8 query producer fanout."""

import os
import socket

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import vllm.envs as envs
from vllm.platforms import current_platform


def _get_free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("localhost", 0))
        return int(sock.getsockname()[1])


def _make_query_bytes(
    rank: int,
    rows: int,
    local_heads: int,
    query_dim: int,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed * 997 + rank)
    raw = torch.randint(
        0,
        256,
        (rows, local_heads, query_dim),
        generator=generator,
        device=device,
        dtype=torch.uint8,
    )
    return raw.view(torch.float8_e4m3fn)


def _reference_gather(local_query: torch.Tensor, world_size: int) -> torch.Tensor:
    rows, local_heads, query_dim = local_query.shape
    gathered_rank_major = torch.empty(
        world_size * rows,
        local_heads,
        query_dim,
        dtype=torch.uint8,
        device=local_query.device,
    )
    dist.all_gather_into_tensor(
        gathered_rank_major,
        local_query.view(torch.uint8),
    )
    return (
        gathered_rank_major.view(
            world_size,
            rows,
            local_heads,
            query_dim,
        )
        .permute(1, 0, 2, 3)
        .reshape(rows, world_size * local_heads, query_dim)
    )


def _assert_byte_equal(actual: torch.Tensor, expected: torch.Tensor) -> None:
    assert actual.dtype == torch.float8_e4m3fn
    torch.testing.assert_close(
        actual.view(torch.uint8),
        expected,
        rtol=0,
        atol=0,
    )


def _worker(rank: int, world_size: int, port: int) -> None:
    os.environ.update(
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
        RANK=str(rank),
        LOCAL_RANK=str(rank),
        WORLD_SIZE=str(world_size),
    )
    device = torch.device(f"cuda:{rank}")
    torch.accelerator.set_device_index(device)
    dist.init_process_group("nccl")
    cpu_group = dist.new_group(
        ranks=list(range(world_size)),
        backend="gloo",
    )
    from vllm.v1.attention.ops.dcp_query_vmm import (
        create_dcp_query_vmm_workspace_for_group,
    )

    local_heads = 16
    query_dim = 576
    max_rows = 128
    workspace = create_dcp_query_vmm_workspace_for_group(
        max_rows,
        local_heads,
        query_dim,
        cpu_group,
        device,
    )
    assert workspace.payload_bytes_per_rank == 4_718_592
    assert workspace.local_consumer_query.shape == (
        max_rows,
        world_size * local_heads,
        query_dim,
    )
    assert workspace.local_consumer_query.is_contiguous()
    assert workspace.peer_consumer_queries.shape == (
        world_size,
        max_rows,
        world_size * local_heads,
        query_dim,
    )

    try:
        for iteration, rows in enumerate((1, 8, 32, 64, 128)):
            local_query = _make_query_bytes(
                rank,
                rows,
                local_heads,
                query_dim,
                iteration,
                device,
            )
            expected = _reference_gather(local_query, world_size)
            fanout_targets = workspace.begin_publish(rows)
            fanout_targets.copy_(local_query.unsqueeze(0))
            workspace.finish_publish()
            actual = workspace.acquire_local_query(rows)
            torch.accelerator.synchronize()
            _assert_byte_equal(actual, expected)
            workspace.acknowledge()

        # Rank/head sentinels verify DCP rank order in the dense destination.
        rows = 8
        sentinel_bytes = torch.empty(
            rows,
            local_heads,
            query_dim,
            dtype=torch.uint8,
            device=device,
        )
        for local_head in range(local_heads):
            sentinel_bytes[:, local_head].fill_(rank * local_heads + local_head)
        fanout_targets = workspace.begin_publish(rows)
        fanout_targets.copy_(sentinel_bytes.view(torch.float8_e4m3fn).unsqueeze(0))
        workspace.finish_publish()
        actual = workspace.acquire_local_query(rows)
        torch.accelerator.synchronize()
        expected_heads = torch.arange(
            world_size * local_heads,
            dtype=torch.uint8,
            device=device,
        )
        torch.testing.assert_close(
            actual.view(torch.uint8)[:, :, 0],
            expected_heads.expand(rows, -1),
            rtol=0,
            atol=0,
        )
        workspace.acknowledge()

        # The unchanged TRTLLM-GEN sparse MLA binding consumes the completed
        # local inbox. Compare output and LSE bit-for-bit with the same call
        # using a contiguous collective result.
        from flashinfer.decode import trtllm_batch_decode_with_kv_cache_mla

        rows = 8
        topk = 128
        page_size = 64
        query_generator = torch.Generator(device=device)
        query_generator.manual_seed(31 * 997 + rank)
        local_query = (
            torch.randn(
                rows,
                local_heads,
                query_dim,
                generator=query_generator,
                dtype=torch.bfloat16,
                device=device,
            )
            / 4
        ).to(torch.float8_e4m3fn)
        dense_query = _reference_gather(local_query, world_size).view(
            torch.float8_e4m3fn
        )
        fanout_targets = workspace.begin_publish(rows)
        fanout_targets.copy_(local_query.unsqueeze(0))
        workspace.finish_publish()
        fanout_query = workspace.acquire_local_query(rows)

        torch.manual_seed(313 + rank)
        kv_cache = (
            torch.randn(
                topk // page_size,
                1,
                page_size,
                query_dim,
                dtype=torch.bfloat16,
                device=device,
            )
            / 4
        ).to(torch.float8_e4m3fn)
        topk_indices = (
            torch.arange(topk, dtype=torch.int32, device=device)
            .view(1, 1, topk)
            .expand(rows, 1, topk)
            .clone()
        )
        seq_lens = torch.full(
            (rows,),
            topk,
            dtype=torch.int32,
            device=device,
        )
        fanout_workspace = torch.zeros(
            512 * 1024 * 1024,
            dtype=torch.int8,
            device=device,
        )
        dense_workspace = torch.zeros_like(fanout_workspace)

        def run_sparse_mla(
            query: torch.Tensor,
            flashinfer_workspace: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            result = trtllm_batch_decode_with_kv_cache_mla(
                query=query.unsqueeze(1),
                kv_cache=kv_cache,
                workspace_buffer=flashinfer_workspace,
                qk_nope_head_dim=128,
                kv_lora_rank=512,
                qk_rope_head_dim=64,
                block_tables=topk_indices,
                seq_lens=seq_lens,
                max_seq_len=topk,
                sparse_mla_top_k=topk,
                bmm1_scale=1.0,
                bmm2_scale=1.0,
                return_lse=True,
                backend="trtllm-gen",
            )
            assert isinstance(result, tuple)
            return result

        fanout_output, fanout_lse = run_sparse_mla(
            fanout_query,
            fanout_workspace,
        )
        workspace.acknowledge()
        dense_output, dense_lse = run_sparse_mla(
            dense_query,
            dense_workspace,
        )
        torch.accelerator.synchronize()
        torch.testing.assert_close(fanout_output, dense_output, rtol=0, atol=0)
        torch.testing.assert_close(fanout_lse, dense_lse, rtol=0, atol=0)

        # Capture once and replay with changing owner-local query bytes.
        rows = 32
        graph_input = _make_query_bytes(
            rank,
            rows,
            local_heads,
            query_dim,
            99,
            device,
        )
        graph_output = torch.empty(
            rows,
            world_size * local_heads,
            query_dim,
            dtype=torch.float8_e4m3fn,
            device=device,
        )
        dist.barrier()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            fanout_targets = workspace.begin_publish(rows)
            fanout_targets.copy_(graph_input.unsqueeze(0))
            workspace.finish_publish()
            graph_output.copy_(workspace.acquire_local_query(rows))
            workspace.acknowledge()

        for replay in range(4):
            if replay:
                graph_input.view(torch.uint8).add_(17)
            expected = _reference_gather(graph_input, world_size)
            dist.barrier()
            graph.replay()
            torch.accelerator.synchronize()
            _assert_byte_equal(graph_output, expected)
    finally:
        workspace.close()
        dist.destroy_process_group(cpu_group)
        dist.destroy_process_group()


@pytest.mark.skipif(
    not current_platform.is_cuda(),
    reason="DCP query producer fanout requires CUDA.",
)
@pytest.mark.skipif(envs.VLLM_TARGET_DEVICE != "cuda", reason="Only test on CUDA")
def test_dcp_query_vmm_matches_collective_and_replays_graph() -> None:
    world_size = 4
    if torch.accelerator.device_count() < world_size:
        pytest.skip(f"Test requires {world_size} GPUs")
    mp.spawn(
        _worker,
        args=(world_size, _get_free_port()),
        nprocs=world_size,
    )
