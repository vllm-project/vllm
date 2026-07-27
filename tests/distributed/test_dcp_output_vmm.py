# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Correctness and CUDA-graph tests for owner-local DCP output/LSE merge."""

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


def _make_inputs(
    rank: int,
    rows: int,
    total_heads: int,
    head_dim: int,
    seed: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed * 997 + rank)
    output = torch.randn(
        rows,
        total_heads,
        head_dim,
        generator=generator,
        device=device,
        dtype=torch.bfloat16,
    )
    lse = torch.randn(
        rows,
        total_heads,
        generator=generator,
        device=device,
        dtype=torch.float32,
    )
    return output, lse


def _reference_merge(
    local_output: torch.Tensor,
    local_lse: torch.Tensor,
    rank: int,
    world_size: int,
    *,
    is_lse_base_on_e: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    rows, total_heads, head_dim = local_output.shape
    gathered_output_flat = torch.empty(
        world_size * rows,
        total_heads,
        head_dim,
        device=local_output.device,
        dtype=local_output.dtype,
    )
    gathered_lse_flat = torch.empty(
        world_size * rows,
        total_heads,
        device=local_lse.device,
        dtype=local_lse.dtype,
    )
    dist.all_gather_into_tensor(gathered_output_flat, local_output)
    dist.all_gather_into_tensor(gathered_lse_flat, local_lse)
    outputs = gathered_output_flat.view(world_size, rows, total_heads, head_dim).float()
    lses = gathered_lse_flat.view(world_size, rows, total_heads)
    lses = torch.where(
        torch.isnan(lses) | torch.isposinf(lses),
        -torch.inf,
        lses,
    )
    lse_max = lses.amax(dim=0)
    lse_max = torch.where(torch.isneginf(lse_max), 0.0, lse_max)
    weights = (
        torch.exp(lses - lse_max) if is_lse_base_on_e else torch.exp2(lses - lse_max)
    )
    weight_sum = weights.sum(dim=0)
    normalized = torch.where(
        weight_sum.unsqueeze(0) == 0,
        0.0,
        weights / weight_sum.unsqueeze(0),
    )
    merged_output = (outputs * normalized.unsqueeze(-1)).sum(dim=0)
    merged_lse = (
        torch.log(weight_sum) if is_lse_base_on_e else torch.log2(weight_sum)
    ) + lse_max

    local_heads = total_heads // world_size
    head_start = rank * local_heads
    head_end = head_start + local_heads
    return (
        merged_output[:, head_start:head_end].to(torch.bfloat16),
        merged_lse[:, head_start:head_end],
    )


def _assert_matches(
    actual_output: torch.Tensor,
    actual_lse: torch.Tensor,
    expected_output: torch.Tensor,
    expected_lse: torch.Tensor,
) -> None:
    torch.testing.assert_close(actual_output, expected_output, rtol=0.02, atol=0.02)
    torch.testing.assert_close(actual_lse, expected_lse, rtol=1e-5, atol=1e-5)


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
    from vllm.v1.attention.ops.dcp_output_vmm import (
        create_dcp_output_vmm_workspace_for_group,
    )

    total_heads = 64
    head_dim = 512
    max_rows = 128
    workspace = create_dcp_output_vmm_workspace_for_group(
        max_rows,
        total_heads,
        head_dim,
        cpu_group,
        device,
    )
    assert workspace.payload_bytes_per_rank == 10_526_720
    assert workspace.peer_partial_outputs.shape == (
        world_size,
        max_rows,
        total_heads,
        head_dim,
    )

    try:
        for iteration, rows in enumerate((1, 8, 32, 64, 128)):
            is_lse_base_on_e = iteration % 2 == 0
            local_output, local_lse = _make_inputs(
                rank,
                rows,
                total_heads,
                head_dim,
                iteration,
                device,
            )
            expected_output, expected_lse = _reference_merge(
                local_output,
                local_lse,
                rank,
                world_size,
                is_lse_base_on_e=is_lse_base_on_e,
            )
            actual_output, actual_lse = workspace.merge(
                local_output,
                local_lse,
                is_lse_base_on_e=is_lse_base_on_e,
                return_lse=True,
            )
            torch.accelerator.synchronize()
            _assert_matches(
                actual_output,
                actual_lse,
                expected_output,
                expected_lse,
            )

        # Exercise empty and invalid owner statistics. Positive infinity and
        # NaN are intentionally treated as an empty shard, matching common.py.
        rows = 8
        local_output, local_lse = _make_inputs(
            rank,
            rows,
            total_heads,
            head_dim,
            17,
            device,
        )
        local_lse[:, 0] = -torch.inf
        if rank == 0:
            local_lse[:, 1] = torch.nan
        elif rank == 1:
            local_lse[:, 1] = torch.inf
        expected_output, expected_lse = _reference_merge(
            local_output,
            local_lse,
            rank,
            world_size,
            is_lse_base_on_e=False,
        )
        actual_output, actual_lse = workspace.merge(
            local_output,
            local_lse,
            is_lse_base_on_e=False,
            return_lse=True,
        )
        torch.accelerator.synchronize()
        _assert_matches(
            actual_output,
            actual_lse,
            expected_output,
            expected_lse,
        )

        # Capture once and replay with changing input bytes. Sequence counters
        # must advance on the device rather than reuse capture-time epochs.
        rows = 32
        graph_output_input, graph_lse_input = _make_inputs(
            rank,
            rows,
            total_heads,
            head_dim,
            99,
            device,
        )
        dist.barrier()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_output, graph_lse = workspace.merge(
                graph_output_input,
                graph_lse_input,
                is_lse_base_on_e=False,
                return_lse=True,
            )

        for replay in range(2):
            if replay:
                graph_output_input.neg_()
                graph_lse_input.add_(0.5)
            expected_output, expected_lse = _reference_merge(
                graph_output_input,
                graph_lse_input,
                rank,
                world_size,
                is_lse_base_on_e=False,
            )
            dist.barrier()
            graph.replay()
            torch.accelerator.synchronize()
            _assert_matches(
                graph_output,
                graph_lse,
                expected_output,
                expected_lse,
            )
    finally:
        workspace.close()
        dist.destroy_process_group(cpu_group)
        dist.destroy_process_group()


@pytest.mark.skipif(
    not current_platform.is_cuda(),
    reason="Owner-local DCP output/LSE merge requires CUDA.",
)
@pytest.mark.skipif(envs.VLLM_TARGET_DEVICE != "cuda", reason="Only test on CUDA")
def test_dcp_output_vmm_matches_collectives_and_replays_graph() -> None:
    world_size = 4
    if torch.accelerator.device_count() < world_size:
        pytest.skip(f"Test requires {world_size} GPUs")
    mp.spawn(
        _worker,
        args=(world_size, _get_free_port()),
        nprocs=world_size,
    )
