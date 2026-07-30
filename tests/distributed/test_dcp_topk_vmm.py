# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exactness and CUDA-graph tests for owner-local VMM DCP top-k."""

import os
import socket

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import vllm.envs as envs
from vllm.platforms import current_platform
from vllm.utils.import_utils import has_cutedsl


def _get_free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("localhost", 0))
        return int(sock.getsockname()[1])


def _make_inputs(
    rank: int,
    rows: int,
    num_cols: int,
    topk: int,
    seed: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed * 997 + rank)
    logits = torch.randn(
        rows,
        num_cols,
        generator=generator,
        device=device,
    )
    indices = torch.topk(logits, k=topk, dim=1).indices.to(torch.int32)
    return logits, indices


def _reference_merge(
    logits: torch.Tensor,
    indices: torch.Tensor,
    topk: int,
    rank: int,
    world_size: int,
    row_starts: torch.Tensor | None = None,
) -> torch.Tensor:
    from vllm.model_executor.kernels.attention.dsa.dcp_indexer_cutedsl import (
        pack_dcp_topk_candidates_cutedsl,
        stable_topk_from_gathered_candidates_cutedsl,
    )

    rows = indices.shape[0]
    packed = torch.empty(
        (rows, topk, 2),
        dtype=torch.float32,
        device=logits.device,
    )
    pack_dcp_topk_candidates_cutedsl(
        logits,
        indices,
        packed,
        rank,
        world_size,
        1,
        row_starts,
    )
    gathered_flat = torch.empty(
        (world_size * rows, topk, 2),
        dtype=torch.float32,
        device=logits.device,
    )
    dist.all_gather_into_tensor(gathered_flat, packed)
    gathered = (
        gathered_flat.view(world_size, rows, topk, 2)
        .permute(1, 0, 2, 3)
        .reshape(rows, world_size * topk, 2)
        .contiguous()
    )
    stable_topk_from_gathered_candidates_cutedsl(
        gathered,
        topk,
        out=indices,
    )
    return indices


def _assert_same_ids(actual: torch.Tensor, expected: torch.Tensor) -> None:
    torch.testing.assert_close(
        actual.sort(dim=1).values,
        expected.sort(dim=1).values,
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
    from vllm.model_executor.kernels.attention.dsa.dcp_topk_vmm import (
        create_dcp_topk_vmm_workspace_for_group,
    )

    topk = 2048
    num_cols = 4096
    max_rows = 96
    workspace = create_dcp_topk_vmm_workspace_for_group(
        max_rows,
        topk,
        cpu_group,
        device,
    )
    assert workspace.candidate_payload_bytes_per_rank == max_rows * topk * 2 * 4
    assert workspace.peer_candidates.shape == (
        world_size,
        max_rows,
        topk,
        2,
    )

    try:
        for iteration, rows in enumerate((1, 17, 64, 96)):
            logits, indices = _make_inputs(
                rank,
                rows,
                num_cols,
                topk,
                iteration,
                device,
            )
            expected = _reference_merge(
                logits,
                indices.clone(),
                topk,
                rank,
                world_size,
            )
            workspace.merge(
                logits,
                indices,
                topk,
                rank,
                world_size,
                1,
                None,
            )
            torch.accelerator.synchronize()
            _assert_same_ids(indices, expected)

        # Stable selection with ties and one empty owner shard.
        rows = 8
        logits = torch.zeros(
            (rows, num_cols),
            dtype=torch.float32,
            device=device,
        )
        indices = (
            torch.arange(topk, dtype=torch.int32, device=device)
            .expand(rows, -1)
            .clone()
        )
        if rank == world_size - 1:
            indices.fill_(-1)
        expected = _reference_merge(
            logits,
            indices.clone(),
            topk,
            rank,
            world_size,
        )
        workspace.merge(
            logits,
            indices,
            topk,
            rank,
            world_size,
            1,
            None,
        )
        torch.accelerator.synchronize()
        _assert_same_ids(indices, expected)

        # Prefill-style row-relative candidate indices.
        rows = 8
        logits = torch.randn(
            (rows, num_cols),
            dtype=torch.float32,
            device=device,
        )
        row_starts = torch.arange(rows, dtype=torch.int32, device=device) * 3
        indices = torch.stack(
            [
                torch.topk(
                    logits[row, int(row_starts[row].item()) :],
                    topk,
                ).indices
                for row in range(rows)
            ]
        ).to(torch.int32)
        expected = _reference_merge(
            logits,
            indices.clone(),
            topk,
            rank,
            world_size,
            row_starts,
        )
        workspace.merge(
            logits,
            indices,
            topk,
            rank,
            world_size,
            1,
            row_starts,
        )
        torch.accelerator.synchronize()
        _assert_same_ids(indices, expected)

        # Capture once, then replay with two different input tensors. Device
        # epochs must advance on replay rather than reusing capture-time values.
        rows = 64
        logits, source_indices = _make_inputs(
            rank,
            rows,
            num_cols,
            topk,
            99,
            device,
        )
        graph_indices = source_indices.clone()
        dist.barrier()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_indices.copy_(source_indices)
            workspace.merge(
                logits,
                graph_indices,
                topk,
                rank,
                world_size,
                1,
                None,
            )

        for replay in range(2):
            if replay:
                logits.neg_()
                source_indices.copy_(
                    torch.topk(logits, k=topk, dim=1).indices.to(torch.int32)
                )
            expected = _reference_merge(
                logits,
                source_indices.clone(),
                topk,
                rank,
                world_size,
            )
            dist.barrier()
            graph.replay()
            torch.accelerator.synchronize()
            _assert_same_ids(graph_indices, expected)
    finally:
        torch.accelerator.synchronize()
        dist.barrier(group=cpu_group)
        workspace.close()
        dist.barrier(group=cpu_group)
        dist.destroy_process_group(cpu_group)
        dist.destroy_process_group()


@pytest.mark.skipif(
    not current_platform.is_cuda(),
    reason="Owner-local DCP top-k requires CUDA.",
)
@pytest.mark.skipif(envs.VLLM_TARGET_DEVICE != "cuda", reason="Only test on CUDA")
@pytest.mark.skipif(not has_cutedsl(), reason="Requires CuTeDSL.")
def test_dcp_topk_vmm_matches_allgather_and_replays_graph() -> None:
    world_size = 4
    if torch.accelerator.device_count() < world_size:
        pytest.skip(f"Test requires {world_size} GPUs")
    mp.spawn(
        _worker,
        args=(world_size, _get_free_port()),
        nprocs=world_size,
    )
