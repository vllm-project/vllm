# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exactness and CUDA-graph tests for the symm-mem fused DCP top-k merge."""

import socket

import pytest
import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
import torch.multiprocessing as mp

import vllm.envs as envs
from vllm.distributed import cleanup_dist_env_and_memory
from vllm.platforms import current_platform
from vllm.utils.import_utils import has_cutedsl
from vllm.utils.system_utils import update_environment_variables

HEADER_BYTES = 256


def _get_free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("localhost", 0))
        return sock.getsockname()[1]


def _make_local_inputs(
    rank: int,
    rows: int,
    num_cols: int,
    topk_tokens: int,
    seed: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator(device=device)
    gen.manual_seed(seed * 997 + rank)
    logits = torch.randn(rows, num_cols, device=device, generator=gen)
    k = min(topk_tokens, num_cols)
    local_topk = torch.topk(logits, k=k, dim=1).indices.to(torch.int32)
    topk_indices = torch.full((rows, topk_tokens), -1, dtype=torch.int32, device=device)
    topk_indices[:, :k] = local_topk
    return logits, topk_indices


def _reference_merge(
    logits: torch.Tensor,
    topk_indices: torch.Tensor,
    topk_tokens: int,
    rank: int,
    world_size: int,
    candidate_count: int,
) -> torch.Tensor:
    """The production NCCL path: pack -> all-gather -> gathered selector."""
    from vllm.model_executor.kernels.attention.dsa.dcp_indexer_cutedsl import (
        pack_dcp_topk_candidates_cutedsl,
        stable_topk_from_gathered_candidates_cutedsl,
    )

    rows = topk_indices.shape[0]
    packed = torch.empty(
        (rows, candidate_count, 2), dtype=torch.float32, device=logits.device
    )
    pack_dcp_topk_candidates_cutedsl(
        logits,
        topk_indices[:, :candidate_count],
        packed,
        rank,
        world_size,
        1,
        None,
    )
    gathered_flat = torch.empty(
        (world_size * rows, candidate_count, 2),
        dtype=torch.float32,
        device=logits.device,
    )
    dist.all_gather_into_tensor(gathered_flat, packed)
    gathered = (
        gathered_flat.reshape((world_size, rows, candidate_count, 2))
        .permute(1, 0, 2, 3)
        .reshape(rows, world_size * candidate_count, 2)
        .contiguous()
    )
    out = torch.empty(rows, topk_tokens, dtype=torch.int32, device=logits.device)
    stable_topk_from_gathered_candidates_cutedsl(gathered, topk_tokens, out=out)
    return out


def _make_workspace(rank: int, world_size: int, max_rows: int, candidates: int):
    from vllm.model_executor.kernels.attention.dsa.dcp_topk_symm_mem import (
        DcpTopkSymmMemWorkspace,
    )

    inbox_bytes = max_rows * world_size * candidates * 2 * 4
    local = symm_mem.empty(
        (HEADER_BYTES + inbox_bytes,), dtype=torch.uint8, device=f"cuda:{rank}"
    )
    local.zero_()
    handle = symm_mem.rendezvous(local, dist.group.WORLD.group_name)
    handle.barrier(channel=0)
    inbox = handle.get_buffer(
        rank,
        (max_rows, world_size, candidates, 2),
        torch.float32,
        HEADER_BYTES // 4,
    )
    return DcpTopkSymmMemWorkspace(
        my_rank=rank,
        world_size=world_size,
        max_rows=max_rows,
        local_candidates=candidates,
        buffer_ptrs_dev=handle.buffer_ptrs_dev,
        inbox=inbox,
        local_buffer=local,
        handle=handle,
    )


def _symm_mem_merge_worker(local_rank: int, world_size: int, port: int):
    monkeypatch = pytest.MonkeyPatch()
    with monkeypatch.context() as m:
        m.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        update_environment_variables(
            {
                "RANK": str(local_rank),
                "LOCAL_RANK": str(local_rank),
                "WORLD_SIZE": str(world_size),
                "MASTER_ADDR": "localhost",
                "MASTER_PORT": str(port),
            }
        )
        device = torch.device(f"cuda:{local_rank}")
        torch.accelerator.set_device_index(device)
        dist.init_process_group("nccl")

        topk_tokens = 2048
        # Production exchanges each rank's full local top-K (exact merge).
        candidates = topk_tokens
        num_cols = 4096
        max_rows = 96

        ws = _make_workspace(local_rank, world_size, max_rows, candidates)

        # Eager: several lockstep iterations exercising the epoch protocol,
        # with varying row counts.
        for it, rows in enumerate((64, 17, 96)):
            logits, topk_indices = _make_local_inputs(
                local_rank, rows, num_cols, topk_tokens, it, device
            )
            expected = _reference_merge(
                logits,
                topk_indices.clone(),
                topk_tokens,
                local_rank,
                world_size,
                candidates,
            )
            ws.merge(
                logits,
                topk_indices,
                topk_tokens,
                local_rank,
                world_size,
                1,
                None,
            )
            torch.accelerator.synchronize()
            torch.testing.assert_close(
                topk_indices.sort(dim=1).values,
                expected.sort(dim=1).values,
                msg=f"iteration {it} rows {rows}",
            )

        # CUDA graph: capture one merge, replay twice, verify both replays.
        rows = 64
        logits, topk_indices = _make_local_inputs(
            local_rank, rows, num_cols, topk_tokens, 99, device
        )
        expected = _reference_merge(
            logits,
            topk_indices.clone(),
            topk_tokens,
            local_rank,
            world_size,
            candidates,
        )
        static_indices = topk_indices.clone()
        torch.accelerator.synchronize()
        dist.barrier()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            ws.merge(
                logits,
                static_indices,
                topk_tokens,
                local_rank,
                world_size,
                1,
                None,
            )
        for _ in range(2):
            static_indices.copy_(topk_indices)
            dist.barrier()
            graph.replay()
            torch.accelerator.synchronize()
            torch.testing.assert_close(
                static_indices.sort(dim=1).values,
                expected.sort(dim=1).values,
            )

        dist.destroy_process_group()


@pytest.mark.skipif(
    not current_platform.is_cuda(),
    reason="Symmetric-memory DCP top-k requires CUDA.",
)
@pytest.mark.skipif(envs.VLLM_TARGET_DEVICE != "cuda", reason="Only test on CUDA")
@pytest.mark.skipif(not has_cutedsl(), reason="Requires CuTeDSL.")
@pytest.mark.parametrize("world_size", [2, 4])
def test_dcp_topk_symm_mem_matches_allgather_reference(world_size: int):
    if world_size > torch.accelerator.device_count():
        pytest.skip("Not enough GPUs to run the test.")

    port = _get_free_port()
    mp.spawn(
        _symm_mem_merge_worker,
        args=(world_size, port),
        nprocs=world_size,
    )
    cleanup_dist_env_and_memory()
