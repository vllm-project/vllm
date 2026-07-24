# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from vllm.platforms import current_platform
from vllm.utils.network_utils import get_open_port


def _make_hidden(
    row_map: torch.Tensor,
    *,
    hidden_size: int,
    dtype: torch.dtype,
    offset: int,
) -> torch.Tensor:
    columns = torch.arange(hidden_size, device=row_map.device, dtype=torch.float32)
    rows = row_map.clamp_min(0).to(torch.float32).unsqueeze(1)
    values = rows * 17 + columns + offset
    values[row_map < 0] = -123
    return values.to(dtype)


def _collective_restore(
    hidden_states: torch.Tensor,
    restore_idx: torch.Tensor,
    *,
    nccl_group: dist.ProcessGroup,
) -> torch.Tensor:
    gathered_parts = [
        torch.empty_like(hidden_states) for _ in range(dist.get_world_size())
    ]
    dist.all_gather(gathered_parts, hidden_states, group=nccl_group)
    return torch.cat(gathered_parts)[restore_idx]


def _hidden_restore_worker(rank: int, world_size: int, port: int) -> None:
    os.environ.update(
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
        RANK=str(rank),
        WORLD_SIZE=str(world_size),
    )
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    nccl_group = dist.new_group(backend="nccl")
    try:
        from vllm.v1.worker.gpu.pcp_hidden_restore import PCPHiddenStateRestorer

        first_maps = (
            [0, 4, 8, 12, 14],
            [1, 5, 9, 13, -1],
            [2, 6, 10, -1, -1],
            [3, 7, 11, -1, -1],
        )
        first_restore = [0, 5, 10, 15, 1, 6, 11, 16, 2, 7, 12, 17, 3, 8, 4]
        second_maps = (
            [0, 4, 8, -1, -1],
            [1, 5, -1, -1, -1],
            [2, 6, -1, -1, -1],
            [3, 7, -1, -1, -1],
        )
        second_restore = [0, 5, 10, 15, 1, 6, 11, 16, 2]

        for dtype in (torch.bfloat16, torch.float16):
            restorer = PCPHiddenStateRestorer(
                group=dist.group.WORLD,
                device=device,
                max_num_tokens=32,
                hidden_size=64,
                dtype=dtype,
            )

            first_map = torch.tensor(first_maps[rank], device=device)
            first_hidden = _make_hidden(
                first_map,
                hidden_size=64,
                dtype=dtype,
                offset=0,
            )
            first_reference = _collective_restore(
                first_hidden,
                torch.tensor(first_restore, device=device),
                nccl_group=nccl_group,
            )
            first_direct = restorer.restore(
                first_hidden,
                first_map,
                num_global_tokens=len(first_restore),
            )
            assert torch.equal(first_direct, first_reference)

            # Rank 3 delays its prior-output consumer. The next call's retire
            # fence must prevent peers from overwriting rank 3's slab early.
            if rank == 3:
                torch.cuda._sleep(20_000_000)
            saved_first = first_direct.clone()

            second_map = torch.tensor(second_maps[rank], device=device)
            second_hidden = _make_hidden(
                second_map,
                hidden_size=64,
                dtype=dtype,
                offset=1000,
            )
            second_direct = restorer.restore(
                second_hidden,
                second_map,
                num_global_tokens=len(second_restore),
            )
            second_reference = _collective_restore(
                second_hidden,
                torch.tensor(second_restore, device=device),
                nccl_group=nccl_group,
            )
            assert torch.equal(second_direct, second_reference)

            # A third restore reuses slab 0. The publication fence from the
            # second restore must have retired first_direct on every rank.
            third_hidden = _make_hidden(
                first_map,
                hidden_size=64,
                dtype=dtype,
                offset=2000,
            )
            third_direct = restorer.restore(
                third_hidden,
                first_map,
                num_global_tokens=len(first_restore),
            )
            third_reference = _collective_restore(
                third_hidden,
                torch.tensor(first_restore, device=device),
                nccl_group=nccl_group,
            )
            assert torch.equal(third_direct, third_reference)
            assert torch.equal(saved_first, first_reference)

            restorer.close()
            restorer.close()
        dist.barrier()
    finally:
        dist.destroy_process_group(nccl_group)
        dist.destroy_process_group()


def _peer_access_available(world_size: int) -> bool:
    if not current_platform.is_cuda() or torch.cuda.device_count() < world_size:
        return False
    return all(
        src == dst or torch.cuda.can_device_access_peer(src, dst)
        for src in range(world_size)
        for dst in range(world_size)
    )


def test_direct_hidden_restore_matches_collective_on_four_gpus() -> None:
    world_size = 4
    if not _peer_access_available(world_size):
        pytest.skip("Direct PCP hidden restore requires four peer-accessible GPUs")
    mp.spawn(
        _hidden_restore_worker,
        args=(world_size, get_open_port()),
        nprocs=world_size,
    )
