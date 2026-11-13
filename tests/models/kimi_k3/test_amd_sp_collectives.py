# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The sequence-parallel prefill collectives must match torch.distributed.

Each rank holds a different bf16 partial; the reduce-scatters must equal the
fp32 sum rounded to bf16 (up to the summation order), and the all-gather must
be exact. Shapes are the Kimi-K3 TP8 ones (hidden 7168, latent 3584), with a
row count that is not a multiple of the padded row count for rs_rows.
"""

import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from tests.utils import multi_gpu_test
from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_rocm(),
    reason="The sequence-parallel prefill is only wired up on ROCm",
)

ROWS = 1024  # padded rows, a multiple of world x 8
VALID = 1000  # rows the producer actually has


def _close(got: torch.Tensor, want: torch.Tensor) -> None:
    # One bf16 ulp: both sides round an fp32 sum of the same bf16 inputs.
    torch.testing.assert_close(got.float(), want.float(), atol=1e-2, rtol=8e-3)


def _worker(rank: int, world: int, port: int) -> None:
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    torch.cuda.set_device(rank)
    dev = torch.device("cuda", rank)
    dist.init_process_group("nccl", rank=rank, world_size=world, device_id=dev)
    from vllm.models.kimi_k3.amd.ops.sp_collectives import SPCollectives

    comm = SPCollectives(dist.group.WORLD.group_name, rank, world, ROWS, dev)
    R = ROWS // world
    own = slice(rank * R, (rank + 1) * R)
    gen = torch.Generator(device=dev).manual_seed(rank)

    # rs_rows: rows past VALID are zero on every rank.
    x = torch.randn(VALID, 7168, device=dev, generator=gen).bfloat16()
    full = torch.zeros(ROWS, 7168, device=dev)
    full[:VALID] = x.float()
    dist.all_reduce(full)
    _close(comm.rs_rows(x, "p", 64, ROWS), full[own])

    # rs_cols: own column block of the sum and its per-row sum of squares.
    lat_cols = 448 * world
    lat = torch.randn(ROWS, lat_cols, device=dev, generator=gen).bfloat16()
    lat_sum = lat.float()
    dist.all_reduce(lat_sum)
    cols, ss = comm.rs_cols(lat, "p", 64)
    _close(cols, lat_sum[:, rank * 448 : (rank + 1) * 448])
    torch.testing.assert_close(ss, cols.float().pow(2).sum(1), rtol=1e-5, atol=1e-3)

    # allreduce_vec
    v = torch.randn(ROWS, device=dev, generator=gen)
    v_sum = v.clone()
    dist.all_reduce(v_sum)
    torch.testing.assert_close(comm.allreduce_vec(v, "v"), v_sum)

    # ag_rows: exact, and repeatable on the same buffer.
    for step in range(2):
        mine = torch.full((R, 7168), float(rank + step), device=dev).bfloat16()
        want = torch.empty(ROWS, 7168, device=dev, dtype=torch.bfloat16)
        dist.all_gather_into_tensor(want, mine)
        assert torch.equal(comm.ag_rows(mine, "x", 64), want)
    torch.cuda.synchronize()
    dist.destroy_process_group()


@multi_gpu_test(num_gpus=2)
@pytest.mark.parametrize("world", [2, 8])
def test_sp_collectives(world: int) -> None:
    if torch.cuda.device_count() < world:
        pytest.skip(f"needs {world} GPUs")
    mp.spawn(_worker, args=(world, 29511 + world), nprocs=world, join=True)
