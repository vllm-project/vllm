# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SDMA all-reduce on two ROCm GPUs: a call consumed by the fused norm on
another stream than its push, after the next call was consumed, still reads its
own messages, and both ranks get identical outputs."""

import hashlib

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from vllm.platforms import current_platform
from vllm.utils.network_utils import get_open_port

HIDDEN, EPS = 6144, 1e-6
CASES = [(tokens, also_bf16) for tokens in (257, 4096) for also_bf16 in (False, True)]
# Holds rank 0's fused consumers, so the peer's next push into the same slot is
# ready while the consumer still has to read it.
SLEEP_CYCLES = 1_000_000


def _partial(tokens: int, seed: int, rank: int) -> torch.Tensor:
    g = torch.Generator(device="cuda").manual_seed(2 * seed + rank)
    x = torch.randn(tokens, HIDDEN, generator=g, device="cuda") * 0.5
    x[:, :8] *= 20
    return x.bfloat16()


def _residual(tokens: int, seed: int) -> torch.Tensor:
    g = torch.Generator(device="cuda").manual_seed(10_000 + seed)
    r = torch.randn(tokens, HIDDEN, generator=g, device="cuda") * 2.0
    r[:, 8:12] *= 200
    return r.bfloat16()


def _worker(rank: int, port: int) -> None:
    from vllm.distributed.device_communicators.dma_all_reduce import DmaAllReduce

    torch.accelerator.set_device_index(rank)
    dist.init_process_group(
        "gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=2
    )
    device = torch.device("cuda", rank)
    dma = DmaAllReduce(dist.group.WORLD, device, max_bytes=4096 * HIDDEN * 2)
    g = torch.Generator(device="cuda").manual_seed(7)
    weight = (torch.randn(HIDDEN, generator=g, device="cuda") * 0.3).bfloat16()
    calls = list(enumerate(CASES * 4))
    # As under DBO, every push runs on the comm stream.
    comm = torch.cuda.Stream(device=device)
    compute = torch.cuda.current_stream(device)

    # References: each call consumed before the next one.
    with torch.cuda.stream(comm):
        sums = [dma.all_reduce(_partial(t, 2 * i, rank)) for i, (t, _) in calls]
        nexts = [dma.all_reduce(_partial(t, 2 * i + 1, rank)) for i, (t, _) in calls]

    results = []
    for i, (tokens, also_bf16) in calls:
        residual = _residual(tokens, i)
        with torch.cuda.stream(comm):
            pending = dma.push(_partial(tokens, 2 * i, rank))
            nxt = dma.all_reduce(_partial(tokens, 2 * i + 1, rank))
        compute.wait_stream(comm)
        if rank == 0:
            torch.cuda._sleep(SLEEP_CYCLES)
        fused = dma.add_rmsnorm_mxfp8(pending, residual, weight, EPS, also_bf16)
        results.append((residual, nxt, fused))
    torch.accelerator.synchronize()

    digests = []
    for i, (residual, nxt, fused) in enumerate(results):
        assert torch.equal(nxt, nexts[i]), f"call {2 * i + 1}"
        expected_residual = (sums[i].float() + residual.float()).bfloat16()
        assert torch.equal(fused[3], expected_residual), f"call {2 * i}"
        h = hashlib.sha256()
        for t in fused:
            if t is not None:
                h.update(t.view(torch.uint8).cpu().numpy().tobytes())
        digests.append(h.hexdigest())
    both: list[list[str] | None] = [None, None]
    dist.all_gather_object(both, digests)
    assert both[0] == both[1]
    dma.close()
    dist.destroy_process_group()


@pytest.mark.skipif(not current_platform.is_rocm(), reason="ROCm only")
def test_fused_consumer_on_another_stream():
    if torch.accelerator.device_count() < 2:
        pytest.skip("needs 2 GPUs")
    mp.spawn(_worker, args=(get_open_port(),), nprocs=2, join=True)
