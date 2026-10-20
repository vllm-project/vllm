# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TP4 stress of the FFN launch's peer hand-offs over XGMI: four ranks replay
graphs of three widths, alternating two payload sets (half of them ending in
the step-end fault handshake), with random per-rank device delays before every
launch. Every launch must reproduce its first result bit for bit, the reduced
output must match across ranks, and no wait may time out."""

import os
import zlib
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from vllm.platforms import current_platform

LAUNCHES = int(os.environ.get("MINIMAX_M3_STRESS_LAUNCHES", "120000"))
WIDTHS = (1, 4, 16)
MAX_SKEW_CYCLES = 200_000  # ~0.1 ms at the shader clock


def _supported() -> bool:
    if not current_platform.is_rocm() or torch.accelerator.device_count() < 4:
        return False
    from vllm.platforms.rocm import on_gfx950

    return on_gfx950()


pytestmark = [
    pytest.mark.optional,
    pytest.mark.distributed(num_gpus=4),
    pytest.mark.skipif(not _supported(), reason="needs 4 gfx950 GPUs"),
]


def _weights(dev):
    import test_ffn_kernel

    return test_ffn_kernel.weights.__wrapped__()


def _rank(rank, port, results):
    from vllm.models.deepseek_v41.amd.mono.common.peer_memory import PeerBuffer
    from vllm.models.minimax_m3.amd.specialized_decode.mono import ffn
    from vllm.models.minimax_m3.amd.specialized_decode.mono.runner import FfnRunner

    torch.accelerator.set_device_index(rank)
    dist.init_process_group(
        "gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=4
    )
    dev = torch.device("cuda", rank)
    w = _weights(dev)
    peer = PeerBuffer(ffn.peer_bytes(4), dist.group.WORLD, rank, 4, dev)
    peer.bytes.zero_()
    runner = FfnRunner(4, rank, peer.local, peer.addresses, dev, WIDTHS)
    runner.warm(w)
    torch.accelerator.synchronize()
    dist.barrier()

    def inputs(s, which):
        g = torch.Generator(device=dev).manual_seed(zlib.crc32(f"{s}/{which}".encode()))
        residual = torch.randn(s, ffn.HIDDEN, device=dev, generator=g).bfloat16()
        g.manual_seed(zlib.crc32(f"{s}/{which}/{rank}".encode()))
        part = torch.randn(s, ffn.HIDDEN, device=dev, generator=g).bfloat16()
        slots = torch.arange(s, dtype=torch.int64, device=dev)
        return part, residual, slots

    cases, graphs, outs, refs, keep = [], {}, {}, {}, {}
    for s in WIDTHS:
        for which in (0, 1):
            case = (s, which)
            cases.append(case)
            # graphs read their inputs by address: they must outlive them
            keep[case] = part, residual, slots = inputs(s, which)
            final = which == 1
            out = runner(part, residual, w, slots, final)[0]
            refs[case] = out.clone()
            graphs[case] = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graphs[case]):
                outs[case] = runner(part, residual, w, slots, final)[0]
    torch.accelerator.synchronize()
    dist.barrier()

    order = torch.Generator().manual_seed(1234)
    skew = torch.Generator().manual_seed(rank)
    mismatches = torch.zeros((), dtype=torch.int64, device=dev)
    for n in range(LAUNCHES):
        case = cases[int(torch.randint(len(cases), (1,), generator=order))]
        torch.cuda._sleep(int(torch.randint(MAX_SKEW_CYCLES, (1,), generator=skew)))
        graphs[case].replay()
        mismatches += (outs[case] != refs[case]).any()
        if n % 4096 == 4095:
            torch.accelerator.synchronize()
    torch.accelerator.synchronize()

    sums = torch.stack(
        [refs[c].view(torch.int16).to(torch.int64).sum() for c in cases]
    ).cpu()
    gathered = [torch.zeros_like(sums) for _ in range(4)]
    dist.all_gather(gathered, sums)
    results[rank] = SimpleNamespace(
        mismatches=int(mismatches.item()),
        error=peer.bytes[: 4 * ffn.ERROR_WORDS].view(torch.int32).tolist(),
        epoch=int(runner.epoch[0].item()),
        same_across_ranks=all(torch.equal(gathered[0], g) for g in gathered),
    )
    dist.barrier()
    graphs.clear()
    peer.close()
    dist.destroy_process_group()


def test_peer_handoffs_under_rank_skew():
    manager = mp.Manager()
    results = manager.dict()
    port = 29500 + os.getpid() % 1000
    mp.start_processes(_rank, args=(port, results), nprocs=4, start_method="spawn")
    for rank in range(4):
        r = results[rank]
        assert r.error[0] == 0, r.error
        assert r.mismatches == 0
        assert r.same_across_ranks
        # every capture-time and replayed launch moved the epoch on
        assert r.epoch == LAUNCHES + 2 * len(WIDTHS)
