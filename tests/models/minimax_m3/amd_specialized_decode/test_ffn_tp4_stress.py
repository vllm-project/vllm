# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TP4 stress of the FFN launch's peer hand-offs over XGMI: four ranks replay
graphs of three widths, alternating two payload sets (half of them ending in
the step-end fault handshake), with random per-rank device delays before every
launch and the executor's collective epoch reset every ``RESET_EVERY``
launches. Every launch must reproduce its first result bit for bit, the
reduced output must match across ranks, and no wait may time out."""

import os
import time
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
TIMEOUT_S = 600 + LAUNCHES // 1000
RESET_EVERY = 997  # launches between collective epoch resets


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


def _resetting_executor(runner, peer, rank):
    """The FFN executor's step hook over this rank's runner, its epoch reset
    every ``RESET_EVERY`` launches agreed over the stress's process group."""
    from vllm.models.deepseek_v41.amd.mono.common import peer_memory
    from vllm.models.minimax_m3.amd.specialized_decode.runtime import executor as ex

    ex.tp_agree = lambda ok: peer_memory.tp_agree(ok, dist.group.WORLD)
    executor = object.__new__(ex.FfnExecutor)
    executor.rank, executor.runner, executor.peer = rank, runner, peer
    executor._layers = {0: None}
    executor.outputs = executor._since_reset = executor.resets = 0
    executor._reset_launches = RESET_EVERY
    executor._epoch_event = None
    executor._epoch_host = torch.zeros(1, dtype=torch.int32)
    return executor


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

    executor = _resetting_executor(runner, peer, rank)
    order = torch.Generator().manual_seed(1234)
    skew = torch.Generator().manual_seed(rank)
    mismatches = torch.zeros((), dtype=torch.int64, device=dev)
    for n in range(LAUNCHES):
        case = cases[int(torch.randint(len(cases), (1,), generator=order))]
        torch.cuda._sleep(int(torch.randint(MAX_SKEW_CYCLES, (1,), generator=skew)))
        graphs[case].replay()
        mismatches += (outs[case] != refs[case]).any()
        executor.output_step()
        if n % 4096 == 4095:
            torch.accelerator.synchronize()
    torch.accelerator.synchronize()

    outs_cpu = torch.cat([refs[c].flatten() for c in cases]).view(torch.int32).cpu()
    gathered = [torch.zeros_like(outs_cpu) for _ in range(4)]
    dist.all_gather(gathered, outs_cpu)
    results[rank] = SimpleNamespace(
        mismatches=int(mismatches.item()),
        error=peer.bytes[: 4 * ffn.ERROR_WORDS].view(torch.int32).tolist(),
        epoch=int(runner.epoch[0].item()),
        resets=executor.resets,
        same_across_ranks=all(torch.equal(gathered[0], g) for g in gathered),
    )
    dist.barrier()
    graphs.clear()
    peer.close()
    dist.destroy_process_group()


def test_peer_handoffs_under_rank_skew():
    from vllm.utils.network_utils import get_open_port

    manager = mp.Manager()
    results = manager.dict()
    ranks = mp.start_processes(
        _rank, args=(get_open_port(), results), nprocs=4, join=False
    )
    deadline = time.monotonic() + TIMEOUT_S
    while not ranks.join(timeout=10):
        if time.monotonic() > deadline:
            for p in ranks.processes:
                p.kill()
            pytest.fail(f"TP4 stress ranks still running after {TIMEOUT_S} s")
    for rank in range(4):
        r = results[rank]
        assert r.error[0] == 0, r.error
        assert r.mismatches == 0
        assert r.same_across_ranks
        # every replayed launch since the last reset moved the epoch on
        assert r.resets == LAUNCHES // RESET_EVERY
        eager = 0 if r.resets else 2 * len(WIDTHS)
        assert r.epoch == LAUNCHES % RESET_EVERY + eager
