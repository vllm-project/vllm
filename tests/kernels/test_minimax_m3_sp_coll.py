# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ops.sp_coll on four GPUs vs CPU references: gathers bitwise, reduce-scatter =
the fp32 rank-order sum; slot reuse, in-place inputs and the RCCL fallback."""

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from vllm.platforms import current_platform

WORLD = 4
HIDDEN = 1024
ROWS = 256  # rows per rank

pytestmark = pytest.mark.skipif(
    not current_platform.is_rocm(), reason="FlyDSL kernel is ROCm-only"
)


class _NoRccl:
    """Records the calls that fell back to RCCL."""

    def __init__(self):
        self.calls = []

    def all_gather(self, outs, ins):
        self.calls.append("all_gather")
        for o, i in zip(outs, ins):
            gathered = [torch.empty_like(i.cpu()) for _ in range(WORLD)]
            dist.all_gather(gathered, i.cpu())
            exp = torch.cat(gathered)
            o.view(exp.shape).copy_(exp.to(o.device))

    def reduce_scatter(self, out, x):
        raise AssertionError("unexpected fallback")

    def all_to_all(self, out, x):
        raise AssertionError("unexpected fallback")


def _gather_cpu(x: torch.Tensor) -> list[torch.Tensor]:
    xs = [torch.empty_like(x.cpu()) for _ in range(WORLD)]
    dist.all_gather(xs, x.cpu().contiguous())
    return xs


def _worker(rank: int, init_file: str, errors):
    try:
        _run(rank, init_file)
    except Exception as e:  # noqa: BLE001
        errors.put(f"rank {rank}: {type(e).__name__}: {e}")
        raise


def _run(rank: int, init_file: str):
    torch.cuda.set_device(rank)
    dev = torch.device("cuda", rank)
    dist.init_process_group(
        "gloo", init_method=f"file://{init_file}", rank=rank, world_size=WORLD
    )
    from vllm.models.minimax_m3.amd.ops import sp_coll

    padded = ROWS * WORLD
    q = 512
    # the largest phase: bf16 partials + two bf16 gathers (as sp_prefill sizes it)
    slot = padded * HIDDEN * 2 + 2 * ROWS * HIDDEN * 2 + 4096
    pools = sp_coll.Pools(rank, WORLD, dist.group.WORLD, dev, slot, sets=3)
    stream = torch.cuda.current_stream()
    gen = torch.Generator().manual_seed(100 + rank)

    def rnd_u8(*shape):
        return torch.randint(0, 256, shape, generator=gen, dtype=torch.uint8).to(dev)

    for phase in range(6):
        set_id = phase % 3  # three slot sets
        rccl = _NoRccl()
        coll = sp_coll.Collectives(pools, set_id).bind(stream, rccl)
        # attention phase: all-to-all, then gathers
        send = rnd_u8(WORLD * ROWS, q + q // 32)
        recv = torch.empty_like(send)
        coll.all_to_all(recv, send)
        codes = rnd_u8(ROWS, HIDDEN).view(torch.float8_e4m3fn)
        scales = rnd_u8(ROWS, HIDDEN // 32)
        routing = torch.randint(
            -(2**31), 2**31 - 1, (ROWS, 18), generator=gen, dtype=torch.int32
        ).to(dev)
        g_codes = torch.empty((padded, HIDDEN), dtype=codes.dtype, device=dev)
        g_scales = torch.empty((padded, HIDDEN // 32), dtype=torch.uint8, device=dev)
        g_routing = torch.empty((padded, 18), dtype=torch.int32, device=dev)
        coll.all_gather([g_codes, g_scales, g_routing], [codes, scales, routing])
        torch.cuda.synchronize()
        sends = _gather_cpu(send)
        for p in range(WORLD):
            exp = sends[p][rank * ROWS : (rank + 1) * ROWS]
            assert torch.equal(recv[p * ROWS : (p + 1) * ROWS].cpu(), exp), (phase, p)
        for g, x in ((g_codes, codes), (g_scales, scales), (g_routing, routing)):
            if x.dtype != torch.int32:
                g, x = g.view(torch.uint8), x.view(torch.uint8)
            assert torch.equal(g.cpu(), torch.cat(_gather_cpu(x))), (phase, x.dtype)
        assert rccl.calls == []
        # MoE phase: reduce-scatter, then gathers
        coll = sp_coll.Collectives(pools, set_id).bind(stream, rccl)
        part = torch.randn((padded, HIDDEN), generator=gen) * (1 + rank)
        part = part.bfloat16().to(dev)
        h = torch.empty((ROWS, HIDDEN), dtype=torch.bfloat16, device=dev)
        coll.reduce_scatter(h, part)
        aux = (torch.randn((ROWS, HIDDEN), generator=gen)).bfloat16().to(dev)
        ctx = (torch.randn((ROWS, HIDDEN), generator=gen)).bfloat16().to(dev)
        g_aux = torch.empty((padded, HIDDEN), dtype=torch.bfloat16, device=dev)
        g_ctx = torch.empty((padded, HIDDEN), dtype=torch.bfloat16, device=dev)
        coll.all_gather([g_aux, g_ctx], [aux, ctx])
        torch.cuda.synchronize()
        parts = _gather_cpu(part)
        acc = torch.zeros((ROWS, HIDDEN), dtype=torch.float32)
        for p in range(WORLD):
            acc += parts[p][rank * ROWS : (rank + 1) * ROWS].float()
        assert torch.equal(h.cpu(), acc.bfloat16()), phase
        assert torch.equal(g_aux.cpu(), torch.cat(_gather_cpu(aux))), phase
        assert torch.equal(g_ctx.cpu(), torch.cat(_gather_cpu(ctx))), phase
        assert rccl.calls == []
    # slot views from alloc are read in place
    rccl = _NoRccl()
    coll = sp_coll.Collectives(pools, 1)
    send = coll.alloc((WORLD * ROWS, q + q // 32), torch.uint8)
    send.copy_(rnd_u8(WORLD * ROWS, q + q // 32))
    recv = torch.empty((WORLD * ROWS, q + q // 32), dtype=torch.uint8, device=dev)
    coll.bind(stream, rccl).all_to_all(recv, send)
    coll = sp_coll.Collectives(pools, 1)
    part = coll.alloc((padded, HIDDEN), torch.bfloat16)
    part.copy_((torch.randn((padded, HIDDEN), generator=gen) * 3).bfloat16())
    h = torch.empty((ROWS, HIDDEN), dtype=torch.bfloat16, device=dev)
    coll.bind(stream, rccl).reduce_scatter(h, part)
    torch.cuda.synchronize()
    sends, parts = _gather_cpu(send), _gather_cpu(part)
    for p in range(WORLD):
        exp = sends[p][rank * ROWS : (rank + 1) * ROWS]
        assert torch.equal(recv[p * ROWS : (p + 1) * ROWS].cpu(), exp), p
    acc = torch.zeros((ROWS, HIDDEN), dtype=torch.float32)
    for p in range(WORLD):
        acc += parts[p][rank * ROWS : (rank + 1) * ROWS].float()
    assert torch.equal(h.cpu(), acc.bfloat16())
    assert rccl.calls == []
    # all_to_all_unpack = all_to_all + mxfp8_unpack
    from vllm.models.minimax_m3.amd.ops.sp_pack import mxfp8_unpack

    rccl = _NoRccl()
    coll = sp_coll.Collectives(pools, 0).bind(stream, rccl)
    send = rnd_u8(WORLD * ROWS, q + q // 32)
    recv = torch.empty_like(send)
    coll.all_to_all(recv, send)
    coll = sp_coll.Collectives(pools, 0).bind(stream, rccl)
    codes, scales = coll.all_to_all_unpack(send, q)
    torch.cuda.synchronize()
    ref_codes, ref_scales = mxfp8_unpack(recv, WORLD, ROWS, q)
    assert torch.equal(codes.view(torch.uint8), ref_codes.view(torch.uint8))
    assert torch.equal(scales, ref_scales)
    assert rccl.calls == []
    # a phase that does not fit the slot falls back to RCCL
    rccl = _NoRccl()
    coll = sp_coll.Collectives(pools, 0).bind(stream, rccl)
    big = rnd_u8(4 * padded, HIDDEN)
    g_big = torch.empty((WORLD, 4 * padded, HIDDEN), dtype=torch.uint8, device=dev)
    coll.all_gather([g_big], [big])
    torch.cuda.synchronize()
    assert rccl.calls == ["all_gather"]
    assert torch.equal(g_big.view(-1, HIDDEN).cpu(), torch.cat(_gather_cpu(big)))
    dist.barrier()
    dist.destroy_process_group()


def test_sp_coll_four_gpus(tmp_path):
    if torch.cuda.device_count() < WORLD:
        pytest.skip("needs four GPUs")
    from vllm.platforms.rocm import on_gfx950

    if not on_gfx950():
        pytest.skip("the kernels use gfx950 instructions")
    ctx = mp.get_context("spawn")
    errors = ctx.Queue()
    init_file = str(tmp_path / "init")
    procs = [
        ctx.Process(target=_worker, args=(r, init_file, errors)) for r in range(WORLD)
    ]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=600)
    msgs = []
    while not errors.empty():
        msgs.append(errors.get())
    assert all(p.exitcode == 0 for p in procs), msgs or [p.exitcode for p in procs]
