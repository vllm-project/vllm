# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DEP EP dispatch / combine (specialized_decode/cutedsl/moe/ep/comm.py) on real ranks.

Dispatch: every received row (static slot source x T + token) is the source token's
MXFP8 latent, bit-identical to ``flashinfer.mxfp8_quantize``, with its ids, weights and
source; exactly the tokens with a local expert arrive; empty rows stay non-local. A
stand-in expert (a per-expert scale of the dequantized row, routing-weighted) fills the
outputs; combine must return vLLM's RMSNorm of the rank-order FP32 sum of the BF16
per-rank partials. Repeated calls and a CUDA-graph replay exercise the flag re-arming.

pytest: 4 local GPUs (spawned ranks). Multi-node (16 / 32 ranks, one process per GPU):
  srun -N4 -n16 python \
    tests/models/kimi_k3/nvidia/specialized_decode/test_kimi_k3_sdec_ep_comm.py
"""

import os
import socket

import pytest
import torch
import torch.distributed as dist

from vllm.platforms import current_platform

LATENT, TOPK, EXPERTS = 3584, 16, 896


def dequant(x, sf):
    """MXFP8 rows [R, 3584] fp8 + E8M0 [R, 112] -> FP32."""
    scale = torch.exp2(sf.float() - 127).repeat_interleave(32, dim=1)
    return x.float() * scale


def stand_in(rows, ids, weights, rank, local):
    """BF16 sum over the row's local experts of w * dequant(row) * (1 + e % 7) / 8."""
    x = dequant(*rows)
    total = torch.zeros_like(x)
    for j in range(TOPK):
        e = ids[:, j]
        mine = (e // local) == rank
        factor = torch.where(mine, weights[:, j] * (1 + e % 7).float() / 8, 0)
        total += factor[:, None] * x
    return total.to(torch.bfloat16)


def worker(rank, world, tokens_list=(1, 5, 37)):
    from flashinfer import mxfp8_quantize

    from vllm import _custom_ops as ops
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.moe.ep.comm import stride
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.moe.ep.layer import (
        EpRouted,
    )

    device = torch.device("cuda", torch.cuda.current_device())
    group = dist.group.WORLD
    local = EXPERTS // world
    ep = EpRouted(group, max(tokens_list), device)
    norm_w = (torch.rand(LATENT, device=device) + 0.5).to(torch.bfloat16)
    gen = torch.Generator(device=device).manual_seed(100 + rank)
    for tokens in tokens_list:
        latent = (torch.randn(tokens, LATENT, device=device, generator=gen) * 3).to(
            torch.bfloat16
        )
        ids = (
            torch.rand(tokens, EXPERTS, device=device, generator=gen)
            .topk(TOPK, dim=1)
            .indices.int()
        )
        weights = torch.rand(tokens, TOPK, device=device, generator=gen) + 0.05
        everyone = [torch.empty_like(v) for v in (latent, ids, weights)]
        gathered = []
        for v, like in zip((latent, ids, weights), everyone):
            out = [torch.empty_like(like) for _ in range(world)]
            dist.all_gather(out, v)
            gathered.append(out)
        lat_all, ids_all, w_all = gathered
        q_all = [mxfp8_quantize(v, is_sf_swizzled_layout=False) for v in lat_all]
        q_all = [(q, s.view(torch.uint8).view(tokens, -1)) for q, s in q_all]

        # Reference: per destination rank the stand-in partial, rank-order sum, norm.
        acc = torch.zeros(tokens, LATENT, device=device)
        for d in range(world):
            part = stand_in(q_all[rank], ids, weights, d, local)
            sent = ((ids // local) == d).any(1)
            acc += torch.where(sent[:, None], part.float(), 0)
        routed = acc.to(torch.bfloat16)
        expected = torch.empty_like(routed)
        ops.rms_norm(expected, routed, norm_w, 1e-5)

        def step(latent=latent, ids=ids, weights=weights, tokens=tokens):
            normed = torch.empty_like(latent)
            ep.dispatch(latent, ids, weights)
            # stand-in experts on the receive rows (rows past the counter: zeros)
            b, rows = ep.buffers, world * stride(tokens)
            ep.out[:rows] = stand_in(
                (b.x(rows), b.sf(rows)), b.ids(rows), b.weights(rows), rank, local
            )
            return ep.combine(ids, norm_w, normed)

        for it in range(3):
            ep.dispatch(latent, ids, weights)
            torch.cuda.synchronize()
            dist.barrier()
            b, rows = ep.buffers, world * stride(tokens)
            src = b.src(rows).tolist()
            for r, v in enumerate(src):
                s, t = divmod(r, stride(tokens))
                sent = t < tokens and bool(((ids_all[s][t] // local) == rank).any())
                if not sent:  # a non-local row: empty, its ids never local
                    assert v == -1 and ((b.ids(rows)[r] // local) != rank).all()
                    continue
                assert v == (s << 16 | t), (rank, tokens, it, r, v)
                assert torch.equal(
                    b.x(rows)[r].view(torch.uint8), q_all[s][0][t].view(torch.uint8)
                )
                assert torch.equal(b.sf(rows)[r], q_all[s][1][t])
                assert torch.equal(b.ids(rows)[r], ids_all[s][t])
                assert torch.equal(b.weights(rows)[r], w_all[s][t])
            ep.out[:rows] = stand_in(
                (b.x(rows), b.sf(rows)), b.ids(rows), b.weights(rows), rank, local
            )
            normed = torch.empty_like(latent)
            ep.combine(ids, norm_w, normed)
            torch.cuda.synchronize()
            assert (b.src(rows) == -1).all()
            assert ((b.ids(rows) // local) != rank).all()
            torch.testing.assert_close(normed, expected, atol=2e-2, rtol=1e-2)
            dist.barrier()

        graph = torch.cuda.CUDAGraph()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            step()
        torch.cuda.synchronize()
        dist.barrier()
        with torch.cuda.graph(graph, stream=s):
            graph_out = step()
        for _ in range(3):
            dist.barrier()
            graph.replay()
            torch.cuda.synchronize()
            torch.testing.assert_close(graph_out, expected, atol=2e-2, rtol=1e-2)
    dist.barrier()


def _spawned(rank, world, port):
    os.environ.update(
        RANK=str(rank),
        WORLD_SIZE=str(world),
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
        LOCAL_RANK=str(rank),
    )
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", device_id=torch.device("cuda", rank))
    try:
        worker(rank, world)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    not current_platform.is_device_capability_family(100)
    or torch.cuda.device_count() < 4,
    reason="needs 4 SM100 GPUs",
)
def test_ep_dispatch_combine_4_ranks():
    with socket.socket() as sock:
        sock.bind(("", 0))
        port = sock.getsockname()[1]
    torch.multiprocessing.spawn(_spawned, args=(4, port), nprocs=4, join=True)


def _rendezvous(init_file):
    """RANK / WORLD_SIZE / LOCAL_RANK / MASTER_ADDR / MASTER_PORT from srun's
    environment and a shared file (torchrun sets them itself)."""
    import time

    if "RANK" in os.environ or "SLURM_PROCID" not in os.environ:
        return
    rank = int(os.environ["SLURM_PROCID"])
    os.environ.update(
        RANK=str(rank),
        WORLD_SIZE=os.environ["SLURM_NTASKS"],
        LOCAL_RANK=os.environ["SLURM_LOCALID"],
    )
    if rank == 0:
        with socket.socket() as s:
            s.bind(("", 0))
            port = s.getsockname()[1]
        address = f"{socket.gethostbyname(socket.gethostname())}:{port}"
        with open(init_file + ".tmp", "w") as f:
            f.write(address)
        os.rename(init_file + ".tmp", init_file)
    else:
        while not os.path.exists(init_file):
            time.sleep(0.2)
        with open(init_file) as f:
            address = f.read()
    host, port = address.split(":")
    os.environ.update(MASTER_ADDR=host, MASTER_PORT=port)


if __name__ == "__main__":  # srun: one process per GPU, any number of nodes
    import sys

    _rendezvous(
        sys.argv[sys.argv.index("--init-file") + 1] if "--init-file" in sys.argv else ""
    )
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", device_id=torch.device("cuda", local_rank))
    worker(dist.get_rank(), dist.get_world_size())
    if dist.get_rank() == 0:
        print("EP COMM TEST PASSED", flush=True)
    dist.destroy_process_group()
