# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from ROCm/aiter#6173 at b02df0db8 (MIT License),
# Copyright (C) 2026, Advanced Micro Devices, Inc.:
# op_tests/multigpu_tests/test_flydsl_glm5_mono.py
# ruff: noqa: E501

"""Runtime checks for the GLM-5 fused decode-layer MonoKernel (``Glm5MonoKernel``).

Every rank is a spawn worker holding random weights in the production storage
contract: block-FP8 attention, AITER-shuffled MXFP4 experts with the shared
expert fused as the last physical expert, and an FP8 KV cache addressed by
global slot. Each case captures ``--layers`` layers in one HIP graph.

Checks, per decode-step size:
  * the graph completes and every output is finite;
  * two replays give bit-identical outputs;
  * CUDA-graph pad rows (slot -1, empty sparse range) write no cache entry and
    leave the real rows' outputs unchanged when the pad-row input changes.

A launch that waits on a missing peer never returns, so run under ``timeout``:
    timeout -s KILL 900 python3 tests/models/glm_mono/test_glm_mono_moe.py
"""

from __future__ import annotations

import argparse
import os
import sys

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from aiter.jit.utils.chip_info import get_gfx_runtime

from vllm.models.deepseek_v32.amd.mono import (
    AttentionWeight,
    Glm5MonoKernel,
    KvCacheLayout,
    LayerWeights,
    Mxfp4ScaleLayout,
    Mxfp4WeightLayout,
    glm5_mono_launch_rows,
    glm5_tp_config,
    prepare_glm5_weights,
)

TOPK = 2048
FP8 = torch.float8_e4m3fn


def _pad(n, m):
    return (n + m - 1) // m * m


def make_layer(cfg, rank, npes, dev):
    heads = cfg.local_heads
    phys = cfg.n_experts + cfg.num_shared_experts
    qb_rows = heads * (cfg.nope_dim + cfg.pe_dim)

    def fp8(rows, cols):
        return (torch.randn(rows, cols, device=dev) * 0.05).to(FP8)

    def block_scale(rows, cols, bk=128):
        return torch.full((_pad(rows, 128) // 128, cols // bk), 0.02, device=dev)

    def mx_values(rows, k):
        t = torch.randint(0, 256, (rows * k // 2,), dtype=torch.uint8, device=dev)
        t.is_shuffled = True
        return t

    def mx_scales(rows, k):
        n = _pad(rows, 256) * _pad(k // 32, 8)
        return torch.randint(118, 124, (n,), dtype=torch.uint8, device=dev)

    ones = lambda n: torch.ones(n, dtype=torch.bfloat16, device=dev)
    t = {
        "g_in": ones(cfg.hidden),
        "g_q": ones(cfg.q_lora),
        "g_kv": ones(cfg.kv_lora),
        "g_post": ones(cfg.hidden),
        "w_qkv_a": fp8(cfg.qkv_a_rows, cfg.hidden),
        "s_qkv_a": block_scale(cfg.qkv_a_rows, cfg.hidden),
        "w_q_b": fp8(qb_rows, cfg.q_lora),
        "s_q_b": block_scale(qb_rows, cfg.q_lora),
        "w_uk": fp8(heads * cfg.kv_lora, cfg.nope_dim),
        "s_uk": block_scale(heads * cfg.kv_lora, cfg.nope_dim, 64),
        "w_uv": fp8(heads * cfg.v_dim, cfg.kv_lora),
        "s_uv": block_scale(heads * cfg.v_dim, cfg.kv_lora),
        "w_o": fp8(cfg.hidden, heads * cfg.v_dim),
        "s_o": block_scale(cfg.hidden, heads * cfg.v_dim),
        "w_r": (torch.randn(cfg.n_experts, cfg.hidden, device=dev) * 0.02).to(
            torch.bfloat16
        ),
        "bias": torch.zeros(cfg.n_experts, dtype=torch.float32, device=dev),
        "w_ug": mx_values(phys * 2 * cfg.inter, cfg.hidden),
        "s_ug": mx_scales(phys * 2 * cfg.inter, cfg.hidden),
        "w_dn": mx_values(phys * cfg.hidden, cfg.inter),
        "s_dn": mx_scales(phys * cfg.hidden, cfg.inter),
    }
    return LayerWeights(
        heads,
        t,
        cfg,
        rank,
        npes,
        mxfp4_weight_layout=Mxfp4WeightLayout.ATOM,
        mxfp4_scale_layout=Mxfp4ScaleLayout.ATOM,
        physical_experts=phys,
    )


def step_inputs(real, padded, ctx, dev, gen):
    """One token per request at the end of a ``ctx``-token context; rows past
    ``real`` are pad rows with slot -1 and an empty sparse range."""
    pos = ctx - 1
    positions = torch.full((padded,), pos, dtype=torch.int64)
    slots = torch.full((padded,), -1, dtype=torch.int64)
    slots[:real] = torch.arange(real) * ctx + pos
    counts = torch.zeros(padded, dtype=torch.int32)
    counts[:real] = TOPK
    indptr = torch.zeros(padded + 1, dtype=torch.int32)
    indptr[1:] = torch.cumsum(counts, 0)
    picks = [
        r * ctx + torch.randperm(pos, generator=gen)[:TOPK].sort().values
        for r in range(real)
    ]
    indices = torch.cat(picks).to(torch.int32)
    return [x.to(dev) for x in (positions, slots, indptr, indices)]


def worker(rank, args, port):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    torch.cuda.set_device(rank)
    dev = torch.device("cuda", rank)
    dist.init_process_group("gloo", rank=rank, world_size=args.tp)
    group = dist.new_group(list(range(args.tp)), backend="gloo")
    cfg = glm5_tp_config(args.tp)
    torch.manual_seed(1234 + rank)
    gen = torch.Generator().manual_seed(42)

    layers = [make_layer(cfg, rank, args.tp, dev) for _ in range(args.layers)]
    prepared = [prepare_glm5_weights(w, AttentionWeight.FP8_BLOCK128) for w in layers]
    kv = (
        torch.randn(max(args.rows) * args.ctx, cfg.kv_lora + cfg.pe_dim, device=dev)
        * 0.1
    ).to(FP8)
    angles = torch.rand(args.ctx, cfg.pe_dim // 2, device=dev) * 6.28
    cos, sin = angles.cos().to(torch.bfloat16), angles.sin().to(torch.bfloat16)
    cur_pos = torch.zeros(1, dtype=torch.int32, device=dev)
    ops = {}
    failures = []

    def run_case(real, padded, label):
        _, chunk = glm5_mono_launch_rows(padded)
        if chunk not in ops:
            ops[chunk] = Glm5MonoKernel(
                layers[0],
                chunk,
                rank=rank,
                npes=args.tp,
                group=group,
                topk=TOPK,
                launches_per_step=1,
                with_indexer=False,
                attention_weight=AttentionWeight.FP8_BLOCK128,
                kv_cache_layout=KvCacheLayout.ATOM,
                kv_cache_dtype="fp8",
                prepared_weights=prepared[0],
                native_fp4_mfma=True,
            )
        op = ops[chunk]
        positions, slots, indptr, indices = step_inputs(
            real, padded, args.ctx, dev, gen
        )
        h = (torch.randn(padded, cfg.hidden, device=dev) * 0.5).to(torch.bfloat16)
        bufs = [h.clone(), torch.empty_like(h)]
        last = bufs[args.layers % 2]

        def step():
            for i, (w, p) in enumerate(zip(layers, prepared)):
                op.W, op.packed = w, p
                op.forward(
                    bufs[i % 2],
                    cur_pos,
                    kv,
                    kv,
                    indices,
                    cos,
                    sin,
                    x_out=bufs[(i + 1) % 2],
                    positions=positions,
                    slot_mapping=slots,
                    sparse_kv_indptr=indptr,
                )

        step()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            step()
        outputs = []
        for pad_noise in (0.0, 1.0):
            bufs[0].copy_(h)
            if padded > real:
                bufs[0][real:] += pad_noise
            dist.barrier(group)
            graph.replay()
            torch.cuda.synchronize()
            outputs.append(last[:real].clone())
        finite = bool(torch.isfinite(outputs[0].float()).all())
        same = torch.equal(outputs[0], outputs[1])
        ok = finite and same
        if rank == 0:
            what = "pad-row input changed" if padded > real else "replayed"
            print(
                f"[{'PASS' if ok else 'FAIL'}] {label}: rows={real} launched={padded} chunk={chunk} "
                f"finite={finite} identical_when_{what.replace(' ', '_')}={same}",
                flush=True,
            )
        if not ok:
            failures.append(label)
        del graph

    for rows in args.rows:
        padded, _ = glm5_mono_launch_rows(rows)
        run_case(rows, padded, f"decode {rows} row(s)")
    run_case(2, 4, "pad rows (2 real + 2 pad)")

    for op in ops.values():
        op.close()
    dist.destroy_process_group()
    if failures:
        raise SystemExit(f"rank {rank}: failed {failures}")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--tp", type=int, default=4)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--ctx", type=int, default=8192)
    parser.add_argument("--rows", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    parser.add_argument("--port", type=int, default=29731)
    args = parser.parse_args(argv)
    if get_gfx_runtime() != "gfx950":
        print("skip: Glm5MonoKernel targets gfx950")
        return 0
    if torch.cuda.device_count() < args.tp:
        print(f"skip: needs {args.tp} GPUs, found {torch.cuda.device_count()}")
        return 0
    mp.spawn(worker, args=(args, args.port), nprocs=args.tp, join=True)
    return 0


def test_moe_layer():
    if get_gfx_runtime() != "gfx950" or torch.cuda.device_count() < 4:
        pytest.skip("needs 4 gfx950 GPUs")
    assert main(["--tp", "4"]) == 0


if __name__ == "__main__":
    sys.exit(main())
