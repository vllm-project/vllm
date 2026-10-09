# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from ROCm/aiter#6173 at b02df0db8 (MIT License),
# Copyright (C) 2026, Advanced Micro Devices, Inc.:
# op_tests/multigpu_tests/test_flydsl_glm5_mono_dense.py

"""Dense-MLP layers in the GLM-5 MonoKernel (``Glm5MonoKernel(dense_experts=6)``).

GLM-5's leading dense layers have a 12288-wide MLP, six times the expert width.
``pack_dense_mlp`` stores it as six expert-shaped slices and the kernel routes
every row to all six with weight 1. Each rank holds random row-major MXFP4 dense
weights; the check compares the kernel's MLP output (``x_out - a``, reduced over
the TP ranks) with a float32 reference on the same dequantized weights:
  * finite, with relative L2 error below ``--tol-q`` against a reference that
    applies the kernel's MXFP8 activation quantization and BF16 roundings, and
    below ``--tol`` against the plain float32 reference;
  * two replays give bit-identical outputs.

    timeout -s KILL 900 python3 tests/models/glm_mono/test_glm_mono_dense.py
"""

from __future__ import annotations

import argparse
import os
import sys

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from aiter.jit.utils.chip_info import get_gfx_runtime  # noqa: E402
from aiter.utility.fp4_utils import e8m0_to_f32, mxfp4_to_f32  # noqa: E402
from test_glm_mono_moe import TOPK, make_layer, step_inputs  # noqa: E402

from vllm.models.deepseek_v32.amd.mono import (  # noqa: E402
    AttentionWeight,
    Glm5MonoKernel,
    KvCacheLayout,
    LayerWeights,
    Mxfp4ScaleLayout,
    Mxfp4WeightLayout,
    glm5_mono_launch_rows,
    glm5_tp_config,
    pack_dense_mlp,
    prepare_glm5_weights,
)

SLICES = 6
EPS = 1e-5


def dequant(values, scales):
    return mxfp4_to_f32(values) * e8m0_to_f32(scales).repeat_interleave(32, dim=-1)


def make_dense_layer(cfg, rank, npes, dev):
    inter = SLICES * cfg.inter

    def mx(rows, k):
        v = torch.randint(0, 256, (rows, k // 2), dtype=torch.uint8, device=dev)
        s = torch.randint(118, 124, (rows, k // 32), dtype=torch.uint8, device=dev)
        return v, s

    gate_up, gate_up_scale = mx(2 * inter, cfg.hidden)
    down, down_scale = mx(cfg.hidden, inter)
    base = make_layer(cfg, rank, npes, dev)
    t = dict(base.t)
    t.update(pack_dense_mlp(gate_up, gate_up_scale, down, down_scale, SLICES))
    t["w_r"] = torch.zeros(16, cfg.hidden, dtype=torch.bfloat16, device=dev)
    reference = (dequant(gate_up, gate_up_scale), dequant(down, down_scale))
    layer = LayerWeights(
        cfg.local_heads,
        t,
        cfg,
        rank,
        npes,
        mxfp4_weight_layout=Mxfp4WeightLayout.ATOM,
        mxfp4_scale_layout=Mxfp4ScaleLayout.ATOM,
        physical_experts=SLICES,
    )
    return layer, reference


def mxfp8(x):
    """The kernel's activation quantization: per 32 values, a power-of-two scale
    rounded up from amax / 448, then saturating E4M3."""
    g = x.view(*x.shape[:-1], -1, 32)
    mant, exp = torch.frexp(g.abs().amax(-1, keepdim=True) / 448.0)
    scale = torch.ldexp(torch.ones_like(mant), torch.where(mant == 0.5, exp - 1, exp))
    q = (g / scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn).float()
    return (q * scale).view_as(x)


def mlp_reference(a, g_post, gate_up, down, quant=False):
    x = a * torch.rsqrt(a.pow(2).mean(-1, keepdim=True) + EPS) * g_post.float()
    if quant:
        x = mxfp8(x)
    gate, up = (x @ gate_up.T).chunk(2, dim=-1)
    mid = torch.nn.functional.silu(gate) * up
    return (mxfp8(mid) if quant else mid) @ down.T


def worker(rank, args, port):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    torch.cuda.set_device(rank)
    dev = torch.device("cuda", rank)
    dist.init_process_group("gloo", rank=rank, world_size=args.tp)
    group = dist.new_group(list(range(args.tp)), backend="gloo")
    cfg = glm5_tp_config(args.tp)
    torch.manual_seed(4321 + rank)
    gen = torch.Generator().manual_seed(7)

    layer, (gate_up, down) = make_dense_layer(cfg, rank, args.tp, dev)
    prepared = prepare_glm5_weights(layer, AttentionWeight.FP8_BLOCK128)
    kv = (
        torch.randn(max(args.rows) * args.ctx, cfg.kv_lora + cfg.pe_dim, device=dev)
        * 0.1
    ).to(torch.float8_e4m3fn)
    angles = torch.rand(args.ctx, cfg.pe_dim // 2, device=dev) * 6.28
    cos, sin = angles.cos().to(torch.bfloat16), angles.sin().to(torch.bfloat16)
    cur_pos = torch.zeros(1, dtype=torch.int32, device=dev)
    failures = []

    for rows in args.rows:
        padded, chunk = glm5_mono_launch_rows(rows)
        op = Glm5MonoKernel(
            layer,
            chunk,
            rank=rank,
            npes=args.tp,
            group=group,
            topk=TOPK,
            attention_weight=AttentionWeight.FP8_BLOCK128,
            kv_cache_layout=KvCacheLayout.ATOM,
            kv_cache_dtype="fp8",
            prepared_weights=prepared,
            native_fp4_mfma=True,
            dense_experts=SLICES,
        )
        positions, slots, indptr, indices = step_inputs(
            padded, padded, args.ctx, dev, gen
        )
        h = (torch.randn(padded, cfg.hidden, device=dev) * 0.5).to(torch.bfloat16)
        outs = []
        for _ in range(2):
            dist.barrier(group)
            out = op.forward(
                h,
                cur_pos,
                kv,
                kv,
                indices,
                cos,
                sin,
                positions=positions,
                slot_mapping=slots,
                sparse_kv_indptr=indptr,
            )
            torch.cuda.synchronize()
            outs.append(out.clone())
        a = op.debug("a", (padded, cfg.hidden), bf2=True)
        partial = mlp_reference(a, layer.t["g_post"], gate_up, down).cpu()
        dist.all_reduce(partial, group=group)
        # Same activation quantization, BF16 peer partials and BF16 output.
        partial_q = mlp_reference(a, layer.t["g_post"], gate_up, down, quant=True)
        partial_q = partial_q.to(torch.bfloat16).float().cpu()
        dist.all_reduce(partial_q, group=group)
        expect = (a.cpu() + partial_q).to(torch.bfloat16).float() - a.cpu()
        got = (outs[0].float() - a).cpu()
        err = float((got - partial).norm() / partial.norm())
        err_q = float((got - expect).norm() / expect.norm())
        finite = bool(torch.isfinite(outs[0].float()).all())
        same = torch.equal(outs[0], outs[1])
        ok = finite and same and err < args.tol and err_q < args.tol_q
        if rank == 0:
            print(
                f"[{'PASS' if ok else 'FAIL'}] dense rows={rows} chunk={chunk} "
                f"finite={finite} replay_identical={same} rel_err={err:.4f} "
                f"rel_err_vs_quantized_ref={err_q:.4f}",
                flush=True,
            )
        if not ok:
            failures.append(rows)
        op.close()

    dist.destroy_process_group()
    if failures:
        raise SystemExit(f"rank {rank}: failed rows {failures}")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--tp", type=int, default=4)
    parser.add_argument("--ctx", type=int, default=8192)
    parser.add_argument("--rows", type=int, nargs="+", default=[2, 4, 8])
    parser.add_argument("--tol", type=float, default=0.08)
    parser.add_argument("--tol-q", type=float, default=0.01)
    parser.add_argument("--port", type=int, default=29751)
    args = parser.parse_args(argv)
    if get_gfx_runtime() != "gfx950":
        print("skip: Glm5MonoKernel targets gfx950")
        return 0
    if torch.cuda.device_count() < args.tp:
        print(f"skip: needs {args.tp} GPUs, found {torch.cuda.device_count()}")
        return 0
    mp.spawn(worker, args=(args, args.port), nprocs=args.tp, join=True)
    return 0


def test_dense_layer():
    if get_gfx_runtime() != "gfx950" or torch.cuda.device_count() < 4:
        pytest.skip("needs 4 gfx950 GPUs")
    assert main(["--tp", "4"]) == 0


if __name__ == "__main__":
    sys.exit(main())
