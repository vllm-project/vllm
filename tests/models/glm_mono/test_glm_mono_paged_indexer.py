# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from ROCm/aiter#6173 at b02df0db8 (MIT License),
# Copyright (C) 2026, Advanced Micro Devices, Inc.:
# op_tests/multigpu_tests/test_flydsl_glm5_mono_paged_indexer.py
# ruff: noqa: E501

"""GLM-5 MonoKernel fused indexer on a vLLM paged FP8 index cache.

Every row is its own request with its own context length and a scattered block
table. Existing index keys sit in vLLM's paged layout (E4M3 values plus one
power-of-two scale per token, optionally 16x16 shuffled); the launch appends each
row's new key, scores every key, selects the exact top-2048 and attends to it.

Checks, per rank:
  * each new key's cache bytes and scale match vLLM's ue8m0 quantization;
  * the published slots are the top-2048 of a PyTorch reference score;
  * the layer output equals the external-indices launch fed those slots;
  * pad rows (slot -1, empty sparse range) publish nothing.

    timeout -s KILL 1800 python3 tests/models/glm_mono/test_glm_mono_paged_indexer.py --tp 8
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
    glm5_tp_config,
    prepare_glm5_weights,
)

TOPK = 2048
FP8 = torch.float8_e4m3fn
BLOCK = 64
INDEX_DIM = 128
INDEX_HEADS = 32
BLOCK_BYTES = BLOCK * (INDEX_DIM + 4)


def _pad(n, m):
    return (n + m - 1) // m * m


def make_layer(cfg, rank, npes, dev):
    heads = cfg.local_heads
    phys = cfg.n_experts + cfg.num_shared_experts
    qb_rows = heads * (cfg.nope_dim + cfg.pe_dim)
    g = torch.Generator(device=dev).manual_seed(1234 + rank)
    shared = torch.Generator(device=dev).manual_seed(77)

    def fp8(rows, cols, gen=g):
        return (torch.randn(rows, cols, device=dev, generator=gen) * 0.05).to(FP8)

    def block_scale(rows, cols, bk=128):
        return torch.full((_pad(rows, 128) // 128, cols // bk), 0.02, device=dev)

    def mx_values(rows, k):
        t = torch.randint(
            0, 256, (rows * k // 2,), dtype=torch.uint8, device=dev, generator=g
        )
        t.is_shuffled = True
        return t

    def mx_scales(rows, k):
        n = _pad(rows, 256) * _pad(k // 32, 8)
        return torch.randint(118, 124, (n,), dtype=torch.uint8, device=dev, generator=g)

    ones = lambda n: torch.ones(n, dtype=torch.bfloat16, device=dev)
    t = {
        "g_in": ones(cfg.hidden),
        "g_q": ones(cfg.q_lora),
        "g_kv": ones(cfg.kv_lora),
        "g_post": ones(cfg.hidden),
        "w_qkv_a": fp8(cfg.qkv_a_rows, cfg.hidden, shared),
        "s_qkv_a": block_scale(cfg.qkv_a_rows, cfg.hidden),
        "w_q_b": fp8(qb_rows, cfg.q_lora),
        "s_q_b": block_scale(qb_rows, cfg.q_lora),
        "w_uk": fp8(heads * cfg.kv_lora, cfg.nope_dim),
        "s_uk": block_scale(heads * cfg.kv_lora, cfg.nope_dim, 64),
        "w_uv": fp8(heads * cfg.v_dim, cfg.kv_lora),
        "s_uv": block_scale(heads * cfg.v_dim, cfg.kv_lora),
        "w_o": fp8(cfg.hidden, heads * cfg.v_dim),
        "s_o": block_scale(cfg.hidden, heads * cfg.v_dim),
        "w_r": (
            torch.randn(cfg.n_experts, cfg.hidden, device=dev, generator=shared) * 0.02
        ).to(torch.bfloat16),
        "bias": torch.zeros(cfg.n_experts, dtype=torch.float32, device=dev),
        "w_ug": mx_values(phys * 2 * cfg.inter, cfg.hidden),
        "s_ug": mx_scales(phys * 2 * cfg.inter, cfg.hidden),
        "w_dn": mx_values(phys * cfg.hidden, cfg.inter),
        "s_dn": mx_scales(phys * cfg.hidden, cfg.inter),
        # vLLM keeps wk / weights_proj in bf16 and wq_b in block FP8; the
        # indexer is replicated, so every rank holds the same values.
        "w_index_k": (
            torch.randn(INDEX_DIM, cfg.hidden, device=dev, generator=shared) * 0.02
        ).to(torch.bfloat16),
        "w_index_w": (
            torch.randn(INDEX_HEADS, cfg.hidden, device=dev, generator=shared)
            / cfg.hidden**0.5
        ).to(torch.bfloat16),
        "w_index_q": fp8(INDEX_HEADS * INDEX_DIM, cfg.q_lora, shared),
        "s_index_q": block_scale(INDEX_HEADS * INDEX_DIM, cfg.q_lora),
        "g_index_k": 1 + 0.1 * torch.randn(INDEX_DIM, device=dev, generator=shared),
        "b_index_k": 0.1 * torch.randn(INDEX_DIM, device=dev, generator=shared),
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


def quant_ue8m0(v):
    amax = v.abs().amax(-1, keepdim=True).clamp_min(1e-4)
    scale = torch.exp2(torch.ceil(torch.log2(amax / 448.0)))
    return (v / scale).to(FP8), scale.squeeze(-1)


def value_offsets(off, shuffled):
    d = torch.arange(INDEX_DIM, device=off.device)
    off = off[:, None]
    if shuffled:
        return (
            (off // 16) * (16 * INDEX_DIM) + (off % 16) * 16 + (d // 16) * 256 + d % 16
        )
    return off * INDEX_DIM + d


def write_keys(cache, blocks, offs, values, scales, shuffled):
    flat = cache.view(cache.shape[0], -1)
    idx = value_offsets(offs, shuffled)
    flat[blocks[:, None], idx] = values.view(torch.uint8)
    flat[:, BLOCK * INDEX_DIM :].view(torch.float32)[blocks, offs] = scales


def read_keys(cache, blocks, offs, shuffled):
    flat = cache.view(cache.shape[0], -1)
    values = flat[blocks[:, None], value_offsets(offs, shuffled)].view(FP8)
    scales = flat[:, BLOCK * INDEX_DIM :].view(torch.float32)[blocks, offs]
    return values, scales


def rope_interleaved(x, cos, sin):
    x = x.clone()
    x0, x1 = x[..., 0:64:2].clone(), x[..., 1:64:2].clone()
    x[..., 0:64:2], x[..., 1:64:2] = x0 * cos - x1 * sin, x0 * sin + x1 * cos
    return x


def worker(rank, args, port):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    torch.cuda.set_device(rank)
    dev = torch.device("cuda", rank)
    dist.init_process_group("gloo", rank=rank, world_size=args.tp)
    group = dist.new_group(list(range(args.tp)), backend="gloo")
    cfg = glm5_tp_config(args.tp)
    layer = make_layer(cfg, rank, args.tp, dev)
    prepared = prepare_glm5_weights(layer, AttentionWeight.FP8_BLOCK128)
    gen = torch.Generator(device=dev).manual_seed(11)

    ctxs = args.ctx
    real = len(ctxs)
    rows = real + args.pad
    max_ctx = max(ctxs)
    bt_stride = _pad(max_ctx, BLOCK) // BLOCK + 1
    total_blocks = sum(_pad(c, BLOCK) // BLOCK for c in ctxs) + 7
    perm = torch.randperm(total_blocks, generator=gen, device=dev).to(torch.int32)
    block_table = torch.zeros(rows + 1, bt_stride, dtype=torch.int32, device=dev)
    used = 0
    for r, c in enumerate(ctxs):
        n = _pad(c, BLOCK) // BLOCK
        block_table[r, :n] = perm[used : used + n]
        used += n

    shuffled = args.shuffled
    index_cache = torch.zeros(
        total_blocks, BLOCK, INDEX_DIM + 4, dtype=torch.uint8, device=dev
    )
    for r, c in enumerate(ctxs):
        p = torch.arange(c - 1, device=dev)
        blocks = block_table[r, p // BLOCK].long()
        k = torch.randn(c - 1, INDEX_DIM, device=dev, generator=gen) * 0.5
        if args.cluster:
            base = torch.randn(1, INDEX_DIM, device=dev, generator=gen) * 0.5
            k = base + args.cluster * k
        if args.zero_frac:
            k[torch.rand(c - 1, device=dev, generator=gen) < args.zero_frac] = 0
        vals, scales = quant_ue8m0(k)
        write_keys(index_cache, blocks, p % BLOCK, vals, scales, shuffled)

    kv = (
        torch.randn(
            total_blocks * BLOCK, cfg.kv_lora + cfg.pe_dim, device=dev, generator=gen
        )
        * 0.1
    ).to(FP8)
    angles = torch.rand(max_ctx, cfg.pe_dim // 2, device=dev, generator=gen) * 6.28
    cos, sin = angles.cos().to(torch.bfloat16), angles.sin().to(torch.bfloat16)
    h = (torch.randn(rows, cfg.hidden, device=dev, generator=gen) * 0.5).to(
        torch.bfloat16
    )

    positions = torch.zeros(rows, dtype=torch.int64, device=dev)
    slots = torch.full((rows,), -1, dtype=torch.int64, device=dev)
    counts = torch.zeros(rows, dtype=torch.int32, device=dev)
    for r, c in enumerate(ctxs):
        positions[r] = c - 1
        slots[r] = block_table[r, (c - 1) // BLOCK].long() * BLOCK + (c - 1) % BLOCK
        counts[r] = min(c, TOPK)
    indptr = torch.zeros(rows + 1, dtype=torch.int32, device=dev)
    indptr[1:] = torch.cumsum(counts, 0)
    req_ids = torch.arange(rows, dtype=torch.int32, device=dev)
    out_indices = torch.full((int(indptr[-1]) + 16,), -7, dtype=torch.int32, device=dev)
    cur_pos = torch.zeros(1, dtype=torch.int32, device=dev)
    common = dict(
        rank=rank,
        npes=args.tp,
        group=group,
        topk=TOPK,
        launches_per_step=1,
        attention_weight=AttentionWeight.FP8_BLOCK128,
        kv_cache_layout=KvCacheLayout.ATOM,
        kv_cache_dtype="fp8",
        prepared_weights=prepared,
        native_fp4_mfma=True,
    )

    kv0 = kv.clone()
    op_fi = Glm5MonoKernel(
        layer,
        rows,
        with_indexer=True,
        index_max_seq=_pad(max_ctx, 64),
        index_paged=True,
        index_block_size=BLOCK,
        index_block_bytes=BLOCK_BYTES,
        index_shuffled=shuffled,
        block_table_stride=bt_stride,
        index_k_bf16=True,
        **common,
    )
    out_fi = op_fi.forward(
        h,
        cur_pos,
        kv,
        kv,
        None,
        cos,
        sin,
        positions=positions,
        slot_mapping=slots,
        sparse_kv_indptr=indptr,
        index_cache=index_cache,
        block_table=block_table,
        req_ids=req_ids,
        out_indices=out_indices,
    )
    torch.cuda.synchronize()
    mid = op_fi.intermediates()
    op_fi.close()

    failures = []
    inv = {}
    for r, c in enumerate(ctxs):
        p = torch.arange(c, device=dev)
        inv[r] = (block_table[r, p // BLOCK].long() * BLOCK + p % BLOCK, p)

    for r, c in enumerate(ctxs):
        pos = c - 1
        block = block_table[r, pos // BLOCK].long().view(1)
        off = torch.tensor([pos % BLOCK], device=dev)
        got_v, got_s = read_keys(index_cache, block, off, shuffled)
        k = mid["index_k"][r].float()
        k = (k - k.mean()) / torch.sqrt(k.var(unbiased=False) + 1e-6)
        k = k * layer.t["g_index_k"] + layer.t["b_index_k"]
        k = rope_interleaved(k, cos[pos].float(), sin[pos].float())
        ref_v, ref_s = quant_ue8m0(k.view(1, -1))
        same_scale = bool(torch.equal(got_s, ref_s))
        byte_match = (
            (got_v.view(torch.uint8) == ref_v.view(torch.uint8)).float().mean().item()
        )

        p_all = torch.arange(c, device=dev)
        vals, scales = read_keys(
            index_cache, block_table[r, p_all // BLOCK].long(), p_all % BLOCK, shuffled
        )
        keys_f = vals.float() * scales[:, None]
        q = rope_interleaved(
            mid["index_q"][r].float(), cos[pos].float(), sin[pos].float()
        )
        q = q.to(torch.bfloat16).float()
        score = (torch.relu(q @ keys_f.T) * mid["index_w"][r].float()[:, None]).sum(0)
        n = min(c, TOPK)
        ref_values, ref_idx = torch.topk(score, n)
        ref = set(ref_idx.tolist())
        got_slots = out_indices[indptr[r] : indptr[r] + n].long()
        slot_to_pos = torch.full(
            (total_blocks * BLOCK,), -1, dtype=torch.long, device=dev
        )
        slot_to_pos[inv[r][0]] = inv[r][1]
        got = slot_to_pos[got_slots].tolist()
        overlap = len(ref & set(got)) / n
        distinct = len(set(got)) == n and min(got) >= 0
        # Ties make the picked set ambiguous; the picked score values are not.
        got_values = score[torch.tensor(got, device=dev).clamp_min(0)]
        got_values = got_values.sort(descending=True).values
        value_err = (
            (got_values - ref_values).abs().max()
            / ref_values.abs().max().clamp_min(1e-30)
        ).item()
        ok = same_scale and byte_match >= 0.99 and value_err <= 1e-4 and distinct
        if rank == 0 or not ok:
            print(
                f"[{'PASS' if ok else 'FAIL'}] rank {rank} row {r} ctx {c}: new-key scale exact={same_scale} "
                f"bytes {byte_match:.4f} | picks distinct={distinct} overlap {overlap:.4f} "
                f"top-{n} value err {value_err:.1e}",
                flush=True,
            )
        if not ok:
            failures.append(f"row {r}")
    tail = out_indices[int(indptr[-1]) :]
    if not bool((tail == -7).all()):
        failures.append("pad rows wrote past the sparse range")

    kv.copy_(kv0)
    op_ext = Glm5MonoKernel(layer, rows, with_indexer=False, **common)
    out_ext = op_ext.forward(
        h,
        cur_pos,
        kv,
        kv,
        out_indices[: int(indptr[-1])].contiguous(),
        cos,
        sin,
        positions=positions,
        slot_mapping=slots,
        sparse_kv_indptr=indptr,
    )
    torch.cuda.synchronize()
    op_ext.close()
    same = torch.equal(out_fi[:real], out_ext[:real])
    finite = bool(torch.isfinite(out_fi[:real].float()).all())
    if rank == 0 or not (same and finite):
        print(
            f"[{'PASS' if same and finite else 'FAIL'}] rank {rank}: fused vs external output identical={same} finite={finite}",
            flush=True,
        )
    if not (same and finite):
        failures.append("output")
    dist.destroy_process_group()
    if failures:
        raise SystemExit(f"rank {rank}: failed {failures}")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--tp", type=int, default=8)
    parser.add_argument(
        "--ctx", type=int, nargs="+", default=[1000, 8192, 30000, 70001]
    )
    parser.add_argument("--pad", type=int, default=0)
    parser.add_argument(
        "--shuffled", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--cluster", type=float, default=0.0)
    parser.add_argument("--zero-frac", type=float, default=0.0)
    parser.add_argument("--port", type=int, default=29741)
    args = parser.parse_args(argv)
    if get_gfx_runtime() != "gfx950":
        print("skip: Glm5MonoKernel targets gfx950")
        return 0
    if torch.cuda.device_count() < args.tp:
        print(f"skip: needs {args.tp} GPUs, found {torch.cuda.device_count()}")
        return 0
    mp.spawn(worker, args=(args, args.port), nprocs=args.tp, join=True)
    return 0


def test_paged_indexer():
    if get_gfx_runtime() != "gfx950" or torch.cuda.device_count() < 4:
        pytest.skip("needs 4 gfx950 GPUs")
    assert main(["--tp", "4"]) == 0


if __name__ == "__main__":
    sys.exit(main())
