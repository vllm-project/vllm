# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The DeepSeek-V4.1 mono decode kernels on an index source: a layer with an
indexer of its own, where vLLM's indexer runs between the two launches and K2
builds the tokens' key table and key counts in the same launch that reads them.

One ``DSV41MonoLayer`` serves every layer, so the key-count buffer holds the
previous layer's counts when an index source's K2 starts. Every stage of that
K2 must read the counts this launch builds: the layer's outputs, after a
window-only layer (fewer keys) or with a window-only layer's counts planted in
the buffer, must be bit for bit those of the same layer with the key table K1
built (the layer as a non-index source).

Random weights and caches in the loaded formats, one request of M decode
tokens at a context past the top-k's 512 compressed entries, CDNA4."""

import os
import socket

import pytest
import torch

from vllm.platforms import current_platform

HIDDEN, HC, WINDOW, TOPK, BLOCK, EXPERTS = 5120, 4, 128, 512, 128, 384
CTX = 3000  # a request's first decode position


def _on_cdna4() -> bool:
    if not current_platform.is_rocm():
        return False
    from vllm.platforms.rocm import get_cdna_version

    return get_cdna_version() == 4


def _fp8(g, *shape):
    return (torch.randn(*shape, generator=g)).clamp(-200, 200).to(torch.float8_e4m3fn)


def _e8m0(g, *shape, lo=120, hi=126):
    return torch.randint(lo, hi, shape, generator=g, dtype=torch.uint8)


def _weights(tp, ratio, dev, seed):
    from vllm.models.deepseek_v41.amd.mono.attention.plan import Dims
    from vllm.models.deepseek_v41.amd.mono.runner import AttnWeights, MonoLayerWeights
    from vllm.models.deepseek_v41.amd.mono.stages.dims import Dims as MoeDims

    g = torch.Generator().manual_seed(seed)
    d, md = Dims(tp), MoeDims(tp)
    bf = torch.bfloat16

    def t(x):
        return x.to(dev).contiguous()

    def fp4(*shape):
        u8 = torch.randint(0, 256, shape, generator=g, dtype=torch.uint8)
        return t(u8.view(torch.float4_e2m1fn_x2))

    attn = AttnWeights(
        layer_id=seed,
        wqkv=t(_fp8(g, 1792, HIDDEN)),
        wqkv_scale=t(_e8m0(g, 56, HIDDEN // 32)),
        q_norm=t((torch.rand(1280, generator=g) + 0.5).to(bf)),
        kv_norm=t((torch.rand(512, generator=g) + 0.5).to(bf)),
        wq_b=t(_fp8(g, d.heads * 512, 1280)),
        wq_b_scale=t(_e8m0(g, d.heads * 16, 40)),
        wo_a=t(_fp8(g, d.groups * 1024, d.group_k)),
        wo_a_scale=t(_e8m0(g, d.groups * 32, d.group_k // 32)),
        wo_b=t(_fp8(g, HIDDEN, d.groups * 1024)),
        wo_b_scale=t(_e8m0(g, HIDDEN // 32, d.groups * 32)),
        attn_sink=t(torch.randn(64, generator=g)),
        cos_sin=t(torch.randn(CTX + 64, 64, generator=g).clamp(-1, 1)),
        ratio=ratio,
    )
    w = MonoLayerWeights(
        attn=attn,
        hc_attn_fn=t(torch.randn(24, HC * HIDDEN, generator=g) * 0.02),
        hc_attn_scale=t(torch.rand(3, generator=g) + 0.5),
        hc_attn_base=t(torch.randn(24, generator=g) * 0.1),
        attn_norm=t((torch.rand(HIDDEN, generator=g) + 0.5).to(bf)),
        hc_ffn_fn=t(torch.randn(24, HC * HIDDEN, generator=g) * 0.02),
        hc_ffn_scale=t(torch.rand(3, generator=g) + 0.5),
        hc_ffn_base=t(torch.randn(24, generator=g) * 0.1),
        ffn_norm=t((torch.rand(HIDDEN, generator=g) + 0.5).to(bf)),
        gate_w=t((torch.randn(EXPERTS, HIDDEN, generator=g) * 0.02).to(bf)),
        bias=t(torch.randn(EXPERTS, generator=g) * 0.05),
        w13=fp4(EXPERTS, 2 * md.inter, HIDDEN // 2),
        w13_s=t(_e8m0(g, EXPERTS * 2 * md.inter, HIDDEN // 32, lo=118, hi=122)),
        w2=fp4(EXPERTS, HIDDEN, md.inter // 2),
        w2_s=t(_e8m0(g, EXPERTS * HIDDEN, md.down_scale_cols, lo=118, hi=122)),
        sgu=t(_fp8(g, 2 * md.sh_inter, HIDDEN)),
        sgu_s=t(_e8m0(g, 2 * md.sh_inter // 32, HIDDEN // 32, lo=118, hi=122)),
        sw2=t(_fp8(g, HIDDEN, md.sh_inter)),
        sw2_s=t(_e8m0(g, HIDDEN // 32, md.sh_inter // 32, lo=118, hi=122)),
    )
    w.check(tp, EXPERTS)
    attn.check(d)
    return w


def _step(M, ratio, dev, seed):
    """A decode step of one request's M tokens from position CTX: the window
    rows, fp8_ds_mla records and, for ratio > 0, a compressed cache and the
    top-k vLLM's indexer would leave (min((p + 1) / ratio, 512) entries)."""
    g = torch.Generator().manual_seed(seed)
    nblocks = 64
    data = (torch.randn(nblocks, BLOCK, 576, generator=g) * 0.5).to(torch.float8_e4m3fn)
    data = data.view(torch.uint8)
    rope = (torch.randn(nblocks, BLOCK, 64, generator=g) * 0.5).to(torch.bfloat16)
    data[:, :, 448:] = rope.view(torch.uint8).reshape(nblocks, BLOCK, 128)
    scales = torch.full((nblocks, BLOCK * 8), 120, dtype=torch.uint8)
    rec = torch.cat([data.reshape(nblocks, -1), scales], 1).view(nblocks, BLOCK, 584)
    pos = torch.arange(CTX, CTX + M, dtype=torch.int64)
    ring = 16 * BLOCK
    swa_idx = torch.full((M, 1, WINDOW), -1, dtype=torch.int32)
    for i, p in enumerate(pos.tolist()):
        swa_idx[i, 0] = (torch.arange(p - WINDOW + 1, p + 1) % ring).to(torch.int32)
    s = dict(
        x=torch.randn(M, HIDDEN, generator=g).to(torch.bfloat16),
        residual=torch.randn(M, HC, HIDDEN, generator=g).to(torch.bfloat16),
        post_mix=torch.rand(M, HC, 1, generator=g) * 2,
        res_mix=torch.rand(M, HC, HC, generator=g) * 0.5,
        pre_mix=torch.rand(M, HC, generator=g) * 0.5,
        positions=pos,
        slot_mapping=pos % ring,
        swa_cache=rec,
        swa_indices=swa_idx,
        swa_lens=torch.full((M,), WINDOW, dtype=torch.int32),
        token_to_req=torch.zeros(M, dtype=torch.int32),
    )
    if ratio:
        topk = torch.full((M, TOPK), -1, dtype=torch.int32)
        for i, p in enumerate(pos.tolist()):
            n = (p + 1) // ratio
            topk[i, : min(n, TOPK)] = torch.randperm(n, generator=g)[:TOPK].int()
        nb = (CTX + M) // ratio // BLOCK + 1
        s.update(
            topk_indices=topk,
            comp_cache=rec.clone(),
            comp_block_table=(torch.arange(nb, dtype=torch.int32) % nblocks)[None],
        )
    return {k: v.to(dev).contiguous() for k, v in s.items()}


def _worker(rank, tp, port):
    import torch.distributed as dist

    os.environ["MASTER_ADDR"], os.environ["MASTER_PORT"] = "127.0.0.1", str(port)
    torch.cuda.set_device(rank)
    dist.init_process_group("gloo", rank=rank, world_size=tp)
    dev = torch.device("cuda", rank)
    from vllm.models.deepseek_v41.amd.mono.runner import DSV41MonoLayer

    runner = DSV41MonoLayer(tp=tp, rank=rank, group=dist.group.WORLD, device=dev)
    window_only = _weights(tp, 0, dev, seed=1)
    for ratio in (2, 1):
        w = _weights(tp, ratio, dev, seed=2 + ratio)
        for M in (1, 6):
            before = _step(M, 0, dev, seed=10 + M)
            step = _step(M, ratio, dev, seed=20 + M)
            # the layer with K1's key table: the reference
            want = [o.clone() for o in runner.forward(w, **step)]
            for trial in range(4):
                if trial % 2:
                    # a window-only layer's counts in the buffer
                    runner.klen[:M].fill_(WINDOW)
                    runner.klen[M : 2 * M].zero_()
                else:
                    # the previous layer, window only, as in DeepSeek-V4.1's
                    # layers 1 -> 2
                    runner.forward(window_only, **before)
                got = runner.forward(w, **step, select=lambda normed, qr: None)
                for name, a, b in zip(("x", "res", "post", "comb", "pre"), got, want):
                    assert torch.equal(a, b), (
                        f"rank {rank}, ratio {ratio}, {M} rows, trial {trial}: "
                        f"the index source's {name} differs from the layer's "
                        "with K1's key table"
                    )
    torch.accelerator.synchronize()
    dist.barrier()
    dist.destroy_process_group()


@pytest.mark.skipif(not _on_cdna4(), reason="the mono kernels' CDNA4 build")
@pytest.mark.parametrize("tp", [2, 4])
def test_index_source_reads_its_own_key_counts(tp):
    if torch.accelerator.device_count() < tp:
        pytest.skip(f"needs {tp} GPUs")
    pytest.importorskip("flydsl")
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    torch.multiprocessing.spawn(_worker, args=(tp, port), nprocs=tp, join=True)
