# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The fused one- and two-shot all-reduce + short conv + AttnRes mix kernel
against the unfused kernels, and the sequence-parallel reduce-scatter (or MoE
combine push) + conv + mix + all-gather against the unsharded conv + mix
kernel, across ranks."""

import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

DIM, BLOCK, HEADS, OFFSET = 4608, 16, 2, 256
WIDTH = DIM // HEADS


def _gather(x, world):
    out = [torch.empty_like(x) for _ in range(world)]
    dist.all_gather(out, x)
    return out


def _worker(rank: int, world: int, port: int, queue) -> None:
    from vllm.models.inkling2.ops.allreduce import (
        Inkling2AllReduceWorkspace,
        all_reduce_conv_mix,
    )
    from vllm.models.inkling2.ops.attn_res import attn_res
    from vllm.models.inkling2.ops.moe import combine as moe_combine
    from vllm.models.inkling2.ops.sconv import short_conv

    os.environ.update(MASTER_ADDR="localhost", MASTER_PORT=str(port))
    torch.accelerator.set_device_index(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world)
    workspace = Inkling2AllReduceWorkspace.create(dist.new_group(backend="gloo"), DIM)
    assert workspace is not None
    failures = []
    # (requests, tokens per request, MoE side, stack blocks, queries, last)
    cases = [
        (1, 1, False, 3, 1, True),
        (1, 8, False, 8, 1, False),
        (4, 8, False, 8, 1, False),
        (8, 1, True, 2, 3, False),
        (2, 8, True, 5, 1, False),
        (16, 8, True, 4, 1, False),
        (8, 8, False, 6, 3, True),
    ]
    for i, (reqs, qlen, moe, num_stack, num_q, with_last) in enumerate(cases * 2):
        tokens = reqs * qlen
        torch.manual_seed(i)  # shared by all ranks
        cache = torch.randn(reqs * 8 + 1, HEADS, BLOCK, OFFSET + WIDTH, device="cuda")
        cache = cache.bfloat16()
        weight = torch.randn(DIM, 4, device="cuda").bfloat16()
        stack = torch.randn(num_stack, tokens, DIM, device="cuda").bfloat16()
        queries = torch.randn(num_q, DIM, device="cuda") * DIM**-0.5
        norm_w = torch.randn(num_q, DIM, device="cuda").bfloat16()
        last = torch.randn(tokens, DIM, device="cuda").bfloat16() if with_last else None
        addend = torch.randn(tokens, DIM, device="cuda").bfloat16() if moe else None
        residual = torch.randn(tokens, DIM, device="cuda").bfloat16() if moe else None
        start = torch.randint(0, 30, (reqs,))
        start[0] = 1
        pos = torch.cat([s + torch.arange(qlen) for s in start]).cuda()
        seq = torch.arange(reqs).repeat_interleave(qlen).cuda()
        table = torch.randperm(reqs * 8).view(reqs, 8).int().cuda() + 1
        slots = table[seq, pos // BLOCK].long() * BLOCK + pos % BLOCK
        meta = (pos, table, seq.int(), slots, (seq * qlen).int())

        torch.manual_seed(100 + 7 * i + rank)  # per-rank partial sums
        if moe:
            down = torch.randn(tokens * 8, DIM, device="cuda").bfloat16()
            src2dst = torch.randperm(tokens * 8, device="cuda").int()
            src2dst[torch.rand(tokens * 8, device="cuda") < 0.6] = -1
            x, combined = None, (down, src2dst)
            # Each rank's fp32 combine is rounded once to bf16.
            partial = moe_combine(down, src2dst, 8, torch.float32).bfloat16().float()
        else:
            x, combined = torch.randn(tokens, DIM, device="cuda").bfloat16(), None
            partial = x.float()
        reduced = torch.zeros_like(partial)
        for p in _gather(partial, world):
            reduced += p
        ref_cache = cache.clone()
        ref_conv = residual.clone() if residual is not None else None
        ref_conv = short_conv(
            reduced, weight, ref_cache, *meta, OFFSET, WIDTH, addend, ref_conv, ref_conv
        )
        if last is None:
            ref_mix = attn_res(stack, ref_conv, None, queries, norm_w, 1e-6, True)
        else:
            ref_mix = attn_res(stack, last, ref_conv, queries, norm_w, 1e-6, True)

        for two_shot in (False, True):
            if tokens > workspace.max_tokens(two_shot):
                continue
            outs = []
            for _ in range(2):
                out_cache = cache.clone()
                conv = (
                    residual.clone()
                    if residual is not None
                    else torch.empty_like(ref_conv)
                )
                mix = all_reduce_conv_mix(
                    workspace,
                    x,
                    combined,
                    weight,
                    out_cache,
                    *meta,
                    OFFSET,
                    WIDTH,
                    conv,
                    conv if moe else None,
                    addend,
                    stack,
                    last,
                    queries,
                    norm_w,
                    1e-6,
                    two_shot=two_shot,
                )
                outs.append((mix, conv, out_cache))
            mix, conv, out_cache = outs[0]
            gathered = _gather(mix, world)
            checks = {
                "deterministic": all(torch.equal(a, b) for a, b in zip(*outs)),
                "same on all ranks": all(torch.equal(m, mix) for m in gathered),
                "conv": torch.equal(conv, ref_conv),
                "cache": torch.equal(out_cache, ref_cache),
                # Only the fp32 summation order of the mix statistics differs.
                "mix": torch.allclose(
                    mix.float(), ref_mix.float(), atol=1e-2, rtol=1e-2
                ),
            }
            failures += [
                f"case {i} two_shot={two_shot}: {k}"
                for k, ok in checks.items()
                if not ok
            ]
    queue.put((rank, failures))
    dist.destroy_process_group()


@pytest.mark.parametrize("world", [2, 4])
def test_fused_all_reduce_matches_unfused(world):
    if torch.accelerator.device_count() < world:
        pytest.skip(f"needs {world} GPUs")
    queue = mp.get_context("spawn").Queue()
    port = 29600 + world
    mp.spawn(_worker, args=(world, port, queue), nprocs=world)
    results = dict(queue.get() for _ in range(world))
    assert all(not f for f in results.values()), results


def _seq_parallel_worker(rank: int, world: int, port: int, queue) -> None:
    import vllm._C_stable_libtorch  # noqa: F401

    from vllm.models.inkling2.ops.conv_mix import conv_mix
    from vllm.models.inkling2.ops.moe import combine as moe_combine
    from vllm.models.inkling2.ops.moe import combine_push
    from vllm.models.inkling2.ops.seq_parallel import SeqParallel

    os.environ.update(MASTER_ADDR="localhost", MASTER_PORT=str(port))
    torch.accelerator.set_device_index(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world)
    tail, align = 8, 16
    sp = SeqParallel(dist.new_group(backend="gloo"), DIM, 2048, tail)
    failures = []
    # (query lengths, padding, MoE side, stack blocks, queries, last)
    cases = [
        ([600], 0, False, 1, 1, True),
        ([8] * 16 + [900], 13, True, 8, 3, False),
        ([1, 8, 8, 500, 2, 700], 0, True, 4, 1, False),
        ([5] * 60 + [3], 7, False, 6, 1, False),
    ]
    for i, (qlens, pad, moe, num_stack, num_q, with_last) in enumerate(cases):
        reqs, tokens = len(qlens), sum(qlens) + pad
        torch.manual_seed(i)  # shared by all ranks
        blocks = max(qlens) // BLOCK + 4
        cache = torch.randn(reqs * blocks + 1, HEADS, BLOCK, OFFSET + WIDTH)
        cache = cache.bfloat16().cuda()
        weight = torch.randn(DIM, 4, device="cuda").bfloat16()
        stack = torch.randn(num_stack, tokens, DIM, device="cuda").bfloat16()
        queries = torch.randn(num_q, DIM, device="cuda") * DIM**-0.5
        norm_w = torch.randn(num_q, DIM, device="cuda").bfloat16()
        last = torch.randn(tokens, DIM, device="cuda").bfloat16() if with_last else None
        addend = torch.randn(tokens, DIM, device="cuda").bfloat16() if moe else None
        residual = torch.randn(tokens, DIM, device="cuda").bfloat16() if moe else None
        start = torch.randint(0, 40, (reqs,))
        lens = torch.tensor(qlens)
        pos = torch.cat([s + torch.arange(q) for s, q in zip(start, qlens)])
        seq = torch.arange(reqs).repeat_interleave(lens)
        first = (lens.cumsum(0) - lens).repeat_interleave(lens)
        table = torch.randperm(reqs * blocks).view(reqs, blocks).int() + 1
        slots = table[seq, pos // BLOCK].long() * BLOCK + pos % BLOCK
        pad_zeros = torch.zeros(pad, dtype=torch.long)
        pos, seq, first = (torch.cat([t, pad_zeros]) for t in (pos, seq, first))
        slots = torch.cat([slots, pad_zeros - 1])
        meta = (
            pos.cuda(),
            table.cuda(),
            seq.int().cuda(),
            slots.cuda(),
            first.int().cuda(),
        )

        torch.manual_seed(100 + i + rank)  # per-rank partial sums
        if moe:
            # The combine of two expert rows per token (-1: none).
            down = torch.randn(2 * tokens, DIM, device="cuda").bfloat16()
            src2dst = torch.randperm(2 * tokens, device="cuda").int()
            src2dst[::7] = -1
            partial = moe_combine(down, src2dst, 2, torch.bfloat16)
        else:
            partial = torch.randn(tokens, DIM, device="cuda").bfloat16()
        partials = _gather(partial, world)
        reduced = partials[0].float()
        for p in partials[1:]:
            reduced += p.float()
        reduced = reduced.bfloat16()
        ref_cache = cache.clone()
        ref_conv = (
            residual.clone() if residual is not None else torch.empty_like(reduced)
        )
        ref_mix = conv_mix(
            reduced,
            weight,
            ref_cache,
            *meta,
            OFFSET,
            WIDTH,
            ref_conv,
            ref_conv if moe else None,
            addend,
            stack,
            last,
            queries,
            norm_w,
            1e-6,
        )

        begin, end, _ = sp.shard(tokens)
        out_cache = cache.clone()
        if moe:
            combine_push(down, src2dst, 2, sp)
            x, halo = sp.pushed(tokens)
        else:
            sp.partial(tokens).copy_(partial)
            x, halo = sp.reduce_scatter(tokens)
        rows = slice(begin, end)
        conv = (
            residual[rows].clone()
            if residual is not None
            else torch.empty_like(x[halo:])
        )
        mix = conv_mix(
            x,
            weight,
            out_cache,
            *meta,
            OFFSET,
            WIDTH,
            conv,
            conv if moe else None,
            None if addend is None else addend[rows],
            stack[:, rows],
            None if last is None else last[rows],
            queries,
            norm_w,
            1e-6,
            sp=sp,
            halo=halo,
            prefix_align=align,
        )
        # Every rank writes the cache rows of its shard and those read later.
        t = torch.arange(tokens)
        later = t + tail
        ends = (
            (later >= tokens)
            | (slots[later.clamp(max=tokens - 1)] < 0)
            | (first[later.clamp(max=tokens - 1)] != first)
        )
        written = (slots >= 0) & (
            ((t >= begin) & (t < end)) | ends | (pos % align >= align - 3)
        )
        expected = cache.clone()
        for slot in slots[written].tolist():
            b, o = divmod(slot, BLOCK)
            expected[b, :, o] = ref_cache[b, :, o]
        checks = {
            "mix": torch.equal(mix, ref_mix),
            "conv": torch.equal(conv, ref_conv[rows]),
            "cache": torch.equal(out_cache, expected),
        }
        failures += [f"case {i}: {k}" for k, ok in checks.items() if not ok]
    queue.put((rank, failures))
    dist.destroy_process_group()


@pytest.mark.parametrize("world", [2, 4])
def test_seq_parallel_matches_unsharded(world):
    """The reduce-scatter (attention) or MoE combine push, sharded conv + mix
    and all-gather of large batches reproduce the unsharded kernel on the
    reduced sum bitwise, and each rank writes the conv-state cache rows that
    later steps read."""
    if torch.accelerator.device_count() < world:
        pytest.skip(f"needs {world} GPUs")
    queue = mp.get_context("spawn").Queue()
    port = 29620 + world
    mp.spawn(_seq_parallel_worker, args=(world, port, queue), nprocs=world)
    results = dict(queue.get() for _ in range(world))
    assert all(not f for f in results.values()), results
