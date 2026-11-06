# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Brute-force exactness of the Kimi K3.1 specialized decode's DCP top-k path
over a real DCP group, on the MXFP4 indexer cache: every rank scores its
interleave-1 shard of the keys with the fused indexer (run_dcp: DeepGEMM-layout
logits and the local top-N B' rows), the one-round merge (#63,
dcp_topk_merge_rows) turns them into this rank's KV rows and the global ids, and
the ids must be the exact global top-K of the brute-force logits of all keys
(DeepGEMM's FP4 paged MQA logits over the unsharded cache), the ranks' rows a
partition of them.

Shapes: 64 requests x q = 5 (320 rows, the widest DSpark verify step), row j of
a request seeing context + 1 + j keys (spec verify), and requests with 0, 1, 2,
3 and up to (K - 1) // N saturated ranks (their keys' scales raised so that they
hold more than N of the row's global top-K), including local lengths in (N, K).

Launch: one process per GPU on one node (2 or 4 ranks), or 8 ranks across two
GB300 nodes with ``srun -n8`` and ``KIMI_K3_SD_DCP_INIT`` (a shared rendezvous
file), one pytest per rank.
"""

import os
import time

import pytest
import torch
import torch.multiprocessing as mp

from vllm.platforms import current_platform
from vllm.utils.network_utils import get_open_port

pytestmark = pytest.mark.skipif(
    not current_platform.is_device_capability_family(100),
    reason="the fused FP4 indexer and the DCP top-k merge need SM100",
)

TOPK, UNIT, ROW, Q, REQUESTS = 2048, 64, 68, 5, 64
DCP_INIT = os.environ.get("KIMI_K3_SD_DCP_INIT")


def _contexts(world, n):
    """Global contexts of the 64 requests: short (no top-k), around K, local
    lengths in (N, K), and long ones up to ~130K."""
    out = [100, 2_040, 2_050, world * (n + 7), world * (TOPK - 9)]
    while len(out) < REQUESTS:
        i = len(out)
        out.append(3_000 + (i * 7_919) % 127_000)
    return out


def _saturated(r, world, max_sat):
    """Request r's saturated ranks: none, one, two, three, ..., max_sat."""
    count = r % (max_sat + 1)
    return [(r * 3 + 5 * i) % world for i in range(count)]


def _global_case(world, n, max_sat):
    """The unsharded MXFP4 cache (identical on every rank), its block table,
    per-row ends, the MXFP4 query rows and head weights."""
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.dsa.index import (
        fp4_query_views,
    )

    gen = torch.Generator(device="cuda").manual_seed(7)
    contexts = _contexts(world, n)
    need = [-(-(c + Q) // UNIT) for c in contexts]
    blocks = sum(need)
    cache = torch.empty(blocks, UNIT, ROW, dtype=torch.uint8, device="cuda")
    pages = cache.view(blocks, UNIT * ROW)
    pages[:, : UNIT * 64] = torch.randint(
        0, 256, (blocks, UNIT * 64), device="cuda", generator=gen
    ).to(torch.uint8)
    scales = torch.randint(118, 132, (blocks, UNIT, 4), device="cuda", generator=gen)
    table = torch.zeros(REQUESTS, max(need), dtype=torch.int32, device="cuda")
    used = 0
    for r, k in enumerate(need):
        table[r, :k] = torch.arange(used, used + k, dtype=torch.int32)
        # Saturate: this request's keys on the chosen ranks get 2^4 larger
        # scales (|logits| x 16), so their best keys fill the global top-K.
        sat = _saturated(r, world, max_sat)
        if sat:
            keys = torch.arange(k * UNIT, device="cuda")
            mine = torch.isin(keys % world, torch.tensor(sat, device="cuda"))
            view = scales[used : used + k].view(-1, 4)
            view[mine] += 4
        used += k
    pages[:, UNIT * 64 :] = scales.to(torch.uint8).view(blocks, -1)
    ends = torch.tensor(
        [c + 1 + j for c in contexts for j in range(Q)],
        dtype=torch.int32,
        device="cuda",
    )
    rows = REQUESTS * Q
    q_rows = torch.zeros(rows, 32, 128, dtype=torch.uint8, device="cuda")
    packed, q_scales = fp4_query_views(q_rows, rows)
    packed.copy_(
        torch.randint(0, 256, packed.shape, device="cuda", generator=gen).to(
            torch.uint8
        )
    )
    exps = torch.randint(118, 132, (rows, 32, 4), device="cuda", generator=gen)
    q_scales.copy_(exps.to(torch.uint8).view(torch.int32).view(rows, 32))
    weights = (torch.randn(rows, 32, device="cuda", generator=gen) * 0.1).bfloat16()
    return cache, table, ends, q_rows, weights


def _global_logits(cache, table, ends, q_rows, weights):
    """Brute force: DeepGEMM's FP4 paged MQA logits of every row over all its
    keys (BF16 [rows, max end])."""
    from vllm.model_executor.layers.sparse_attn_indexer import (
        kv_cache_as_quant_view,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.dsa.index import (
        fp4_query_views,
    )
    from vllm.utils.deep_gemm import (
        fp8_fp4_paged_mqa_logits,
        get_num_sms,
        get_paged_mqa_logits_metadata,
        mqa_logits_mx_inputs,
    )

    rows = ends.shape[0]
    packed, q_scales = fp4_query_views(q_rows, rows)
    query = packed.contiguous().view(torch.int8).view(rows, 1, 32, 64)
    q_scale = q_scales.contiguous().view(rows, 1, 32)
    lens = ends.view(rows, 1).contiguous()
    row_table = table.repeat_interleave(Q, 0).contiguous()
    indices = torch.arange(rows, dtype=torch.int32, device="cuda")
    schedule = get_paged_mqa_logits_metadata(lens, UNIT, get_num_sms(), indices)
    q_scale, w = mqa_logits_mx_inputs(query, q_scale, weights, torch.bfloat16)
    return fp8_fp4_paged_mqa_logits(
        (query, q_scale),
        kv_cache_as_quant_view(cache, 128, True),
        w,
        lens,
        row_table,
        schedule,
        max_model_len=int(ends.max()) + 1,
        logits_dtype=torch.bfloat16,
        indices=indices,
    )


def _local_shard(cache, table, ends, world, rank):
    """This rank's interleave-1 shard (global key g -> rank g % world, local
    index g // world) as its own paged MXFP4 cache (blocks in scrambled order),
    block table and row ends."""
    local_ends = ((ends - rank + world - 1) // world).clamp_min(0).int()
    lengths = local_ends.view(REQUESTS, Q)[:, -1].tolist()
    need = [-(-max(x, 1) // UNIT) for x in lengths]
    blocks = sum(need) + 1

    def split(pages):  # [blocks, 64 x 68] -> E2M1 [blocks, 64, 64], UE8M0 x4
        codes = pages[:, : UNIT * 64].view(-1, UNIT, 64)
        return codes, pages[:, UNIT * 64 :].view(-1, UNIT, 4)

    local = torch.zeros(blocks, UNIT, ROW, dtype=torch.uint8, device="cuda")
    src_codes, src_scales = split(cache.view(cache.shape[0], UNIT * ROW))
    dst_codes, dst_scales = split(local.view(blocks, UNIT * ROW))
    local_table = torch.zeros(REQUESTS, max(need), dtype=torch.int32, device="cuda")
    order = torch.randperm(blocks - 1, generator=torch.Generator().manual_seed(rank))
    used = 0
    for r, (length, k) in enumerate(zip(lengths, need)):
        ids = order[used : used + k].to("cuda").int()
        local_table[r, :k] = ids
        used += k
        if not length:
            continue
        i = torch.arange(length, device="cuda")
        g = i * world + rank
        blk, pos = table[r][g // UNIT].long(), g % UNIT
        lb, lp = ids[i // UNIT].long(), i % UNIT
        dst_codes[lb, lp] = src_codes[blk, pos]
        dst_scales[lb, lp] = src_scales[blk, pos]
    return local, local_table, local_ends


def _check(group):
    from types import SimpleNamespace

    from vllm.models.kimi_k3.nvidia.ops.dcp_topk_merge import (
        dcp_merge_local_topk,
        dcp_topk_merge_fits,
    )
    from vllm.models.kimi_k3.nvidia.ops.dcp_topk_merge.kernels import (
        dcp_topk_merge_rows,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.dsa.indexer import (
        FusedIndexer,
    )

    world, rank = group.world_size, group.rank_in_group
    n = dcp_merge_local_topk(TOPK, world)
    max_sat = (TOPK - 1) // n
    rows = REQUESTS * Q
    assert dcp_topk_merge_fits(rows, TOPK, n), "320 rows must fit the merge"
    cache, table, ends, q_rows, weights = _global_case(world, n, max_sat)
    reference = _global_logits(cache, table, ends, q_rows, weights)
    local, local_table, local_ends = _local_shard(cache, table, ends, world, rank)
    max_model_len = int(ends.max()) + 64
    indexer = SimpleNamespace(
        k_cache=SimpleNamespace(kv_cache=local),
        topk_tokens=TOPK,
        max_model_len=max_model_len,
        dcp_world_size=world,
    )
    decode = SimpleNamespace(
        seq_lens=local_ends.view(rows, 1),
        block_table=local_table.repeat_interleave(Q, 0),
    )
    device = torch.device("cuda", torch.accelerator.current_device_index())
    results = {}
    for packed_select in (True, False):
        fused = FusedIndexer(
            indexer, [(rows, Q)], device, rows, local_topk=n, packed=packed_select
        )
        assert fused.fp4
        packed, logits, row_ends = fused.run_dcp(q_rows, weights, decode, rows, Q)
        assert packed_select or packed is None
        # This rank's KV rows of the merged top-K: an MLA block table over
        # pages of 64 local tokens (reversed ids, so rows != local indices).
        mla_blocks = -(-int(local_ends.max()) // UNIT) + 1
        mla_table = torch.arange(
            REQUESTS * mla_blocks, device="cuda", dtype=torch.int32
        )
        mla_table = mla_table.flip(0).view(REQUESTS, mla_blocks).contiguous()
        rows_out = torch.full((rows, TOPK), -7, dtype=torch.int32, device="cuda")
        valid_out = torch.full((rows,), -7, dtype=torch.int32, device="cuda")
        ids_out = torch.full((rows, TOPK), -7, dtype=torch.int32, device="cuda")
        dcp_topk_merge_rows(
            logits=logits,
            row_ends=row_ends,
            packed=packed,
            max_local_len=-(-max_model_len // world),
            local_topk=n,
            topk=TOPK,
            req_id=torch.arange(REQUESTS, device="cuda", dtype=torch.int32)
            .repeat_interleave(Q)
            .contiguous(),
            block_table=mla_table,
            block_size=UNIT,
            stride_rows=UNIT,
            dcp_rank=rank,
            rows_out=rows_out,
            valid_out=valid_out,
            ids_out=ids_out,
        )
        torch.accelerator.synchronize()
        # The local logits are the brute force's at this rank's keys.
        bits = logits.view(torch.int16)
        ref_bits = reference.view(torch.int16)
        for t in range(0, rows, 37):
            length = int(local_ends[t])
            g = torch.arange(length, device="cuda") * world + rank
            got, want = bits[t, :length].int(), ref_bits[t, g].int()
            got = torch.where(got == -32768, 0, got)
            want = torch.where(want == -32768, 0, want)
            assert torch.equal(got, want), (rank, t)
        saturated_rows = 0
        for t in range(rows):
            end = int(ends[t])
            got = ids_out[t]
            count = int((got >= 0).sum())
            assert bool((got[count:] == -1).all()), (rank, t)
            sel = got[:count].long()
            assert sel.unique().numel() == count, (rank, t, "repeated ids")
            assert bool((sel < end).all()), (rank, t)
            v = reference[t, :end].float()
            want = v.topk(min(TOPK, end)).values
            have = reference[t].float()[sel].sort(descending=True).values
            assert torch.equal(have, want), (rank, t, "not the exact global top-K")
            share = torch.bincount(sel % world, minlength=world)
            saturated_rows += int((share > n).any())
            # This rank's rows: exactly the selected ids it owns, as KV rows.
            mine = sel[sel % world == rank] // world
            assert int(valid_out[t]) == mine.numel(), (rank, t)
            r = t // Q
            want_rows = mla_table[r][mine // UNIT].long() * UNIT + mine % UNIT
            got_rows = rows_out[t, : mine.numel()].long()
            assert torch.equal(got_rows.sort().values, want_rows.sort().values)
        total = group.all_reduce(valid_out.clone())
        assert torch.equal(total, (ids_out >= 0).sum(1).int())
        every = group.all_gather(ids_out.sort(dim=1).values.unsqueeze(0), dim=0)
        for other in range(world):
            assert torch.equal(every[other], every[0]), f"rank {other} differs"
        results[packed_select] = ids_out.sort(dim=1).values
        # Several saturated ranks per row occur (the rescue round ran).
        assert saturated_rows >= REQUESTS, saturated_rows
    assert torch.equal(results[True], results[False])


def run_worker(rank, world, local_rank, init_method):
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed import (
        cleanup_dist_env_and_memory,
        get_dcp_group,
        init_distributed_environment,
        initialize_model_parallel,
    )
    from vllm.models.kimi_k3.nvidia.ops.dcp_topk_merge import init_dcp_topk_merge

    torch.accelerator.set_device_index(local_rank)
    config = VllmConfig()
    config.parallel_config.tensor_parallel_size = world
    config.parallel_config.decode_context_parallel_size = world
    # The rescue workspace holds a step of 64 requests x q = 5.
    config.scheduler_config.max_num_seqs = REQUESTS
    config.scheduler_config.max_num_batched_tokens = 8192
    with set_current_vllm_config(config):
        init_distributed_environment(
            world_size=world,
            rank=rank,
            local_rank=local_rank,
            distributed_init_method=init_method,
        )
        initialize_model_parallel(
            tensor_model_parallel_size=world, decode_context_model_parallel_size=world
        )
        try:
            from types import SimpleNamespace

            gate = SimpleNamespace(
                parallel_config=config.parallel_config,
                kernel_config=SimpleNamespace(
                    sparse_indexer_topk_backend="auto", enable_jit_warmup=False
                ),
                model_config=SimpleNamespace(max_model_len=world << 17),
                scheduler_config=config.scheduler_config,
                compilation_config=config.compilation_config,
                speculative_config=SimpleNamespace(
                    num_speculative_tokens=Q - 1, parallel_drafting=False
                ),
                num_speculative_tokens=Q - 1,
                use_v2_model_runner=True,
            )
            assert init_dcp_topk_merge(gate, TOPK), "merge disabled"
            _check(get_dcp_group())
        finally:
            cleanup_dist_env_and_memory()


def _spawned(rank, world, init_method):
    run_worker(rank, world, rank, init_method)


@pytest.mark.skipif(DCP_INIT is None, reason="needs a multi-node launch")
def test_dcp_fp4_indexer_merge_exact_launched():
    """One rank of an ``srun -n<world>`` launch (DCP8: 2 nodes x 4 GPUs)."""
    rank, world = int(os.environ["SLURM_PROCID"]), int(os.environ["SLURM_NTASKS"])
    local = int(os.environ["SLURM_LOCALID"])
    run_worker(rank, world, local, f"file://{DCP_INIT}")


@pytest.mark.skipif(DCP_INIT is not None, reason="multi-node launch")
@pytest.mark.parametrize(
    "world",
    [
        pytest.param(
            w,
            marks=pytest.mark.skipif(
                torch.accelerator.device_count() < w, reason=f"Need {w} GPUs."
            ),
        )
        for w in (2, 4)
    ],
)
def test_dcp_fp4_indexer_merge_exact(world):
    init_method = f"tcp://127.0.0.1:{get_open_port()}"
    ctx = mp.get_context("spawn")
    procs = [
        ctx.Process(target=_spawned, args=(r, world, init_method)) for r in range(world)
    ]
    for p in procs:
        p.start()
    deadline = time.monotonic() + 600
    for p in procs:
        p.join(timeout=max(0.0, deadline - time.monotonic()))
    for p in procs:
        if p.is_alive():
            p.kill()
            p.join()
    assert [p.exitcode for p in procs] == [0] * world
