# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The Kimi K3.1 specialized decode's DCP top-k exchange over a real DCP group
(4 GPUs of one node, NVLink; the local select needs N = 1.5 K / W < 1024, so
W >= 4): the fused indexer's local select pushes its B' rows into
every rank's symmetric exchange slots and ExchangeMerge merges them
(FusedIndexer.run_exchange, topk_exchange.py), against the #63 path it
replaces (run_dcp's B' rows, NCCL all-gather, select + rescue:
dcp_topk_merge_rows) on the same MXFP4 shards: this rank's KV rows, counts and
the global ids bit-equal, in the same order; eagerly over several calls (both
epoch parities), replayed from a CUDA graph, and over a padded batch (rows
past the step's decodes come out empty).

Shapes: 1, 4, 12, 28 and 62 requests x q = 5 (5-310 rows: one select row per
SM up to 140, rows sharing SMs at 310, the HIGH hybrid decode's batch) at
~136.5K and ~32K contexts plus short ones, with rows whose keys on one rank
lead the row's global top-K (a saturated rank: the rescue round). At 310 rows
the stock path selects candidates with DeepSelect, which may resolve ties at a
rank's N-th logit to other keys: there both paths must select the exact global
top-K logits, and the exchange must equal #63's merge of the same candidates.
"""

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

TOPK, UNIT, ROW, Q = 2048, 64, 68, 5
WORLD = 4
REQUESTS = (1, 4, 12, 28, 62)


def _contexts(requests, world, n):
    base = [136_505, 32_773, 100, world * (n + 7), 2_050, 120_000]
    return [base[r % len(base)] + 997 * (r // len(base)) for r in range(requests)]


def _case(requests, world, n, seed):
    """The unsharded MXFP4 cache (identical on every rank), block table, per-row
    ends, MXFP4 query rows and head weights; odd requests' keys on rank
    r % world get 2^4 larger scales (that rank saturates)."""
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.dsa.index import (
        fp4_query_views,
    )

    gen = torch.Generator(device="cuda").manual_seed(seed)
    contexts = _contexts(requests, world, n)
    need = [-(-(c + Q) // UNIT) for c in contexts]
    blocks = sum(need)
    cache = torch.empty(blocks, UNIT, ROW, dtype=torch.uint8, device="cuda")
    pages = cache.view(blocks, UNIT * ROW)
    pages[:, : UNIT * 64] = torch.randint(
        0, 256, (blocks, UNIT * 64), device="cuda", generator=gen
    ).to(torch.uint8)
    scales = torch.randint(118, 132, (blocks, UNIT, 4), device="cuda", generator=gen)
    table = torch.zeros(requests, max(need), dtype=torch.int32, device="cuda")
    used = 0
    for r, k in enumerate(need):
        table[r, :k] = torch.arange(used, used + k, dtype=torch.int32)
        if r % 2:
            keys = torch.arange(k * UNIT, device="cuda")
            mine = keys % world == r % world
            scales[used : used + k].view(-1, 4)[mine] += 4
        used += k
    pages[:, UNIT * 64 :] = scales.to(torch.uint8).view(blocks, -1)
    ends = torch.tensor(
        [c + 1 + j for c in contexts for j in range(Q)],
        dtype=torch.int32,
        device="cuda",
    )
    rows = requests * Q
    q_rows = torch.zeros(max(rows, 64), 32, 128, dtype=torch.uint8, device="cuda")
    packed, q_scales = fp4_query_views(q_rows, rows)
    packed.copy_(
        torch.randint(0, 256, packed.shape, device="cuda", generator=gen).to(
            torch.uint8
        )
    )
    exps = torch.randint(118, 132, (rows, 32, 4), device="cuda", generator=gen)
    q_scales.copy_(exps.to(torch.uint8).view(torch.int32).view(rows, 32))
    weights = torch.randn(max(rows, 64), 32, device="cuda", generator=gen) * 0.1
    return cache, table, ends, q_rows, weights.bfloat16()


def _shard(cache, table, ends, requests, world, rank):
    """This rank's interleave-1 shard as its own paged cache, table, ends."""
    local_ends = ((ends - rank + world - 1) // world).clamp_min(0).int()
    lengths = local_ends.view(requests, Q)[:, -1].tolist()
    need = [-(-max(x, 1) // UNIT) for x in lengths]
    blocks = sum(need) + 1
    local = torch.zeros(blocks, UNIT, ROW, dtype=torch.uint8, device="cuda")
    src = cache.view(cache.shape[0], UNIT * ROW)
    dst = local.view(blocks, UNIT * ROW)
    s_codes, s_scales = src[:, : UNIT * 64].view(-1, UNIT, 64), src[:, UNIT * 64 :]
    d_codes, d_scales = dst[:, : UNIT * 64].view(-1, UNIT, 64), dst[:, UNIT * 64 :]
    s_scales, d_scales = s_scales.view(-1, UNIT, 4), d_scales.view(-1, UNIT, 4)
    local_table = torch.zeros(requests, max(need), dtype=torch.int32, device="cuda")
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
        d_codes[lb, lp] = s_codes[blk, pos]
        d_scales[lb, lp] = s_scales[blk, pos]
    return local, local_table, local_ends


def _check(group):
    from vllm.models.kimi_k3.nvidia.ops.dcp_topk_merge import dcp_merge_local_topk
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.dsa.topk_exchange import (  # noqa: E501
        reserve,
    )

    world, rank = group.world_size, group.rank_in_group
    n = dcp_merge_local_topk(TOPK, world)
    device = torch.device("cuda", torch.accelerator.current_device_index())
    exchange = reserve(group.device_group, device, world, rank, 64 * Q, n)
    saturated = 0
    for requests in REQUESTS:
        saturated += _check_requests(group, exchange, n, requests)
        torch.distributed.barrier(group=group.cpu_group)
    assert saturated > 0, "no saturated row: the rescue round never ran"


def _merge_local_select(fused, rows, logits, row_ends, args):
    """#63's NCCL merge over the fused local select's B' rows (its push-variant
    kernel; the pushes it also makes carry the current epoch and are rewritten
    by the next exchange call)."""
    from vllm.models.kimi_k3.nvidia.ops.dcp_topk_merge.kernels import (
        dcp_topk_merge_rows,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common.host import (
        stream,
        to_cute,
    )

    local, _ = fused.exchange_kernels[rows, Q]
    out, valid = fused.out[:rows], fused.valid[:rows]
    local(
        to_cute(fused.logits),
        to_cute(row_ends, 4),
        to_cute(out, 4),
        to_cute(valid, 4),
        *fused.exchange_args(),
        stream(fused.device),
    )
    dcp_topk_merge_rows(logits=logits, row_ends=row_ends, packed=out, **args)


def _same_logits(group, logits, row_ends, stock, ours, rows, max_local_len):
    """Per row: this rank's same count of KV rows, and the selected global ids'
    logits (gathered from every rank's local logits) the multiset of the row's
    exact top-K logits, for both paths."""
    world, width = group.world_size, min(max_local_len, logits.shape[1])
    local = logits[:rows, :width].contiguous()
    every = group.all_gather(local.unsqueeze(0), dim=0)  # [W, rows, width]
    ends = group.all_gather(row_ends[:rows].contiguous().unsqueeze(0), dim=0)
    assert torch.equal(stock[1], ours[1]), "counts"
    bad: dict[str, list] = {"stock": [], "exchange": []}
    for t in range(rows):
        row = torch.cat([every[r, t, : int(ends[r, t])] for r in range(world)])
        c = min(TOPK, row.numel())
        exact = row.float().topk(c).values.sort().values
        for name, out in (("stock", stock), ("exchange", ours)):
            ids = out[2][t].long()
            ids = ids[ids >= 0]
            if ids.numel() != c:
                bad[name].append(t)
                continue
            got = every[ids % world, t, ids // world].float().sort().values
            if not torch.equal(got, exact):
                bad[name].append(t)
    assert not bad["exchange"], ("exchange not the exact top-K", bad)
    assert not bad["stock"], ("stock not the exact top-K", bad)


def _check_requests(group, exchange, n, requests):
    """One shape: the exchange path against the stock one; returns the rows
    with a saturated rank."""
    from types import SimpleNamespace

    from vllm.models.kimi_k3.nvidia.ops.dcp_topk_merge.kernels import (
        dcp_topk_merge_rows,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.dsa.indexer import (
        FusedIndexer,
    )

    world, rank = group.world_size, group.rank_in_group
    device = torch.device("cuda", torch.accelerator.current_device_index())
    rows = requests * Q
    cache, table, ends, q_rows, weights = _case(requests, world, n, requests)
    local, local_table, local_ends = _shard(cache, table, ends, requests, world, rank)
    max_model_len = int(ends.max()) + 64
    max_local_len = -(-max_model_len // world)
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
    fused = FusedIndexer(
        indexer,
        [(rows, Q)],
        device,
        rows,
        local_topk=n,
        packed=True,
        page=UNIT,
        exchange=exchange,
    )
    assert fused.exchanges(rows), rows
    mla_blocks = -(-int(local_ends.max()) // UNIT) + 1
    mla_table = torch.arange(requests * mla_blocks, device="cuda").int()
    mla_table = mla_table.flip(0).view(requests, mla_blocks).contiguous()
    req_id = torch.arange(requests, device="cuda", dtype=torch.int32)
    req_id = req_id.repeat_interleave(Q).contiguous()

    def outputs(fill):
        return (
            torch.full((rows, TOPK), fill, dtype=torch.int32, device="cuda"),
            torch.full((rows,), fill, dtype=torch.int32, device="cuda"),
            torch.full((rows, TOPK), fill, dtype=torch.int32, device="cuda"),
        )

    def merge_args(out):
        return dict(
            max_local_len=max_local_len,
            local_topk=n,
            topk=TOPK,
            req_id=req_id,
            block_table=mla_table,
            block_size=UNIT,
            stride_rows=UNIT,
            dcp_rank=rank,
            rows_out=out[0],
            valid_out=out[1],
            ids_out=out[2],
        )

    # Stock: B' rows, NCCL all-gather, select + rescue (#63).
    ref = outputs(-9)
    packed, logits, row_ends = fused.run_dcp(q_rows, weights, decode, rows, Q)
    dcp_topk_merge_rows(
        logits=logits, row_ends=row_ends, packed=packed, **merge_args(ref)
    )
    torch.accelerator.synchronize()
    if packed is None:
        # Past the SM count the stock path selects the B' rows with DeepSelect
        # + #63's pack, the exchange with the fused local select: the same
        # exact top-N, but ties at a rank's N-th logit may resolve to other
        # keys. The global selection must hold the same logits; the exact
        # comparison below is against the NCCL merge over the local select's
        # own B' rows.
        stock = ref
        ref = outputs(-9)
        _merge_local_select(fused, rows, logits, row_ends, merge_args(ref))
        torch.accelerator.synchronize()
        _same_logits(group, logits, row_ends, stock, ref, rows, max_local_len)
    assert fused.workspace is not None
    sat = (fused.workspace.sat_word[:rows] >> 16).ne(0)

    def exchange_call(out):
        merge = req_id, mla_table, UNIT, out[0], out[1], max_local_len
        fused.run_exchange(q_rows, weights, decode, rows, Q, merge, out[2])

    for call in range(3):  # both epoch parities
        got = outputs(-7)
        exchange_call(got)
        torch.accelerator.synchronize()
        for a, b, what in zip(got, ref, ("rows", "counts", "ids")):
            assert torch.equal(a, b), (rank, requests, call, what)
        assert torch.equal((fused.workspace.sat_word[:rows] >> 16).ne(0), sat), (
            rank,
            requests,
            call,
        )
    # Replayed from a CUDA graph (the epochs advance on the device).
    got = outputs(-5)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        exchange_call(got)  # warm
    torch.cuda.current_stream().wait_stream(side)
    torch.accelerator.synchronize()
    torch.distributed.barrier(group=group.cpu_group)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        exchange_call(got)
    for _ in range(2):
        for t in got:
            t.fill_(-3)
        graph.replay()
        torch.accelerator.synchronize()
        for a, b, what in zip(got, ref, ("rows", "counts", "ids")):
            assert torch.equal(a, b), (rank, requests, "graph", what)
    # A padded batch (the step's decodes fill only the first rows; merge_rows
    # runs the exchange over every row of the batch): the live rows as above,
    # the rows past them empty, as the stock path's fill.
    if requests > 1:
        live = rows - Q
        short = SimpleNamespace(
            seq_lens=decode.seq_lens[:live], block_table=decode.block_table[:live]
        )
        got = outputs(-7)
        merge = req_id, mla_table, UNIT, got[0], got[1], max_local_len
        fused.run_exchange(q_rows, weights, short, rows, Q, merge, got[2])
        torch.accelerator.synchronize()
        for a, b, what in zip(got, ref, ("rows", "counts", "ids")):
            assert torch.equal(a[:live], b[:live]), (rank, requests, "padded", what)
        assert bool((got[0][live:] == -1).all()), (rank, requests)
        assert bool((got[1][live:] == 0).all()), (rank, requests)
        assert bool((got[2][live:] == -1).all()), (rank, requests)
    # Every rank selects the same ids.
    every = group.all_gather(ref[2].unsqueeze(0), dim=0)
    for other in range(world):
        assert torch.equal(every[other], every[0]), (requests, other)
    return int(sat.sum())


def run_worker(rank, world, init_method):
    from types import SimpleNamespace

    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed import (
        cleanup_dist_env_and_memory,
        get_dcp_group,
        init_distributed_environment,
        initialize_model_parallel,
    )
    from vllm.models.kimi_k3.nvidia.ops.dcp_topk_merge import init_dcp_topk_merge

    torch.accelerator.set_device_index(rank)
    config = VllmConfig()
    config.parallel_config.tensor_parallel_size = world
    config.parallel_config.decode_context_parallel_size = world
    config.scheduler_config.max_num_seqs = 64
    config.scheduler_config.max_num_batched_tokens = 8192
    with set_current_vllm_config(config):
        init_distributed_environment(
            world_size=world,
            rank=rank,
            local_rank=rank,
            distributed_init_method=init_method,
        )
        initialize_model_parallel(
            tensor_model_parallel_size=world, decode_context_model_parallel_size=world
        )
        try:
            gate = SimpleNamespace(
                parallel_config=config.parallel_config,
                kernel_config=SimpleNamespace(
                    sparse_indexer_topk_backend="auto", enable_jit_warmup=False
                ),
                model_config=SimpleNamespace(max_model_len=world << 17),
                scheduler_config=config.scheduler_config,
                compilation_config=config.compilation_config,
                speculative_config=SimpleNamespace(num_speculative_tokens=Q - 1),
                num_speculative_tokens=Q - 1,
            )
            assert init_dcp_topk_merge(gate, TOPK), "merge disabled"
            _check(get_dcp_group())
        finally:
            cleanup_dist_env_and_memory()


@pytest.mark.skipif(torch.accelerator.device_count() < WORLD, reason="Need 4 GPUs.")
def test_k31_dsa_topk_exchange_matches_stock():
    init_method = f"tcp://127.0.0.1:{get_open_port()}"
    ctx = mp.get_context("spawn")
    procs = [
        ctx.Process(target=run_worker, args=(r, WORLD, init_method))
        for r in range(WORLD)
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
    assert [p.exitcode for p in procs] == [0] * WORLD
