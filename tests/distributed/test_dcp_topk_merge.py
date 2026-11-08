# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exact one-round DCP decode top-k merge over a real DCP group.

One process per GPU (DCP = TP = world size), the production construction
gate (`init_dcp_topk_merge`: rank agreement, symmetric rescue workspace)
and the real all-gather. Every rank builds the same global bf16 logits and
keeps its interleave-1 shard (columns `rank::W`); the reference is the
top-K of the union of all ranks' logits. Each world size spawns one group
and runs every case inside it.
"""

import time

import pytest
import torch
import torch.multiprocessing as mp

from vllm.platforms import current_platform
from vllm.utils.import_utils import has_cutedsl
from vllm.utils.network_utils import get_open_port

TOPK = 2048
# DeepSelect reads logits rows at a 1024-byte aligned stride.
_ROW_ALIGN = 512
_BLOCK = 64


def _has_deep_select() -> bool:
    try:
        import vllm._deepselect_C  # noqa: F401
    except ImportError:
        return False
    return True


pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda()
    or not current_platform.is_device_capability_family(100)
    or not has_cutedsl()
    or not _has_deep_select(),
    reason="The DCP top-k merge needs SM100, CuTe DSL and DeepSelect",
)


# --- Reference helpers -------------------------------------------------------


def _make_case(
    lengths: list[int], seed: int, levels: int = 0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Global bf16 logits `[rows, max_len]` (identical on every rank) and
    per-row global ends. `levels` > 0 quantizes the scores (tie-heavy)."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    logits = torch.randn((len(lengths), max(lengths)), generator=gen, device="cuda")
    if levels:
        logits = torch.floor(logits * levels / 8) / levels
    ends = torch.tensor(lengths, dtype=torch.int32, device="cuda")
    return logits.to(torch.bfloat16), ends


def _boost(logits: torch.Tensor, ranks, rows, world: int) -> torch.Tensor:
    """Raise the given ranks' tokens on the given rows above all others."""
    out = logits.float()
    for rank in ranks:
        out[rows, rank::world] += 8.0
    return out.to(torch.bfloat16)


def _shard(
    logits: torch.Tensor, ends: torch.Tensor, world: int, rank: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """This rank's local bf16 logits (DeepSelect-aligned rows) and row ends."""
    num_rows, width = logits.shape
    local_width = -(-width // world)
    padded = -(-local_width // _ROW_ALIGN) * _ROW_ALIGN
    local = torch.full(
        (num_rows, padded), float("-inf"), dtype=torch.bfloat16, device="cuda"
    )
    cols = logits[:, rank::world]
    local[:, : cols.shape[1]] = cols
    local_ends = ((ends - rank + world - 1) // world).clamp_min(0).int()
    return local, local_ends


def _as_lists(out: torch.Tensor) -> list[list[int]]:
    """Sorted selections per row, checking the selected-then--1 layout."""
    lists = []
    for row in out.tolist():
        valid = [i for i in row if i >= 0]
        assert row == valid + [-1] * (len(row) - len(valid)), "bad -1 tail"
        lists.append(sorted(valid))
    return lists


def _assert_exact(ids: torch.Tensor, logits: torch.Tensor, ends: torch.Tensor):
    """Repeat-free ids whose score multiset is the exact global top-K (ties
    at the cutoff resolved either way)."""
    for row, got in enumerate(_as_lists(ids)):
        assert len(set(got)) == len(got), f"row {row}: repeated ids"
        v = logits[row, : int(ends[row])].float()
        want = v.topk(min(TOPK, v.numel())).values
        sel = torch.tensor(got, dtype=torch.long, device="cuda")
        scores = logits[row].float()[sel].sort(descending=True).values
        assert torch.equal(scores, want), f"row {row}: not the exact top-K"


def _assert_same_on_every_rank(group, ids: torch.Tensor, what: str) -> None:
    every = group.all_gather(ids.sort(dim=1).values.unsqueeze(0), dim=0)
    for r in range(group.world_size):
        assert torch.equal(every[r], every[0]), f"{what}: rank {r} differs"


def _block_table(num_rows: int, max_local: int) -> torch.Tensor:
    blocks = -(-max(max_local, 1) // _BLOCK) + 1
    table = torch.arange(num_rows * blocks, dtype=torch.int32, device="cuda")
    return table.flip(0).view(num_rows, blocks).contiguous()


def _assert_rows(group, ids, rows_out, valid_out, req_id, block_table) -> None:
    """Rows and counts are the stock filter/convert of the merged ids, and the
    ranks' owned rows partition the merged set."""
    from vllm.v1.attention.backends.mla.sparse_utils import (
        triton_filter_and_convert_dcp_index,
    )

    ref_rows, ref_valid = triton_filter_and_convert_dcp_index(
        req_id,
        block_table,
        ids,
        dcp_size=group.world_size,
        dcp_rank=group.rank_in_group,
        cp_kv_cache_interleave_size=1,
        BLOCK_SIZE=_BLOCK,
        BLOCK_STRIDE_ROWS=_BLOCK,
        NUM_TOPK_TOKENS=ids.shape[1],
        return_valid_counts=True,
    )
    assert torch.equal(valid_out, ref_valid)
    assert _as_lists(rows_out) == _as_lists(ref_rows)
    total = group.all_reduce(valid_out.clone())
    assert torch.equal(total, (ids >= 0).sum(1).int())


def _sat_masks(ws, num_rows: int) -> list[int]:
    return ((ws.sat_word[:num_rows] >> 16) & 0xFFFF).tolist()


# --- The worker --------------------------------------------------------------


def _check_cases(group, ws, n: int) -> None:
    """Normal, saturated (1 rank and the (K - 1) // N bound), mixed, short and
    tie-heavy rows through both entry points (ids; rows from the logits and
    from packed input)."""
    from vllm.v1.attention.ops.dcp_topk_merge.kernels import (
        dcp_topk_merge,
        dcp_topk_merge_rows,
        dcp_topk_pack,
    )

    world, rank = group.world_size, group.rank_in_group
    max_sat = (TOPK - 1) // n
    long = [100_000, 40_000, 9_000, 70_123, 150_000, 6_000]
    # Normal rows also cover rows below K, empty rows and local indices
    # straddling 2**16 (the bit plane).
    straddle = world * 66_000
    normal = [100_000, 40_000, 9_000, 70_123, 2_049, 1, 0, straddle]
    cases = [
        # (name, lengths, levels, saturated ranks, saturated rows)
        ("normal", normal, 0, [], None),
        ("sat1", long, 0, [world - 1], slice(None)),
        ("sat-max", long, 0, list(range(max_sat)), slice(None)),
        ("mixed", long, 0, [world // 2], slice(0, None, 2)),
        # Local lengths in (N, K): a saturated rank has fewer than K tokens.
        ("short", [300, 1_000, 3_000, 9_000, 20_000, 120], 0, [0], slice(None)),
        ("ties", [50_000, 30_000, 120_000], 16, [], None),
        ("ties-sat", [50_000, 30_000, 120_000], 16, [1], slice(None)),
    ]
    for seed, (name, lengths, levels, sat, sat_rows) in enumerate(cases):
        num_rows = len(lengths)
        logits, ends = _make_case(lengths, seed=seed, levels=levels)
        if name == "normal":
            boundary = world * 65_536
            logits[-1, boundary - world * 300 : boundary + world * 300] += 6
        if sat:
            logits = _boost(logits, sat, sat_rows, world)
        local, local_end = _shard(logits, ends, world, rank)
        if name == "mixed":
            # The per-row rescue sequence wraps like any int32.
            ws.seq2.fill_(2**31 - 1)

        ids = torch.full((num_rows, TOPK), -7, dtype=torch.int32, device="cuda")
        dcp_topk_merge(local, local_end, ids, n)
        masks = _sat_masks(ws, num_rows)
        _assert_exact(ids, logits, ends)
        _assert_same_on_every_rank(group, ids, name)
        sat_mask = sum(1 << r for r in sat)
        assert all(bin(m).count("1") <= max_sat for m in masks), (name, masks)
        if levels == 0 and not sat:
            assert masks == [0] * num_rows, (name, masks)
        elif levels == 0:
            local_len = [-(-x // world) for x in lengths]
            for row, m in enumerate(masks):
                boosted = sat_rows == slice(None) or row % 2 == 0
                if boosted and local_len[row] > n and lengths[row] > TOPK:
                    assert m & sat_mask, (name, row, masks)
                if not boosted:
                    assert m == 0, (name, row, masks)
        if name == "mixed":
            want = [-(2**31) if m else 2**31 - 1 for m in masks]
            assert ws.seq2[:num_rows].tolist() == want, name

        req_id = torch.arange(num_rows, dtype=torch.int32, device="cuda")
        block_table = _block_table(num_rows, int(local_end.max()))
        packed = dcp_topk_pack(local, local_end, n)
        for mode in ("logits", "packed"):
            rows_out = torch.full_like(ids, -7)
            valid_out = torch.full((num_rows,), -7, dtype=torch.int32, device="cuda")
            ids2 = torch.full_like(ids, -7)
            dcp_topk_merge_rows(
                logits=local,
                row_ends=local_end,
                packed=packed if mode == "packed" else None,
                local_topk=n,
                topk=TOPK,
                req_id=req_id,
                block_table=block_table,
                block_size=_BLOCK,
                stride_rows=_BLOCK,
                dcp_rank=rank,
                rows_out=rows_out,
                valid_out=valid_out,
                ids_out=ids2,
            )
            _assert_exact(ids2, logits, ends)
            _assert_same_on_every_rank(group, ids2, f"{name}/{mode}")
            if levels == 0:
                assert _as_lists(ids2) == _as_lists(ids), (name, mode)
            _assert_rows(group, ids2, rows_out, valid_out, req_id, block_table)


def _check_nonfinite(group, n: int) -> None:
    """NaN / +-inf logits: ids stay in range, never select a NaN, agree on
    every rank, and rows without a NaN keep the exact top-K."""
    from vllm.v1.attention.ops.dcp_topk_merge.kernels import dcp_topk_merge

    world, rank = group.world_size, group.rank_in_group
    lengths = [30_000] * 6 + [1_500, 30_000]
    logits, ends = _make_case(lengths, seed=41)
    logits[0, ::7] = float("nan")
    logits[1] = float("nan")
    logits[2, ::5] = float("inf")
    logits[3] = float("-inf")
    logits[4] = float("inf")
    logits[5, (world - 1) :: world] = float("nan")  # NaN on one rank
    logits[6, ::3] = float("nan")
    logits[7, 0::world] = float("inf")  # +inf on one rank: it saturates
    local, local_end = _shard(logits, ends, world, rank)
    ids = torch.full((len(lengths), TOPK), -7, dtype=torch.int32, device="cuda")
    dcp_topk_merge(local, local_end, ids, n)
    _assert_same_on_every_rank(group, ids, "nonfinite")
    for row, got in enumerate(_as_lists(ids)):
        end = int(ends[row])
        assert all(0 <= g < end for g in got), row
        row_logits = logits[row, :end]
        if got:
            sel = torch.tensor(got, device="cuda")
            assert not logits[row, sel].isnan().any(), row
        if not row_logits.isnan().any():
            _assert_exact(ids[row : row + 1], logits[row : row + 1], ends[row:])
    assert _as_lists(ids)[1] == []


def _check_cuda_graph(group, ws, n: int) -> None:
    """One graph of both entry points, replayed over inputs alternating
    clean, one saturated row and every row saturated (1 rank, and the bound):
    each replay is exact, and the rescue sequence advances only on saturated
    rows."""
    from vllm.v1.attention.ops.dcp_topk_merge.kernels import (
        dcp_topk_merge,
        dcp_topk_merge_rows,
    )

    world, rank = group.world_size, group.rank_in_group
    lengths = [100_000, 40_000, 9_000, 70_123, 150_000, 6_000, 3_000, 220_000]
    num_rows = len(lengths)
    base, ends = _make_case(lengths, seed=29)
    variants = [
        base,
        _boost(base, [0], slice(7, 8), world),
        _boost(base, [world - 1], slice(None), world),
        _boost(base, list(range((TOPK - 1) // n)), slice(None), world),
    ]
    shards = [_shard(v, ends, world, rank) for v in variants]
    static = shards[0][0].clone()
    local_end = shards[0][1]
    ids = torch.empty((num_rows, TOPK), dtype=torch.int32, device="cuda")
    ids2, rows_out = torch.empty_like(ids), torch.empty_like(ids)
    valid_out = torch.empty(num_rows, dtype=torch.int32, device="cuda")
    req_id = torch.arange(num_rows, dtype=torch.int32, device="cuda")
    block_table = _block_table(num_rows, int(local_end.max()))

    def step() -> None:
        dcp_topk_merge(static, local_end, ids, n)
        dcp_topk_merge_rows(
            logits=static,
            row_ends=local_end,
            local_topk=n,
            topk=TOPK,
            req_id=req_id,
            block_table=block_table,
            block_size=_BLOCK,
            stride_rows=_BLOCK,
            dcp_rank=rank,
            rows_out=rows_out,
            valid_out=valid_out,
            ids_out=ids2,
        )

    step()  # compile outside capture
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with group.graph_capture() as ctx, torch.cuda.graph(graph, stream=ctx.stream):
        step()
    torch.accelerator.synchronize()
    seq = ws.seq2[:num_rows].clone()
    for i in range(2 * len(variants)):
        v = i % len(variants)
        static.copy_(shards[v][0])
        graph.replay()
        torch.accelerator.synchronize()
        _assert_exact(ids, variants[v], ends)
        _assert_exact(ids2, variants[v], ends)
        _assert_same_on_every_rank(group, ids, f"graph replay {i}")
        _assert_rows(group, ids2, rows_out, valid_out, req_id, block_table)
        triggered = torch.tensor(
            [int(m != 0) for m in _sat_masks(ws, num_rows)],
            dtype=torch.int32,
            device="cuda",
        )
        assert bool(triggered.any()) == (v != 0), (i, triggered.tolist())
        # Two merges per replay: each saturated row advances twice.
        assert torch.equal(ws.seq2[:num_rows], seq + 2 * triggered), i
        seq = ws.seq2[:num_rows].clone()


def _check_late_peer(group, ws, n: int) -> None:
    """The saturated rank starts its select ~100 ms after the all-gather,
    while the other ranks are already in their rescue waiting on its flag:
    they must wait for its push (and take that long), and the result is the
    exact top-K."""
    from vllm.v1.attention.ops.dcp_topk_merge.kernels import (
        dcp_topk_pack,
        dcp_topk_select,
    )

    world, rank = group.world_size, group.rank_in_group
    late = world - 1
    lengths = [150_000, 100_000, 70_123]
    logits, ends = _make_case(lengths, seed=31)
    logits = _boost(logits, [late], slice(None), world)
    local, local_end = _shard(logits, ends, world, rank)
    ids = torch.full((len(lengths), TOPK), -7, dtype=torch.int32, device="cuda")
    packed = dcp_topk_pack(local, local_end, n)
    gathered = group.all_gather(packed, dim=0).view(world, *packed.shape)
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    start.record()
    if rank == late:
        torch.cuda._sleep(200_000_000)
    dcp_topk_select(gathered, n, ids, exact=ws.args(local, local_end))
    stop.record()
    torch.accelerator.synchronize()
    assert _sat_masks(ws, len(lengths)) == [1 << late] * len(lengths)
    _assert_exact(ids, logits, ends)
    _assert_same_on_every_rank(group, ids, "late peer")
    if rank != late:
        assert start.elapsed_time(stop) > 20, "the rescue did not wait"


def _check_deterministic(group, ws, n: int, repeats: int = 8) -> None:
    """Repeated merges of the same logits are bit-identical, order included,
    through both entry points (ids; rows with and without ids), unsaturated
    and saturated (the rescue round): the sparse attention downstream sums
    the selected tokens in this order."""
    from vllm.v1.attention.ops.dcp_topk_merge.kernels import (
        dcp_topk_merge,
        dcp_topk_merge_rows,
    )

    world, rank = group.world_size, group.rank_in_group
    lengths = [150_000, 100_000, 70_123, 40_000, 220_000, 9_000, 120_000, 60_000]
    num_rows = len(lengths)
    base, ends = _make_case(lengths, seed=53)
    for sat in (False, True):
        logits = _boost(base, [world - 1], slice(None), world) if sat else base
        local, local_end = _shard(logits, ends, world, rank)
        req_id = torch.arange(num_rows, dtype=torch.int32, device="cuda")
        block_table = _block_table(num_rows, int(local_end.max()))
        runs: list[list[torch.Tensor]] = []
        for _ in range(repeats):
            ids = torch.full((num_rows, TOPK), -7, dtype=torch.int32, device="cuda")
            dcp_topk_merge(local, local_end, ids, n)
            masks = _sat_masks(ws, num_rows)
            assert all(masks) if sat else not any(masks), (sat, masks)
            outs = [ids]
            for with_ids in (True, False):
                rows_out = torch.full_like(ids, -7)
                valid_out = torch.full(
                    (num_rows,), -7, dtype=torch.int32, device="cuda"
                )
                ids2 = torch.full_like(ids, -7) if with_ids else None
                dcp_topk_merge_rows(
                    logits=local,
                    row_ends=local_end,
                    local_topk=n,
                    topk=TOPK,
                    req_id=req_id,
                    block_table=block_table,
                    block_size=_BLOCK,
                    stride_rows=_BLOCK,
                    dcp_rank=rank,
                    rows_out=rows_out,
                    valid_out=valid_out,
                    ids_out=ids2,
                )
                outs += [rows_out, valid_out] + ([ids2] if with_ids else [])
            runs.append(outs)
        torch.accelerator.synchronize()
        _assert_exact(runs[0][0], logits, ends)
        _assert_rows(group, runs[0][3], runs[0][1], runs[0][2], req_id, block_table)
        what = ["ids", "rows", "valid", "rows ids", "rows (no ids)", "valid (no ids)"]
        for i, run in enumerate(runs[1:], 1):
            for name, got, want in zip(what, run, runs[0]):
                assert torch.equal(got, want), f"sat={sat} run {i}: {name} changed"
        # Without the ids the rows come out the same.
        assert torch.equal(runs[0][4], runs[0][1])


def run_worker(rank: int, world: int, local_rank: int, init_method: str) -> None:
    """One DCP rank. Also callable from a multi-node launcher."""
    from types import SimpleNamespace

    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed import (
        cleanup_dist_env_and_memory,
        get_dcp_group,
        init_distributed_environment,
        initialize_model_parallel,
    )
    from vllm.v1.attention.ops.dcp_topk_merge import (
        dcp_merge_local_topk,
        dcp_topk_merge_max_rows,
        init_dcp_topk_merge,
    )
    from vllm.v1.attention.ops.dcp_topk_merge.kernels import (
        get_dcp_topk_rescue_workspace,
    )

    torch.accelerator.set_device_index(local_rank)
    config = VllmConfig()
    config.parallel_config.tensor_parallel_size = world
    config.parallel_config.decode_context_parallel_size = world
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
            group = get_dcp_group()
            n = dcp_merge_local_topk(TOPK, world)
            gate_config = SimpleNamespace(
                parallel_config=config.parallel_config,
                kernel_config=SimpleNamespace(
                    sparse_indexer_topk_backend="auto", enable_jit_warmup=False
                ),
                # The longest context whose local ids fit the packed words.
                model_config=SimpleNamespace(max_model_len=world << 17),
                scheduler_config=config.scheduler_config,
                compilation_config=config.compilation_config,
                speculative_config=None,
                num_speculative_tokens=0,
            )
            assert init_dcp_topk_merge(gate_config, TOPK) == n, "merge disabled"
            ws = get_dcp_topk_rescue_workspace(TOPK, n)
            assert ws is not None and ws.max_rows >= 8
            assert dcp_topk_merge_max_rows(TOPK, n) == ws.max_rows
            _check_cases(group, ws, n)
            _check_deterministic(group, ws, n)
            _check_nonfinite(group, n)
            _check_cuda_graph(group, ws, n)
            _check_late_peer(group, ws, n)
        finally:
            cleanup_dist_env_and_memory()


def _spawned(rank: int, world: int, init_method: str) -> None:
    run_worker(rank, world, rank, init_method)


@pytest.mark.parametrize(
    "world",
    [
        pytest.param(
            w,
            marks=pytest.mark.skipif(
                torch.accelerator.device_count() < w, reason=f"Need {w} GPUs."
            ),
        )
        for w in (2, 4, 8)
    ],
)
def test_dcp_topk_merge_exact(world: int):
    init_method = f"tcp://127.0.0.1:{get_open_port()}"
    ctx = mp.get_context("spawn")
    procs = [
        ctx.Process(target=_spawned, args=(r, world, init_method)) for r in range(world)
    ]
    for p in procs:
        p.start()
    # One deadline for the group: a hung rank fails the test, never CI.
    deadline = time.monotonic() + 600
    for p in procs:
        p.join(timeout=max(0.0, deadline - time.monotonic()))
    for p in procs:
        if p.is_alive():
            p.kill()
            p.join()
    assert [p.exitcode for p in procs] == [0] * world
