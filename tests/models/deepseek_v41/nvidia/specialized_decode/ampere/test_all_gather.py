# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The SM80 Lamport all-gathers (`AmpereAllGather`) at TP8: the indexer's
decode top-k shares (rank-major) and the logits and Engram columns (side by
side, `columns.py`, which SM90 runs too).

Each check runs one rank late, so a missing wait reads a peer's stale data:

* calls of 1 to 16 rows a rank, in an order that cycles every unit's
  generations, every other call on five CTAs: every rank's rows in place,
  bitwise, with the sentinel pattern (INT32_MIN) arriving as zero and this
  rank's own rows untouched;
* every mailbox slot back to the sentinel after the calls, and every
  unit's generation in 0..2;
* one CUDA graph of three calls replayed with new inputs;
* the indexer's share gather (`_gather_topk_shares`) for the row splits of
  steps of 4, 8, 13 and 16 requests of six rows, in place with the kernel,
  and of 24 requests, above its capacity, with the custom all-gather: the
  step's rows rebuilt in order;
* side by side, BF16 rows of the Engram heads (1,536 bytes a rank), the
  Engram wkv output (6,400) and the logits (32,320, more fragments than a
  CTA has threads), 1 to 128 rows: every rank's columns of each row in rank
  order, bitwise, the sentinel pattern arriving as zero from every rank
  including this one; the mailboxes back to the sentinel; a CUDA graph of
  three calls of the three sizes replayed with new inputs; 129 rows through
  the TP group's all-gather; and `AmpereLogitsProcessor`'s gather.

Needs eight SM80 GPUs with peer access; about a minute. On eight SM90 GPUs
`test_hopper_column_gathers` runs the side-by-side checks.

For a stress run, run this file with `python -m` as the module
`tests.models.deepseek_v41.nvidia.specialized_decode.ampere.test_all_gather`, with
`--replays 2000000`: CUDA graphs of three rank-major and three
side-by-side gathers of different sizes each (side-by-side only on SM90), some
with the ranks skewed, replayed in one seeded random order with new contents
every replay; a device-side count of the words that differ from the expected
gather must stay 0. A gather that read an earlier call's data would show up as
a mismatch.
"""

import argparse
import os
import random
import tempfile
from types import SimpleNamespace

import pytest
import torch

from vllm.platforms import current_platform

TP = 8
HIDDEN = 5120
TOPK = 512
LATE_CYCLES = 20_000
REPLAYS = 50
INT32_MIN = torch.iinfo(torch.int32).min
# Bytes of a rank's row: Engram heads (3 x 256 BF16), Engram wkv output
# (3,200 BF16) and logits (16,160 BF16).
COLUMN_ROW_BYTES = (1536, 6400, 32320)
# The stress run's graphs: rows a rank of three rank-major gathers, and rows of
# a side-by-side gather of each COLUMN_ROW_BYTES size.
STRESS_GRAPHS = (
    ((6, 16, 1), (96, 6, 128)),
    ((16, 3, 12), (1, 128, 48)),
    ((1, 6, 16), (128, 24, 6)),
)


def _late(rank: int, step: int) -> None:
    """Delay one rank per step, a different one each step."""
    if rank == step % TP:
        torch.cuda._sleep(LATE_CYCLES)


def _rows(rows: int, rank: int, seed: int) -> torch.Tensor:
    """This rank's rows: indices from -1 up, and the sentinel pattern."""
    generator = torch.Generator(device="cuda").manual_seed(97 * rank + 1009 * seed)
    x = torch.randint(
        -1, 1 << 20, (rows, TOPK), generator=generator, device="cuda"
    ).int()
    x[:, 0] = INT32_MIN
    return x


def _buffer(rows: int, rank: int, seed: int) -> torch.Tensor:
    """A gather's output: this rank's rows in place, garbage elsewhere."""
    out = torch.full((TP * rows, TOPK), 12345, dtype=torch.int32, device="cuda")
    out[rank * rows : (rank + 1) * rows] = _rows(rows, rank, seed)
    return out


def _assert_gathered(out: torch.Tensor, rank: int, seed: int) -> None:
    """Every rank's rows in `out`, the sentinel arriving as zero from peers."""
    rows = out.shape[0] // TP
    for source in range(TP):
        want = _rows(rows, source, seed)
        if source != rank:
            want[want == INT32_MIN] = 0
        assert torch.equal(out[source * rows : (source + 1) * rows], want), source


def _check_shares(rank: int) -> None:
    """`_gather_topk_shares` rebuilds a step's rows from the ranks' shares,
    in place up to the kernel's capacity, with the custom all-gather above."""
    from vllm.models.deepseek_v41.nvidia.specialized_decode.ampere import indexer

    gather = indexer._ampere_all_gather
    assert gather is not None
    calls: list[int] = []
    original = type(gather).__call__

    def counted(self, local, out):
        calls.append(local.shape[0])
        return original(self, local, out)

    type(gather).__call__ = counted  # type: ignore[method-assign]
    try:
        for num_requests in (4, 8, 13, 16, 24):
            rows = 6 * num_requests
            shard = indexer._decode_row_shard(rows, num_requests, TP, rank)
            assert shard is not None
            start, end, share_rows = shard
            # Row r's top-k is r + 1 everywhere: the full step every rank wants.
            want = (torch.arange(rows, device="cuda") + 1).int()
            want = want[:, None].expand(rows, TOPK).contiguous()
            buffer = torch.full((256, TOPK), 777, dtype=torch.int32, device="cuda")

            def score(out: torch.Tensor, start=start, end=end, want=want) -> None:
                out.copy_(want[start:end])

            indexer._gather_topk_shares(
                buffer, TOPK, rows, share_rows, end - start, score
            )
            assert torch.equal(buffer[:rows], want), num_requests
    finally:
        type(gather).__call__ = original  # type: ignore[method-assign]
    assert calls == [6, 6, 12, 12]


def _columns(rows: int, row_bytes: int, rank: int, seed: int) -> torch.Tensor:
    """This rank's BF16 columns of every row, the sentinel pattern in the
    first 32-bit word of each row."""
    generator = torch.Generator(device="cuda").manual_seed(
        131 * rank + 7919 * seed + row_bytes
    )
    x = torch.randn(rows, row_bytes // 2, generator=generator, device="cuda")
    x = x.bfloat16()
    x.view(torch.int32)[:, 0] = INT32_MIN
    return x


def _assert_columns(out: torch.Tensor, row_bytes: int, seed: int) -> None:
    """Every rank's columns side by side, the sentinel arriving as zero."""
    rows = out.shape[0]
    want = torch.cat([_columns(rows, row_bytes, s, seed) for s in range(TP)], 1)
    words = want.view(torch.int32)
    words[words == INT32_MIN] = 0
    assert torch.equal(out.view(torch.int32), words), (rows, row_bytes, seed)


def _check_columns(rank: int) -> None:
    """The side-by-side all-gathers of `columns.py`."""
    from vllm.distributed import get_tp_group
    from vllm.models.deepseek_v41.nvidia.specialized_decode.ampere import columns

    columns.init_column_gathers(list(COLUMN_ROW_BYTES))
    gathers = [columns._gathers[size] for size in COLUMN_ROW_BYTES]
    assert all(g.capacity == 128 and g.side_by_side for g in gathers)
    ctas = gathers[0].ctas
    step = 100
    for seed, rows in enumerate([1, 6, 48, 128, 96, 3, 80, 128, 24, 1] * 2):
        for gather in gathers:
            gather.ctas = 5 if seed % 2 else ctas
            local = _columns(rows, gather.row_bytes, rank, seed)
            _late(rank, step := step + 1)
            _assert_columns(columns.all_gather_columns(local), gather.row_bytes, seed)
    for gather in gathers:
        gather.ctas = ctas
    torch.accelerator.synchronize()
    torch.distributed.barrier(get_tp_group().cpu_group)
    for gather in gathers:
        assert (gather.mailbox == INT32_MIN).all()
        assert int(gather.generations.min()) >= 0
        assert int(gather.generations.max()) < 3

    # One graph of a call of each size, replayed with new inputs.
    sizes = list(zip(COLUMN_ROW_BYTES, (96, 6, 80)))
    locals_ = [_columns(rows, size, rank, 0) for size, rows in sizes]
    for local in locals_:
        columns.all_gather_columns(local)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outs = [columns.all_gather_columns(local) for local in locals_]
    for replay in range(REPLAYS):
        seeds = [300 + 3 * replay + i for i in range(len(sizes))]
        for local, (size, rows), seed in zip(locals_, sizes, seeds):
            local.copy_(_columns(rows, size, rank, seed))
        _late(rank, step := step + 1)
        graph.replay()
        for out, (size, _), seed in zip(outs, sizes, seeds):
            _assert_columns(out, size, seed)

    # Above the capacity, the TP group's all-gather (no sentinel rewrite).
    local = _columns(129, COLUMN_ROW_BYTES[0], rank, 7)
    out = columns.all_gather_columns(local)
    want = torch.cat([_columns(129, COLUMN_ROW_BYTES[0], s, 7) for s in range(TP)], 1)
    assert torch.equal(out.view(torch.int32), want.view(torch.int32))

    # The logits processor's gather of a step's vocabulary shards.
    processor = columns.AmpereLogitsProcessor(TP * COLUMN_ROW_BYTES[2] // 2)
    local = _columns(96, COLUMN_ROW_BYTES[2], rank, 11)
    _assert_columns(processor._gather_logits(local), COLUMN_ROW_BYTES[2], 11)


def _check(rank: int) -> None:
    from vllm.distributed import get_tp_group
    from vllm.models.deepseek_v41.nvidia.specialized_decode.ampere import indexer

    gather = indexer._ampere_all_gather
    assert gather is not None
    assert gather.capacity == 16
    step = 0
    ctas = gather.ctas
    for seed, rows in enumerate([1, 6, 12, 16, 3, 6, 16, 1, 12, 7] * 2):
        gather.ctas = 5 if seed % 2 else ctas
        out = _buffer(rows, rank, seed)
        local = out[rank * rows : (rank + 1) * rows]
        _late(rank, step := step + 1)
        gather(local, out)
        _assert_gathered(out, rank, seed)
    gather.ctas = ctas
    torch.accelerator.synchronize()
    torch.distributed.barrier(get_tp_group().cpu_group)
    assert (gather.mailbox == INT32_MIN).all()
    assert int(gather.generations.min()) >= 0 and int(gather.generations.max()) < 3

    # One graph of three calls, replayed with new inputs.
    sizes = [6, 12, 6]
    outs = [_buffer(rows, rank, 100 + i) for i, rows in enumerate(sizes)]
    locals_ = [out[rank * rows : (rank + 1) * rows] for out, rows in zip(outs, sizes)]
    for local, out in zip(locals_, outs):
        gather(local, out)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for local, out in zip(locals_, outs):
            gather(local, out)
    for replay in range(REPLAYS):
        seeds = [200 + 3 * replay + i for i in range(len(sizes))]
        for out, rows, seed in zip(outs, sizes, seeds):
            out.copy_(_buffer(rows, rank, seed))
        _late(rank, step := step + 1)
        graph.replay()
        for out, seed in zip(outs, seeds):
            _assert_gathered(out, rank, seed)

    _check_shares(rank)
    _check_columns(rank)


def _pattern(rank: int, kind: int, words: int) -> torch.Tensor:
    """(rank, kind, index) as int32 bits, never negative and so never the
    sentinel; the replay counter goes into bits 26-30."""
    index = torch.arange(words, dtype=torch.int32, device="cuda") & 0xFFFFF
    return (rank << 23) | (kind << 20) | index


def _stress_graph(rank, rank_major, side_by_side, counter, errors, skew):
    """A CUDA graph of rank-major gathers (`rank_major` rows a rank, the
    indexer's gather) and side-by-side gathers (`side_by_side` rows of each
    COLUMN_ROW_BYTES size) whose contents follow the replay counter on the
    device. It adds the words that differ from the expected gather to
    `errors`."""
    from vllm.models.deepseek_v41.nvidia.specialized_decode.ampere import (
        columns,
        indexer,
    )

    calls = []
    for kind, rows in enumerate(rank_major):
        out = torch.zeros(TP * rows, TOPK, dtype=torch.int32, device="cuda")
        want = torch.cat([_pattern(r, kind, rows * TOPK) for r in range(TP)])
        local = out[rank * rows : (rank + 1) * rows]
        mine = _pattern(rank, kind, rows * TOPK).view(rows, TOPK)
        calls.append((local, out, mine, want.view(TP * rows, TOPK)))
    for kind, (row_bytes, rows) in enumerate(zip(COLUMN_ROW_BYTES, side_by_side), 3):
        words = row_bytes // 4
        local = torch.zeros(rows, row_bytes // 2, dtype=torch.bfloat16, device="cuda")
        mine = _pattern(rank, kind, rows * words).view(rows, words)
        want = [_pattern(r, kind, rows * words).view(rows, words) for r in range(TP)]
        calls.append((local, None, mine, torch.cat(want, 1)))

    def body():
        counter.add_(1)
        bits = (counter & 31) << 26
        for i, (local, out, mine, want) in enumerate(calls):
            torch.add(mine, bits, out=local.view(torch.int32))
            if skew:
                torch.cuda._sleep(LATE_CYCLES * ((rank + i) % TP))
            if out is None:
                got = columns.all_gather_columns(local)
            else:
                gather = indexer._ampere_all_gather
                assert gather is not None
                gather(local, out)
                got = out
            errors.add_((got.view(torch.int32) != want + bits).sum())

    body()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        body()
    # The graph keeps raw pointers: keep its tensors alive too.
    return graph, calls


def _check_replays(rank: int, replays: int, columns_only: bool) -> tuple[int, int]:
    """The stress graphs replayed in one seeded random order on every rank;
    returns the replays run and the mismatched words summed over the
    ranks, which all stop after the first block of 1000 replays with a
    mismatch."""
    from vllm.distributed import get_tp_group
    from vllm.models.deepseek_v41.nvidia.specialized_decode.ampere import columns

    columns.init_column_gathers(list(COLUMN_ROW_BYTES))
    counter = torch.zeros(1, dtype=torch.int32, device="cuda")
    errors = torch.zeros(1, dtype=torch.int64, device="cuda")
    graphs = [
        _stress_graph(
            rank,
            () if columns_only else rank_major,
            side_by_side,
            counter,
            errors,
            skew,
        )
        for rank_major, side_by_side in STRESS_GRAPHS
        for skew in (False, True)
    ]
    order = random.Random(1234)
    total = torch.zeros(1, dtype=torch.int64)
    done = 0
    while done < replays:
        for _ in range(min(1000, replays - done)):
            graphs[order.randrange(len(graphs))][0].replay()
        done = min(done + 1000, replays)
        torch.accelerator.synchronize()
        # Every rank stops after the same block, or the others would wait.
        total[0] = errors.item()
        torch.distributed.all_reduce(total, group=get_tp_group().cpu_group)
        if total.item():
            break
    return done, int(total.item())


def _worker(
    rank: int, port: str, skip_path: str, columns_only: bool, replays: int
) -> None:
    from tests.utils import init_test_distributed_environment
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed import cleanup_dist_env_and_memory, get_tp_group
    from vllm.models.deepseek_v41.nvidia.ops import mhc
    from vllm.models.deepseek_v41.nvidia.specialized_decode.ampere import (
        columns,
        indexer,
    )

    torch.accelerator.set_device_index(rank)
    with set_current_vllm_config(VllmConfig()), torch.inference_mode():
        init_test_distributed_environment(TP, 1, rank, port, local_rank=rank)
        config = SimpleNamespace(
            parallel_config=SimpleNamespace(
                tensor_parallel_size=TP, enable_expert_parallel=False
            ),
            model_config=SimpleNamespace(
                hf_config=SimpleNamespace(hidden_size=HIDDEN, index_topk=TOPK)
            ),
        )
        if columns_only:
            supported = torch.tensor([int(columns.supports_column_gathers(config))])
        else:
            supported = torch.tensor([int(mhc.supports_ampere_all_reduce(config))])
        torch.distributed.all_reduce(
            supported, torch.distributed.ReduceOp.MIN, get_tp_group().cpu_group
        )
        if not supported.item():
            if rank == 0:
                with open(skip_path, "w") as f:
                    f.write("vLLM's custom all-reduce is off: no full peer access")
            cleanup_dist_env_and_memory()
            return
        if not columns_only:
            indexer.init_ampere_all_gather(config)
        if replays:
            done, errors = _check_replays(rank, replays, columns_only)
            if rank == 0:
                gathers = 3 if columns_only else 6
                print(
                    f"{done} replays ({gathers * done} gathers) on {TP} "
                    f"ranks: {errors} mismatched words"
                )
            assert errors == 0
        elif columns_only:
            _check_columns(rank)
        else:
            _check(rank)
        cleanup_dist_env_and_memory()


def _spawn(columns_only: bool, replays: int = 0) -> str | None:
    """Run the checks (or the stress replays) on eight GPUs; returns why they
    were skipped, if they were."""
    if torch.accelerator.device_count() < TP:
        return "needs eight GPUs"
    from vllm.utils.network_utils import get_open_port

    with tempfile.TemporaryDirectory() as tmp:
        skip_path = os.path.join(tmp, "skipped")
        torch.multiprocessing.spawn(
            _worker,
            args=(str(get_open_port()), skip_path, columns_only, replays),
            nprocs=TP,
        )
        if os.path.exists(skip_path):
            with open(skip_path) as f:
                return f.read()
    return None


@pytest.mark.distributed(num_gpus=8)
@pytest.mark.skipif(not current_platform.is_device_capability(80), reason="needs SM80")
def test_ampere_all_gather():
    skipped = _spawn(columns_only=False)
    if skipped:
        pytest.skip(skipped)


@pytest.mark.distributed(num_gpus=8)
@pytest.mark.skipif(not current_platform.is_device_capability(90), reason="needs SM90")
def test_hopper_column_gathers():
    skipped = _spawn(columns_only=True)
    if skipped:
        pytest.skip(skipped)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--replays", type=int, default=2_000_000)
    columns_only = not current_platform.is_device_capability(80)
    skipped = _spawn(columns_only, parser.parse_args().replays)
    if skipped:
        print(f"skipped: {skipped}")
