# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The specialized decode's TP4 PLE row gather and replicated token embedding
against the stock path, on four GPUs.

`PleRowGather` (`specialized_decode/ple/gather.py`) must return what the
stock PLE finalize and dequant return (the int8 all-reduce of the ranks'
masked FP8 lookups, then `rows.to(bf16) * scale.to(bf16)`), negative zeros
as positive zeros, the same bits on every rank: for calls of alternating sizes
(1 to 256 tokens of 16 rows, so 64 and 256 tokens take every CTA's
grid-stride loop), ids at every rank's range boundaries, with and without
the early trigger, eagerly and under CUDA graph replay with changed inputs.
The race stress of `replay_stress.py` replays graphs of three back-to-back
gathers (4 to 256 tokens), some with ranks delayed, in a seeded random order;
in half of them one rank owns no row of the middle call, which an all-reduce
follows. Every replay writes new FP8 bytes and the graph counts the output
words that differ from their dequantized values, so a word left from an
earlier call counts as bad.
Run the file as a module on four GPUs for a long stress run (`--replays`).
A rank that skips a call makes the others report the stall and raise instead
of hanging.

`ReplicatedEmbedding` (`runtime/embedding.py`) must leave the stock
lookup and all-reduce to eager forwards and, in CUDA graphs, look up the full
table: the stock rows with negative zeros kept; the module's shard becomes a
view of the table. The TP4 cases need four GPUs with NVLink and Ray and skip
without them.

`PaddedProjection` (`runtime/ple.py`) must keep the PLE projection of
fewer than 16 rows within BF16 rounding of the stock one (it multiplies 16
rows, so cuBLAS sums K in another order), whatever the buffer's other rows
hold, and leave every other input to the stock method.
"""

import pytest
import torch

from vllm.platforms import current_platform

from ....utils import (
    ensure_model_parallel_initialized,
    init_test_distributed_environment,
    multi_process_parallel,
)
from .replay_stress import REPLAYS, SKEWS, ReplayStress, StressGraph, report, spawn

try:
    import ray
except ImportError:
    ray = None

TP = 4
HEADS = 16
HEAD_DIM = 160
# The table's rows split in four ranges of RANGE rows (the last one shorter),
# as the PLE table's ranges at TP4 (about 80M rows each).
RANGE = 1000
TABLE_ROWS = 4 * RANGE - 37
SCALE = 0.01234
# The largest call: the decode's TP4 steps of up to 64 requests of four rows.
MAX_TOKENS = 256
# The race stress: three calls a graph, at token counts of the decode's TP4
# steps (4 to 64 tokens of 16 rows, and some up to 256), as the logits
# gather's stress.
STRESS_TOKENS = (
    (4, 64, 16),
    (64, 8, 32),
    (12, 48, 4),
    (40, 60, 28),
    (20, 36, 56),
    (44, 24, 52),
    (256, 4, 128),
)

requires_sm100 = pytest.mark.skipif(
    not current_platform.is_cuda()
    or not current_platform.is_device_capability_family(100),
    reason="the specialized decode's PLE gather requires SM100",
)
requires_tp4 = pytest.mark.skipif(
    ray is None
    or not torch.cuda.is_available()
    or torch.accelerator.device_count() < TP,
    reason="the TP4 cases need four GPUs with NVLink and Ray",
)


def _on_ray_workers(monkeypatch: pytest.MonkeyPatch, worker) -> None:
    """Run `worker` on one Ray worker per TP4 rank (one task per worker
    process)."""
    multi_process_parallel(
        monkeypatch, TP, 1, ray.remote(num_gpus=1, max_calls=1)(worker)
    )


def _join(monkeypatch, tp_size, pp_size, rank, distributed_init_port):
    """Join the TP4 group on `rank`'s GPU; returns the device."""
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    device = torch.device(f"cuda:{rank}")
    torch.accelerator.set_device_index(device)
    init_test_distributed_environment(tp_size, pp_size, rank, distributed_init_port)
    ensure_model_parallel_initialized(tp_size, pp_size)
    # The stock dequant's FP32 global scale, as the PLE table holds it.
    _SCALES[device] = torch.tensor([SCALE], dtype=torch.float32, device=device)
    return device


def _ids(
    tokens: int, seed: int, device: torch.device, idle: int | None = None
) -> torch.Tensor:
    """`[tokens, HEADS]` ids, the same on every rank: random rows, and the
    first and last row of every rank's range among them; with `idle`, none
    of that rank's rows."""
    generator = torch.Generator(device=device).manual_seed(seed)
    ranks = [r for r in range(TP) if r != idle]
    skipped = 0 if idle is None else min((idle + 1) * RANGE, TABLE_ROWS) - idle * RANGE
    ids = torch.randint(
        0, TABLE_ROWS - skipped, (tokens * HEADS,), generator=generator, device=device
    )
    if idle is not None:
        ids = torch.where(ids >= idle * RANGE, ids + skipped, ids)
    edges = [r * RANGE for r in ranks] + [
        min((r + 1) * RANGE, TABLE_ROWS) - 1 for r in ranks
    ]
    count = min(len(edges), ids.numel())
    ids[:count] = torch.tensor(edges[:count], device=device)
    return ids.view(tokens, HEADS)


def _lookup(ids: torch.Tensor, rank: int, seed: int) -> torch.Tensor:
    """The rank's FP8 lookup `[tokens, HEADS, HEAD_DIM]`: the rows of the ids
    it owns (random, a few negative zeros), zeros elsewhere."""
    generator = torch.Generator(device=ids.device).manual_seed(seed)
    rows = (
        torch.randn(*ids.shape, HEAD_DIM, generator=generator, device=ids.device) * 4
    ).to(torch.float8_e4m3fn)
    rows.view(torch.uint8)[..., :3] = 0x80
    owned = (ids >= rank * RANGE) & (ids < min((rank + 1) * RANGE, TABLE_ROWS))
    bytes_ = torch.where(owned[..., None], rows.view(torch.uint8), 0)
    return bytes_.view(torch.float8_e4m3fn)


_SCALES: dict[torch.device, torch.Tensor] = {}


def _stock(lookup: torch.Tensor) -> torch.Tensor:
    """The stock finalize and dequant of the ranks' lookups, negative zeros as
    positive zeros: `[tokens, HEADS * HEAD_DIM]` BF16. Capturable once the
    device's scale exists (`_scale`)."""
    from vllm.distributed import get_tp_group

    reduced = get_tp_group().all_reduce(lookup.view(torch.int8))
    rows = reduced.view(torch.float8_e4m3fn).reshape(lookup.shape[0], -1)
    scale = _SCALES[lookup.device]
    out = rows.to(torch.bfloat16) * scale.to(torch.bfloat16)
    return torch.where(out == 0, torch.zeros_like(out), out)


def _gather(rank: int, device: torch.device, trigger_early: bool, timeout=30.0):
    """A `PleRowGather` over the TP group for up to `MAX_TOKENS` tokens."""
    from vllm.distributed import get_tp_group
    from vllm.models.qwen4_exp.nvidia.specialized_decode.ple.gather import (
        PleRowGather,
    )

    return PleRowGather(
        get_tp_group().device_group,
        HEAD_DIM,
        MAX_TOKENS * HEADS,
        rank * RANGE,
        min((rank + 1) * RANGE, TABLE_ROWS),
        SCALE,
        device,
        stall_timeout=timeout,
        trigger_early=trigger_early,
    )


def _same_on_every_rank(t: torch.Tensor) -> None:
    """`t` holds the same bits on every rank."""
    from vllm.distributed import get_tp_group

    parts = [torch.empty_like(t) for _ in range(TP)]
    torch.distributed.all_gather(parts, t, group=get_tp_group().device_group)
    for part in parts:
        assert torch.equal(part, t)


def _tp_gather(monkeypatch, tp_size, pp_size, rank, distributed_init_port):
    with monkeypatch.context() as m:
        device = _join(m, tp_size, pp_size, rank, distributed_init_port)
        for trigger_early in (False, True):
            gather = _gather(rank, device, trigger_early)
            sms = torch.cuda.get_device_properties(device).multi_processor_count
            assert sms <= gather.resident
            assert gather.ctas(HEADS) < gather.ctas(64 * HEADS) <= gather.resident
            for step in range(35):
                tokens = (1, 64, 4, MAX_TOKENS, 16, 3, 2)[step % 7]
                ids = _ids(tokens, 100 * step, device)
                lookup = _lookup(ids, rank, 7 * step + 1)
                out = torch.empty(
                    tokens, HEADS * HEAD_DIM, dtype=torch.bfloat16, device=device
                )
                gather(lookup, ids, out)
                want = _stock(lookup)
                assert torch.equal(out.view(torch.int16), want.view(torch.int16))
                _same_on_every_rank(out)

            # Captured calls replay to the stock rows, with changed inputs.
            ids = _ids(16, 5, device)
            lookup = _lookup(ids, rank, 11)
            out = torch.empty(16, HEADS * HEAD_DIM, dtype=torch.bfloat16, device=device)
            small = torch.empty(
                1, HEADS * HEAD_DIM, dtype=torch.bfloat16, device=device
            )
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                gather(lookup[:1], ids[:1], small)
                gather(lookup, ids, out)
            for _ in range(25):
                graph.replay()
            torch.accelerator.synchronize()
            assert torch.equal(out.view(torch.int16), _stock(lookup).view(torch.int16))
            ids.copy_(_ids(16, 6, device))
            lookup.copy_(_lookup(ids, rank, 12))
            graph.replay()
            torch.accelerator.synchronize()
            assert torch.equal(out.view(torch.int16), _stock(lookup).view(torch.int16))
            assert torch.equal(
                small.view(torch.int16), _stock(lookup[:1]).view(torch.int16)
            )
        _stress(rank, REPLAYS)


def _stress_graphs(stress: ReplayStress, rank: int) -> None:
    """Add the PLE gather graphs to `stress` (`replay_stress.py`): each
    replay writes new FP8 bytes for every call (the same bytes on every rank,
    a function of the element, the call and the replay; zeros in the rows
    another rank owns), gathers them, and counts the output words that differ
    from the bytes' dequantized values. In half of the graphs one rank owns no
    row of the middle call, and an all-reduce over the group follows that call
    (`ple/gather.py`: a rank that owns no row does not hold the others
    back)."""
    from vllm.distributed import get_tp_group

    device = stress.device
    gather = _gather(rank, device, True)
    scale = torch.tensor(SCALE, dtype=torch.float32, device=device).to(torch.bfloat16)
    end = min((rank + 1) * RANGE, TABLE_ROWS)
    separator = torch.ones(16, dtype=torch.int32, device=device)
    for index, token_counts in enumerate(STRESS_TOKENS * 2):
        idle = index % TP if index >= len(STRESS_TOKENS) else None
        calls = []
        for k, tokens in enumerate(token_counts):
            ids = _ids(tokens, 1000 + 10 * index + k, device, idle if k == 1 else None)
            shape = (tokens, HEADS, HEAD_DIM)
            calls.append(
                {
                    "ids": ids,
                    "owned": ((ids >= rank * RANGE) & (ids < end)).long()[..., None],
                    "pattern": torch.arange(
                        tokens * HEADS * HEAD_DIM, dtype=torch.int64, device=device
                    ).view(shape)
                    * 7,
                    "bytes": torch.empty(shape, dtype=torch.uint8, device=device),
                    "lookup": torch.empty(shape, dtype=torch.uint8, device=device),
                    "want": torch.empty(
                        tokens, HEADS * HEAD_DIM, dtype=torch.bfloat16, device=device
                    ),
                    "out": torch.empty(
                        tokens, HEADS * HEAD_DIM, dtype=torch.bfloat16, device=device
                    ),
                }
            )
        for skew in SKEWS:

            def body(calls=calls, skew=skew, index=index, idle=idle) -> None:
                stress.counter.add_(1)
                for k, call in enumerate(calls):
                    # Positive finite E4M3 bytes (8 to 111): another call or
                    # replay gives other bytes. Kernels write every input.
                    t = stress.counter * len(calls) + k
                    values = (call["pattern"] + 5 * t) % 104 + 8
                    call["bytes"].copy_(values)
                    call["lookup"].copy_(values * call["owned"])
                    rows = call["bytes"].view(torch.float8_e4m3fn).to(torch.bfloat16)
                    torch.mul(rows.flatten(1), scale, out=call["want"])
                for k, call in enumerate(calls):
                    stress.delay(skew, index, k)
                    gather(
                        call["lookup"].view(torch.float8_e4m3fn),
                        call["ids"],
                        call["out"],
                    )
                    if idle is not None and k == 1:
                        get_tp_group().all_reduce(separator)
                for call in calls:
                    stress.count_words(call["out"], call["want"])

            stress.graphs.append(
                StressGraph(
                    body,
                    len(token_counts),
                    sum(n * HEADS * HEAD_DIM // 2 for n in token_counts),
                    0,
                    (calls, gather, separator),
                )
            )


def _stress(rank: int, replays: int, log: bool = False) -> None:
    """The race stress of the PLE gather on this rank of the TP4 group; every
    rank must call it."""
    stress = ReplayStress(torch.device("cuda", rank))
    _stress_graphs(stress, rank)
    result = report("ple gather", stress.run(replays, log=log))
    assert result.bad_words == 0, result
    assert result.eager_bad == 0, result


@requires_sm100
@requires_tp4
def test_tp4_ple_gather_matches_stock(monkeypatch):
    _on_ray_workers(monkeypatch, _tp_gather)


# The stall test's gathers, kept until the worker process exits (one task per
# worker, ``max_calls=1``): freeing symmetric memory after a trap aborts.
_STALLED: list[object] = []


def _tp_stall(monkeypatch, tp_size, pp_size, rank, distributed_init_port):
    from vllm.distributed import get_tp_group

    with monkeypatch.context() as m:
        device = _join(m, tp_size, pp_size, rank, distributed_init_port)
        gather = _gather(rank, device, True, timeout=1.0)
        _STALLED.append(gather)
        skipper = tp_size - 1
        ids = torch.full((1, HEADS), skipper * RANGE, dtype=torch.int64, device=device)
        lookup = _lookup(ids, rank, 3)
        out = torch.empty(1, HEADS * HEAD_DIM, dtype=torch.bfloat16, device=device)
        if rank != skipper:
            with pytest.raises(RuntimeError):
                gather(lookup, ids, out)
                torch.accelerator.synchronize(device)
            report = gather.stall_report()
            assert report is not None and report.startswith(f"rank {rank} timed out")
        torch.distributed.barrier(group=get_tp_group().cpu_group)


@requires_sm100
@requires_tp4
def test_tp4_ple_gather_reports_a_stall(monkeypatch):
    _on_ray_workers(monkeypatch, _tp_stall)


def _tp_embedding(monkeypatch, tp_size, pp_size, rank, distributed_init_port):
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.vocab_parallel_embedding import (
        VocabParallelEmbedding,
    )
    from vllm.models.qwen4_exp.nvidia.specialized_decode.runtime.embedding import (
        replicate_embedding,
    )

    vocab, hidden = 248320, 2560
    with monkeypatch.context() as m, set_current_vllm_config(VllmConfig()):
        device = _join(m, tp_size, pp_size, rank, distributed_init_port)
        generator = torch.Generator(device=device).manual_seed(3)
        full = torch.randn(vocab, hidden, generator=generator, device=device).to(
            torch.bfloat16
        )
        full[::7, :5] = -0.0
        embed = VocabParallelEmbedding(vocab, hidden, params_dtype=torch.bfloat16).to(
            device
        )
        start = embed.shard_indices.org_vocab_start_index
        end = embed.shard_indices.org_vocab_end_index
        embed.weight.data[: end - start].copy_(full[start:end])
        # The same ids on every rank (the stock all-reduce sums the ranks'
        # lookups of their own ids): random ones, the vocabulary's ends and
        # every rank's first and last row.
        ids = torch.randint(0, vocab, (64,), generator=generator, device=device).to(
            torch.int32
        )
        part = end - start
        edges = [0, vocab - 1]
        edges += [r * part for r in range(tp_size)]
        edges += [(r + 1) * part - 1 for r in range(tp_size)]
        ids[: len(edges)] = torch.tensor(edges, device=device)
        stock = embed(ids).clone()

        replicated = replicate_embedding(embed)
        assert replicated is not None
        assert torch.equal(replicated.table, full)
        # The shard is a view of the table now.
        assert embed.weight.data_ptr() == replicated.table[start:].data_ptr()
        # Eager: the stock lookup and all-reduce.
        assert torch.equal(embed(ids).view(torch.int16), stock.view(torch.int16))
        # Captured: the full table's rows, negative zeros kept.
        rows = (64, 16, 4, 1)
        outs = []
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for n in rows:
                outs.append(embed(ids[:n]))
            outs.append(embed(ids[:8].view(2, 4)))
        graph.replay()
        torch.accelerator.synchronize()
        for n, out in zip(rows, outs):
            want = full[ids[:n].long()]
            assert torch.equal(out.view(torch.int16), want.view(torch.int16))
            normalized = torch.where(out == 0, torch.zeros_like(out), out)
            assert torch.equal(
                normalized.view(torch.int16), stock[:n].view(torch.int16)
            )
        assert outs[-1].shape == (2, 4, hidden)
        assert torch.equal(outs[-1].reshape(8, hidden), full[ids[:8].long()])


@requires_sm100
@requires_tp4
def test_tp4_replicated_embedding_matches_stock(monkeypatch):
    _on_ray_workers(monkeypatch, _tp_embedding)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("rows", [1, 4, 8, 12])
def test_padded_projection_is_within_bf16_rounding(rows: int):
    # The PLE projection's padded rows (runtime/ple.py): within BF16 rounding
    # of the stock projection of the step's rows, and no further from an FP32
    # reference than one BF16 unit of the largest output.
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod
    from vllm.models.qwen4_exp.nvidia.specialized_decode.runtime.ple import (
        PAD_ROWS,
        PaddedProjection,
    )

    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(rows)
    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(
        (0.02 * torch.randn(12800, 2560, generator=generator, device=device)).to(
            torch.bfloat16
        ),
        requires_grad=False,
    )
    stock = UnquantizedLinearMethod()
    buffer = torch.randn(PAD_ROWS, 2560, generator=generator, device=device).to(
        torch.bfloat16
    )
    x = buffer[:rows]
    padded = PaddedProjection(stock, buffer, PAD_ROWS)
    got = padded.apply(layer, x)
    want = stock.apply(layer, x)
    reference = x.float() @ layer.weight.float().T
    assert got.shape == want.shape == (rows, 12800)
    scale = float(reference.abs().max())
    assert float((got.float() - want.float()).abs().max()) <= scale * 2**-7
    assert float((got.float() - reference).abs().max()) <= scale * 2**-7
    # Rows past the step's do not change the step's rows.
    buffer[rows:] = float("nan")
    assert torch.equal(padded.apply(layer, x), got)
    # Anything but the buffer's leading rows takes the stock method.
    other = x.clone()
    assert torch.equal(padded.apply(layer, other), stock.apply(layer, other))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Race stress of the PLE gather on four GPUs (replay_stress.py)"
    )
    parser.add_argument("--replays", type=int, default=REPLAYS)
    args = parser.parse_args()
    spawn(_stress, args.replays, True)
