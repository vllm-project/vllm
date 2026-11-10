# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The specialized decode's logits kernels against the stock path.

`TokenChoice` (the MTP draft passes' greedy token on the vocabulary shard)
must return `greedy_argmax` of the gathered logits, the stock draft choice,
on every rank: ties to the lowest index, NaN as -inf, -0 equal to +0, padding
never chosen; eagerly and under CUDA graph replay (the exchange epochs rotate),
at TP 1 and TP 4, with both chunk sizes (16 and 64 rows at TP 1 and 64 rows at
TP 4 take the large chunks, interleaved with the small ones at TP 4).
`LogitsAllGather` must return what
`tensor_model_parallel_all_gather` returns, with negative zeros as positive
zeros, the same bits on every rank, for calls of alternating sizes (64 and
256 rows take every CTA's grid-stride loop). At TP 4, a rank that skips a call
makes the others report the stall and raise instead of hanging, and the token
choice refuses as many rows as the GPU has SMs (its liveness bound). At TP 4
the race stress of `replay_stress.py` replays graphs of back-to-back gathers
(4 to 64 rows) and token choices (1 to 64 rows), some with ranks delayed, in a
random order, and finds no stale word or token. The TP 4 cases need four GPUs
and Ray and skip without them. Run as a module on four GPUs (`--kernel gather` or
`--kernel choice`) for a long stress run (`--replays`) or the control on
vLLM's custom all-gather (`--control all-gather`)."""

import pytest
import torch

from vllm.platforms import current_platform

from ....utils import (
    ensure_model_parallel_initialized,
    init_test_distributed_environment,
    multi_process_parallel,
)
from .replay_stress import (
    REPLAYS,
    SKEWS,
    ReplayStress,
    StressGraph,
    custom_comm,
    report,
    spawn,
)

try:
    import ray
except ImportError:  # a job without Ray runs the TP1 cases only
    ray = None

VOCAB = 248320
TP = 4
SHARD = VOCAB // TP
WORDS = SHARD // 2  # 32-bit words of a rank's logits row
# The race stress: three calls a graph; the gather takes every target row
# count of the decode's TP4 steps of up to 16 requests (4 to 64; its words
# encode the row in six bits), the token choice every draft row count up to 16
# and some up to 64 (the large chunks from 34 rows at TP4).
GATHER_STRESS_ROWS = (
    (4, 64, 16),
    (64, 8, 32),
    (12, 48, 4),
    (40, 60, 28),
    (20, 36, 56),
    (44, 24, 52),
)
CHOICE_STRESS_ROWS = (
    (1, 16, 4),
    (16, 2, 8),
    (3, 12, 1),
    (10, 15, 7),
    (5, 9, 14),
    (11, 6, 13),
    (64, 33, 48),
)
# vLLM's custom all-gather (the control) takes at most 4 MiB a rank: 32 rows.
CONTROL_GATHER_ROWS = 32

requires_sm100 = pytest.mark.skipif(
    not current_platform.is_cuda()
    or not current_platform.is_device_capability_family(100),
    reason="the specialized decode's logits kernels require SM100",
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


def _logits(rows: int, vocab: int, seed: int, device: torch.device) -> torch.Tensor:
    """Full-vocabulary BF16 logits with the cases the choice must order like
    the stock argmax: ties of the row maximum (within a shard and across
    shards), NaN, -inf, +-0, and a row of zeros with one -0 first."""
    generator = torch.Generator(device=device).manual_seed(seed)
    logits = (torch.randn(rows, vocab, generator=generator, device=device) * 4).to(
        torch.bfloat16
    )
    for row in range(rows):
        kind = row % 6
        top = float(logits[row].float().max()) + 1.0
        if kind == 1:
            # Tie inside one shard and across shards: the lowest index wins.
            for index in (vocab // 2 + 7, vocab // 2 + 3, vocab - 1, vocab // 50):
                logits[row, index] = top
        elif kind == 2:
            # NaN counts as -inf.
            logits[row, 11] = float("nan")
            logits[row, vocab - 2] = float("nan")
        elif kind == 3:
            logits[row] = float("-inf")
            logits[row, vocab - 5] = 0.0
        elif kind == 4:
            logits[row] = 0.0
            logits[row, 3] = -0.0
            # Subnormals order as values (no flush to zero): the positive
            # one wins, the negative one stays below the zeros.
            tiny = torch.tensor([1, -32767], dtype=torch.int16).view(torch.bfloat16)
            logits[row, vocab // 3] = tiny[0]
            logits[row, vocab // 7] = tiny[1]
        elif kind == 5:
            logits[row] = float("-inf")
    return logits


def _stock_tokens(logits: torch.Tensor) -> torch.Tensor:
    from vllm.v1.worker.gpu.sample.greedy_argmax import greedy_argmax

    return greedy_argmax(logits)


def _check_graph(choice, logits: torch.Tensor, expected) -> None:
    """Captured launches replay to the right tokens, after many replays (the
    epochs rotate) and with changed inputs."""
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(3):
            choice(logits)
        captured = choice(logits)
    for _ in range(25):
        graph.replay()
    torch.accelerator.synchronize()
    assert torch.equal(captured, expected(logits))
    logits.mul_(-1)
    graph.replay()
    torch.accelerator.synchronize()
    assert torch.equal(captured, expected(logits))


@requires_sm100
@pytest.mark.parametrize("rows", [1, 4, 16, 64])
def test_local_choice_matches_greedy_argmax(rows: int):
    from vllm.models.qwen4_exp.nvidia.specialized_decode.logits.choice import (
        TokenChoice,
    )

    device = torch.device("cuda")
    choice = TokenChoice(None, VOCAB, VOCAB, 0, 64, device)
    logits = _logits(rows, VOCAB, rows, device)
    assert torch.equal(choice(logits), _stock_tokens(logits))
    _check_graph(choice, logits, _stock_tokens)


@requires_sm100
def test_local_choice_skips_padding_and_offsets_the_index():
    from vllm.models.qwen4_exp.nvidia.specialized_decode.logits.choice import (
        TokenChoice,
    )

    device = torch.device("cuda")
    width, valid, offset = 4104, 4099, 1000
    choice = TokenChoice(None, width, valid, offset, 8, device)
    logits = _logits(8, width, 3, device)
    logits[:, valid:] = 1e4  # padding never wins
    expected = _stock_tokens(logits[:, :valid].contiguous()) + offset
    assert torch.equal(choice(logits), expected)


def _tp_logits(monkeypatch, tp_size, pp_size, rank, distributed_init_port):
    from vllm.distributed import get_tp_group, tensor_model_parallel_all_gather
    from vllm.models.qwen4_exp.nvidia.specialized_decode.logits.choice import (
        TokenChoice,
    )
    from vllm.models.qwen4_exp.nvidia.specialized_decode.logits.gather import (
        LogitsAllGather,
    )

    with monkeypatch.context() as m:
        m.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        device = torch.device(f"cuda:{rank}")
        torch.accelerator.set_device_index(device)
        init_test_distributed_environment(tp_size, pp_size, rank, distributed_init_port)
        ensure_model_parallel_initialized(tp_size, pp_size)
        group = get_tp_group()
        shard = VOCAB // tp_size
        # One CTA a row polls: at least one SM must stay free (choice.py).
        sms = torch.cuda.get_device_properties(device).multi_processor_count
        with pytest.raises(ValueError, match="SMs"):
            TokenChoice(group.device_group, shard, shard, rank * shard, sms, device)
        choice = TokenChoice(group.device_group, shard, shard, rank * shard, 64, device)
        gather = LogitsAllGather(group.device_group, shard, 256, device)
        # 64 rows exceed what the GPU holds at once: the grid stops there.
        assert sms <= gather.resident <= sms * 2048 // 128
        assert gather.ctas(1) < gather.ctas(64) == gather.resident

        def same_on_every_rank(t: torch.Tensor) -> None:
            parts = [torch.empty_like(t) for _ in range(tp_size)]
            torch.distributed.all_gather(parts, t, group=group.device_group)
            for part in parts:
                assert torch.equal(part, t)

        def local(full: torch.Tensor) -> torch.Tensor:
            return full[:, rank * shard : (rank + 1) * shard].contiguous()

        # The token choice: greedy_argmax of the full logits on every rank.
        for rows in (1, 4, 16, 64):
            full = _logits(rows, VOCAB, rows, device)
            tokens = choice(local(full))
            assert torch.equal(tokens, _stock_tokens(full))
            same_on_every_rank(tokens)
            mine = local(full)
            _check_graph(
                choice,
                mine,
                lambda shard_logits: _stock_tokens(
                    tensor_model_parallel_all_gather(shard_logits)
                ),
            )

        # The gather: NCCL's bits with -0 as +0, calls of alternating sizes.
        def expected(shard_logits: torch.Tensor) -> torch.Tensor:
            out = tensor_model_parallel_all_gather(shard_logits)
            return torch.where(out == 0, torch.zeros_like(out), out)

        for step in range(42):
            rows = (1, 64, 4, 256, 16, 3)[step % 6]
            generator = torch.Generator(device=device).manual_seed(100 * step + rank)
            shard_logits = torch.randn(
                rows, shard, generator=generator, device=device
            ).to(torch.bfloat16)
            shard_logits[:, :8] = -0.0
            shard_logits[:, 8:16] = 0.0
            out = gather(shard_logits)
            want = expected(shard_logits)
            assert torch.equal(out.view(torch.int16), want.view(torch.int16))
            same_on_every_rank(out)

        shard_logits = torch.randn(16, shard, device=device).to(torch.bfloat16)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            gather(shard_logits)
            captured = gather(shard_logits)
        for _ in range(25):
            graph.replay()
        torch.accelerator.synchronize()
        assert torch.equal(captured, expected(shard_logits))
        shard_logits.mul_(-2)
        graph.replay()
        torch.accelerator.synchronize()
        assert torch.equal(captured, expected(shard_logits))

        _stress(rank, "gather", REPLAYS, None)
        _stress(rank, "choice", REPLAYS, None)


def _gather_words(rank: int, call: int, rows: int, device: torch.device):
    """A rank's logits in one stress call as 32-bit words: the rank (bits
    29-30), the call (27-28), the row (16-21) and the word's column (0-14);
    the replay adds bits 22-26. Bits 15 and 31 stay clear, so no BF16 half is
    a negative zero, the empty-slot sentinel."""
    row = torch.arange(rows, dtype=torch.int32, device=device)[:, None] << 16
    column = torch.arange(WORDS, dtype=torch.int32, device=device)
    return (rank << 29) | (call << 27) | row | column


def _gather_stress_graphs(stress: ReplayStress, rank: int, control: str | None):
    """Add the logits gather graphs (or the same graphs on vLLM's custom
    all-gather, at most `CONTROL_GATHER_ROWS` rows a call) to `stress`."""
    from vllm.distributed import get_tp_group
    from vllm.models.qwen4_exp.nvidia.specialized_decode.logits.gather import (
        LogitsAllGather,
    )

    device = stress.device
    if control is None:
        gather = LogitsAllGather(get_tp_group().device_group, SHARD, 64, device)
        gather.warm_up()
    else:
        ca_comm = custom_comm(control)

    def exchange(logits: torch.Tensor) -> torch.Tensor:
        if control is None:
            return gather(logits)
        rows = logits.shape[0]
        out = ca_comm.custom_all_gather(logits)
        return out.view(TP, rows, SHARD).transpose(0, 1).reshape(rows, VOCAB)

    for index, row_counts in enumerate(GATHER_STRESS_ROWS):
        if control is not None:
            row_counts = tuple(min(n, CONTROL_GATHER_ROWS) for n in row_counts)
        mine = [_gather_words(rank, k, n, device) for k, n in enumerate(row_counts)]
        want = [
            torch.cat([_gather_words(r, k, n, device) for r in range(TP)], dim=1)
            for k, n in enumerate(row_counts)
        ]
        inputs = [
            torch.empty(n, SHARD, dtype=torch.bfloat16, device=device)
            for n in row_counts
        ]
        for skew in SKEWS:

            def body(mine=mine, want=want, inputs=inputs, skew=skew, index=index):
                stress.counter.add_(1)
                bits = ((stress.counter & 0x1F) << 22).to(torch.int32)
                for k in range(len(inputs)):
                    torch.add(mine[k], bits, out=inputs[k].view(torch.int32))
                outs = []
                for k in range(len(inputs)):
                    stress.delay(skew, index, k)
                    outs.append(exchange(inputs[k]))
                for k in range(len(inputs)):
                    stress.count_words(outs[k], want[k] + bits)

            stress.graphs.append(
                StressGraph(
                    body,
                    len(row_counts),
                    sum(n * TP * WORDS for n in row_counts),
                    0,
                    (mine, want, inputs),
                )
            )


def _choice_stress_graphs(stress: ReplayStress, rank: int, control: str | None):
    """Add the token choice graphs (or, with `control`, the stock choice on
    vLLM's custom all-gather: `greedy_argmax` of the gathered logits) to
    `stress`.

    Each row's logits are small integers, `((7 p + 11 i + 13 r + 5 t) mod 61)
    - 30` at position `p` of row `i` on rank `r` in call `t`, and one
    peak a rank: at a position given by the row, call and rank; on the row's
    winner (`(i + t) mod 4`) of height `254 - 2 (e mod 64)` for the row's
    use count `e`, one lower on the other ranks. The token is the winner's
    peak. A rank's record (its peak) from a use of the row since its use count
    last passed a multiple of 64 is higher than every current one, so such a
    stale record changes the token; an older stale record can be lower and
    go unseen. The control also checks every gathered logit."""
    from vllm.distributed import get_tp_group
    from vllm.models.qwen4_exp.nvidia.specialized_decode import MAX_DECODE_REQUESTS
    from vllm.models.qwen4_exp.nvidia.specialized_decode.logits.choice import (
        TokenChoice,
    )
    from vllm.v1.worker.gpu.sample.greedy_argmax import greedy_argmax

    device = stress.device
    capacity = MAX_DECODE_REQUESTS[TP]
    if control is None:
        choice = TokenChoice(
            get_tp_group().device_group, SHARD, SHARD, rank * SHARD, capacity, device
        )
        choice.warm_up()
    else:
        ca_comm = custom_comm(control)
    uses = torch.zeros(capacity, dtype=torch.int64, device=device)
    ranks = torch.arange(TP, dtype=torch.int64, device=device)
    position = torch.arange(SHARD, dtype=torch.int64, device=device)

    def peak_position(rows, t, owner):
        """Where rank `owner`'s peak of each row lies in its shard."""
        return (
            977 * torch.arange(rows, device=device) + 131 * t + 7919 * owner
        ) % SHARD

    for index, row_counts in enumerate(CHOICE_STRESS_ROWS):
        calls = []
        for rows in row_counts:
            row = torch.arange(rows, dtype=torch.int64, device=device)
            base = (7 * position + 11 * row[:, None] + 13 * ranks[:, None, None]) % 61
            calls.append(
                {
                    "rows": rows,
                    "row": row,
                    # Every rank's background for the control, else this rank's.
                    "base": (base if control else base[rank]).to(torch.int32),
                    "logits": torch.empty(
                        rows, SHARD, dtype=torch.bfloat16, device=device
                    ),
                    "full": torch.empty(
                        TP, rows, SHARD, dtype=torch.bfloat16, device=device
                    ),
                    "want": torch.empty(rows, dtype=torch.int64, device=device),
                }
            )
        for skew in SKEWS:

            def body(calls=calls, skew=skew, index=index) -> None:
                stress.counter.add_(1)
                for k, call in enumerate(calls):
                    rows, row = call["rows"], call["row"]
                    t = stress.counter * len(calls) + k
                    background = (call["base"] + 5 * t) % 61 - 30
                    peak = 254 - 2 * (uses[:rows] % 64)
                    winner = (row + t) % TP
                    call["want"].copy_(winner * SHARD + peak_position(rows, t, winner))
                    if control is None:
                        call["logits"].copy_(background)
                        height = peak - (winner != rank).long()
                        call["logits"].scatter_(
                            1,
                            peak_position(rows, t, rank)[:, None],
                            height[:, None].bfloat16(),
                        )
                    else:
                        call["full"].copy_(background)
                        height = peak - (winner != ranks[:, None]).long()
                        call["full"].scatter_(
                            2,
                            peak_position(rows, t, ranks[:, None])[..., None],
                            height[..., None].bfloat16(),
                        )
                        call["logits"].copy_(call["full"][rank])
                    uses[:rows] += 1
                outs = []
                for k, call in enumerate(calls):
                    stress.delay(skew, index, k)
                    if control is None:
                        outs.append(choice(call["logits"]))
                    else:
                        outs.append(ca_comm.custom_all_gather(call["logits"]))
                for out, call in zip(outs, calls):
                    if control is None:
                        stress.count_tokens(out, call["want"])
                        continue
                    rows = call["rows"]
                    gathered = out.view(TP, rows, SHARD)
                    stress.count_words(gathered, call["full"])
                    logits = gathered.transpose(0, 1).reshape(rows, VOCAB)
                    stress.count_tokens(greedy_argmax(logits), call["want"])

            tokens = sum(row_counts)
            stress.graphs.append(
                StressGraph(
                    body,
                    len(calls),
                    0 if control is None else tokens * TP * WORDS,
                    tokens,
                    calls,
                )
            )


def _stress(
    rank: int, kernel: str, replays: int, control: str | None, log: bool = False
) -> None:
    """The race stress of the logits `kernel` (`gather` or `choice`,
    or the control) on this rank of the TP4 group; every rank must call it."""
    stress = ReplayStress(torch.device("cuda", rank))
    if kernel == "gather":
        _gather_stress_graphs(stress, rank, control)
    else:
        _choice_stress_graphs(stress, rank, control)
    name = kernel if control is None else f"{kernel} control ({control})"
    result = report(name, stress.run(replays, log=log))
    if control is None:
        assert result.bad_words + result.bad_tokens == 0, result
        assert result.eager_bad == 0, result


@requires_sm100
@requires_tp4
def test_tp4_logits_kernels_match_stock(monkeypatch):
    _on_ray_workers(monkeypatch, _tp_logits)


# The stall test's kernels, kept until the worker process exits (one task per
# worker, ``max_calls=1``): freeing symmetric memory after a trap aborts.
_STALLED: list[object] = []


def _tp_stall(monkeypatch, tp_size, pp_size, rank, distributed_init_port):
    from vllm.distributed import get_tp_group
    from vllm.models.qwen4_exp.nvidia.specialized_decode.logits.choice import (
        TokenChoice,
    )
    from vllm.models.qwen4_exp.nvidia.specialized_decode.logits.gather import (
        LogitsAllGather,
    )

    with monkeypatch.context() as m:
        m.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        device = torch.device(f"cuda:{rank}")
        torch.accelerator.set_device_index(device)
        init_test_distributed_environment(tp_size, pp_size, rank, distributed_init_port)
        ensure_model_parallel_initialized(tp_size, pp_size)
        group = get_tp_group()
        shard = VOCAB // tp_size
        kind = "choice" if rank % 2 == 0 else "gather"
        # Every rank builds both (collective), ranks 0-2 then call one.
        choice = TokenChoice(
            group.device_group, shard, shard, rank * shard, 4, device, stall_timeout=1.0
        )
        gather = LogitsAllGather(
            group.device_group, shard, 4, device, stall_timeout=1.0
        )
        _STALLED.extend((choice, gather))
        skipper = tp_size - 1
        logits = torch.zeros(1, shard, dtype=torch.bfloat16, device=device)
        if rank != skipper:
            kernel = choice if kind == "choice" else gather
            with pytest.raises(RuntimeError):
                kernel(logits)
                torch.accelerator.synchronize(device)
            report = kernel.stall_report()
            assert report is not None and report.startswith(f"rank {rank} timed out")
        torch.distributed.barrier(group=group.cpu_group)


@requires_sm100
@requires_tp4
def test_tp4_logits_kernels_report_a_stall(monkeypatch):
    _on_ray_workers(monkeypatch, _tp_stall)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Race stress of the logits kernels on four GPUs (replay_stress.py)"
    )
    parser.add_argument("--kernel", choices=("gather", "choice"), required=True)
    parser.add_argument("--replays", type=int, default=REPLAYS)
    parser.add_argument(
        "--control",
        choices=("all-gather",),
        help="replay the graphs on vLLM's custom all-gather and only count the "
        "bad words and tokens",
    )
    args = parser.parse_args()
    spawn(_stress, args.kernel, args.replays, args.control, True)
