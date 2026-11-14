# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
r"""Race stress of the specialized decode's cross-rank kernels at TP4.

The boundary kernel (`SublayerBoundary`), the target logits all-gather
(`LogitsAllGather`), the draft token choice (`TokenChoice`) and the PLE
row gather (`PleRowGather`) exchange data through mailboxes in symmetric
memory that their threads poll, under programmatic dependent launch, inside
CUDA graphs. A protocol slip (a slot
read before its sender wrote it, a slot reset late, a generation or epoch
reused too early) would not hang: the receiver would take a word of an
earlier call and go on. These checks look for such words.

`test_boundary.py`, `test_logits.py` and `test_ple.py` build the stress
graphs: each runs three calls of one kernel back to back, at row counts the
decode takes, with no rank, one rank or every rank delayed before its calls;
each `StressGraph` keeps the tensors its graph addresses. Every replay bumps a
device counter and writes new inputs from it, so every input word encodes its
rank, its call and the replay; the graph then counts, with torch ops on the
GPU, the output words that differ from the ones those inputs determine. The
graphs replay in a seeded random order, the same on every rank, so calls of
different sizes follow each other in every combination. A word left from an
earlier call carries another replay or call and counts as bad. The boundary's
normed output, whose last bit the GPU's approximate division and reciprocal
square root decide, counts as bad where it is more than one BF16 unit in the
last place from the expected value ("near words").

Under pytest the TP4 tests of these files replay the graphs `REPLAYS` times
and require no bad word. For a long run, run the files as modules on four
GPUs (the stress skips nothing and takes no pytest options)::

    python -m tests.models.qwen4_exp.specialized_decode.test_boundary \
        --replays 2000000
    python -m tests.models.qwen4_exp.specialized_decode.test_logits \
        --kernel gather --replays 2000000
    python -m tests.models.qwen4_exp.specialized_decode.test_ple \
        --replays 2000000

`--control` replays the same graphs with one of vLLM's own collectives
doing the exchange (its custom all-reduce, or the MNNVL Lamport all-gather of
`CustomAllreduce.custom_all_gather`) and only counts: a control that finds
bad words shows that the check sees them.
"""

import gc
import random
import time
from collections.abc import Callable
from dataclasses import dataclass

import torch

from ....utils import (
    ensure_model_parallel_initialized,
    init_test_distributed_environment,
)

TP = 4
# Replays of the TP4 tests under pytest.
REPLAYS = 2000
# Graph replays between two host synchronizations.
BLOCK = 1000
# Cycles of one delay step (about 10 us).
LATE_CYCLES = 20_000
# How the graphs of one set of row counts delay the ranks: not at all; every
# call on every rank by ``(rank + call) % TP`` steps; one rank (the graph
# set's index modulo TP) by three steps before the first call only.
SKEWS = ("none", "every call", "one rank")
# The control's exchanges.
CONTROLS = ("all-reduce", "all-gather")


@dataclass
class StressGraph:
    """One graph of the stress: `body` runs one replay's work (new inputs
    from the counter, the calls, the checks); `words`, `tokens` and
    `near_words` are what one replay checks on this rank; `keep` holds
    the tensors the captured graph addresses."""

    body: Callable[[], None]
    calls: int
    words: int
    tokens: int
    keep: object
    near_words: int = 0


@dataclass
class StressResult:
    """The counts of one rank, or of all ranks after `report`;
    `eager_bad` is None when the eager runs are not checked."""

    replays: int
    calls: int
    words: int
    bad_words: int
    tokens: int
    bad_tokens: int
    near_words: int
    bad_near_words: int
    eager_bad: int | None


class ReplayStress:
    """The counter the graphs derive their inputs from, the device counts of
    bad words, bad tokens and bad near words, and the graphs.

    `check_eager` False leaves the eager run before each capture unchecked:
    inside vLLM's `graph_capture` its custom all-reduce returns an
    uninitialized tensor outside the capture itself."""

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.rank = torch.distributed.get_rank()
        self.counter = torch.zeros(1, dtype=torch.int64, device=device)
        # Bad words, bad tokens, bad near words.
        self.errors = torch.zeros(3, dtype=torch.int64, device=device)
        self.graphs: list[StressGraph] = []
        self.check_eager = True

    def delay(self, skew: str, graph_set: int, call: int) -> None:
        """Delay this rank before `call` as `skew` says (`SKEWS`)."""
        if skew == "every call":
            steps = (self.rank + call) % TP
        elif skew == "one rank":
            steps = 3 if call == 0 and self.rank == graph_set % TP else 0
        else:
            steps = 0
        if steps:
            torch.cuda._sleep(LATE_CYCLES * steps)

    def count_words(self, got: torch.Tensor, want: torch.Tensor) -> None:
        """Add the 32-bit words where `got` and `want` differ."""
        self.errors[0:1].add_((got.view(torch.int32) != want.view(torch.int32)).sum())

    def count_tokens(self, got: torch.Tensor, want: torch.Tensor) -> None:
        """Add the tokens where `got` and `want` differ."""
        self.errors[1:2].add_((got != want).sum())

    def count_near(self, got: torch.Tensor, want: torch.Tensor) -> None:
        """Add the 32-bit words of the BF16 `got` with a half more than one
        unit in the last place from `want` (broadcast to `got`): for
        outputs whose rounding the GPU's approximate instructions decide."""
        ulps = (got.view(torch.int16).int() - want.view(torch.int16).int()).abs()
        far = (ulps > 1).view(*got.shape[:-1], got.shape[-1] // 2, 2).any(-1)
        self.errors[2:3].add_(far.sum())

    def run(self, replays: int, seed: int = 1234, log: bool = False) -> StressResult:
        """Capture every graph (each after one eager run), then replay them
        `replays` times in a seeded random order; the counts of this rank."""
        from vllm.distributed.parallel_state import graph_capture

        # An earlier stress's graphs and kernels form reference cycles; a
        # collection during a capture would free their symmetric and pinned
        # memory there, which CUDA does not permit.
        gc.collect()
        captured = []
        with graph_capture(device=self.device) as context:
            for graph in self.graphs:
                graph.body()
                cuda_graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(cuda_graph, stream=context.stream):
                    graph.body()
                captured.append(cuda_graph)
        torch.accelerator.synchronize()
        eager_bad = int(self.errors.sum().item()) if self.check_eager else None
        self.errors.zero_()
        order = random.Random(seed)
        calls = words = tokens = near_words = 0
        start = time.monotonic()
        for first in range(0, replays, BLOCK):
            for _ in range(min(BLOCK, replays - first)):
                index = order.randrange(len(captured))
                captured[index].replay()
                graph = self.graphs[index]
                calls += graph.calls
                words += graph.words
                tokens += graph.tokens
                near_words += graph.near_words
            torch.accelerator.synchronize()
            done = min(first + BLOCK, replays)
            if log and self.rank == 0 and done % 200_000 == 0:
                bad_words, bad_tokens, bad_near = self.errors.tolist()
                print(
                    f"{done} replays in {time.monotonic() - start:.0f} s: "
                    f"{bad_words} bad words, {bad_tokens} bad tokens, "
                    f"{bad_near} bad near words on rank 0",
                    flush=True,
                )
        bad_words, bad_tokens, bad_near = self.errors.tolist()
        return StressResult(
            replays,
            calls,
            words,
            bad_words,
            tokens,
            bad_tokens,
            near_words,
            bad_near,
            eager_bad,
        )


def custom_comm(control: str):
    """The TP group's `CustomAllreduce` (vLLM's own collectives), which the
    control's exchange (`CONTROLS`) runs on."""
    from vllm.distributed import get_tp_group

    ca_comm = getattr(get_tp_group().device_communicator, "ca_comm", None)
    if ca_comm is None or ca_comm.disabled:
        raise RuntimeError("vLLM's custom all-reduce is not available")
    if control == "all-gather" and not ca_comm.mnnvl_lamport_ag_multicast_ptr:
        raise RuntimeError("vLLM's MNNVL Lamport all-gather is not available")
    return ca_comm


def report(name: str, result: StressResult) -> StressResult:
    """Sum the ranks' counts (collective over the TP group's CPU group), print
    them on rank 0 and return the sums."""
    from vllm.distributed import get_tp_group

    fields = (
        "calls",
        "words",
        "bad_words",
        "tokens",
        "bad_tokens",
        "near_words",
        "bad_near_words",
    )
    # Every rank checks its eager runs or none does; -1: not checked.
    eager = -1 if result.eager_bad is None else result.eager_bad
    counts = torch.tensor([getattr(result, field) for field in fields] + [eager])
    torch.distributed.all_reduce(counts, group=get_tp_group().cpu_group)
    sums = dict(zip(fields, (int(c) for c in counts)))
    total = StressResult(
        replays=result.replays,
        **sums,
        eager_bad=None if result.eager_bad is None else int(counts[-1]),
    )
    if torch.distributed.get_rank() == 0:
        near = (
            f", bad near words (more than one BF16 ulp off) {total.bad_near_words} "
            f"of {total.near_words}"
            if total.near_words
            else ""
        )
        eager_runs = "not checked" if total.eager_bad is None else total.eager_bad
        print(
            f"STRESS {name}: {total.replays} replays, {total.calls} calls on "
            f"{TP} ranks; bad words {total.bad_words} of {total.words}, bad "
            f"tokens {total.bad_tokens} of {total.tokens}{near}; eager runs "
            f"before capture: {eager_runs} bad",
            flush=True,
        )
    return total


def _process(
    rank: int,
    port: int,
    target: Callable[..., None],
    args: tuple,
) -> None:
    """One rank of `spawn`: join the TP group, run `target(rank,
    *args)`, leave."""
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed import cleanup_dist_env_and_memory

    torch.accelerator.set_device_index(rank)
    with set_current_vllm_config(VllmConfig()):
        init_test_distributed_environment(TP, 1, rank, str(port), local_rank=rank)
        ensure_model_parallel_initialized(TP, 1)
        target(rank, *args)
        torch.accelerator.synchronize()
        cleanup_dist_env_and_memory()


def spawn(target: Callable[..., None], *args: object) -> None:
    """Run `target(rank, *args)` on four GPUs, one process a rank, each in
    the TP group of all four."""
    from vllm.utils.network_utils import get_open_port

    if torch.accelerator.device_count() < TP:
        raise SystemExit("the stress needs four GPUs with NVLink")
    torch.multiprocessing.spawn(
        _process, args=(get_open_port(), target, args), nprocs=TP
    )
