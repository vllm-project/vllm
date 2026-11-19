# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The row-split reads of Qwen4Exp's specialized decode (`runtime/split.py`)
on four GPUs: chains of reads through the boundary kernel's split calls
(attention, folded MoE and MoE boundaries), the up/mix all-gather
(`HcUpMixGather`) and the final rows all-gather (`RowsAllGather`) against
the replicated reads (the one-shot boundary, then `hc_project`), bit for bit
(the gathers' negative zeros as positive zeros), eagerly and under CUDA graph
replay with changed inputs. The race stress of `replay_stress.py` replays
graphs of such chains at the decode's TP4 row counts, some with ranks delayed,
in a random order, each against the replicated chain on the same inputs, and
finds no differing word. A rank that skips a call of any of the three exchanges
makes the others report the stall and raise instead of hanging. Run as a module
on four GPUs for a long stress run (`--replays`)."""

import pytest
import ray
import torch

from vllm.platforms import current_platform

from ....utils import (
    ensure_model_parallel_initialized,
    init_test_distributed_environment,
    multi_process_parallel,
)
from .replay_stress import REPLAYS, SKEWS, TP, ReplayStress, StressGraph, report, spawn

STREAMS = 4
HIDDEN = 2560
WIDTH = STREAMS * HIDDEN
LOW_RANK = 320
TOP_K = 10
EPS = 1e-6
# The largest split step (64 rows a rank), and the most rows the replicated
# reference's read kernels take in one call.
MAX_ROWS = 256
READ_ROWS = 64
# A chain's boundaries and the reads after them: a decoder layer's reads after
# an attention and a folded MoE block, then the final mixer's after a MoE
# block (the unfolded variant).
KINDS = ("attention", "moe folded", "moe")
# The rows of the checked chains: one row a rank, the smallest split step, the
# largest of 16 requests, of 32 and of 64.
CHECK_ROWS = (4, 32, 64, 128, 256)
# The stress's chains: every row count a rank can own up to 16 at least once,
# then the first and last owned count of each larger group count (eight rows a
# group) up to 64 a rank.
STRESS_ROWS = (
    *range(4, 68, 4),
    *(4 * owned for owned in (17, 24, 25, 32, 33, 40, 41, 48, 49, 56, 57, 64)),
)
NEGATIVE_ZERO = -32768

requires_sm100 = pytest.mark.skipif(
    not current_platform.is_cuda()
    or not current_platform.is_device_capability_family(100),
    reason="the specialized decode's boundary kernel requires SM100",
)


def _hyper_connection(use_combine: bool, seed: int):
    """A decoder layer's hyper-connection (`use_combine`) or the final
    mixer on the current GPU, with random weights of the checkpoint's scale,
    the same on every rank."""
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.models.qwen4_exp.common.hyperconnection import HyperConnectionConfig
    from vllm.models.qwen4_exp.nvidia.hyperconnection import GatedResidual

    config = HyperConnectionConfig(
        hc_count=STREAMS,
        hidden_size=HIDDEN,
        params_dtype=torch.bfloat16,
        hc_lowrank=LOW_RANK,
        rms_norm_eps=EPS,
        hc_per_branch_norm=True,
    )
    with set_current_vllm_config(VllmConfig()), torch.device("cuda"):
        hc = GatedResidual(config, use_combine=use_combine)
    generator = torch.Generator(device="cuda").manual_seed(seed)
    down = (
        hc.input_mix_weight_down_block_inject.weight
        if use_combine
        else hc.input_mix_weight_down.weight
    )
    for weight, scale in (
        (hc.hc_norm.weight, 0.1),
        (down, 0.02),
        (hc.input_mix_weight_up.weight, 0.05),
    ):
        weight.data.copy_(
            torch.randn(weight.shape, generator=generator, device="cuda") * scale
        )
    if use_combine:
        down.data[LOW_RANK + STREAMS :] = 0  # the merged weight's padding rows
    return hc


def _block(kind: str, rows: int, rank: int, device: torch.device) -> dict:
    """This rank's operands of one boundary of `kind`: its partial output
    (attention), or a MoE block's routed output (or GEMM2 rows, routing
    weights and slots under expert parallelism, about a quarter of each
    token's experts on this rank), shared expert output and gate logits."""
    generator = torch.Generator(device=device).manual_seed(
        1000 * rows + 10 * rank + KINDS.index(kind)
    )

    def randn(*shape: int, scale: float = 0.5) -> torch.Tensor:
        return (torch.randn(*shape, generator=generator, device=device) * scale).to(
            torch.bfloat16
        )

    if kind == "attention":
        partial = randn(rows, HIDDEN)
        # Packed +0/-0 values that collide with the Lamport sentinel.
        partial[:, :16] = 0
        partial[:, 9:16:2] = -0.0
        return {"partial": partial}
    block = {"shared": randn(rows, HIDDEN), "gate": randn(rows, 1, scale=1.0)}
    if kind == "moe":
        block["partial"] = randn(rows, HIDDEN)
        return block
    held = torch.rand(rows, TOP_K, generator=generator, device=device) < 0.25
    count = int(held.sum())
    slots = torch.full((rows, TOP_K), -1, dtype=torch.int32, device=device)
    slots[held] = torch.randperm(count, generator=generator, device=device).to(
        torch.int32
    )
    block["weights"] = (
        torch.rand(rows, TOP_K, generator=generator, device=device) * 0.2
    ).to(torch.bfloat16)
    block["slots"] = slots
    block["partial"] = randn(max(count, 1), HIDDEN)
    return block


def _boundary(boundary, kind: str, block: dict, streams, injection, norm, split):
    """The boundary of `kind` on `block`'s operands."""
    site = (streams, injection, norm, EPS, split)
    if kind == "attention":
        return boundary(block["partial"], *site)
    if kind == "moe":
        return boundary.moe(block["partial"], block["shared"], block["gate"], *site)
    return boundary.moe_folded(
        block["partial"],
        block["weights"],
        block["slots"],
        block["shared"],
        block["gate"],
        *site,
    )


def _project(hc, xn: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
    """`hc_project` of every row of `xn`, `READ_ROWS` rows a call (each
    row's projections depend on that row alone)."""
    from vllm.models.qwen4_exp.nvidia.specialized_decode.hc.read import (
        hc_project,
    )

    parts = [hc_project(hc, chunk) for chunk in xn.split(READ_ROWS)]
    if len(parts) == 1:
        return parts[0]
    block_input = torch.cat([part[0] for part in parts])
    if parts[0][1] is None:
        return block_input, None
    return block_input, torch.cat([part[1] for part in parts])


class Chain:
    """One chain of `KINDS` reads of `rows` rows on this rank: the
    operands (the starting streams and injection logits, the same on every
    rank, and this rank's blocks), replicated and split."""

    def __init__(self, rows: int, rank: int, boundary, split, hcs) -> None:
        self.rows = rows
        self.boundary = boundary
        self.split = split
        self.hcs = hcs
        device = hcs[0].hc_norm.weight.device
        generator = torch.Generator(device=device).manual_seed(rows)
        self.streams = torch.randn(rows, WIDTH, generator=generator, device=device).to(
            torch.bfloat16
        )
        # A column view of a down projection's output, as the reads return
        # the injection logits.
        down = torch.randn(rows, 336, generator=generator, device=device)
        self.injection = down.to(torch.bfloat16)[:, LOW_RANK : LOW_RANK + STREAMS]
        self.blocks = [_block(kind, rows, rank, device) for kind in KINDS]

    def replicated(self) -> list[tuple]:
        """Per read: the streams, `xn`, the block input and the injection
        logits, of every row."""
        streams, injection = self.streams, self.injection
        reads = []
        for kind, block, hc in zip(KINDS, self.blocks, self.hcs):
            streams, xn = _boundary(
                self.boundary, kind, block, streams, injection, hc.hc_norm.weight, False
            )
            block_input, injection = _project(hc, xn)
            reads.append((streams, xn, block_input, injection))
        return reads

    def split_reads(self, delay=None) -> tuple[list[tuple], torch.Tensor]:
        """Per read, as `replicated`: this rank's rows of the streams,
        `xn` and injection logits, and the block input of every row; then
        the final streams of every row (`RowSplit.streams`). `delay(k)`
        runs before the `k`-th exchange."""
        split = self.split
        streams = split.own(self.streams)
        injection = split.own(self.injection)
        reads = []
        for k, (kind, block, hc) in enumerate(zip(KINDS, self.blocks, self.hcs)):
            if delay is not None:
                delay(k)
            streams, xn = _boundary(
                self.boundary, kind, block, streams, injection, hc.hc_norm.weight, True
            )
            block_input, injection = split.project(hc, xn)
            reads.append((streams, xn, block_input, injection))
        if delay is not None:
            delay(len(KINDS))
        return reads, split.streams(streams)


def positive_zeros(x: torch.Tensor) -> torch.Tensor:
    """`x` with its negative zeros as positive zeros (the gathers' bits)."""
    bits = x.contiguous().view(torch.int16)
    return torch.where(bits == NEGATIVE_ZERO, 0, bits).view(torch.bfloat16)


def _bits(x: torch.Tensor) -> torch.Tensor:
    """The 16-bit patterns of `x`."""
    return x.contiguous().view(torch.int16)


def _check(chain: Chain, reads: list[tuple], streams: torch.Tensor) -> None:
    """The split chain's outputs against the replicated chain's, bit for bit."""
    own = chain.split.own
    replicated = chain.replicated()
    for k, (want, got) in enumerate(zip(replicated, reads)):
        streams_want, xn_want, block_input_want, injection_want = want
        streams_got, xn_got, block_input_got, injection_got = got
        assert torch.equal(_bits(streams_got), _bits(own(streams_want))), k
        assert torch.equal(_bits(xn_got), _bits(own(xn_want))), k
        assert torch.equal(
            _bits(block_input_got), _bits(positive_zeros(block_input_want))
        ), k
        if injection_want is None:
            assert injection_got is None
        else:
            assert torch.equal(_bits(injection_got), _bits(own(injection_want))), k
    final_streams = replicated[-1][0]
    assert torch.equal(_bits(streams), _bits(positive_zeros(final_streams)))


def _build(rank: int, device: torch.device):
    """The boundary kernel with its split calls, the `RowSplit` of the TP
    group, and a layer's and the final mixer's hyper-connections; warmed up
    (collective)."""
    from vllm.distributed import get_tp_group
    from vllm.models.qwen4_exp.nvidia.specialized_decode.boundary import (
        SublayerBoundary,
    )
    from vllm.models.qwen4_exp.nvidia.specialized_decode.hc.read import (
        warm_up_hc_read,
    )
    from vllm.models.qwen4_exp.nvidia.specialized_decode.runtime.split import (
        RowSplit,
    )

    group = get_tp_group().device_group
    boundary = SublayerBoundary(group, MAX_ROWS, device, split=True)
    split = RowSplit(group, MAX_ROWS, 0, device)
    layer_hc = _hyper_connection(True, seed=0)
    final_hc = _hyper_connection(False, seed=1)
    warm_up_hc_read(layer_hc, final_hc, READ_ROWS)
    boundary.warm_up(
        layer_hc.hc_norm.weight, EPS, attention=True, moe=True, folded=True
    )
    split.warm_up()
    return boundary, split, (layer_hc, layer_hc, final_hc)


@ray.remote(num_gpus=1, max_calls=1)
def _tp_split(monkeypatch, tp_size, pp_size, rank, distributed_init_port):
    from vllm.distributed import get_tp_group

    with monkeypatch.context() as m:
        m.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        device = torch.device(f"cuda:{rank}")
        torch.accelerator.set_device_index(device)
        init_test_distributed_environment(tp_size, pp_size, rank, distributed_init_port)
        ensure_model_parallel_initialized(tp_size, pp_size)
        group = get_tp_group().device_group
        boundary, split, hcs = _build(rank, device)
        for rows in CHECK_ROWS:
            chain = Chain(rows, rank, boundary, split, hcs)
            reads, streams = chain.split_reads()
            _check(chain, reads, streams)
            # Every rank holds the same block inputs and final streams.
            for output in [read[2] for read in reads] + [streams]:
                gathered = [torch.empty_like(output) for _ in range(tp_size)]
                torch.distributed.all_gather(gathered, output, group=group)
                for other in gathered:
                    assert torch.equal(_bits(other), _bits(output))
            # Captured, replayed many times (the generations rotate), then
            # with changed inputs.
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                reads, streams = chain.split_reads()
            for _ in range(25):
                graph.replay()
            torch.accelerator.synchronize()
            _check(chain, reads, streams)
            for block in chain.blocks:
                block["partial"].mul_(0.5)
            chain.streams.mul_(2)
            graph.replay()
            torch.accelerator.synchronize()
            _check(chain, reads, streams)
        _stress(rank, REPLAYS, (boundary, split, hcs))


def _stress_graphs(stress: ReplayStress, rank: int, built) -> None:
    """Add the split chains' graphs to `stress`: each replay scales every
    block's operands by a factor that the counter sets (so a word left from
    another replay or chain differs), runs the replicated chain, then the split
    chain with this rank delayed as the graph's skew says, and counts the split
    outputs' words that differ from the replicated ones."""
    boundary, split, hcs = built
    for index, rows in enumerate(STRESS_ROWS):
        chain = Chain(rows, rank, boundary, split, hcs)
        bases = [
            {
                name: block[name].clone()
                for name in ("partial", "shared")
                if name in block
            }
            for block in chain.blocks
        ]
        owned = rows // TP
        words = len(KINDS) * (2 * owned * WIDTH + rows * HIDDEN) // 2
        words += rows * WIDTH // 2
        for skew in SKEWS:

            def delay(k: int, skew=skew, index=index) -> None:
                stress.delay(skew, index, k)

            def body(chain=chain, bases=bases, delay=delay) -> None:
                stress.counter.add_(1)
                for k, (block, base) in enumerate(zip(chain.blocks, bases)):
                    # A factor of 1 to 61 that changes every replay and call.
                    factor = (stress.counter * len(KINDS) + k) % 61 + 1
                    for name, value in base.items():
                        block[name].copy_(value * factor.to(value.dtype))
                want = chain.replicated()
                reads, streams = chain.split_reads(delay)
                own = chain.split.own
                for (s_want, xn_want, bi_want, _), (s_got, xn_got, bi_got, _) in zip(
                    want, reads
                ):
                    stress.count_words(s_got, own(s_want))
                    stress.count_words(xn_got, own(xn_want))
                    stress.count_words(bi_got, positive_zeros(bi_want))
                stress.count_words(streams, positive_zeros(want[-1][0]))

            stress.graphs.append(StressGraph(body, len(KINDS) + 1, words, 0, chain))


def _stress(rank: int, replays: int, built=None, log: bool = False) -> None:
    """The race stress of the split reads on this rank of the TP4 group;
    every rank must call it."""
    device = torch.device("cuda", rank)
    if built is None:
        built = _build(rank, device)
    stress = ReplayStress(device)
    _stress_graphs(stress, rank, built)
    result = report("split reads", stress.run(replays, log=log))
    assert result.bad_words == 0 and result.eager_bad == 0, result


@requires_sm100
def test_tp4_split_reads_match_replicated(monkeypatch):
    if torch.accelerator.device_count() < 4:
        pytest.skip("Requires four GPUs with NVLink multicast")
    multi_process_parallel(monkeypatch, 4, 1, _tp_split)


# The stall test's exchanges, kept until the worker process exits (one task
# per worker, ``max_calls=1``): freeing a symmetric-memory mailbox after the
# kernel trapped aborts the process.
_STALLED: list[object] = []
EXCHANGES = ("boundary", "up/mix gather", "rows gather")


def _tp_stall(monkeypatch, tp_size, pp_size, rank, distributed_init_port, exchange):
    """One rank of the stall test of `exchange` (`EXCHANGES`)."""
    from vllm.distributed import get_tp_group
    from vllm.models.qwen4_exp.nvidia.specialized_decode.boundary import (
        SublayerBoundary,
    )
    from vllm.models.qwen4_exp.nvidia.specialized_decode.hc.up_mix import (
        HcUpMixGather,
    )
    from vllm.models.qwen4_exp.nvidia.specialized_decode.logits.gather import (
        RowsAllGather,
    )

    with monkeypatch.context() as m:
        m.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        device = torch.device(f"cuda:{rank}")
        torch.accelerator.set_device_index(device)
        init_test_distributed_environment(tp_size, pp_size, rank, distributed_init_port)
        ensure_model_parallel_initialized(tp_size, pp_size)
        group = get_tp_group()
        zeros = torch.zeros(1, WIDTH, dtype=torch.bfloat16, device=device)
        kernel: SublayerBoundary | HcUpMixGather | RowsAllGather
        if exchange == "boundary":
            boundary = SublayerBoundary(
                group.device_group, MAX_ROWS, device, stall_timeout=1.0, split=True
            )
            partial = torch.zeros(tp_size, HIDDEN, dtype=torch.bfloat16, device=device)
            injection = torch.zeros(1, STREAMS, dtype=torch.bfloat16, device=device)
            norm = torch.zeros(WIDTH, dtype=torch.bfloat16, device=device)

            def call() -> None:
                boundary(partial, zeros, injection, norm, EPS, split=True)

            kernel = boundary
            waited_for = "the data of"
        elif exchange == "up/mix gather":
            gather = HcUpMixGather(group.device_group, 16, device, stall_timeout=1.0)
            low = torch.zeros(1, LOW_RANK, dtype=torch.bfloat16, device=device)
            up = torch.zeros(WIDTH, LOW_RANK, dtype=torch.bfloat16, device=device)

            def call() -> None:
                gather(low, zeros, up)

            kernel = gather
            waited_for = "the block input of"
        else:
            rows = RowsAllGather(
                group.device_group, WIDTH, 16, device, stall_timeout=1.0
            )

            def call() -> None:
                rows(zeros)

            kernel = rows
            waited_for = "the rows of"
        _STALLED.append(kernel)
        skipper = tp_size - 1
        if rank != skipper:
            # The last rank never calls: the others' polls time out, and their
            # kernel traps (the CUDA context is unusable afterwards).
            with pytest.raises(RuntimeError):
                call()
                torch.accelerator.synchronize(device)
            stall = kernel.stall_report()
            assert stall is not None
            assert stall.startswith(f"rank {rank} timed out in generation 0"), stall
            assert f"{waited_for} rank(s) [{skipper}]" in stall, stall
        else:
            assert kernel.stall_report() is None
        # The skipping rank keeps its mailbox mapped until the others are done.
        torch.distributed.barrier(group=group.cpu_group)


def _stall_task(exchange: str):
    """The Ray task of a rank of the stall test of `exchange`."""

    def task(monkeypatch, tp_size, pp_size, rank, distributed_init_port):
        _tp_stall(monkeypatch, tp_size, pp_size, rank, distributed_init_port, exchange)

    return ray.remote(num_gpus=1, max_calls=1)(task)


@requires_sm100
@pytest.mark.parametrize("exchange", EXCHANGES)
def test_tp4_split_exchanges_report_a_stall(monkeypatch, exchange):
    if torch.accelerator.device_count() < 4:
        pytest.skip("Requires four GPUs with NVLink multicast")
    multi_process_parallel(monkeypatch, 4, 1, _stall_task(exchange))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Race stress of the row-split reads on four GPUs (replay_stress.py)"
    )
    parser.add_argument("--replays", type=int, default=REPLAYS)
    args = parser.parse_args()
    spawn(_stress, args.replays, None, True)
