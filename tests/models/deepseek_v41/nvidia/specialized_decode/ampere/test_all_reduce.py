# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The SM80 boundaries' two-shot Lamport all-reduce (`AmpereAllReduce`),
alone and fused with the mHC boundary, at TP8.

Each check runs one rank late, so a missing wait reads a peer's stale data:

* calls of 1 to 128 rows in an order that cycles every row's generations,
  every other call on five CTAs (several rows each): within one BF16 ulp of
  the FP32 sum of the partials (which include +0/-0 pairs that look like
  the Lamport sentinel), and bitwise equal on every rank;
* the all-reduce fused with a boundary's post-mix, collapse and RMSNorm
  (`post_collapse`), with and without the pre-mix and the stream mean,
  every other call on five CTAs: bitwise the all-reduce followed by
  `mhc_post_collapse`;
* the same with the MoE's top-k reduction open (`finalize_post_collapse`,
  six routes a token): bitwise `post_collapse` of the partial row formed
  as the kernel forms it (an FP32 fused multiply-add per route in route
  order, then the shared row, rounded once to BF16);
* every mailbox slot back to the sentinel after the calls, and every row's
  generation in 0..2;
* one CUDA graph of two calls (24 and 48 rows), a fused one and a
  finalizing one (6 rows each), replayed with new inputs;
* the boundary dispatch: the kernel up to its capacity, the TP group's
  all-reduce above it.

Needs eight SM80 GPUs with peer access; under a minute. For a stress run, run
this file with `python -m` as the module
`tests.models.deepseek_v41.nvidia.specialized_decode.ampere.test_all_reduce`, with
`--replays 2000000`: CUDA graphs of three all-reduces of different row
counts each, some with the ranks skewed, replayed in one seeded random order
with new inputs every replay; a device-side count of the elements that differ
from the expected sum must stay 0. Rank r feeds `(t * (2r + 3)) % 61 - 30`
with `t` the element index plus 7 times the replay plus 13 times the call,
mod 61: every sum is an integer of magnitude at most 240, so exact in BF16,
and the 61 residues give 61 different sums. An all-reduce that read an
earlier call's data, wrote a sum to another offset or left an element
unwritten would show up as a mismatch.
"""

import argparse
import os
import random
import tempfile
from functools import partial
from types import SimpleNamespace

import pytest
import torch

from vllm.platforms import current_platform

TP = 8
HIDDEN = 5120
TOP_K = 6
EPS = 1e-6
LATE_CYCLES = 20_000
REPLAYS = 50
# The stress run's graphs: three all-reduces of these rows each.
STRESS_ROWS = ((6, 96, 24), (128, 1, 48), (12, 128, 6))


def _late(rank: int, step: int) -> None:
    """Delay one rank per step, a different one each step."""
    if rank == step % TP:
        torch.cuda._sleep(LATE_CYCLES)


def _partial(rows: int, rank: int, seed: int) -> torch.Tensor:
    """This rank's partial sum, with +0/-0 pairs that look like the Lamport
    sentinel."""
    generator = torch.Generator(device="cuda").manual_seed(97 * rank + 1009 * seed)
    x = torch.randn(rows, HIDDEN, generator=generator, device="cuda").bfloat16()
    x[:, :16] = 0
    x[:, 9:16:2] = -0.0
    return x


def _assert_reduced(out: torch.Tensor, x: torch.Tensor) -> None:
    """`out` is within one BF16 ulp of the FP32 sum of every rank's `x`
    and bitwise the same on every rank."""
    from vllm.distributed import get_tp_group

    group = get_tp_group()
    rows = x.shape[0]
    partials = group.all_gather(x, dim=0).view(TP, rows, HIDDEN)
    want = partials.float().sum(0)
    ulp = torch.maximum(want.abs(), torch.full_like(want, 2**-126)) * 2**-7
    assert ((out.float() - want).abs() <= ulp).all(), rows
    outs = group.all_gather(out, dim=0).view(TP, rows, HIDDEN)
    assert torch.equal(outs.view(torch.int16), out.expand(TP, -1, -1).view(torch.int16))


def _moe(
    rows: int, rank: int, seed: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """This rank's unweighted route rows, routing weights and shared row."""
    generator = torch.Generator(device="cuda").manual_seed(89 * rank + 1013 * seed)
    routes = _partial(rows * TOP_K, rank, seed)
    weights = torch.rand(rows, TOP_K, generator=generator, device="cuda")
    return routes, weights, _partial(rows, rank, seed + 7)


def _moe_partial(routes, weights, shared) -> torch.Tensor:
    """The partial row as the finalize kernel forms it: an FP32 fused
    multiply-add per route (the FP64 product of a BF16 and an FP32 value is
    exact), the shared row added in FP32, rounded to BF16."""
    v = routes.view(-1, TOP_K, HIDDEN).double()
    w = weights.double()
    acc = (v[:, 0] * w[:, :1]).float()
    for k in range(1, TOP_K):
        acc = (v[:, k] * w[:, k : k + 1] + acc.double()).float()
    return (acc + shared.float()).bfloat16()


def _boundary(
    rows: int, seed: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """A boundary's streams, post-, comb- and pre-mix and norm weight, the
    same on every rank."""
    generator = torch.Generator(device="cuda").manual_seed(seed)
    streams = torch.randn(rows, 4, HIDDEN, generator=generator, device="cuda")
    post = torch.rand(rows, 4, 1, generator=generator, device="cuda") * 2
    comb = torch.rand(rows, 4, 4, generator=generator, device="cuda")
    pre = torch.rand(rows, 4, generator=generator, device="cuda") + 1e-6
    weight = torch.randn(HIDDEN, generator=generator, device="cuda")
    return streams.bfloat16(), post, comb, pre, weight.bfloat16()


def _assert_post_collapsed(reduce, x, boundary, pre, write_aux, got) -> None:
    """`got` is bitwise the all-reduce of `x` followed by
    `mhc_post_collapse`."""
    from vllm.models.deepseek_v41.nvidia.specialized_decode.ampere.mhc import (
        mhc_post_collapse,
    )

    streams, post, comb, _, weight = boundary
    want = mhc_post_collapse(
        reduce(x), streams, post.view(-1, 4), comb, pre, weight, EPS, write_aux
    )
    for out, expected in zip(got, want):
        assert torch.equal(out.view(torch.int16), expected.view(torch.int16))


def _check(rank: int) -> None:
    from vllm.distributed import get_tp_group
    from vllm.models.deepseek_v41.nvidia.ops import mhc

    reduce = mhc._ampere_all_reduce
    assert reduce is not None
    step = 0
    ctas = reduce.ctas
    for seed, rows in enumerate([1, 6, 24, 96, 128, 7, 24, 96, 1, 128] * 2):
        reduce.ctas = 5 if seed % 2 else ctas
        x = _partial(rows, rank, seed)
        _late(rank, step := step + 1)
        _assert_reduced(reduce(x), x)
    for seed, rows in enumerate([1, 6, 96, 128, 7, 24]):
        x = _partial(rows, rank, 400 + seed)
        boundary = _boundary(rows, seed)
        streams, post, comb, pre, weight = boundary
        pre = None if seed % 3 == 2 else pre
        write_aux = seed % 2 == 1
        reduce.ctas = 5 if seed % 2 else ctas
        _late(rank, step := step + 1)
        got = reduce.post_collapse(x, streams, post, comb, pre, weight, EPS, write_aux)
        _assert_post_collapsed(reduce, x, boundary, pre, write_aux, got)
    for seed, rows in enumerate([1, 6, 96, 128, 7, 24]):
        moe = _moe(rows, rank, 500 + seed)
        boundary = _boundary(rows, 500 + seed)
        streams, post, comb, pre, weight = boundary
        pre = None if seed % 3 == 2 else pre
        write_aux = seed % 2 == 1
        reduce.ctas = 5 if seed % 2 else ctas
        _late(rank, step := step + 1)
        got = reduce.finalize_post_collapse(
            *moe, streams, post, comb, pre, weight, EPS, write_aux
        )
        want = reduce.post_collapse(
            _moe_partial(*moe), streams, post, comb, pre, weight, EPS, write_aux
        )
        for out, expected in zip(got, want):
            assert torch.equal(out.view(torch.int16), expected.view(torch.int16))
    reduce.ctas = ctas
    torch.accelerator.synchronize()
    torch.distributed.barrier(get_tp_group().cpu_group)
    assert (reduce.mailbox.view(torch.int16) == -32768).all()
    assert int(reduce.generations.min()) >= 0 and int(reduce.generations.max()) < 3

    # One graph of two calls, a fused and a finalizing one, replayed with
    # new inputs.
    xa = _partial(24, rank, 100)
    xb = _partial(48, rank, 101)
    xc = _partial(6, rank, 102)
    moe = _moe(6, rank, 103)
    boundary = _boundary(6, 102)
    fused = partial(reduce.post_collapse, xc, *boundary, EPS, False)
    finalize = partial(reduce.finalize_post_collapse, *moe, *boundary, EPS, False)
    reduce(xa), reduce(xb), fused(), finalize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out_a = reduce(xa)
        out_b = reduce(xb)
        out_c = fused()
        out_d = finalize()
    for replay in range(REPLAYS):
        xa.copy_(_partial(24, rank, 200 + replay))
        xb.copy_(_partial(48, rank, 300 + replay))
        xc.copy_(_partial(6, rank, 400 + replay))
        for tensor, new in zip(moe, _moe(6, rank, 600 + replay)):
            tensor.copy_(new)
        _late(rank, step := step + 1)
        graph.replay()
        _assert_reduced(out_a, xa)
        _assert_reduced(out_b, xb)
        _assert_post_collapsed(reduce, xc, boundary, boundary[3], False, out_c)
        want = reduce.post_collapse(_moe_partial(*moe), *boundary, EPS, False)
        for out, expected in zip(out_d[:2], want[:2]):
            assert torch.equal(out.view(torch.int16), expected.view(torch.int16))

    # The boundary dispatch falls back above the kernel's capacity.
    calls: list[str] = []
    group = get_tp_group()
    original = group.all_reduce

    def all_reduce(x):
        calls.append("group")
        return original(x)

    group.all_reduce = all_reduce  # type: ignore[method-assign]
    try:
        for rows in (reduce.capacity, reduce.capacity + 1):
            x = _partial(rows, rank, rows)
            _assert_reduced(mhc.boundary_all_reduce(x), x)
    finally:
        group.all_reduce = original  # type: ignore[method-assign]
    assert calls == ["group"]


def _stress_value(t: torch.Tensor, rank: int) -> torch.Tensor:
    """Rank `rank`'s stress input for residues `t` (0..60)."""
    return (t * (2 * rank + 3)) % 61 - 30


def _stress_graph(reduce, rank, sizes, counter, errors, skew):
    """A CUDA graph of three all-reduces whose inputs follow the replay
    counter on the device: small integers, so every sum is exact. It adds the
    output elements that differ from the expected sum to `errors`."""
    index = [
        torch.arange(rows * HIDDEN, dtype=torch.int32, device="cuda").view(rows, -1)
        for rows in sizes
    ]
    inputs = [
        torch.empty(rows, HIDDEN, dtype=torch.bfloat16, device="cuda") for rows in sizes
    ]

    def body():
        counter.add_(1)
        for kind, (idx, x) in enumerate(zip(index, inputs)):
            t = (idx + 7 * (counter % 61) + 13 * kind) % 61
            x.copy_(_stress_value(t, rank))
            if skew:
                torch.cuda._sleep(LATE_CYCLES * ((rank + kind) % TP))
            out = reduce(x)
            want = sum(_stress_value(t, r) for r in range(TP))
            errors.add_((out.float() != want).sum())

    body()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        body()
    # The graph keeps raw pointers: keep its tensors alive too.
    return graph, (index, inputs)


def _check_replays(rank: int, replays: int) -> tuple[int, int]:
    """The stress graphs replayed in one seeded random order on every rank;
    returns the replays run and the mismatched elements summed over the
    ranks, which all stop after the first block of 1000 replays with a
    mismatch."""
    from vllm.distributed import get_tp_group
    from vllm.models.deepseek_v41.nvidia.ops import mhc

    reduce = mhc._ampere_all_reduce
    assert reduce is not None
    counter = torch.zeros(1, dtype=torch.int32, device="cuda")
    errors = torch.zeros(1, dtype=torch.int64, device="cuda")
    graphs = [
        _stress_graph(reduce, rank, sizes, counter, errors, skew)
        for sizes in STRESS_ROWS
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


def _worker(rank: int, port: str, skip_path: str, replays: int) -> None:
    from tests.utils import init_test_distributed_environment
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed import cleanup_dist_env_and_memory, get_tp_group
    from vllm.models.deepseek_v41.nvidia.ops import mhc

    torch.accelerator.set_device_index(rank)
    with set_current_vllm_config(VllmConfig()), torch.inference_mode():
        init_test_distributed_environment(TP, 1, rank, port, local_rank=rank)
        config = SimpleNamespace(
            parallel_config=SimpleNamespace(
                tensor_parallel_size=TP, enable_expert_parallel=False
            ),
            model_config=SimpleNamespace(
                hf_config=SimpleNamespace(hidden_size=HIDDEN, num_experts_per_tok=TOP_K)
            ),
        )
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
        mhc.init_ampere_all_reduce(config)
        if replays:
            done, errors = _check_replays(rank, replays)
            if rank == 0:
                print(
                    f"{done} replays ({3 * done} all-reduces) on {TP} "
                    f"ranks: {errors} mismatched elements"
                )
            assert errors == 0
        else:
            _check(rank)
        cleanup_dist_env_and_memory()


def _spawn(replays: int) -> str | None:
    """Run the checks (or the stress replays) on eight GPUs; returns why they
    were skipped, if they were."""
    if torch.accelerator.device_count() < TP:
        return "needs eight GPUs"
    from vllm.utils.network_utils import get_open_port

    with tempfile.TemporaryDirectory() as tmp:
        skip_path = os.path.join(tmp, "skipped")
        torch.multiprocessing.spawn(
            _worker, args=(str(get_open_port()), skip_path, replays), nprocs=TP
        )
        if os.path.exists(skip_path):
            with open(skip_path) as f:
                return f.read()
    return None


@pytest.mark.distributed(num_gpus=8)
@pytest.mark.skipif(not current_platform.is_device_capability(80), reason="needs SM80")
def test_ampere_all_reduce():
    skipped = _spawn(0)
    if skipped:
        pytest.skip(skipped)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--replays", type=int, default=2_000_000)
    skipped = _spawn(parser.parse_args().replays)
    if skipped:
        print(f"skipped: {skipped}")
