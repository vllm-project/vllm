# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The sublayer boundary of Qwen4Exp's specialized decode (`SublayerBoundary`,
a CuTe DSL kernel) against the stock kernels: the all-reduce (a rank-order FP32
sum, the order of the stock MNNVL kernel), the shared expert's gate and the
routed add, and the Triton combine-norm; eagerly and under CUDA graph replay, at
TP 1 and TP 4. At TP 4, a rank that skips a call makes the others report the
stall and raise instead of hanging, and the race stress of `replay_stress.py`
replays graphs of back-to-back boundaries of every kind (attention, MoE, MoE
with the folded finalize) at the decode's row counts, some with ranks delayed,
in a random order, and finds no stale word. Run as a module on four GPUs for a
long stress run (`--replays`) or the control on vLLM's custom collectives
(`--control`)."""

import pytest
import ray
import torch

from vllm.platforms import current_platform

from ....utils import (
    ensure_model_parallel_initialized,
    init_test_distributed_environment,
    multi_process_parallel,
)
from .replay_stress import (
    CONTROLS,
    REPLAYS,
    SKEWS,
    TP,
    ReplayStress,
    StressGraph,
    custom_comm,
    report,
    spawn,
)

STREAMS = 4
HIDDEN = 2560
WIDTH = STREAMS * HIDDEN
EPS = 1e-6
MAX_ROWS = 256
# The race stress: three boundaries a graph, every row count of the decode's
# TP4 steps of up to 16 requests (4 to 64) at least once and some larger ones
# (up to 64 requests), the calls rotating through the kinds.
STRESS_ROWS = (
    (4, 64, 16),
    (64, 8, 32),
    (12, 48, 4),
    (40, 60, 28),
    (20, 36, 56),
    (44, 24, 52),
    (256, 4, 128),
    (96, 192, 160),
)
KINDS = ("attention", "moe", "moe folded")
# The folded calls' experts of each token, in two finalize groups; the
# others are another rank's (slot -1).
FOLD_EXPERTS = (0, 4, 7)

requires_sm100 = pytest.mark.skipif(
    not current_platform.is_cuda()
    or not current_platform.is_device_capability_family(100),
    reason="the specialized decode's boundary kernel requires SM100",
)


def _site(rows: int, rank: int, device: torch.device) -> dict[str, torch.Tensor]:
    """One boundary's operands: streams, injection logits (a column view of
    the down projection's output, as the read returns them), norm weight and
    shared-gate logits equal on every rank; partials per rank, with packed
    +0/-0 values that collide with the Lamport sentinel."""
    common = torch.Generator(device=device).manual_seed(rows)
    local = torch.Generator(device=device).manual_seed(1000 * rows + rank)

    def randn(generator: torch.Generator, *shape: int, scale: float = 1.0):
        return (torch.randn(*shape, generator=generator, device=device) * scale).to(
            torch.bfloat16
        )

    down_output = randn(common, rows, 336)
    partial = randn(local, rows, HIDDEN)
    partial[:, :16] = 0
    partial[:, 9:16:2] = -0.0
    return {
        "streams": randn(common, rows, WIDTH),
        "injection": down_output[:, 320:324],
        "norm": randn(common, WIDTH, scale=0.1),
        "logit": randn(common, rows, 1),
        "partial": partial,
        "shared": randn(local, rows, HIDDEN),
    }


def _contribution(site: dict[str, torch.Tensor], moe: bool) -> torch.Tensor:
    """This rank's partial block output as the stock decode forms it."""
    if not moe:
        return site["partial"]
    return torch.sigmoid(site["logit"]) * site["shared"] + site["partial"]


def _boundary(boundary, site: dict[str, torch.Tensor], moe: bool):
    """The fused boundary of `site`."""
    args = (site["streams"], site["injection"], site["norm"], EPS)
    if moe:
        return boundary.moe(site["partial"], site["shared"], site["logit"], *args)
    return boundary(site["partial"], *args)


def _check(fused, block_output: torch.Tensor, site: dict[str, torch.Tensor]):
    """Streams bitwise and `xn` within one BF16 ulp of the stock
    combine-norm on the reduced block output."""
    from vllm.models.qwen4_exp.nvidia.ops.hc import hc_combine_norm

    streams, xn = hc_combine_norm(
        site["streams"], block_output, site["injection"], site["norm"], EPS, STREAMS
    )
    torch.testing.assert_close(fused[0], streams, rtol=0, atol=0)
    ulps = (fused[1].view(torch.int16).int() - xn.view(torch.int16).int()).abs()
    assert int(ulps.max()) <= 1


def _check_graph(boundary, site: dict[str, torch.Tensor], moe: bool, reduce):
    """Captured calls replay to the same bits, also after many replays (the
    mailbox generations rotate) and with changed inputs."""
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(4):
            _boundary(boundary, site, moe)
        captured = _boundary(boundary, site, moe)
    for _ in range(25):
        graph.replay()
    torch.accelerator.synchronize()
    _check(captured, reduce(_contribution(site, moe)), site)
    site["partial"].mul_(0.5)
    site["streams"].mul_(2)
    graph.replay()
    torch.accelerator.synchronize()
    _check(captured, reduce(_contribution(site, moe)), site)


@requires_sm100
@pytest.mark.parametrize("early_trigger", [False, True])
@pytest.mark.parametrize("moe", [False, True])
@pytest.mark.parametrize("rows", [1, 4, 33, 128, 256])
def test_local_boundary_matches_stock(rows: int, moe: bool, early_trigger: bool):
    from vllm.models.qwen4_exp.nvidia.specialized_decode.boundary import (
        SublayerBoundary,
    )

    device = torch.device("cuda")
    boundary = SublayerBoundary(None, 256, device, early_trigger)
    site = _site(rows, 0, device)
    _check(_boundary(boundary, site, moe), _contribution(site, moe), site)
    _check_graph(boundary, site, moe, lambda x: x)


@ray.remote(num_gpus=1, max_calls=1)
def _tp_boundary(monkeypatch, tp_size, pp_size, rank, distributed_init_port):
    from vllm.distributed import get_tp_group
    from vllm.models.qwen4_exp.nvidia.specialized_decode.boundary import (
        SublayerBoundary,
    )

    with monkeypatch.context() as m:
        m.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        device = torch.device(f"cuda:{rank}")
        torch.accelerator.set_device_index(device)
        init_test_distributed_environment(tp_size, pp_size, rank, distributed_init_port)
        ensure_model_parallel_initialized(tp_size, pp_size)
        group = get_tp_group().device_group
        boundary = SublayerBoundary(group, MAX_ROWS, device)

        def reduce(contribution: torch.Tensor) -> torch.Tensor:
            parts = [torch.empty_like(contribution) for _ in range(tp_size)]
            torch.distributed.all_gather(parts, contribution, group=group)
            total = torch.zeros_like(contribution, dtype=torch.float32)
            for part in parts:
                total += part.float()
            return total.to(torch.bfloat16)

        for rows in (1, 4, 16, 64, 128, MAX_ROWS):
            for moe in (False, True):
                site = _site(rows, rank, device)
                fused = _boundary(boundary, site, moe)
                _check(fused, reduce(_contribution(site, moe)), site)
                # Every rank holds the same bits.
                for output in fused:
                    gathered = [torch.empty_like(output) for _ in range(tp_size)]
                    torch.distributed.all_gather(gathered, output, group=group)
                    for other in gathered:
                        assert torch.equal(other, output)
                _check_graph(boundary, site, moe, reduce)
        _stress(rank, REPLAYS, None)


def _stress_call(rows: int, kind: str, device: torch.device) -> dict:
    """The operands of one stress call of `rows` tokens.

    Element `j` of rank `r`'s partial in call `t` (the replay counter
    times three plus the call) is `((7 j + 13 r + 5 t) mod 61) - 30` times
    `2^(((t + i) mod 16) + q)`, for the element's row `i` and `q` 1 in
    the second and fourth fifth of the row, else 0. Powers of two keep every
    value exact: the reduced block output is the exact sum, a BF16 value, and
    a stale contribution of any of the 60 calls before changes it. The MoE
    kinds split it into a shared expert output, gated by 1
    (`bf16(sigmoid(16))`), and a routed output, the folded one into three
    GEMM2 rows of weight 1; the streams and injection logits are zero, so
    every stream of the new streams is the block output.

    The boundary normalizes each stream by its sum of squares, which the five
    CTAs of a token's cluster add from their shares, a fifth of the row each,
    through distributed shared memory. The scale sets each share's weight in
    that sum (1 or 4 relative to the others), so a share that is missing or
    left from another call or row moves `xn` by several BF16 units in the
    last place, unless it had the same scale (a call a multiple of 16 calls
    away with the row, for one)."""
    from vllm.models.qwen4_exp.nvidia.specialized_decode.boundary.all_reduce_hc import (  # noqa: E501
        CLUSTER,
        TOP_K,
    )

    j = torch.arange(rows * HIDDEN, dtype=torch.int64, device=device)
    ranks = torch.arange(TP, dtype=torch.int64, device=device)[:, None]
    shared = j % 7 - 3
    first, second = j % 5 - 2, j % 3 - 1
    weights = torch.zeros(rows, TOP_K, dtype=torch.bfloat16, device=device)
    slots = torch.full((rows, TOP_K), -1, dtype=torch.int32, device=device)
    for i, expert in enumerate(FOLD_EXPERTS):
        weights[:, expert] = 1.0
        slots[:, expert] = torch.arange(rows, device=device) + i * rows
    return {
        "rows": rows,
        "kind": kind,
        "base": (7 * j + 13 * ranks) % 61,
        "row": j // HIDDEN,
        "odd_share": (j % HIDDEN) // (HIDDEN // CLUSTER) % 2,
        "shared_int": shared,
        "first_int": first,
        "second_int": second,
        "routed_offset": shared + first + second,
        "shared": torch.empty(rows, HIDDEN, dtype=torch.bfloat16, device=device),
        "gate": torch.full((rows, 1), 16.0, dtype=torch.bfloat16, device=device),
        "partial": torch.empty(rows, HIDDEN, dtype=torch.bfloat16, device=device),
        "expert_rows": torch.empty(
            3 * rows, HIDDEN, dtype=torch.bfloat16, device=device
        ),
        "weights": weights,
        "slots": slots,
        "streams": torch.zeros(rows, WIDTH, dtype=torch.bfloat16, device=device),
        "injection": torch.zeros(rows, STREAMS, dtype=torch.bfloat16, device=device),
        "want": torch.empty(rows, HIDDEN, dtype=torch.bfloat16, device=device),
        "xn_want": torch.empty(rows, HIDDEN, dtype=torch.bfloat16, device=device),
    }


def _stress_graphs(stress: ReplayStress, rank: int, control: str | None) -> None:
    """Add the boundary graphs (or, with `control`, the same graphs on
    vLLM's custom all-reduce or all-gather) to `stress`."""
    from vllm.distributed import get_tp_group
    from vllm.models.qwen4_exp.nvidia.specialized_decode.boundary import (
        SublayerBoundary,
    )

    device = stress.device
    norm = torch.zeros(WIDTH, dtype=torch.bfloat16, device=device)
    if control is None:
        boundary = SublayerBoundary(get_tp_group().device_group, MAX_ROWS, device)
        boundary.warm_up(norm, EPS, attention=True, moe=True, folded=True)
    else:
        ca_comm = custom_comm(control)

    def exchange(call: dict) -> tuple[torch.Tensor, torch.Tensor | None]:
        """The call's block output (every stream of the new streams, or the
        control's sum) and the boundary's `xn` (None for the control)."""
        rows, kind = call["rows"], call["kind"]
        site = (call["streams"], call["injection"], norm, EPS)
        if control == "all-reduce":
            return ca_comm.custom_all_reduce(call["partial"]), None
        if control == "all-gather":
            gathered = ca_comm.custom_all_gather(call["partial"])
            return gathered.view(TP, rows, HIDDEN).float().sum(0).bfloat16(), None
        if kind == "attention":
            out, xn = boundary(call["partial"], *site)
        elif kind == "moe":
            out, xn = boundary.moe(call["partial"], call["shared"], call["gate"], *site)
        else:
            out, xn = boundary.moe_folded(
                call["expert_rows"],
                call["weights"],
                call["slots"],
                call["shared"],
                call["gate"],
                *site,
            )
        return out.view(rows, STREAMS, HIDDEN), xn.view(rows, STREAMS, HIDDEN)

    for index, row_counts in enumerate(STRESS_ROWS):
        calls = [
            _stress_call(rows, KINDS[(index + k) % len(KINDS)], device)
            for k, rows in enumerate(row_counts)
        ]
        for skew in SKEWS:

            def body(calls=calls, skew=skew, index=index) -> None:
                stress.counter.add_(1)
                for k, call in enumerate(calls):
                    rows = call["rows"]
                    t = stress.counter * len(calls) + k
                    scale = 2 ** ((t + call["row"]) % 16 + call["odd_share"])
                    values = ((call["base"] + 5 * t) % 61 - 30) * scale
                    call["want"].copy_(values.sum(0).view(rows, HIDDEN))
                    if control is None:
                        # Every stream of xn: the block output normalized
                        # (the norm weight is zero).
                        y = call["want"].float()
                        norm_scale = torch.rsqrt(y.square().mean(-1, True) + EPS)
                        call["xn_want"].copy_(y * norm_scale)
                    mine = values[rank]
                    if control is not None or call["kind"] == "attention":
                        call["partial"].copy_(mine.view(rows, HIDDEN))
                        continue
                    shared = call["shared_int"] * scale
                    call["shared"].copy_(shared.view(rows, HIDDEN))
                    if call["kind"] == "moe":
                        call["partial"].copy_((mine - shared).view(rows, HIDDEN))
                        continue
                    routed = mine - call["routed_offset"] * scale
                    expert_rows = call["expert_rows"].view(3, rows * HIDDEN)
                    expert_rows[0].copy_(routed)
                    expert_rows[1].copy_(call["first_int"] * scale)
                    expert_rows[2].copy_(call["second_int"] * scale)
                outs = []
                for k, call in enumerate(calls):
                    stress.delay(skew, index, k)
                    outs.append(exchange(call))
                for (out, xn), call in zip(outs, calls):
                    want = call["want"]
                    stress.count_words(out, want if control else want[:, None])
                    if xn is not None:
                        stress.count_near(xn, call["xn_want"][:, None])

            words = sum(rows * HIDDEN // 2 for rows in row_counts)
            stress.graphs.append(
                StressGraph(
                    body,
                    len(calls),
                    words if control else words * STREAMS,
                    0,
                    calls,
                    0 if control else words * STREAMS,
                )
            )


def _stress(rank: int, replays: int, control: str | None, log: bool = False):
    """The race stress of the boundary kernel (or the control) on this rank
    of the TP4 group; every rank must call it."""
    stress = ReplayStress(torch.device("cuda", rank))
    # Inside graph_capture, vLLM's custom all-reduce returns an uninitialized
    # tensor outside the capture itself: its eager runs say nothing.
    stress.check_eager = control != "all-reduce"
    _stress_graphs(stress, rank, control)
    name = "boundary" if control is None else f"boundary control ({control})"
    result = report(name, stress.run(replays, log=log))
    if control is None:
        assert result.bad_words == 0 and result.bad_near_words == 0, result
        assert result.eager_bad == 0, result


@requires_sm100
def test_tp4_boundary_matches_stock(monkeypatch):
    if torch.accelerator.device_count() < 4:
        pytest.skip("Requires four GPUs with NVLink multicast")
    multi_process_parallel(monkeypatch, 4, 1, _tp_boundary)


# The stall test's boundaries, kept until the worker process exits (one task
# per worker, ``max_calls=1``): freeing a symmetric-memory mailbox after the
# kernel trapped aborts the process.
_STALLED: list[object] = []


@ray.remote(num_gpus=1, max_calls=1)
def _tp_stall(monkeypatch, tp_size, pp_size, rank, distributed_init_port):
    from vllm.distributed import get_tp_group
    from vllm.models.qwen4_exp.nvidia.specialized_decode.boundary import (
        SublayerBoundary,
    )

    with monkeypatch.context() as m:
        m.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        device = torch.device(f"cuda:{rank}")
        torch.accelerator.set_device_index(device)
        init_test_distributed_environment(tp_size, pp_size, rank, distributed_init_port)
        ensure_model_parallel_initialized(tp_size, pp_size)
        group = get_tp_group()
        boundary = SublayerBoundary(
            group.device_group, MAX_ROWS, device, stall_timeout=1.0
        )
        _STALLED.append(boundary)
        skipper = tp_size - 1
        if rank != skipper:
            # The last rank never publishes: the others' polls time out, and
            # their kernel traps (the CUDA context is unusable afterwards).
            with pytest.raises(RuntimeError):
                _boundary(boundary, _site(1, rank, device), False)
                torch.accelerator.synchronize(device)
            report = boundary.stall_report()
            assert report is not None
            assert report.startswith(f"rank {rank} timed out in generation 0")
            assert f"rank(s) [{skipper}]" in report
        else:
            assert boundary.stall_report() is None
        # The skipping rank keeps its mailbox mapped until the others are done.
        torch.distributed.barrier(group=group.cpu_group)


@requires_sm100
def test_tp4_boundary_reports_a_stall(monkeypatch):
    if torch.accelerator.device_count() < 4:
        pytest.skip("Requires four GPUs with NVLink multicast")
    multi_process_parallel(monkeypatch, 4, 1, _tp_stall)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Race stress of the boundary kernel on four GPUs (replay_stress.py)"
    )
    parser.add_argument("--replays", type=int, default=REPLAYS)
    parser.add_argument(
        "--control",
        choices=CONTROLS,
        help="replay the graphs on vLLM's custom all-reduce or all-gather "
        "and only count the bad words",
    )
    args = parser.parse_args()
    spawn(_stress, args.replays, args.control, True)
