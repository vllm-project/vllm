# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reduce-scatter must not depend on the number of tokens.

Library collectives pick their algorithm, channel count and chunk boundaries
from the message size, so the order in which a given element's contributions are
summed changes with the batch size. Under ``VLLM_BATCH_INVARIANT`` the
communicator is expected to route around that, whichever backend it lands on.

Requires at least 4 GPUs: a 2-rank sum is order independent, so TP=2 passes even
with a batch-variant collective. Runs again at 8 where they are available, which
is the more sensitive probe.
"""

import pytest
import ray
import torch
import torch.distributed as dist

from tests.utils import (
    init_test_distributed_environment,
    multi_gpu_marks,
    multi_process_parallel,
)
from vllm.distributed.parallel_state import get_tp_group, set_custom_all_reduce
from vllm.model_executor.layers.batch_invariant import override_envs_for_invariance

from .utils import order_sensitive_elements, skip_if_not_rocm

# ROCm-only for now: that is where `CudaCommunicator` replaces the collective
# under the mode. Off ROCm it pins NCCL's algorithm, protocol and channel count
# instead.
pytestmark = [skip_if_not_rocm, *multi_gpu_marks(num_gpus=4)]

# Divisible by 8 so the same counts work at both world sizes, and spread across
# the small-message thresholds where the collective changes protocol. They also
# straddle the custom kernel's default 16MiB bound -- bf16/fp16 above 2048
# tokens and fp32 above 1024 take the all-to-all fallback instead -- so the
# sweep below spans both implementations rather than one.
TOKEN_COUNTS = [32, 40, 64, 128, 256, 512, 1024, 3000, 4096]
HIDDEN_SIZE = 4096

# Row 0 sits at offset 0, so it lands in the first chunk of every decomposition
# and stays invariant even when the rest of the tensor does not. The rest of the
# probe is checked in full rather than sampled: summing four 16-bit values in an
# fp32 accumulator is exact for all but a handful of elements, so a sampled set
# of rows is regularly insensitive to reordering by chance.
CHECK_ROWS = list(range(1, 32))

# (dtype, exponent_spread), as in the all-reduce test: the spread widens the
# operand range until the fp32 accumulator inside the reduction has to round,
# without which reordering it is unobservable and the sweep asserts nothing.
CASES = [
    (torch.bfloat16, 20),
    (torch.float16, 12),
    (torch.float32, 0),
]


def _make_input(rows: int, dtype: torch.dtype, spread: int, device, seed: int):
    generator = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn(rows, HIDDEN_SIZE, generator=generator, device=device)
    if spread:
        exponents = torch.randint(
            -spread,
            spread,
            x.shape,
            generator=generator,
            device=device,
            dtype=torch.int32,
        )
        x = x * torch.exp2(exponents.float())
    return x.to(dtype)


def _splits(num_tokens: int, world_size: int) -> list[list[int]]:
    """Uniform plus lopsided row distributions, all summing to num_tokens."""
    tail = num_tokens - (world_size - 1)
    return [
        [num_tokens // world_size] * world_size,
        [1] * (world_size - 1) + [tail],
        [tail] + [1] * (world_size - 1),
        # Half the ranks get nothing: they still have rows to send.
        [0 if r % 2 else 2 * num_tokens // world_size for r in range(world_size)],
    ]


def _reconstruct(shard: torch.Tensor, sizes: list[int]) -> torch.Tensor:
    """The full reduced tensor, assembled on every rank from the shards.

    All-gather is pure data movement, so this adds no arithmetic of its own and
    lets a fixed logical row be compared across sweeps that scatter it to
    different ranks.
    """
    group = get_tp_group()
    padded = shard.new_zeros((max(sizes), *shard.shape[1:]))
    padded[: shard.shape[0]] = shard
    gathered = torch.empty(
        (group.world_size * padded.shape[0], *padded.shape[1:]),
        dtype=shard.dtype,
        device=shard.device,
    )
    dist.all_gather_into_tensor(gathered, padded, group=group.device_group)
    gathered = gathered.view(group.world_size, *padded.shape)
    return torch.cat([gathered[r, : sizes[r]] for r in range(group.world_size)])


def _check_reduce_scatter(
    monkeypatch: pytest.MonkeyPatch,
    tp_size: int,
    pp_size: int,
    rank: int,
    distributed_init_port: str,
):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    override_envs_for_invariance()

    device = torch.device(f"cuda:{rank}")
    torch.accelerator.set_device_index(device)
    init_test_distributed_environment(tp_size, pp_size, rank, distributed_init_port)
    group = get_tp_group()

    failures = []
    vacuous = []
    checked = 0
    for dtype, spread in CASES:
        full = _make_input(max(TOKEN_COUNTS), dtype, spread, device, 1234 + rank)
        sensitive = order_sensitive_elements(full[: CHECK_ROWS[-1] + 1])[CHECK_ROWS]
        if not sensitive.any():
            vacuous.append(f"{dtype} spread=+-{spread}")

        uniform = {
            n: _reconstruct(
                group.reduce_scatter(full[:n].contiguous(), dim=0),
                [n // tp_size] * tp_size,
            )
            for n in TOKEN_COUNTS
        }
        # reduce_scatterv only changes which rows a rank receives, never the
        # ascending-rank order they are summed in, so its results have to fall
        # in the same equivalence class as the uniform ones.
        n = TOKEN_COUNTS[-1]
        variable = [
            _reconstruct(
                group.reduce_scatterv(full[:n].contiguous(), dim=0, sizes=sizes), sizes
            )
            for sizes in _splits(n, tp_size)
        ]

        for index, row in enumerate(CHECK_ROWS):
            # A row whose sum is exact in either order cannot observe a
            # reordering, so comparing it would pass regardless.
            if not sensitive[index].any():
                continue
            checked += 1
            reference = uniform[TOKEN_COUNTS[0]][row]
            variant = [
                n for n in TOKEN_COUNTS if not torch.equal(uniform[n][row], reference)
            ]
            if variant:
                failures.append(
                    f"{dtype} spread=+-{spread} row={row} changed at token "
                    f"counts {variant}"
                )
            split_variant = [
                sizes
                for sizes, out in zip(_splits(n, tp_size), variable)
                if not torch.equal(out[row], reference)
            ]
            if split_variant:
                failures.append(
                    f"{dtype} spread=+-{spread} row={row} changed under splits "
                    f"{split_variant}"
                )

    assert not vacuous, (
        "these cases cannot observe a reduction reordering, so the sweep passes "
        "without asserting anything:\n  " + "\n  ".join(vacuous)
    )
    assert checked, "no order-sensitive row was compared"
    assert not failures, (
        f"reduce-scatter depends on the token count or the shard split over "
        f"{tp_size} ranks:\n  " + "\n  ".join(failures)
    )


def _check_default_unchanged(
    monkeypatch: pytest.MonkeyPatch,
    tp_size: int,
    pp_size: int,
    rank: int,
    distributed_init_port: str,
):
    """Without the mode, both entry points must still be the library collective."""
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "0")

    device = torch.device(f"cuda:{rank}")
    torch.accelerator.set_device_index(device)
    init_test_distributed_environment(tp_size, pp_size, rank, distributed_init_port)
    group = get_tp_group()

    for dtype, spread in CASES:
        x = _make_input(256, dtype, spread, device, 1234 + rank)
        expected = torch.empty(
            (256 // tp_size, HIDDEN_SIZE), dtype=dtype, device=device
        )
        dist.reduce_scatter_tensor(expected, x, group=group.device_group)
        assert torch.equal(group.reduce_scatter(x, dim=0), expected)
        assert torch.equal(
            group.reduce_scatterv(x, dim=0, sizes=[256 // tp_size] * tp_size), expected
        )


def _check_implementations_agree(
    monkeypatch: pytest.MonkeyPatch,
    tp_size: int,
    pp_size: int,
    rank: int,
    distributed_init_port: str,
):
    """The custom kernel and the all-to-all fallback must be interchangeable.

    Batch invariance serves reduce-scatters below the custom all-reduce's
    ``max_reduce_scatter_size`` with ``cross_device_reduce_scatter`` and
    everything above it with an all-to-all plus a fixed rank-order sum. That is
    a size-dependent switch between two *implementations* -- benign only while
    they agree bitwise, and a batch-variance bug otherwise, since the switch
    point is a token count.

    Both sum ``world_size`` contributions in ascending rank order into an fp32
    accumulator and round once, which is a per-element property independent of
    message size, so comparing them below the bound settles the boundary as well.
    """
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    override_envs_for_invariance()

    device = torch.device(f"cuda:{rank}")
    torch.accelerator.set_device_index(device)
    set_custom_all_reduce(True)
    init_test_distributed_environment(tp_size, pp_size, rank, distributed_init_port)
    group = get_tp_group()

    from vllm.model_executor.layers.batch_invariant import (
        reduce_scatter_batch_invariant,
    )

    ca_comm = group.device_communicator.ca_comm
    assert ca_comm is not None and not ca_comm.disabled, (
        "custom all-reduce is not live, so there is no second implementation "
        "to compare against and this test asserts nothing"
    )

    failures = []
    compared = 0
    for dtype, spread in CASES:
        full = _make_input(max(TOKEN_COUNTS), dtype, spread, device, 1234 + rank)
        for num_tokens in TOKEN_COUNTS:
            probe = full[:num_tokens].contiguous()
            custom = ca_comm.custom_reduce_scatter(probe)
            if custom is None:
                continue
            # Only elements whose accumulation is inexact can tell the two
            # implementations apart; the rest agree for free.
            sensitive = order_sensitive_elements(probe)
            fallback = reduce_scatter_batch_invariant(probe, group.device_group)
            offset = rank * (num_tokens // tp_size)
            for row in range(custom.shape[0]):
                if not sensitive[offset + row].any():
                    continue
                compared += 1
                if not torch.equal(custom[row], fallback[row]):
                    failures.append(
                        f"{dtype} spread=+-{spread} tokens={num_tokens} "
                        f"row={offset + row}"
                    )

    assert compared, (
        "no order-sensitive row was served by the custom kernel, so the two "
        "implementations were never actually compared"
    )
    assert not failures, (
        "the custom reduce-scatter and the all-to-all fallback disagree, so the "
        "size-based switch between them is itself batch variance:\n  "
        + "\n  ".join(failures)
    )


@ray.remote(num_gpus=1, max_calls=1)
def reduce_scatter_worker(monkeypatch, tp_size, pp_size, rank, port):
    _check_reduce_scatter(monkeypatch, tp_size, pp_size, rank, port)


@ray.remote(num_gpus=1, max_calls=1)
def implementations_agree_worker(monkeypatch, tp_size, pp_size, rank, port):
    _check_implementations_agree(monkeypatch, tp_size, pp_size, rank, port)


@ray.remote(num_gpus=1, max_calls=1)
def default_worker(monkeypatch, tp_size, pp_size, rank, port):
    _check_default_unchanged(monkeypatch, tp_size, pp_size, rank, port)


@pytest.mark.parametrize(
    "tp_size", [4, pytest.param(8, marks=multi_gpu_marks(num_gpus=8))]
)
@pytest.mark.parametrize(
    "worker",
    [reduce_scatter_worker, implementations_agree_worker, default_worker],
    ids=["batch_invariant", "implementations_agree", "default"],
)
def test_tp_reduce_scatter_is_batch_invariant(
    tp_size: int,
    worker,
    monkeypatch: pytest.MonkeyPatch,
):
    multi_process_parallel(monkeypatch, tp_size, 1, worker)
