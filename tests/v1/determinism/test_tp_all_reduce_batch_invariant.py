# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tensor-parallel all-reduce must not depend on the number of tokens.

Library collectives pick their algorithm, channel count and chunk boundaries
from the message size, so the order in which a given element's contributions are
summed changes with the batch size. Under ``VLLM_BATCH_INVARIANT`` the
communicator is expected to route around that, whichever backend it lands on.

Requires at least 4 GPUs: a 2-rank sum is order independent, so TP=2 passes even
with a batch-variant collective. Runs again at 8 where they are available, which
is the more sensitive probe -- see the parametrization note below.
"""

import os
from pathlib import Path

import pytest
import ray
import torch

from tests.utils import (
    init_test_distributed_environment,
    multi_gpu_marks,
    multi_process_parallel,
)
from vllm.distributed import tensor_model_parallel_all_reduce
from vllm.distributed.parallel_state import get_tp_group, set_custom_all_reduce

from .utils import order_sensitive_elements, skip_if_not_cuda_alike

# multi_gpu_test would also wrap the test in create_new_process_for_each_test,
# whose re-import breaks the ray workers below, so take its marks alone: the
# registered `distributed` selector keeps `-m distributed` picking this test up,
# and its skipif enforces the 4 GPUs the module docstring explains are needed.
pytestmark = [skip_if_not_cuda_alike, *multi_gpu_marks(num_gpus=4)]

# Token counts spanning the small-message thresholds where the collectives
# switch protocol, chunking, or algorithm. At world size 4 the custom all-reduce
# switches from its one-shot to its two-shot kernel at 512KiB, i.e. between 32
# and 64 tokens for the 16-bit cases and between 17 and 32 for fp32.
TOKEN_COUNTS = [1, 2, 3, 4, 5, 8, 16, 17, 32, 64, 128, 256, 512]
HIDDEN_SIZE = 4096

# Row 0 always sits at offset 0, so it lands in the first chunk of every
# decomposition and stays invariant even when the rest of the tensor does not.
# Checking it alone hides real failures.
CHECK_ROWS = [0, 1, 2, 3, 7, 15, 31]

# (dtype, exponent_spread). The spread widens the operand range until the fp32
# accumulator inside the reduction has to round -- without that, reduction order
# is unobservable and the sweep asserts nothing. `order_sensitive_elements`
# enforces this per case. fp16 stops at 14: from 15 the operands themselves
# overflow to +-inf, and inf + -inf gives NaN, which would register as
# "sensitive" while failing the comparison for reasons unrelated to the
# collective. fp32 needs no spread at all.
CASES = [
    (torch.bfloat16, 20),
    (torch.float16, 14),
    (torch.float32, 0),
]

# fp32 at 512 tokens is exactly 8MiB, the custom all-reduce's default max_size,
# whose bound is strict -- so that one point takes the all-gather fallback even
# in the custom_ar worker. Pinned so that a change to max_size, HIDDEN_SIZE or
# TOKEN_COUNTS surfaces here rather than silently moving which implementation
# the sweep tests. (max_size is the library default here because the group is
# built without an engine config; CudaCommunicator sizes it from
# max_num_batched_tokens under the mode.)
FALLBACK_POINTS = {(torch.float32, 512)}


def _check_all_reduce(
    monkeypatch: pytest.MonkeyPatch,
    tp_size: int,
    pp_size: int,
    rank: int,
    distributed_init_port: str,
    use_custom_all_reduce: bool,
):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")

    device = torch.device(f"cuda:{rank}")
    torch.accelerator.set_device_index(device)

    # Both of the paths batch invariance can take have to hold. With the custom
    # all-reduce enabled it serves everything under its size limit; with it
    # disabled -- and above its limit either way -- every message falls through
    # to all-gather plus a fixed rank-order sum. The library collective is never
    # reached under the mode.
    set_custom_all_reduce(use_custom_all_reduce)
    init_test_distributed_environment(tp_size, pp_size, rank, distributed_init_port)

    # Without this the two parametrizations can silently become the same test:
    # the custom all-reduce disables itself when P2P is unavailable, and on ROCm
    # an enabled AITER custom all-reduce leaves `ca_comm` unset as well. Batch
    # invariance dispatches through `ca_comm` alone, so in either case every
    # message would take the all-gather fallback and both workers would pass
    # having covered one path.
    ca_comm = get_tp_group().device_communicator.ca_comm
    if use_custom_all_reduce:
        assert ca_comm is not None and not ca_comm.disabled, (
            "custom all-reduce is not live, so this worker exercises the same "
            "all-gather fallback as fallback_ar and covers nothing extra"
        )
    else:
        assert ca_comm is None, (
            "custom all-reduce was constructed with set_custom_all_reduce(False)"
        )

    failures = []
    vacuous = []
    fell_through: set[tuple[torch.dtype, int]] = set()
    for dtype, spread in CASES:
        generator = torch.Generator(device=device).manual_seed(1234 + rank)
        full = torch.randn(
            max(TOKEN_COUNTS), HIDDEN_SIZE, generator=generator, device=device
        )
        if spread:
            exponents = torch.randint(
                -spread,
                spread,
                full.shape,
                generator=generator,
                device=device,
                dtype=torch.int32,
            )
            full = full * torch.exp2(exponents.float())
        full = full.to(dtype)

        # Row 0 is excluded: it never moves under a decomposition, so its
        # sensitivity would not make the sweep able to fail.
        sensitive = order_sensitive_elements(full[: CHECK_ROWS[-1] + 1])[CHECK_ROWS[1:]]
        if not sensitive.any():
            vacuous.append(
                f"{dtype} spread=+-{spread}: reversing the rank order leaves "
                f"every checked element unchanged, so the fp32 accumulation is "
                f"exact for these operands and no reordering is observable"
            )

        if ca_comm is not None:
            fell_through.update(
                (dtype, num_tokens)
                for num_tokens in TOKEN_COUNTS
                if not ca_comm.should_custom_ar(full[:num_tokens])
            )

        reduced = {
            num_tokens: tensor_model_parallel_all_reduce(full[:num_tokens].clone())
            for num_tokens in TOKEN_COUNTS
        }
        for row in CHECK_ROWS:
            # A row is only comparable across launches that actually contain it.
            counts = [n for n in TOKEN_COUNTS if n > row]
            reference = reduced[counts[0]][row]
            variant = [n for n in counts if not torch.equal(reduced[n][row], reference)]
            if variant:
                failures.append(
                    f"{dtype} spread=+-{spread} row={row} changed at token "
                    f"counts {variant}"
                )

    assert not vacuous, (
        "these cases cannot observe a reduction reordering, so their sweep "
        "below passes without asserting anything:\n  " + "\n  ".join(vacuous)
    )

    if ca_comm is not None:
        assert fell_through == FALLBACK_POINTS, (
            f"which token counts the custom all-reduce serves has moved: it "
            f"declines {sorted((str(d), n) for d, n in fell_through)}, expected "
            f"{sorted((str(d), n) for d, n in FALLBACK_POINTS)} (max_size="
            f"{ca_comm.max_size})"
        )

    assert not failures, (
        f"all-reduce depends on the token count over {tp_size} ranks "
        f"(custom_all_reduce={use_custom_all_reduce}):\n  " + "\n  ".join(failures)
    )


# multi_process_parallel does not forward extra arguments to the remote, so bind
# the two paths as separate workers.
@ray.remote(num_gpus=1, max_calls=1)
def custom_ar_worker(monkeypatch, tp_size, pp_size, rank, port):
    _check_all_reduce(monkeypatch, tp_size, pp_size, rank, port, True)


def _check_implementations_agree(
    monkeypatch: pytest.MonkeyPatch,
    tp_size: int,
    pp_size: int,
    rank: int,
    distributed_init_port: str,
):
    """The custom kernel and the all-gather fallback must be interchangeable.

    Batch invariance serves messages below the custom all-reduce's ``max_size``
    with the custom kernel and everything above it with all-gather plus a fixed
    rank-order sum. That is a size-dependent switch between two *implementations*
    -- benign only while they agree bitwise, and a batch-variance bug otherwise,
    since the switch point is a token count.

    Both sum ``world_size`` contributions in ascending rank order into an fp32
    accumulator and round once. That is a per-element property, independent of
    message size, so comparing them below ``max_size`` -- where the custom kernel
    can actually serve the call -- settles the boundary wherever
    ``CudaCommunicator`` places it for a given engine config.
    """
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")

    device = torch.device(f"cuda:{rank}")
    torch.accelerator.set_device_index(device)
    set_custom_all_reduce(True)
    init_test_distributed_environment(tp_size, pp_size, rank, distributed_init_port)

    from vllm.model_executor.layers.batch_invariant import all_reduce_batch_invariant

    ca_comm = get_tp_group().device_communicator.ca_comm
    assert ca_comm is not None and not ca_comm.disabled, (
        "custom all-reduce is not live, so there is no second implementation "
        "to compare against and this test asserts nothing"
    )

    failures = []
    compared = 0
    for dtype, spread in CASES:
        generator = torch.Generator(device=device).manual_seed(1234 + rank)
        full = torch.randn(
            max(TOKEN_COUNTS), HIDDEN_SIZE, generator=generator, device=device
        )
        if spread:
            exponents = torch.randint(
                -spread,
                spread,
                full.shape,
                generator=generator,
                device=device,
                dtype=torch.int32,
            )
            full = full * torch.exp2(exponents.float())
        full = full.to(dtype)

        for num_tokens in TOKEN_COUNTS:
            probe = full[:num_tokens].contiguous()
            if not ca_comm.should_custom_ar(probe):
                continue
            # Only rows whose accumulation is inexact can tell the two
            # implementations apart; the rest agree for free and would let this
            # pass while blind.
            sensitive = order_sensitive_elements(probe)
            custom = ca_comm.custom_all_reduce(probe.clone())
            assert custom is not None
            fallback = all_reduce_batch_invariant(
                probe.clone(), get_tp_group().device_group
            )
            for row in CHECK_ROWS:
                if row >= num_tokens or not sensitive[row].any():
                    continue
                compared += 1
                if not torch.equal(custom[row], fallback[row]):
                    failures.append(
                        f"{dtype} spread=+-{spread} tokens={num_tokens} row={row}"
                    )

    assert compared, (
        "no order-sensitive row was served by the custom all-reduce, so the two "
        "implementations were never actually compared"
    )
    assert not failures, (
        "the custom all-reduce and the all-gather fallback disagree, so the "
        "size-based switch between them is itself batch variance:\n  "
        + "\n  ".join(failures)
    )


@ray.remote(num_gpus=1, max_calls=1)
def implementations_agree_worker(monkeypatch, tp_size, pp_size, rank, port):
    _check_implementations_agree(monkeypatch, tp_size, pp_size, rank, port)


@ray.remote(num_gpus=1, max_calls=1)
def fallback_ar_worker(monkeypatch, tp_size, pp_size, rank, port):
    _check_all_reduce(monkeypatch, tp_size, pp_size, rank, port, False)


# Eight ranks where the hardware allows: an fp32 accumulator often sums four
# contributions exactly, so world size 4 is the weaker probe. Doubling the ranks
# roughly doubles how many checked rows can observe a reordering at all --
# measured on gfx950, 148 order-sensitive comparisons at 4 and 256 at 8, fp32
# rising from 68 to 136 because fp32 operands leave the accumulator no headroom.
@pytest.mark.parametrize(
    "tp_size", [4, pytest.param(8, marks=multi_gpu_marks(num_gpus=8))]
)
@pytest.mark.parametrize(
    "worker",
    [custom_ar_worker, fallback_ar_worker, implementations_agree_worker],
    ids=["custom_ar", "fallback_ar", "implementations_agree"],
)
def test_tp_all_reduce_is_batch_invariant(
    tp_size: int,
    worker,
    monkeypatch: pytest.MonkeyPatch,
):
    # multi_process_parallel forwards PYTHONPATH to the ray workers, which have
    # to import this module to unpickle the remote worker.
    test_dir = str(Path(__file__).parent)
    monkeypatch.setenv(
        "PYTHONPATH",
        os.pathsep.join(filter(None, [test_dir, os.environ.get("PYTHONPATH")])),
    )
    multi_process_parallel(monkeypatch, tp_size, 1, worker)
