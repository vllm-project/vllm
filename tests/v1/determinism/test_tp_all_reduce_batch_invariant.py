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
from vllm.model_executor.layers.batch_invariant import override_envs_for_invariance

from .utils import order_sensitive_elements, skip_if_not_rocm

# ROCm-only for now: that is where `CudaCommunicator` replaces the collective
# under the mode. Off ROCm it pins NCCL's algorithm, protocol and channel count
# instead.
pytestmark = [skip_if_not_rocm, *multi_gpu_marks(num_gpus=4)]

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
# enforces this per case. fp16 stops at 12, past which a world_size sum
# saturates the dtype. fp32 needs no spread at all.
CASES = [
    (torch.bfloat16, 20),
    (torch.float16, 12),
    (torch.float32, 0),
]


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
    override_envs_for_invariance()

    device = torch.device(f"cuda:{rank}")
    torch.accelerator.set_device_index(device)

    set_custom_all_reduce(use_custom_all_reduce)
    init_test_distributed_environment(tp_size, pp_size, rank, distributed_init_port)

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

    assert not failures, (
        f"all-reduce depends on the token count over {tp_size} ranks "
        f"(custom_all_reduce={use_custom_all_reduce}):\n  " + "\n  ".join(failures)
    )


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
    can actually serve the call -- settles the boundary as well.
    """
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    override_envs_for_invariance()

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
# roughly doubles how many checked rows can observe a reordering at all.
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
    multi_process_parallel(monkeypatch, tp_size, 1, worker)
