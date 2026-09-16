# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Parity and layout checks for native MRV2 Uno input preparation."""

import inspect
from types import SimpleNamespace

import pytest
import torch

from vllm.v1.attention.backends.utils import PAD_SLOT_ID
from vllm.v1.worker.gpu.input_batch import InputBuffers
from vllm.v1.worker.gpu.spec_decode.uno import prepare_uno_inputs_reference
from vllm.v1.worker.gpu.spec_decode.uno_prepare import (
    _target_input_lengths,
    prepare_uno_inputs_fused,
    prepare_uno_launch_key,
)

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires a CUDA device"
)


def _batch(k: int, scenario: str, device: torch.device):
    """Build one batch with native MRV2 request-state tensor layouts."""
    del k  # The target batch is intentionally independent of draft K.
    n = 3
    max_reqs = 5
    max_tokens = 5 * 8
    # Two tokens for request 0, two for request 1, and three for request 2.
    query_start_loc = torch.tensor([0, 2, 4, 7], dtype=torch.int32, device=device)
    positions = torch.tensor(
        [4, 5, 10, 11, 20, 21, 22], dtype=torch.int64, device=device
    )
    if scenario == "reorder":
        idx_mapping = torch.tensor([4, 1, 3], dtype=torch.int64, device=device)
        num_sampled = torch.tensor([1, 1, 1], dtype=torch.int32, device=device)
        num_rejected = torch.tensor([0, 1, 2], dtype=torch.int32, device=device)
    elif scenario == "mixed_prefill":
        idx_mapping = torch.tensor([0, 2, 1], dtype=torch.int64, device=device)
        num_sampled = torch.tensor([0, 1, 0], dtype=torch.int32, device=device)
        num_rejected = torch.tensor([0, 0, 1], dtype=torch.int32, device=device)
    elif scenario == "boundary":
        idx_mapping = torch.tensor([2, 0, 4], dtype=torch.int64, device=device)
        num_sampled = torch.tensor([1, 0, 1], dtype=torch.int32, device=device)
        num_rejected = torch.tensor([0, 1, 2], dtype=torch.int32, device=device)
    else:
        raise AssertionError(f"unknown scenario: {scenario}")

    input_batch = SimpleNamespace(
        num_reqs=n,
        idx_mapping=idx_mapping,
        query_start_loc=query_start_loc,
        positions=positions,
    )
    last_sampled = torch.tensor(
        [[100], [101], [102], [103], [104]],
        dtype=torch.int64,
        device=device,
    )
    next_prefill_tokens = torch.tensor(
        [[200, 201, 202, 203, 204], [900, 901, 902, 903, 904]],
        dtype=torch.int32,
        device=device,
    )
    seeds = torch.tensor([11, 13, 17, 19, 23], dtype=torch.int64, device=device)

    if scenario == "boundary":
        # Logical block 0 is null, row 1 has a null block at logical block 2,
        # and the table ends before later positions.  The short table and the
        # max-length limit exercise distinct out-of-context paths.
        block_table = torch.tensor(
            [[0, 7, 8], [0, 7, 0], [0, 9, 10]],
            dtype=torch.int32,
            device=device,
        )
        max_model_len = 14
    else:
        block_table = torch.tensor(
            [
                [0, 7, 8, 9, 10, 11, 12, 13],
                [0, 7, 0, 15, 16, 17, 18, 19],
                [0, 9, 10, 11, 12, 13, 14, 15],
            ],
            dtype=torch.int32,
            device=device,
        )
        max_model_len = 32

    buffers = InputBuffers(max_reqs, max_tokens, device)
    slot_mapping = torch.full((max_tokens,), 12345, dtype=torch.int64, device=device)
    sample_idx_mapping = torch.full(
        (max_tokens,), 12345, dtype=torch.int32, device=device
    )
    return SimpleNamespace(
        buffers=buffers,
        slot_mapping=slot_mapping,
        sample_idx_mapping=sample_idx_mapping,
        input_batch=input_batch,
        num_sampled=num_sampled,
        num_rejected=num_rejected,
        last_sampled=last_sampled,
        next_prefill_tokens=next_prefill_tokens,
        seeds=seeds,
        block_table=block_table,
        max_model_len=max_model_len,
    )


def _cpu_clone(case):
    """Clone a GPU case to CPU for the eager reference path."""

    def copy(value):
        if isinstance(value, torch.Tensor):
            return value.cpu()
        return value

    cpu = SimpleNamespace(
        buffers=InputBuffers(
            case.buffers.max_num_reqs,
            case.buffers.max_num_tokens,
            torch.device("cpu"),
        ),
        slot_mapping=copy(case.slot_mapping),
        sample_idx_mapping=copy(case.sample_idx_mapping),
        input_batch=SimpleNamespace(
            num_reqs=case.input_batch.num_reqs,
            idx_mapping=copy(case.input_batch.idx_mapping),
            query_start_loc=copy(case.input_batch.query_start_loc),
            positions=copy(case.input_batch.positions),
        ),
        num_sampled=copy(case.num_sampled),
        num_rejected=copy(case.num_rejected),
        last_sampled=copy(case.last_sampled),
        next_prefill_tokens=copy(case.next_prefill_tokens),
        seeds=copy(case.seeds),
        block_table=copy(case.block_table),
        max_model_len=case.max_model_len,
    )
    return cpu


def _outputs(case):
    return (
        case.buffers.input_ids.clone(),
        case.buffers.positions.clone(),
        case.buffers.seq_lens.clone(),
        case.buffers.query_start_loc.clone(),
        case.slot_mapping.clone(),
        case.sample_idx_mapping.clone(),
    )


def _run_reference(case, k: int, step: int):
    prepare_uno_inputs_reference(
        case.buffers,
        case.slot_mapping,
        case.sample_idx_mapping,
        case.input_batch,
        case.num_sampled,
        case.num_rejected,
        case.last_sampled,
        case.next_prefill_tokens,
        case.seeds,
        case.block_table,
        4,
        k,
        case.max_model_len,
        29,
        100_003,
        step,
    )


@pytest.mark.parametrize("k", [1, 4, 8])
@pytest.mark.parametrize("scenario", ["reorder", "mixed_prefill", "boundary"])
@requires_cuda
def test_fused_uno_inputs_match_eager_reference(k, scenario):
    device = torch.device("cuda")
    gpu = _batch(k, scenario, device)
    cpu = _cpu_clone(gpu)
    step = 37
    _run_reference(cpu, k, step)
    expected = tuple(value.cpu() for value in _outputs(cpu))

    prepare_uno_inputs_fused(
        gpu.buffers,
        gpu.slot_mapping,
        gpu.sample_idx_mapping,
        gpu.input_batch,
        gpu.num_sampled,
        gpu.num_rejected,
        gpu.last_sampled,
        gpu.next_prefill_tokens,
        gpu.seeds,
        gpu.block_table,
        4,
        k,
        gpu.max_model_len,
        29,
        100_003,
        step,
    )
    torch.accelerator.synchronize()
    actual = tuple(value.cpu() for value in _outputs(gpu))
    assert all(torch.equal(got, want) for got, want in zip(actual, expected))


@pytest.mark.parametrize("k", [1, 4, 8])
@requires_cuda
def test_fused_uno_boundaries_write_padding_slots(k):
    case = _batch(k, "boundary", torch.device("cuda"))
    cpu = _cpu_clone(case)
    _run_reference(cpu, k, 37)
    expected = tuple(value.cpu() for value in _outputs(cpu))
    prepare_uno_inputs_fused(
        case.buffers,
        case.slot_mapping,
        case.sample_idx_mapping,
        case.input_batch,
        case.num_sampled,
        case.num_rejected,
        case.last_sampled,
        case.next_prefill_tokens,
        case.seeds,
        case.block_table,
        4,
        k,
        case.max_model_len,
        29,
        100_003,
        37,
    )
    torch.accelerator.synchronize()
    actual = tuple(value.cpu() for value in _outputs(case))
    assert all(torch.equal(got, want) for got, want in zip(actual, expected))
    # Request 2 starts beyond max_model_len, and request 1 reaches a null
    # logical block.  Both paths must leave only PAD_SLOT_ID in those rows.
    slots = actual[4][: case.input_batch.num_reqs * k]
    assert torch.all(slots[2 * k : 3 * k] == PAD_SLOT_ID)
    assert torch.all(slots[k : 2 * k] == PAD_SLOT_ID)


@requires_cuda
def test_fused_uno_covers_large_graph_padding_capacity():
    # Keep the live request batch small while forcing multiple fixed-size
    # Triton programs to cover the persistent 8K-token graph buffers.
    case = _batch(4, "reorder", torch.device("cuda"))
    case.buffers = InputBuffers(5, 8192, torch.device("cuda"))
    case.slot_mapping = torch.full((8192,), 12345, dtype=torch.int64, device="cuda")
    case.sample_idx_mapping = torch.full(
        (8192,), 12345, dtype=torch.int32, device="cuda"
    )
    cpu = _cpu_clone(case)
    _run_reference(cpu, 4, 37)
    expected = tuple(value.cpu() for value in _outputs(cpu))

    prepare_uno_inputs_fused(
        case.buffers,
        case.slot_mapping,
        case.sample_idx_mapping,
        case.input_batch,
        case.num_sampled,
        case.num_rejected,
        case.last_sampled,
        case.next_prefill_tokens,
        case.seeds,
        case.block_table,
        4,
        4,
        case.max_model_len,
        29,
        100_003,
        37,
    )
    torch.accelerator.synchronize()
    actual = tuple(value.cpu() for value in _outputs(case))
    assert all(torch.equal(got, want) for got, want in zip(actual, expected))


@requires_cuda
def test_fused_uno_preserves_large_runtime_step_arithmetic():
    case = _batch(4, "mixed_prefill", torch.device("cuda"))
    step = 2**40 + 19
    cpu = _cpu_clone(case)
    _run_reference(cpu, 4, step)
    expected = tuple(value.cpu() for value in _outputs(cpu))
    prepare_uno_inputs_fused(
        case.buffers,
        case.slot_mapping,
        case.sample_idx_mapping,
        case.input_batch,
        case.num_sampled,
        case.num_rejected,
        case.last_sampled,
        case.next_prefill_tokens,
        case.seeds,
        case.block_table,
        4,
        4,
        case.max_model_len,
        29,
        100_003,
        step,
    )
    torch.accelerator.synchronize()
    actual = tuple(value.cpu() for value in _outputs(case))
    assert all(torch.equal(got, want) for got, want in zip(actual, expected))


def test_fused_uno_rejects_cpu_buffers_before_launch():
    case = _batch(4, "reorder", torch.device("cpu"))
    with pytest.raises(ValueError, match="requires CUDA"):
        prepare_uno_inputs_fused(
            case.buffers,
            case.slot_mapping,
            case.sample_idx_mapping,
            case.input_batch,
            case.num_sampled,
            case.num_rejected,
            case.last_sampled,
            case.next_prefill_tokens,
            case.seeds,
            case.block_table,
            4,
            4,
            case.max_model_len,
            29,
            100_003,
            37,
        )


@pytest.mark.parametrize(
    ("num_reqs", "served_tokens"),
    [(1, 32), (4, 2048)],
)
def test_uno_prepare_specialization_ignores_dynamic_target_view_lengths(
    num_reqs, served_tokens
):
    """Warmup and serving retain logical bounds without recompiling for views."""
    device = torch.device("cpu")
    buffers = InputBuffers(4, 2048, device)
    slot_mapping = torch.empty(2048, dtype=torch.int64, device=device)
    sample_idx_mapping = torch.empty(2048, dtype=torch.int32, device=device)
    block_table = torch.empty((4, 256), dtype=torch.int32, device=device)
    warmup = SimpleNamespace(
        query_start_loc=torch.empty(num_reqs + 1, dtype=torch.int32, device=device),
        positions=torch.empty(num_reqs, dtype=torch.int64, device=device),
    )
    served = SimpleNamespace(
        query_start_loc=torch.empty(num_reqs + 1, dtype=torch.int32, device=device),
        positions=torch.empty(served_tokens, dtype=torch.int64, device=device),
    )

    # These are the C=1 and full-chunk C=4 target views respectively.  They
    # remain runtime bounds, including on a second full-chunk iteration.
    assert _target_input_lengths(warmup) == (num_reqs + 1, num_reqs)
    assert _target_input_lengths(served) == (num_reqs + 1, served_tokens)

    # Warmup gets fresh result allocations. Serving may pass a view into a
    # persistent sampler result buffer, which deliberately differs in both
    # pointer alignment and storage offset.
    warmup_num_sampled = torch.empty(num_reqs, dtype=torch.int32, device=device)
    warmup_num_rejected = torch.empty(num_reqs, dtype=torch.int32, device=device)
    served_num_sampled = torch.empty(num_reqs + 1, dtype=torch.int32, device=device)[1:]
    served_num_rejected = torch.empty(num_reqs + 1, dtype=torch.int32, device=device)[
        1:
    ]
    assert warmup_num_sampled.storage_offset() == 0
    assert served_num_sampled.storage_offset() == 1

    kwargs = dict(
        num_reqs=num_reqs,
        k=8,
        state_capacity=4,
        block_size=16,
        max_model_len=4096,
        noise_seed=0,
        noise_high=151_669,
        has_rejected=True,
        block=256,
    )
    warmup_specialization = prepare_uno_launch_key(
        buffers,
        slot_mapping,
        sample_idx_mapping,
        warmup,
        warmup_num_sampled,
        warmup_num_rejected,
        block_table,
        **kwargs,
    )
    served_specialization = prepare_uno_launch_key(
        buffers,
        slot_mapping,
        sample_idx_mapping,
        served,
        served_num_sampled,
        served_num_rejected,
        block_table,
        **kwargs,
    )

    assert warmup_specialization == served_specialization
    assert "TARGET_QUERY_CAP" not in dict(warmup_specialization)
    assert "TARGET_POSITION_CAP" not in dict(warmup_specialization)
    import vllm.v1.worker.gpu.spec_decode.uno_prepare as uno_prepare

    assert (
        'do_not_specialize_on_alignment=["num_sampled_ptr", "num_rejected_ptr"]'
        in inspect.getsource(uno_prepare)
    )


def test_fused_uno_rejects_non_native_last_sampled_layout_on_cpu():
    case = _batch(4, "reorder", torch.device("cpu"))
    case.last_sampled = torch.zeros(5, 2, dtype=torch.int64)
    with pytest.raises(ValueError, match="last_sampled"):
        prepare_uno_inputs_fused(
            case.buffers,
            case.slot_mapping,
            case.sample_idx_mapping,
            case.input_batch,
            case.num_sampled,
            case.num_rejected,
            case.last_sampled,
            case.next_prefill_tokens,
            case.seeds,
            case.block_table,
            4,
            4,
            case.max_model_len,
            29,
            100_003,
            37,
        )


def test_fused_uno_rejects_non_native_next_prefill_layout_on_cpu():
    case = _batch(4, "reorder", torch.device("cpu"))
    case.next_prefill_tokens = torch.zeros(5, 2, dtype=torch.int32)
    with pytest.raises(ValueError, match="next_prefill_tokens"):
        prepare_uno_inputs_fused(
            case.buffers,
            case.slot_mapping,
            case.sample_idx_mapping,
            case.input_batch,
            case.num_sampled,
            case.num_rejected,
            case.last_sampled,
            case.next_prefill_tokens,
            case.seeds,
            case.block_table,
            4,
            4,
            case.max_model_len,
            29,
            100_003,
            37,
        )
