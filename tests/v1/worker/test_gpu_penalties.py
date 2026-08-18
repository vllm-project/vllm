# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.platforms import current_platform

pytest.importorskip("triton")
if current_platform.device_count() == 0:
    pytest.skip(
        "Accelerator required for Model Runner V2 penalty tests",
        allow_module_level=True,
    )

from vllm.v1.worker.gpu.input_batch import post_update
from vllm.v1.worker.gpu.sample.penalties import (
    PenaltiesState,
    apply_penalties,
    bincount,
)
from vllm.v1.worker.gpu.states import RequestState

DEVICE = torch.device(current_platform.device_type)


@pytest.mark.parametrize(
    ("max_model_len", "expected_dtype", "expected_width"),
    [
        (65_535, torch.uint16, 6),
        (65_536, torch.int32, 5),
    ],
)
def test_output_counts_use_narrowest_exact_dtype(
    max_model_len: int, expected_dtype: torch.dtype, expected_width: int
):
    req_states = RequestState(
        max_num_reqs=1,
        max_model_len=max_model_len,
        max_num_batched_tokens=1,
        num_speculative_steps=0,
        vocab_size=5,
        device=DEVICE,
    )

    state = PenaltiesState(req_states)

    assert state.output_bin_counts.dtype == expected_dtype
    assert state.output_bin_counts.shape == (1, expected_width)


@pytest.mark.parametrize(
    ("count0", "count1"),
    [(65_535, 0), (0, 65_535), (40_000, 7), (7, 65_528), (32_767, 32_768)],
)
def test_uint16_bincount_keeps_adjacent_counters_independent(count0: int, count1: int):
    token_ids = torch.tensor([0] * count0 + [1] * count1, device=DEVICE)
    all_token_ids = token_ids.unsqueeze(0).to(torch.int32)
    prompt_bin_mask = torch.zeros((1, 1), dtype=torch.int32, device=DEVICE)
    output_bin_counts = torch.zeros((1, 6), dtype=torch.uint16, device=DEVICE)
    output_bin_counts_int32 = torch.zeros((1, 5), dtype=torch.int32, device=DEVICE)

    for counts in (output_bin_counts, output_bin_counts_int32):
        bincount(
            expanded_idx_mapping=torch.tensor([0], dtype=torch.int32, device=DEVICE),
            all_token_ids=all_token_ids,
            prompt_len=torch.tensor([0], dtype=torch.int32, device=DEVICE),
            prefill_len=torch.tensor(
                [token_ids.numel()], dtype=torch.int32, device=DEVICE
            ),
            prompt_bin_mask=prompt_bin_mask,
            output_bin_counts=counts,
            max_prefill_len=token_ids.numel(),
        )

    expected = torch.tensor([count0, count1, 0, 0, 0, 0], dtype=torch.int64)
    torch.testing.assert_close(output_bin_counts.cpu().to(torch.int64)[0], expected)
    torch.testing.assert_close(
        output_bin_counts_int32.cpu().to(torch.int64)[0], expected[:5]
    )


def test_uint16_bincount_handles_odd_vocab_last_token():
    output_bin_counts = torch.zeros((1, 6), dtype=torch.uint16, device=DEVICE)

    bincount(
        expanded_idx_mapping=torch.tensor([0], dtype=torch.int32, device=DEVICE),
        all_token_ids=torch.tensor([[4, 4, 4]], dtype=torch.int32, device=DEVICE),
        prompt_len=torch.tensor([0], dtype=torch.int32, device=DEVICE),
        prefill_len=torch.tensor([3], dtype=torch.int32, device=DEVICE),
        prompt_bin_mask=torch.zeros((1, 1), dtype=torch.int32, device=DEVICE),
        output_bin_counts=output_bin_counts,
        max_prefill_len=3,
    )

    expected = torch.tensor([0, 0, 0, 0, 3, 0], dtype=torch.int64)
    torch.testing.assert_close(output_bin_counts.cpu().to(torch.int64)[0], expected)


def test_uint16_bincount_clears_only_reused_rows():
    output_bin_counts = torch.tensor(
        [[9, 8, 7, 6, 5, 4], [6, 5, 4, 3, 2, 1]],
        dtype=torch.uint16,
        device=DEVICE,
    )
    prompt_bin_mask = torch.tensor(
        [[0x12345], [0x54321]], dtype=torch.int32, device=DEVICE
    )

    bincount(
        expanded_idx_mapping=torch.tensor([1], dtype=torch.int32, device=DEVICE),
        all_token_ids=torch.tensor(
            [[4, 4, 4], [0, 1, 1]], dtype=torch.int32, device=DEVICE
        ),
        prompt_len=torch.tensor([0, 0], dtype=torch.int32, device=DEVICE),
        prefill_len=torch.tensor([3, 3], dtype=torch.int32, device=DEVICE),
        prompt_bin_mask=prompt_bin_mask,
        output_bin_counts=output_bin_counts,
        max_prefill_len=3,
    )

    expected_counts = torch.tensor(
        [[9, 8, 7, 6, 5, 4], [1, 2, 0, 0, 0, 0]], dtype=torch.int64
    )
    torch.testing.assert_close(output_bin_counts.cpu().to(torch.int64), expected_counts)
    torch.testing.assert_close(
        prompt_bin_mask.cpu(), torch.tensor([[0x12345], [0]], dtype=torch.int32)
    )


def test_uint16_penalties_match_int32():
    vocab_size = 257
    num_tokens = 6
    logits = torch.linspace(
        -3.0, 3.0, steps=num_tokens * vocab_size, device=DEVICE
    ).reshape(num_tokens, vocab_size)
    logits_uint16 = logits.clone()
    logits_int32 = logits.clone()
    counts_int32 = (
        torch.arange(2 * vocab_size, dtype=torch.int32, device=DEVICE)
        .reshape(2, vocab_size)
        .remainder(101)
    )
    counts_uint16 = torch.zeros((2, 258), dtype=torch.uint16, device=DEVICE)
    counts_uint16[:, :vocab_size] = counts_int32
    prompt_bin_mask = torch.tensor(
        [[0x10101010] * 9, [0x01010101] * 9],
        dtype=torch.int32,
        device=DEVICE,
    )
    expanded_idx_mapping = torch.tensor(
        [0, 0, 0, 1, 1, 1], dtype=torch.int32, device=DEVICE
    )
    token_ids = torch.tensor([3, 5, 7, 11, 13, 17], dtype=torch.int32, device=DEVICE)
    local_pos = torch.tensor([0, 1, 2, 0, 1, 2], dtype=torch.int32, device=DEVICE)
    repetition = torch.tensor([1.1, 0.9], device=DEVICE)
    frequency = torch.tensor([0.5, -0.3], device=DEVICE)
    presence = torch.tensor([0.8, -0.6], device=DEVICE)

    for current_logits, counts in (
        (logits_uint16, counts_uint16),
        (logits_int32, counts_int32),
    ):
        apply_penalties(
            current_logits,
            expanded_idx_mapping,
            token_ids,
            local_pos,
            repetition,
            frequency,
            presence,
            prompt_bin_mask,
            counts,
        )

    torch.testing.assert_close(logits_uint16, logits_int32, rtol=0, atol=0)


def test_penalties_read_uint16_counts_without_truncation():
    vocab_size = 5
    logits = torch.zeros((1, vocab_size), device=DEVICE)
    output_bin_counts = torch.tensor(
        [[40_000, 7, 0, 0, 0, 0]], dtype=torch.uint16, device=DEVICE
    )

    apply_penalties(
        logits=logits,
        expanded_idx_mapping=torch.tensor([0], dtype=torch.int32, device=DEVICE),
        token_ids=torch.tensor([0], dtype=torch.int32, device=DEVICE),
        expanded_local_pos=torch.tensor([0], dtype=torch.int32, device=DEVICE),
        repetition_penalty=torch.tensor([1.0], device=DEVICE),
        frequency_penalty=torch.tensor([0.0001], device=DEVICE),
        presence_penalty=torch.tensor([0.0], device=DEVICE),
        prompt_bin_mask=torch.zeros((1, 1), dtype=torch.int32, device=DEVICE),
        output_bin_counts=output_bin_counts,
    )

    expected = torch.tensor([[-4.0, -0.0007, 0.0, 0.0, 0.0]])
    torch.testing.assert_close(logits.cpu(), expected)


def test_post_update_increments_uint16_counts():
    output_bin_counts = torch.tensor(
        [[0, 40_000, 0, 0]], dtype=torch.uint16, device=DEVICE
    )
    output_bin_counts_int32 = output_bin_counts.to(torch.int32)

    for counts in (output_bin_counts, output_bin_counts_int32):
        post_update(
            idx_mapping=torch.tensor([0], dtype=torch.int32, device=DEVICE),
            num_computed_tokens=torch.zeros(1, dtype=torch.int32, device=DEVICE),
            last_sampled_tokens=torch.zeros((1, 1), dtype=torch.int64, device=DEVICE),
            output_bin_counts=counts,
            sampled_tokens=torch.tensor([[1, 2, 1]], dtype=torch.int64, device=DEVICE),
            num_sampled=torch.tensor([3], dtype=torch.int32, device=DEVICE),
            num_rejected=torch.tensor([0], dtype=torch.int32, device=DEVICE),
            query_start_loc=None,
            all_token_ids=torch.zeros((1, 8), dtype=torch.int32, device=DEVICE),
            total_len=torch.zeros(1, dtype=torch.int32, device=DEVICE),
        )

    expected = torch.tensor([0, 40_002, 1, 0], dtype=torch.int64)
    torch.testing.assert_close(output_bin_counts.cpu().to(torch.int64)[0], expected)
    torch.testing.assert_close(
        output_bin_counts_int32.cpu().to(torch.int64)[0], expected
    )
