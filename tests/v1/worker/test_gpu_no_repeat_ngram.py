# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.model_executor.models.deepseek_ocr import NoRepeatNGramLogitsProcessor
from vllm.sampling_params import SamplingParams
from vllm.v1.worker.gpu.sample.no_repeat_ngram import (
    NoRepeatNGramState,
    apply_no_repeat_ngram,
)


def _reference_banned_tokens(
    output_ids: list[int],
    ngram_size: int,
    window_size: int,
    whitelist: set[int] | None = None,
) -> set[int]:
    if ngram_size < 1 or len(output_ids) < ngram_size:
        return set()
    if ngram_size == 1:
        search_start = max(0, len(output_ids) - window_size)
        return set(output_ids[search_start:]) - (whitelist or set())
    prefix = output_ids[-(ngram_size - 1) :]
    search_start = max(0, len(output_ids) - window_size)
    banned = {
        output_ids[start + ngram_size - 1]
        for start in range(search_start, len(output_ids) - ngram_size + 1)
        if output_ids[start : start + ngram_size - 1] == prefix
    }
    return banned - (whitelist or set())


@pytest.mark.parametrize("ngram_sizes", [[1, 2, 3, 0], [5, 7, 11, 17]])
def test_no_repeat_ngram_matches_reference(ngram_sizes: list[int]):
    torch.manual_seed(17)
    device = "cuda"
    batch_size = len(ngram_sizes)
    prompt_len = 3
    output_len = 192
    vocab_size = 31
    max_model_len = prompt_len + output_len

    # A small vocabulary deliberately creates repeated prefixes. Prompt tokens
    # use an out-of-vocabulary sentinel so the test also checks output-only
    # semantics.
    outputs = torch.randint(
        0,
        vocab_size,
        (batch_size, output_len),
        dtype=torch.int32,
        device=device,
    )
    all_token_ids = torch.full(
        (batch_size, max_model_len),
        vocab_size + 1,
        dtype=torch.int32,
        device=device,
    )
    all_token_ids[:, prompt_len:] = outputs
    prompt_lens = torch.full(
        (batch_size,), prompt_len, dtype=torch.int32, device=device
    )
    total_lens = torch.full(
        (batch_size,), max_model_len, dtype=torch.int32, device=device
    )
    sizes = torch.tensor(ngram_sizes, dtype=torch.int32, device=device)
    window_sizes = torch.full(
        (batch_size,), output_len, dtype=torch.int32, device=device
    )
    whitelist_ids = torch.zeros((batch_size, 1), dtype=torch.int32, device=device)
    whitelist_lens = torch.zeros(batch_size, dtype=torch.int32, device=device)
    mapping = torch.arange(batch_size, dtype=torch.int32, device=device)
    logits = torch.zeros((batch_size, vocab_size), device=device)

    apply_no_repeat_ngram(
        logits,
        mapping,
        all_token_ids,
        prompt_lens,
        total_lens,
        sizes,
        window_sizes,
        whitelist_ids,
        whitelist_lens,
    )
    torch.accelerator.synchronize()

    for row, ngram_size in enumerate(ngram_sizes):
        expected = _reference_banned_tokens(
            outputs[row].cpu().tolist(), ngram_size, output_len
        )
        actual = set(torch.isneginf(logits[row]).nonzero().flatten().cpu().tolist())
        assert actual == expected


def test_no_repeat_ngram_uses_request_slot_mapping():
    device = "cuda"
    # Logical row zero maps to request slot one. Only that slot uses the
    # constraint, which catches accidental indexing by compacted batch row.
    all_token_ids = torch.tensor(
        [[1, 2, 3, 1, 2], [4, 5, 6, 4, 5]], dtype=torch.int32, device=device
    )
    prompt_lens = torch.zeros(2, dtype=torch.int32, device=device)
    total_lens = torch.full((2,), 5, dtype=torch.int32, device=device)
    sizes = torch.tensor([0, 3], dtype=torch.int32, device=device)
    window_sizes = torch.full((2,), 100, dtype=torch.int32, device=device)
    whitelist_ids = torch.zeros((2, 1), dtype=torch.int32, device=device)
    whitelist_lens = torch.zeros(2, dtype=torch.int32, device=device)
    mapping = torch.tensor([1], dtype=torch.int32, device=device)
    logits = torch.zeros((1, 8), device=device)

    apply_no_repeat_ngram(
        logits,
        mapping,
        all_token_ids,
        prompt_lens,
        total_lens,
        sizes,
        window_sizes,
        whitelist_ids,
        whitelist_lens,
    )
    torch.accelerator.synchronize()
    assert torch.isneginf(logits[0, 6])
    assert torch.isfinite(logits[0, 3])


def test_no_repeat_ngram_honors_window_and_whitelist():
    device = "cuda"
    all_token_ids = torch.tensor(
        [[1, 2, 1, 3, 1], [4, 5, 4, 6, 4]], dtype=torch.int32, device=device
    )
    prompt_lens = torch.zeros(2, dtype=torch.int32, device=device)
    total_lens = torch.full((2,), 5, dtype=torch.int32, device=device)
    sizes = torch.full((2,), 2, dtype=torch.int32, device=device)
    windows = torch.tensor([3, 100], dtype=torch.int32, device=device)
    whitelist_ids = torch.tensor([[0], [5]], dtype=torch.int32, device=device)
    whitelist_lens = torch.tensor([0, 1], dtype=torch.int32, device=device)
    mapping = torch.arange(2, dtype=torch.int32, device=device)
    logits = torch.zeros((2, 8), device=device)

    apply_no_repeat_ngram(
        logits,
        mapping,
        all_token_ids,
        prompt_lens,
        total_lens,
        sizes,
        windows,
        whitelist_ids,
        whitelist_lens,
    )
    torch.accelerator.synchronize()

    assert torch.isfinite(logits[0, 2])
    assert torch.isneginf(logits[0, 3])
    assert torch.isfinite(logits[1, 5])
    assert torch.isneginf(logits[1, 6])


@pytest.mark.parametrize(
    ("ngram_size", "window_size", "whitelist", "banned_token"),
    [(2, 5, set(), 10), (3, 100, {6}, 6), (5, 17, {2, 9}, 10)],
)
def test_no_repeat_ngram_matches_legacy_ocr_processor(
    ngram_size: int,
    window_size: int,
    whitelist: set[int],
    banned_token: int,
):
    device = "cuda"
    vocab_size = 13
    prefix = list(range(1, ngram_size))
    output_ids = prefix + [banned_token, 12, 11] + prefix

    expected = torch.zeros(vocab_size, device=device)
    legacy = NoRepeatNGramLogitsProcessor(
        ngram_size=ngram_size,
        window_size=window_size,
        whitelist_token_ids=whitelist,
    )
    legacy(output_ids, expected)

    all_token_ids = torch.tensor([output_ids], dtype=torch.int32, device=device)
    prompt_lens = torch.zeros(1, dtype=torch.int32, device=device)
    total_lens = torch.tensor([len(output_ids)], dtype=torch.int32, device=device)
    sizes = torch.tensor([ngram_size], dtype=torch.int32, device=device)
    windows = torch.tensor([window_size], dtype=torch.int32, device=device)
    whitelist_ids = torch.tensor(
        [sorted(whitelist) or [0]], dtype=torch.int32, device=device
    )
    whitelist_lens = torch.tensor([len(whitelist)], dtype=torch.int32, device=device)
    mapping = torch.zeros(1, dtype=torch.int32, device=device)
    actual = torch.zeros_like(expected).unsqueeze(0)

    apply_no_repeat_ngram(
        actual,
        mapping,
        all_token_ids,
        prompt_lens,
        total_lens,
        sizes,
        windows,
        whitelist_ids,
        whitelist_lens,
    )
    torch.accelerator.synchronize()

    torch.testing.assert_close(actual[0], expected)


def test_no_repeat_ngram_validates_ocr_arguments():
    NoRepeatNGramState.validate_params(SamplingParams())
    NoRepeatNGramState.validate_params(
        SamplingParams(
            extra_args={
                "ngram_size": 3,
                "window_size": 100,
                "whitelist_token_ids": [1, 2],
            }
        )
    )
    with pytest.raises(ValueError, match="Specify only one"):
        NoRepeatNGramState.validate_params(
            SamplingParams(extra_args={"ngram_size": 3, "no_repeat_ngram_size": 3})
        )
    with pytest.raises(ValueError, match="window_size"):
        NoRepeatNGramState.validate_params(
            SamplingParams(extra_args={"ngram_size": 3, "window_size": 0})
        )
    with pytest.raises(ValueError, match="whitelist_token_ids"):
        NoRepeatNGramState.validate_params(
            SamplingParams(
                extra_args={"ngram_size": 3, "whitelist_token_ids": [1, "2"]}
            )
        )
