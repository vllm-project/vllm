# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.v1.worker.gpu.sample.no_repeat_ngram import apply_no_repeat_ngram


def _reference_banned_tokens(output_ids: list[int], ngram_size: int) -> set[int]:
    if ngram_size < 1 or len(output_ids) < ngram_size:
        return set()
    if ngram_size == 1:
        return set(output_ids)
    prefix = output_ids[-(ngram_size - 1) :]
    return {
        output_ids[start + ngram_size - 1]
        for start in range(len(output_ids) - ngram_size + 1)
        if output_ids[start : start + ngram_size - 1] == prefix
    }


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
    mapping = torch.arange(batch_size, dtype=torch.int32, device=device)
    logits = torch.zeros((batch_size, vocab_size), device=device)

    apply_no_repeat_ngram(
        logits,
        mapping,
        all_token_ids,
        prompt_lens,
        total_lens,
        sizes,
    )
    torch.cuda.synchronize()

    for row, ngram_size in enumerate(ngram_sizes):
        expected = _reference_banned_tokens(outputs[row].cpu().tolist(), ngram_size)
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
    mapping = torch.tensor([1], dtype=torch.int32, device=device)
    logits = torch.zeros((1, 8), device=device)

    apply_no_repeat_ngram(
        logits,
        mapping,
        all_token_ids,
        prompt_lens,
        total_lens,
        sizes,
    )
    torch.cuda.synchronize()
    assert torch.isneginf(logits[0, 6])
    assert torch.isfinite(logits[0, 3])
