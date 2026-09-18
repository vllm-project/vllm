# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the V2 no-repeat-ngram logits processor core."""

import numpy as np
import pytest
import torch

from vllm.model_executor.models.deepseek_ocr import (
    NGramPerReqLogitsProcessorV2,
    NoRepeatNGramLogitsProcessor,
    _ban_repeated_ngrams,
)
from vllm.sampling_params import SamplingParams

VOCAB = 64


def _reference_banned(
    output_ids: list[int],
    ngram_size: int,
    window_size: int,
    whitelist: set[int] | None,
) -> set[int]:
    logits = torch.zeros(VOCAB)
    NoRepeatNGramLogitsProcessor(ngram_size, window_size, whitelist)(output_ids, logits)
    return set(torch.nonzero(logits == -float("inf"))[:, 0].tolist())


def _run_helper(
    seqs: list[list[int]],
    prompt_lens: list[int],
    ngram_size: int,
    window_sizes: list[int],
    whitelists: list[set[int] | None],
) -> list[set[int]]:
    """Run `_ban_repeated_ngrams` on one row per slot; return per-row bans."""
    max_len = max(len(s) for s in seqs)
    all_token_ids = torch.zeros(len(seqs), max_len, dtype=torch.int64)
    for i, s in enumerate(seqs):
        all_token_ids[i, : len(s)] = torch.tensor(s)
    width = max(
        min(len(s) - p, w) - ngram_size + 1
        for s, p, w in zip(seqs, prompt_lens, window_sizes)
    )
    if any(whitelists):
        max_wl = max(len(w) for w in whitelists if w)
        whitelist_ids = torch.zeros(len(seqs), max_wl, dtype=torch.int64)
        whitelist_len = torch.zeros(len(seqs), dtype=torch.int64)
        for i, w in enumerate(whitelists):
            if w:
                whitelist_ids[i, : len(w)] = torch.tensor(sorted(w))
                whitelist_len[i] = len(w)
    else:
        whitelist_ids = whitelist_len = None
    logits = torch.zeros(len(seqs), VOCAB)
    _ban_repeated_ngrams(
        logits,
        all_token_ids,
        slots=torch.arange(len(seqs)),
        total_len=torch.tensor([len(s) for s in seqs]),
        prompt_len=torch.tensor(prompt_lens),
        ngram_size=ngram_size,
        window_size=torch.tensor(window_sizes),
        width=width,
        whitelist_ids=whitelist_ids,
        whitelist_len=whitelist_len,
    )
    return [set(torch.nonzero(row == -float("inf"))[:, 0].tolist()) for row in logits]


@pytest.mark.parametrize("ngram_size", [1, 2, 3])
@pytest.mark.parametrize("window_hi", [5, 100])
@pytest.mark.parametrize("with_whitelist", [False, True])
def test_ban_repeated_ngrams_matches_v1(
    ngram_size: int, window_hi: int, with_whitelist: bool
):
    rng = np.random.default_rng(42)
    seqs, prompt_lens, window_sizes, whitelists = [], [], [], []
    for _ in range(8):
        p = int(rng.integers(0, 10))
        out_len = int(rng.integers(ngram_size, 40))
        seqs.append(rng.integers(0, 16, size=p + out_len).tolist())
        prompt_lens.append(p)
        window_sizes.append(int(rng.integers(ngram_size, window_hi + ngram_size)))
        whitelists.append(
            set(rng.integers(0, 16, size=3).tolist()) if with_whitelist else None
        )
    banned = _run_helper(seqs, prompt_lens, ngram_size, window_sizes, whitelists)
    for i, seq in enumerate(seqs):
        ref = _reference_banned(
            seq[prompt_lens[i] :], ngram_size, window_sizes[i], whitelists[i]
        )
        assert banned[i] == ref


def test_ban_repeated_ngrams_ignores_prompt():
    # prompt=[4, 9, 9], output=[4, 5, 4]; the (4, 9) bigram in the prompt must
    # not cause 9 to be banned; only the output bigram (4, 5) bans 5.
    banned = _run_helper([[4, 9, 9, 4, 5, 4]], [3], 2, [100], [None])[0]
    assert banned == {5}


def test_validate_params():
    NGramPerReqLogitsProcessorV2.validate_params(SamplingParams())
    NGramPerReqLogitsProcessorV2.validate_params(
        SamplingParams(extra_args={"ngram_size": 2})
    )
    with pytest.raises(ValueError, match="ngram_size"):
        NGramPerReqLogitsProcessorV2.validate_params(
            SamplingParams(extra_args={"ngram_size": 0})
        )
    with pytest.raises(ValueError, match="window_size"):
        NGramPerReqLogitsProcessorV2.validate_params(
            SamplingParams(extra_args={"ngram_size": 2, "window_size": -1})
        )
    with pytest.raises(ValueError, match="whitelist_token_ids"):
        NGramPerReqLogitsProcessorV2.validate_params(
            SamplingParams(extra_args={"ngram_size": 2, "whitelist_token_ids": 5})
        )
