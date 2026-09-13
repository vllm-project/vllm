# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from transformers import SynthIDTextWatermarkLogitsProcessor

from vllm.v1.watermarking.synthid import SynthIDWatermarker

NGRAM_LEN = 5
KEYS = [654, 400, 836, 123, 340, 443, 597, 160, 57]
SAMPLING_TABLE_SIZE = 10007
SAMPLING_TABLE_SEED = 123


def test_synthid_logits_match_huggingface():
    device = torch.device("cpu")

    reference = SynthIDTextWatermarkLogitsProcessor(
        ngram_len=NGRAM_LEN,
        keys=KEYS,
        sampling_table_size=SAMPLING_TABLE_SIZE,
        sampling_table_seed=SAMPLING_TABLE_SEED,
        context_history_size=1024,
        device=device,
    )

    watermarker = SynthIDWatermarker(
        ngram_len=NGRAM_LEN,
        keys=KEYS,
        sampling_table_size=SAMPLING_TABLE_SIZE,
        sampling_table_seed=SAMPLING_TABLE_SEED,
    )

    contexts = torch.tensor(
        [
            [10, 20, 30, 40],
            [50, 60, 70, 80],
        ],
        dtype=torch.long,
    )

    generator = torch.Generator().manual_seed(42)
    logits = torch.randn(
        2,
        128,
        generator=generator,
    )

    vocab_size = logits.shape[-1]
    vocabulary = torch.arange(vocab_size)

    expanded_contexts = contexts[:, None, :].expand(
        -1,
        vocab_size,
        -1,
    )
    candidates = vocabulary[None, :, None].expand(
        contexts.shape[0],
        -1,
        -1,
    )

    ngrams = torch.cat(
        (expanded_contexts, candidates),
        dim=-1,
    )

    reference_keys = reference.compute_ngram_keys(ngrams)
    reference_g_values = reference.sample_g_values(reference_keys)
    expected = reference.update_scores(
        logits.clone(),
        reference_g_values,
    )

    actual = watermarker.watermark_logits(
        logits.clone(),
        contexts,
    )

    torch.testing.assert_close(actual, expected)
