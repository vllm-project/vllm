# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
from transformers import SynthIDTextWatermarkLogitsProcessor

from vllm.v1.watermarking.synthid import SynthIDWatermarker

NGRAM_LEN = 5
KEYS = [654, 400, 836, 123, 340, 443, 597, 160, 57]
SAMPLING_TABLE_SIZE = 10007
SAMPLING_TABLE_SEED = 123


@pytest.mark.parametrize(
    ("batch_size", "vocab_size", "ngram_len", "keys"),
    [
        (1, 32, 2, [11]),
        (2, 128, 5, KEYS),
        (4, 257, 3, [7, 13, 29]),
    ],
)
def test_synthid_logits_match_huggingface(
    batch_size,
    vocab_size,
    ngram_len,
    keys,
):
    device = torch.device("cpu")
    reference = SynthIDTextWatermarkLogitsProcessor(
        ngram_len=ngram_len,
        keys=keys,
        sampling_table_size=SAMPLING_TABLE_SIZE,
        sampling_table_seed=SAMPLING_TABLE_SEED,
        context_history_size=1024,
        device=device,
    )

    watermarker = SynthIDWatermarker(
        ngram_len=ngram_len,
        keys=keys,
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
    contexts = torch.randint(
        0,
        vocab_size,
        (batch_size, ngram_len - 1),
        generator=generator,
    )

    logits = torch.randn(
        batch_size,
        vocab_size,
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


def test_synthid_sample_uses_watermarked_logits():
    watermarker = SynthIDWatermarker(
        ngram_len=NGRAM_LEN,
        keys=KEYS,
        sampling_table_size=SAMPLING_TABLE_SIZE,
        sampling_table_seed=SAMPLING_TABLE_SEED,
    )

    logits = torch.randn(2, 128)
    contexts = torch.tensor(
        [
            [10, 20, 30, 40],
            [50, 60, 70, 80],
        ],
        dtype=torch.long,
    )

    expected_logits = watermarker.watermark_logits(
        logits,
        contexts,
    )

    sampled_logits = None

    def random_sampler(scores):
        nonlocal sampled_logits
        sampled_logits = scores
        return scores.argmax(dim=-1)

    result = watermarker.sample(
        logits,
        contexts,
        random_sampler,
    )

    assert sampled_logits is not None
    torch.testing.assert_close(
        sampled_logits,
        expected_logits,
    )
    torch.testing.assert_close(
        result.logits,
        expected_logits,
    )
    assert torch.equal(
        result.token_ids,
        expected_logits.argmax(dim=-1),
    )


def test_synthid_sample_respects_skip_mask():
    watermarker = SynthIDWatermarker(
        ngram_len=NGRAM_LEN,
        keys=KEYS,
        sampling_table_size=SAMPLING_TABLE_SIZE,
        sampling_table_seed=SAMPLING_TABLE_SEED,
    )

    logits = torch.randn(2, 128)
    contexts = torch.tensor(
        [
            [10, 20, 30, 40],
            [50, 60, 70, 80],
        ],
        dtype=torch.long,
    )
    skip_mask = torch.tensor([False, True])

    watermarked_logits = watermarker.watermark_logits(
        logits,
        contexts,
    )

    sampled_logits = None

    def random_sampler(scores):
        nonlocal sampled_logits
        sampled_logits = scores
        return scores.argmax(dim=-1)

    watermarker.sample(
        logits,
        contexts,
        random_sampler,
        skip_mask,
    )

    assert sampled_logits is not None

    # Watermark enabled for row 0.
    torch.testing.assert_close(
        sampled_logits[0],
        watermarked_logits[0],
    )

    # Watermark skipped for row 1.
    torch.testing.assert_close(
        sampled_logits[1],
        logits[1],
    )


def test_synthid_sample_requires_random_sampler():
    watermarker = SynthIDWatermarker(
        ngram_len=NGRAM_LEN,
        keys=KEYS,
        sampling_table_size=SAMPLING_TABLE_SIZE,
        sampling_table_seed=SAMPLING_TABLE_SEED,
    )

    logits = torch.randn(1, 128)
    contexts = torch.tensor(
        [[10, 20, 30, 40]],
        dtype=torch.long,
    )

    with pytest.raises(
        ValueError,
        match="SynthID requires a random sampler",
    ):
        watermarker.sample(
            logits,
            contexts,
        )


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"ngram_len": 1}, "ngram_len must be at least 2"),
        ({"keys": []}, "keys must not be empty"),
        (
            {"sampling_table_size": 0},
            "sampling_table_size must be positive",
        ),
    ],
)
def test_synthid_rejects_invalid_config(kwargs, match):
    config = {
        "ngram_len": NGRAM_LEN,
        "keys": KEYS,
        "sampling_table_size": SAMPLING_TABLE_SIZE,
        "sampling_table_seed": SAMPLING_TABLE_SEED,
    }
    config.update(kwargs)

    with pytest.raises(ValueError, match=match):
        SynthIDWatermarker(**config)
