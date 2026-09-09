# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.platforms import current_platform
from vllm.v1.watermarking import (
    DualKeyGumbelWatermarkDetector,
    GumbelWatermarkDetector,
    GumbelWatermarker,
    derive_watermark_key,
)
from vllm.v1.watermarking.gumbel import _gamma_survival_integer_shape
from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample
from vllm.v1.worker.gpu.sample.watermark import philox_gumbel_sample


def test_gamma_survival_integer_shape():
    assert _gamma_survival_integer_shape(0.0, 4) == 1.0
    assert _gamma_survival_integer_shape(2.0, 1) == pytest.approx(0.1353352832)


def test_detector_deduplicates_repeated_prf_inputs_by_default():
    detection = GumbelWatermarkDetector(key=42, context_width=1).detect([1, 1, 1, 1, 1])

    assert detection.num_scored_tokens == 2


def test_detector_can_score_repeated_prf_inputs():
    detection = GumbelWatermarkDetector(
        key=42, context_width=1, deduplicate_contexts=False
    ).detect([1, 1, 1, 1, 1])

    assert detection.num_scored_tokens == 5


def test_detector_deduplicates_context_even_when_target_differs():
    detection = GumbelWatermarkDetector(key=42, context_width=1).detect([1, 2, 1, 3])

    assert detection.num_scored_tokens == 3


def test_dual_key_detector_scores_each_token_against_both_keys():
    token_ids = [1, 2, 3, 4, 5]
    key_a_detector = GumbelWatermarkDetector(
        key=derive_watermark_key(42, b"key_a"),
        context_width=1,
        deduplicate_contexts=False,
    )
    key_b_detector = GumbelWatermarkDetector(
        key=derive_watermark_key(42, b"key_b"),
        context_width=1,
        deduplicate_contexts=False,
    )
    detector = DualKeyGumbelWatermarkDetector(
        key=42,
        context_width=1,
        deduplicate_contexts=False,
    )

    key_a = key_a_detector.detect(token_ids)
    key_b = key_b_detector.detect(token_ids)
    dual = detector.detect(token_ids)

    assert dual.score == pytest.approx((key_a.score + key_b.score) / 2)
    assert dual.p_value == pytest.approx(
        _gamma_survival_integer_shape(dual.score * 2, dual.num_scored_tokens * 2)
    )


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(), reason="requires a CUDA-like accelerator"
)
@pytest.mark.parametrize("key", [1, 2**32, 42, 15726070495360670683])
@pytest.mark.parametrize("context_width", [1, 4, 16])
def test_fused_watermarker_matches_cpu(key: int, context_width: int):
    torch.manual_seed(0)
    contexts = torch.randint(0, 248320, (32, context_width), dtype=torch.int64)
    logits = torch.randn(32, 8193)
    watermarker = GumbelWatermarker(key, context_width)

    expected = watermarker.sample(logits, contexts).token_ids
    actual = watermarker.sample(logits.cuda(), contexts.cuda()).token_ids.cpu()

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(), reason="requires a CUDA-like accelerator"
)
def test_fused_watermarker_handles_nan_logits():
    contexts = torch.zeros((2, 4), dtype=torch.int64, device="cuda")
    logits = torch.full((2, 1025), float("nan"), device="cuda")
    watermarker = GumbelWatermarker(key=42, context_width=4)

    token_ids = watermarker.sample(logits, contexts).token_ids

    assert torch.all((token_ids >= 0) & (token_ids < logits.shape[-1]))


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(), reason="requires a CUDA-like accelerator"
)
def test_fused_watermarker_handles_noncontiguous_inputs():
    contexts = torch.randint(0, 248320, (8, 8), dtype=torch.int64, device="cuda")[
        :, ::2
    ]
    logits = torch.randn(8, 2050, device="cuda")[:, ::2]
    watermarker = GumbelWatermarker(key=42, context_width=4)

    expected = watermarker.sample(logits.contiguous(), contexts.contiguous()).token_ids
    actual = watermarker.sample(logits, contexts).token_ids

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(), reason="requires a CUDA-like accelerator"
)
@pytest.mark.parametrize("use_fp64", [False, True])
def test_philox_gumbel_sample_skip_mask_matches_separate_samplers(use_fp64: bool):
    torch.manual_seed(0)
    logits = torch.randn(8, 8193, device="cuda")
    context_storage = torch.randint(0, 248320, (8, 8), dtype=torch.int64, device="cuda")
    contexts = context_storage[:, ::2]
    repeated_mask = torch.tensor(
        [False, False, False, True, False, False, True, False], device="cuda"
    )
    watermarking = torch.tensor(
        [True, False, True, True, False, True, True, False], device="cuda"
    )
    req_indices = torch.tensor([3, 1, 7, 0, 4, 2, 6, 5], device="cuda")
    temperatures = torch.ones(8, dtype=torch.float32, device="cuda")
    temperatures[2] = 0
    seeds = torch.arange(8, dtype=torch.int64, device="cuda") + 1000
    positions = torch.arange(8, dtype=torch.int64, device="cuda") + 100
    watermark_mask = (
        watermarking[req_indices] & (temperatures[req_indices] != 0) & ~repeated_mask
    )

    watermarked = philox_gumbel_sample(logits, contexts, 42)
    ordinary = gumbel_sample(
        logits,
        req_indices,
        temperatures,
        seeds,
        positions,
        apply_temperature=False,
        is_drafting=False,
        use_fp64=use_fp64,
    )
    expected = torch.where(watermark_mask, watermarked, ordinary)
    actual = philox_gumbel_sample(
        logits,
        contexts,
        42,
        skip_mask=~watermark_mask,
        expanded_idx_mapping=req_indices,
        temperatures=temperatures,
        seeds=seeds,
        positions=positions,
        use_fp64=use_fp64,
    )

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
