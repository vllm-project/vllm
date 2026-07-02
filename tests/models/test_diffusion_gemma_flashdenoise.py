# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math

import pytest
import torch

import vllm._custom_ops as ops


def _splitmix64(x: int) -> int:
    mask = (1 << 64) - 1
    x = (x + 0x9E3779B97F4A7C15) & mask
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & mask
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & mask
    return (x ^ (x >> 31)) & mask


def _uniform01(seed: int, offset: int, row: int, vocab_idx: int) -> float:
    mask = (1 << 64) - 1
    key = seed & mask
    key ^= ((offset & mask) * 0x9E3779B97F4A7C15) & mask
    key ^= (row * 0xBF58476D1CE4E5B9) & mask
    key ^= (vocab_idx * 0x94D049BB133111EB) & mask
    z = _splitmix64(key)
    mantissa = (z >> 40) & 0xFFFFFF
    return (mantissa + 0.5) * (2.0**-24)


def _reference_from_logits(
    logits: torch.Tensor,
    lm_head_weight: torch.Tensor,
    normalizer: float,
    rng_seed: int,
    rng_offset: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    logits_cpu = logits.detach().cpu().float()
    weight_cpu = lm_head_weight.detach().cpu().float()
    rows, vocab_size = logits_cpu.shape

    noisy_logits = logits_cpu.clone()
    for row in range(rows):
        for vocab_idx in range(vocab_size):
            u = _uniform01(rng_seed, rng_offset, row, vocab_idx)
            noisy_logits[row, vocab_idx] += -math.log(-math.log(u))

    sample_values, sample_indices = noisy_logits.max(dim=-1)
    clean_values, clean_indices = logits_cpu.max(dim=-1)
    log_probs = logits_cpu.log_softmax(dim=-1)
    probs = log_probs.exp()
    entropy = -(probs * log_probs).sum(dim=-1)
    soft_embed = torch.matmul(probs.to(torch.bfloat16).float(), weight_cpu)
    return (
        entropy,
        sample_values,
        sample_indices,
        clean_values,
        clean_indices,
        soft_embed * normalizer,
    )


def test_flashdenoise_python_wrappers_are_available():
    assert hasattr(ops, "diffusion_gemma_flashdenoise")
    assert hasattr(ops, "diffusion_gemma_flashdenoise_scaled")


def test_flashdenoise_wrapper_reports_unavailable_without_native_op():
    if hasattr(torch.ops, "_C") and hasattr(
        torch.ops._C, "diffusion_gemma_flashdenoise"
    ):
        pytest.skip("native FlashDenoise op is registered")

    tensor = torch.empty(1)
    with pytest.raises(RuntimeError, match="native op is unavailable"):
        ops.diffusion_gemma_flashdenoise(
            tensor,
            tensor,
            torch.empty(1, dtype=torch.int64),
            tensor,
            torch.empty(1, dtype=torch.int64),
            torch.empty(1, 1),
            torch.empty(1, 1),
            torch.empty(1, 1),
            1.0,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_native_flashdenoise_mode16_matches_dense_reference():
    if not hasattr(torch.ops, "_C") or not hasattr(
        torch.ops._C, "diffusion_gemma_flashdenoise"
    ):
        pytest.skip("native FlashDenoise op is unavailable")

    rows, hidden_size, vocab_size = 19, 64, 512
    normalizer = 0.75
    rng_seed = 24680
    rng_offset = 11
    generator = torch.Generator(device="cuda").manual_seed(17)
    hidden = (
        torch.randn(
            rows,
            hidden_size,
            device="cuda",
            dtype=torch.float32,
            generator=generator,
        )
        * 0.125
    ).to(torch.bfloat16)
    lm_head_weight = (
        torch.randn(
            vocab_size,
            hidden_size,
            device="cuda",
            dtype=torch.float32,
            generator=generator,
        )
        * 0.125
    ).to(torch.bfloat16)

    entropy = torch.empty(rows, device="cuda", dtype=torch.float32)
    sample_values = torch.empty(rows, device="cuda", dtype=torch.float32)
    sample_indices = torch.empty(rows, device="cuda", dtype=torch.int64)
    clean_values = torch.empty(rows, device="cuda", dtype=torch.float32)
    clean_indices = torch.empty(rows, device="cuda", dtype=torch.int64)
    soft_embed = torch.empty(rows, hidden_size, device="cuda", dtype=torch.float32)

    ops.diffusion_gemma_flashdenoise(
        entropy,
        sample_values,
        sample_indices,
        clean_values,
        clean_indices,
        soft_embed,
        hidden,
        lm_head_weight,
        normalizer,
        mode_flags=16,
        rng_seed=rng_seed,
        rng_offset=rng_offset,
    )
    torch.cuda.synchronize()

    dense_logits = torch.mm(hidden, lm_head_weight.t(), out_dtype=torch.float32)
    expected = _reference_from_logits(
        dense_logits,
        lm_head_weight,
        normalizer,
        rng_seed,
        rng_offset,
    )

    torch.testing.assert_close(sample_indices.cpu(), expected[2])
    torch.testing.assert_close(clean_indices.cpu(), expected[4])
    torch.testing.assert_close(sample_values.cpu(), expected[1], atol=2e-3, rtol=0)
    torch.testing.assert_close(clean_values.cpu(), expected[3], atol=2e-3, rtol=0)
    torch.testing.assert_close(entropy.cpu(), expected[0], atol=2e-3, rtol=0)
    torch.testing.assert_close(soft_embed.cpu(), expected[5], atol=2e-3, rtol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_native_flashdenoise_scaled_mode16_matches_dense_reference():
    if not hasattr(torch.ops, "_C") or not hasattr(
        torch.ops._C, "diffusion_gemma_flashdenoise_scaled"
    ):
        pytest.skip("native scaled FlashDenoise op is unavailable")

    rows, hidden_size, vocab_size = 13, 64, 512
    normalizer = 0.75
    rng_seed = 13579
    rng_offset = 23
    generator = torch.Generator(device="cuda").manual_seed(29)
    hidden = (
        torch.randn(
            rows,
            hidden_size,
            device="cuda",
            dtype=torch.float32,
            generator=generator,
        )
        * 0.125
    ).to(torch.bfloat16)
    lm_head_weight = (
        torch.randn(
            vocab_size,
            hidden_size,
            device="cuda",
            dtype=torch.float32,
            generator=generator,
        )
        * 0.125
    ).to(torch.bfloat16)
    logit_scale = torch.linspace(
        0.5, 1.75, rows, device="cuda", dtype=torch.float32
    )

    entropy = torch.empty(rows, device="cuda", dtype=torch.float32)
    sample_values = torch.empty(rows, device="cuda", dtype=torch.float32)
    sample_indices = torch.empty(rows, device="cuda", dtype=torch.int64)
    clean_values = torch.empty(rows, device="cuda", dtype=torch.float32)
    clean_indices = torch.empty(rows, device="cuda", dtype=torch.int64)
    soft_embed = torch.empty(rows, hidden_size, device="cuda", dtype=torch.float32)

    ops.diffusion_gemma_flashdenoise_scaled(
        entropy,
        sample_values,
        sample_indices,
        clean_values,
        clean_indices,
        soft_embed,
        hidden,
        lm_head_weight,
        logit_scale,
        normalizer,
        mode_flags=16,
        rng_seed=rng_seed,
        rng_offset=rng_offset,
    )
    torch.cuda.synchronize()

    dense_logits = torch.mm(hidden, lm_head_weight.t(), out_dtype=torch.float32)
    expected = _reference_from_logits(
        dense_logits * logit_scale.unsqueeze(-1),
        lm_head_weight,
        normalizer,
        rng_seed,
        rng_offset,
    )

    torch.testing.assert_close(sample_indices.cpu(), expected[2])
    torch.testing.assert_close(clean_indices.cpu(), expected[4])
    torch.testing.assert_close(sample_values.cpu(), expected[1], atol=2e-3, rtol=0)
    torch.testing.assert_close(clean_values.cpu(), expected[3], atol=2e-3, rtol=0)
    torch.testing.assert_close(entropy.cpu(), expected[0], atol=2e-3, rtol=0)
    torch.testing.assert_close(soft_embed.cpu(), expected[5], atol=2e-3, rtol=2e-2)
