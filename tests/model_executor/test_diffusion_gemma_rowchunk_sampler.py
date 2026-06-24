# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.models.diffusion_gemma import (
    _get_diffusion_gemma_row_chunk_size,
    _is_diffusion_gemma_hidden_state_input,
    _is_diffusion_gemma_sampler_warmup_batch,
    _materialize_diffusion_gemma_logits_from_hidden,
    _should_use_diffusion_gemma_row_chunked_sampler,
)
from vllm.model_executor.models.diffusion_gemma_rowchunk import (
    _INT64_MIX_A,
    _stable_gumbel_argmax_from_scaled,
    _stable_uniform_from_row_base,
    diffusion_gemma_softcap_row_chunked_sample_soft_embeds,
    stable_uniform_from_indices,
)


def test_diffusion_gemma_row_chunk_size_from_scratch_budget(monkeypatch):
    monkeypatch.delenv("VLLM_DIFFUSION_GEMMA_ROW_CHUNK", raising=False)
    monkeypatch.setenv("VLLM_DIFFUSION_GEMMA_ROW_CHUNK_SCRATCH_MIB", "1024")
    assert _get_diffusion_gemma_row_chunk_size(4096, 262144) == 64
    assert _get_diffusion_gemma_row_chunk_size(64, 262144) == 64

    monkeypatch.setenv("VLLM_DIFFUSION_GEMMA_ROW_CHUNK", "17")
    assert _get_diffusion_gemma_row_chunk_size(4096, 262144) == 17
    assert _get_diffusion_gemma_row_chunk_size(8, 262144) == 8


@pytest.mark.parametrize(
    ("row_chunked_input", "real_logprobs_enabled", "expected"),
    [
        (True, False, True),
        (True, True, False),
        (False, False, False),
    ],
)
def test_diffusion_gemma_row_chunked_guard(
    row_chunked_input, real_logprobs_enabled, expected
):
    assert (
        _should_use_diffusion_gemma_row_chunked_sampler(
            row_chunked_input=row_chunked_input,
            real_logprobs_enabled=real_logprobs_enabled,
        )
        is expected
    )


def test_diffusion_gemma_row_chunked_logprobs_fallback_materializes_logits():
    torch.manual_seed(0)
    hidden = torch.randn(4, 8, dtype=torch.float32)
    weight = torch.randn(11, 8, dtype=torch.float32)
    cap = 30.0

    actual = _materialize_diffusion_gemma_logits_from_hidden(hidden, weight, 7, cap)
    raw = hidden @ weight[:7].t()
    expected = torch.tanh(raw.float() / cap) * cap

    assert actual.shape == (4, 7)
    torch.testing.assert_close(actual, expected)


def test_diffusion_gemma_sampler_warmup_batch_detection():
    warmup = SimpleNamespace(num_reqs=2, req_ids=["_warmup_0_", "_warmup_1_"])
    mixed = SimpleNamespace(num_reqs=2, req_ids=["_warmup_0_", "real-req"])
    empty = SimpleNamespace(num_reqs=0, req_ids=[])

    assert _is_diffusion_gemma_sampler_warmup_batch(warmup)
    assert not _is_diffusion_gemma_sampler_warmup_batch(mixed)
    assert not _is_diffusion_gemma_sampler_warmup_batch(empty)


def test_diffusion_gemma_hidden_state_input_classification():
    hidden = torch.empty((3, 64))
    logits = torch.empty((3, 257))
    bad = torch.empty((3, 128))

    assert _is_diffusion_gemma_hidden_state_input(hidden, 257, 64)
    assert not _is_diffusion_gemma_hidden_state_input(logits, 257, 64)
    with pytest.raises(ValueError, match="expected hidden states"):
        _is_diffusion_gemma_hidden_state_input(bad, 257, 64)


def _legacy_stable_uniform_from_indices(
    row_offsets: torch.Tensor, token_offsets: torch.Tensor, seed: int
) -> torch.Tensor:
    x = (
        token_offsets[None, :].to(torch.int64)
        + (row_offsets[:, None].to(torch.int64) + 1) * -7046029254386353131
        + int(seed)
    )
    x = (x ^ (x >> 30)) * -4658895280553007687
    x = (x ^ (x >> 27)) * -7723592293110705685
    x = x ^ (x >> 31)
    mantissa = torch.bitwise_and(x, (1 << 53) - 1).to(torch.float64)
    return ((mantissa + 0.5) * (1.0 / float(1 << 53))).to(torch.float32)


def _make_inputs(
    rows: int,
    hidden_size: int,
    vocab_size: int,
    dtype: torch.dtype = torch.bfloat16,
):
    generator = torch.Generator(device="cuda").manual_seed(2026061801)
    hidden = torch.randn(
        rows,
        hidden_size,
        device="cuda",
        dtype=dtype,
        generator=generator,
    )
    weight = torch.randn(
        vocab_size,
        hidden_size,
        device="cuda",
        dtype=dtype,
        generator=generator,
    )
    return hidden, weight


def _make_embed_weight(
    vocab_size: int,
    embed_size: int,
    dtype: torch.dtype = torch.bfloat16,
):
    generator = torch.Generator(device="cuda").manual_seed(2026061802)
    return torch.randn(
        vocab_size,
        embed_size,
        device="cuda",
        dtype=dtype,
        generator=generator,
    )


def _reference_sample_soft_embeds(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    embed_weight: torch.Tensor,
    softcap: float,
    temperature: torch.Tensor,
    seed: int,
    row_seed_offsets: torch.Tensor,
):
    logits = hidden @ weight.t()
    unscaled = torch.tanh(logits.float() / softcap) * softcap
    zero_temp_rows = temperature <= 0
    scaled = unscaled / temperature.clamp(min=1e-10)[:, None]
    scaled = torch.where(zero_temp_rows[:, None], unscaled, scaled)
    lse = scaled.logsumexp(dim=-1)
    probs = scaled.softmax(dim=-1)
    entropy = lse - (probs * scaled).sum(dim=-1)
    entropy = torch.where(zero_temp_rows, torch.zeros_like(entropy), entropy)
    greedy = scaled.argmax(dim=-1)
    soft = (probs.to(embed_weight.dtype) @ embed_weight).float()
    token_offsets = torch.arange(weight.shape[0], device="cuda", dtype=torch.int64)
    uniform = stable_uniform_from_indices(row_seed_offsets, token_offsets, seed)
    uniform = uniform.clamp(
        min=torch.finfo(uniform.dtype).tiny,
        max=1.0 - torch.finfo(uniform.dtype).eps,
    )
    noisy = scaled + (-torch.log(-torch.log(uniform))) * (
        temperature > 0
    ).float()[:, None]
    sample = noisy.argmax(dim=-1)
    if zero_temp_rows.any():
        sample = torch.where(zero_temp_rows, greedy, sample)
        soft[zero_temp_rows] = embed_weight[greedy[zero_temp_rows]].float()
    return lse, entropy, sample, greedy, soft


def _materialized_gumbel_argmax_from_scaled(
    scaled: torch.Tensor,
    row_offsets: torch.Tensor,
    token_offsets: torch.Tensor,
    seed: int,
    noise_scale: float | torch.Tensor,
) -> torch.Tensor:
    uniform = stable_uniform_from_indices(row_offsets, token_offsets, seed)
    uniform = uniform.clamp(
        min=torch.finfo(uniform.dtype).tiny,
        max=1.0 - torch.finfo(uniform.dtype).eps,
    )
    uniform = -torch.log(-torch.log(uniform))
    if isinstance(noise_scale, torch.Tensor):
        uniform = uniform * noise_scale
    elif float(noise_scale) != 1.0:
        uniform = uniform * float(noise_scale)
    return (scaled + uniform).argmax(dim=-1)


def _assert_rowchunk_outputs_close(actual, expected):
    torch.testing.assert_close(actual[0], expected[0], rtol=3e-4, atol=2e-3)
    torch.testing.assert_close(actual[1], expected[1], rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(actual[2], expected[2], rtol=0, atol=0)
    torch.testing.assert_close(actual[3], expected[3], rtol=0, atol=0)
    torch.testing.assert_close(actual[4], expected[4], rtol=2e-2, atol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_stable_uniform_matches_legacy_float64_stream():
    rows = torch.tensor([0, 1_000_003, 17_000_051], device="cuda", dtype=torch.int64)
    tokens = torch.arange(1024, device="cuda", dtype=torch.int64)

    actual = stable_uniform_from_indices(rows, tokens, seed=2026061807)
    expected = _legacy_stable_uniform_from_indices(rows, tokens, seed=2026061807)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_stable_uniform_row_base_matches_inline_stream():
    rows = torch.tensor([0, 1_000_003, 17_000_051], device="cuda", dtype=torch.int64)
    tokens = torch.arange(32769, device="cuda", dtype=torch.int64)
    seed = 2026062235
    row_base = (rows[:, None] + 1) * _INT64_MIX_A + seed

    actual = _stable_uniform_from_row_base(row_base, tokens)
    expected = stable_uniform_from_indices(rows, tokens, seed=seed)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_diffusion_gemma_row_chunked_matches_materialized_reference():
    hidden, weight = _make_inputs(rows=9, hidden_size=96, vocab_size=677)
    embed_weight = _make_embed_weight(vocab_size=677, embed_size=48)
    temperature = torch.tensor(
        [0.3, 0.5, 1e-6, 1.0, 1e-9, 0.9, 0.8, 0.0, 1.1],
        device="cuda",
        dtype=torch.float32,
    )
    row_seed_offsets = torch.tensor(
        [8, 1, 7, 3, 5, 11, 13, 17, 19], device="cuda", dtype=torch.int64
    )
    token_offsets = torch.arange(weight.shape[0], device="cuda", dtype=torch.int64)

    expected = _reference_sample_soft_embeds(
        hidden,
        weight,
        embed_weight,
        softcap=30.0,
        temperature=temperature,
        seed=2026061803,
        row_seed_offsets=row_seed_offsets,
    )
    actual_uncached = diffusion_gemma_softcap_row_chunked_sample_soft_embeds(
        hidden,
        weight,
        embed_weight,
        softcap=30.0,
        temperature=temperature,
        seed=2026061803,
        row_chunk_size=4,
        row_seed_offsets=row_seed_offsets,
    )
    actual_cached = diffusion_gemma_softcap_row_chunked_sample_soft_embeds(
        hidden,
        weight,
        embed_weight,
        softcap=30.0,
        temperature=temperature,
        seed=2026061803,
        row_chunk_size=4,
        row_seed_offsets=row_seed_offsets,
        token_offsets=token_offsets,
    )

    for actual in (actual_uncached, actual_cached):
        torch.testing.assert_close(actual[0], expected[0], rtol=3e-4, atol=2e-3)
        torch.testing.assert_close(actual[1], expected[1], rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(actual[2], expected[2], rtol=0, atol=0)
        torch.testing.assert_close(actual[3], expected[3], rtol=0, atol=0)
        torch.testing.assert_close(actual[4], expected[4], rtol=2e-2, atol=2e-2)

    for cached_item, uncached_item in zip(actual_cached, actual_uncached):
        torch.testing.assert_close(cached_item, uncached_item, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_diffusion_gemma_row_chunked_token_offsets_validation():
    hidden, weight = _make_inputs(rows=2, hidden_size=16, vocab_size=17)
    embed_weight = _make_embed_weight(vocab_size=17, embed_size=8)
    temperature = torch.ones(2, device="cuda", dtype=torch.float32)

    with pytest.raises(ValueError, match="shape"):
        diffusion_gemma_softcap_row_chunked_sample_soft_embeds(
            hidden,
            weight,
            embed_weight,
            softcap=30.0,
            temperature=temperature,
            seed=2026061821,
            row_chunk_size=2,
            token_offsets=torch.arange(16, device="cuda", dtype=torch.int64),
        )

    with pytest.raises(ValueError, match="int64"):
        diffusion_gemma_softcap_row_chunked_sample_soft_embeds(
            hidden,
            weight,
            embed_weight,
            softcap=30.0,
            temperature=temperature,
            seed=2026061821,
            row_chunk_size=2,
            token_offsets=torch.arange(17, device="cuda", dtype=torch.int32),
        )

    with pytest.raises(ValueError, match="gumbel_vocab_chunk_size"):
        diffusion_gemma_softcap_row_chunked_sample_soft_embeds(
            hidden,
            weight,
            embed_weight,
            softcap=30.0,
            temperature=temperature,
            seed=2026061821,
            row_chunk_size=2,
            gumbel_vocab_chunk_size=0,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_blocked_gumbel_uniform_slices_match_full_uniform():
    rows = torch.tensor([0, 19, 1_000_003], device="cuda", dtype=torch.int64)
    tokens = torch.arange(32769, device="cuda", dtype=torch.int64)
    full = stable_uniform_from_indices(rows, tokens, seed=2026062203).clamp(
        min=torch.finfo(torch.float32).tiny,
        max=1.0 - torch.finfo(torch.float32).eps,
    )
    pieces = []
    for start in range(0, tokens.shape[0], 16384):
        pieces.append(
            stable_uniform_from_indices(
                rows, tokens[start : start + 16384], seed=2026062203
            ).clamp(
                min=torch.finfo(torch.float32).tiny,
                max=1.0 - torch.finfo(torch.float32).eps,
            )
        )

    torch.testing.assert_close(torch.cat(pieces, dim=-1), full, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("vocab_size", [17, 1024, 16383, 16384, 16385, 32769])
def test_blocked_gumbel_matches_materialized_reference_at_vocab_boundaries(
    vocab_size,
):
    hidden, weight = _make_inputs(rows=5, hidden_size=16, vocab_size=vocab_size)
    embed_weight = _make_embed_weight(vocab_size=vocab_size, embed_size=8)
    temperature = torch.tensor(
        [0.3, 0.7, 1e-6, 0.0, 1.1], device="cuda", dtype=torch.float32
    )
    row_seed_offsets = torch.tensor(
        [11, 3, 19, 5, 23], device="cuda", dtype=torch.int64
    )
    expected = diffusion_gemma_softcap_row_chunked_sample_soft_embeds(
        hidden,
        weight,
        embed_weight,
        softcap=30.0,
        temperature=temperature,
        seed=2026062205,
        row_chunk_size=2,
        row_seed_offsets=row_seed_offsets,
        gumbel_vocab_chunk_size=vocab_size + 1,
    )

    gumbel_vocab_chunk_sizes = (
        (1, 7, 1024, 16384) if vocab_size <= 1024 else (1024, 16384)
    )
    for gumbel_vocab_chunk_size in gumbel_vocab_chunk_sizes:
        actual = diffusion_gemma_softcap_row_chunked_sample_soft_embeds(
            hidden,
            weight,
            embed_weight,
            softcap=30.0,
            temperature=temperature,
            seed=2026062205,
            row_chunk_size=2,
            row_seed_offsets=row_seed_offsets,
            gumbel_vocab_chunk_size=gumbel_vocab_chunk_size,
        )
        _assert_rowchunk_outputs_close(actual, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_blocked_gumbel_row_chunk_and_vocab_chunk_invariant():
    hidden, weight = _make_inputs(rows=7, hidden_size=32, vocab_size=4097)
    embed_weight = _make_embed_weight(vocab_size=4097, embed_size=16)
    temperature = torch.tensor(
        [0.3, 0.5, 0.7, 1.0, 0.0, 1e-9, 1.1],
        device="cuda",
        dtype=torch.float32,
    )
    row_seed_offsets = torch.arange(7, device="cuda", dtype=torch.int64) * 13
    expected = diffusion_gemma_softcap_row_chunked_sample_soft_embeds(
        hidden,
        weight,
        embed_weight,
        softcap=30.0,
        temperature=temperature,
        seed=2026062207,
        row_chunk_size=7,
        row_seed_offsets=row_seed_offsets,
        gumbel_vocab_chunk_size=weight.shape[0] + 1,
    )

    for row_chunk_size, gumbel_vocab_chunk_size in (
        (1, 1),
        (2, 7),
        (3, 1024),
        (4, 4096),
    ):
        actual = diffusion_gemma_softcap_row_chunked_sample_soft_embeds(
            hidden,
            weight,
            embed_weight,
            softcap=30.0,
            temperature=temperature,
            seed=2026062207,
            row_chunk_size=row_chunk_size,
            row_seed_offsets=row_seed_offsets,
            gumbel_vocab_chunk_size=gumbel_vocab_chunk_size,
        )
        _assert_rowchunk_outputs_close(actual, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_blocked_gumbel_preserves_cross_block_tie_first_index():
    scaled = torch.zeros((3, 12), device="cuda", dtype=torch.float32)
    scaled[0, 3] = 2.0
    scaled[0, 4] = 2.0
    scaled[1, 7] = 5.0
    scaled[1, 11] = 5.0
    # Row 2 is all equal and should choose token 0.
    row_offsets = torch.arange(3, device="cuda", dtype=torch.int64)
    token_offsets = torch.arange(12, device="cuda", dtype=torch.int64)
    zero_noise = torch.zeros((3, 1), device="cuda", dtype=torch.float32)

    actual = _stable_gumbel_argmax_from_scaled(
        scaled,
        row_offsets,
        token_offsets,
        seed=2026062211,
        noise_scale=zero_noise,
        gumbel_vocab_chunk_size=4,
    )

    torch.testing.assert_close(
        actual, torch.tensor([3, 7, 0], device="cuda"), rtol=0, atol=0
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_blocked_gumbel_matches_independent_materialized_reference():
    generator = torch.Generator(device="cuda").manual_seed(2026062213)
    scaled = torch.randn(
        (5, 1031), device="cuda", dtype=torch.float32, generator=generator
    )
    row_offsets = torch.tensor([0, 7, 11, 23, 101], device="cuda", dtype=torch.int64)
    token_offsets = torch.arange(scaled.shape[1], device="cuda", dtype=torch.int64)
    noise_scale = torch.tensor(
        [1.0, 0.7, 0.0, 2.0, 0.3], device="cuda", dtype=torch.float32
    )[:, None]

    actual = _stable_gumbel_argmax_from_scaled(
        scaled,
        row_offsets,
        token_offsets,
        seed=2026062213,
        noise_scale=noise_scale,
        gumbel_vocab_chunk_size=17,
    )
    expected = _materialized_gumbel_argmax_from_scaled(
        scaled,
        row_offsets,
        token_offsets,
        seed=2026062213,
        noise_scale=noise_scale,
    )

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_blocked_gumbel_applies_scalar_noise_scale():
    generator = torch.Generator(device="cuda").manual_seed(2026062214)
    scaled = torch.randn(
        (4, 257), device="cuda", dtype=torch.float32, generator=generator
    )
    row_offsets = torch.arange(4, device="cuda", dtype=torch.int64)
    token_offsets = torch.arange(scaled.shape[1], device="cuda", dtype=torch.int64)

    actual = _stable_gumbel_argmax_from_scaled(
        scaled,
        row_offsets,
        token_offsets,
        seed=2026062214,
        noise_scale=0.125,
        gumbel_vocab_chunk_size=19,
    )
    expected = _materialized_gumbel_argmax_from_scaled(
        scaled,
        row_offsets,
        token_offsets,
        seed=2026062214,
        noise_scale=0.125,
    )

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_blocked_gumbel_handles_infinite_scores_like_argmax():
    scaled = torch.full((3, 12), float("-inf"), device="cuda", dtype=torch.float32)
    scaled[1, 5] = 1.0
    scaled[1, 8] = float("inf")
    scaled[2, 6] = float("inf")
    scaled[2, 9] = float("inf")
    row_offsets = torch.arange(3, device="cuda", dtype=torch.int64)
    token_offsets = torch.arange(12, device="cuda", dtype=torch.int64)
    zero_noise = torch.zeros((3, 1), device="cuda", dtype=torch.float32)

    actual = _stable_gumbel_argmax_from_scaled(
        scaled,
        row_offsets,
        token_offsets,
        seed=2026062217,
        noise_scale=zero_noise,
        gumbel_vocab_chunk_size=4,
    )
    expected = scaled.argmax(dim=-1)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actual[0].item() == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_blocked_gumbel_handles_nan_scores_like_argmax():
    scaled = torch.zeros((4, 12), device="cuda", dtype=torch.float32)
    scaled[0, 5] = float("nan")
    scaled[1, 9] = float("nan")
    scaled[1, 2] = float("nan")
    scaled[2, 1] = 3.0
    scaled[2, 10] = float("nan")
    scaled[3, 0] = float("nan")
    scaled[3, :] = float("nan")
    row_offsets = torch.arange(4, device="cuda", dtype=torch.int64)
    token_offsets = torch.arange(12, device="cuda", dtype=torch.int64)
    zero_noise = torch.zeros((4, 1), device="cuda", dtype=torch.float32)

    actual = _stable_gumbel_argmax_from_scaled(
        scaled,
        row_offsets,
        token_offsets,
        seed=2026062218,
        noise_scale=zero_noise,
        gumbel_vocab_chunk_size=4,
    )
    expected = scaled.argmax(dim=-1)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    # Row 1 has NaNs in block 0 and block 2; the global first NaN wins.
    assert actual[1].item() == 2
    # Row 2 has a finite score in block 0 and a NaN in block 2; NaN wins like
    # torch.argmax over the full materialized row.
    assert actual[2].item() == 10
    # Row 3 is all NaN; first vocab id wins.
    assert actual[3].item() == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_blocked_gumbel_matches_materialized_nan_inf_scores_with_noise():
    scaled = torch.zeros((5, 16), device="cuda", dtype=torch.float32)
    # Row 0 has an all-finite first block and a later NaN in the same block
    # as a larger finite value. Match materialized torch.argmax behavior.
    scaled[0, 3] = 9.0
    scaled[0, 4] = 99.0
    scaled[0, 6] = float("nan")
    # Row 1 has NaNs in multiple blocks. The first full-row NaN wins.
    scaled[1, 2] = float("nan")
    scaled[1, 7] = float("nan")
    # Row 2 has +inf in multiple blocks. The first +inf wins.
    scaled[2, 1] = float("inf")
    scaled[2, 11] = float("inf")
    # Row 3 is all -inf. Full-row argmax returns the first token.
    scaled[3, :] = float("-inf")
    # Row 4 mixes an earlier +inf with a later NaN. NaN wins like torch.argmax.
    scaled[4, 3] = float("inf")
    scaled[4, 10] = float("nan")

    row_offsets = torch.arange(scaled.shape[0], device="cuda", dtype=torch.int64)
    token_offsets = torch.arange(scaled.shape[1], device="cuda", dtype=torch.int64)
    noise_scale = torch.ones((scaled.shape[0], 1), device="cuda", dtype=torch.float32)

    actual = _stable_gumbel_argmax_from_scaled(
        scaled,
        row_offsets,
        token_offsets,
        seed=2026062231,
        noise_scale=noise_scale,
        gumbel_vocab_chunk_size=4,
    )
    expected = _materialized_gumbel_argmax_from_scaled(
        scaled,
        row_offsets,
        token_offsets,
        seed=2026062231,
        noise_scale=noise_scale,
    )

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actual[0].item() == expected[0].item()
    assert actual[1].item() == 2
    assert actual[2].item() == 1
    assert actual[3].item() == 0
    assert actual[4].item() == 10


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_blocked_gumbel_rejects_non_float32_scaled_scores():
    scaled = torch.zeros((2, 8), device="cuda", dtype=torch.float64)
    row_offsets = torch.arange(2, device="cuda", dtype=torch.int64)
    token_offsets = torch.arange(8, device="cuda", dtype=torch.int64)

    with pytest.raises(ValueError, match="scaled must be float32"):
        _stable_gumbel_argmax_from_scaled(
            scaled,
            row_offsets,
            token_offsets,
            seed=2026062219,
            noise_scale=1.0,
            gumbel_vocab_chunk_size=4,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_blocked_gumbel_rejects_bad_noise_scale_shape():
    scaled = torch.zeros((2, 8), device="cuda", dtype=torch.float32)
    row_offsets = torch.arange(2, device="cuda", dtype=torch.int64)
    token_offsets = torch.arange(8, device="cuda", dtype=torch.int64)
    noise_scale = torch.ones((2,), device="cuda", dtype=torch.float32)

    with pytest.raises(ValueError, match=r"noise_scale tensor.*\[rows, 1\]"):
        _stable_gumbel_argmax_from_scaled(
            scaled,
            row_offsets,
            token_offsets,
            seed=2026062220,
            noise_scale=noise_scale,
            gumbel_vocab_chunk_size=4,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_blocked_gumbel_rejects_bad_noise_scale_dtype():
    scaled = torch.zeros((2, 8), device="cuda", dtype=torch.float32)
    row_offsets = torch.arange(2, device="cuda", dtype=torch.int64)
    token_offsets = torch.arange(8, device="cuda", dtype=torch.int64)
    noise_scale = torch.ones((2, 1), device="cuda", dtype=torch.float64)

    with pytest.raises(ValueError, match="noise_scale tensor must be float32"):
        _stable_gumbel_argmax_from_scaled(
            scaled,
            row_offsets,
            token_offsets,
            seed=2026062221,
            noise_scale=noise_scale,
            gumbel_vocab_chunk_size=4,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_blocked_gumbel_matches_reference_for_supported_input_dtypes(dtype):
    hidden, weight = _make_inputs(
        rows=4, hidden_size=16, vocab_size=1025, dtype=dtype
    )
    embed_weight = _make_embed_weight(vocab_size=1025, embed_size=8, dtype=dtype)
    temperature = torch.tensor(
        [0.3, 0.7, 0.0, 1e-6], device="cuda", dtype=torch.float32
    )
    row_seed_offsets = torch.arange(4, device="cuda", dtype=torch.int64) * 17
    expected = diffusion_gemma_softcap_row_chunked_sample_soft_embeds(
        hidden,
        weight,
        embed_weight,
        softcap=30.0,
        temperature=temperature,
        seed=2026062223,
        row_chunk_size=2,
        row_seed_offsets=row_seed_offsets,
        gumbel_vocab_chunk_size=weight.shape[0] + 1,
    )
    actual = diffusion_gemma_softcap_row_chunked_sample_soft_embeds(
        hidden,
        weight,
        embed_weight,
        softcap=30.0,
        temperature=temperature,
        seed=2026062223,
        row_chunk_size=2,
        row_seed_offsets=row_seed_offsets,
        gumbel_vocab_chunk_size=128,
    )

    _assert_rowchunk_outputs_close(actual, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_blocked_gumbel_scalar_temperature_matches_reference():
    hidden, weight = _make_inputs(rows=4, hidden_size=16, vocab_size=1025)
    embed_weight = _make_embed_weight(vocab_size=1025, embed_size=8)
    temperature = torch.full((4,), 0.7, device="cuda", dtype=torch.float32)
    row_seed_offsets = torch.arange(4, device="cuda", dtype=torch.int64) * 19
    expected = _reference_sample_soft_embeds(
        hidden,
        weight,
        embed_weight,
        softcap=30.0,
        temperature=temperature,
        seed=2026062229,
        row_seed_offsets=row_seed_offsets,
    )
    actual = diffusion_gemma_softcap_row_chunked_sample_soft_embeds(
        hidden,
        weight,
        embed_weight,
        softcap=30.0,
        temperature=0.7,
        seed=2026062229,
        row_chunk_size=2,
        row_seed_offsets=row_seed_offsets,
        gumbel_vocab_chunk_size=128,
    )

    _assert_rowchunk_outputs_close(actual, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_blocked_gumbel_scalar_zero_temperature_skips_sampling():
    hidden, weight = _make_inputs(rows=4, hidden_size=16, vocab_size=257)
    embed_weight = _make_embed_weight(vocab_size=257, embed_size=8)
    lse, entropy, sample, greedy, soft = (
        diffusion_gemma_softcap_row_chunked_sample_soft_embeds(
            hidden,
            weight,
            embed_weight,
            softcap=30.0,
            temperature=0.0,
            seed=2026062233,
            row_chunk_size=2,
            gumbel_vocab_chunk_size=64,
        )
    )

    assert lse.shape == entropy.shape == sample.shape == greedy.shape == (4,)
    torch.testing.assert_close(sample, greedy, rtol=0, atol=0)
    torch.testing.assert_close(entropy, torch.zeros_like(entropy), rtol=0, atol=0)
    torch.testing.assert_close(soft, embed_weight[greedy].float(), rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_diffusion_gemma_row_chunked_positive_temperature_fast_path_matches():
    hidden, weight = _make_inputs(rows=6, hidden_size=64, vocab_size=257)
    embed_weight = _make_embed_weight(vocab_size=257, embed_size=32)
    temperature = torch.tensor(
        [0.3, 0.5, 0.7, 1.0, 0.9, 1.1],
        device="cuda",
        dtype=torch.float32,
    )
    row_seed_offsets = torch.arange(6, device="cuda", dtype=torch.int64) * 5

    expected = diffusion_gemma_softcap_row_chunked_sample_soft_embeds(
        hidden,
        weight,
        embed_weight,
        softcap=30.0,
        temperature=temperature,
        seed=2026061819,
        row_chunk_size=3,
        row_seed_offsets=row_seed_offsets,
    )
    actual = diffusion_gemma_softcap_row_chunked_sample_soft_embeds(
        hidden,
        weight,
        embed_weight,
        softcap=30.0,
        temperature=temperature,
        seed=2026061819,
        row_chunk_size=3,
        row_seed_offsets=row_seed_offsets,
        temperature_is_positive=True,
    )

    for idx, (actual_item, expected_item) in enumerate(zip(actual, expected)):
        if idx in (2, 3):
            torch.testing.assert_close(actual_item, expected_item, rtol=0, atol=0)
        elif idx == 4:
            torch.testing.assert_close(
                actual_item, expected_item, rtol=2e-2, atol=2e-2
            )
        else:
            torch.testing.assert_close(actual_item, expected_item, rtol=3e-4, atol=2e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_diffusion_gemma_row_chunked_chunk_size_invariant():
    hidden, weight = _make_inputs(rows=7, hidden_size=64, vocab_size=257)
    embed_weight = _make_embed_weight(vocab_size=257, embed_size=32)
    temperature = torch.tensor(
        [0.3, 0.5, 0.7, 1.0, 0.0, 0.9, 1.1],
        device="cuda",
        dtype=torch.float32,
    )
    row_seed_offsets = torch.arange(7, device="cuda", dtype=torch.int64) * 3
    expected = diffusion_gemma_softcap_row_chunked_sample_soft_embeds(
        hidden,
        weight,
        embed_weight,
        softcap=30.0,
        temperature=temperature,
        seed=2026061804,
        row_chunk_size=7,
        row_seed_offsets=row_seed_offsets,
    )

    for row_chunk_size in (1, 2, 3):
        actual = diffusion_gemma_softcap_row_chunked_sample_soft_embeds(
            hidden,
            weight,
            embed_weight,
            softcap=30.0,
            temperature=temperature,
            seed=2026061804,
            row_chunk_size=row_chunk_size,
            row_seed_offsets=row_seed_offsets,
        )
        for idx, (actual_item, expected_item) in enumerate(zip(actual, expected)):
            if idx in (2, 3):
                torch.testing.assert_close(actual_item, expected_item, rtol=0, atol=0)
            elif idx == 4:
                torch.testing.assert_close(
                    actual_item, expected_item, rtol=2e-2, atol=2e-2
                )
            else:
                torch.testing.assert_close(
                    actual_item, expected_item, rtol=3e-4, atol=2e-3
                )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_diffusion_gemma_row_chunked_row_seed_offsets_are_reorder_stable():
    hidden, weight = _make_inputs(rows=8, hidden_size=64, vocab_size=257)
    embed_weight = _make_embed_weight(vocab_size=257, embed_size=32)
    temperature = torch.tensor(
        [0.3, 0.5, 0.7, 1.0, 0.6, 0.9, 0.8, 1.1],
        device="cuda",
        dtype=torch.float32,
    )
    row_seed_offsets = torch.tensor(
        [11, 3, 20, 7, 15, 99, 2, 42], device="cuda", dtype=torch.int64
    )
    expected = diffusion_gemma_softcap_row_chunked_sample_soft_embeds(
        hidden,
        weight,
        embed_weight,
        softcap=30.0,
        temperature=temperature,
        seed=2026061805,
        row_chunk_size=3,
        row_seed_offsets=row_seed_offsets,
    )
    perm = torch.tensor([3, 0, 7, 1, 6, 4, 2, 5], device="cuda")
    actual = diffusion_gemma_softcap_row_chunked_sample_soft_embeds(
        hidden[perm],
        weight,
        embed_weight,
        softcap=30.0,
        temperature=temperature[perm],
        seed=2026061805,
        row_chunk_size=2,
        row_seed_offsets=row_seed_offsets[perm],
    )
    inv = torch.argsort(perm)
    for idx, (actual_item, expected_item) in enumerate(zip(actual, expected)):
        if idx in (2, 3):
            torch.testing.assert_close(actual_item[inv], expected_item, rtol=0, atol=0)
        elif idx == 4:
            torch.testing.assert_close(
                actual_item[inv], expected_item, rtol=2e-2, atol=2e-2
            )
        else:
            torch.testing.assert_close(
                actual_item[inv], expected_item, rtol=3e-4, atol=2e-3
            )
