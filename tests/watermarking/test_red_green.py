# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.config.watermarking import WatermarkConfig
from vllm.platforms import current_platform
from vllm.v1.watermarking import (
    RedGreenWatermarkDetector,
    RedGreenWatermarker,
    create_watermarker,
)
from vllm.v1.watermarking.gpu_sampler import GPUWatermarkSampler
from vllm.v1.watermarking.red_green import _binomial_survival
from vllm.v1.watermarking.spec_decode import (
    create_speculative_draft_watermarker,
    speculative_target_watermark_key,
)
from vllm.v1.watermarking.watermarker import RandomSampler
from vllm.v1.worker.gpu.spec_decode import rejection_sampler as rejection_sampler_module
from vllm.v1.worker.gpu.spec_decode.rejection_sampler import RejectionSampler


def _argmax_sampler(logits: torch.Tensor) -> torch.Tensor:
    return logits.argmax(dim=-1)


@pytest.mark.parametrize(
    "successes, trials, probability",
    [(0, 10, 0.25), (3, 10, 0.25), (10, 10, 0.25), (11, 10, 0.25), (40, 100, 0.5)],
)
def test_binomial_survival(successes: int, trials: int, probability: float):
    expected = sum(
        math.comb(trials, count)
        * probability**count
        * (1 - probability) ** (trials - count)
        for count in range(max(successes, 0), trials + 1)
    )

    assert _binomial_survival(successes, trials, probability) == pytest.approx(
        min(expected, 1.0)
    )


def test_config_selects_red_green():
    config = WatermarkConfig(
        algorithm="red_green", key=42, context_width=2, delta=3.0, gamma=0.5
    )
    watermarker = create_watermarker(config)

    assert isinstance(watermarker, RedGreenWatermarker)
    assert watermarker.context_width == 2
    assert (watermarker.delta, watermarker.gamma) == (3.0, 0.5)
    assert not config.supports_speculative_decoding


@pytest.mark.parametrize("overrides", [{"delta": 0.0}, {"gamma": 0.0}, {"gamma": 1.0}])
def test_config_rejects_invalid_red_green_parameters(overrides: dict):
    with pytest.raises(ValueError):
        WatermarkConfig(algorithm="red_green", key=42, **overrides)


def test_green_list_holds_a_gamma_fraction_of_the_vocabulary():
    contexts = torch.randint(
        0, 50_000, (8, 4), generator=torch.Generator().manual_seed(0)
    )
    green = RedGreenWatermarker(key=42, gamma=0.25).green_list(contexts, 50_000)

    assert green.shape == (8, 50_000)
    assert green.float().mean().item() == pytest.approx(0.25, abs=0.01)
    # Each context has its own green list.
    assert not torch.equal(green[0], green[1])


def test_watermark_logits_bias_green_tokens_of_unskipped_rows():
    watermarker = RedGreenWatermarker(key=42, context_width=2, delta=2.0, gamma=0.5)
    logits = torch.randn(3, 64)
    logits[:, :4] = -torch.inf
    contexts = torch.tensor([[1, 2], [3, 4], [5, 6]])
    skip_mask = torch.tensor([False, True, False])

    output = watermarker.watermark_logits(logits, contexts, skip_mask)
    green = watermarker.green_list(contexts, 64)

    expected = torch.where(green, logits + 2.0, logits)
    assert torch.equal(output[0], expected[0])
    assert torch.equal(output[1], logits[1])
    assert torch.equal(output[2], expected[2])
    # Filtered tokens stay filtered.
    assert torch.isneginf(output[:, :4]).all()


def test_sample_draws_from_the_biased_logits():
    watermarker = RedGreenWatermarker(key=42, context_width=1, delta=5.0, gamma=0.25)
    contexts = torch.tensor([[7], [8]])

    logits = torch.zeros(2, 32)
    sample = watermarker.sample(logits, contexts, _argmax_sampler)
    green = watermarker.green_list(contexts, 32)

    assert green[torch.arange(2), sample.token_ids].all()
    # Processed logprobs see the logits before the watermark.
    assert sample.logits is logits


def test_sample_requires_a_random_sampler():
    with pytest.raises(ValueError, match="random sampler"):
        RedGreenWatermarker(key=42).sample(torch.zeros(1, 8), torch.zeros(1, 4))


def _generate(watermarker: RedGreenWatermarker, num_tokens: int, vocab_size: int):
    """Sample from flat logits under the watermark, building each context the
    way the GPU sampler does (-1 before the first generated tokens)."""
    generator = torch.Generator().manual_seed(0)
    width = watermarker.context_width
    token_ids: list[int] = []
    for _ in range(num_tokens):
        context = ([-1] * width + token_ids)[-width:]
        logits = watermarker.watermark_logits(
            torch.zeros(1, vocab_size), torch.tensor([context])
        )
        token_ids.append(
            int(torch.multinomial(logits.softmax(-1), 1, generator=generator))
        )
    return token_ids


def test_detector_recognizes_watermarked_tokens():
    watermarker = RedGreenWatermarker(key=42, context_width=2, delta=4.0, gamma=0.25)
    detector = RedGreenWatermarkDetector(key=42, context_width=2, gamma=0.25)
    token_ids = _generate(watermarker, 128, 1000)

    watermarked = detector.detect(token_ids)
    unwatermarked = detector.detect(
        torch.randint(
            0, 1000, (128,), generator=torch.Generator().manual_seed(1)
        ).tolist()
    )
    other_key = RedGreenWatermarkDetector(key=43, context_width=2).detect(token_ids)

    assert watermarked.is_watermarked
    assert watermarked.p_value < 1e-10
    assert not unwatermarked.is_watermarked
    assert not other_key.is_watermarked


def test_detector_scores_green_tokens():
    detector = RedGreenWatermarkDetector(
        key=42, context_width=1, gamma=0.5, deduplicate_contexts=False
    )
    token_ids = [3, 1, 4, 1, 5, 9, 2, 6]
    green = RedGreenWatermarker(key=42, context_width=1, gamma=0.5).green_list(
        torch.tensor([[-1]] + [[t] for t in token_ids[:-1]]), 10
    )

    detection = detector.detect(token_ids)

    assert detection.num_scored_tokens == len(token_ids)
    assert (
        detection.score == green[torch.arange(len(token_ids)), token_ids].sum().item()
    )
    assert detection.p_value == pytest.approx(
        _binomial_survival(round(detection.score), len(token_ids), 0.5)
    )


def test_target_only_speculative_decoding_keys_only_the_target():
    config = WatermarkConfig(
        algorithm="red_green", key=42, allow_target_only_watermarking=True
    )
    watermarker = create_watermarker(config)

    draft = create_speculative_draft_watermarker(
        watermarker,
        max_num_reqs=2,
        device=torch.device("cpu"),
        allow_target_only=True,
    )

    assert draft is None
    # Verification samples from the watermarked target logits: no keyed recovery.
    assert speculative_target_watermark_key(config) is None


# Request 0 verifies drafts [9, 7] and then its bonus row; request 1 opted out.
_VERIFY_ROWS = {
    "expanded_idx_mapping": torch.tensor([0, 0, 0, 1]),
    "expanded_local_pos": torch.tensor([0, 1, 2, 0]),
    "draft_sampled": torch.tensor([8, 9, 7, 6]),
}
_VERIFY_CONTEXTS = torch.tensor([[7, 8], [8, 9], [9, 7], [5, 6]])


def _verification_sampler(watermarker: RedGreenWatermarker) -> GPUWatermarkSampler:
    sampler = object.__new__(GPUWatermarkSampler)
    sampler.watermarker = watermarker
    sampler.deduplicate_contexts = "single_turn"
    sampler.deduplicate_contexts_max_history = None
    sampler.num_speculative_tokens = 2
    sampler.use_fp64_gumbel = False
    sampler.req_states = SimpleNamespace(
        all_token_ids=SimpleNamespace(
            gpu=torch.tensor([[100, 7, 8, 0, 0], [100, 101, 5, 6, 0]])
        ),
        prompt_len=SimpleNamespace(gpu=torch.tensor([1, 2])),
        total_len=SimpleNamespace(gpu=torch.tensor([3, 4])),
    )
    sampler.watermarking = SimpleNamespace(gpu=torch.tensor([True, False]))
    sampler.sampling_states = SimpleNamespace(
        temperature=SimpleNamespace(gpu=torch.tensor([1.0, 1.0])),
        seeds=SimpleNamespace(gpu=torch.zeros(2, dtype=torch.int64)),
    )
    return sampler


def test_gpu_sampler_watermarks_verification_logits():
    watermarker = RedGreenWatermarker(key=42, context_width=2, delta=2.0, gamma=0.5)
    logits = torch.randn(4, 16)

    output = _verification_sampler(watermarker).watermark_verification_logits(
        logits, **_VERIFY_ROWS
    )

    green = watermarker.green_list(_VERIFY_CONTEXTS, 16)
    green[3] = False
    assert torch.equal(output, torch.where(green, logits + 2.0, logits))


def test_rejection_sampler_verifies_the_watermarked_logits(monkeypatch):
    watermarker = RedGreenWatermarker(key=42, context_width=2, delta=2.0, gamma=0.5)
    sampler = _verification_sampler(watermarker)
    logits = torch.randn(4, 16)
    sampler.apply_sampling_params = lambda *args, **kwargs: logits
    calls = {}

    def rejection_sample(target_logits, *args, **kwargs):
        calls["target_logits"] = target_logits
        calls["kwargs"] = kwargs
        return torch.zeros(2, 3, dtype=torch.int64), torch.ones(2, dtype=torch.int32)

    monkeypatch.setattr(rejection_sampler_module, "rejection_sample", rejection_sample)
    rejection_sampler = object.__new__(RejectionSampler)
    rejection_sampler.sampler = sampler
    rejection_sampler.num_speculative_steps = 2
    rejection_sampler.synthetic_conditional_rates = None
    rejection_sampler.use_block_verification = False
    rejection_sampler.watermark_key = None

    processed_logits, _, _ = rejection_sampler._verify(
        logits,
        None,
        _VERIFY_ROWS["draft_sampled"],
        torch.arange(4),
        torch.tensor([0, 3, 4]),
        torch.tensor([0, 1]),
        np.array([0, 1]),
        _VERIFY_ROWS["expanded_idx_mapping"],
        _VERIFY_ROWS["expanded_local_pos"],
        np.array([8, 8]),
        _VERIFY_ROWS["draft_sampled"],
    )

    expected = sampler.watermark_verification_logits(logits, **_VERIFY_ROWS)
    assert torch.equal(calls["target_logits"], expected)
    # Processed logprobs see the logits before the watermark.
    assert processed_logits is logits
    # The watermark is in the verified logits; recovery is not keyed.
    assert "watermark_key" not in calls["kwargs"]


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(), reason="requires a CUDA-like accelerator"
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("context_width", [1, 4, 5])
def test_green_bias_kernel_matches_the_prf(dtype: torch.dtype, context_width: int):
    from vllm.v1.worker.gpu.sample.watermark import philox_green_bias

    watermarker = RedGreenWatermarker(
        key=2**40 + 7, context_width=context_width, gamma=0.3
    )
    vocab_size = 50_257  # not a multiple of the kernel's block size
    logits = torch.randn(6, vocab_size, device="cuda").to(dtype)
    logits[:, :10] = -torch.inf
    contexts = torch.randint(-1, vocab_size, (6, context_width), device="cuda")
    skip_mask = torch.tensor([False, True, False, False, True, False], device="cuda")

    output = philox_green_bias(
        logits, contexts, watermarker.prf.key, 2.5, 0.3, skip_mask=skip_mask
    )
    green = watermarker.green_list(contexts, vocab_size) & ~skip_mask.unsqueeze(-1)

    assert output.dtype == torch.float32
    assert torch.equal(output, torch.where(green, logits.float() + 2.5, logits.float()))


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(), reason="requires a CUDA-like accelerator"
)
@pytest.mark.parametrize("use_fp64", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("context_width", [1, 5])
def test_fused_sample_draws_the_token_of_the_biased_logits(
    dtype: torch.dtype, context_width: int, use_fp64: bool
):
    watermarker = RedGreenWatermarker(
        key=2**40 + 7, context_width=context_width, delta=2.5, gamma=0.3
    )
    generator = torch.Generator(device="cuda").manual_seed(0)
    num_rows, vocab_size = 64, 50_257  # not a multiple of the kernel's block size
    logits = torch.randn(num_rows, vocab_size, device="cuda", generator=generator)
    logits = (3 * logits).to(dtype)
    logits[:, :10] = -torch.inf
    contexts = torch.randint(
        -1, vocab_size, (num_rows, context_width), device="cuda", generator=generator
    )
    skip_mask = torch.rand(num_rows, device="cuda", generator=generator) < 0.25
    # Several rows per request state, one of them greedy.
    random_sampler = RandomSampler(
        expanded_idx_mapping=torch.randint(
            0, 4, (num_rows,), device="cuda", generator=generator
        ),
        temperatures=torch.tensor([1.0, 0.7, 0.0, 1.3], device="cuda"),
        seeds=torch.randint(0, 2**62, (4,), device="cuda", generator=generator),
        positions=torch.randint(
            0, 4096, (num_rows,), device="cuda", generator=generator
        ),
        use_fp64=use_fp64,
    )

    sample = watermarker.sample(logits, contexts, random_sampler, skip_mask)
    biased = watermarker.watermark_logits(logits, contexts, skip_mask)

    assert torch.equal(sample.token_ids, random_sampler(biased))
    assert sample.logits is logits
