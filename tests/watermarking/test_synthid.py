# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import logging
import math
from types import SimpleNamespace

import pytest
import torch
from pydantic import ValidationError

from vllm.config import VllmConfig
from vllm.config.watermarking import WatermarkConfig
from vllm.logger import _print_warning_once
from vllm.v1.watermarking.factory import create_watermarker
from vllm.v1.watermarking.gumbel import GumbelWatermarker
from vllm.v1.watermarking.prfs import PhiloxPRF
from vllm.v1.watermarking.synthid import (
    _STREAM_DOMAIN,
    SynthIDWatermarkDetector,
    SynthIDWatermarker,
)
from vllm.v1.watermarking.watermarker import RandomSampler
from vllm.v1.worker.gpu.sample import synthid as fused_synthid
from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample
from vllm.v1.worker.gpu.sample.synthid import is_supported, synthid_sample


def test_synthid_config_and_factory():
    config = WatermarkConfig(key=42, algorithm="synthid_text", context_width=3, depth=5)
    watermarker = create_watermarker(config)

    assert isinstance(watermarker, SynthIDWatermarker)
    assert watermarker.context_width == 3
    assert watermarker.depth == 5
    assert watermarker.prf.key == 42
    assert not config.supports_speculative_decoding
    assert WatermarkConfig(key=42, algorithm="gumbel").depth == 22


@pytest.mark.parametrize(
    ("depth", "match"),
    [(0, "greater than or equal to 1"), (33, "less than or equal to 32")],
)
def test_synthid_config_rejects_invalid_depth(depth, match):
    with pytest.raises(ValidationError, match=match):
        WatermarkConfig(
            key=42,
            algorithm="synthid_text",
            depth=depth,
        )


def test_synthid_config_rejects_bool_depth():
    with pytest.raises(ValidationError, match="valid integer"):
        WatermarkConfig(key=42, algorithm="synthid_text", depth=True)


@pytest.mark.parametrize("depth", [1, 32])
def test_synthid_accepts_supported_depth(depth):
    config = WatermarkConfig(
        key=42,
        algorithm="synthid_text",
        depth=depth,
    )
    assert config.depth == depth


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"context_width": 0}, "context_width must be positive"),
        ({"depth": 0}, "SynthID-Text depth must be between 1 and 32"),
        ({"depth": 33}, "SynthID-Text depth must be between 1 and 32"),
        ({"depth": True}, "SynthID-Text depth must be an integer, not a bool"),
        ({"key": -1}, "Philox keys must fit in 64 bits"),
    ],
)
def test_synthid_rejects_invalid_constructor_args(kwargs, match):
    with pytest.raises(ValueError, match=match):
        SynthIDWatermarker(**({"key": 42} | kwargs))


@pytest.mark.parametrize(
    "key, depth",
    [
        (0, 1),
        (42, 5),
        (2**64 - 1, 32),
    ],
)
def test_synthid_logits_follow_philox_bits_and_reweighting(key, depth):
    contexts = torch.tensor([[-1, -1, 7], [11, 12, 13]])
    logits = torch.tensor([[0.2, -0.3, 0.7, -1.1], [0.4, 1.3, -0.8, 0.1]])
    prf = PhiloxPRF(key)

    expected_probs = torch.softmax(logits, dim=1)

    words = prf.uint32(
        contexts,
        torch.arange(logits.shape[1]),
        stream=_STREAM_DOMAIN,
    )

    for depth_index in range(depth):
        g = ((words >> depth_index) & 1).to(logits.dtype)
        mass = (expected_probs * g).sum(dim=1, keepdim=True)
        expected_probs *= 1 + g - mass

    actual = SynthIDWatermarker(
        key,
        context_width=3,
        depth=depth,
    ).watermark_logits(logits, contexts)

    torch.testing.assert_close(actual.exp(), expected_probs)


@pytest.mark.parametrize(
    "constructor",
    [SynthIDWatermarker, SynthIDWatermarkDetector],
)
def test_synthid_warns_for_large_context_width(constructor):
    with pytest.warns(
        UserWarning,
        match="context_width values greater than 16 reduce robustness to edits",
    ):
        constructor(42, context_width=17)


def test_synthid_sample_uses_watermarked_logits_and_skip_mask():
    watermarker = SynthIDWatermarker(42, context_width=3, depth=5)
    logits = torch.tensor([[0.2, 0.4, -0.1], [0.3, -0.2, 0.5]])
    contexts = torch.tensor([[-1, -1, 7], [11, 12, 13]])
    skip_mask = torch.tensor([False, True])
    sampled_logits = []

    def random_sampler(scores):
        sampled_logits.append(scores)
        return scores.argmax(dim=-1)

    result = watermarker.sample(logits, contexts, random_sampler, skip_mask)
    expected = watermarker.watermark_logits(logits, contexts)

    assert len(sampled_logits) == 1
    torch.testing.assert_close(sampled_logits[0][0], expected[0])
    torch.testing.assert_close(sampled_logits[0][1], logits[1])
    assert torch.equal(result.token_ids, sampled_logits[0].argmax(dim=-1))
    assert result.logits is logits

    # A peaked row underflows in-support tokens; only the input's -inf stay -inf.
    peaked = torch.tensor([[20.0, 0.0, 0.0, float("-inf")]])
    actual = SynthIDWatermarker(42, context_width=3, depth=32).watermark_logits(
        peaked, contexts[:1]
    )
    assert torch.equal(actual.isneginf(), peaked.isneginf())


def test_synthid_requires_random_sampler():
    with pytest.raises(ValueError, match="SynthID-Text requires a random sampler"):
        SynthIDWatermarker(42).sample(torch.zeros(1, 3), torch.zeros(1, 4))


@pytest.mark.parametrize("depth", [0, 33])
def test_synthid_detector_rejects_invalid_depth(depth):
    with pytest.raises(
        ValueError,
        match="SynthID-Text depth must be between 1 and 32",
    ):
        SynthIDWatermarkDetector(42, depth=depth)


@pytest.mark.parametrize(
    "key, context_width, depth",
    [
        (0, 1, 1),
        (42, 2, 3),
        (2**64 - 1, 4, 32),
    ],
)
def test_synthid_detector_matches_independent_bit_count(
    key,
    context_width,
    depth,
):
    tokens = [3, 5, 7, 11, 13, 17]
    contexts = torch.tensor(
        [
            ([-1] * context_width + tokens[:position])[-context_width:]
            for position in range(len(tokens))
        ]
    )
    targets = torch.tensor(tokens)[:, None]
    prf = PhiloxPRF(key)

    words = prf.uint32(
        contexts,
        targets,
        stream=_STREAM_DOMAIN,
    ).flatten()

    hits = 0
    for depth_index in range(depth):
        hits += sum((int(word) >> depth_index) & 1 for word in words)

    trials = len(tokens) * depth
    expected_p_value = sum(math.comb(trials, k) for k in range(hits, trials + 1)) / (
        2**trials
    )

    result = SynthIDWatermarkDetector(
        key,
        context_width,
        depth,
    ).detect(tokens)

    assert result.num_scored_tokens == len(tokens)
    assert result.score == hits / trials
    assert result.p_value == pytest.approx(expected_p_value)


def test_synthid_detector_empty_and_repeated_contexts():
    detector = SynthIDWatermarkDetector(42, context_width=1, depth=4)

    empty = detector.detect([])
    repeated = detector.detect([1, 1, 1, 1, 1])

    assert empty.num_scored_tokens == 0
    assert empty.p_value == 1
    assert not empty.is_watermarked
    assert repeated.num_scored_tokens == 2


def test_synthid_detector_recognizes_generated_tokens():
    key = 42
    width = 4
    depth = 4
    watermarker = SynthIDWatermarker(key, width, depth)
    generator = torch.Generator().manual_seed(123)
    logits = torch.zeros(1, 32)
    tokens: list[int] = []

    def random_sampler(scores):
        return torch.multinomial(
            scores.softmax(dim=-1), 1, generator=generator
        ).flatten()

    for _ in range(96):
        context = ([-1] * width + tokens)[-width:]
        sample = watermarker.sample(logits, torch.tensor([context]), random_sampler)
        tokens.append(int(sample.token_ids.item()))

    matched = SynthIDWatermarkDetector(key, width, depth).detect(tokens)

    assert matched.is_watermarked
    assert matched.p_value < 0.01


def test_synthid_runs_tournament_in_fp32_for_bf16_logits():
    logits = torch.randn(4, 1000, generator=torch.Generator().manual_seed(0))
    logits = logits.to(torch.bfloat16)
    contexts = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
    watermarker = SynthIDWatermarker(42, context_width=3, depth=32)

    actual = watermarker.watermark_logits(logits, contexts)
    expected = watermarker.watermark_logits(logits.float(), contexts)

    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected)


def test_synthid_detector_ignores_gumbel_text_with_same_key():
    key = 42
    width = 4
    generator = torch.Generator().manual_seed(0)
    watermarker = GumbelWatermarker(key, width)
    tokens: list[int] = []

    for _ in range(400):
        logits = torch.randn(1, 4096, generator=generator) * 2
        context = ([-1] * width + tokens)[-width:]
        sample = watermarker.sample(logits, torch.tensor([context]))
        tokens.append(int(sample.token_ids.item()))

    result = SynthIDWatermarkDetector(key, width, 32).detect(tokens)

    assert not result.is_watermarked
    assert result.p_value > 1e-4


requires_fused = pytest.mark.skipif(
    not is_supported(),
    reason="requires the fused SynthID-Text kernels (sm80 or newer)",
)


def _cuda_sampler_inputs(num_rows, vocab_size, seed=0):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    logits = 3 * torch.randn(num_rows, vocab_size, device="cuda", generator=generator)
    logits[0] *= 10  # peaked row
    if num_rows > 3:
        logits[1, ::3] = float("-inf")  # masked tokens
        # Increasing logits move the running max in every histogram tile.
        logits[2] = torch.linspace(-30, 30, vocab_size, device="cuda")
        # Fully masked leading tiles leave a split's running max at -inf.
        logits[3, : vocab_size - 100] = float("-inf")
    contexts = torch.randint(
        0, vocab_size, (num_rows, 4), device="cuda", generator=generator
    )
    # -1 pads the context of a sequence shorter than the context width.
    contexts[4:8, :2] = -1
    if num_rows > 8:
        contexts[8] = -1
    temperatures = torch.tensor([0.0, 0.7, 1.0, 1.3], device="cuda")[
        torch.randint(0, 4, (num_rows,), device="cuda", generator=generator)
    ]
    random_sampler = RandomSampler(
        expanded_idx_mapping=torch.arange(num_rows, device="cuda", dtype=torch.int32),
        temperatures=temperatures,
        seeds=torch.randint(0, 2**62, (num_rows,), device="cuda", generator=generator),
        positions=torch.randint(
            0, 1000, (num_rows,), device="cuda", generator=generator
        ),
    )
    skip_mask = torch.rand(num_rows, device="cuda", generator=generator) < 0.2
    skip_mask[:4] = False
    return logits, contexts, random_sampler, skip_mask


def _reference_probs(watermarker, logits, contexts):
    """The tournament in fp64 on torch, independent of the fused kernels."""
    vocabulary = torch.arange(logits.shape[-1], device=logits.device)
    words = watermarker.prf.uint32(contexts, vocabulary, stream=_STREAM_DOMAIN)
    probs = torch.softmax(logits.double(), dim=1)
    return watermarker._update_probs(probs, words, watermarker.depth)


def _sample_fused(watermarker, logits, contexts, random_sampler, skip_mask):
    return synthid_sample(
        logits,
        contexts,
        watermarker.prf.key,
        watermarker.depth,
        random_sampler.expanded_idx_mapping,
        random_sampler.temperatures,
        random_sampler.seeds,
        random_sampler.positions,
        stream=_STREAM_DOMAIN,
        skip_mask=skip_mask,
    )


@requires_fused
@pytest.mark.parametrize(
    ("depth", "num_rows", "vocab_size"),
    [
        (1, 256, 5000),
        (2, 256, 5000),
        (11, 256, 5000),
        (12, 256, 5000),
        (22, 256, 5000),
        (23, 256, 5000),
        (32, 256, 5000),
        # 33 tiles of 2048 tokens at one row: the vocabulary is split across programs.
        (22, 12, 131849),
        (32, 12, 131849),
        # More rows than programs and more tiles than one program may histogram.
        (22, 600, 65000),
        # One row at a production vocabulary size.
        (22, 1, 151936),
    ],
)
def test_synthid_fused_sample_matches_reference(depth, num_rows, vocab_size):
    logits, contexts, random_sampler, skip_mask = _cuda_sampler_inputs(
        num_rows, vocab_size
    )
    watermarker = SynthIDWatermarker(42, context_width=4, depth=depth)
    reference = _reference_probs(watermarker, logits, contexts).log().float()
    scores = torch.where(skip_mask.unsqueeze(-1), logits, reference)
    expected = random_sampler(scores)

    actual = _sample_fused(watermarker, logits, contexts, random_sampler, skip_mask)

    assert torch.equal(actual[skip_mask], expected[skip_mask])
    # Rounding differences may flip a near-tie of the noised maximum. Rows
    # whose sample survives a tiny perturbation of the reference are no tie.
    perturbation = 1e-3 * torch.randn_like(scores)
    stable = random_sampler(scores + perturbation) == random_sampler(
        scores - perturbation
    )
    assert stable.float().mean() > 0.99
    assert torch.equal(actual[stable], expected[stable])


@requires_fused
@pytest.mark.parametrize("use_fp64", [False, True])
def test_synthid_fused_skipped_rows_share_gumbel_noise(use_fp64):
    logits, contexts, random_sampler, _ = _cuda_sampler_inputs(64, 5000)
    skip_mask = torch.ones(64, dtype=torch.bool, device="cuda")

    actual = synthid_sample(
        logits,
        contexts,
        42,
        22,
        random_sampler.expanded_idx_mapping,
        random_sampler.temperatures,
        random_sampler.seeds,
        random_sampler.positions,
        stream=_STREAM_DOMAIN,
        skip_mask=skip_mask,
        use_fp64=use_fp64,
    )

    expected = gumbel_sample(
        logits,
        random_sampler.expanded_idx_mapping,
        random_sampler.temperatures,
        random_sampler.seeds,
        random_sampler.positions,
        apply_temperature=False,
        is_drafting=False,
        use_fp64=use_fp64,
    )
    assert torch.equal(actual, expected)


@requires_fused
@pytest.mark.parametrize("use_fp64", [False, True])
def test_synthid_fused_sample_follows_tournament_distribution(use_fp64):
    num_samples, vocab_size, depth = 200_000, 40, 3
    watermarker = SynthIDWatermarker(7, context_width=2, depth=depth)
    logits = torch.randn(vocab_size, generator=torch.Generator().manual_seed(1))
    contexts = torch.tensor([[5, 9]])
    expected = _reference_probs(watermarker, logits[None], contexts)[0]

    sampled = synthid_sample(
        logits.cuda()[None].expand(num_samples, vocab_size),
        contexts.cuda().expand(num_samples, 2),
        watermarker.prf.key,
        depth,
        torch.zeros(num_samples, device="cuda", dtype=torch.int32),
        torch.ones(1, device="cuda"),
        torch.tensor([1234], device="cuda"),
        torch.arange(num_samples, device="cuda"),
        stream=_STREAM_DOMAIN,
        use_fp64=use_fp64,
    )

    frequencies = torch.bincount(sampled.cpu(), minlength=vocab_size) / num_samples
    torch.testing.assert_close(
        frequencies.double(), expected.double(), rtol=0, atol=5e-3
    )


@requires_fused
def test_synthid_detector_recognizes_fused_samples():
    key, width, depth, vocab_size = 42, 4, 32, 1000
    generator = torch.Generator(device="cuda").manual_seed(0)
    tokens: list[int] = []
    for step in range(64):
        context = ([-1] * width + tokens)[-width:]
        sampled = synthid_sample(
            torch.randn(1, vocab_size, device="cuda", generator=generator),
            torch.tensor([context], device="cuda"),
            key,
            depth,
            torch.zeros(1, device="cuda", dtype=torch.int32),
            torch.ones(1, device="cuda"),
            torch.tensor([7], device="cuda"),
            torch.tensor([step], device="cuda"),
            stream=_STREAM_DOMAIN,
        )
        tokens.append(int(sampled.item()))

    detected = SynthIDWatermarkDetector(key, width, depth).detect(tokens)

    assert detected.is_watermarked
    assert detected.p_value < 0.01


def _fake_fused_sampler(monkeypatch, sample_fn):
    monkeypatch.setattr(SynthIDWatermarker, "_use_fused", lambda self, logits: True)
    monkeypatch.setattr("vllm.v1.worker.gpu.sample.synthid.synthid_sample", sample_fn)


def _cpu_sampler_inputs():
    logits = torch.randn(2, 16, generator=torch.Generator().manual_seed(0))
    contexts = torch.tensor([[-1, -1, 7], [11, 12, 13]])
    random_sampler = RandomSampler(
        expanded_idx_mapping=torch.arange(2),
        temperatures=torch.ones(2),
        seeds=torch.zeros(2, dtype=torch.int64),
        positions=torch.zeros(2, dtype=torch.int64),
    )
    return logits, contexts, random_sampler


def test_failed_first_fused_call_is_a_clear_error(monkeypatch):
    def failing_synthid_sample(*args, **kwargs):
        raise RuntimeError("compilation failed")

    _fake_fused_sampler(monkeypatch, failing_synthid_sample)
    logits, contexts, random_sampler = _cpu_sampler_inputs()
    watermarker = SynthIDWatermarker(42, context_width=3, depth=5)

    with pytest.raises(RuntimeError, match="Triton 3.8 or newer") as error:
        watermarker.sample(logits, contexts, random_sampler)
    assert str(error.value.__cause__) == "compilation failed"


def test_cpu_tensors_use_the_torch_tournament(monkeypatch):
    def failing_synthid_sample(*args, **kwargs):
        raise AssertionError("the fused kernels need CUDA tensors")

    monkeypatch.setattr("vllm.v1.worker.gpu.sample.synthid.is_supported", lambda: True)
    monkeypatch.setattr(
        "vllm.v1.worker.gpu.sample.synthid.synthid_sample", failing_synthid_sample
    )
    monkeypatch.setattr(
        RandomSampler, "__call__", lambda self, scores: scores.argmax(dim=-1)
    )
    logits, contexts, random_sampler = _cpu_sampler_inputs()
    watermarker = SynthIDWatermarker(42, context_width=3, depth=5)
    expected = watermarker.watermark_logits(logits, contexts).argmax(dim=-1)

    sample = watermarker.sample(logits, contexts, random_sampler)

    assert torch.equal(sample.token_ids, expected)


def test_fused_failure_after_the_first_call_is_not_swallowed(monkeypatch):
    outcomes = [torch.tensor([3, 4]), RuntimeError("illegal memory access")]

    def synthid_sample_once(*args, **kwargs):
        outcome = outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    _fake_fused_sampler(monkeypatch, synthid_sample_once)
    logits, contexts, random_sampler = _cpu_sampler_inputs()
    watermarker = SynthIDWatermarker(42, context_width=3, depth=5)

    first = watermarker.sample(logits, contexts, random_sampler)
    assert first.token_ids.tolist() == [3, 4]
    with pytest.raises(RuntimeError, match="illegal memory access"):
        watermarker.sample(logits, contexts, random_sampler)


def test_torch_tournament_sampling_warns_once(monkeypatch, caplog):
    monkeypatch.setattr("vllm.v1.worker.gpu.sample.synthid.is_supported", lambda: False)
    monkeypatch.setattr(
        RandomSampler, "__call__", lambda self, scores: scores.argmax(dim=-1)
    )
    _print_warning_once.cache_clear()
    logits, contexts, random_sampler = _cpu_sampler_inputs()
    watermarker = SynthIDWatermarker(42, context_width=3, depth=5)

    with caplog.at_level(logging.WARNING):
        watermarker.sample(logits, contexts, random_sampler)
        SynthIDWatermarker(7, context_width=3, depth=5).sample(
            logits, contexts, random_sampler
        )

    assert caplog.text.count("sampling with the torch tournament") == 1
    assert "meant for correctness, not serving" in caplog.text


@pytest.mark.parametrize(
    ("has_gluon", "is_cuda", "has_capability", "expected"),
    [
        (True, True, True, True),
        (False, True, True, False),  # Triton < 3.8
        (True, True, False, False),
        (True, False, True, False),
    ],
)
def test_fused_kernels_require_nvidia_sm80(
    monkeypatch, has_gluon, is_cuda, has_capability, expected
):
    platform = SimpleNamespace(
        is_cuda=lambda: is_cuda,
        has_device_capability=lambda capability: has_capability and capability == 80,
    )
    monkeypatch.setattr(fused_synthid, "current_platform", platform)
    monkeypatch.setattr(fused_synthid, "HAS_GLUON", has_gluon)

    assert fused_synthid.is_supported() is expected


def _platform(name, capability=(9, 0)):
    return SimpleNamespace(
        device_name=name,
        is_cuda=lambda: name == "cuda",
        is_cuda_alike=lambda: name in ("cuda", "rocm"),
        has_device_capability=lambda minimum: capability >= divmod(minimum, 10),
    )


@pytest.mark.parametrize(
    ("platform", "has_gluon", "fails"),
    [
        (_platform("cuda"), True, False),
        (_platform("cuda"), False, True),  # Triton < 3.8
        (_platform("cuda", capability=(7, 5)), False, False),  # torch tournament
        (_platform("rocm"), True, True),
        (_platform("cpu"), False, False),  # torch tournament
    ],
    ids=["cuda", "cuda-old-triton", "cuda-sm75", "rocm", "cpu"],
)
def test_synthid_fails_at_startup_without_fused_kernels(
    monkeypatch, platform, has_gluon, fails
):
    monkeypatch.setattr("vllm.v1.watermarking.synthid.current_platform", platform)
    monkeypatch.setattr(fused_synthid, "current_platform", platform)
    monkeypatch.setattr(fused_synthid, "HAS_GLUON", has_gluon)
    watermark_config = WatermarkConfig(key=42, algorithm="synthid_text")
    if fails:
        with pytest.raises(ValueError, match="Triton 3.8 or newer"):
            VllmConfig(watermark_config=watermark_config)
    else:
        VllmConfig(watermark_config=watermark_config)


@requires_fused
@pytest.mark.parametrize("enforce_eager", [True, False])
def test_engine_fails_at_startup_when_fused_kernels_fail(
    vllm_runner, monkeypatch, enforce_eager
):
    """The profiling run's dummy sampler pass is the first fused call.

    It comes before the kernel warmup, which does not run at all with
    ``enforce_eager``.
    """
    fused_calls = []

    def failing_synthid_sample(*args, **kwargs):
        fused_calls.append(kwargs)
        raise RuntimeError("kernel failed to compile")

    # The patch reaches the worker only when the engine runs in-process.
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    monkeypatch.setattr(
        "vllm.v1.worker.gpu.sample.synthid.synthid_sample", failing_synthid_sample
    )
    with (
        pytest.raises(RuntimeError, match="Triton 3.8 or newer") as error,
        vllm_runner(
            "hmellor/tiny-random-LlamaForCausalLM",
            dtype="half",
            enforce_eager=enforce_eager,
            max_model_len=128,
            max_num_seqs=4,
            watermark_config={"algorithm": "synthid_text", "key": 42},
        ),
    ):
        pass

    assert len(fused_calls) == 1
    assert str(error.value.__cause__) == "kernel failed to compile"
