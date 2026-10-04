# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for SBW watermarking (SBWWatermarker + SBWWatermarkDetector).

Test structure mirrors tests/watermarking/test_gumbel.py and
tests/watermarking/test_watermarking.py.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.config.watermarking import WatermarkConfig
from vllm.platforms import current_platform
from vllm.v1.watermarking import SBWWatermarkDetector, SBWWatermarker
from vllm.v1.watermarking.factory import create_watermarker
from vllm.v1.watermarking.gpu_sampler import GPUWatermarkSampler
from vllm.v1.worker.gpu.sample.sampler import Sampler

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_watermarker(
    key: int = 15485863,
    scheme: str = "selfhash",
    context_width: int | None = None,
    gamma: float = 0.25,
    delta: float = 2.0,
) -> SBWWatermarker:
    return SBWWatermarker(
        key=key,
        scheme=scheme,
        context_width=context_width,
        gamma=gamma,
        delta=delta,
    )


# SBWWatermarker.sample() accepts an optional random_sampler; without one it
# falls back to argmax. In production the sampler delegates to FlashInfer or gumbel.


# ---------------------------------------------------------------------------
# WatermarkConfig integration
# ---------------------------------------------------------------------------


def test_sbw_config_creates_watermarker():
    wm = create_watermarker(WatermarkConfig(key=42, algorithm="sbw"))
    assert isinstance(wm, SBWWatermarker)


def test_sbw_config_selfhash_default_context_width():
    wm = create_watermarker(
        WatermarkConfig(key=42, algorithm="sbw", sbw_scheme="selfhash")
    )
    assert wm.context_width == 4


def test_sbw_config_lefthash_default_context_width():
    wm = create_watermarker(
        WatermarkConfig(key=42, algorithm="sbw", sbw_scheme="lefthash")
    )
    assert wm.context_width == 1


def test_sbw_config_context_width_override():
    wm = create_watermarker(WatermarkConfig(key=42, algorithm="sbw", context_width=2))
    assert wm.context_width == 2


def test_sbw_config_supports_speculative_decoding():
    cfg = WatermarkConfig(key=42, algorithm="sbw")
    assert cfg.supports_speculative_decoding is True


def test_gumbel_config_does_not_support_speculative_decoding(monkeypatch):
    monkeypatch.setattr(
        "vllm.config.watermarking.logger.warning_once",
        lambda *a, **kw: None,
    )
    cfg = WatermarkConfig(key=42, algorithm="gumbel")
    assert cfg.supports_speculative_decoding is False


# ---------------------------------------------------------------------------
# SBWWatermarker construction validation
# ---------------------------------------------------------------------------


class TestSBWWatermarkerValidation:
    def test_invalid_key_too_large(self):
        with pytest.raises(ValueError, match="64 bits"):
            SBWWatermarker(key=2**65)

    def test_invalid_scheme(self):
        with pytest.raises(ValueError, match="scheme"):
            SBWWatermarker(key=42, scheme="unknown")

    def test_gamma_zero_invalid(self):
        with pytest.raises(ValueError, match="gamma"):
            SBWWatermarker(key=42, gamma=0.0)

    def test_gamma_one_invalid(self):
        with pytest.raises(ValueError, match="gamma"):
            SBWWatermarker(key=42, gamma=1.0)

    def test_delta_negative_invalid(self):
        with pytest.raises(ValueError, match="delta"):
            SBWWatermarker(key=42, delta=-0.1)

    def test_context_width_zero_invalid(self):
        with pytest.raises(ValueError, match="context_width"):
            SBWWatermarker(key=42, context_width=0)

    def test_selfhash_default_context_width(self):
        wm = SBWWatermarker(key=42, scheme="selfhash")
        assert wm.context_width == 4

    def test_lefthash_default_context_width(self):
        wm = SBWWatermarker(key=42, scheme="lefthash")
        assert wm.context_width == 1


# ---------------------------------------------------------------------------
# Watermarker contract (mirrors test_watermarker_contract in test_watermarking.py)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("scheme", ["selfhash", "lefthash"])
def test_watermarker_contract(scheme: str):
    """sample() returns correct shapes and deterministic biased logits."""
    cw = 4 if scheme == "selfhash" else 1
    wm = _make_watermarker(scheme=scheme, context_width=cw)
    logits = torch.zeros(2, 128)
    contexts = torch.randint(0, 1000, (2, cw))

    first = wm.sample(logits, contexts)
    second = wm.sample(logits, contexts)

    assert first.token_ids.shape == (2,)  # placeholder shape is correct
    assert first.logits.shape == logits.shape
    # Biased logits are deterministic for the same input
    assert torch.equal(first.logits, second.logits)


def test_sample_returns_biased_logits():
    """sample() must return biased (not original) logits."""
    wm = _make_watermarker(delta=3.0)
    logits = torch.zeros(1, 50)
    contexts = torch.zeros(1, 4, dtype=torch.long)
    result = wm.sample(logits, contexts)
    # At least some tokens must be biased up by delta.
    assert (result.logits > 0).any()


def test_sample_delta_zero_unchanged():
    """delta=0 → logits unchanged, token drawn from unbiased distribution."""
    wm = _make_watermarker(delta=0.0)
    logits = torch.randn(2, 100)
    contexts = torch.zeros(2, 4, dtype=torch.long)
    result = wm.sample(logits, contexts)
    torch.testing.assert_close(result.logits, logits)


def test_sample_gamma_controls_fraction():
    """Green-list fraction should approximate gamma over a large vocabulary."""
    gamma = 0.3
    wm = _make_watermarker(scheme="lefthash", context_width=1, gamma=gamma, delta=1.0)
    logits = torch.zeros(1, 2000)
    contexts = torch.tensor([[7]], dtype=torch.long)
    result = wm.sample(logits, contexts)
    frac = (result.logits > 0).float().mean().item()
    assert abs(frac - gamma) < 0.05, f"Expected ~{gamma:.0%} green, got {frac:.2%}"


def test_sample_different_keys_different_greenlists():
    wm1 = _make_watermarker(key=111, scheme="lefthash", context_width=1)
    wm2 = _make_watermarker(key=222, scheme="lefthash", context_width=1)
    logits = torch.zeros(1, 500)
    ctx = torch.tensor([[3]], dtype=torch.long)
    r1 = wm1.sample(logits.clone(), ctx)
    r2 = wm2.sample(logits.clone(), ctx)
    assert not torch.equal(r1.logits, r2.logits)


def test_sample_selfhash_and_lefthash_differ():
    wm_s = _make_watermarker(key=42, scheme="selfhash", context_width=4)
    wm_l = _make_watermarker(key=42, scheme="lefthash", context_width=4)
    logits = torch.zeros(1, 200)
    ctx = torch.randint(0, 100, (1, 4))
    rs = wm_s.sample(logits.clone(), ctx)
    rl = wm_l.sample(logits.clone(), ctx)
    assert not torch.equal(rs.logits, rl.logits)


def test_sample_context_width_1_selfhash_does_not_crash():
    """selfhash with cw=1 triggers the empty-prefix path; must not raise."""
    wm = _make_watermarker(scheme="selfhash", context_width=1)
    logits = torch.zeros(2, 50)
    contexts = torch.zeros(2, 1, dtype=torch.long)
    result = wm.sample(logits, contexts)
    assert result.token_ids.shape == (2,)


# ---------------------------------------------------------------------------
# CUDA tests (skip if no GPU)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not current_platform.is_cuda_alike(), reason="requires CUDA")
@pytest.mark.parametrize("scheme", ["selfhash", "lefthash"])
@pytest.mark.parametrize("key", [42, 2**32 + 7, 15485863])
def test_cuda_matches_cpu(scheme: str, key: int):
    """CUDA and CPU must produce bit-identical biased logits."""
    cw = 4 if scheme == "selfhash" else 1
    wm = _make_watermarker(key=key, scheme=scheme, context_width=cw)
    torch.manual_seed(0)
    logits_cpu = torch.randn(8, 512)
    contexts_cpu = torch.randint(0, 50000, (8, cw))

    result_cpu = wm.sample(logits_cpu.clone(), contexts_cpu)
    result_cuda = wm.sample(logits_cpu.clone().cuda(), contexts_cpu.cuda())

    torch.testing.assert_close(
        result_cuda.logits.cpu(), result_cpu.logits, rtol=0, atol=0
    )


@pytest.mark.skipif(not current_platform.is_cuda_alike(), reason="requires CUDA")
def test_cuda_handles_noncontiguous_inputs():
    wm = _make_watermarker(scheme="lefthash", context_width=1)
    logits = torch.randn(8, 200, device="cuda")[:, ::2]  # non-contiguous
    contexts = torch.zeros(8, 1, dtype=torch.long, device="cuda")
    # Must not raise or produce wrong shapes.
    result = wm.sample(logits, contexts)
    assert result.token_ids.shape == (8,)


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------


class TestSBWDetector:
    def test_watermarked_text_detected(self):
        """Tokens sampled with realistic params should yield a detectable signal.

        Uses context_width=4 and diverse random logits so the context changes
        at every step, giving the detector many independently-seeded positions
        to score.
        """
        key, gamma, delta = 15485863, 0.25, 2.0
        cw = 4
        wm = _make_watermarker(
            key=key, scheme="selfhash", context_width=cw, gamma=gamma, delta=delta
        )
        detector = SBWWatermarkDetector(
            key=key, scheme="selfhash", context_width=cw, gamma=gamma
        )
        # Generate 300 tokens with diverse random logits (varied context each step).
        torch.manual_seed(0)
        tokens: list[int] = list(torch.randint(0, 500, (cw,)).tolist())
        for _ in range(300):
            logits = torch.randn(1, 500)
            ctx = torch.tensor([tokens[-cw:]], dtype=torch.long)
            result = wm.sample(logits, ctx)
            # SBW returns biased logits; sample greedily from them.
            tokens.append(result.logits.argmax(dim=-1)[0].item())

        detection = detector.detect(tokens)
        assert detection.p_value < 0.01

    def test_unwatermarked_text_not_detected(self):
        """Random tokens should not be flagged as watermarked."""
        torch.manual_seed(42)
        detector = SBWWatermarkDetector(key=15485863)
        tokens = torch.randint(0, 1000, (300,)).tolist()
        detection = detector.detect(tokens)
        # p-value should not be very small (test at very conservative threshold).
        assert detection.p_value > 0.001

    def test_detector_wrong_key_does_not_detect(self):
        """Detection with a wrong key should not find the watermark."""
        key_gen, key_det = 111, 222
        wm = _make_watermarker(
            key=key_gen, scheme="lefthash", context_width=1, delta=5.0
        )
        detector = SBWWatermarkDetector(key=key_det, scheme="lefthash", context_width=1)
        tokens: list[int] = [0]
        for _ in range(199):
            logits = torch.zeros(1, 500)
            ctx = torch.tensor([[tokens[-1]]], dtype=torch.long)
            result = wm.sample(logits, ctx)
            tokens.append(result.logits.argmax(dim=-1)[0].item())
        detection = detector.detect(tokens)
        assert detection.p_value > 0.01

    def test_detector_deduplicates_by_default(self):
        """Repeated contexts should be deduplicated (same as Gumbel detector)."""
        detector = SBWWatermarkDetector(key=42, scheme="lefthash", context_width=1)
        # All tokens are the same → all contexts are identical → only 2 scored.
        detection = detector.detect([1, 1, 1, 1, 1])
        assert detection.num_scored_tokens == 2

    def test_detector_can_disable_deduplication(self):
        detector = SBWWatermarkDetector(
            key=42, scheme="lefthash", context_width=1, deduplicate_contexts=False
        )
        detection = detector.detect([1, 1, 1, 1, 1])
        assert detection.num_scored_tokens == 5

    def test_p_value_empty_sequence(self):
        detector = SBWWatermarkDetector(key=42)
        detection = detector.detect([])
        assert detection.p_value == 1.0 or detection.num_scored_tokens == 0

    def test_p_value_is_valid_probability(self):
        detector = SBWWatermarkDetector(key=42, scheme="lefthash", context_width=1)
        tokens = list(range(50))
        detection = detector.detect(tokens)
        assert 0.0 <= detection.p_value <= 1.0


# ---------------------------------------------------------------------------
# GPUWatermarkSampler -- SBW bias path
# ---------------------------------------------------------------------------
# These tests verify that GPUWatermarkSampler.sample() applies the SBW bias
# on raw logits (before apply_sampling_params) and that per-row delta=0.0
# correctly leaves non-watermarked rows untouched.
# ---------------------------------------------------------------------------


def _make_sbw_gpu_sampler(
    watermarked: list[bool],
    temperatures: list[float] | None = None,
) -> GPUWatermarkSampler:
    """Build a minimal GPUWatermarkSampler stub for SBW path tests."""
    B = len(watermarked)
    if temperatures is None:
        temperatures = [1.0] * B
    wm = _make_watermarker(delta=2.0, gamma=0.5)
    sampler = object.__new__(GPUWatermarkSampler)
    sampler.watermarker = wm
    sampler.watermarking = SimpleNamespace(
        np=np.array(watermarked),
        gpu=torch.tensor(watermarked),
    )
    sampler.sampling_states = SimpleNamespace(
        temperature=SimpleNamespace(
            np=np.array(temperatures), gpu=torch.tensor(temperatures)
        ),
        seeds=SimpleNamespace(gpu=torch.zeros(B, dtype=torch.int64)),
    )
    sampler._get_contexts = lambda idx_mapping: torch.zeros(
        len(idx_mapping), wm.context_width, dtype=torch.int64
    )
    return sampler


def test_sbw_gpu_sampler_biases_raw_logits_before_super(monkeypatch):
    """sample() must pass biased logits to super().sample(), not raw ones.

    We intercept super().sample() and verify that the logits it receives
    are different from the raw logits passed in (i.e. bias was applied).
    """
    sampler = _make_sbw_gpu_sampler(watermarked=[True])
    received: list[torch.Tensor] = []

    def fake_super_sample(self, logits, *args, **kwargs):
        received.append(logits.clone())
        return torch.tensor([0]), logits

    monkeypatch.setattr(Sampler, "sample", fake_super_sample)

    raw_logits = torch.zeros(1, 100)
    sampler.sample(
        raw_logits,
        torch.tensor([0]),
        torch.tensor([0]),
        np.array([0]),
        torch.zeros(1, dtype=torch.int64),
        torch.zeros(1, dtype=torch.int64),
        torch.zeros(1, dtype=torch.int64),
    )

    assert len(received) == 1
    # At least some tokens must have been biased
    assert not torch.equal(received[0], raw_logits), (
        "super().sample() received unchanged logits -- bias was not applied"
    )


def test_sbw_gpu_sampler_does_not_bias_when_no_watermarked_rows(monkeypatch):
    """If no rows are watermarked, raw logits must reach super() unchanged."""
    sampler = _make_sbw_gpu_sampler(watermarked=[False, False])
    received: list[torch.Tensor] = []

    def fake_super_sample(self, logits, *args, **kwargs):
        received.append(logits.clone())
        return torch.tensor([0, 0]), logits

    monkeypatch.setattr(Sampler, "sample", fake_super_sample)

    raw_logits = torch.zeros(2, 100)
    sampler.sample(
        raw_logits,
        torch.tensor([0, 1]),
        torch.tensor([0, 1]),
        np.array([0, 1]),
        torch.zeros(2, dtype=torch.int64),
        torch.zeros(2, dtype=torch.int64),
        torch.zeros(2, dtype=torch.int64),
    )

    assert len(received) == 1
    assert torch.equal(received[0], raw_logits), (
        "super().sample() received modified logits despite no watermarked rows"
    )


def test_sbw_gpu_sampler_mixed_batch_non_watermarked_rows_unchanged(monkeypatch):
    """Non-watermarked rows must receive delta=0 bias (i.e. unchanged logits)."""
    sampler = _make_sbw_gpu_sampler(watermarked=[True, False, True, False])
    received: list[torch.Tensor] = []

    def fake_super_sample(self, logits, *args, **kwargs):
        received.append(logits.clone())
        return torch.zeros(4, dtype=torch.int64), logits

    monkeypatch.setattr(Sampler, "sample", fake_super_sample)

    raw_logits = torch.zeros(4, 100)
    sampler.sample(
        raw_logits,
        torch.tensor([0, 1, 2, 3]),
        torch.tensor([0, 1, 2, 3]),
        np.array([0, 1, 2, 3]),
        torch.zeros(4, dtype=torch.int64),
        torch.zeros(4, dtype=torch.int64),
        torch.zeros(4, dtype=torch.int64),
    )

    assert len(received) == 1
    biased = received[0]
    # Non-watermarked rows (1, 3) must be identical to raw
    assert torch.equal(biased[1], raw_logits[1]), "row 1 (unwatermarked) was modified"
    assert torch.equal(biased[3], raw_logits[3]), "row 3 (unwatermarked) was modified"
    # Watermarked rows (0, 2) must differ
    assert not torch.equal(biased[0], raw_logits[0]), (
        "row 0 (watermarked) was not biased"
    )
    assert not torch.equal(biased[2], raw_logits[2]), (
        "row 2 (watermarked) was not biased"
    )


def test_sbw_gpu_sampler_biases_greedy_requests(monkeypatch):
    """SBW must bias greedy (temperature=0) requests unlike Gumbel."""
    sampler = _make_sbw_gpu_sampler(watermarked=[True], temperatures=[0.0])
    received: list[torch.Tensor] = []

    def fake_super_sample(self, logits, *args, **kwargs):
        received.append(logits.clone())
        return torch.tensor([0]), logits

    monkeypatch.setattr(Sampler, "sample", fake_super_sample)

    raw_logits = torch.zeros(1, 100)
    sampler.sample(
        raw_logits,
        torch.tensor([0]),
        torch.tensor([0]),
        np.array([0]),
        torch.zeros(1, dtype=torch.int64),
        torch.zeros(1, dtype=torch.int64),
        torch.zeros(1, dtype=torch.int64),
    )

    assert len(received) == 1
    assert not torch.equal(received[0], raw_logits), (
        "SBW bias was not applied to greedy request"
    )
