# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import random
from typing import NamedTuple

import numpy as np
import pytest
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from vllm.benchmarks.datasets import BimodalDataset, SampleRequest


@pytest.fixture(scope="session")
def hf_tokenizer() -> PreTrainedTokenizerBase:
    # Use a small, commonly available tokenizer
    return AutoTokenizer.from_pretrained("gpt2")


class BimodalParams(NamedTuple):
    num_requests: int
    short_ratio: float
    short_input_range: tuple[int, int]
    short_output_range: tuple[int, int]
    long_input_range: tuple[int, int]
    long_output_len: int


@pytest.fixture(scope="session")
def bimodal_params() -> BimodalParams:
    return BimodalParams(
        num_requests=100,
        short_ratio=BimodalDataset.DEFAULT_SHORT_RATIO,
        short_input_range=(
            BimodalDataset.DEFAULT_SHORT_INPUT_MIN,
            BimodalDataset.DEFAULT_SHORT_INPUT_MAX,
        ),
        short_output_range=(
            BimodalDataset.DEFAULT_SHORT_OUTPUT_MIN,
            BimodalDataset.DEFAULT_SHORT_OUTPUT_MAX,
        ),
        long_input_range=(
            BimodalDataset.DEFAULT_LONG_INPUT_MIN,
            BimodalDataset.DEFAULT_LONG_INPUT_MAX,
        ),
        long_output_len=BimodalDataset.DEFAULT_LONG_OUTPUT_LEN,
    )


def _fingerprint_sample(req: SampleRequest) -> tuple[str, int, int]:
    """Project a SampleRequest into a comparable tuple."""
    return (req.prompt, req.prompt_len, req.expected_output_len)


def _collect_samples(
    dataset: BimodalDataset,
    tokenizer: PreTrainedTokenizerBase,
    num_requests: int = 100,
    short_ratio: float = BimodalDataset.DEFAULT_SHORT_RATIO,
    short_input_range: tuple[int, int] = (
        BimodalDataset.DEFAULT_SHORT_INPUT_MIN,
        BimodalDataset.DEFAULT_SHORT_INPUT_MAX,
    ),
    short_output_range: tuple[int, int] = (
        BimodalDataset.DEFAULT_SHORT_OUTPUT_MIN,
        BimodalDataset.DEFAULT_SHORT_OUTPUT_MAX,
    ),
    long_input_range: tuple[int, int] = (
        BimodalDataset.DEFAULT_LONG_INPUT_MIN,
        BimodalDataset.DEFAULT_LONG_INPUT_MAX,
    ),
    long_output_len: int = BimodalDataset.DEFAULT_LONG_OUTPUT_LEN,
) -> list[tuple[str, int, int]]:
    samples = dataset.sample(
        tokenizer=tokenizer,
        num_requests=num_requests,
        short_ratio=short_ratio,
        short_input_range=short_input_range,
        short_output_range=short_output_range,
        long_input_range=long_input_range,
        long_output_len=long_output_len,
    )
    return [_fingerprint_sample(s) for s in samples]


# -------------------------------------------
# Seed determinism
# -------------------------------------------


@pytest.mark.benchmark
def test_bimodal_dataset_same_seed(
    hf_tokenizer: PreTrainedTokenizerBase,
    bimodal_params: BimodalParams,
) -> None:
    """Same seed should yield identical outputs, even if global RNGs change.

    This guards against accidental reliance on Python's random or np.random
    in BimodalDataset after moving to numpy.default_rng.
    """
    p = bimodal_params
    common_seed = 123
    dataset_a = BimodalDataset(random_seed=common_seed)
    dataset_b = BimodalDataset(random_seed=common_seed)

    a = _collect_samples(
        dataset_a,
        hf_tokenizer,
        num_requests=p.num_requests,
        short_ratio=p.short_ratio,
        short_input_range=p.short_input_range,
        short_output_range=p.short_output_range,
        long_input_range=p.long_input_range,
        long_output_len=p.long_output_len,
    )

    # Perturb global RNG state to ensure isolation.
    random.seed(999)
    _ = [random.random() for _ in range(100)]
    np.random.seed(888)
    _ = [np.random.random() for _ in range(100)]

    b = _collect_samples(
        dataset_b,
        hf_tokenizer,
        num_requests=p.num_requests,
        short_ratio=p.short_ratio,
        short_input_range=p.short_input_range,
        short_output_range=p.short_output_range,
        long_input_range=p.long_input_range,
        long_output_len=p.long_output_len,
    )
    assert a == b


@pytest.mark.benchmark
def test_bimodal_dataset_different_seeds(
    hf_tokenizer: PreTrainedTokenizerBase,
    bimodal_params: BimodalParams,
) -> None:
    """Different seeds should change outputs with overwhelming likelihood."""
    p = bimodal_params
    dataset_a = BimodalDataset(random_seed=0)
    a = _collect_samples(
        dataset_a,
        hf_tokenizer,
        num_requests=p.num_requests,
        short_ratio=p.short_ratio,
        short_input_range=p.short_input_range,
        short_output_range=p.short_output_range,
        long_input_range=p.long_input_range,
        long_output_len=p.long_output_len,
    )

    dataset_b = BimodalDataset(random_seed=999)
    # Perturb global RNG with same seed as dataset_a to ensure isolation.
    random.seed(0)
    np.random.seed(0)
    b = _collect_samples(
        dataset_b,
        hf_tokenizer,
        num_requests=p.num_requests,
        short_ratio=p.short_ratio,
        short_input_range=p.short_input_range,
        short_output_range=p.short_output_range,
        long_input_range=p.long_input_range,
        long_output_len=p.long_output_len,
    )
    assert a != b


# -------------------------------------------
# Distribution tests
# -------------------------------------------


@pytest.mark.benchmark
def test_bimodal_dataset_default_distribution(
    hf_tokenizer: PreTrainedTokenizerBase,
) -> None:
    """Default 80/20 split should produce the expected ratio.

    With a fixed seed and 500 samples, the actual split is deterministic.
    We assert within a tighter tolerance (+/- 5%) than the statistical
    bound to catch regressions in sampling logic.
    """
    dataset = BimodalDataset(random_seed=42)
    num_requests = 500
    requests = dataset.sample(
        tokenizer=hf_tokenizer,
        num_requests=num_requests,
    )
    assert len(requests) == num_requests

    # Thresholds derived from class constants with tokenizer margin.
    short_upper = BimodalDataset.DEFAULT_SHORT_INPUT_MAX + 50
    long_lower = BimodalDataset.DEFAULT_LONG_INPUT_MIN - 100

    short_count = sum(
        1
        for r in requests
        if r.prompt_len <= short_upper
        and BimodalDataset.DEFAULT_SHORT_OUTPUT_MIN
        <= r.expected_output_len
        <= BimodalDataset.DEFAULT_SHORT_OUTPUT_MAX
    )
    long_count = sum(
        1
        for r in requests
        if r.prompt_len >= long_lower
        and r.expected_output_len == BimodalDataset.DEFAULT_LONG_OUTPUT_LEN
    )

    # With fixed seed=42 and 500 samples, expect ~80% short, ~20% long.
    # Use +/- 5% tolerance (75%-85% short, 15%-25% long).
    assert short_count >= num_requests * 0.75, (
        f"Expected ~80% short, got {short_count}/{num_requests}"
    )
    assert short_count <= num_requests * 0.85, (
        f"Expected ~80% short, got {short_count}/{num_requests}"
    )
    assert long_count >= num_requests * 0.15, (
        f"Expected ~20% long, got {long_count}/{num_requests}"
    )
    assert long_count <= num_requests * 0.25, (
        f"Expected ~20% long, got {long_count}/{num_requests}"
    )


@pytest.mark.benchmark
def test_bimodal_dataset_custom_ratio(
    hf_tokenizer: PreTrainedTokenizerBase,
) -> None:
    """Custom short_ratio=0.5 should shift the distribution."""
    dataset = BimodalDataset(random_seed=42)
    num_requests = 500
    requests = dataset.sample(
        tokenizer=hf_tokenizer,
        num_requests=num_requests,
        short_ratio=0.5,
        short_input_range=(50, 100),
        short_output_range=(10, 20),
        long_input_range=(500, 1000),
        long_output_len=64,
    )
    assert len(requests) == num_requests

    short_count = sum(
        1 for r in requests
        if r.prompt_len <= 150 and r.expected_output_len <= 20
    )
    long_count = sum(
        1 for r in requests
        if r.prompt_len >= 400 and r.expected_output_len == 64
    )
    # 50/50 split with +/- 5% tolerance.
    assert short_count >= num_requests * 0.40, (
        f"Expected ~50% short, got {short_count}/{num_requests}"
    )
    assert long_count >= num_requests * 0.40, (
        f"Expected ~50% long, got {long_count}/{num_requests}"
    )


# -------------------------------------------
# Edge case: all short / all long
# -------------------------------------------


@pytest.mark.benchmark
def test_bimodal_dataset_all_short(
    hf_tokenizer: PreTrainedTokenizerBase,
) -> None:
    """short_ratio=1.0 should produce only short requests."""
    dataset = BimodalDataset(random_seed=42)
    num_requests = 50
    requests = dataset.sample(
        tokenizer=hf_tokenizer,
        num_requests=num_requests,
        short_ratio=1.0,
        short_input_range=(50, 100),
        short_output_range=(10, 20),
        long_input_range=(500, 1000),
        long_output_len=64,
    )
    assert len(requests) == num_requests
    for r in requests:
        assert r.expected_output_len <= 20, (
            f"Expected all short outputs, got output_len={r.expected_output_len}"
        )


@pytest.mark.benchmark
def test_bimodal_dataset_all_long(
    hf_tokenizer: PreTrainedTokenizerBase,
) -> None:
    """short_ratio=0.0 should produce only long requests."""
    dataset = BimodalDataset(random_seed=42)
    num_requests = 50
    requests = dataset.sample(
        tokenizer=hf_tokenizer,
        num_requests=num_requests,
        short_ratio=0.0,
        short_input_range=(50, 100),
        short_output_range=(10, 20),
        long_input_range=(500, 1000),
        long_output_len=64,
    )
    assert len(requests) == num_requests
    for r in requests:
        assert r.expected_output_len == 64, (
            f"Expected all long outputs (64), got {r.expected_output_len}"
        )


# -------------------------------------------
# Input validation
# -------------------------------------------


@pytest.mark.benchmark
def test_bimodal_dataset_invalid_short_ratio(
    hf_tokenizer: PreTrainedTokenizerBase,
) -> None:
    """Raise ValueError when short_ratio is outside [0, 1]."""
    dataset = BimodalDataset(random_seed=42)
    with pytest.raises(ValueError, match="bimodal-short-ratio"):
        dataset.sample(
            tokenizer=hf_tokenizer,
            num_requests=10,
            short_ratio=1.5,
        )
    with pytest.raises(ValueError, match="bimodal-short-ratio"):
        dataset.sample(
            tokenizer=hf_tokenizer,
            num_requests=10,
            short_ratio=-0.1,
        )


@pytest.mark.benchmark
def test_bimodal_dataset_invalid_range(
    hf_tokenizer: PreTrainedTokenizerBase,
) -> None:
    """Raise ValueError when min > max in any range."""
    dataset = BimodalDataset(random_seed=42)
    with pytest.raises(ValueError, match="bimodal-short-input-min"):
        dataset.sample(
            tokenizer=hf_tokenizer,
            num_requests=10,
            short_input_range=(200, 100),
        )
    with pytest.raises(ValueError, match="bimodal-short-output-min"):
        dataset.sample(
            tokenizer=hf_tokenizer,
            num_requests=10,
            short_output_range=(50, 10),
        )
    with pytest.raises(ValueError, match="bimodal-long-input-min"):
        dataset.sample(
            tokenizer=hf_tokenizer,
            num_requests=10,
            long_input_range=(4000, 2000),
        )


# -------------------------------------------
# max_model_len validation
# -------------------------------------------


@pytest.mark.benchmark
def test_bimodal_dataset_max_model_len_rejects_oversized_long(
    hf_tokenizer: PreTrainedTokenizerBase,
) -> None:
    """Raise ValueError when long-arm max sequence exceeds max_model_len."""
    dataset = BimodalDataset(random_seed=42)
    with pytest.raises(ValueError, match="long-arm.*exceeds max_model_len"):
        dataset.sample(
            tokenizer=hf_tokenizer,
            num_requests=10,
            long_input_range=(2000, 4000),
            long_output_len=128,
            max_model_len=2048,
        )


@pytest.mark.benchmark
def test_bimodal_dataset_max_model_len_rejects_oversized_short(
    hf_tokenizer: PreTrainedTokenizerBase,
) -> None:
    """Raise ValueError when short-arm max sequence exceeds max_model_len."""
    dataset = BimodalDataset(random_seed=42)
    with pytest.raises(ValueError, match="short-arm.*exceeds max_model_len"):
        dataset.sample(
            tokenizer=hf_tokenizer,
            num_requests=10,
            short_input_range=(1500, 2000),
            short_output_range=(500, 600),
            long_input_range=(500, 1000),
            long_output_len=128,
            max_model_len=2048,
        )


@pytest.mark.benchmark
def test_bimodal_dataset_max_model_len_accepts_valid(
    hf_tokenizer: PreTrainedTokenizerBase,
) -> None:
    """Succeed when max sequence fits within max_model_len."""
    dataset = BimodalDataset(random_seed=42)
    requests = dataset.sample(
        tokenizer=hf_tokenizer,
        num_requests=10,
        long_input_range=(500, 1000),
        long_output_len=128,
        max_model_len=2048,
    )
    assert len(requests) == 10
