# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math

import pytest
import torch

from vllm.platforms import CpuArchEnum, current_platform
from vllm.triton_utils import HAS_TRITON
from vllm.v1.worker.cpu.gumbel import gumbel_sample
from vllm.v1.worker.cpu.sampler import CPUSamplingStates

if current_platform.is_cpu():
    import vllm._C  # noqa: F401

pytestmark = pytest.mark.skipif(
    not current_platform.is_cpu()
    or HAS_TRITON
    or current_platform.get_cpu_architecture() == CpuArchEnum.X86,
    reason="requires a non-x86 CPU platform without Triton",
)

_UINT32_MASK = (1 << 32) - 1
_UINT64_MASK = (1 << 64) - 1
VOCAB_SIZE = 200_000
NUM_SAMPLES = 500_000
HEAD_LOG_GAP = 18.0
Z_TOLERANCE = 10.0


def _sample(
    logits: torch.Tensor,
    expanded_idx_mapping: torch.Tensor,
    temperature: torch.Tensor,
    seeds: torch.Tensor,
    positions: torch.Tensor,
    apply_temperature: bool = True,
    use_fp64: bool = False,
) -> torch.Tensor:
    return gumbel_sample(
        logits,
        expanded_idx_mapping,
        temperature,
        seeds,
        positions,
        apply_temperature=apply_temperature,
        is_drafting=False,
        use_fp64=use_fp64,
    )


def _sample_many(
    logits_1d: torch.Tensor,
    num_samples: int,
    *,
    use_fp64: bool = False,
) -> torch.Tensor:
    vocab_size = logits_1d.shape[0]
    logits = logits_1d.unsqueeze(0).expand(num_samples, vocab_size)
    mapping = torch.zeros(num_samples, dtype=torch.int64)
    temperature = torch.ones(1)
    seeds = torch.tensor([0xABCD], dtype=torch.int64)
    positions = torch.arange(num_samples, dtype=torch.int64)
    return _sample(
        logits,
        mapping,
        temperature,
        seeds,
        positions,
        use_fp64=use_fp64,
    )


def _sample_histogram(
    logits_1d: torch.Tensor,
    num_samples: int,
    *,
    chunk_size: int = 1_000_000,
) -> torch.Tensor:
    histogram = torch.zeros(logits_1d.shape[0], dtype=torch.float64)
    for start in range(0, num_samples, chunk_size):
        size = min(chunk_size, num_samples - start)
        vocab_size = logits_1d.shape[0]
        logits = logits_1d.unsqueeze(0).expand(size, vocab_size)
        mapping = torch.zeros(size, dtype=torch.int64)
        temperature = torch.ones(1)
        seeds = torch.tensor([0xABCD], dtype=torch.int64)
        positions = torch.arange(start, start + size, dtype=torch.int64)
        sampled = _sample(logits, mapping, temperature, seeds, positions)
        histogram += torch.bincount(sampled, minlength=vocab_size).double()
    return histogram


def _murmur3_rotl32(value: int, shift: int) -> int:
    return ((value << shift) | (value >> (32 - shift))) & _UINT32_MASK


def _murmur3_mix(hash_: int, key: int) -> int:
    key = (key * 0xCC9E2D51) & _UINT32_MASK
    key = _murmur3_rotl32(key, 15)
    key = (key * 0x1B873593) & _UINT32_MASK
    hash_ ^= key
    hash_ = _murmur3_rotl32(hash_, 13)
    return (hash_ * 5 + 0xE6546B64) & _UINT32_MASK


def _murmur3_fmix32(hash_: int) -> int:
    hash_ ^= hash_ >> 16
    hash_ = (hash_ * 0x85EBCA6B) & _UINT32_MASK
    hash_ ^= hash_ >> 13
    hash_ = (hash_ * 0xC2B2AE35) & _UINT32_MASK
    return (hash_ ^ (hash_ >> 16)) & _UINT32_MASK


def _murmur3_hash32(seed: int, position: int, token_id: int, domain: int = 0) -> int:
    seed_bits = seed & _UINT64_MASK
    hash_ = domain
    hash_ = _murmur3_mix(hash_, seed_bits & _UINT32_MASK)
    hash_ = _murmur3_mix(hash_, seed_bits >> 32)
    hash_ = _murmur3_mix(hash_, position & _UINT32_MASK)
    hash_ = _murmur3_mix(hash_, token_id & _UINT32_MASK)
    return _murmur3_fmix32(hash_ ^ 16)


def _murmur3_uniform32(seed: int, position: int, vocab_size: int) -> torch.Tensor:
    hashes = torch.tensor(
        [_murmur3_hash32(seed, position, token_id) for token_id in range(vocab_size)],
        dtype=torch.int64,
    )
    high = (hashes >> 16).to(torch.float32)
    low = (hashes & 0xFFFF).to(torch.float32)
    return high * 1.52587890625e-05 + (low + 0.5) * 2.3283064365386963e-10


def _murmur3_uniform64(seed: int, position: int, vocab_size: int) -> torch.Tensor:
    random53 = []
    for token_id in range(vocab_size):
        low = _murmur3_hash32(seed, position, token_id)
        high = _murmur3_hash32(seed, position, token_id, 0x9E3779B9)
        random53.append(((high << 32) | low) >> 11)

    uniform = (torch.tensor(random53, dtype=torch.float64) + 0.5) * 2.0**-53
    return torch.where(uniform == 1.0, uniform - 2.0**-53, uniform)


def _log1p_neg_stable(value: torch.Tensor) -> torch.Tensor:
    polynomial = torch.full_like(value, 1.0 / 8.0)
    for denominator in range(7, 1, -1):
        polynomial = 1.0 / denominator + value * polynomial
    polynomial = 1.0 + value * polynomial
    series = -value * polynomial
    direct = torch.log(torch.clamp(1.0 - value, min=2.0**-24))
    return torch.where(value < 0.25, series, direct)


def _reference_sample(
    logits: torch.Tensor,
    expanded_idx_mapping: torch.Tensor,
    temperature: torch.Tensor,
    seeds: torch.Tensor,
    positions: torch.Tensor,
    apply_temperature: bool = True,
    use_fp64: bool = False,
) -> torch.Tensor:
    output = []
    for row_idx, row in enumerate(logits.float()):
        req_state_idx = int(expanded_idx_mapping[row_idx])
        temp = float(temperature[req_state_idx]) if req_state_idx >= 0 else 0.0
        if temp == 0.0:
            output.append(int(row.argmax()))
            continue

        position = int(positions[row_idx])
        seed = int(seeds[req_state_idx])

        if use_fp64:
            uniform = _murmur3_uniform64(seed, position, row.numel())
            noise = -torch.log(-torch.log(uniform))
        else:
            uniform = _murmur3_uniform32(seed, position, row.numel())
            noise = -torch.log(-_log1p_neg_stable(uniform))

        if apply_temperature:
            row = row / temp
        output.append(int((row.to(noise.dtype) + noise).argmax()))

    return torch.tensor(output, dtype=torch.int64)


def _make_heavy_tailed_counts(seed: int = 1234) -> torch.Tensor:
    counts = torch.randint(
        1,
        4,
        (VOCAB_SIZE,),
        generator=torch.Generator().manual_seed(seed),
        dtype=torch.int64,
    )
    counts[0] = round(math.exp(HEAD_LOG_GAP))
    return counts


def _counts_to_logits(counts: torch.Tensor) -> torch.Tensor:
    return counts.double().log().float()


def _z_score(observed: int, expected: float, num_trials: int) -> float:
    probability = expected / num_trials
    return (observed - expected) / math.sqrt(
        num_trials * probability * (1.0 - probability)
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("use_fp64", [False, True])
def test_random_and_mixed_rows_match_reference(
    dtype: torch.dtype, use_fp64: bool
) -> None:
    logits = torch.randn(4, 257, generator=torch.Generator().manual_seed(1234)).to(
        dtype
    )
    mapping = torch.tensor([3, 0, 2, 1], dtype=torch.int64)
    temperature = torch.tensor([0.7, 0.0, 1.3, 1.0])
    seeds = torch.tensor([0xABCD, -1, (1 << 63) - 1, 17], dtype=torch.int64)
    positions = torch.tensor([0, 11, 123456789, 19], dtype=torch.int64)

    expected = _reference_sample(
        logits,
        mapping,
        temperature,
        seeds,
        positions,
        use_fp64=use_fp64,
    )
    actual = _sample(
        logits,
        mapping,
        temperature,
        seeds,
        positions,
        use_fp64=use_fp64,
    )

    assert torch.equal(actual, expected)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_greedy_supports_dtypes_strides_and_invalid_mapping(
    dtype: torch.dtype,
) -> None:
    storage = torch.full((3, 48), -100.0, dtype=dtype)
    logits = storage[:, 1:46]
    logits[0, [0, 17, 44]] = torch.tensor([1.0, 5.0, 3.0], dtype=dtype)
    logits[1, [0, 31, 44]] = torch.tensor([-3.0, -2.0, -1.0], dtype=dtype)
    logits[2, [3, 40]] = 7.0
    mapping = torch.tensor([2, -1, 0], dtype=torch.int64)
    temperature = torch.tensor([0.0, 1.0, 0.0])
    seeds = torch.tensor([11, 22, 33], dtype=torch.int64)
    positions = torch.tensor([100, 101, 102], dtype=torch.int64)

    actual = _sample(logits, mapping, temperature, seeds, positions)

    assert torch.equal(actual, logits.float().argmax(dim=-1))


def test_pre_scaled_logits_match_temperature_application() -> None:
    logits = torch.randn(4, 1031, generator=torch.Generator().manual_seed(7))
    mapping = torch.tensor([2, 0, 3, 1], dtype=torch.int64)
    temperature = torch.tensor([0.5, 1.5, 0.8, 2.0])
    seeds = torch.tensor([19, 23, 29, 31], dtype=torch.int64)
    positions = torch.tensor([31, 37, 41, 43], dtype=torch.int64)
    scaled_logits = logits / temperature[mapping].unsqueeze(1)

    expected = _sample(
        scaled_logits,
        mapping,
        temperature,
        seeds,
        positions,
        apply_temperature=False,
    )
    actual = _sample(
        logits,
        mapping,
        temperature,
        seeds,
        positions,
        apply_temperature=True,
    )

    assert torch.equal(actual, expected)


def test_sampling_states_apply_temperature() -> None:
    states = CPUSamplingStates(4, 17, torch.device("cpu"))
    temperatures = torch.tensor([0.5, 0.0, 2.0, 1.0])
    states.temperature.copy_(temperatures)

    logits = torch.arange(1, 69, dtype=torch.float32).reshape(4, 17)
    expanded_idx_mapping = torch.tensor([2, 0, 3, 1], dtype=torch.int64)
    expected = logits / torch.tensor([2.0, 0.5, 1.0, 1.0]).unsqueeze(1)

    states.apply_temperature(logits, expanded_idx_mapping, torch.arange(4).numpy())

    assert torch.equal(logits, expected)


def test_sampling_states_apply_min_p() -> None:
    states = CPUSamplingStates(4, 4, torch.device("cpu"))
    min_p = torch.tensor([0.0, 0.25, 0.5, 1.0])
    states.min_p.copy_(min_p)

    logits = torch.tensor(
        [
            [2.0, 1.0, 0.0, -1.0],
            [3.0, 2.0, 1.0, 0.0],
            [2.0, 2.0, 1.0, 0.0],
            [4.0, 3.0, 2.0, 1.0],
        ]
    )
    expanded_idx_mapping = torch.tensor([2, 0, 3, 1], dtype=torch.int64)
    expected = torch.tensor(
        [
            [2.0, -torch.inf, -torch.inf, -torch.inf],
            [3.0, 2.0, 1.0, 0.0],
            [2.0, 2.0, -torch.inf, -torch.inf],
            [4.0, 3.0, -torch.inf, -torch.inf],
        ]
    )

    states.apply_min_p(logits, expanded_idx_mapping, torch.arange(4).numpy())

    assert torch.equal(logits, expected)


@pytest.mark.parametrize("use_fp64", [False, True])
def test_sampling_matches_target_distribution(use_fp64: bool) -> None:
    counts = _make_heavy_tailed_counts()
    total = counts.sum().item()
    sampled = _sample_many(_counts_to_logits(counts), NUM_SAMPLES, use_fp64=use_fp64)

    assert sampled.min() >= 0 and sampled.max() < VOCAB_SIZE
    tail_probability = (total - counts[0].item()) / total
    tail_count = (sampled != 0).sum().item()
    z_score = _z_score(tail_count, NUM_SAMPLES * tail_probability, NUM_SAMPLES)
    assert abs(z_score) < Z_TOLERANCE


def test_full_vocab_distribution_fidelity() -> None:
    counts = torch.randint(
        500,
        1500,
        (VOCAB_SIZE,),
        generator=torch.Generator().manual_seed(2024),
        dtype=torch.int64,
    )
    total = counts.sum().item()
    num_samples = 4_000_000
    histogram = _sample_histogram(_counts_to_logits(counts), num_samples)

    coverage = (histogram > 0).sum().item() / VOCAB_SIZE
    assert coverage > 0.99

    expected = counts.double() / total * num_samples
    chi_square = (((histogram - expected) ** 2) / expected).sum().item()
    degrees_of_freedom = VOCAB_SIZE - 1
    assert chi_square < degrees_of_freedom + 10 * math.sqrt(2 * degrees_of_freedom)


def test_greedy_temperature_zero_returns_argmax() -> None:
    torch.manual_seed(0)
    num_requests = 128
    logits = torch.randn(num_requests, VOCAB_SIZE)
    mapping = torch.arange(num_requests, dtype=torch.int64)
    temperature = torch.zeros(num_requests)
    seeds = torch.arange(num_requests, dtype=torch.int64)
    positions = torch.arange(num_requests, dtype=torch.int64)

    sampled = _sample(logits, mapping, temperature, seeds, positions)

    assert torch.equal(sampled, logits.argmax(dim=-1))


def test_zero_count_tokens_are_never_sampled() -> None:
    counts = _make_heavy_tailed_counts(seed=7)
    counts[1::2] = 0
    sampled = _sample_many(_counts_to_logits(counts), NUM_SAMPLES)

    zero_count = torch.zeros(VOCAB_SIZE, dtype=torch.bool)
    zero_count[1::2] = True
    assert sampled.min() >= 0 and sampled.max() < VOCAB_SIZE
    assert not zero_count[sampled].any()


def test_single_nonzero_token_is_always_sampled() -> None:
    counts = torch.zeros(VOCAB_SIZE, dtype=torch.int64)
    counts[123_456] = 1000

    sampled = _sample_many(_counts_to_logits(counts), 10_000)

    assert (sampled == 123_456).all()


@pytest.mark.parametrize("vocab_size", [1, 999, 1024, 4097])
def test_vocab_size_not_multiple_of_block(vocab_size: int) -> None:
    counts = torch.randint(
        20,
        200,
        (vocab_size,),
        generator=torch.Generator().manual_seed(vocab_size),
        dtype=torch.int64,
    )
    total = counts.sum().item()
    num_samples = max(40 * vocab_size, 50_000)
    sampled = _sample_many(_counts_to_logits(counts), num_samples)

    assert sampled.min() >= 0 and sampled.max() < vocab_size
    observed = torch.bincount(sampled, minlength=vocab_size).double()
    expected = counts.double() / total * num_samples
    degrees_of_freedom = vocab_size - 1
    if degrees_of_freedom > 0:
        chi_square = (((observed - expected) ** 2) / expected).sum().item()
        assert chi_square < degrees_of_freedom + 10 * math.sqrt(2 * degrees_of_freedom)
