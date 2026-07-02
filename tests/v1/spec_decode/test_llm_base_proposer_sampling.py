# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import dataclasses

import torch

from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed
from vllm.v1.sample.logits_processor import LogitsProcessors
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.spec_decode import llm_base_proposer
from vllm.v1.spec_decode.llm_base_proposer import (
    SpecDecodeBaseProposer,
    compute_probs_and_sample_next_token,
)

DEVICE_TYPE = current_platform.device_type


def _seed_default_generator(seed: int) -> None:
    set_random_seed(seed)


def _make_sampling_metadata(batch_size: int) -> SamplingMetadata:
    return SamplingMetadata(
        temperature=torch.ones(batch_size, dtype=torch.float32, device=DEVICE_TYPE),
        all_greedy=False,
        all_random=True,
        top_p=None,
        top_k=None,
        generators={},
        max_num_logprobs=None,
        no_penalties=True,
        prompt_token_ids=None,
        frequency_penalties=torch.empty(0, device=DEVICE_TYPE),
        presence_penalties=torch.empty(0, device=DEVICE_TYPE),
        repetition_penalties=torch.empty(0, device=DEVICE_TYPE),
        output_token_ids=[[] for _ in range(batch_size)],
        spec_token_ids=[[] for _ in range(batch_size)],
        allowed_token_ids_mask=None,
        bad_words_token_ids={},
        logitsprocs=LogitsProcessors(),
    )


def test_compute_probs_and_sample_next_token_uses_fp64_exponential_race():
    batch_size = 4
    vocab_size = 32
    generator = torch.Generator(device=DEVICE_TYPE).manual_seed(11)
    logits = torch.randn(
        batch_size,
        vocab_size,
        dtype=torch.float32,
        device=DEVICE_TYPE,
        generator=generator,
    )
    metadata = _make_sampling_metadata(batch_size)

    _seed_default_generator(12345)
    probs = logits.softmax(dim=-1, dtype=torch.float32)
    q = torch.empty(probs.shape, dtype=torch.float64, device=probs.device)
    q.exponential_()
    expected_ids = q.reciprocal_().mul_(probs).argmax(dim=-1).view(-1)

    _seed_default_generator(12345)
    actual_ids, actual_probs = compute_probs_and_sample_next_token(
        logits.clone(),
        metadata,
        use_fp64_gumbel=True,
    )

    assert torch.equal(actual_ids, expected_ids)
    assert torch.allclose(actual_probs, probs)


def test_sample_from_logits_does_not_prerepeat_temperature(monkeypatch):
    """Regression: ``_sample_from_logits`` must pass the PER-REQUEST sampling
    metadata through unchanged for parallel drafting (logits has batch_size * K
    rows). If it pre-repeats ``temperature`` to the row count,
    ``compute_probs_and_sample_next_token`` sees ``temperature.shape[0] ==
    num_tokens`` and skips the K-fold seeded-generator expansion, so a seeded
    request's later draft slots fall back to the default RNG (broken determinism).
    """
    batch_size, k, vocab_size = 2, 3, 8
    logits = torch.randn(
        batch_size * k, vocab_size, dtype=torch.float32, device=DEVICE_TYPE
    )
    metadata = _make_sampling_metadata(batch_size)  # per-request temperature [bs]

    captured: dict[str, int] = {}

    def _spy(logits_arg, sampling_metadata, use_fp64_gumbel=False):
        captured["temperature_rows"] = sampling_metadata.temperature.shape[0]
        return logits_arg.argmax(dim=-1), logits_arg

    monkeypatch.setattr(
        llm_base_proposer, "compute_probs_and_sample_next_token", _spy
    )

    class _Stub:
        _enable_probabilistic_draft_probs = True
        use_fp64_gumbel = False

    SpecDecodeBaseProposer._sample_from_logits(_Stub(), logits, metadata)

    # Per-request temperature (batch_size) must reach the sampler, NOT the
    # repeated batch_size * k. (Bug pre-repeated -> would be batch_size * k.)
    assert captured["temperature_rows"] == batch_size


def test_parallel_draft_seeded_generators_are_deterministic_per_request():
    """With per-request temperature + per-request seeded generators and
    parallel-draft logits (batch_size * K request-major rows), every draft slot
    of a seeded request is driven by that request's generator, so re-running
    with the same seeds is fully deterministic across ALL rows. Under the old
    bug the K-1 trailing slots used the default RNG and were non-deterministic.
    """
    batch_size, k, vocab_size = 2, 3, 32
    logits = torch.randn(
        batch_size * k, vocab_size, dtype=torch.float32, device=DEVICE_TYPE
    )

    def _seeded_metadata() -> SamplingMetadata:
        gens = {
            r: torch.Generator(device=DEVICE_TYPE).manual_seed(31 + r)
            for r in range(batch_size)
        }
        return dataclasses.replace(
            _make_sampling_metadata(batch_size), generators=gens
        )

    out1, _ = compute_probs_and_sample_next_token(
        logits.clone(), _seeded_metadata(), use_fp64_gumbel=False
    )
    out2, _ = compute_probs_and_sample_next_token(
        logits.clone(), _seeded_metadata(), use_fp64_gumbel=False
    )
    assert out1.shape[0] == batch_size * k
    assert torch.equal(out1, out2)
