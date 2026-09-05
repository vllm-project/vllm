# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import patch

import numpy as np
import pytest
import torch

pytest.importorskip("triton")
if not torch.cuda.is_available():
    pytest.skip("CUDA required for sampler flag tests", allow_module_level=True)

import vllm.v1.worker.gpu.sample.sampler as sampler_module
from vllm.sampling_params import SamplingParams
from vllm.v1.worker.gpu.sample.sampler import Sampler
from vllm.v1.worker.gpu.spec_decode.rejection_sampler import RejectionSampler
from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import (
    reduced_greedy_verify,
)
from vllm.v1.worker.gpu.states import RequestState

DEVICE = torch.device("cuda")
VOCAB_SIZE = 128


class MockReasoningConfig:
    reasoning_start_token_ids = [90]
    reasoning_end_token_ids = [91]
    natural_reasoning_end_token_ids = [91]


def _make_sampler() -> Sampler:
    req_states = RequestState(
        max_num_reqs=4,
        max_model_len=64,
        max_num_batched_tokens=16,
        num_speculative_steps=1,
        vocab_size=VOCAB_SIZE,
        device=DEVICE,
    )
    with patch.object(
        sampler_module,
        "get_tensor_model_parallel_world_size",
        return_value=2,
    ):
        return Sampler(
            max_num_reqs=4,
            vocab_size=VOCAB_SIZE,
            device=DEVICE,
            req_states=req_states,
            reasoning_config=MockReasoningConfig(),
        )


def _can_use_reduced_sampling(
    sampler: Sampler,
    idx_mapping_np: np.ndarray,
    local_vocab_size: int = 16,
    dtype: torch.dtype = torch.float32,
) -> bool:
    local_logits = torch.empty(
        (len(idx_mapping_np), local_vocab_size), dtype=dtype, device=DEVICE
    )
    return sampler.can_use_reduced_sampling(idx_mapping_np, local_logits)


@pytest.mark.parametrize(
    ("sampling_params", "expected"),
    [
        pytest.param(SamplingParams(), False, id="defaults"),
        pytest.param(SamplingParams(temperature=0.0), False, id="greedy"),
        pytest.param(
            SamplingParams(thinking_token_budget=3), True, id="thinking-budget"
        ),
        pytest.param(SamplingParams(logit_bias={1: 1.0}), True, id="logit-bias"),
        pytest.param(SamplingParams(frequency_penalty=0.1), True, id="penalty"),
        pytest.param(SamplingParams(_bad_words_token_ids=[[1]]), True, id="bad-words"),
        pytest.param(SamplingParams(temperature=0.7), True, id="temperature"),
        pytest.param(SamplingParams(min_p=0.1), True, id="min-p"),
        pytest.param(SamplingParams(top_k=10), True, id="top-k"),
        pytest.param(SamplingParams(top_p=0.9), True, id="top-p"),
        pytest.param(
            SamplingParams.for_sampler_warmup(), True, id="all-logits-processors"
        ),
    ],
)
def test_logits_processing_cache_matches_request_features(
    sampling_params: SamplingParams, expected: bool
):
    sampler = _make_sampler()
    sampler.add_request(3, prompt_len=1, sampling_params=sampling_params)

    assert sampler.needs_logits_processing[3] == expected


def test_logits_processing_cache_is_overwritten_when_slot_is_reused():
    sampler = _make_sampler()
    sampler.add_request(3, 1, SamplingParams.for_sampler_warmup())
    sampler.add_request(3, 1, SamplingParams())

    assert not sampler.needs_logits_processing[3]


def test_logits_processing_cache_only_checks_active_requests():
    sampler = _make_sampler()
    sampler.add_request(0, 1, SamplingParams(temperature=0.0))
    sampler.add_request(2, 1, SamplingParams.for_sampler_warmup())

    sampling_only = np.array([0], dtype=np.int32)
    with_processing = np.array([0, 2], dtype=np.int32)

    assert not np.any(sampler.needs_logits_processing[sampling_only])
    assert np.any(sampler.needs_logits_processing[with_processing])


def test_reduced_sampling_requires_bounded_top_k():
    sampler = _make_sampler()
    idx_mapping_np = np.array([0], dtype=np.int32)

    sampler.add_request(0, 1, SamplingParams(temperature=0.7, top_p=0.9))
    assert not _can_use_reduced_sampling(sampler, idx_mapping_np)

    sampler.add_request(0, 1, SamplingParams(temperature=0.7, top_k=2, top_p=0.9))
    assert _can_use_reduced_sampling(sampler, idx_mapping_np)

    sampler.add_request(0, 1, SamplingParams(temperature=0.7, top_k=8))
    assert not _can_use_reduced_sampling(sampler, idx_mapping_np)


def test_reduced_sampling_ignores_greedy_top_k_in_mixed_batch():
    sampler = _make_sampler()
    sampler.add_request(0, 1, SamplingParams(temperature=0.0))
    sampler.add_request(1, 1, SamplingParams(temperature=0.7, top_k=2))

    assert _can_use_reduced_sampling(sampler, np.array([0, 1], dtype=np.int32))


@pytest.mark.parametrize("attribute", ["return_sampling_mask", "compute_nans"])
def test_reduced_sampling_falls_back_for_full_vocab_outputs(attribute):
    sampler = _make_sampler()
    sampler.add_request(0, 1, SamplingParams(temperature=0.0))
    setattr(sampler, attribute, True)

    assert not _can_use_reduced_sampling(sampler, np.array([0], dtype=np.int32))


def test_reduced_sampling_falls_back_for_inexact_fp32_token_ids():
    sampler = _make_sampler()
    sampler.add_request(0, 1, SamplingParams(temperature=0.0))
    sampler.sampling_states.vocab_size = 2**24

    assert not _can_use_reduced_sampling(sampler, np.array([0], dtype=np.int32))


@pytest.mark.parametrize(
    "sampling_params",
    [
        SamplingParams(temperature=0.0, logit_bias={1: 1.0}),
        SamplingParams(temperature=0.0, frequency_penalty=0.1),
        SamplingParams(temperature=0.0, _bad_words_token_ids=[[1]]),
        SamplingParams(temperature=0.0, thinking_token_budget=3),
        SamplingParams(temperature=0.7, top_k=2, min_p=0.1),
        SamplingParams(temperature=0.0, logprobs=1),
        SamplingParams(temperature=0.0, logprob_token_ids=[1]),
    ],
)
def test_reduced_sampling_falls_back_for_full_vocab_features(sampling_params):
    sampler = _make_sampler()
    sampler.add_request(0, 1, sampling_params)

    assert not _can_use_reduced_sampling(sampler, np.array([0], dtype=np.int32))


def test_reduced_sampling_falls_back_for_explicit_random_seed():
    sampler = _make_sampler()
    sampler.add_request(
        0,
        1,
        SamplingParams(temperature=0.7, top_k=2, seed=1),
    )

    assert not _can_use_reduced_sampling(sampler, np.array([0], dtype=np.int32))


def test_reduced_sampling_accounts_for_logits_dtype_in_traffic_check():
    sampler = _make_sampler()
    sampler.add_request(0, 1, SamplingParams(temperature=0.7, top_k=4))
    idx_mapping_np = np.array([0], dtype=np.int32)

    assert _can_use_reduced_sampling(sampler, idx_mapping_np, dtype=torch.float32)
    assert not _can_use_reduced_sampling(sampler, idx_mapping_np, dtype=torch.bfloat16)


def test_reduced_greedy_requires_smaller_candidate_payload():
    sampler = _make_sampler()
    sampler.add_request(0, 1, SamplingParams(temperature=0.0))
    idx_mapping_np = np.array([0], dtype=np.int32)

    assert not _can_use_reduced_sampling(sampler, idx_mapping_np, local_vocab_size=2)
    assert _can_use_reduced_sampling(sampler, idx_mapping_np, local_vocab_size=3)


def test_reduced_speculative_verification_requires_all_greedy():
    sampler = _make_sampler()
    sampler.add_request(0, 1, SamplingParams(temperature=0.0))
    sampler.add_request(1, 1, SamplingParams(temperature=0.7))
    sampler.sampling_states.apply_staged_writes()
    rejection_sampler = object.__new__(RejectionSampler)
    rejection_sampler.sampler = sampler
    rejection_sampler.draft_sample_method = "greedy"
    rejection_sampler.synthetic_conditional_rates = None

    assert rejection_sampler.can_use_reduced_sampling(np.array([0], dtype=np.int32))
    assert not rejection_sampler.can_use_reduced_sampling(
        np.array([0, 1], dtype=np.int32)
    )

    rejection_sampler.draft_sample_method = "probabilistic"
    assert not rejection_sampler.can_use_reduced_sampling(np.array([0], dtype=np.int32))

    rejection_sampler.draft_sample_method = "greedy"
    rejection_sampler.synthetic_conditional_rates = torch.ones(1, device=DEVICE)
    assert not rejection_sampler.can_use_reduced_sampling(np.array([0], dtype=np.int32))


def test_reduced_greedy_verify_accepts_prefix_and_emits_target():
    # Request 0 accepts token 5, then rejects token 8 in favor of token 6.
    # Request 1 accepts token 3 and emits bonus token 4.
    target_argmax = torch.tensor([5, 6, 7, 3, 4], device=DEVICE)
    draft_sampled = torch.tensor([99, 5, 8, 99, 3], device=DEVICE)
    cu_num_logits = torch.tensor([0, 3, 5], device=DEVICE, dtype=torch.int32)

    sampled, num_sampled = reduced_greedy_verify(
        target_argmax, draft_sampled, cu_num_logits, num_speculative_steps=2
    )

    assert num_sampled.tolist() == [2, 2]
    assert sampled[0, :2].tolist() == [5, 6]
    assert sampled[1, :2].tolist() == [3, 4]


def test_reduced_random_sampling_maps_global_candidate(monkeypatch):
    sampler = _make_sampler()
    sampler.tp_size = 2
    sampler.add_request(0, 1, SamplingParams(temperature=0.7, top_k=2))
    sampler.sampling_states.apply_staged_writes()
    idx_mapping_np = np.array([0], dtype=np.int32)
    expanded_idx_mapping = torch.tensor([0], device=DEVICE)
    local_logits = torch.tensor([[0.1, 0.9, 0.8, 0.2]], device=DEVICE)
    top_k, top_p = sampler.sampling_states.get_top_k_top_p(
        expanded_idx_mapping, idx_mapping_np
    )

    def fake_all_gather(candidates: torch.Tensor, dim: int):
        remote = candidates.clone()
        remote[:, 2:] += local_logits.shape[-1]
        return torch.cat((candidates, remote), dim=dim)

    monkeypatch.setattr(
        sampler_module, "tensor_model_parallel_all_gather", fake_all_gather
    )
    monkeypatch.setattr(
        sampler_module,
        "gumbel_sample",
        lambda *args, **kwargs: torch.tensor([2], device=DEVICE),
    )

    sampled = sampler._sample_reduced(
        local_logits,
        top_k,
        top_p,
        expanded_idx_mapping,
        idx_mapping_np,
        sampler.sampling_states.temperature.gpu,
        sampler.sampling_states.seeds.gpu,
        torch.tensor([3], device=DEVICE),
        vocab_start_index=10,
    )

    assert sampled.tolist() == [15]


def test_full_logits_fallback_skips_reduced_sampling(monkeypatch):
    sampler = _make_sampler()
    sampler.tp_size = 2
    sampler.use_flashinfer = False
    sampler.add_request(
        0, prompt_len=1, sampling_params=SamplingParams(temperature=0.7, top_k=2)
    )
    sampler.sampling_states.apply_staged_writes()

    idx_mapping_np = np.array([0], dtype=np.int32)
    expanded_idx_mapping = torch.tensor([0], device=DEVICE)
    idx_mapping = expanded_idx_mapping.clone()
    pos = torch.tensor([3], device=DEVICE)
    input_ids = torch.tensor([1], device=DEVICE)
    expanded_local_pos = torch.tensor([0], device=DEVICE)
    full_logits = torch.zeros((1, VOCAB_SIZE), device=DEVICE)

    def fail_reduced_sample(*args, **kwargs):
        pytest.fail("Full logits must not enter the reduced sampling path")

    def fake_gumbel_sample(*args, **kwargs):
        return torch.tensor([0], device=DEVICE)

    monkeypatch.setattr(
        sampler,
        "apply_sampling_params",
        lambda logits, *args, **kwargs: logits,
    )
    monkeypatch.setattr(sampler, "_sample_reduced", fail_reduced_sample)
    monkeypatch.setattr(
        sampler_module,
        "apply_top_k_top_p",
        lambda logits, *args, **kwargs: logits,
    )
    monkeypatch.setattr(sampler_module, "gumbel_sample", fake_gumbel_sample)

    sampled, processed_logits = sampler.sample(
        full_logits,
        expanded_idx_mapping,
        idx_mapping,
        idx_mapping_np,
        pos,
        input_ids,
        expanded_local_pos,
        return_logprobs=True,
        use_reduced_sampling=False,
    )

    assert sampled.tolist() == [0]
    assert processed_logits is full_logits


def test_reduced_all_greedy_skips_topk_filter_and_gumbel(monkeypatch):
    sampler = _make_sampler()
    sampler.tp_size = 2
    sampler.add_request(
        0, prompt_len=1, sampling_params=SamplingParams(temperature=0.0)
    )
    sampler.sampling_states.apply_staged_writes()
    idx_mapping_np = np.array([0], dtype=np.int32)
    expanded_idx_mapping = torch.tensor([0], device=DEVICE)
    pos = torch.tensor([3], device=DEVICE)
    local_logits = torch.tensor([[0.1, 0.9, 0.8, 0.2]], device=DEVICE)

    def fake_all_gather(packed_candidates: torch.Tensor, dim: int):
        assert dim == -1
        assert packed_candidates.shape == (1, 2)
        remote_candidates = torch.tensor([[1.1, 5.0]], device=DEVICE)
        return torch.cat((packed_candidates, remote_candidates), dim=dim)

    def fail_redundant_path(*args, **kwargs):
        pytest.fail("all-greedy reduced sampling must only use max/argmax")

    monkeypatch.setattr(
        sampler_module, "tensor_model_parallel_all_gather", fake_all_gather
    )
    monkeypatch.setattr(torch, "topk", fail_redundant_path)
    monkeypatch.setattr(sampler_module, "apply_top_k_top_p", fail_redundant_path)
    monkeypatch.setattr(sampler_module, "gumbel_sample", fail_redundant_path)
    monkeypatch.setattr(sampler_module, "flashinfer_sample", fail_redundant_path)

    sampled = sampler._sample_reduced(
        local_logits,
        top_k=None,
        top_p=None,
        expanded_idx_mapping=expanded_idx_mapping,
        idx_mapping_np=idx_mapping_np,
        temperature=sampler.sampling_states.temperature.gpu,
        seeds=sampler.sampling_states.seeds.gpu,
        pos=pos,
        vocab_start_index=0,
    )

    assert sampled.tolist() == [5]
