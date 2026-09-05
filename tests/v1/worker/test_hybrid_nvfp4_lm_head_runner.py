# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from torch import nn

import vllm.envs as envs
from vllm.model_executor.layers.hybrid_nvfp4_lm_head import (
    HybridNvfp4LmHead,
    _attach_state,
)
from vllm.v1.worker.gpu.model_runner import GPUModelRunner as V2GPUModelRunner
from vllm.v1.worker.gpu_model_runner import GPUModelRunner as LegacyGPUModelRunner


def _legacy_runner_stub(
    *,
    top_k: int = 8,
    top_p: float = 0.9,
    temperature: float = 0.7,
    presence: float = 0.0,
    repetition: float = 1.0,
    frequency: float = 0.0,
    num_reqs: int = 1,
):
    sampling_metadata = SimpleNamespace(
        all_random=True,
        generators=[],
        no_penalties=(
            presence == 0.0 and repetition == 1.0 and frequency == 0.0
        ),
        max_num_logprobs=None,
        logprob_token_ids=[],
        allowed_token_ids_mask=None,
        bad_words_token_ids=[],
        logitsprocs=SimpleNamespace(all=[]),
    )
    input_batch = SimpleNamespace(
        num_reqs=num_reqs,
        sampling_metadata=sampling_metadata,
        top_k_cpu=np.full(num_reqs, top_k, dtype=np.int32),
        top_p_cpu=np.full(num_reqs, top_p, dtype=np.float32),
        temperature_cpu=np.full(num_reqs, temperature, dtype=np.float32),
        repetition_penalties_cpu=np.full(num_reqs, repetition, dtype=np.float32),
        frequency_penalties_cpu=np.full(num_reqs, frequency, dtype=np.float32),
        presence_penalties_cpu=np.full(num_reqs, presence, dtype=np.float32),
    )
    lm_head = SimpleNamespace(
        num_added_embeddings=0,
        shard_indices=SimpleNamespace(num_added_elements=0),
        _hybrid_nvfp4_lm_head_state=object(),
    )
    return SimpleNamespace(
        input_batch=input_batch,
        model_config=SimpleNamespace(head_dtype=torch.bfloat16),
        dtype=torch.bfloat16,
        lora_config=None,
        sampler=SimpleNamespace(use_fp64_gumbel=False, logprobs_mode=None),
        model=SimpleNamespace(lm_head=lm_head, sample_topk_tokens=Mock()),
        broadcast_pp_output=False,
        vocab_size=128,
    )


def test_legacy_topk_gate_rejects_fp64_gumbel(monkeypatch):
    runner = _legacy_runner_stub()
    monkeypatch.setattr(envs, "VLLM_HYBRID_NVFP4_LM_HEAD", True)
    runner.sampler.use_fp64_gumbel = True

    assert (
        LegacyGPUModelRunner._get_hybrid_topk_sampling_params(
            runner, SimpleNamespace(has_structured_output_requests=False), None
        )
        is None
    )


def test_legacy_topk_gate_accepts_presence_only_and_rejects_other_penalties(
    monkeypatch,
):
    monkeypatch.setattr(envs, "VLLM_HYBRID_NVFP4_LM_HEAD", True)
    runner = _legacy_runner_stub(presence=0.5)
    params = LegacyGPUModelRunner._get_hybrid_topk_sampling_params(
        runner, SimpleNamespace(has_structured_output_requests=False), None
    )
    assert params == (8, pytest.approx(0.9), pytest.approx(0.7), True)

    runner = _legacy_runner_stub(frequency=0.1)
    assert (
        LegacyGPUModelRunner._get_hybrid_topk_sampling_params(
            runner, SimpleNamespace(has_structured_output_requests=False), None
        )
        is None
    )

    runner = _legacy_runner_stub(repetition=1.1)
    assert (
        LegacyGPUModelRunner._get_hybrid_topk_sampling_params(
            runner, SimpleNamespace(has_structured_output_requests=False), None
        )
        is None
    )


def test_legacy_presence_output_table_deduplicates_and_filters_invalid_ids(
    monkeypatch,
):
    # Keep this metadata-only test runnable on a CPU-only CI worker even when
    # the process was configured with the CUDA target device.
    monkeypatch.setattr("vllm.v1.worker.gpu_model_runner.PIN_MEMORY", False)
    runner = SimpleNamespace(
        input_batch=SimpleNamespace(
            num_reqs=2,
            num_prompt_tokens=np.array([2, 1], dtype=np.int32),
            num_tokens_no_spec=np.array([8, 6], dtype=np.int32),
            token_ids_cpu=np.array(
                [
                    [90, 91, 5, 5, 7, -1, 128, 9],
                    [92, 3, 4, 3, 6, 6, 0, 0],
                ],
                dtype=np.int64,
            ),
        ),
        vocab_size=128,
    )

    output_ids = LegacyGPUModelRunner._make_hybrid_presence_output_token_ids(
        runner, torch.zeros((2, 4), dtype=torch.bfloat16)
    )

    assert torch.equal(
        output_ids,
        torch.tensor([[5, 7, 9], [3, 4, 6]], dtype=torch.int64),
    )


def test_legacy_presence_output_table_rejects_chunked_prefill_rows():
    runner = SimpleNamespace(
        input_batch=SimpleNamespace(
            num_reqs=2,
            num_prompt_tokens=np.array([0, 0], dtype=np.int32),
            num_tokens_no_spec=np.array([1, 1], dtype=np.int32),
            token_ids_cpu=np.zeros((2, 1), dtype=np.int64),
        ),
        vocab_size=128,
    )

    assert (
        LegacyGPUModelRunner._make_hybrid_presence_output_token_ids(
            runner, torch.zeros((3, 4), dtype=torch.bfloat16)
        )
        is None
    )


class _FakeV2Model(nn.Module):
    def __init__(self, *, state: object | None = object()) -> None:
        super().__init__()
        self.lm_head = nn.Module()
        self.lm_head.num_added_embeddings = 0
        self.lm_head.shard_indices = SimpleNamespace(num_added_elements=0)
        self.lm_head._hybrid_nvfp4_lm_head_state = state
        self.topk_calls: list[dict[str, object]] = []
        self.greedy_calls = 0
        self.compute_logits_calls = 0

    def sample_topk_tokens(self, hidden_states, **kwargs):
        self.topk_calls.append({"hidden_states": hidden_states, **kwargs})
        return torch.arange(hidden_states.shape[0], dtype=torch.int64)

    def get_top_tokens(self, hidden_states):
        self.greedy_calls += 1
        return torch.full((hidden_states.shape[0],), 7, dtype=torch.int64)

    def compute_logits(self, hidden_states):
        self.compute_logits_calls += 1
        return torch.zeros((hidden_states.shape[0], 8), dtype=torch.float32)


class _FakeV2Sampler:
    def __init__(self, params, *, presence_inputs=None):
        self.params = params
        self.presence_inputs = presence_inputs
        self.sampling_states = SimpleNamespace(
            seeds=SimpleNamespace(gpu=torch.tensor([123, 456], dtype=torch.int64)),
            temperature=SimpleNamespace(gpu=torch.tensor([0.7, 0.7])),
        )
        self.make_calls = []
        self.fallback_calls = 0

    def get_vocab_parallel_sampling_params(self, _input_batch):
        return self.params

    def get_vocab_parallel_presence_inputs(self, _input_batch):
        assert self.presence_inputs is not None
        return self.presence_inputs

    def make_sampler_output(self, sampled, _input_batch):
        self.make_calls.append(sampled)
        return SimpleNamespace(
            sampled_token_ids=sampled,
            num_sampled=torch.ones(sampled.shape[0], dtype=torch.int64),
            num_rejected=torch.zeros(sampled.shape[0], dtype=torch.int64),
        )

    def __call__(self, _logits, _input_batch):
        self.fallback_calls += 1
        return SimpleNamespace(
            sampled_token_ids=torch.tensor([[2]], dtype=torch.int64),
            num_sampled=torch.ones(1, dtype=torch.int64),
            num_rejected=torch.zeros(1, dtype=torch.int64),
        )


def _v2_input_batch(num_rows: int = 2):
    return SimpleNamespace(
        logits_indices=torch.arange(num_rows, dtype=torch.int64),
        num_reqs=num_rows,
        num_draft_tokens=0,
        expanded_idx_mapping=torch.arange(num_rows, dtype=torch.int64),
        positions=torch.arange(10, 10 + num_rows, dtype=torch.int64),
        has_structured_output_reqs=False,
    )


def _v2_runner(model, sampler):
    return SimpleNamespace(
        model=model,
        sampler=sampler,
        batch_sharder=None,
        lora_config=None,
        adaptive_verification=None,
        model_config=SimpleNamespace(head_dtype=torch.bfloat16),
        dtype=torch.bfloat16,
        vocab_size=128,
        device=torch.device("cpu"),
        rejection_sampler=None,
        speculator=None,
    )


def test_v2_topk_presence_dispatch_passes_persistent_metadata(monkeypatch):
    monkeypatch.setattr(envs, "VLLM_HYBRID_NVFP4_LM_HEAD", True)
    sampler = _FakeV2Sampler(
        ("topk", 8, 0.9, 0.7, True),
        presence_inputs=(
            torch.tensor([0.5, 0.5]),
            torch.tensor([[1, 0, 0], [0, 2, 0]], dtype=torch.int32),
            torch.tensor([0, 1], dtype=torch.int64),
        ),
    )
    model = _FakeV2Model()
    runner = _v2_runner(model, sampler)

    output, _, _ = V2GPUModelRunner.sample(
        runner,
        torch.zeros((2, 4), dtype=torch.bfloat16),
        _v2_input_batch(),
        grammar_output=None,
    )

    assert len(model.topk_calls) == 1
    call = model.topk_calls[0]
    assert torch.equal(call["presence_penalties"], torch.tensor([0.5, 0.5]))
    assert torch.equal(
        call["output_token_counts"],
        torch.tensor([[1, 0, 0], [0, 2, 0]], dtype=torch.int32),
    )
    assert torch.equal(
        call["presence_request_indices"], torch.tensor([0, 1], dtype=torch.int64)
    )
    assert output.sampled_token_ids.tolist() == [0, 1]


def test_v2_greedy_speculative_dispatch_uses_target_fast_path(monkeypatch):
    monkeypatch.setattr(envs, "VLLM_HYBRID_NVFP4_LM_HEAD", True)
    sampler = _FakeV2Sampler(("greedy", 128, 1.0, 0.0, False))
    rejection_sampler = Mock()
    rejection_sampler.draft_logits = None
    rejection_sampler.synthetic_conditional_rates = None
    rejection_sampler.use_block_verification = False
    rejection_sampler.sample_from_greedy_tokens.return_value = SimpleNamespace(
        sampled_token_ids=torch.tensor([[7]], dtype=torch.int64),
        num_sampled=torch.ones(1, dtype=torch.int64),
        num_rejected=torch.zeros(1, dtype=torch.int64),
    )
    model = _FakeV2Model()
    runner = _v2_runner(model, sampler)
    runner.rejection_sampler = rejection_sampler
    runner.speculator = SimpleNamespace(draft_logits=None)
    input_batch = _v2_input_batch(1)
    input_batch.num_draft_tokens = 2

    output, _, _ = V2GPUModelRunner.sample(
        runner,
        torch.zeros((1, 4), dtype=torch.bfloat16),
        input_batch,
        grammar_output=None,
    )

    assert model.greedy_calls == 1
    rejection_sampler.sample_from_greedy_tokens.assert_called_once()
    assert output.sampled_token_ids.tolist() == [[7]]


def test_v2_incompatible_runtime_config_releases_hybrid_state(monkeypatch):
    monkeypatch.setattr(envs, "VLLM_HYBRID_NVFP4_LM_HEAD", True)
    monkeypatch.setattr(envs, "VLLM_COMPUTE_NANS_IN_LOGITS", True)
    model = _FakeV2Model(state=None)
    state = HybridNvfp4LmHead(
        weight=torch.ones((4, 2), dtype=torch.uint8),
        scale=torch.ones((4, 1), dtype=torch.float32),
        global_scale=torch.ones((), dtype=torch.float32),
        input_size=4,
        output_size=4,
        candidates=2,
    )
    model.lm_head.weight = torch.zeros((4, 4), dtype=torch.bfloat16)
    _attach_state(model.lm_head, state)
    sampler = _FakeV2Sampler(None)
    runner = _v2_runner(model, sampler)

    output, _, _ = V2GPUModelRunner.sample(
        runner,
        torch.zeros((1, 4), dtype=torch.bfloat16),
        _v2_input_batch(1),
        grammar_output=None,
    )

    assert not hasattr(model.lm_head, "_hybrid_nvfp4_lm_head_state")
    assert model.compute_logits_calls == 1
    assert sampler.fallback_calls == 1
    assert output.sampled_token_ids.tolist() == [[2]]
