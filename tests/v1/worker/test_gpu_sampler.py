# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm import SamplingParams
from vllm.v1.worker.gpu.sample import sampler as sampler_module
from vllm.v1.worker.gpu.sample.penalties import PenaltiesState
from vllm.v1.worker.gpu.sample.sampler import Sampler
from vllm.v1.worker.gpu.sample.states import SamplingStates


def _new_rapid_sampler(
    *,
    rapid_penalties: torch.Tensor,
    all_token_ids: torch.Tensor | None = None,
) -> Sampler:
    sampler = object.__new__(Sampler)
    sampler.rapid_penalties = rapid_penalties
    sampler.req_states = SimpleNamespace(
        max_num_reqs=rapid_penalties.shape[0],
        vocab_size=rapid_penalties.shape[1],
        device=rapid_penalties.device,
        all_token_ids=SimpleNamespace(gpu=all_token_ids),
    )
    return sampler


def test_worker_sampler_seeds_rapid_repetition_penalty_from_prompt():
    all_token_ids = torch.tensor(
        [
            [2, 5, 2, 99],
            [0, 0, 0, 0],
        ],
        dtype=torch.int32,
    )
    rapid_penalties = torch.zeros(2, 8, dtype=torch.float32)
    sampler = _new_rapid_sampler(
        rapid_penalties=rapid_penalties,
        all_token_ids=all_token_ids,
    )

    sampler._seed_rapid_prompt_penalties(
        req_idx=0,
        prompt_len=4,
        sampling_params=SamplingParams(repetition_penalty=1.2),
    )

    assert torch.isclose(rapid_penalties[0, 2], torch.tensor(1.2))
    assert torch.isclose(rapid_penalties[0, 5], torch.tensor(1.2))
    assert torch.count_nonzero(rapid_penalties[0]) == 2


def test_worker_sampler_clears_rapid_penalties_when_reusing_slot():
    rapid_penalties = torch.full((1, 8), 3.0)
    sampler = _new_rapid_sampler(rapid_penalties=rapid_penalties)
    sampler.use_rapid = True
    sampler.require_rapid = False
    for name in (
        "sampling_states",
        "penalties_state",
        "logit_bias_state",
        "bad_words_state",
        "logprob_token_ids_state",
    ):
        setattr(sampler, name, SimpleNamespace(add_request=lambda *args: None))

    sampler.add_request(0, 0, SamplingParams())

    assert torch.count_nonzero(rapid_penalties) == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="UVA tensors require CUDA")
def test_sampling_states_can_collapse_uniform_rapid_params():
    states = SamplingStates(max_num_reqs=4, vocab_size=8)
    for req_idx in range(3):
        states.add_request(
            req_idx,
            SamplingParams(temperature=1.0, top_k=4, top_p=0.28),
        )
    states.apply_staged_writes()

    device = states.temperature.gpu.device
    expanded_idx_mapping = torch.tensor([0, 1, 2], dtype=torch.int32, device=device)
    idx_mapping_np = np.array([0, 1, 2])

    temperatures = states.get_temperatures(
        expanded_idx_mapping,
        idx_mapping_np,
        scalar_if_uniform=True,
    )
    top_k, top_p = states.get_top_k_top_p(
        expanded_idx_mapping,
        idx_mapping_np,
        scalar_if_uniform=True,
    )

    assert temperatures == pytest.approx(1.0)
    assert top_k == 4
    assert top_p == pytest.approx(0.28)

    vector_top_k, vector_top_p = states.get_top_k_top_p(
        expanded_idx_mapping,
        idx_mapping_np,
    )

    assert vector_top_k is not None and vector_top_k.shape == (3,)
    assert vector_top_p is not None and vector_top_p.shape == (3,)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="UVA tensors require CUDA")
def test_penalties_state_can_collapse_uniform_rapid_params():
    req_states = SimpleNamespace(
        max_num_reqs=4,
        vocab_size=8,
        device=torch.device("cuda"),
    )
    state = PenaltiesState(req_states)
    state.presence_penalty.np[:3] = 0.2
    state.repetition_penalty.np[:3] = 1.1
    state.penalty_decay.np[:3] = 0.95
    state.presence_penalty.copy_to_uva()
    state.repetition_penalty.copy_to_uva()
    state.penalty_decay.copy_to_uva()

    device = state.presence_penalty.gpu.device
    expanded_idx_mapping = torch.tensor([0, 1, 2], dtype=torch.int32, device=device)
    idx_mapping_np = np.array([0, 1, 2])

    presence, repetition, decay = state.rapid_penalty_params(
        expanded_idx_mapping,
        idx_mapping_np,
        scalar_if_uniform=True,
    )

    assert presence == pytest.approx(0.2)
    assert repetition == pytest.approx(1.1)
    assert decay == pytest.approx(0.95)

    vector_presence, vector_repetition, vector_decay = state.rapid_penalty_params(
        expanded_idx_mapping,
        idx_mapping_np,
    )

    assert vector_presence.shape == (3,)
    assert vector_repetition.shape == (3,)
    assert vector_decay.shape == (3,)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="UVA tensors require CUDA")
def test_penalties_state_keeps_mixed_rapid_params_as_vectors():
    req_states = SimpleNamespace(
        max_num_reqs=4,
        vocab_size=8,
        device=torch.device("cuda"),
    )
    state = PenaltiesState(req_states)
    state.presence_penalty.np[:2] = [0.2, 0.3]
    state.repetition_penalty.np[:2] = 1.1
    state.penalty_decay.np[:2] = 0.95
    state.presence_penalty.copy_to_uva()
    state.repetition_penalty.copy_to_uva()
    state.penalty_decay.copy_to_uva()

    device = state.presence_penalty.gpu.device
    expanded_idx_mapping = torch.tensor([0, 1], dtype=torch.int32, device=device)
    idx_mapping_np = np.array([0, 1])

    presence, repetition, decay = state.rapid_penalty_params(
        expanded_idx_mapping,
        idx_mapping_np,
        scalar_if_uniform=True,
    )

    assert isinstance(presence, torch.Tensor)
    assert isinstance(repetition, torch.Tensor)
    assert isinstance(decay, torch.Tensor)
    assert presence.shape == (2,)


def test_rapid_sampler_recomputes_processed_logits_for_logprobs(monkeypatch):
    sampler = object.__new__(Sampler)
    sampler.use_rapid = True
    sampler.require_rapid = False
    sampler.use_flashinfer = False
    sampler.logprobs_mode = "processed_logprobs"
    sampler.rapid_penalties = None

    class FakeSamplingStates:
        def get_temperatures(
            self,
            expanded_idx_mapping,
            idx_mapping_np,
            *,
            scalar_if_uniform=False,
        ):
            assert scalar_if_uniform is True
            return 1.0

        def get_top_k_top_p(
            self,
            expanded_idx_mapping,
            idx_mapping_np,
            *,
            scalar_if_uniform=False,
        ):
            assert scalar_if_uniform is True
            return None, None

        def any_greedy(self, idx_mapping_np):
            return False

        def any_explicit_seed(self, idx_mapping_np):
            return False

        def any_min_p(self, idx_mapping_np):
            return False

    class FakePenaltiesState:
        def any_frequency_penalty(self, idx_mapping_np):
            return False

        def use_rapid_penalty(self, idx_mapping_np):
            return False

    sampler.sampling_states = FakeSamplingStates()
    sampler.penalties_state = FakePenaltiesState()

    sampling_param_calls = []
    first_processed_logits = torch.full((1, 4), 1.0)
    logprob_processed_logits = torch.full((1, 4), 2.0)

    def fake_apply_sampling_params(*args, **kwargs):
        sampling_param_calls.append(kwargs)
        return (
            first_processed_logits
            if len(sampling_param_calls) == 1
            else logprob_processed_logits
        )

    sampler.apply_sampling_params = fake_apply_sampling_params
    monkeypatch.setattr(
        sampler_module, "rapid_sample_input_supported", lambda logits: True
    )
    monkeypatch.setattr(
        sampler_module,
        "rapid_sample",
        lambda logits, top_k, top_p, temperatures: torch.tensor([3]),
    )

    sampled, processed_logits = sampler.sample(
        logits=torch.zeros((1, 4)),
        expanded_idx_mapping=torch.tensor([0]),
        idx_mapping_np=np.array([0]),
        pos=torch.tensor([0]),
        input_ids=torch.tensor([0]),
        expanded_local_pos=torch.tensor([0]),
        return_logprobs=True,
    )

    assert sampled.tolist() == [3]
    assert processed_logits is logprob_processed_logits
    assert len(sampling_param_calls) == 2
    assert sampling_param_calls[0]["skip_top_k_top_p"] is True
    assert sampling_param_calls[0]["skip_temperature"] is True
    assert sampling_param_calls[1].get("skip_top_k_top_p", False) is False
    assert sampling_param_calls[1].get("skip_temperature", False) is False


@pytest.mark.parametrize("require_rapid", [False, True])
def test_rapid_sampler_native_fallback_is_forbidden_when_required(
    monkeypatch,
    require_rapid,
):
    monkeypatch.delenv("VLLM_USE_RAPID_SAMPLER", raising=False)

    sampler = object.__new__(Sampler)
    sampler.use_rapid = True
    sampler.use_flashinfer = False
    sampler.logprobs_mode = "raw_logprobs"
    sampler.rapid_penalties = None
    sampler.use_fp64_gumbel = False
    sampler.require_rapid = require_rapid

    top_k_calls = []

    class FakeSamplingStates:
        temperature = SimpleNamespace(gpu=torch.tensor([1.0]))
        seeds = SimpleNamespace(gpu=torch.tensor([0]))

        def get_temperatures(
            self,
            expanded_idx_mapping,
            idx_mapping_np,
            *,
            scalar_if_uniform=False,
        ):
            assert scalar_if_uniform is True
            return 1.0

        def get_top_k_top_p(
            self,
            expanded_idx_mapping,
            idx_mapping_np,
            *,
            scalar_if_uniform=False,
        ):
            top_k_calls.append(scalar_if_uniform)
            return None, None

        def any_greedy(self, idx_mapping_np):
            return False

        def any_explicit_seed(self, idx_mapping_np):
            return False

        def any_min_p(self, idx_mapping_np):
            return False

    class FakePenaltiesState:
        def any_frequency_penalty(self, idx_mapping_np):
            return False

        def use_rapid_penalty(self, idx_mapping_np):
            return False

    sampler.sampling_states = FakeSamplingStates()
    sampler.penalties_state = FakePenaltiesState()

    sampling_param_calls = []
    rapid_processed_logits = torch.full((1, 4), 1.0)
    native_processed_logits = torch.full((1, 4), 2.0)

    def fake_apply_sampling_params(*args, **kwargs):
        sampling_param_calls.append(kwargs)
        return (
            rapid_processed_logits
            if len(sampling_param_calls) == 1
            else native_processed_logits
        )

    sampler.apply_sampling_params = fake_apply_sampling_params
    monkeypatch.setattr(
        sampler_module, "rapid_sample_input_supported", lambda logits: False
    )
    monkeypatch.setattr(
        sampler_module,
        "rapid_sample",
        lambda *args, **kwargs: pytest.fail("unsupported input should use native"),
    )
    monkeypatch.setattr(
        sampler_module,
        "gumbel_sample",
        lambda *args, **kwargs: torch.tensor([2]),
    )

    sample = lambda: sampler.sample(
        logits=torch.zeros((1, 4)),
        expanded_idx_mapping=torch.tensor([0]),
        idx_mapping_np=np.array([0]),
        pos=torch.tensor([0]),
        input_ids=torch.tensor([0]),
        expanded_local_pos=torch.tensor([0]),
    )

    if require_rapid:
        with pytest.raises(RuntimeError, match="rapid-sampling requires"):
            sample()
        return

    sampled, processed_logits = sample()

    assert sampled.tolist() == [2]
    assert processed_logits is native_processed_logits
    assert top_k_calls == [True, False]
    assert len(sampling_param_calls) == 2
    assert sampling_param_calls[0]["skip_top_k_top_p"] is True
    assert sampling_param_calls[0]["skip_temperature"] is True
    assert sampling_param_calls[1].get("skip_top_k_top_p", False) is False
    assert sampling_param_calls[1].get("skip_temperature", False) is False
