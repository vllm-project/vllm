# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Sequence
from types import SimpleNamespace

import numpy as np
import pytest
import torch

pytest.importorskip("triton")
if not torch.cuda.is_available():
    pytest.skip("CUDA required for sampler flag tests", allow_module_level=True)

from vllm.sampling_params import SamplingParams
from vllm.v1.worker.gpu.input_batch import InputBuffers
from vllm.v1.worker.gpu.model_runner import GPUModelRunner
from vllm.v1.worker.gpu.sample import sampler as sampler_module
from vllm.v1.worker.gpu.sample.logits_processor import (
    LogitsContext,
    LogitsProcessor,
)
from vllm.v1.worker.gpu.sample.sampler import Sampler
from vllm.v1.worker.gpu.states import RequestState

DEVICE = torch.device("cuda")
VOCAB_SIZE = 128


class MockReasoningConfig:
    reasoning_start_token_ids = [90]
    reasoning_end_token_ids = [91]
    natural_reasoning_end_token_ids = [91]


def _make_sampler(custom_logits_processors: Sequence[LogitsProcessor] = ()) -> Sampler:
    req_states = RequestState(
        max_num_reqs=4,
        max_model_len=64,
        max_num_batched_tokens=16,
        num_speculative_steps=1,
        vocab_size=VOCAB_SIZE,
        device=DEVICE,
    )
    return Sampler(
        vllm_config=SimpleNamespace(reasoning_config=MockReasoningConfig()),
        max_num_reqs=4,
        vocab_size=VOCAB_SIZE,
        device=DEVICE,
        req_states=req_states,
        custom_logits_processors=custom_logits_processors,
    )


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
    sampler.add_request(3, sampling_params=sampling_params)

    assert sampler.needs_logits_processing[3] == expected


def test_logits_processing_cache_is_overwritten_when_slot_is_reused():
    sampler = _make_sampler()
    sampler.add_request(3, SamplingParams.for_sampler_warmup())
    sampler.add_request(3, SamplingParams())

    assert not sampler.needs_logits_processing[3]


class _GateProcessor(LogitsProcessor):
    """Reports a fixed admission decision from add_request()."""

    def __init__(self, admitted: bool):
        self.admitted = admitted

    def add_request(self, req_idx: int, sampling_params: SamplingParams) -> bool:
        return self.admitted

    def apply(self, logits: torch.Tensor, ctx: LogitsContext) -> torch.Tensor:
        raise AssertionError("the pipeline never runs in this test")


@pytest.mark.parametrize("admitted", [True, False])
def test_custom_processor_add_request_gates_pipeline_flag(admitted: bool):
    """A custom processor's add_request() return value is OR-ed into the
    per-request needs_logits_processing flag."""
    sampler = _make_sampler(custom_logits_processors=[_GateProcessor(admitted)])
    sampler.add_request(3, SamplingParams())

    assert sampler.needs_logits_processing[3] == admitted


def test_has_topp_only_rows():
    """A row is top-p-only when it has top-p and no effective top-k; top_k
    of 0 (the default) or at least the logits width disables top-k."""
    sampler = _make_sampler()
    sampler.add_request(0, SamplingParams(top_k=10, top_p=0.9))
    sampler.add_request(1, SamplingParams(top_k=10))
    sampler.add_request(2, SamplingParams(top_p=0.9))
    sampler.add_request(3, SamplingParams(top_k=VOCAB_SIZE + 1, top_p=0.9))
    states = sampler.sampling_states
    states.apply_staged_writes()

    def has_topp_only_rows(idx: list[int], vocab_size: int = VOCAB_SIZE) -> bool:
        idx_np = np.array(idx, dtype=np.int32)
        idx_gpu = torch.from_numpy(idx_np).to(DEVICE)
        return states.get_top_k_top_p(idx_gpu, idx_np, vocab_size)[2]

    assert not has_topp_only_rows([0, 1])
    assert has_topp_only_rows([0, 2])
    assert has_topp_only_rows([3])
    # The kernel compares top_k against the logits width, so a top_k at or
    # above a narrower logits width counts as no top-k.
    assert has_topp_only_rows([0], vocab_size=10)


def _flashinfer_top_k(monkeypatch, params: list[SamplingParams]):
    """Run Sampler.sample on one row per request and return the top_k argument
    it hands to FlashInfer."""
    sampler = _make_sampler()
    # flashinfer_sample is replaced below, so this also runs on GPUs where the
    # FlashInfer sampler is disabled (e.g. CI's 16-SM MIG slices).
    sampler.use_flashinfer = True
    received = []

    def fake_flashinfer_sample(logits, k, p):
        received.append(k)
        return torch.zeros(logits.shape[0], dtype=torch.int32, device=logits.device)

    monkeypatch.setattr(sampler_module, "flashinfer_sample", fake_flashinfer_sample)
    for req_idx, sampling_params in enumerate(params):
        sampler.add_request(req_idx, sampling_params)
    sampler.apply_staged_writes()

    num_reqs = len(params)
    idx_mapping = torch.arange(num_reqs, dtype=torch.int32, device=DEVICE)
    sampler.sample(
        torch.randn(num_reqs, VOCAB_SIZE, device=DEVICE),
        expanded_idx_mapping=idx_mapping,
        idx_mapping=idx_mapping,
        idx_mapping_np=np.arange(num_reqs, dtype=np.int32),
        pos=torch.zeros(num_reqs, dtype=torch.int64, device=DEVICE),
        input_ids=torch.zeros(num_reqs, dtype=torch.int32, device=DEVICE),
        expanded_local_pos=torch.zeros(num_reqs, dtype=torch.int32, device=DEVICE),
        seq_lens_upper_bound_np=np.ones(num_reqs, dtype=np.int64),
    )
    assert len(received) == 1
    return received[0]


@pytest.mark.parametrize(
    ("top_ks", "expected"),
    [
        pytest.param([20, 20, 20], 20, id="shared"),
        pytest.param([20, 30], [20, 30], id="different"),
        pytest.param([20, 0], [20, VOCAB_SIZE], id="one-without-top-k"),
        pytest.param([0, 0], None, id="none-with-top-k"),
    ],
)
def test_flashinfer_gets_int_top_k_only_when_requests_share_it(
    monkeypatch, top_ks: list[int], expected: int | list[int] | None
):
    """FlashInfer only takes its faster top-k-first path for an int top_k."""
    params = [SamplingParams(temperature=0.7, top_k=k, top_p=0.9) for k in top_ks]
    top_k = _flashinfer_top_k(monkeypatch, params)

    if isinstance(expected, list):
        assert isinstance(top_k, torch.Tensor)
        assert top_k.tolist() == expected
    else:
        assert type(top_k) is type(expected)
        assert top_k == expected


def test_dummy_sampler_run_profiles_requests_with_different_top_k():
    """Requests that share a top_k take FlashInfer's leaner path, so the
    profiling pass must mix top_k values to measure the per-row path."""
    num_reqs = 4
    sampler = _make_sampler()
    runner = SimpleNamespace(
        model=SimpleNamespace(
            compute_logits=lambda hidden: torch.randn(
                num_reqs, VOCAB_SIZE, device=DEVICE
            )
        ),
        input_buffers=InputBuffers(
            max_num_reqs=num_reqs, max_num_tokens=16, device=DEVICE
        ),
        sampler=sampler,
    )

    GPUModelRunner._dummy_sampler_run(runner, torch.zeros(num_reqs, 8, device=DEVICE))

    assert len(set(sampler.sampling_states.top_k.np[:num_reqs].tolist())) > 1


def test_logits_processing_cache_only_checks_active_requests():
    sampler = _make_sampler()
    sampler.add_request(0, SamplingParams(temperature=0.0))
    sampler.add_request(2, SamplingParams.for_sampler_warmup())

    sampling_only = np.array([0], dtype=np.int32)
    with_processing = np.array([0, 2], dtype=np.int32)

    assert not np.any(sampler.needs_logits_processing[sampling_only])
    assert np.any(sampler.needs_logits_processing[with_processing])
