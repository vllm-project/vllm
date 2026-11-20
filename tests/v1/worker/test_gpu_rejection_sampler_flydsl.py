# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""VLLM_ROCM_USE_AITER_FLYDSL_REJECTION_SAMPLER: RejectionSampler verifies the
steps the AITER FlyDSL rejection sampler supports with it, with the tokens of
the Triton path, and keeps every other step (and any build of AITER without the
op) on Triton."""

import sys

import numpy as np
import pytest
import torch

from vllm.config import SpeculativeConfig, VllmConfig
from vllm.platforms import current_platform
from vllm.sampling_params import SamplingParams

pytest.importorskip("triton")
if not current_platform.is_cuda_alike():
    pytest.skip("CUDA or ROCm required", allow_module_level=True)

from vllm.v1.worker.gpu.sample.sampler import Sampler
from vllm.v1.worker.gpu.spec_decode.draft_logits import DraftLogitsCache
from vllm.v1.worker.gpu.spec_decode.rejection_sampler import RejectionSampler
from vllm.v1.worker.gpu.states import RequestState

DEVICE = torch.device("cuda")
NUM_STEPS = 5
VOCAB_SIZE = 129280
MAX_NUM_REQS = 128
FLYDSL_MODULE = "aiter.ops.flydsl.rejection_sample"
# Of request state i.
TEMPERATURES = (0.0, 1.0, 0.6, 1.5)


def _set_flydsl_flag(monkeypatch: pytest.MonkeyPatch, enabled: bool) -> None:
    monkeypatch.setenv("VLLM_ROCM_USE_AITER", "1" if enabled else "0")
    monkeypatch.setenv(
        "VLLM_ROCM_USE_AITER_FLYDSL_REJECTION_SAMPLER", "1" if enabled else "0"
    )
    if current_platform.is_rocm():
        from vllm._aiter_ops import rocm_aiter_ops

        rocm_aiter_ops.refresh_env_variables()
        rocm_aiter_ops._flydsl_rejection_sampler_module.cache_clear()


@pytest.fixture(autouse=True)
def _restore_aiter_env(monkeypatch: pytest.MonkeyPatch):
    yield
    monkeypatch.undo()
    if current_platform.is_rocm():
        from vllm._aiter_ops import rocm_aiter_ops

        rocm_aiter_ops.refresh_env_variables()
        rocm_aiter_ops._flydsl_rejection_sampler_module.cache_clear()


def _make_rejection_sampler(
    spec_config: SpeculativeConfig, requests: list[SamplingParams]
) -> RejectionSampler:
    req_states = RequestState(
        max_num_reqs=MAX_NUM_REQS,
        max_model_len=64,
        max_num_batched_tokens=MAX_NUM_REQS * (NUM_STEPS + 1),
        num_speculative_steps=NUM_STEPS,
        vocab_size=VOCAB_SIZE,
        device=DEVICE,
    )
    sampler = Sampler(
        vllm_config=VllmConfig(),
        max_num_reqs=MAX_NUM_REQS,
        vocab_size=VOCAB_SIZE,
        device=DEVICE,
        req_states=req_states,
    )
    for req_idx, sampling_params in enumerate(requests):
        sampler.add_request(req_idx, sampling_params)
    sampler.apply_staged_writes()
    return RejectionSampler(sampler, spec_config, DEVICE)


def _spec_config(method: str = "standard", **kwargs) -> SpeculativeConfig:
    if method == "synthetic":
        kwargs["synthetic_acceptance_rates"] = [0.9, 0.7, 0.5, 0.3, 0.1]
    return SpeculativeConfig(
        method="suffix",
        num_speculative_tokens=kwargs.pop("num_speculative_tokens", NUM_STEPS),
        rejection_sample_method=method,
        **kwargs,
    )


def _verify_inputs(num_reqs: int, seed: int = 0) -> tuple[dict, list[list[int]]]:
    """A verify step whose decisions are far from every rounding margin, and
    the tokens the standard method must return for it.

    Each target row peaks at one token by 60 and each draft row by 10, so at
    `TEMPERATURES` a draft on the target peak is accepted (p > q), a draft
    elsewhere is rejected (log p - log q < -30 < log u), and the resampled
    token is the target peak whatever the Gumbel noise. For greedy and
    temperature-1 requests the first draft off the peak is the runner-up, 0.5
    below it: greedy rejects it, temperature 1 accepts it (p 0.38 > q 0.15).
    """
    gen = torch.Generator().manual_seed(seed)
    rows = NUM_STEPS + 1
    num_logits = num_reqs * rows
    idx_mapping_np = torch.randperm(MAX_NUM_REQS, generator=gen)[:num_reqs].numpy()
    peaks = torch.randint(0, VOCAB_SIZE - 1, (num_logits,), generator=gen)
    target = torch.zeros(num_logits, VOCAB_SIZE, dtype=torch.bfloat16)
    target[torch.arange(num_logits), peaks] = 60.0

    num_accepted = torch.randint(0, rows, (num_reqs,), generator=gen).tolist()
    # Entry j + 1 is the draft of step j, verified by row j: its peak, or the
    # next token once the request stops accepting. Entry 0 holds no draft.
    draft_sampled = torch.zeros(num_reqs, rows, dtype=torch.int64)
    for b, req_idx in enumerate(idx_mapping_np):
        k = num_accepted[b]
        for j in range(NUM_STEPS):
            peak = int(peaks[b * rows + j])
            draft_sampled[b, j + 1] = peak if j < k else peak + 1
        if b > 0 and k < NUM_STEPS and TEMPERATURES[req_idx % 4] in (0.0, 1.0):
            target[b * rows + k, draft_sampled[b, k + 1]] = 59.5
            num_accepted[b] += TEMPERATURES[req_idx % 4] == 1.0
    # A placeholder draft is always rejected.
    draft_sampled[0, NUM_STEPS] = -1
    num_accepted[0] = min(num_accepted[0], NUM_STEPS - 1)
    expected = [
        draft_sampled[b, 1 : k + 1].tolist() + [int(peaks[b * rows + k])]
        for b, k in enumerate(num_accepted)
    ]

    draft_values = torch.zeros(
        MAX_NUM_REQS, NUM_STEPS, VOCAB_SIZE, dtype=torch.bfloat16
    )
    for b, req_idx in enumerate(idx_mapping_np):
        for j in range(NUM_STEPS):
            token = int(draft_sampled[b, j + 1])
            if token >= 0:
                draft_values[req_idx, j, token] = 10.0
    draft_logits = DraftLogitsCache(
        MAX_NUM_REQS,
        NUM_STEPS,
        VOCAB_SIZE,
        dtype=torch.bfloat16,
        fill=0.0,
        device=DEVICE,
    )
    draft_logits.values.copy_(draft_values)

    idx_mapping = torch.from_numpy(idx_mapping_np).to(DEVICE, torch.int64)
    inputs = dict(
        logits=target.to(DEVICE),
        draft_logits=draft_logits,
        draft_sampled=draft_sampled.view(-1).to(DEVICE, torch.int32),
        pos=torch.arange(100, 100 + num_logits, device=DEVICE, dtype=torch.int64),
        cu_num_logits=torch.arange(
            0, num_logits + 1, rows, device=DEVICE, dtype=torch.int32
        ),
        idx_mapping=idx_mapping,
        idx_mapping_np=idx_mapping_np,
        expanded_idx_mapping=idx_mapping.repeat_interleave(rows),
        expanded_local_pos=torch.arange(rows, device=DEVICE, dtype=torch.int32).repeat(
            num_reqs
        ),
        seq_lens_upper_bound_np=np.full(num_reqs, 64, dtype=np.int32),
    )
    return inputs, expected


def _check_sampled(
    sampled: torch.Tensor, num_sampled: torch.Tensor, expected: list[list[int]]
) -> None:
    assert num_sampled.tolist() == [len(tokens) for tokens in expected]
    for b, tokens in enumerate(expected):
        assert sampled[b, : len(tokens)].tolist() == tokens


def _flydsl_rejection_sampler_or_skip():
    if not current_platform.is_rocm():
        pytest.skip("AITER FlyDSL rejection sampler is ROCm-only")
    from vllm.platforms.rocm import on_gfx950

    if not on_gfx950():
        pytest.skip("AITER FlyDSL rejection sampler is gfx950-only")
    pytest.importorskip(FLYDSL_MODULE)


def _requests() -> list[SamplingParams]:
    return [
        SamplingParams(temperature=TEMPERATURES[i % 4], seed=i)
        for i in range(MAX_NUM_REQS)
    ]


def test_flydsl_rejection_sampler_falls_back_without_op(
    monkeypatch: pytest.MonkeyPatch,
):
    """With the flag on but an AITER build that lacks the op, verification
    keeps the Triton path."""
    monkeypatch.setitem(sys.modules, FLYDSL_MODULE, None)
    _set_flydsl_flag(monkeypatch, enabled=True)
    rejection_sampler = _make_rejection_sampler(_spec_config(), _requests())
    assert rejection_sampler.flydsl_rejection_sampler is None

    inputs, expected = _verify_inputs(8)
    _, sampled, num_sampled = rejection_sampler._verify(**inputs)
    _check_sampled(sampled, num_sampled, expected)


@pytest.mark.parametrize("method", ["standard", "synthetic"])
@pytest.mark.parametrize("num_reqs", [1, 8, 96])
def test_flydsl_rejection_sampler_matches_triton(
    monkeypatch: pytest.MonkeyPatch, method: str, num_reqs: int
):
    _flydsl_rejection_sampler_or_skip()
    requests = _requests()
    _set_flydsl_flag(monkeypatch, enabled=False)
    triton_sampler = _make_rejection_sampler(_spec_config(method), requests)
    _set_flydsl_flag(monkeypatch, enabled=True)
    flydsl_sampler = _make_rejection_sampler(_spec_config(method), requests)
    assert triton_sampler.flydsl_rejection_sampler is None
    assert flydsl_sampler.flydsl_rejection_sampler is not None

    inputs, expected = _verify_inputs(num_reqs)
    taken = flydsl_sampler._flydsl_rejection_sample(
        inputs["logits"],
        inputs["draft_logits"],
        inputs["draft_sampled"],
        inputs["pos"],
        inputs["idx_mapping"],
        inputs["idx_mapping_np"],
    )
    assert taken is not None

    _, ref_sampled, ref_num_sampled = triton_sampler._verify(**inputs)
    _, sampled, num_sampled = flydsl_sampler._verify(**inputs)
    if method == "standard":
        _check_sampled(sampled, num_sampled, expected)
    assert torch.equal(num_sampled, ref_num_sampled)
    for b, n in enumerate(ref_num_sampled.tolist()):
        assert torch.equal(sampled[b, :n], ref_sampled[b, :n])
        assert (sampled[b, n:] == -1).all()


@pytest.mark.parametrize(
    "spec_kwargs",
    [
        dict(method="block"),
        dict(method="fly", fly_window_size=2),
        dict(method="standard", num_speculative_tokens=3),
    ],
    ids=["block", "fly", "3-steps"],
)
def test_flydsl_rejection_sampler_declines_config(
    monkeypatch: pytest.MonkeyPatch, spec_kwargs: dict
):
    _flydsl_rejection_sampler_or_skip()
    _set_flydsl_flag(monkeypatch, enabled=True)
    rejection_sampler = _make_rejection_sampler(
        _spec_config(**spec_kwargs), _requests()
    )
    assert rejection_sampler.flydsl_rejection_sampler is None


@pytest.mark.parametrize(
    "case", ["top-k", "penalty", "fp32-logits", "short-request", "sharded"]
)
def test_flydsl_rejection_sampler_declines_step(
    monkeypatch: pytest.MonkeyPatch, case: str
):
    """Steps whose logits need more than temperature, or that are not a full
    verify step on a replicated cache, stay on Triton."""
    _flydsl_rejection_sampler_or_skip()
    requests = _requests()
    inputs, _ = _verify_inputs(8)
    req_idx = int(inputs["idx_mapping_np"][3])
    if case == "top-k":
        requests[req_idx] = SamplingParams(temperature=0.8, top_k=20)
    elif case == "penalty":
        requests[req_idx] = SamplingParams(temperature=0.8, presence_penalty=0.5)
    elif case == "fp32-logits":
        inputs["logits"] = inputs["logits"].float()
    elif case == "short-request":
        inputs["logits"] = inputs["logits"][:-1]
    elif case == "sharded":
        inputs["draft_logits"].row_mapping = torch.arange(
            MAX_NUM_REQS, device=DEVICE, dtype=torch.int32
        )
    _set_flydsl_flag(monkeypatch, enabled=True)
    rejection_sampler = _make_rejection_sampler(_spec_config(), requests)
    assert rejection_sampler.flydsl_rejection_sampler is not None

    taken = rejection_sampler._flydsl_rejection_sample(
        inputs["logits"],
        inputs["draft_logits"],
        inputs["draft_sampled"],
        inputs["pos"],
        inputs["idx_mapping"],
        inputs["idx_mapping_np"],
    )
    assert taken is None
