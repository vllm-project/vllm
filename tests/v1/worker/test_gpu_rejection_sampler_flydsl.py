# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""VLLM_ROCM_USE_AITER_FLYDSL_REJECTION_SAMPLER: RejectionSampler verifies the
steps the AITER FlyDSL rejection sampler supports with it, with the tokens of
the Triton path, and keeps every other step (and any build of AITER without the
op) on Triton."""

import sys
from collections.abc import Sequence
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.config import SpeculativeConfig
from vllm.platforms import current_platform
from vllm.sampling_params import SamplingParams

pytest.importorskip("triton")
if not current_platform.is_cuda_alike():
    pytest.skip("CUDA or ROCm required", allow_module_level=True)

from vllm.v1.worker.gpu.sample.logits_processor import (
    LogitsContext,
    LogitsProcessor,
)
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


class _ReasoningConfig:
    """Enables the thinking-budget stage."""

    reasoning_start_token_ids = [90]
    reasoning_end_token_ids = [91]
    natural_reasoning_end_token_ids = [91]


class _ExtraArgsProcessor(LogitsProcessor):
    """A custom logits processor that takes requests with `extra_args`."""

    def __init__(self):
        pass

    def add_request(self, req_idx: int, sampling_params: SamplingParams) -> bool:
        return bool(sampling_params.extra_args)

    def apply(self, logits: torch.Tensor, ctx: LogitsContext) -> torch.Tensor:
        return logits


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
    spec_config: SpeculativeConfig,
    requests: list[SamplingParams],
    custom_logits_processors: Sequence[LogitsProcessor] = (),
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
        vllm_config=SimpleNamespace(reasoning_config=_ReasoningConfig()),
        max_num_reqs=MAX_NUM_REQS,
        vocab_size=VOCAB_SIZE,
        device=DEVICE,
        req_states=req_states,
        custom_logits_processors=custom_logits_processors,
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


def _verify_inputs(
    num_reqs: int, seed: int = 0, grammar_masked: bool = False
) -> tuple[dict, list[list[int]]]:
    """A verify step whose decisions are far from every rounding margin, and
    the tokens the standard method must return for it.

    Each target row peaks at one token by 60 and each draft row by 10, so at
    `TEMPERATURES` a draft on the target peak is accepted (p > q), a draft
    elsewhere is rejected (log p - log q < -30 < log u), and the resampled
    token is the target peak whatever the Gumbel noise. For greedy and
    temperature-1 requests the first draft off the peak is the runner-up, 0.5
    below it: greedy rejects it, temperature 1 accepts it (p 0.38 > q 0.15).
    With `grammar_masked`, 99% of the other target entries are -inf, as a
    structured-output bitmask leaves them in the raw logits; a masked draft
    has p = 0 and is rejected either way.
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
    if grammar_masked:
        masked = (target == 0) & (torch.rand(target.shape, generator=gen) < 0.99)
        target[masked] = float("-inf")
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
    cu_num_logits_np = np.arange(0, num_logits + 1, rows, dtype=np.int32)
    inputs = dict(
        logits=target.to(DEVICE),
        draft_logits=draft_logits,
        draft_sampled=draft_sampled.view(-1).to(DEVICE, torch.int32),
        pos=torch.arange(100, 100 + num_logits, device=DEVICE, dtype=torch.int64),
        cu_num_logits=torch.from_numpy(cu_num_logits_np).to(DEVICE),
        idx_mapping=idx_mapping,
        idx_mapping_np=idx_mapping_np,
        expanded_idx_mapping=idx_mapping.repeat_interleave(rows),
        expanded_local_pos=torch.arange(rows, device=DEVICE, dtype=torch.int32).repeat(
            num_reqs
        ),
        seq_lens_upper_bound_np=np.full(num_reqs, 64, dtype=np.int32),
        cu_num_logits_np=cu_num_logits_np,
    )
    return inputs, expected


def _flydsl_rejection_sample(rejection_sampler: RejectionSampler, inputs: dict):
    return rejection_sampler._flydsl_rejection_sample(
        inputs["logits"],
        inputs["draft_logits"],
        inputs["draft_sampled"],
        inputs["pos"],
        inputs["idx_mapping"],
        inputs["idx_mapping_np"],
        inputs["cu_num_logits_np"],
    )


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


@pytest.mark.parametrize("aiter_op", ["missing", "incomplete"])
def test_flydsl_rejection_sampler_falls_back_without_op(
    monkeypatch: pytest.MonkeyPatch, aiter_op: str
):
    """With the flag on but an AITER build that lacks the op, or has it
    without the API vLLM uses, verification keeps the Triton path."""
    module = None if aiter_op == "missing" else ModuleType(FLYDSL_MODULE)
    monkeypatch.setitem(sys.modules, FLYDSL_MODULE, module)
    _set_flydsl_flag(monkeypatch, enabled=True)
    rejection_sampler = _make_rejection_sampler(_spec_config(), _requests())
    assert rejection_sampler.flydsl_rejection_sampler is None

    inputs, expected = _verify_inputs(8)
    _, sampled, num_sampled = rejection_sampler._verify(**inputs)
    _check_sampled(sampled, num_sampled, expected)


@pytest.mark.parametrize("method", ["standard", "synthetic"])
@pytest.mark.parametrize(
    ("num_reqs", "grammar_masked"),
    [(1, False), (8, False), (96, False), (96, True)],
    ids=["1", "8", "96", "96-grammar-masked"],
)
def test_flydsl_rejection_sampler_matches_triton(
    monkeypatch: pytest.MonkeyPatch, method: str, num_reqs: int, grammar_masked: bool
):
    _flydsl_rejection_sampler_or_skip()
    requests = _requests()
    _set_flydsl_flag(monkeypatch, enabled=False)
    triton_sampler = _make_rejection_sampler(_spec_config(method), requests)
    _set_flydsl_flag(monkeypatch, enabled=True)
    flydsl_sampler = _make_rejection_sampler(_spec_config(method), requests)
    assert triton_sampler.flydsl_rejection_sampler is None
    assert flydsl_sampler.flydsl_rejection_sampler is not None

    inputs, expected = _verify_inputs(num_reqs, grammar_masked=grammar_masked)
    assert _flydsl_rejection_sample(flydsl_sampler, inputs) is not None

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


def _input_batch(inputs: dict) -> SimpleNamespace:
    """The InputBatch fields RejectionSampler.__call__ reads, for a step whose
    tokens are exactly its logits rows."""
    num_reqs = len(inputs["idx_mapping_np"])
    num_logits = inputs["logits"].shape[0]
    return SimpleNamespace(
        num_reqs=num_reqs,
        input_ids=inputs["draft_sampled"],
        positions=inputs["pos"],
        logits_indices=torch.arange(num_logits, device=DEVICE, dtype=torch.int64),
        idx_mapping=inputs["idx_mapping"],
        idx_mapping_np=inputs["idx_mapping_np"],
        expanded_idx_mapping=inputs["expanded_idx_mapping"],
        expanded_local_pos=inputs["expanded_local_pos"],
        cu_num_logits=inputs["cu_num_logits"],
        cu_num_logits_np=inputs["cu_num_logits_np"],
        seq_lens=torch.full((num_reqs,), 106, device=DEVICE, dtype=torch.int32),
        seq_lens_cpu_upper_bound=torch.full((num_reqs,), 106, dtype=torch.int32),
    )


@pytest.mark.parametrize("max_chunk_logits", [None, 8 * (NUM_STEPS + 1)])
def test_flydsl_rejection_sampler_call_matches_triton(
    monkeypatch: pytest.MonkeyPatch, max_chunk_logits: int | None
):
    """Through RejectionSampler.__call__ (one chunk), or _verify_in_chunks
    with 8-request chunks, every chunk takes the FlyDSL path and the output
    matches Triton's."""
    _flydsl_rejection_sampler_or_skip()
    requests = _requests()
    _set_flydsl_flag(monkeypatch, enabled=False)
    triton_sampler = _make_rejection_sampler(_spec_config(), requests)
    _set_flydsl_flag(monkeypatch, enabled=True)
    flydsl_sampler = _make_rejection_sampler(_spec_config(), requests)
    taken = []
    flydsl_verify = flydsl_sampler._flydsl_rejection_sample

    def spy(*args):
        out = flydsl_verify(*args)
        taken.append(out is not None)
        return out

    monkeypatch.setattr(flydsl_sampler, "_flydsl_rejection_sample", spy)

    inputs, expected = _verify_inputs(40, seed=1)
    batch = _input_batch(inputs)
    if max_chunk_logits is None:
        ref = triton_sampler(inputs["logits"], batch, inputs["draft_logits"])
        out = flydsl_sampler(inputs["logits"], batch, inputs["draft_logits"])
        assert taken == [True]
        assert torch.equal(out.num_sampled, ref.num_sampled)
        assert torch.equal(out.num_rejected, ref.num_rejected)
        sampled, num_sampled = out.sampled_token_ids, out.num_sampled
    else:
        args = (
            inputs["logits"],
            batch,
            inputs["draft_logits"],
            inputs["draft_sampled"],
            inputs["pos"],
            max_chunk_logits,
            None,
        )
        ref_sampled, ref_num_sampled, _, _ = triton_sampler._verify_in_chunks(*args)
        sampled, num_sampled, _, _ = flydsl_sampler._verify_in_chunks(*args)
        assert taken == [True] * 5
        assert torch.equal(num_sampled, ref_num_sampled)
        for b, n in enumerate(ref_num_sampled.tolist()):
            assert torch.equal(sampled[b, :n], ref_sampled[b, :n])
    _check_sampled(sampled, num_sampled, expected)


# Every kind of logits processing besides temperature, and every step that
# is not a full verify step on a replicated bf16 draft cache.
_DECLINED_REQUESTS = {
    "top-k": SamplingParams(temperature=0.8, top_k=20),
    "top-p": SamplingParams(temperature=0.8, top_p=0.9),
    "min-p": SamplingParams(temperature=0.8, min_p=0.1),
    "presence-penalty": SamplingParams(temperature=0.8, presence_penalty=0.5),
    "repetition-penalty": SamplingParams(repetition_penalty=1.2),
    "logit-bias": SamplingParams(logit_bias={7: 2.0}),
    "allowed-token-ids": SamplingParams(allowed_token_ids=[1, 2, 3]),
    "min-tokens": SamplingParams(min_tokens=4, stop_token_ids=[5]),
    "bad-words": SamplingParams(_bad_words_token_ids=[[7]]),
    "thinking-budget": SamplingParams(thinking_token_budget=3),
    "logits-processor": SamplingParams(extra_args={"custom": 1}),
}


@pytest.mark.parametrize(
    "case",
    [
        *_DECLINED_REQUESTS,
        "fp32-logits",
        "short-request",
        "uneven-rows",
        "sharded",
        "greedy-draft",
    ],
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
    if case in _DECLINED_REQUESTS:
        requests[req_idx] = _DECLINED_REQUESTS[case]
    elif case == "fp32-logits":
        inputs["logits"] = inputs["logits"].float()
    elif case == "short-request":
        inputs["logits"] = inputs["logits"][:-1]
    elif case == "uneven-rows":
        # Same total, but request 1 has 7 rows and request 2 has 5.
        inputs["cu_num_logits_np"][2] += 1
    elif case == "sharded":
        inputs["draft_logits"].row_mapping = torch.arange(
            MAX_NUM_REQS, device=DEVICE, dtype=torch.int32
        )
    elif case == "greedy-draft":
        # draft_sample_method=greedy keeps no draft-logits cache.
        inputs["draft_logits"] = None
    _set_flydsl_flag(monkeypatch, enabled=True)
    rejection_sampler = _make_rejection_sampler(
        _spec_config(), requests, [_ExtraArgsProcessor()]
    )
    assert rejection_sampler.flydsl_rejection_sampler is not None
    assert _flydsl_rejection_sample(rejection_sampler, inputs) is None

    # The same step without the declined request is taken.
    if case in _DECLINED_REQUESTS:
        rejection_sampler.sampler.add_request(req_idx, SamplingParams(seed=0))
        rejection_sampler.sampler.apply_staged_writes()
        assert _flydsl_rejection_sample(rejection_sampler, inputs) is not None
