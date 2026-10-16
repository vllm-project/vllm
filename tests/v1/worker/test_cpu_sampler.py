# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import numpy as np
import pytest
import torch
import vllm._C  # noqa: F401

from vllm.platforms import current_platform
from vllm.sampling_params import SamplingParams
from vllm.v1.watermarking.gumbel import (
    DualKeyGumbelWatermarker,
    GumbelWatermarker,
)
from vllm.v1.worker.cpu.logits_processors import (
    CPUBadWords,
    CPULogitBiasState,
    CPUPenalties,
)
from vllm.v1.worker.cpu.sampler import (
    CPUSampler,
    CPUSamplingMaskTensors,
    CPUThinkingBudgetState,
    CPUTraceReplayState,
    CPUWatermarkSampler,
)
from vllm.v1.worker.gpu.sample.logits_processor import LogitsContext, LogitsProcessor
from vllm.v1.worker.gpu.states import RequestState

pytestmark = pytest.mark.skipif(not current_platform.is_cpu(), reason="CPU only")

if current_platform.is_cpu():
    import vllm.v1.worker.cpu.shm  # noqa: F401

START = 90
END = 91
END_A = 92
END_B = 93
VOCAB_SIZE = 128


class ReasoningConfig:
    reasoning_start_token_ids = [START]
    reasoning_end_token_ids = [END]
    natural_reasoning_end_token_ids = [END]


class MultiTokenEndReasoningConfig:
    reasoning_start_token_ids = [START]
    reasoning_end_token_ids = [END_A, END_B]
    natural_reasoning_end_token_ids = [END_A, END_B]


class DistinctEndReasoningConfig:
    reasoning_start_token_ids = [START]
    reasoning_end_token_ids = [END_A, END_B]
    natural_reasoning_end_token_ids = [END]


def _make_req_states(tokens: list[int], prompt_len: int = 1) -> RequestState:
    req_states = RequestState(
        max_num_reqs=4,
        max_model_len=max(64, len(tokens) + 2),
        max_num_batched_tokens=32,
        num_speculative_steps=4,
        vocab_size=VOCAB_SIZE,
        device=torch.device("cpu"),
    )
    req_states.add_request(
        req_id="req",
        prompt_len=prompt_len,
        all_token_ids=tokens,
        num_computed_tokens=len(tokens),
        max_tokens=32,
    )
    req_states.apply_staged_writes()
    return req_states


def _thinking_context(req_idx: int, input_ids: list[int]) -> LogitsContext:
    num_rows = len(input_ids)
    return LogitsContext(
        expanded_idx_mapping=torch.full((num_rows,), req_idx, dtype=torch.int64),
        idx_mapping=torch.tensor([req_idx], dtype=torch.int64),
        idx_mapping_np=np.array([req_idx], dtype=np.intp),
        expanded_local_pos=torch.arange(num_rows, dtype=torch.int32),
        input_ids=torch.tensor(input_ids, dtype=torch.int32),
        pos=torch.zeros(num_rows, dtype=torch.int32),
        seq_lens_upper_bound_np=np.full(1, num_rows, dtype=np.int64),
    )


def test_cpu_sampler_selects_cpu_state_implementations() -> None:
    req_states = _make_req_states([1])
    sampler = CPUSampler(
        vllm_config=SimpleNamespace(reasoning_config=None),
        max_num_reqs=4,
        vocab_size=VOCAB_SIZE,
        device=torch.device("cpu"),
        req_states=req_states,
    )

    assert isinstance(sampler.sampling_states, type(sampler).sampling_states_cls)
    assert sampler.sampling_mask_tensors_cls is CPUSamplingMaskTensors
    assert isinstance(sampler.logits_processors[0], CPULogitBiasState)
    assert isinstance(sampler.penalties_state, CPUPenalties)
    assert isinstance(sampler.logits_processors[2], CPUBadWords)
    assert isinstance(sampler.thinking_budget_state, CPUThinkingBudgetState)


def test_cpu_sampler_suppresses_incomplete_prefill_samples() -> None:
    sampler = object.__new__(CPUSampler)
    sampler.req_states = SimpleNamespace(
        prefill_len=SimpleNamespace(gpu=torch.tensor([10, 7, 4, 20], dtype=torch.int32))
    )
    input_batch = SimpleNamespace(
        num_reqs=4,
        seq_lens=torch.tensor([4, 6, 22, 7], dtype=torch.int32),
        cu_num_logits=torch.tensor([0, 1, 2, 3, 4], dtype=torch.int32),
        idx_mapping=torch.tensor([2, 0, 3, 1], dtype=torch.int64),
    )

    num_sampled, num_rejected = sampler._get_num_sampled_and_rejected(input_batch)

    assert num_sampled.tolist() == [1, 0, 1, 1]
    assert num_rejected.tolist() == [0, 0, 0, 0]


def test_cpu_thinking_budget_forces_and_continues_end_marker() -> None:
    tokens = [1, START, 10, 11, 12]
    req_states = _make_req_states(tokens)
    req_idx = req_states.req_id_to_index["req"]
    state = CPUThinkingBudgetState(req_states, MultiTokenEndReasoningConfig())
    state.add_request(req_idx, SamplingParams(thinking_token_budget=3))

    logits = torch.zeros((2, VOCAB_SIZE))
    logits[:, [END_A, END_B]] = -torch.inf
    state.apply(logits, _thinking_context(req_idx, [12, END_A]))

    assert logits[0, END_A] == pytest.approx(1.0e9)
    assert logits[1, END_B] == pytest.approx(1.0e9)


def test_cpu_thinking_budget_uses_distinct_forced_end_marker() -> None:
    req_states = _make_req_states([1, START, 10, 11, 12])
    req_idx = req_states.req_id_to_index["req"]
    state = CPUThinkingBudgetState(req_states, DistinctEndReasoningConfig())
    state.add_request(req_idx, SamplingParams(thinking_token_budget=3))

    logits = torch.zeros((2, VOCAB_SIZE))
    state.apply(logits, _thinking_context(req_idx, [12, END_A]))

    assert logits[0, END_A] == pytest.approx(1.0e9)
    assert logits[1, END_B] == pytest.approx(1.0e9)


def test_cpu_thinking_budget_stops_after_natural_end() -> None:
    tokens = [1, START, 10, END, 20, 21, 22]
    req_states = _make_req_states(tokens)
    req_idx = req_states.req_id_to_index["req"]
    state = CPUThinkingBudgetState(req_states, ReasoningConfig())
    state.add_request(req_idx, SamplingParams(thinking_token_budget=2))

    logits = torch.zeros((1, VOCAB_SIZE))
    state.apply(logits, _thinking_context(req_idx, [22]))
    assert torch.count_nonzero(logits) == 0


def test_cpu_thinking_budget_does_not_force_before_budget() -> None:
    req_states = _make_req_states([1, START, 10, 11])
    req_idx = req_states.req_id_to_index["req"]
    state = CPUThinkingBudgetState(req_states, ReasoningConfig())
    state.add_request(req_idx, SamplingParams(thinking_token_budget=3))

    logits = torch.zeros((1, VOCAB_SIZE))
    state.apply(logits, _thinking_context(req_idx, [11]))
    assert torch.count_nonzero(logits) == 0


def test_cpu_thinking_budget_uses_latest_start_boundary() -> None:
    tokens = [1, START, 10, 11, 12, END, 13, START, 14, 15, 16]
    req_states = _make_req_states(tokens)
    req_idx = req_states.req_id_to_index["req"]
    state = CPUThinkingBudgetState(req_states, ReasoningConfig())
    state.add_request(req_idx, SamplingParams(thinking_token_budget=3))

    logits = torch.zeros((1, VOCAB_SIZE))
    state.apply(logits, _thinking_context(req_idx, [16]))
    assert logits[0, END] == pytest.approx(1.0e9)


def test_cpu_thinking_budget_continues_end_prefix_from_resumed_prompt() -> None:
    tokens = [1, START, 10, 11, END_A]
    req_states = _make_req_states(tokens, prompt_len=len(tokens))
    req_idx = req_states.req_id_to_index["req"]
    state = CPUThinkingBudgetState(req_states, MultiTokenEndReasoningConfig())
    state.add_request(req_idx, SamplingParams(thinking_token_budget=3))

    logits = torch.zeros((1, VOCAB_SIZE))
    state.apply(logits, _thinking_context(req_idx, [END_A]))
    assert logits[0, END_A] == 0
    assert logits[0, END_B] == pytest.approx(1.0e9)


def test_cpu_thinking_budget_clamps_oversized_budget() -> None:
    req_states = _make_req_states([1, START, 10, 11, 12])
    req_idx = req_states.req_id_to_index["req"]
    state = CPUThinkingBudgetState(req_states, ReasoningConfig())
    state.add_request(req_idx, SamplingParams(thinking_token_budget=2**40))

    logits = torch.zeros((1, VOCAB_SIZE))
    state.apply(logits, _thinking_context(req_idx, [12]))
    assert torch.count_nonzero(logits) == 0


def test_cpu_thinking_budget_scans_new_committed_tokens_incrementally() -> None:
    tokens = [1, START, *([10] * 4094)]
    req_states = _make_req_states(tokens)
    req_idx = req_states.req_id_to_index["req"]
    state = CPUThinkingBudgetState(req_states, ReasoningConfig())
    state.add_request(req_idx, SamplingParams(thinking_token_budget=8192))

    state.apply(torch.zeros((1, VOCAB_SIZE)), _thinking_context(req_idx, [10]))
    assert state.cached_scan_pos[req_idx] == len(tokens)

    req_states.all_token_ids.stage_write(req_idx, len(tokens), [10])
    req_states.total_len.stage_write_elem(req_idx, len(tokens) + 1)
    req_states.apply_staged_writes()
    state.apply(torch.zeros((1, VOCAB_SIZE)), _thinking_context(req_idx, [10]))
    assert state.cached_scan_pos[req_idx] == len(tokens) + 1


def test_cpu_thinking_budget_clears_reused_slot() -> None:
    req_states = _make_req_states([1, START, 10, 11, 12])
    req_idx = req_states.req_id_to_index["req"]
    state = CPUThinkingBudgetState(req_states, ReasoningConfig())
    state.add_request(req_idx, SamplingParams(thinking_token_budget=3))
    state.add_request(req_idx, SamplingParams())

    logits = torch.zeros((1, VOCAB_SIZE))
    state.apply(logits, _thinking_context(req_idx, [12]))
    assert torch.count_nonzero(logits) == 0


def test_cpu_greedy_sampling_applies_thinking_budget() -> None:
    req_states = _make_req_states([1, START, 10, 11, 12])
    req_idx = req_states.req_id_to_index["req"]
    sampler = CPUSampler(
        vllm_config=SimpleNamespace(reasoning_config=ReasoningConfig()),
        max_num_reqs=4,
        vocab_size=VOCAB_SIZE,
        device=torch.device("cpu"),
        req_states=req_states,
    )
    sampler.add_request(
        req_idx,
        SamplingParams(temperature=0.0, thinking_token_budget=3),
    )

    idx_mapping = torch.tensor([req_idx], dtype=torch.int64)
    logits = sampler.apply_sampling_params(
        torch.zeros((1, VOCAB_SIZE)),
        idx_mapping,
        idx_mapping,
        idx_mapping.numpy(),
        torch.tensor([4], dtype=torch.int32),
        torch.tensor([12], dtype=torch.int32),
        torch.tensor([0], dtype=torch.int32),
        np.array([4], dtype=np.int64),
    )

    assert logits[0, END] == pytest.approx(1.0e9)


def test_cpu_trace_replay_uses_current_decode_step() -> None:
    req_states = _make_req_states([1, 2, 7], prompt_len=2)
    req_idx = req_states.req_id_to_index["req"]
    state = CPUTraceReplayState(req_states)
    state.add_request(req_idx, SamplingParams(trace_decode_token_ids=[11, 12, 13]))
    state.apply_staged_writes()

    for step, expected in enumerate([11, 12, 13]):
        req_states.total_len.gpu[req_idx] = 2 + step
        sampled = torch.tensor([99], dtype=torch.int64)
        state.apply_trace(sampled, torch.tensor([req_idx], dtype=torch.int64))
        assert sampled.tolist() == [expected]


def test_cpu_trace_replay_skips_inactive_and_exhausted_rows() -> None:
    req_states = _make_req_states([1, 2, 7], prompt_len=2)
    traced_idx = req_states.req_id_to_index["req"]
    state = CPUTraceReplayState(req_states)
    state.add_request(traced_idx, SamplingParams(trace_decode_token_ids=[11]))
    state.apply_staged_writes()

    sampled = torch.tensor([20, 21], dtype=torch.int64)
    state.apply_trace(
        sampled,
        torch.tensor([traced_idx, -1], dtype=torch.int64),
    )
    assert sampled.tolist() == [20, 21]


def test_cpu_trace_replay_clears_reused_slot() -> None:
    req_states = _make_req_states([1, 2], prompt_len=2)
    req_idx = req_states.req_id_to_index["req"]
    state = CPUTraceReplayState(req_states)
    state.add_request(req_idx, SamplingParams(trace_decode_token_ids=[11]))
    state.add_request(req_idx, SamplingParams())
    state.apply_staged_writes()

    sampled = torch.tensor([20], dtype=torch.int64)
    state.apply_trace(sampled, torch.tensor([req_idx], dtype=torch.int64))
    assert sampled.tolist() == [20]


def test_cpu_nan_count_and_sampling_mask() -> None:
    logits = torch.tensor(
        [
            [0.0, -torch.inf, torch.inf, 1.0, torch.nan, -torch.inf, 2.0, -torch.inf],
            [0.0] * 8,
        ]
    )
    num_nans = CPUSampler._get_num_nans(logits)
    assert num_nans.tolist() == [1, 0]

    mask = CPUSamplingMaskTensors.from_logits(
        logits,
        torch.tensor([0, 1, 2], dtype=torch.int32),
        torch.tensor([1, 0], dtype=torch.int32),
        max_num_kept=2,
    )
    assert mask.counts.tolist() == [3, 0]
    assert mask.token_ids[0].tolist() == [0, 3]
    assert mask.packed_mask[0].tolist() == [73]


def test_cpu_sampling_mask_uses_request_boundaries_and_exact_fallback() -> None:
    logits = torch.full((6, 10), -torch.inf)
    logits[0, [1, 3]] = 0
    logits[2, [2, 4, 6, 8]] = 0
    logits[5, [0, 9]] = 0

    mask = CPUSamplingMaskTensors.from_logits(
        logits,
        torch.tensor([0, 2, 5, 6], dtype=torch.int32),
        torch.tensor([1, 0, 1], dtype=torch.int32),
        max_num_kept=2,
    )

    assert mask.tolists().to_nested_list() == [[1, 3], [], [0, 9]]


@pytest.mark.parametrize("max_num_kept", [512, 20_001])
def test_cpu_sampling_mask_matches_wide_finite_support(max_num_kept: int) -> None:
    vocab_size = 20_001
    sizes = [0, 1, 511, 512, 513, 2053, vocab_size]
    generator = torch.Generator().manual_seed(0)
    logits = torch.full((len(sizes), vocab_size), -torch.inf)
    expected = []
    for row, size in enumerate(sizes):
        kept = torch.randperm(vocab_size, generator=generator)[:size].sort().values
        logits[row, kept] = 0
        expected.append(kept.tolist())
    num_sampled = torch.ones(len(sizes), dtype=torch.int32)
    num_sampled[1] = 0
    expected[1] = []

    mask = CPUSamplingMaskTensors.from_logits(
        logits,
        torch.arange(len(sizes) + 1, dtype=torch.int32),
        num_sampled,
        max_num_kept,
    )

    assert mask.tolists().to_nested_list() == expected


class _AddOneProcessor(LogitsProcessor):
    def __init__(self) -> None:
        pass

    def add_request(self, req_idx: int, sampling_params: SamplingParams) -> bool:
        return True

    def apply(self, logits: torch.Tensor, ctx: LogitsContext) -> torch.Tensor:
        return logits.add_(1)


def test_cpu_sampler_runs_custom_logits_processors() -> None:
    req_states = _make_req_states([1])
    req_idx = req_states.req_id_to_index["req"]
    sampler = CPUSampler(
        vllm_config=SimpleNamespace(reasoning_config=None),
        max_num_reqs=4,
        vocab_size=VOCAB_SIZE,
        device=torch.device("cpu"),
        req_states=req_states,
        custom_logits_processors=[_AddOneProcessor()],
    )
    sampler.add_request(req_idx, SamplingParams(temperature=0.0))
    idx_mapping = torch.tensor([req_idx], dtype=torch.int64)

    logits = sampler.apply_sampling_params(
        torch.zeros((1, VOCAB_SIZE)),
        idx_mapping,
        idx_mapping,
        idx_mapping.numpy(),
        torch.tensor([0], dtype=torch.int32),
        torch.tensor([1], dtype=torch.int32),
        torch.tensor([0], dtype=torch.int32),
        np.array([1], dtype=np.int64),
    )

    assert torch.all(logits == 1)


@pytest.mark.parametrize("algorithm", ["gumbel", "dual-key"])
@pytest.mark.parametrize("watermarking", [False, True])
def test_cpu_watermark_sampler_uses_cpu_sampler_pipeline(
    algorithm: str,
    watermarking: bool,
) -> None:
    assert issubclass(CPUWatermarkSampler, CPUSampler)
    req_states = _make_req_states([1])
    req_idx = req_states.req_id_to_index["req"]
    watermarker = (
        GumbelWatermarker(key=42, context_width=1)
        if algorithm == "gumbel"
        else DualKeyGumbelWatermarker(key=42, context_width=1)
    )
    sampler = CPUWatermarkSampler(
        watermarker,
        vllm_config=SimpleNamespace(reasoning_config=None),
        max_num_reqs=4,
        vocab_size=VOCAB_SIZE,
        device=torch.device("cpu"),
        req_states=req_states,
    )
    sampler.add_request(
        req_idx,
        SamplingParams(
            temperature=1.0,
            seed=7,
            watermarking=watermarking,
        ),
    )
    sampler.apply_staged_writes()

    sampled, output_logits = sampler._sample_random(
        torch.linspace(-1, 1, VOCAB_SIZE).unsqueeze(0),
        torch.tensor([req_idx], dtype=torch.int64),
        np.array([req_idx], dtype=np.intp),
        torch.tensor([1], dtype=torch.int64),
        top_k=None,
        top_p=None,
        use_fused_sampler=False,
    )

    assert sampled.shape == (1,)
    assert 0 <= sampled.item() < VOCAB_SIZE
    assert output_logits.shape == (1, VOCAB_SIZE)


def test_cpu_watermark_sampler_detects_repeated_context() -> None:
    req_states = _make_req_states([9, 1, 2, 3, 1, 2], prompt_len=1)
    req_idx = req_states.req_id_to_index["req"]
    sampler = CPUWatermarkSampler(
        GumbelWatermarker(key=42, context_width=2),
        vllm_config=SimpleNamespace(reasoning_config=None),
        max_num_reqs=4,
        vocab_size=VOCAB_SIZE,
        device=torch.device("cpu"),
        req_states=req_states,
    )

    req_indices = torch.tensor([req_idx], dtype=torch.int64)
    contexts = sampler._get_contexts(req_indices)

    assert contexts.tolist() == [[1, 2]]
    assert sampler._get_repeated_contexts(req_indices, contexts).tolist() == [True]
