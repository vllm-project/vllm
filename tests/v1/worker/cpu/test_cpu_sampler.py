# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import numpy as np
import pytest
import torch
from transformers import LlamaConfig

import vllm.config.reasoning as reasoning_config_module
from vllm.config import ModelConfig, VllmConfig
from vllm.config.reasoning import ReasoningConfig as VllmReasoningConfig
from vllm.platforms import CpuArchEnum, current_platform
from vllm.sampling_params import SamplingParams
from vllm.triton_utils import HAS_TRITON
from vllm.v1.watermarking.gumbel import (
    DualKeyGumbelWatermarker,
    GumbelWatermarker,
)
from vllm.v1.worker.cpu.logits_processors import LogitsContext
from vllm.v1.worker.cpu.sampler import (
    CPUSampler,
    CPUSamplingMaskTensors,
    CPUThinkingBudgetState,
    CPUTraceReplayState,
    CPUWatermarkSampler,
    get_num_nans,
)
from vllm.v1.worker.cpu.states import CPURequestState

if current_platform.is_cpu():
    import vllm._C  # noqa: F401

pytestmark = pytest.mark.skipif(
    not current_platform.is_cpu()
    or HAS_TRITON
    or current_platform.get_cpu_architecture() == CpuArchEnum.X86,
    reason="requires a non-x86 CPU platform without Triton",
)

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


class _ReasoningTokenizer:
    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        del add_special_tokens
        return {"<reasoning>": [START], "</reasoning>": [END]}[text]


def _make_req_states(tokens: list[int], prompt_len: int = 1) -> CPURequestState:
    req_states = CPURequestState(
        max_num_reqs=4,
        max_model_len=max(64, len(tokens) + 2),
        max_num_batched_tokens=32,
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
    return req_states


def _vllm_config(monkeypatch: pytest.MonkeyPatch, tmp_path) -> VllmConfig:
    LlamaConfig(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=1,
        num_key_value_heads=1,
        max_position_embeddings=64,
        vocab_size=VOCAB_SIZE,
        architectures=["LlamaForCausalLM"],
    ).save_pretrained(tmp_path)
    model_config = ModelConfig(
        model=str(tmp_path),
        dtype="float32",
        max_model_len=64,
        skip_tokenizer_init=True,
    )
    reasoning_config = VllmReasoningConfig(
        reasoning_start_str="<reasoning>",
        reasoning_end_str="</reasoning>",
    )
    monkeypatch.setattr(
        reasoning_config_module,
        "cached_tokenizer_from_config",
        lambda **_: _ReasoningTokenizer(),
    )
    reasoning_config.initialize_token_ids(model_config)
    return VllmConfig(
        model_config=model_config,
        reasoning_config=reasoning_config,
    )


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


def test_cpu_thinking_budget_forces_and_continues_end_marker() -> None:
    tokens = [1, START, 10, 11, 12]
    req_states = _make_req_states(tokens)
    req_idx = req_states.req_id_to_index["req"]
    state = CPUThinkingBudgetState(req_states, MultiTokenEndReasoningConfig())
    state.add_request(req_idx, SamplingParams(thinking_token_budget=3))

    logits = torch.zeros((1, VOCAB_SIZE))
    logits[:, [END_A, END_B]] = -torch.inf
    state.apply(logits, _thinking_context(req_idx, [12]))

    assert logits[0, END_A] == pytest.approx(1.0e9)

    total_len = int(req_states.total_len[req_idx])
    req_states.all_token_ids[req_idx, total_len] = END_A
    req_states.total_len[req_idx] = total_len + 1
    logits = torch.zeros((1, VOCAB_SIZE))
    state.apply(logits, _thinking_context(req_idx, [END_A]))

    assert logits[0, END_B] == pytest.approx(1.0e9)


def test_cpu_thinking_budget_uses_distinct_forced_end_marker() -> None:
    req_states = _make_req_states([1, START, 10, 11, 12])
    req_idx = req_states.req_id_to_index["req"]
    state = CPUThinkingBudgetState(req_states, DistinctEndReasoningConfig())
    state.add_request(req_idx, SamplingParams(thinking_token_budget=3))

    logits = torch.zeros((1, VOCAB_SIZE))
    state.apply(logits, _thinking_context(req_idx, [12]))

    assert logits[0, END_A] == pytest.approx(1.0e9)

    total_len = int(req_states.total_len[req_idx])
    req_states.all_token_ids[req_idx, total_len] = END_A
    req_states.total_len[req_idx] = total_len + 1
    logits = torch.zeros((1, VOCAB_SIZE))
    state.apply(logits, _thinking_context(req_idx, [END_A]))

    assert logits[0, END_B] == pytest.approx(1.0e9)


def test_cpu_thinking_budget_stops_after_natural_end() -> None:
    tokens = [1, START, 10, END, 20, 21, 22]
    req_states = _make_req_states(tokens)
    req_idx = req_states.req_id_to_index["req"]
    state = CPUThinkingBudgetState(req_states, ReasoningConfig())
    state.add_request(req_idx, SamplingParams(thinking_token_budget=2))

    logits = torch.zeros((1, VOCAB_SIZE))
    state.apply(logits, _thinking_context(req_idx, [22]))
    assert torch.count_nonzero(logits) == 0


def test_cpu_thinking_budget_ignores_plain_request() -> None:
    req_states = _make_req_states([1, START, 10, 11, 12])
    req_idx = req_states.req_id_to_index["req"]
    state = CPUThinkingBudgetState(req_states, ReasoningConfig())
    state.add_request(req_idx, SamplingParams())

    logits = torch.zeros((1, VOCAB_SIZE))
    state.apply(logits, _thinking_context(req_idx, [12]))

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

    req_states.all_token_ids[req_idx, len(tokens)] = 10
    req_states.total_len[req_idx] = len(tokens) + 1
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


def test_cpu_greedy_sampling_applies_thinking_budget(monkeypatch, tmp_path) -> None:
    req_states = _make_req_states([1, START, 10, 11, 12])
    req_idx = req_states.req_id_to_index["req"]
    sampler = CPUSampler(
        vllm_config=_vllm_config(monkeypatch, tmp_path),
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

    for step, expected in enumerate([11, 12, 13]):
        req_states.total_len[req_idx] = 2 + step
        sampled = torch.tensor([99], dtype=torch.int64)
        state.apply_trace(sampled, torch.tensor([req_idx], dtype=torch.int64))
        assert sampled.tolist() == [expected]


def test_cpu_trace_replay_maps_requests_and_skips_inactive_rows() -> None:
    req_states = CPURequestState(
        max_num_reqs=4,
        max_model_len=64,
        max_num_batched_tokens=16,
        vocab_size=VOCAB_SIZE,
        device=torch.device("cpu"),
    )
    req_states.add_request("plain", 2, [1, 2], 2, 16)
    req_states.add_request("traced", 2, [3, 4, 7], 3, 16)
    plain_idx = req_states.req_id_to_index["plain"]
    traced_idx = req_states.req_id_to_index["traced"]
    state = CPUTraceReplayState(req_states)
    state.add_request(plain_idx, SamplingParams())
    state.add_request(traced_idx, SamplingParams(trace_decode_token_ids=[11, 12]))

    sampled = torch.tensor([20, 21, 22], dtype=torch.int64)
    state.apply_trace(
        sampled,
        torch.tensor([traced_idx, -1, plain_idx], dtype=torch.int64),
    )
    assert sampled.tolist() == [12, 21, 22]

    req_states.total_len[traced_idx] = 4
    sampled = torch.tensor([30], dtype=torch.int64)
    state.apply_trace(sampled, torch.tensor([traced_idx], dtype=torch.int64))
    assert sampled.tolist() == [30]


def test_cpu_trace_replay_clears_reused_slot() -> None:
    req_states = _make_req_states([1, 2], prompt_len=2)
    req_idx = req_states.req_id_to_index["req"]
    state = CPUTraceReplayState(req_states)
    state.add_request(req_idx, SamplingParams(trace_decode_token_ids=[11]))
    state.add_request(req_idx, SamplingParams())

    sampled = torch.tensor([20], dtype=torch.int64)
    state.apply_trace(sampled, torch.tensor([req_idx], dtype=torch.int64))
    assert sampled.tolist() == [20]


def test_cpu_nan_count() -> None:
    logits = torch.tensor(
        [
            [0.0, -torch.inf, torch.inf, 1.0, torch.nan, -torch.inf, 2.0, -torch.inf],
            [0.0] * 8,
        ]
    )
    num_nans = get_num_nans(logits)
    assert num_nans.tolist() == [1, 0]


def test_cpu_sampling_mask_tracks_finite_tokens() -> None:
    logits = torch.tensor(
        [
            [0.0, -torch.inf, torch.inf, 1.0, torch.nan, -torch.inf, 2.0, -torch.inf],
            [0.0] * 8,
        ]
    )
    mask = CPUSamplingMaskTensors.from_logits(
        logits,
        torch.tensor([0, 1, 2], dtype=torch.int32),
        torch.tensor([1, 0], dtype=torch.int32),
        max_num_kept=2,
    )
    assert mask.counts.tolist() == [3, 0]
    assert mask.token_ids[0].tolist() == [0, 3]
    assert mask.packed_mask[0].tolist() == [73]


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


def _make_sampler(
    sampling_params: list[SamplingParams],
) -> tuple[CPUSampler, torch.Tensor]:
    req_states = CPURequestState(
        max_num_reqs=4,
        max_model_len=64,
        max_num_batched_tokens=16,
        vocab_size=VOCAB_SIZE,
        device=torch.device("cpu"),
    )
    sampler = CPUSampler(
        vllm_config=VllmConfig(),
        max_num_reqs=4,
        vocab_size=VOCAB_SIZE,
        device=torch.device("cpu"),
        req_states=req_states,
    )
    req_indices = []
    for i, params in enumerate(sampling_params):
        req_id = f"req-{i}"
        req_states.add_request(req_id, 1, [i + 1], 1, 16)
        req_idx = req_states.req_id_to_index[req_id]
        req_indices.append(req_idx)
        sampler.add_request(req_idx, params)
    return sampler, torch.tensor(req_indices, dtype=torch.int64)


@pytest.mark.parametrize(
    ("sampling_params", "expected_finite"),
    [
        (SamplingParams(top_k=2), [2, 3]),
        (SamplingParams(top_p=0.7), [2, 3]),
        (SamplingParams(top_k=3, top_p=0.7), [2, 3]),
        (SamplingParams(top_k=VOCAB_SIZE + 1, top_p=0.7), [2, 3]),
    ],
)
def test_cpu_sampler_applies_top_k_top_p(
    sampling_params: SamplingParams,
    expected_finite: list[int],
) -> None:
    sampler, idx_mapping = _make_sampler([sampling_params])

    logits = sampler.apply_sampling_params(
        torch.tensor([[0.0, 1.0, 2.0, 3.0]]),
        idx_mapping,
        idx_mapping,
        idx_mapping.numpy(),
        torch.tensor([1], dtype=torch.int64),
        torch.tensor([1], dtype=torch.int32),
        torch.tensor([0], dtype=torch.int32),
        np.array([1], dtype=np.int64),
    )

    assert torch.isfinite(logits[0]).nonzero().flatten().tolist() == expected_finite


def test_cpu_sampler_applies_temperature() -> None:
    sampler, idx_mapping = _make_sampler([SamplingParams(temperature=0.5)])
    logits = torch.tensor([[0.0, 1.0, 2.0, 3.0]])

    actual = sampler.apply_sampling_params(
        logits,
        idx_mapping,
        idx_mapping,
        idx_mapping.numpy(),
        torch.tensor([1], dtype=torch.int64),
        torch.tensor([1], dtype=torch.int32),
        torch.tensor([0], dtype=torch.int32),
        np.array([1], dtype=np.int64),
    )

    torch.testing.assert_close(actual, logits / 0.5)


def test_cpu_sampler_only_processes_active_requests() -> None:
    sampler, idx_mapping = _make_sampler(
        [SamplingParams(temperature=0.0), SamplingParams(temperature=0.5)]
    )
    logits = torch.tensor([[0.0, 1.0, 2.0, 3.0]])

    greedy = sampler.apply_sampling_params(
        logits,
        idx_mapping[:1],
        idx_mapping[:1],
        idx_mapping[:1].numpy(),
        torch.tensor([1], dtype=torch.int32),
        torch.tensor([1], dtype=torch.int32),
        torch.tensor([0], dtype=torch.int32),
        np.ones(1, dtype=np.int64),
    )
    scaled = sampler.apply_sampling_params(
        logits,
        idx_mapping[1:],
        idx_mapping[1:],
        idx_mapping[1:].numpy(),
        torch.tensor([1], dtype=torch.int32),
        torch.tensor([2], dtype=torch.int32),
        torch.tensor([0], dtype=torch.int32),
        np.ones(1, dtype=np.int64),
    )

    torch.testing.assert_close(greedy, logits)
    torch.testing.assert_close(scaled, logits / 0.5)


def test_cpu_sampler_clears_processing_when_slot_is_reused() -> None:
    sampler, idx_mapping = _make_sampler([SamplingParams(temperature=0.5)])
    req_idx = int(idx_mapping[0])
    sampler.add_request(req_idx, SamplingParams(temperature=0.0))
    logits = torch.tensor([[0.0, 1.0, 2.0, 3.0]])

    actual = sampler.apply_sampling_params(
        logits,
        idx_mapping,
        idx_mapping,
        idx_mapping.numpy(),
        torch.tensor([1], dtype=torch.int32),
        torch.tensor([1], dtype=torch.int32),
        torch.tensor([0], dtype=torch.int32),
        np.ones(1, dtype=np.int64),
    )

    torch.testing.assert_close(actual, logits)


def test_cpu_sampler_greedy_returns_argmax() -> None:
    sampler, idx_mapping = _make_sampler([SamplingParams(temperature=0.0)])
    logits = torch.tensor([[0.0, 3.0, 2.0, 1.0]])

    sampled, _ = sampler.sample(
        logits,
        idx_mapping,
        idx_mapping,
        idx_mapping.numpy(),
        torch.tensor([1], dtype=torch.int64),
        torch.tensor([1], dtype=torch.int32),
        torch.tensor([0], dtype=torch.int32),
        np.ones(1, dtype=np.int64),
    )

    assert sampled.tolist() == [1]


@pytest.mark.parametrize("algorithm", ["gumbel", "dual-key"])
@pytest.mark.parametrize("watermarking", [False, True])
def test_cpu_watermark_sampler_uses_cpu_sampler_pipeline(
    algorithm: str,
    watermarking: bool,
) -> None:
    req_states = _make_req_states([1])
    req_idx = req_states.req_id_to_index["req"]
    watermarker = (
        GumbelWatermarker(key=42, context_width=1)
        if algorithm == "gumbel"
        else DualKeyGumbelWatermarker(key=42, context_width=1)
    )
    sampler = CPUWatermarkSampler(
        watermarker,
        vllm_config=VllmConfig(),
        max_num_reqs=4,
        vocab_size=VOCAB_SIZE,
        device=torch.device("cpu"),
        req_states=req_states,
    )
    params = SamplingParams(
        temperature=1.0,
        top_k=4,
        top_p=0.9,
        seed=7,
        watermarking=watermarking,
    )
    sampler.add_request(req_idx, params)

    idx_mapping = torch.tensor([req_idx], dtype=torch.int64)
    logits = torch.linspace(-1, 1, VOCAB_SIZE).unsqueeze(0)
    sample_args = (
        logits,
        idx_mapping,
        idx_mapping,
        idx_mapping.numpy(),
        torch.tensor([1], dtype=torch.int64),
        torch.tensor([1], dtype=torch.int32),
        torch.tensor([0], dtype=torch.int32),
        np.ones(1, dtype=np.int64),
    )
    sampled, output_logits = sampler.sample(*sample_args)
    repeated, repeated_logits = sampler.sample(*sample_args)

    assert sampled.item() in range(VOCAB_SIZE - 4, VOCAB_SIZE)
    assert torch.equal(repeated, sampled)
    torch.testing.assert_close(repeated_logits, output_logits)
    assert torch.isfinite(output_logits[0, sampled.item()])

    ordinary = CPUSampler(
        vllm_config=VllmConfig(),
        max_num_reqs=4,
        vocab_size=VOCAB_SIZE,
        device=torch.device("cpu"),
        req_states=req_states,
    )
    ordinary.add_request(req_idx, params)
    expected, expected_logits = ordinary.sample(*sample_args)
    if watermarking:
        assert not torch.equal(sampled, expected)
    else:
        assert torch.equal(sampled, expected)
        torch.testing.assert_close(output_logits, expected_logits)


def test_cpu_watermark_sampler_handles_mixed_rows() -> None:
    req_states = CPURequestState(
        max_num_reqs=4,
        max_model_len=64,
        max_num_batched_tokens=16,
        vocab_size=VOCAB_SIZE,
        device=torch.device("cpu"),
    )
    params = [
        SamplingParams(temperature=1.0, top_k=4, seed=7, watermarking=True),
        SamplingParams(temperature=1.0, top_k=4, seed=11, watermarking=False),
        SamplingParams(temperature=0.0, top_k=4, watermarking=True),
    ]
    req_indices = []
    for i in range(len(params)):
        req_id = f"req-{i}"
        req_states.add_request(req_id, 1, [9, i + 1], 2, 16)
        req_indices.append(req_states.req_id_to_index[req_id])
    idx_mapping = torch.tensor(req_indices, dtype=torch.int64)

    sampler = CPUWatermarkSampler(
        GumbelWatermarker(key=42, context_width=1),
        vllm_config=VllmConfig(),
        max_num_reqs=4,
        vocab_size=VOCAB_SIZE,
        device=torch.device("cpu"),
        req_states=req_states,
        deduplicate_contexts="none",
    )
    ordinary = CPUSampler(
        vllm_config=VllmConfig(),
        max_num_reqs=4,
        vocab_size=VOCAB_SIZE,
        device=torch.device("cpu"),
        req_states=req_states,
    )
    for req_idx, sampling_params in zip(req_indices, params, strict=True):
        sampler.add_request(req_idx, sampling_params)
        ordinary.add_request(req_idx, sampling_params)

    logits = torch.linspace(-1, 1, VOCAB_SIZE).repeat(3, 1)
    sample_args = (
        logits,
        idx_mapping,
        idx_mapping,
        idx_mapping.numpy(),
        torch.tensor([2, 2, 2], dtype=torch.int64),
        torch.tensor([1, 2, 3], dtype=torch.int32),
        torch.zeros(3, dtype=torch.int32),
        np.full(3, 2, dtype=np.int64),
    )
    sampled, _ = sampler.sample(*sample_args)
    ordinary_sampled, _ = ordinary.sample(*sample_args)

    assert sampled[0] != ordinary_sampled[0]
    assert sampled[1] == ordinary_sampled[1]
    assert sampled[2] == logits[2].argmax()
    assert torch.all(sampled >= VOCAB_SIZE - 4)


def test_cpu_watermark_sampler_detects_repeated_context() -> None:
    req_states = _make_req_states([9, 1, 2, 3, 1, 2], prompt_len=1)
    req_idx = req_states.req_id_to_index["req"]
    sampler = CPUWatermarkSampler(
        GumbelWatermarker(key=42, context_width=2),
        vllm_config=VllmConfig(),
        max_num_reqs=4,
        vocab_size=VOCAB_SIZE,
        device=torch.device("cpu"),
        req_states=req_states,
    )

    req_indices = torch.tensor([req_idx], dtype=torch.int64)
    contexts = sampler._get_contexts(req_indices)

    assert contexts.tolist() == [[1, 2]]
    assert sampler._get_repeated_contexts(req_indices, contexts).tolist() == [True]


def test_cpu_watermark_deduplication_scope_and_partial_context() -> None:
    req_states = _make_req_states([1, 2, 1, 2], prompt_len=2)
    req_idx = req_states.req_id_to_index["req"]
    req_indices = torch.tensor([req_idx], dtype=torch.int64)

    single_turn = CPUWatermarkSampler(
        GumbelWatermarker(key=42, context_width=2),
        vllm_config=VllmConfig(),
        max_num_reqs=4,
        vocab_size=VOCAB_SIZE,
        device=torch.device("cpu"),
        req_states=req_states,
        deduplicate_contexts="single_turn",
    )
    all_history = CPUWatermarkSampler(
        GumbelWatermarker(key=42, context_width=2),
        vllm_config=VllmConfig(),
        max_num_reqs=4,
        vocab_size=VOCAB_SIZE,
        device=torch.device("cpu"),
        req_states=req_states,
        deduplicate_contexts="all",
    )
    contexts = single_turn._get_contexts(req_indices)

    assert single_turn._get_repeated_contexts(req_indices, contexts).tolist() == [False]
    assert all_history._get_repeated_contexts(req_indices, contexts).tolist() == [True]

    partial_states = _make_req_states([9, 1], prompt_len=1)
    partial_idx = partial_states.req_id_to_index["req"]
    partial_sampler = CPUWatermarkSampler(
        GumbelWatermarker(key=42, context_width=2),
        vllm_config=VllmConfig(),
        max_num_reqs=4,
        vocab_size=VOCAB_SIZE,
        device=torch.device("cpu"),
        req_states=partial_states,
        deduplicate_contexts="all",
    )
    partial_indices = torch.tensor([partial_idx], dtype=torch.int64)
    partial_context = partial_sampler._get_contexts(partial_indices)

    assert partial_context.tolist() == [[-1, 1]]
    assert partial_sampler._get_repeated_contexts(
        partial_indices, partial_context
    ).tolist() == [True]
