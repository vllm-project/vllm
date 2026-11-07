# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Sequence
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.config import ModelConfig, VllmConfig

pytest.importorskip("triton")
if not torch.cuda.is_available():
    pytest.skip("CUDA required for sampler flag tests", allow_module_level=True)

from vllm.sampling_params import SamplingParams
from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p
from vllm.v1.worker.gpu.sample.gumbel import _gumbel_sample_kernel, gumbel_sample
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
    ("sampling_params", "expected", "may_mask"),
    [
        pytest.param(SamplingParams(), False, False, id="defaults"),
        pytest.param(SamplingParams(temperature=0.0), False, False, id="greedy"),
        pytest.param(
            SamplingParams(thinking_token_budget=3), True, True, id="thinking-budget"
        ),
        pytest.param(SamplingParams(logit_bias={1: 1.0}), True, True, id="logit-bias"),
        pytest.param(SamplingParams(frequency_penalty=0.1), True, True, id="penalty"),
        pytest.param(
            SamplingParams(_bad_words_token_ids=[[1]]), True, True, id="bad-words"
        ),
        pytest.param(SamplingParams(temperature=0.7), True, False, id="temperature"),
        pytest.param(SamplingParams(min_p=0.1), True, True, id="min-p"),
        pytest.param(SamplingParams(top_k=10), True, True, id="top-k"),
        pytest.param(SamplingParams(top_p=0.9), True, True, id="top-p"),
        pytest.param(
            SamplingParams.for_sampler_warmup(), True, True, id="all-logits-processors"
        ),
    ],
)
def test_logits_processing_cache_matches_request_features(
    sampling_params: SamplingParams, expected: bool, may_mask: bool
):
    sampler = _make_sampler()
    sampler.add_request(3, sampling_params=sampling_params)

    assert sampler.needs_logits_processing[3] == expected
    # Temperature alone only scales the logits, so it never leaves -inf behind.
    assert sampler.may_mask_logits[3] == may_mask


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


def test_logits_processing_cache_only_checks_active_requests():
    sampler = _make_sampler()
    sampler.add_request(0, SamplingParams(temperature=0.0))
    sampler.add_request(2, SamplingParams.for_sampler_warmup())

    sampling_only = np.array([0], dtype=np.int32)
    with_processing = np.array([0, 2], dtype=np.int32)

    assert not np.any(sampler.needs_logits_processing[sampling_only])
    assert np.any(sampler.needs_logits_processing[with_processing])


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("consumer", ["tokens", "logprobs", "mask"])
@pytest.mark.parametrize(
    "extra",
    [
        {},
        {"top_k": 17},
        {"top_p": 0.8},
        {"min_p": 0.1},
        {"logit_bias": {7: 2.0}},
        {"frequency_penalty": 0.5},
    ],
)
def test_temperature_only_sampling_preserves_tokens_and_consumed_logits(
    dtype, consumer, extra
):
    """Compare fused temperature with the original materialized sampling path.

    Filtering, processors, and consumers of scaled logits must retain their
    original behavior. Reordered slots and padded row strides exercise indexing.
    """
    vocab_size = 8193
    req_states = RequestState(4, 64, 16, 0, vocab_size, DEVICE)
    sampler = Sampler(
        VllmConfig(),
        4,
        vocab_size,
        DEVICE,
        req_states,
        return_sampling_mask=consumer == "mask",
    )
    for i, temperature in enumerate([0.7, 1.0, 1.4, 0.8]):
        req_states.add_request(str(i), 1, [7], 0, 16)
        req_idx = req_states.req_id_to_index[str(i)]
        sampler.add_request(
            req_idx, SamplingParams(temperature=temperature, seed=100 + i, **extra)
        )
    req_states.apply_staged_writes()
    sampler.apply_staged_writes()
    idx_np = np.array([1, 3, 0, 2], dtype=np.int32)
    idx = torch.from_numpy(idx_np).to(DEVICE)
    pos = torch.tensor([1, 3, 2, 4], device=DEVICE, dtype=torch.int64)
    input_ids = torch.full((4,), 7, device=DEVICE, dtype=torch.int32)
    local_pos = torch.zeros_like(input_ids)
    lengths = np.ones(4, dtype=np.int32)
    torch.manual_seed(42)
    logits = torch.randn(4, vocab_size + 15, dtype=dtype, device=DEVICE)[:, :vocab_size]
    original = logits.clone()
    args = (idx, idx, idx_np, pos, input_ids, local_pos, lengths)

    expected_logits = sampler.apply_sampling_params(
        logits, *args, skip_top_k_top_p=True
    )
    top_k, top_p, has_topp_only_rows = sampler.sampling_states.get_top_k_top_p(
        idx, idx_np, vocab_size
    )
    expected_logits = apply_top_k_top_p(
        expected_logits, top_k, top_p, has_topp_only_rows=has_topp_only_rows
    )
    expected_tokens = gumbel_sample(
        expected_logits,
        idx,
        sampler.sampling_states.temperature.gpu,
        sampler.sampling_states.seeds.gpu,
        pos,
        apply_temperature=False,
        is_drafting=False,
    )
    actual_tokens, actual_logits = sampler.sample(
        logits, *args, return_logprobs=consumer == "logprobs"
    )
    assert torch.equal(actual_tokens, expected_tokens)
    assert torch.equal(logits, original)
    if consumer != "tokens":
        assert torch.equal(actual_logits, expected_logits)


@pytest.mark.parametrize("use_fp64", [False, True])
def test_temperature_sampling_warmup_covers_logits_dtypes(tmp_path, use_fp64):
    """Expand real compile keys without loading weights or allocating GPU tensors."""
    from transformers import Qwen3Config

    Qwen3Config(
        architectures=["Qwen3ForCausalLM"],
        vocab_size=8193,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
    ).save_pretrained(tmp_path)
    config = VllmConfig(
        model_config=ModelConfig(
            model=str(tmp_path),
            dtype="bfloat16",
            skip_tokenizer_init=True,
            max_model_len=64,
            use_fp64_gumbel=use_fp64,
        )
    )
    keys = _gumbel_sample_kernel.get_warmup_keys(vllm_config=config)
    inputs = [dict(key.inputs) for key in keys]
    assert {item["logits_ptr"].dtype for item in inputs} == {
        torch.bfloat16,
        torch.float32,
    }
    assert {item["local_max_ptr"].dtype for item in inputs} == {
        torch.float64 if use_fp64 else torch.float32
    }
