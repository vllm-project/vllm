# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU speculative-decoding end-to-end correctness."""

import os

import pytest
import torch

from tests.models.utils import check_logprobs_close
from tests.v1.e2e.spec_decode.utils import compute_acceptance_len
from vllm import LLM, SamplingParams
from vllm.platforms import current_platform
from vllm.triton_utils import HAS_TRITON
from vllm.triton_utils.dispatcher import KernelOverride
from vllm.utils.cpu_triton_utils import (
    _rejection_random_sample_kernel_impl,
    _sample_recovered_tokens_kernel_impl,
)
from vllm.v1.sample.rejection_sampler import (
    rejection_random_sample_kernel,
    sample_recovered_tokens_kernel,
)

pytestmark = pytest.mark.cpu_model

if not current_platform.is_cpu():
    pytest.skip("skipping CPU-only tests", allow_module_level=True)

# Bound the KV cache so the run does not scale with host memory.
os.environ.setdefault("VLLM_CPU_KVCACHE_SPACE", "1")

MODEL = "Qwen/Qwen3-0.6B"
NUM_LOGPROBS = 5
SP = SamplingParams(max_tokens=48, temperature=0, logprobs=NUM_LOGPROBS)

# Repetitive prompts so the ngram proposer finds matches and actually drafts;
# without hits the run is indistinguishable from plain decoding.
PROMPTS = [
    "The capital of France is Paris. The capital of Italy is Rome. "
    "The capital of Spain is Madrid. The capital of France is",
    "def add(a, b):\n    return a + b\n\ndef sub(a, b):\n    return a - b\n\n"
    "def mul(a, b):\n    return",
    "one two three four five one two three four five one two three four",
]


def _make_llm(**overrides) -> LLM:
    base = dict(
        model=MODEL,
        dtype="bfloat16",
        max_model_len=1024,
        disable_log_stats=False,
    )
    base.update(overrides)
    return LLM(**base)


def _tuples(outputs) -> list[tuple[list[int], str, object]]:
    return [
        (list(o.outputs[0].token_ids), o.outputs[0].text, o.outputs[0].logprobs)
        for o in outputs
    ]


@pytest.fixture(scope="module")
def baseline_refs():
    llm = _make_llm()
    refs = _tuples(llm.generate(PROMPTS, SP))
    del llm
    return refs


def _spec_metric(llm: LLM, name: str) -> float:
    for metric in llm.get_metrics():
        if metric.name == name:
            return float(getattr(metric, "value", 0.0))
    return 0.0


def test_cpu_rejection_sampler_supports_narrow_draft_probs():
    current_platform.import_kernels()
    batch_size = 32
    draft_probs = torch.tensor([[1.0, 0.0]]).repeat(batch_size, 1)
    target_probs = torch.tensor([[0.0, 0.0, 1.0]]).repeat(batch_size, 1)
    draft_token_ids = torch.zeros(batch_size, dtype=torch.int32)
    cu_num_draft_tokens = torch.arange(1, batch_size + 1, dtype=torch.int32)
    recovered_token_ids = torch.empty(batch_size, dtype=torch.int32)

    recovered_kernel = KernelOverride(
        sample_recovered_tokens_kernel,
        _sample_recovered_tokens_kernel_impl,
    )
    recovered_kernel[(batch_size, 1)](
        output_token_ids_ptr=recovered_token_ids,
        cu_num_draft_tokens_ptr=cu_num_draft_tokens,
        draft_token_ids_ptr=draft_token_ids,
        draft_probs_ptr=draft_probs,
        draft_probs_stride=draft_probs.stride(0),
        target_probs_ptr=target_probs,
        inv_q_ptr=torch.ones((batch_size, 3)),
        vocab_size=3,
        draft_vocab_size=2,
        BLOCK_SIZE=8192,
        NO_DRAFT_PROBS=False,
        USE_FP64_GUMBEL=False,
    )

    expected_target_only = torch.full((batch_size,), 2, dtype=torch.int32)
    assert torch.equal(recovered_token_ids, expected_target_only)

    output_token_ids = torch.full((batch_size, 2), -1, dtype=torch.int32)
    random_kernel = KernelOverride(
        rejection_random_sample_kernel,
        _rejection_random_sample_kernel_impl,
    )
    random_kernel[(batch_size,)](
        output_token_ids_ptr=output_token_ids,
        cu_num_draft_tokens_ptr=cu_num_draft_tokens,
        draft_token_ids_ptr=torch.full((batch_size,), 2, dtype=torch.int32),
        draft_probs_ptr=draft_probs,
        draft_probs_stride=draft_probs.stride(0),
        target_probs_ptr=target_probs,
        bonus_token_ids_ptr=torch.zeros((batch_size, 1), dtype=torch.int32),
        recovered_token_ids_ptr=torch.ones(batch_size, dtype=torch.int32),
        uniform_probs_ptr=torch.full((batch_size,), 0.5, dtype=torch.float64),
        is_greedy_ptr=torch.zeros(batch_size, dtype=torch.bool),
        max_spec_len=1,
        vocab_size=3,
        draft_vocab_size=2,
        synthetic_conditional_rates_ptr=None,
        NO_DRAFT_PROBS=False,
        SYNTHETIC_MODE=False,
    )

    expected_recovered = torch.ones(batch_size, dtype=torch.int32)
    assert torch.equal(output_token_ids[:, 0], expected_recovered)


def test_ngram_spec_decode_matches_baseline(baseline_refs):
    """Greedy output must be unchanged by speculative decoding.

    The drafted-token check keeps the test from passing vacuously when the
    proposer silently produces nothing.
    """
    llm = _make_llm(
        speculative_config={
            "method": "ngram",
            "prompt_lookup_max": 5,
            "prompt_lookup_min": 3,
            "num_speculative_tokens": 3,
        },
    )
    got = _tuples(llm.generate(PROMPTS, SP))
    drafted = _spec_metric(llm, "vllm:spec_decode_num_draft_tokens")
    accepted = _spec_metric(llm, "vllm:spec_decode_num_accepted_tokens")
    del llm

    assert drafted > 0, "ngram proposer drafted no tokens; the test would be vacuous"
    assert accepted > 0, f"no draft tokens accepted out of {drafted} drafted"

    check_logprobs_close(
        outputs_0_lst=baseline_refs,
        outputs_1_lst=got,
        name_0="no_spec_decode",
        name_1="ngram_spec_decode",
    )


SMOKE_PROMPTS = [
    "The capital of France is",
    "2 + 2 equals",
    "In one word, the color of the sky is",
    "Q: If a train travels 60 miles in 1.5 hours, what is its average speed?\nA:",
]
SMOKE_SP = SamplingParams(temperature=0.0, max_tokens=32, ignore_eos=True)

# Methods routed to the V2 runner, using the same target/draft pairs as the GPU
# suite. MTP needs no separate draft checkpoint: its speculator layer is
# trained jointly with, and shipped inside, the target model's own weights.
SMOKE_CONFIGS = [
    pytest.param(
        "meta-llama/Llama-3.2-1B-Instruct",
        {
            "method": "eagle3",
            "model": "nm-testing/Llama3_2_1B_speculator.eagle3",
            "num_speculative_tokens": 3,
        },
        id="eagle3",
    ),
    pytest.param(
        "Qwen/Qwen3-4B",
        {
            "method": "dflash",
            "model": "z-lab/Qwen3-4B-DFlash-b16",
            # The checkpoint allows 16, but accepted tokens plateau around 64
            # regardless, so a larger draft only burns CPU on rejected tokens.
            "num_speculative_tokens": 4,
        },
        id="dflash",
    ),
    pytest.param(
        "Qwen/Qwen3.5-0.8B",
        {
            "method": "mtp",
            "num_speculative_tokens": 3,
        },
        id="mtp",
    ),
]

# Measured 1.85 (eagle3), 1.99 (dflash), 2.29 (mtp) on CPU; this floor leaves
# room for numeric drift while still failing if speculation stops paying off.
MIN_ACCEPTANCE_LEN = 1.5


@pytest.mark.skipif(
    not HAS_TRITON, reason="the V2 runner needs triton-cpu, which CI builds"
)
@pytest.mark.parametrize("model,speculative_config", SMOKE_CONFIGS)
def test_v2_speculator_smoke(model: str, speculative_config: dict):
    """Gates that drafts are accepted often enough for speculation to pay off."""
    llm = _make_llm(model=model, speculative_config=speculative_config)
    try:
        assert llm.llm_engine.vllm_config.use_v2_model_runner, (
            f"{speculative_config['method']} fell back to the V1 runner; "
            "this test is meant to cover the V2 path"
        )
        llm.generate(SMOKE_PROMPTS, SMOKE_SP)
        acceptance = compute_acceptance_len(llm.get_metrics())
        assert acceptance >= MIN_ACCEPTANCE_LEN, (
            f"acceptance length {acceptance:.3f} below {MIN_ACCEPTANCE_LEN}; "
            "drafts are being rejected far more than expected"
        )
    finally:
        del llm
