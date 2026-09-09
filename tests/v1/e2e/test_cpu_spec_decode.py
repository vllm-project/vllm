# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU speculative-decoding end-to-end correctness."""

import os

import pytest

from tests.models.utils import check_logprobs_close
from tests.v1.e2e.spec_decode.utils import compute_acceptance_len
from vllm import LLM, SamplingParams
from vllm.platforms import current_platform
from vllm.triton_utils import HAS_TRITON

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
        enforce_eager=True,
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

# Methods routed to the V2 runner, using the same target/draft pairs as the
# GPU suite. DFlash is left out: its only checkpoint drafts for Qwen3-8B.
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
        "Qwen/Qwen3.5-0.8B-Base",
        {"method": "mtp", "num_speculative_tokens": 2},
        id="mtp",
    ),
]


@pytest.mark.skipif(
    not HAS_TRITON, reason="the V2 runner needs triton-cpu, which CI builds"
)
@pytest.mark.parametrize("model,speculative_config", SMOKE_CONFIGS)
def test_v2_speculator_smoke(model: str, speculative_config: dict):
    """Acceptance length above 1 means drafts were verified, not just proposed."""
    llm = _make_llm(model=model, speculative_config=speculative_config)
    try:
        assert llm.llm_engine.vllm_config.use_v2_model_runner, (
            f"{speculative_config['method']} fell back to the V1 runner; "
            "this test is meant to cover the V2 path"
        )
        llm.generate(SMOKE_PROMPTS, SMOKE_SP)
        acceptance = compute_acceptance_len(llm.get_metrics())
        assert acceptance > 1, (
            f"no draft tokens accepted (acceptance length {acceptance:.3f})"
        )
    finally:
        del llm
