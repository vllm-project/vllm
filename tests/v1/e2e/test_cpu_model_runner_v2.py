# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Model Runner V2 on CPU must match the V1 runner.

Without Triton (Arm, macOS), V2 runs its Triton kernels through the torch
implementations in `vllm.v1.worker.cpu.kernels`; each sampling feature below
routes through a different one.
"""

import os
from typing import Any

import pytest

from tests.models.utils import check_logprobs_close
from vllm import LLM, SamplingParams
from vllm.platforms import current_platform

pytestmark = pytest.mark.cpu_model

if not current_platform.is_cpu():
    pytest.skip("skipping CPU-only tests", allow_module_level=True)

# Bound the KV cache so the run does not scale with host memory.
os.environ.setdefault("VLLM_CPU_KVCACHE_SPACE", "1")

MODEL = "Qwen/Qwen3-0.6B"
PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):",
    "List three colors:",
    "1, 2, 3, 4,",
]
NUM_LOGPROBS = 5
FEATURES = {
    "greedy": {},
    "penalties": dict(
        presence_penalty=0.5, frequency_penalty=0.5, repetition_penalty=1.2
    ),
    "logit_bias": dict(logit_bias={11: -100.0, 13: 5.0}),
    "bad_words": dict(bad_words=["Paris", " return"]),
    # 12095 is " Paris", the greedy first token for the first prompt.
    "min_tokens": dict(min_tokens=4, stop_token_ids=[12095]),
    "allowed_token_ids": dict(allowed_token_ids=list(range(100, 1000))),
}


def _generate(use_v2: bool) -> dict[str, Any]:
    with pytest.MonkeyPatch.context() as m:
        m.setenv("VLLM_USE_V2_MODEL_RUNNER", str(int(use_v2)))
        llm = LLM(model=MODEL, dtype="bfloat16", max_model_len=1024)
        results: dict[str, Any] = {}
        for name, kwargs in FEATURES.items():
            params = SamplingParams(
                temperature=0, max_tokens=24, logprobs=NUM_LOGPROBS, **kwargs
            )
            results[name] = [
                (o.outputs[0].token_ids, o.outputs[0].text, o.outputs[0].logprobs)
                for o in llm.generate(PROMPTS, params)
            ]
        sampled = SamplingParams(
            temperature=0.8, top_p=0.95, min_p=0.05, max_tokens=24, seed=7
        )
        results["seeded"] = [
            [o.outputs[0].token_ids for o in llm.generate(PROMPTS, sampled)]
            for _ in range(2)
        ]
        del llm
    return results


@pytest.fixture(scope="module")
def outputs() -> tuple[dict[str, Any], dict[str, Any]]:
    return _generate(use_v2=False), _generate(use_v2=True)


@pytest.mark.parametrize("feature", list(FEATURES))
def test_v2_matches_v1(outputs, feature: str):
    v1, v2 = outputs
    check_logprobs_close(
        outputs_0_lst=v1[feature],
        outputs_1_lst=v2[feature],
        name_0="v1",
        name_1="v2",
    )


def test_v2_seeded_sampling_is_reproducible(outputs):
    first, second = outputs[1]["seeded"]
    assert first == second
