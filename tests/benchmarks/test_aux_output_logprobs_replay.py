# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Performance benchmark for auxiliary logprob replay."""

import time

import pytest

from vllm import LLM, SamplingParams
from vllm.platforms import current_platform

pytestmark = [
    pytest.mark.benchmark,
    pytest.mark.skipif(
        not current_platform.is_cuda_alike(),
        reason="auxiliary output benchmark requires a CUDA-like worker",
    ),
    pytest.mark.skip_global_cleanup,
]

MODEL = "facebook/opt-125m"
TOP_K = 5
MAX_TOKENS = 16
PROMPTS = [
    "Explain why deterministic replay is useful in reinforcement learning.",
    "List three properties of a reliable token logprob cache.",
]
WARM_PROMPT = "A shared prefix for replay performance measurements. " * 16


def _params(*, prompt_logprobs: int | None, replay: bool = True) -> SamplingParams:
    return SamplingParams(
        temperature=0,
        max_tokens=MAX_TOKENS,
        logprobs=TOP_K,
        prompt_logprobs=prompt_logprobs,
        extra_args={"aux_output_replay": True} if replay else None,
    )


def _make_llm(*, replay: bool = True):
    config = dict(
        model=MODEL,
        dtype="float16",
        max_model_len=512,
        enforce_eager=True,
        enable_chunked_prefill=True,
        max_num_batched_tokens=64,
        enable_prefix_caching=True,
    )
    if replay:
        config["aux_output_config"] = {
            "enable_logprobs_replay": True,
            "enable_prompt_logprobs_replay": True,
        }
    else:
        config["enable_prefix_caching"] = False
    return LLM(**config)


@pytest.mark.parametrize("prompt_logprobs", [None, TOP_K])
def test_aux_output_logprobs_replay_latency(prompt_logprobs: int | None):
    """Report cold and warm replay latency without imposing a device threshold."""
    baseline_llm = _make_llm(replay=False)
    baseline_params = _params(prompt_logprobs=prompt_logprobs, replay=False)
    start = time.perf_counter()
    baseline_outputs = baseline_llm.generate(PROMPTS, baseline_params)
    baseline_seconds = time.perf_counter() - start
    baseline_tokens = sum(
        len(output.outputs[0].token_ids) for output in baseline_outputs
    )
    assert baseline_tokens > 0
    del baseline_llm

    llm = _make_llm()
    params = _params(prompt_logprobs=prompt_logprobs)

    start = time.perf_counter()
    cold_outputs = llm.generate(PROMPTS, params)
    cold_seconds = time.perf_counter() - start

    llm.generate([WARM_PROMPT], params)
    start = time.perf_counter()
    warm_outputs = llm.generate([WARM_PROMPT], params)
    warm_seconds = time.perf_counter() - start

    assert len(cold_outputs) == len(PROMPTS)
    assert warm_outputs[0].num_cached_tokens > 0
    cold_tokens = sum(len(output.outputs[0].token_ids) for output in cold_outputs)
    warm_tokens = len(warm_outputs[0].outputs[0].token_ids)
    assert cold_tokens > 0 and warm_tokens > 0

    print(
        "aux_output_logprobs_replay "
        f"prompt_logprobs={prompt_logprobs is not None} "
        f"baseline_ms_per_request="
        f"{baseline_seconds * 1000 / len(baseline_outputs):.2f} "
        f"baseline_ms_per_token={baseline_seconds * 1000 / baseline_tokens:.2f} "
        f"cold_ms_per_request="
        f"{cold_seconds * 1000 / len(cold_outputs):.2f} "
        f"cold_ms_per_token={cold_seconds * 1000 / cold_tokens:.2f} "
        f"warm_ms_per_request={warm_seconds * 1000:.2f} "
        f"warm_ms_per_token={warm_seconds * 1000 / warm_tokens:.2f} "
        f"warm_cached_tokens={warm_outputs[0].num_cached_tokens}"
    )
    del llm
