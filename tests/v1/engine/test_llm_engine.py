# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

import random
from typing import TYPE_CHECKING, Optional

import pytest

from vllm import LLM, EngineArgs, LLMEngine
from vllm.sampling_params import (GuidedDecodingParams, RequestOutputKind,
                                  SamplingParams)
from vllm.v1.metrics.reader import Counter, Gauge, Histogram, Metric, Vector

if TYPE_CHECKING:
    from tests.conftest import VllmRunner

MODEL = "facebook/opt-125m"
DTYPE = "half"


def _vllm_model(
    apc: bool,
    vllm_runner: type[VllmRunner],
    monkeypatch: pytest.MonkeyPatch,
    *,
    skip_tokenizer_init: bool = False,
):
    """Set up VllmRunner instance."""
    monkeypatch.setenv("VLLM_USE_V1", "1")
    return vllm_runner(
        MODEL,
        dtype=DTYPE,
        max_model_len=128,
        enforce_eager=True,
        enable_prefix_caching=apc,
        gpu_memory_utilization=0.5,
        skip_tokenizer_init=skip_tokenizer_init,
    )


@pytest.fixture(
    # Function scope decouples tests & allows
    # env var adjustment via monkeypatch
    scope="function",
    # Prefix caching
    params=[False, True])
def vllm_model(vllm_runner, request, monkeypatch):
    """VllmRunner test fixture parameterized by APC True/False."""
    with _vllm_model(request.param, vllm_runner, monkeypatch) as vllm_model:
        yield vllm_model


@pytest.fixture(scope="function")
def vllm_model_apc(vllm_runner, monkeypatch):
    """VllmRunner test fixture with APC."""
    with _vllm_model(True, vllm_runner, monkeypatch) as vllm_model:
        yield vllm_model


@pytest.fixture(
    # Function scope decouples tests & allows
    # env var adjustment via monkeypatch
    scope="function",
    # Prefix caching
    params=[False, True])
def vllm_model_skip_tokenizer_init(vllm_runner, request, monkeypatch):
    """VllmRunner test fixture with APC."""
    with _vllm_model(
            request.param,
            vllm_runner,
            monkeypatch,
            skip_tokenizer_init=True,
    ) as vllm_model:
        yield vllm_model


def _get_test_sampling_params(
    prompt_list: list[str],
    seed: Optional[int] = 42,
    structured_outputs: bool = False,
) -> tuple[list[SamplingParams], list[int]]:
    """Generate random sampling params for a batch."""

    def get_mostly_n_gt1() -> int:
        r"""Mostly n \in [2,20], ~1/3 n=1"""
        x = random.randint(0, 28)
        if x < 10:
            return 1
        else:
            return x - 8

    n_list = [get_mostly_n_gt1() for _ in range(len(prompt_list))]
    # High temperature to maximize the chance of unique completions
    return [
        SamplingParams(
            temperature=0.95,
            top_p=0.95,
            n=n,
            seed=seed,
            guided_decoding=GuidedDecodingParams(
                regex="[0-9]+") if structured_outputs else None,
        ) for n in n_list
    ], n_list


def test_compatibility_with_skip_tokenizer_init(
    vllm_model_skip_tokenizer_init: VllmRunner,
    example_prompts: list[str],
):
    # Case 1: Structured output request should raise an error.
    sampling_params_list, _ = _get_test_sampling_params(
        example_prompts,
        structured_outputs=True,
    )
    llm: LLM = vllm_model_skip_tokenizer_init.llm
    with pytest.raises(ValueError):
        _ = llm.generate(example_prompts, sampling_params_list)


def test_parallel_sampling(vllm_model, example_prompts) -> None:
    """Test passes if parallel sampling `n>1` yields `n` unique completions.

    Args:
      vllm_model: VllmRunner instance under test.
      example_prompt: test fixture providing prompts for testing.
    """
    sampling_params_list, n_list = _get_test_sampling_params(example_prompts)
    llm: LLM = vllm_model.llm
    outputs = llm.generate(example_prompts, sampling_params_list)

    # Validate each request response
    for out, n in zip(outputs, n_list):
        completion_counts: dict[str, int] = {}
        # Assert correct number of completions
        assert len(out.outputs) == n, (
            f"{len(out.outputs)} completions; {n} expected.")
        for idx in range(n):
            comp = out.outputs[idx]
            # Assert correct completion indices
            assert comp.index == idx, (f"Index {comp.index}; expected {idx}.")
            text = comp.text
            completion_counts[text] = completion_counts.get(text, 0) + 1
        # Assert unique completions
        if len(completion_counts) != n:
            repeats = {
                txt: num
                for (txt, num) in completion_counts.items() if num > 1
            }
            raise AssertionError(
                f"{len(completion_counts)} unique completions; expected"
                f" {n}. Repeats: {repeats}")


@pytest.mark.parametrize("model", ["Qwen/Qwen3-0.6B"])
@pytest.mark.parametrize("num_index", [2, 5])
@pytest.mark.parametrize("output_kind", [
    None, RequestOutputKind.CUMULATIVE, RequestOutputKind.DELTA,
    RequestOutputKind.FINAL_ONLY
])
def test_llmengine_streaming_with_parallel_sampling(model, output_kind,
                                                    num_index) -> None:
    """Test output_kind in LLMEngine when parallel sampling (index) `n>1`.
    """
    engine_args = EngineArgs(model=model, gpu_memory_utilization=0.5)
    engine = LLMEngine.from_engine_args(engine_args)

    NUM_REQUESTS = 10
    NUM_TOKENS = 20

    sampling_params = SamplingParams(
        max_tokens=NUM_TOKENS,
        min_tokens=NUM_TOKENS,
        n=num_index,
        **({
            "output_kind": output_kind
        } if output_kind is not None else {}),
        temperature=0.9,
    )

    if output_kind == None:
        assert sampling_params.output_kind == RequestOutputKind.CUMULATIVE, "default is CUMULATIVE"

    prompts = ["The history of the Earth is" for i in range(NUM_REQUESTS)]
    request_ids = []
    from collections import defaultdict
    partial_tokens: defaultdict[
        str, defaultdict[int, list[int]]
    ] = defaultdict(lambda: defaultdict(list))
    finished_requests: set[str] = set()

    for i, prompt in enumerate(prompts):
        request_id = f"req-{i}"
        request_ids.append(request_id)
        engine.add_request(request_id, prompt, sampling_params)

    while len(finished_requests) < len(request_ids):
        for request_output in engine.step():
            request_id = request_output.request_id

            # Accumulate outputs by using the `index` field
            for output in request_output.outputs:
                partial_tokens[request_id][output.index].extend(
                    output.token_ids)

            if request_output.finished:
                finished_requests.add(request_id)
                index_count = len(
                    set(comp.index for comp in request_output.outputs))
                token_count = len(request_output.outputs[0].token_ids)
                # total generated token counts per index
                token_all = [
                    len(partial_tokens[request_id][i])
                    for i in partial_tokens[request_id]
                ]

                # 1st assert: checks the last output
                # 2nd assert: checks the number of parallel sampling number
                # 3rd assert: verifies each index generated the same number of tokens in sum
                if sampling_params.output_kind == RequestOutputKind.CUMULATIVE:
                    assert index_count == 1 and token_count == NUM_TOKENS
                    assert len(partial_tokens[request_id]) == num_index
                    assert (len(set(token_all)) == 1
                            and token_all[0] == NUM_TOKENS *
                            (NUM_TOKENS + 1) // 2)
                elif sampling_params.output_kind == RequestOutputKind.DELTA:
                    assert index_count == 1 and token_count == 1
                    assert len(partial_tokens[request_id]) == num_index
                    assert len(
                        set(token_all)) == 1 and token_all[0] == NUM_TOKENS
                elif sampling_params.output_kind == RequestOutputKind.FINAL_ONLY:
                    assert index_count == num_index and token_count == NUM_TOKENS
                    assert len(partial_tokens[request_id]) == num_index
                    assert len(
                        set(token_all)) == 1 and token_all[0] == NUM_TOKENS
                else:
                    assert False, "output_kind is missing"


@pytest.mark.parametrize("num_index", [2, 5])
@pytest.mark.parametrize("output_kind", [
    None, RequestOutputKind.CUMULATIVE, RequestOutputKind.DELTA,
    RequestOutputKind.FINAL_ONLY
])
def test_llm_streaming_with_parallel_sampling(vllm_model, num_index,
                                              output_kind) -> None:
    """Test output_kind in LLM class when parallel sampling (index) `n>1`.
    """
    NUM_REQUESTS = 10
    NUM_TOKENS = 20
    prompts = ["The history of the Earth is" for i in range(NUM_REQUESTS)]

    sampling_params = SamplingParams(
        max_tokens=NUM_TOKENS,
        min_tokens=NUM_TOKENS,
        n=num_index,
        **({
            "output_kind": output_kind
        } if output_kind is not None else {}),
        temperature=0.9,
    )

    llm = vllm_model.llm
    outputs = llm.generate(prompts, sampling_params)

    # ATM output_kind is overwritten to FINAL_ONLY
    assert sampling_params.output_kind == RequestOutputKind.FINAL_ONLY, (
        f"output_kind={output_kind} overwritten to FINAL_ONLY,"
        f"got {sampling_params.output_kind} instead")

    for output in outputs:
        index_count = len(set(comp.index for comp in output.outputs))
        token_ids = output.outputs[0].token_ids

        assert index_count == num_index

        if index_count == 1:
            if len(token_ids) == 1:
                raise AssertionError("works like streaming DELTA")
            else:
                raise AssertionError("works like streaming CUMULATIVE")


def test_engine_metrics(vllm_runner, monkeypatch, example_prompts):
    max_tokens = 100
    # Use spec decoding to test num_accepted_tokens_per_pos
    speculative_config = {
        "method": "ngram",
        "prompt_lookup_max": 5,
        "prompt_lookup_min": 3,
        "num_speculative_tokens": 5,
    }
    monkeypatch.setenv("VLLM_USE_V1", "1")
    with vllm_runner(
            MODEL,
            speculative_config=speculative_config,
            disable_log_stats=False,
    ) as vllm_model:
        llm: LLM = vllm_model.llm
        sampling_params = SamplingParams(temperature=0.0,
                                         max_tokens=max_tokens)
        outputs = llm.generate(example_prompts, sampling_params)

        n_prompts = len(example_prompts)
        assert len(outputs) == n_prompts

        total_tokens = 0
        for out in outputs:
            assert len(out.outputs) == 1
            total_tokens += len(out.outputs[0].token_ids)
        assert total_tokens == max_tokens * n_prompts

        metrics = llm.get_metrics()

        def find_metric(name) -> list[Metric]:
            found = []
            for metric in metrics:
                if metric.name == name:
                    found.append(metric)
            return found

        num_requests_running = find_metric("vllm:num_requests_running")
        assert len(num_requests_running) == 1
        assert isinstance(num_requests_running[0], Gauge)
        assert num_requests_running[0].value == .0

        generation_tokens = find_metric("vllm:generation_tokens")
        assert len(generation_tokens) == 1
        assert isinstance(generation_tokens[0], Counter)
        assert generation_tokens[0].value == total_tokens

        request_generation_tokens = find_metric(
            "vllm:request_generation_tokens")
        assert len(request_generation_tokens) == 1
        assert isinstance(request_generation_tokens[0], Histogram)
        assert "+Inf" in request_generation_tokens[0].buckets
        assert request_generation_tokens[0].buckets["+Inf"] == n_prompts
        assert request_generation_tokens[0].count == n_prompts
        assert request_generation_tokens[0].sum == total_tokens

        num_accepted_tokens_per_pos = find_metric(
            "vllm:spec_decode_num_accepted_tokens_per_pos")
        assert len(num_accepted_tokens_per_pos) == 1
        assert isinstance(num_accepted_tokens_per_pos[0], Vector)
        assert len(num_accepted_tokens_per_pos[0].values) == 5


@pytest.mark.parametrize("model", ["meta-llama/Llama-3.2-1B-Instruct"])
def test_skip_tokenizer_initialization(model: str,
                                       monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("VLLM_USE_V1", "1")
    # This test checks if the flag skip_tokenizer_init skips the initialization
    # of tokenizer and detokenizer. The generated output is expected to contain
    # token ids.
    llm = LLM(
        model=model,
        skip_tokenizer_init=True,
        enforce_eager=True,
    )
    sampling_params = SamplingParams(prompt_logprobs=True, detokenize=True)

    with pytest.raises(ValueError, match="cannot pass text prompts when"):
        llm.generate("abc", sampling_params)

    outputs = llm.generate({"prompt_token_ids": [1, 2, 3]},
                           sampling_params=sampling_params)
    assert len(outputs) > 0
    completions = outputs[0].outputs
    assert len(completions) > 0
    assert completions[0].text == ""
    assert completions[0].token_ids
