# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
from dataclasses import replace
from typing import Any

import pytest
import torch

from tests.conftest import HfRunner, VllmRunner
from tests.utils import (
    large_gpu_mark,
    single_gpu_only,
)
from tests.v1.e2e.general.async_scheduling_utils import (
    AccuracyTolerance,
    check_accuracy_budget,
    check_greedy_token,
    check_logprobs,
    check_request_accuracy,
    check_stopping,
)
from vllm import SamplingParams
from vllm.logprobs import Logprob
from vllm.platforms import current_platform
from vllm.sampling_params import StructuredOutputsParams
from vllm.v1.metrics.reader import Metric

MODEL = "Qwen/Qwen3-0.6B"
MTP_MODEL = "meta-llama/Llama-3.2-1B-Instruct"

# Need to enforce eager for MRV2 while we sort out cudagraph issues.
ENFORCE_EAGER = os.getenv("ENFORCE_EAGER", "0") == "1"

first_prompt = (
    "The following numbers of the sequence "
    + ", ".join(str(i) for i in range(10))
    + " are:"
)
example_prompts = [first_prompt, "In one word, the capital of France is "] + [
    f"Tell me about the number {i}: " for i in range(32)
]

default_params = dict(
    temperature=0.0,  # greedy
    max_tokens=30,
    min_tokens=28,
)


@single_gpu_only
def test_without_spec_decoding(
    vllm_runner: type[VllmRunner],
    hf_runner: type[HfRunner],
    sample_json_schema,
):
    """Check target-model accuracy across scheduling and executor configurations."""
    struct_outputs = StructuredOutputsParams(json=sample_json_schema)
    test_sampling_params: list[dict[str, Any]] = [
        dict(),
        # dict(min_tokens=20),
        dict(frequency_penalty=-1.0),
        dict(bad_words=["the", " the"]),
        dict(logprobs=2),
        dict(logprobs=2, frequency_penalty=-1.0),
        dict(prompt_logprobs=2),
        dict(prompt_logprobs=2, logprobs=2),
        dict(structured_outputs=struct_outputs),
        dict(
            structured_outputs=struct_outputs,
            logprobs=2,
        ),
        dict(
            structured_outputs=struct_outputs,
            frequency_penalty=-1.0,
        ),
        dict(
            structured_outputs=struct_outputs,
            logprobs=2,
            frequency_penalty=-1.0,
        ),
    ]

    # test_preemption, executor, async_scheduling,
    # spec_config, test_prefill_chunking
    test_configs = [
        (False, "mp", False, None, False),
        (True, "mp", False, None, True),
        (False, "mp", True, None, False),
        (False, "uni", True, None, False),
        (True, "mp", True, None, False),
        (True, "uni", True, None, False),
        (False, "mp", True, None, True),
        (True, "mp", True, None, True),
        (True, "uni", True, None, True),
    ]

    run_tests(vllm_runner, hf_runner, MODEL, test_configs, test_sampling_params)


@single_gpu_only
@large_gpu_mark(min_gb=16)
def test_with_eagle3_spec_decoding(
    vllm_runner: type[VllmRunner],
    hf_runner: type[HfRunner],
    sample_json_schema,
):
    """Check accuracy and acceptance, including drafts that exceed model length."""

    spec_config = {
        "method": "eagle3",
        "num_speculative_tokens": 2,
        "model": "nm-testing/Llama3_2_1B_speculator.eagle3",
    }
    # Set small draft model len to force doesn't-fit-in-drafter case.
    spec_config_short = spec_config | {"max_model_len": 50}

    struct_outputs = StructuredOutputsParams(json=sample_json_schema)

    test_sampling_params: list[dict[str, Any]] = [
        dict(),
        dict(frequency_penalty=-1.0),
        dict(bad_words=["the", " the"]),
        dict(logprobs=2),
        dict(logprobs=2, frequency_penalty=-1.0),
        dict(prompt_logprobs=2),
        dict(prompt_logprobs=2, logprobs=2),
        dict(structured_outputs=struct_outputs),
        dict(
            structured_outputs=struct_outputs,
            logprobs=2,
            frequency_penalty=-1.0,
        ),
    ]

    # test_preemption, executor, async_scheduling,
    # spec_config, test_prefill_chunking
    test_configs = [
        (False, "mp", False, None, False),
        (False, "mp", False, spec_config, False),
        (True, "mp", False, spec_config, True),
        (True, "uni", False, spec_config_short, True),
        (False, "mp", True, spec_config, False),
        (True, "mp", True, spec_config, False),
        (False, "mp", True, spec_config_short, True),
        (True, "uni", True, spec_config, False),
        (True, "uni", True, spec_config_short, False),
        (True, "mp", True, spec_config, True),
        (True, "uni", True, spec_config_short, True),
    ]

    run_tests(vllm_runner, hf_runner, MTP_MODEL, test_configs, test_sampling_params)


@pytest.mark.skipif(
    current_platform.is_xpu(),
    reason=("XPU matmul/attention kernels are not batch-invariant"),
)
def test_with_ngram_gpu_spec_decoding(
    vllm_runner: type[VllmRunner], hf_runner: type[HfRunner]
):
    """Check ngram accuracy and acceptance across scheduling configurations."""

    # Variant with larger speculation window
    ngram_gpu_config = {
        "method": "ngram_gpu",
        "num_speculative_tokens": 3,
        "prompt_lookup_max": 3,
        "prompt_lookup_min": 2,
    }

    # Test configurations covering various scenarios
    # test_preemption, executor, async_scheduling,
    # spec_config, test_prefill_chunking
    test_configs = [
        (False, "mp", False, None, False),
        (False, "mp", False, ngram_gpu_config, False),
        (True, "mp", False, ngram_gpu_config, True),
        (False, "mp", True, ngram_gpu_config, False),
        (True, "mp", True, ngram_gpu_config, False),
        (True, "uni", True, ngram_gpu_config, False),
        (True, "mp", True, ngram_gpu_config, True),
    ]

    # Use MODEL (Qwen) for ngram_gpu tests as it's lighter weight
    # and ngram_gpu doesn't require a specific draft model
    run_tests(vllm_runner, hf_runner, MODEL, test_configs, [{}])


def run_tests(
    vllm_runner: type[VllmRunner],
    hf_runner: type[HfRunner],
    model: str,
    test_configs: list[tuple],
    test_sampling_params: list[dict[str, Any]],
):
    """Validate every original batch against an independent target model."""
    outputs = []
    for n, config in enumerate(test_configs, 1):
        outputs.append(
            run_test(
                vllm_runner,
                model,
                f"{n}/{len(test_configs)}",
                test_sampling_params,
                *config,
            )
        )

    # Natural BF16 continuations can diverge at near ties, even between two
    # synchronous calls. Score each actual history instead of comparing text
    # generated from different histories. Neither engine forces an attention
    # backend, model dtype, or matmul precision.
    # Qwen/Llama BF16 controls and Qwen holdouts measured max score error <0.38,
    # per-stream request mean <0.073, max greedy gap 0.125, and request sum 0.25.
    # Per-request budgets also reject systematic errors below the scalar bounds.
    tolerance = AccuracyTolerance(logprob_atol=0.5, greedy_atol=0.25)
    with hf_runner(model) as hf:
        hf.model.eval()
        for config, batches, _ in outputs:
            assert len(batches) == len(test_sampling_params)
            for batch, overrides in zip(batches, test_sampling_params, strict=True):
                assert len(batch) == len(example_prompts), config
                params = SamplingParams(**default_params, **overrides)
                for i, (request, prompt) in enumerate(
                    zip(batch, example_prompts, strict=True)
                ):
                    assert request.prompt_token_ids == hf.tokenizer.encode(prompt)
                    check_request_accuracy(
                        hf,
                        request,
                        params,
                        tolerance,
                        f"config=[{config}], params={overrides}, request={i}",
                    )
                print(f"ACCURACY PASSED: config=[{config}], params={overrides}")

    baseline_acceptances = next((o[2] for o in outputs if o[2] is not None), None)
    if baseline_acceptances is not None:
        for config, _, acceptances in outputs:
            if acceptances is None:
                continue
            for baseline, actual, params in zip(
                baseline_acceptances, acceptances, test_sampling_params, strict=True
            ):
                context = f"config=[{config}], params={params}"
                if "spec_mml=None" in config:
                    # Keep the original acceptance-quality floor per batch.
                    relative_drop = (
                        0.10
                        if current_platform.is_rocm() and "preemption=True" in config
                        else 0.05
                    )
                    assert actual >= baseline * (1 - relative_drop), (
                        f"{context}: acceptance={actual}, baseline={baseline}"
                    )
                else:
                    assert actual > 0.1, f"{context}: acceptance={actual}"


def run_test(
    vllm_runner: type[VllmRunner],
    model: str,
    test_str: str,
    sampling_param_tests: list[dict[str, Any]],
    test_preemption: bool,
    executor: str,
    async_scheduling: bool,
    spec_config: dict[str, Any] | None,
    test_prefill_chunking: bool,
):
    spec_decoding = spec_config is not None
    spec_method = (spec_config or {}).get("method", "none")
    # Chunked ngram decoding admits fewer concurrent requests under the
    # 48-token budget. Its original 33-block cache did not preempt on ROCm.
    # A 17-block cache forces contention while every original request fits
    # comfortably within 256 tokens (the longest prompt plus output is 67).
    cache_blocks = 17 if test_prefill_chunking and spec_method == "ngram_gpu" else 33
    cache_arg: dict[str, Any] = (
        dict(
            num_gpu_blocks_override=cache_blocks,
            max_model_len=(cache_blocks - 1) * 16,
        )
        if test_preemption
        else dict(gpu_memory_utilization=0.9, max_model_len=4096)
    )
    spec_mml = (spec_config or {}).get("max_model_len")
    test_config = (
        f"executor={executor}, preemption={test_preemption}, "
        f"async_sched={async_scheduling}, "
        f"chunk_prefill={test_prefill_chunking}, "
        f"spec_decoding={spec_decoding}, spec_method={spec_method}, spec_mml={spec_mml}"
    )
    print("-" * 80)
    print(f"---- TESTING {test_str}: {test_config}")
    print("-" * 80)

    with vllm_runner(
        model,
        enable_chunked_prefill=test_prefill_chunking,
        # Force prefill chunking
        max_num_batched_tokens=48 if test_prefill_chunking else None,
        enforce_eager=ENFORCE_EAGER,
        async_scheduling=async_scheduling,
        distributed_executor_backend=executor,
        speculative_config=spec_config,
        disable_log_stats=False,
        enable_prefix_caching=False if current_platform.is_rocm() else None,
        **cache_arg,
    ) as vllm_model:
        results = []
        acceptance_rates: list[float] | None = [] if spec_decoding else None
        for override_params in sampling_param_tests:
            metrics_before = vllm_model.llm.get_metrics()
            print(f"----------- RUNNING PARAMS: {override_params}")
            results.append(
                vllm_model.llm.generate(
                    vllm_model.get_inputs(example_prompts),
                    sampling_params=SamplingParams(**default_params, **override_params),
                )
            )
            metrics_after = vllm_model.llm.get_metrics()
            if acceptance_rates is not None:
                acceptance_rate = _get_acceptance_rate(metrics_before, metrics_after)
                acceptance_rates.append(acceptance_rate)
                print(f"ACCEPTANCE RATE {acceptance_rate}")

            if test_preemption:
                preemptions = _get_count(
                    metrics_before, metrics_after, "vllm:num_preemptions"
                )
                assert preemptions > 0, "preemption test had no preemptions"

    # Preserve the original parameter-effect checks across repeated batches.
    if len(results) > 1:
        baseline = _result_signature(results[0])
        for outputs, params in zip(results[1:], sampling_param_tests[1:], strict=True):
            assert _result_signature(outputs) != baseline, (
                f"{test_config}: sampling parameters had no observable effect: {params}"
            )
        # The old ROCm filter used schema-only as its first parameter set.
        # Retain those comparisons after restoring the ordinary cases too.
        schema_batches = [
            batch
            for batch, params in zip(results, sampling_param_tests, strict=True)
            if params.get("structured_outputs") is not None
        ]
        for batch in schema_batches[1:]:
            assert _result_signature(batch) != _result_signature(schema_batches[0]), (
                f"{test_config}: structured sampling parameters had no effect"
            )

    return test_config, results, acceptance_rates


def _result_signature(requests):
    return [
        (
            list(request.outputs[0].token_ids),
            request.outputs[0].text,
            request.prompt_logprobs,
            request.outputs[0].logprobs,
        )
        for request in requests
    ]


def _get_acceptance_rate(before: list[Metric], after: list[Metric]) -> float:
    draft = _get_count(before, after, "vllm:spec_decode_num_draft_tokens")
    accept = _get_count(before, after, "vllm:spec_decode_num_accepted_tokens")
    assert draft > 0, "speculative test did not draft any tokens"
    assert 0 <= accept <= draft, f"invalid acceptance counters: {accept=}, {draft=}"
    return accept / draft


def _get_count(before: list[Metric], after: list[Metric], name: str) -> int:
    before_val = next(m.value for m in before if m.name == name)
    after_val = next(m.value for m in after if m.name == name)
    return after_val - before_val


@pytest.mark.parametrize(
    "corruption", ["score", "rank", "top_k", "missing", "nan", "positive", "order"]
)
def test_accuracy_checks_reject_incorrect_logprobs(corruption):
    second = 2.875 if corruption == "order" else 1.0
    reference = torch.tensor([3.0, second, -2.0, -5.0]).log_softmax(-1)
    tolerance = AccuracyTolerance(logprob_atol=0.5, greedy_atol=0.25)
    scores = {
        0: Logprob(logprob=float(reference[0]), rank=1, decoded_token="a"),
        1: Logprob(logprob=float(reference[1]), rank=2, decoded_token="b"),
    }
    check_logprobs(scores, reference, 0, 2, tolerance, "control")
    if corruption == "score":
        scores[0] = replace(scores[0], logprob=scores[0].logprob - 1.0)
    elif corruption == "rank":
        scores[1] = replace(scores[1], rank=1)
    elif corruption == "top_k":
        del scores[1]
        scores[3] = Logprob(logprob=float(reference[3]), rank=2, decoded_token="d")
    elif corruption == "missing":
        del scores[0]
    elif corruption == "positive":
        scores[0] = replace(scores[0], logprob=0.1)
    elif corruption == "order":
        scores[0] = replace(scores[0], rank=2)
        scores[1] = replace(scores[1], rank=1)
    else:
        scores[0] = replace(scores[0], logprob=float("nan"))
    with pytest.raises(AssertionError):
        check_logprobs(scores, reference, 0, 2, tolerance, corruption)


def test_accuracy_checks_distinguish_ties_from_wrong_tokens():
    tolerance = AccuracyTolerance(logprob_atol=0.5, greedy_atol=0.25)
    logits = torch.tensor([3.0, 2.875, -2.0, -torch.inf])
    check_greedy_token(logits, 0, tolerance, "best")
    check_greedy_token(logits, 1, tolerance, "near tie")
    for token in (2, 3):
        with pytest.raises(AssertionError):
            check_greedy_token(logits, token, tolerance, "incorrect token")

    # top-k may omit a sampled token tied at its boundary. The sampled rank
    # can duplicate a sequential top-k rank without corrupting the result.
    reference = torch.zeros(4).log_softmax(-1)
    scores = {
        token: Logprob(logprob=float(reference[token]), rank=rank)
        for token, rank in [(3, 1), (0, 1), (1, 2)]
    }
    check_logprobs(scores, reference, 3, 2, tolerance, "tied top-k")

    # A grammar with one allowed token still returns the requested top-k,
    # including masked entries whose score is exactly negative infinity.
    reference = torch.tensor([0.0, -torch.inf, -torch.inf])
    scores = {0: Logprob(0.0, 1), 1: Logprob(-float("inf"), 2)}
    check_logprobs(scores, reference, 0, 2, tolerance, "masked top-k")


def test_accuracy_checks_reject_continuing_after_eos():
    params = SamplingParams(max_tokens=3)
    stop_ids = {2}
    check_stopping([0, 1, 2], "stop", stop_ids, params, "EOS at length cap")
    check_stopping([0, 1, 0], "length", stop_ids, params, "length cap")
    for tokens, reason in [([0, 2, 1], "length"), ([0, 1, 2], "length")]:
        with pytest.raises(AssertionError):
            check_stopping(tokens, reason, stop_ids, params, "incorrect stopping")


def test_accuracy_checks_reject_systematic_sub_bound_errors():
    tolerance = AccuracyTolerance(logprob_atol=0.5, greedy_atol=0.25)
    reference = torch.tensor([3.0, 2.75, -2.0]).log_softmax(-1)
    scores = {i: Logprob(float(reference[i]) - 0.4, i + 1) for i in (0, 1)}
    # Each corrupted score fits the scalar bound, but the bias must not pass.
    errors = check_logprobs(scores, reference, 0, 2, tolerance, "biased scores")
    with pytest.raises(AssertionError, match="mean logprob error"):
        check_accuracy_budget([errors], [], tolerance, "biased scores")

    gap = check_greedy_token(reference, 1, tolerance, "one ambiguous choice")
    with pytest.raises(AssertionError, match="total greedy logit gap"):
        check_accuracy_budget([], [gap] * 30, tolerance, "systematically worse choices")
    check_accuracy_budget(
        [torch.tensor([0.03, 0.07])], [0.125, 0.125], tolerance, "control variation"
    )
