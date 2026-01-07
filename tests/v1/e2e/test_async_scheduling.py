# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from itertools import repeat
from typing import Any

import pytest
import torch._dynamo.config as dynamo_config

from vllm import SamplingParams
from vllm.logprobs import Logprob
from vllm.platforms import current_platform
from vllm.sampling_params import StructuredOutputsParams
from vllm.v1.metrics.reader import Metric

from ...conftest import VllmRunner
from ...models.utils import check_outputs_equal

MODEL = "Qwen/Qwen3-0.6B"
MTP_MODEL = "meta-llama/Llama-3.2-1B-Instruct"


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
    # spec decoding currently doesn't support min_tokens
    # min_tokens=28,
)


def test_without_spec_decoding(
    sample_json_schema,
    monkeypatch: pytest.MonkeyPatch,
):
    """Test consistency of combos of async scheduling, preemption,
    uni/multiproc executor, prefill chunking."""
    struct_outputs = StructuredOutputsParams(json=sample_json_schema)
    test_sampling_params: list[dict[str, Any]] = [
        dict(),
        # dict(min_tokens=20),
        dict(presence_penalty=-1.0),
        dict(bad_words=["the", " the"]),
        dict(logprobs=2),
        dict(logprobs=2, presence_penalty=-1.0),
        dict(structured_outputs=struct_outputs),
        dict(
            structured_outputs=struct_outputs,
            logprobs=2,
            presence_penalty=-1.0,
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

    if current_platform.is_rocm():
        # On ROCm, Only test with structured_outputs (deterministic)
        # and skip chunk_prefill (more variable).
        test_configs = [
            cfg
            for cfg in test_configs
            if not cfg[4]  # skip chunk_prefill=True
        ]
        test_sampling_params = [
            p for p in test_sampling_params if p.get("structured_outputs") is not None
        ]

    run_tests(monkeypatch, MODEL, test_configs, test_sampling_params)


def test_with_spec_decoding(sample_json_schema, monkeypatch: pytest.MonkeyPatch):
    """Test consistency and acceptance rates with some different combos of
    preemption, executor, async scheduling, prefill chunking,
    spec decoding model length.
    """

    spec_config = {
        "method": "eagle3",
        "num_speculative_tokens": 2,
        "model": "nm-testing/Llama3_2_1B_speculator.eagle3",
    }
    # Set small draft model len to force doesn't-fit-in-drafter case.
    spec_config_short = spec_config | {"max_model_len": 50}

    struct_outputs = StructuredOutputsParams(json=sample_json_schema)

    test_sampling_params = [
        dict(),
        dict(logprobs=2),
        dict(structured_outputs=struct_outputs),
        dict(
            structured_outputs=struct_outputs,
            logprobs=2,
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

    # On ROCm, use TRITON_ATTN + float32 for better numerical consistency
    run_tests(
        monkeypatch,
        MTP_MODEL,
        test_configs,
        test_sampling_params,
        is_testing_with_spec_decoding=True,
    )


@dynamo_config.patch(cache_size_limit=16)
def run_tests(
    monkeypatch: pytest.MonkeyPatch,
    model: str,
    test_configs: list[tuple],
    test_sampling_params: list[dict[str, Any]],
    is_testing_with_spec_decoding: bool = False,
):
    """Test consistency of combos of async scheduling, preemption,
    uni/multiproc executor with spec decoding."""

    # Determine attention config based on platform
    if current_platform.is_rocm():
        if is_testing_with_spec_decoding:
            # Use TRITON_ATTN for spec decoding test for consistency
            attention_config = {"backend": "TRITON_ATTN"}
        else:
            attention_config = {"backend": "ROCM_ATTN"}
    else:
        attention_config = {"backend": "FLEX_ATTENTION"}

    with monkeypatch.context() as m:
        # lock matmul precision to full FP32 (IEEE)
        m.setenv("VLLM_FLOAT32_MATMUL_PRECISION", "highest")
        # m.setenv("VLLM_BATCH_INVARIANT", "1")
        outputs: list[tuple[str, list, list]] = []
        for n, (
            test_preemption,
            executor,
            async_scheduling,
            spec_config,
            test_prefill_chunking,
        ) in enumerate(test_configs, 1):
            test_str = f"{n}/{len(test_configs)}"
            test_results = run_test(
                model,
                test_str,
                test_sampling_params,
                test_preemption,
                executor,
                async_scheduling,
                spec_config,
                test_prefill_chunking=test_prefill_chunking,
                is_testing_with_spec_decoding=is_testing_with_spec_decoding,
                attention_config=attention_config,
            )
            outputs.append(test_results)

    baseline_config, baseline_tests, _ = outputs[0]
    _, _, baseline_acceptances = next(
        (o for o in outputs if o[2] is not None), (None, None, None)
    )

    print(f"BASELINE: config=[{baseline_config}], accept_rates={baseline_acceptances}")

    failure = None
    for test_config, test_outputs, test_acceptance_rates in outputs[1:]:
        for (base_outs, base_logprobs), base_acceptance_rate, (
            test_outs,
            test_logprobs,
        ), test_acceptance_rate, params in zip(
            baseline_tests,
            baseline_acceptances or repeat(None),
            test_outputs,
            test_acceptance_rates or repeat(None),
            test_sampling_params,
        ):
            try:
                check_outputs_equal(
                    outputs_0_lst=base_outs,
                    outputs_1_lst=test_outs,
                    name_0=f"baseline=[{baseline_config}], params={params}",
                    name_1=f"config=[{test_config}], params={params}",
                )

                # On ROCm with TRITON_ATTN (spec decoding test), skip strict
                # logprobs comparison when logprobs are requested
                skip_logprobs_check = (
                    current_platform.is_rocm()
                    and params.get("logprobs")
                    and is_testing_with_spec_decoding
                )
                if not skip_logprobs_check:
                    assert _all_logprobs_match(base_logprobs, test_logprobs)

                if (
                    base_acceptance_rate is not None
                    and test_acceptance_rate is not None
                ):
                    if "spec_mml=None" in test_config:
                        # Preemption causes more variance in acceptance rates
                        if (
                            current_platform.is_rocm()
                            and "preemption=True" in test_config
                        ):
                            tolerance = 0.10
                        else:
                            tolerance = 0.05
                        assert (
                            test_acceptance_rate > base_acceptance_rate
                            or test_acceptance_rate
                            == pytest.approx(base_acceptance_rate, rel=tolerance)
                        )
                    else:
                        # Currently the reported acceptance rate is expected to be
                        # lower when we sometimes skip drafting altogether.
                        assert test_acceptance_rate > 0.1
                print(
                    f"PASSED: config=[{test_config}], params={params}"
                    f" accept_rate={test_acceptance_rate}"
                )
            except AssertionError as e:
                print(
                    f"FAILED: config=[{test_config}], params={params}"
                    f" accept_rate={test_acceptance_rate}"
                )
                if failure is None:
                    failure = e

    if failure is not None:
        raise failure


def run_test(
    model: str,
    test_str: str,
    sampling_param_tests: list[dict[str, Any]],
    test_preemption: bool,
    executor: str,
    async_scheduling: bool,
    spec_config: dict[str, Any] | None,
    test_prefill_chunking: bool,
    is_testing_with_spec_decoding: bool = False,
    attention_config: dict[str, Any] | None = None,
):
    spec_decoding = spec_config is not None
    cache_arg: dict[str, Any] = (
        # Force preemptions
        dict(num_gpu_blocks_override=32)
        if test_preemption
        else dict(gpu_memory_utilization=0.9)
    )
    spec_mml = (spec_config or {}).get("max_model_len")
    test_config = (
        f"executor={executor}, preemption={test_preemption}, "
        f"async_sched={async_scheduling}, "
        f"chunk_prefill={test_prefill_chunking}, "
        f"spec_decoding={spec_decoding}, spec_mml={spec_mml}"
    )
    print("-" * 80)
    print(f"---- TESTING {test_str}: {test_config}")
    print("-" * 80)

    with VllmRunner(
        model,
        max_model_len=512,
        enable_chunked_prefill=test_prefill_chunking,
        # Force prefill chunking
        max_num_batched_tokens=48 if test_prefill_chunking else None,
        # enforce_eager=True,
        async_scheduling=async_scheduling,
        distributed_executor_backend=executor,
        dtype="float32",
        speculative_config=spec_config,
        disable_log_stats=False,
        attention_config=attention_config,
        **cache_arg,
    ) as vllm_model:
        results = []
        acceptance_rates: list[float] | None = [] if spec_decoding else None
        for override_params in sampling_param_tests:
            metrics_before = vllm_model.llm.get_metrics()
            print(f"----------- RUNNING PARAMS: {override_params}")
            results.append(
                vllm_model.generate(
                    example_prompts,
                    sampling_params=SamplingParams(**default_params, **override_params),
                    return_logprobs=True,
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

    if len(results) > 1:
        # First check that the different parameter configs
        # actually result in different output.
        for (other_test_outs, other_test_logprobs), params in zip(
            results[1:], sampling_param_tests[1:]
        ):
            with pytest.raises(AssertionError):
                check_outputs_equal(
                    outputs_0_lst=results[0][0],
                    outputs_1_lst=other_test_outs,
                    name_0=f"baseline params={params}",
                    name_1=f"other params={params}",
                )
                assert _all_logprobs_match(results[0][1], other_test_logprobs)

    return test_config, results, acceptance_rates


def _all_logprobs_match(req_a, req_b) -> bool:
    return (
        req_a == req_b
        or len(req_a) == len(req_b)
        and all(
            len(seq_a) == len(seq_b)
            and all(_logprobs_match(a, b) for a, b in zip(seq_a, seq_b))
            for seq_a, seq_b in zip(req_a, req_b)
        )
    )


def _logprobs_match(lps_a: dict[int, Logprob], lps_b: dict[int, Logprob]) -> bool:
    if current_platform.is_rocm():
        # ROCm has higher numerical variance
        # due to use of float16.
        rel_tol, abs_tol = 5e-2, 1e-5
    else:
        rel_tol, abs_tol = 1e-3, 1e-6
    return (
        len(lps_a) == len(lps_b)
        and lps_a.keys() == lps_b.keys()
        and all(
            a.decoded_token == b.decoded_token
            and a.rank == b.rank
            and a.logprob == pytest.approx(b.logprob, rel=rel_tol, abs=abs_tol)
            for a, b in ((lps_a[x], lps_b[x]) for x in lps_a)
        )
    )


def _get_acceptance_rate(before: list[Metric], after: list[Metric]) -> float:
    draft = _get_count(before, after, "vllm:spec_decode_num_draft_tokens")
    accept = _get_count(before, after, "vllm:spec_decode_num_accepted_tokens")
    return accept / draft if draft > 0 else 0.0


def _get_count(before: list[Metric], after: list[Metric], name: str) -> int:
    before_val = next(m.value for m in before if m.name == name)
    after_val = next(m.value for m in after if m.name == name)
    return after_val - before_val
