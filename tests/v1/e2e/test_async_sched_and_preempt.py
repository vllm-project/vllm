# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any

import pytest
import torch._dynamo.config as dynamo_config

from vllm import SamplingParams
from vllm.logprobs import Logprob

from ...conftest import VllmRunner
from ...models.utils import check_outputs_equal

MODEL = "Qwen/Qwen3-0.6B"


@dynamo_config.patch(cache_size_limit=16)
def test_preempt_and_async_scheduling_e2e(monkeypatch: pytest.MonkeyPatch):
    """Test consistency of combos of async scheduling, preemption,
    uni/multiproc executor, and various sampling parameters."""

    first_prompt = (
        "The following numbers of the sequence "
        + ", ".join(str(i) for i in range(10))
        + " are:"
    )
    example_prompts = [first_prompt, "In one word, the capital of France is "] + [
        f"Tell me about the number {i}: " for i in range(32)
    ]

    sampling_param_tests: list[dict[str, Any]] = [
        dict(),
        # dict(min_tokens=20),
        dict(presence_penalty=-1.0),
        dict(bad_words=["the", " the"]),
        dict(logprobs=2),
        dict(logprobs=2, presence_penalty=-1.0),
    ]

    default_params = dict(
        temperature=0.0,  # greedy
        max_tokens=20,
    )

    with monkeypatch.context() as m:
        m.setenv("VLLM_ATTENTION_BACKEND", "FLEX_ATTENTION")
        # m.setenv("VLLM_BATCH_INVARIANT", "1")

        outputs: list[tuple[str, list]] = []
        for test_preemption in [False, True]:
            for executor in ["mp", "uni"]:
                for async_scheduling in [False, True]:
                    cache_arg: dict[str, Any] = (
                        dict(num_gpu_blocks_override=32)
                        if test_preemption
                        else dict(gpu_memory_utilization=0.7)
                    )
                    test_config = (
                        f"executor={executor}, preemption={test_preemption},"
                        f" async_sched={async_scheduling}"
                    )
                    print("-" * 80)
                    print(f"---- TESTING: {test_config}")
                    print("-" * 80)
                    with VllmRunner(
                        MODEL,
                        max_model_len=512,
                        enforce_eager=True,
                        async_scheduling=async_scheduling,
                        distributed_executor_backend=executor,
                        dtype="float32",  # avoid precision errors
                        **cache_arg,
                    ) as vllm_model:
                        results = []
                        for override_params in sampling_param_tests:
                            print(f"----------- RUNNING PARAMS: {override_params}")
                            results.append(
                                vllm_model.generate(
                                    example_prompts,
                                    sampling_params=SamplingParams(
                                        **default_params, **override_params
                                    ),
                                    return_logprobs=True,
                                )
                            )

                        if not outputs:
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
                                    assert _all_logprobs_match(
                                        results[0][1], other_test_logprobs
                                    )

                        outputs.append((test_config, results))

    baseline_config, baseline_tests = outputs[0]

    for test_config, test_outputs in outputs[1:]:
        for (base_outs, base_logprobs), (test_outs, test_logprobs), params in zip(
            baseline_tests, test_outputs, sampling_param_tests
        ):
            check_outputs_equal(
                outputs_0_lst=base_outs,
                outputs_1_lst=test_outs,
                name_0=f"baseline=[{baseline_config}], params={params}",
                name_1=f"config=[{test_config}], params={params}",
            )
            assert _all_logprobs_match(base_logprobs, test_logprobs)

            print(f"PASSED: config=[{test_config}], params={params}")


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
    return len(lps_a) == len(lps_b) and all(
        a.decoded_token == b.decoded_token
        and a.rank == b.rank
        and a.logprob == pytest.approx(b.logprob, rel=1e-3, abs=1e-6)
        for a, b in ((lps_a[x], lps_b[x]) for x in lps_a)
    )
