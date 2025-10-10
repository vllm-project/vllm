# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any

import pytest

from vllm import SamplingParams

from ...conftest import VllmRunner
from ...models.utils import check_outputs_equal

MODEL = "Qwen/Qwen3-0.6B"


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
        # TODO enable these with https://github.com/vllm-project/vllm/pull/26467.
        # dict(repetition_penalty=0.1),
        # dict(bad_words=[]),
    ]

    default_params = dict(
        temperature=0.0,  # greedy
        max_tokens=20,
    )

    with monkeypatch.context() as m:
        m.setenv("VLLM_ATTENTION_BACKEND", "FLEX_ATTENTION")
        # m.setenv("VLLM_KERNEL_OVERRIDE_BATCH_INVARIANT", "1")

        outputs = []
        for test_preemption in [False, True]:
            for executor in ["uni", "mp"]:
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
                                )
                            )
                        outputs.append((test_config, results))

    baseline_config, baseline_tests = outputs[0]

    for test_config, test_outputs in outputs[1:]:
        for base_outs, test_outs, params in zip(
            baseline_tests, test_outputs, sampling_param_tests
        ):
            check_outputs_equal(
                outputs_0_lst=base_outs,
                outputs_1_lst=test_outs,
                name_0=f"baseline=[{baseline_config}], params={params}",
                name_1=f"config=[{test_config}], params={params}",
            )

            print(f"PASSED: config=[{test_config}], params={params}")
