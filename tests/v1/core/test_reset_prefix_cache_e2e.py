# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm import EngineArgs, LLMEngine, SamplingParams

PROMPTS = [
    "A robot may not injure a human being ",
    "To be or not to be,",
    "What is the meaning of life?",
    "What does the fox say? " * 20,  # Test long prompt
]


@pytest.mark.parametrize("keep_running_requests", [False, True])
def test_reset_prefix_cache_e2e(monkeypatch, keep_running_requests: bool):
    # "spawn" is required for test to be deterministic
    monkeypatch.setenv("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    engine_args = EngineArgs(
        model="Qwen/Qwen3-0.6B",
        gpu_memory_utilization=0.2,
        async_scheduling=True,
        max_num_batched_tokens=32,
        max_model_len=2048,
        compilation_config={"mode": 0},
        dtype="float16",
    )
    engine = LLMEngine.from_engine_args(engine_args)
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=16,
    )

    # No preempt case:
    for i, prompt in enumerate(PROMPTS):
        engine.add_request("ground_truth_" + str(i), prompt, sampling_params)

    ground_truth_results = {}
    while engine.has_unfinished_requests():
        request_outputs = engine.step()
        for request_output in request_outputs:
            if request_output.finished:
                ground_truth_results[request_output.request_id] = request_output

    # Preempt case:
    for i, prompt in enumerate(PROMPTS):
        engine.add_request("preempted_" + str(i), prompt, sampling_params)

    step_id = 0
    preempted_results = {}
    while engine.has_unfinished_requests():
        if step_id == 10:
            assert engine.reset_prefix_cache(
                reset_running_requests=not keep_running_requests,
                keep_running_requests=keep_running_requests,
            )

        request_outputs = engine.step()

        for request_output in request_outputs:
            if request_output.finished:
                preempted_results[request_output.request_id] = request_output
        step_id += 1

    for i in range(len(PROMPTS)):
        assert (
            ground_truth_results["ground_truth_" + str(i)].outputs[0].text
            == preempted_results["preempted_" + str(i)].outputs[0].text
        ), (
            f"ground_truth_results['ground_truth_{i}'].outputs[0].text="
            f"{ground_truth_results['ground_truth_' + str(i)].outputs[0].text} "
            f"preempted_results['preempted_{i}'].outputs[0].text="
            f"{preempted_results['preempted_' + str(i)].outputs[0].text}"
        )

    if keep_running_requests:
        # Prefixes cached before the reset can no longer be hit.
        engine.add_request("after_reset", PROMPTS[3], sampling_params)
        after_reset_results = []
        while engine.has_unfinished_requests():
            for request_output in engine.step():
                if request_output.finished:
                    after_reset_results.append(request_output)
        assert len(after_reset_results) == 1
        assert after_reset_results[0].num_cached_tokens == 0
