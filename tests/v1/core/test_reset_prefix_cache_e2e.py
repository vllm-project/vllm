# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm import EngineArgs, LLMEngine, SamplingParams

PROMPTS = [
    "A robot may not injure a human being ",
    "To be or not to be,",
    "What is the meaning of life?",
    "What does the fox say? " * 20,  # Test long prompt
]


def test_reset_prefix_cache_e2e():
    engine_args = EngineArgs(
        model="Qwen/Qwen3-0.6B",
        gpu_memory_utilization=0.2,
        async_scheduling=True,
        max_num_batched_tokens=32,
        max_model_len=2048,
        compilation_config={"mode": 0},
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
            engine.reset_prefix_cache(reset_running_requests=True)

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
