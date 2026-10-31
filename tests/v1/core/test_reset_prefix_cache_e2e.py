# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from tests.models.utils import check_logprobs_close
from vllm import EngineArgs, LLMEngine, SamplingParams

PROMPTS = [
    "A robot may not injure a human being ",
    "To be or not to be,",
    "What is the meaning of life?",
    "What does the fox say? " * 20,  # Test long prompt
]


def test_reset_prefix_cache_e2e(monkeypatch):
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


def test_mixed_prompt_embeds_cache_history(hf_runner, vllm_runner, monkeypatch):
    """Changing only the routing mask must not reuse another prompt's KV."""
    monkeypatch.setenv("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    model = "Qwen/Qwen3-0.6B"
    with hf_runner(model, dtype="half") as hf_model:
        prompt_embeds = (
            hf_model.model.get_input_embeddings()
            .weight[198]
            .detach()
            .cpu()
            .repeat(33, 1)
        )

    prompt_b = {
        "prompt_token_ids": [0] * 33,
        "prompt_embeds": prompt_embeds,
        "prompt_is_token_ids": [True] * 33,
    }
    prompt_a = {**prompt_b, "prompt_is_token_ids": [False] * 32 + [True]}
    params = SamplingParams(temperature=0, max_tokens=1, logprobs=5)

    with vllm_runner(
        model,
        dtype="half",
        enable_prompt_embeds=True,
        enable_prefix_caching=True,
        enable_chunked_prefill=True,
        enforce_eager=True,
        block_size=16,
        max_model_len=256,
        max_num_seqs=1,
        gpu_memory_utilization=0.3,
    ) as vllm_model:
        llm = vllm_model.llm
        cold_b = llm.generate([prompt_b], params)[0]
        assert llm.reset_prefix_cache()
        cold_a = llm.generate([prompt_a], params)[0]
        repeated_a = llm.generate([prompt_a], params)[0]
        after_a = llm.generate([prompt_b], params)[0]

    assert cold_b.num_cached_tokens == cold_a.num_cached_tokens == 0
    assert repeated_a.num_cached_tokens == 32
    assert after_a.num_cached_tokens == 0
    check_logprobs_close(
        outputs_0_lst=[
            (list(o.token_ids), o.text, o.logprobs)
            for result in (cold_b, cold_a)
            for o in result.outputs
        ],
        outputs_1_lst=[
            (list(o.token_ids), o.text, o.logprobs)
            for result in (after_a, repeated_a)
            for o in result.outputs
        ],
        name_0="cold",
        name_1="after_a",
        always_check_logprobs=True,
    )
