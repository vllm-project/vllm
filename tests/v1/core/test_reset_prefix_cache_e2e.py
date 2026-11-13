# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm import EngineArgs, LLMEngine, SamplingParams


def test_prefix_cache_diagnostics_e2e(vllm_runner, monkeypatch, capfd):
    """Real requests distinguish bounded eviction evidence from key misses."""
    monkeypatch.setenv("VLLM_PREFIX_CACHE_DIAGNOSTICS", "1")
    monkeypatch.setenv("VLLM_LOGGING_LEVEL", "DEBUG")
    monkeypatch.setenv("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    with vllm_runner(
        "Qwen/Qwen3-0.6B",
        dtype="bfloat16",
        enforce_eager=True,
        enable_prefix_caching=True,
        block_size=16,
        num_gpu_blocks_override=32,
        max_model_len=496,
        max_num_batched_tokens=496,
        max_num_seqs=1,
        gpu_memory_utilization=0.2,
        attention_config={"backend": "TRITON_ATTN"},
    ) as runner:
        params = SamplingParams(temperature=0, max_tokens=1, ignore_eos=True)
        tokens = list(range(100, 133))

        def generate(prompt, salt="A"):
            return runner.llm.generate(
                {"prompt_token_ids": prompt, "cache_salt": salt},
                params,
                use_tqdm=False,
            )[0].num_cached_tokens

        assert generate(tokens) == 0
        assert generate(tokens) == 32
        assert generate(tokens, "B") == 0
        assert generate(list(range(1000, 1495))) == 0
        assert generate(tokens) == 0
    captured = capfd.readouterr()
    diagnostics = "\n".join(
        line
        for line in (captured.out + captured.err).splitlines()
        if "prefix_cache_diagnostic " in line
    )
    assert "reason=no_match" in diagnostics
    assert "reason=lookup_limit" in diagnostics
    assert "reason=evicted" in diagnostics


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
