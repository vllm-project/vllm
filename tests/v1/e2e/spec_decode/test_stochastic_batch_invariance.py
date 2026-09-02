# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from tests.utils import large_gpu_mark, single_gpu_only
from vllm import SamplingParams

from .utils import compute_acceptance_len, get_spec_decode_metric_value


@single_gpu_only
@large_gpu_mark(min_gb=32)
def test_spec_batch_invariance_with_preemption(monkeypatch, vllm_runner):
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    monkeypatch.setenv("VLLM_USE_FLASHINFER_SAMPLER", "0")

    prompts = [
        "The capital of France is",
        "Explain why the sky is blue.",
        "Write a short story about a robot.",
        "List three practical uses of water.",
    ]
    params = [
        SamplingParams(temperature=0.8, max_tokens=48, seed=42 + i)
        for i in range(len(prompts))
    ]
    speculative_config = {
        "model": "deepseek-ai/dspark_qwen3_4b_block7",
        "revision": "3457dff1417cb84927f6098a5fcb7cee85c934b7",
        "method": "dspark",
        "num_speculative_tokens": 7,
        "draft_sample_method": "probabilistic",
        "rejection_sample_method": "standard",
    }

    with vllm_runner(
        "Qwen/Qwen3-4B-FP8",
        revision="f3ecd40dbe901708a7557adbecbabab618178ae4",
        speculative_config=speculative_config,
        max_model_len=128,
        max_num_seqs=len(prompts),
        max_num_batched_tokens=128,
        max_num_scheduled_tokens=23,
        num_gpu_blocks_override=9,
        enable_chunked_prefill=True,
        enable_prefix_caching=True,
        enforce_eager=True,
        disable_log_stats=False,
    ) as runner:
        serial = [
            runner.llm.generate([prompt], param, use_tqdm=False)[0].outputs[0]
            for prompt, param in zip(prompts, params)
        ]
        assert runner.llm.reset_prefix_cache()
        metrics_before = runner.llm.get_metrics()
        batched = [
            output.outputs[0]
            for output in runner.llm.generate(prompts, params, use_tqdm=False)
        ]
        metrics_after = runner.llm.get_metrics()

    assert (
        get_spec_decode_metric_value(metrics_after, "vllm:num_preemptions")
        - get_spec_decode_metric_value(metrics_before, "vllm:num_preemptions")
        > 0
    )
    assert compute_acceptance_len(metrics_after, metrics_before) > 1
    assert [tuple(item.token_ids) for item in batched] == [
        tuple(item.token_ids) for item in serial
    ]
