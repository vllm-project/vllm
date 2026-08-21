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
        "model": "RedHatAI/Qwen3-8B-speculator.eagle3",
        "revision": "08610ffa01dd9f16731fe8f627b85905b6aa51c4",
        "method": "eagle3",
        "num_speculative_tokens": 3,
        "draft_sample_method": "probabilistic",
        "rejection_sample_method": "standard",
        "enable_adaptive_verification": False,
    }

    with vllm_runner(
        "Qwen/Qwen3-8B",
        revision="b968826d9c46dd6066d109eabc6255188de91218",
        speculative_config=speculative_config,
        max_model_len=128,
        max_num_seqs=len(prompts),
        max_num_batched_tokens=128,
        num_gpu_blocks_override=8,
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
    assert (
        get_spec_decode_metric_value(metrics_after, "vllm:spec_decode_num_draft_tokens")
        - get_spec_decode_metric_value(
            metrics_before, "vllm:spec_decode_num_draft_tokens"
        )
        > 0
    )
    assert compute_acceptance_len(metrics_after, metrics_before) > 1
    assert [tuple(item.token_ids) for item in batched] == [
        tuple(item.token_ids) for item in serial
    ]
