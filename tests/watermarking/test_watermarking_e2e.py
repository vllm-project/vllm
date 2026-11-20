# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm import SamplingParams
from vllm.platforms import current_platform
from vllm.v1.metrics.reader import Counter
from vllm.v1.watermarking import DualKeyGumbelWatermarkDetector, GumbelWatermarkDetector


@pytest.mark.parametrize("enforce", [False, True])
@pytest.mark.skipif(
    not current_platform.is_cuda_alike(), reason="requires a CUDA-like accelerator"
)
def test_llm_generated_sequence_is_watermarked(vllm_runner, enforce):
    """Engine-level watermarking must produce tokens the detector recognizes."""
    watermark_config = {"key": 42, "context_width": 4}
    detector = GumbelWatermarkDetector(**watermark_config)

    runner = vllm_runner(
        "facebook/opt-125m",
        watermark_config={**watermark_config, "enforce": enforce},
        seed=0,
        enforce_eager=True,
        max_model_len=512,
        gpu_memory_utilization=0.2,
        disable_log_stats=False,
    )
    request = "Tell me a story about an explorer who discovers a mysterious island."
    # Openings use ordinary sampling, so the recorded p-value needs a fixed seed.
    params_use_wm = SamplingParams(
        temperature=1.0, max_tokens=256, watermarking=True, seed=0
    )
    params_no_wm = SamplingParams(temperature=1.0, max_tokens=256, watermarking=False)
    params_greedy = SamplingParams(temperature=0.0, max_tokens=16)

    with runner:
        output_use_wm = runner.llm.generate(request, params_use_wm)
        output_no_wm = runner.llm.generate(request, params_no_wm)
        runner.llm.generate(request, params_greedy)
        metrics = runner.llm.get_metrics()

    # The greedy request is skipped. The opt-out is not watermarked unless
    # enforce overrides it.
    counters = [m for m in metrics if isinstance(m, Counter)]
    status = {
        m.labels["status"]: m.value
        for m in counters
        if m.name == "vllm:watermark_requests"
    }
    assert status == {
        "watermarked": 1 + enforce,
        "skipped": 1,
        "not_watermarked": 1 - enforce,
    }
    num_success = sum(m.value for m in counters if m.name == "vllm:request_success")
    assert sum(status.values()) == num_success

    result_use_wm = detector.detect(list(output_use_wm[0].outputs[0].token_ids))
    result_no_wm = detector.detect(list(output_no_wm[0].outputs[0].token_ids))

    assert result_use_wm.is_watermarked, result_use_wm
    recorded_p_value = 2.68e-29
    assert result_use_wm.p_value < recorded_p_value * 10  # tolerance

    assert result_no_wm.is_watermarked == enforce, result_no_wm


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(), reason="requires a CUDA-like accelerator"
)
def test_dflash2_drafts_are_watermarked(vllm_runner):
    """DFlash2 candidate-head drafts must be accepted and their output detected."""
    watermark_config = {"algorithm": "dual_key_gumbel", "key": 42, "context_width": 4}
    detector = DualKeyGumbelWatermarkDetector(key=42, context_width=4)
    runner = vllm_runner(
        "Qwen/Qwen3.5-4B",
        speculative_config={
            "method": "dflash",
            "model": "vllm-project/Qwen3.5-4B-DFlash2",
            "num_speculative_tokens": 7,
            "draft_sample_method": "probabilistic",
        },
        watermark_config=watermark_config,
        language_model_only=True,
        seed=0,
        max_model_len=1024,
        max_num_seqs=8,
        gpu_memory_utilization=0.6,
        disable_log_stats=False,
    )
    prompts = [
        "Tell me a story about an explorer who discovers a mysterious island.",
        "Write a letter from a lighthouse keeper to their sister.",
        "Describe a busy market in a city that floats on the sea.",
        "Explain to a child why the sky changes colour at sunset.",
        "Write a diary entry of a cat that learned to read.",
        "Describe the first day at a school for young wizards.",
        "Tell a story about a robot that wants to become a gardener.",
        "Write a short speech for the opening of a mountain library.",
    ]

    def params(watermarking: bool) -> list[SamplingParams]:
        return [
            SamplingParams(
                temperature=1.0, max_tokens=256, watermarking=watermarking, seed=i
            )
            for i in range(len(prompts))
        ]

    with runner:
        outputs_wm = runner.llm.generate(prompts, params(True))
        outputs_no_wm = runner.llm.generate(prompts, params(False))
        metrics = runner.llm.get_metrics()

    accepted = sum(
        m.value
        for m in metrics
        if isinstance(m, Counter) and m.name == "vllm:spec_decode_num_accepted_tokens"
    )
    assert accepted > 0, "DFlash2 accepted no drafts"

    def num_detected(outputs) -> int:
        return sum(
            detector.detect(list(o.outputs[0].token_ids)).is_watermarked
            for o in outputs
        )

    # H100: 6-8 of 8 detected over 10 seed sets, 0 of 8 without watermarking.
    assert num_detected(outputs_wm) >= 5
    assert num_detected(outputs_no_wm) <= 1
