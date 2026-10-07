# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm import SamplingParams
from vllm.platforms import current_platform
from vllm.v1.watermarking import GumbelWatermarkDetector


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(), reason="requires a CUDA-like accelerator"
)
def test_llm_generated_sequence_is_watermarked(vllm_runner):
    """Engine-level watermarking must produce tokens the detector recognizes."""
    watermark_config = {"key": 42, "context_width": 4}
    detector = GumbelWatermarkDetector(**watermark_config)

    runner = vllm_runner(
        "facebook/opt-125m",
        watermark_config=watermark_config,
        seed=0,
        enforce_eager=True,
        max_model_len=512,
        gpu_memory_utilization=0.2,
    )
    request = "Tell me a story about an explorer who discovers a mysterious island."
    params_use_wm = SamplingParams(temperature=1.0, max_tokens=256, watermarking=True)
    params_no_wm = SamplingParams(temperature=1.0, max_tokens=256, watermarking=False)

    with runner:
        output_use_wm = runner.llm.generate(request, params_use_wm)
        output_no_wm = runner.llm.generate(request, params_no_wm)

    result_use_wm = detector.detect(list(output_use_wm[0].outputs[0].token_ids))
    result_no_wm = detector.detect(list(output_no_wm[0].outputs[0].token_ids))

    assert result_use_wm.is_watermarked, result_use_wm
    recorded_p_value = 9.83e-20
    assert result_use_wm.p_value < recorded_p_value * 10  # tolerance

    assert not result_no_wm.is_watermarked, result_no_wm



@pytest.mark.skipif(
    not current_platform.is_cuda_alike(), reason="requires a CUDA-like accelerator"
)
def test_sbw_greedy_decoding_is_watermarked(vllm_runner):
    """SBW watermarking must work with greedy decoding (temperature=0).

    This is the key advantage of SBW over Gumbel: bias-based watermarking shifts
    logits before argmax, so the watermark signal is embedded even without
    sampling randomness. Gumbel requires temperature > 0 because its noise-based
    approach needs the randomness from sampling.
    """
    from vllm.v1.watermarking import SBWWatermarkDetector

    watermark_config = {"key": 42, "algorithm": "sbw", "context_width": 4}
    detector = SBWWatermarkDetector(
        key=watermark_config["key"], context_width=watermark_config["context_width"]
    )

    runner = vllm_runner(
        "facebook/opt-125m",
        watermark_config=watermark_config,
        seed=0,
        enforce_eager=True,
        max_model_len=512,
        gpu_memory_utilization=0.2,
    )
    request = "Tell me a story about an explorer who discovers a mysterious island."
    # temperature=0 is greedy decoding
    params_greedy_wm = SamplingParams(temperature=0, max_tokens=256, watermarking=True)
    params_greedy_no_wm = SamplingParams(
        temperature=0, max_tokens=256, watermarking=False
    )

    with runner:
        output_greedy_wm = runner.llm.generate(request, params_greedy_wm)
        output_greedy_no_wm = runner.llm.generate(request, params_greedy_no_wm)

    result_greedy_wm = detector.detect(list(output_greedy_wm[0].outputs[0].token_ids))
    result_greedy_no_wm = detector.detect(
        list(output_greedy_no_wm[0].outputs[0].token_ids)
    )

    assert result_greedy_wm.is_watermarked, (
        f"SBW greedy watermark not detected: {result_greedy_wm}"
    )
    assert not result_greedy_no_wm.is_watermarked, (
        f"Non-watermarked greedy output falsely detected: {result_greedy_no_wm}"
    )
