# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm import SamplingParams
from vllm.platforms import current_platform
from vllm.v1.metrics.reader import Counter
from vllm.v1.watermarking import GumbelWatermarkDetector


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


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(), reason="requires a CUDA-like accelerator"
)
def test_sbw_beam_search_is_watermarked(vllm_runner):
    """SBW watermarking must watermark beam search output.

    Beam expansion ranks candidates by the returned logprobs, so the engine
    must run a processed logprobs mode for the bias to be visible; on the raw
    default the request runs unwatermarked with a warning.
    """
    from vllm.sampling_params import BeamSearchParams
    from vllm.v1.watermarking import SBWWatermarkDetector

    watermark_config = {"key": 42, "algorithm": "sbw", "context_width": 4}
    detector = SBWWatermarkDetector(key=42, context_width=4)

    runner = vllm_runner(
        "facebook/opt-125m",
        watermark_config=watermark_config,
        logprobs_mode="processed_logprobs",
        seed=0,
        enforce_eager=True,
        max_model_len=512,
        gpu_memory_utilization=0.2,
    )
    request = "Tell me a story about an explorer who discovers a mysterious island."
    params_wm = BeamSearchParams(beam_width=4, max_tokens=64)
    params_no_wm = BeamSearchParams(beam_width=4, max_tokens=64, watermarking=False)

    with runner:
        output_wm = runner.llm.beam_search([request], params_wm)[0]
        output_no_wm = runner.llm.beam_search([request], params_no_wm)[0]

    best_wm = output_wm.sequences[0]
    best_no_wm = output_no_wm.sequences[0]
    assert len(best_wm.tokens) > 0

    # The detector consumes the generated tokens only, matching the
    # detector's convention elsewhere (cf. offline.py's generated split).
    prompt_len = len(best_wm.orig_prompt["prompt_token_ids"])
    result_wm = detector.detect(list(best_wm.tokens[prompt_len:]))
    result_no_wm = detector.detect(list(best_no_wm.tokens[prompt_len:]))

    assert result_wm.is_watermarked, f"SBW beam watermark not detected: {result_wm}"
    assert result_no_wm.num_scored_tokens > 0
    assert not result_no_wm.is_watermarked, (
        f"Non-watermarked beam output falsely detected: {result_no_wm}"
    )
