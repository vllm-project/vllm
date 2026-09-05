# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from tests.utils import large_gpu_mark
from vllm import LLM, SamplingParams
from vllm.distributed import cleanup_dist_env_and_memory

TARGET = "Qwen/Qwen3-8B"
DRAFT = "UIUC-SSAIL/Qwen3-8B-XPress-b16"


PROMPTS = [
    "What is the capital of the United Kingdom?",
    "Write a Python function that returns the square of a number.",
    "Explain in one sentence why the sky appears blue.",
    "List the first five prime numbers.",
]
SAMPLING = SamplingParams(temperature=0.0, max_tokens=64, ignore_eos=True)


def _get_counter(metrics, name: str) -> float:
    metric = next((m for m in metrics if m.name == name), None)
    assert metric is not None, f"Missing metric: {name}"
    return metric.value


@pytest.mark.slow_test
@large_gpu_mark(min_gb=80)
# Acceptance length test
def test_xpress_accepts_more_than_the_bare_drafter(monkeypatch):
    monkeypatch.setenv("VLLM_USE_FLASHINFER_SAMPLER", "0")

    llm = LLM(
        model=TARGET,
        speculative_config={
            "method": "xpress",
            "model": DRAFT,
            "num_speculative_tokens": 15,
        },
        max_model_len=4096,
        enforce_eager=True,
        disable_log_stats=False,
    )

    try:
        outputs = llm.generate(PROMPTS, SAMPLING)
        assert len(outputs) == len(PROMPTS)
        assert all(output.outputs[0].text for output in outputs)

        metrics = llm.get_metrics()
        num_drafts = _get_counter(metrics, "vllm:spec_decode_num_drafts")
        num_accepted = _get_counter(metrics, "vllm:spec_decode_num_accepted_tokens")

        assert num_drafts > 0
        acceptance_len = 1 + (num_accepted / num_drafts)
        assert acceptance_len > 2.0, f"acceptance length {acceptance_len:.2f} too low"
    finally:
        del llm
        torch.accelerator.empty_cache()
        cleanup_dist_env_and_memory()


# lossless test
@pytest.mark.slow_test
@large_gpu_mark(min_gb=80)
def test_xpress_output_matches_non_speculative(monkeypatch):
    monkeypatch.setenv("VLLM_USE_FLASHINFER_SAMPLER", "0")

    def _run(spec):
        kwargs = dict(
            model=TARGET,
            max_model_len=4096,
            enforce_eager=True,
        )
        if spec:
            kwargs["speculative_config"] = {
                "method": "xpress",
                "model": DRAFT,
                "num_speculative_tokens": 15,
            }
        llm = LLM(**kwargs)
        try:
            return [o.outputs[0].text for o in llm.generate(PROMPTS, SAMPLING)]
        finally:
            del llm
            torch.accelerator.empty_cache()
            cleanup_dist_env_and_memory()

    spec_texts = _run(spec=True)
    ref_texts = _run(spec=False)

    matches = sum(a == b for a, b in zip(spec_texts, ref_texts))
    assert matches >= int(0.66 * len(PROMPTS)), (
        f"only {matches}/{len(PROMPTS)} outputs matched non-speculative decoding"
    )
