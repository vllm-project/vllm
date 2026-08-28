# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from tests.utils import large_gpu_mark
from vllm import LLM, SamplingParams
from vllm.distributed import cleanup_dist_env_and_memory

TARGET = "Qwen/Qwen3-8B"
DRAFT = "UIUC-SSAIL/Qwen3-8B-XPress-b16"


def _get_counter(metrics, name: str) -> float:
    metric = next((m for m in metrics if m.name == name), None)
    assert metric is not None, f"Missing metric: {name}"
    return metric.value


@pytest.mark.slow_test
@large_gpu_mark(min_gb=80)
def test_xpress_accepts_more_than_the_bare_drafter(monkeypatch):
    """XPress must accept more per step than the drafter it refines.

    The refiner only adds a bias on top of the same block-diffusion draft, so if
    acceptance is not above the block drafter's own, the head is contributing
    nothing and something is mis-wired (weights not loaded, K read as 0, the
    mixer left unfolded).
    """
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
        outputs = llm.generate(
            [
                "What is the capital of the United Kingdom?",
                "Write a Python function that returns the square of a number.",
            ],
            SamplingParams(temperature=0.0, max_tokens=64, ignore_eos=True),
        )
        assert len(outputs) == 2
        assert all(output.outputs[0].text for output in outputs)

        metrics = llm.get_metrics()
        num_drafts = _get_counter(metrics, "vllm:spec_decode_num_drafts")
        num_accepted = _get_counter(metrics, "vllm:spec_decode_num_accepted_tokens")

        assert num_drafts > 0
        acceptance_len = 1 + (num_accepted / num_drafts)
        # The bare block drafter sits near 6.5 on this pair; anything at or below
        # ~2 means the refiner is not actually in the loop.
        assert acceptance_len > 2.0, f"acceptance length {acceptance_len:.2f} too low"
    finally:
        del llm
        torch.accelerator.empty_cache()
        cleanup_dist_env_and_memory()
