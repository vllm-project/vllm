# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from tests.utils import large_gpu_mark
from vllm import LLM, SamplingParams
from vllm.distributed import cleanup_dist_env_and_memory


def _get_counter(metrics, name: str) -> float:
    metric = next((m for m in metrics if m.name == name), None)
    assert metric is not None, f"Missing metric: {name}"
    return metric.value


@pytest.mark.slow_test
@large_gpu_mark(min_gb=80)
def test_laguna_dflash_hf_pair_smoke(monkeypatch):
    """Smoke-test the public Laguna XS-2.1 base/DFlash checkpoint pair."""
    monkeypatch.setenv("VLLM_USE_FLASHINFER_SAMPLER", "0")

    llm = LLM(
        model="poolside/Laguna-XS-2.1",
        trust_remote_code=True,
        speculative_config={
            "method": "dflash",
            "model": "poolside/Laguna-XS-2.1-DFlash",
            "num_speculative_tokens": 15,
        },
        max_model_len=8192,
        max_num_batched_tokens=65536,
        max_num_seqs=32,
        enforce_eager=True,
        disable_log_stats=False,
    )

    try:
        outputs = llm.generate(
            [
                "What is the capital of the United Kingdom?",
                "Write a Python function that returns the square of a number.",
            ],
            SamplingParams(temperature=0.0, max_tokens=32, ignore_eos=True),
        )
        assert len(outputs) == 2
        assert all(output.outputs[0].text for output in outputs)

        metrics = llm.get_metrics()
        num_drafts = _get_counter(metrics, "vllm:spec_decode_num_drafts")
        num_accepted = _get_counter(metrics, "vllm:spec_decode_num_accepted_tokens")

        assert num_drafts > 0
        acceptance_len = 1 + (num_accepted / num_drafts)
        assert acceptance_len > 1.0
    finally:
        del llm
        torch.accelerator.empty_cache()
        cleanup_dist_env_and_memory()
