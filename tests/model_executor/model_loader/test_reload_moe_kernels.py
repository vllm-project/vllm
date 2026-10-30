# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reloading the same weights must not change what a served MoE model
generates: post-load processing reruns, and the tensors it derives for the
kernels must stay where the captured CUDA graphs read them."""

import pytest

from tests.models.utils import check_logprobs_close
from vllm.platforms import current_platform

PROMPTS = [
    "The capital of France is",
    "List the first ten prime numbers:",
    "Explain why the sky is blue in two sentences.",
    "Q: A train travels 60 km in 45 minutes. What is its speed in km/h?\nA:",
]


@pytest.mark.skipif(
    not current_platform.is_device_capability_family(100),
    reason="TRT-LLM and CuteDSL MoE kernels need SM100",
)
@pytest.mark.parametrize(
    "model,moe_backend,reload_mode",
    [
        ("openai/gpt-oss-20b", "flashinfer_trtllm", "direct"),
        ("RedHatAI/Qwen3-30B-A3B-NVFP4", "flashinfer_cutedsl", "direct"),
        ("RedHatAI/Qwen3-30B-A3B-NVFP4", "flashinfer_cutedsl", "layerwise"),
    ],
)
def test_reload_keeps_moe_output(
    model: str, moe_backend: str, reload_mode: str, vllm_runner, monkeypatch
) -> None:
    monkeypatch.setenv("VLLM_RELOAD_MODE", reload_mode)
    with vllm_runner(
        model,
        max_model_len=1024,
        max_num_seqs=4,
        enable_prefix_caching=False,
        kernel_config={"enable_jit_warmup": False, "moe_backend": moe_backend},
    ) as llm:
        before = llm.generate_greedy_logprobs(PROMPTS, max_tokens=32, num_logprobs=5)
        llm.collective_rpc("reload_weights")  # the served checkpoint
        after = llm.generate_greedy_logprobs(PROMPTS, max_tokens=32, num_logprobs=5)
    check_logprobs_close(
        outputs_0_lst=before,
        outputs_1_lst=after,
        name_0="before reload",
        name_1="after reload",
    )
