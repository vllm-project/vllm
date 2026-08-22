# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from tests.evals.gsm8k.gsm8k_eval import _build_gsm8k_prompts
from vllm import SamplingParams
from vllm.config import CompilationConfig

from ...utils import compute_acceptance_len

TARGET_MODEL = "unsloth/Meta-Llama-3.1-8B-Instruct"
PARD2_DRAFT = "amd/PARD2-Llama-3.1-8B"


def test_pard2_acceptance_length(vllm_runner):
    """PARD-2 (https://arxiv.org/abs/2504.18583) target-dependent fusion should
    accept multiple draft tokens per step; guards against a fusion regression that
    collapses acceptance length toward 1.0."""
    prompts = _build_gsm8k_prompts(num_questions=50, num_shots=5)[0]

    with vllm_runner(
        TARGET_MODEL,
        block_size=None,
        trust_remote_code=False,
        speculative_config={
            "model": PARD2_DRAFT,
            "num_speculative_tokens": 5,
        },
        max_model_len=4096,
        enforce_eager=True,
        enable_chunked_prefill=None,
        disable_log_stats=False,
        compilation_config=CompilationConfig(),
    ) as spec_runner:
        spec_runner.llm.generate(prompts, SamplingParams(temperature=0, max_tokens=256))
        acceptance_len = compute_acceptance_len(spec_runner.llm.get_metrics())

    min_acceptance_len = 3.0
    print(f"PARD-2 acceptance length: {acceptance_len:.4f} (min {min_acceptance_len})")
    assert acceptance_len >= min_acceptance_len, (
        f"PARD-2 acceptance length {acceptance_len:.4f} below min {min_acceptance_len}"
    )
