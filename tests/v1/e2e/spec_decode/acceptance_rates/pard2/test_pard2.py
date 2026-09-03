# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from tests.evals.gsm8k.gsm8k_eval import _build_gsm8k_prompts
from vllm import SamplingParams
from vllm.config import CompilationConfig

from ...utils import compute_acceptance_len

TARGET_MODEL = "RedHatAI/Meta-Llama-3.1-8B-Instruct-quantized.w8a8"
PARD2_DRAFT = "amd/PARD2-Llama-3.1-8B"

# The Qwen3 draft binds its attention sink to the first token of the sequence,
# so it is the checkpoint that actually notices when the draft is fed a
# left-shifted sequence: dropping token 0 takes it from ~7.7 down to ~2.0.
ANCHOR_TARGET_MODEL = "Qwen/Qwen3-8B"
ANCHOR_PARD2_DRAFT = "amd/PARD2-Qwen3-8B"


def test_pard2_acceptance_length(vllm_runner):
    """PARD-2 (https://arxiv.org/abs/2504.18583) fusion should accept multiple draft
    tokens per step; acceptance near 1.0 means it broke. The quantized target with a
    bf16 draft also covers building draft layers from the draft's own quant config."""
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


def test_pard2_anchor_acceptance_length(vllm_runner):
    """PARD-2 must see token 0 (with a zero target feature) as the draft's first row.

    Feeding the draft an EAGLE-style left-shifted sequence drops that row. This
    checkpoint depends on it, so acceptance collapses from ~7.7 to ~2.0 --- a
    threshold the Llama test above cannot catch, since that draft is only mildly
    affected. Measured 7.72 with the anchor row, 1.99 without.
    """
    prompts = _build_gsm8k_prompts(num_questions=50, num_shots=5)[0]

    with vllm_runner(
        ANCHOR_TARGET_MODEL,
        block_size=None,
        trust_remote_code=False,
        speculative_config={
            "model": ANCHOR_PARD2_DRAFT,
            "num_speculative_tokens": 15,
        },
        max_model_len=4096,
        enforce_eager=True,
        enable_chunked_prefill=None,
        disable_log_stats=False,
        compilation_config=CompilationConfig(),
    ) as spec_runner:
        spec_runner.llm.generate(prompts, SamplingParams(temperature=0, max_tokens=256))
        acceptance_len = compute_acceptance_len(spec_runner.llm.get_metrics())

    min_acceptance_len = 6.0
    print(
        f"PARD-2 anchor acceptance length: {acceptance_len:.4f} "
        f"(min {min_acceptance_len})"
    )
    assert acceptance_len >= min_acceptance_len, (
        f"PARD-2 acceptance length {acceptance_len:.4f} below min "
        f"{min_acceptance_len}; the draft is likely not seeing the anchor row"
    )
