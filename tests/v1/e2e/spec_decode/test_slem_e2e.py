# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End-to-end correctness tests for SLEM speculative decoding.

Validates that SLEM (heterogeneous vocab) produces identical outputs to
baseline (no spec decode) under greedy decoding, confirming losslessness.
"""

import pytest
import torch

from tests.utils import large_gpu_mark, single_gpu_only
from vllm import LLM, SamplingParams
from vllm.distributed import cleanup_dist_env_and_memory

TARGET_MODEL = "Qwen/Qwen3-1.7B"
DRAFT_MODEL = "Qwen/Qwen3-0.6B"

PROMPTS = [
    "The capital of France is",
    "Albert Einstein was born in",
    "The largest planet in the solar system is",
    "Water boils at",
    "The speed of light is approximately",
    "The chemical symbol for gold is",
    "Shakespeare wrote",
    "The first human to walk on the moon was",
    "The mitochondria is known as",
    "Python was created by",
    "The Great Wall of China was built to",
    "DNA stands for",
    "The Pythagorean theorem states that",
    "The invention of the printing press is attributed to",
    "Photosynthesis is the process by which",
]


def greedy_params(max_tokens: int = 64) -> SamplingParams:
    return SamplingParams(temperature=0, max_tokens=max_tokens, ignore_eos=False)


@single_gpu_only
@large_gpu_mark(min_gb=20)
def test_slem_lossless_greedy():
    """SLEM with heterogeneous vocab must match baseline under greedy decoding."""
    sampling = greedy_params()

    ref_llm = LLM(
        model=TARGET_MODEL,
        max_model_len=2048,
        enforce_eager=True,
    )
    ref_outputs = ref_llm.generate(PROMPTS, sampling)
    del ref_llm
    torch.accelerator.empty_cache()
    cleanup_dist_env_and_memory()

    spec_llm = LLM(
        model=TARGET_MODEL,
        speculative_config={
            "method": "draft_model",
            "model": DRAFT_MODEL,
            "num_speculative_tokens": 5,
            "use_heterogeneous_vocab": True,
            "heterogeneous_vocab_method": "slem",
        },
        max_model_len=2048,
        enforce_eager=True,
        disable_log_stats=False,
    )
    spec_outputs = spec_llm.generate(PROMPTS, sampling)

    mismatches = []
    for i, (ref, spec) in enumerate(zip(ref_outputs, spec_outputs)):
        ref_text = ref.outputs[0].text
        spec_text = spec.outputs[0].text
        if ref_text != spec_text:
            mismatches.append(
                f"  [{i}] prompt={PROMPTS[i]!r}\n"
                f"       ref ={ref_text!r}\n"
                f"       slem={spec_text!r}"
            )

    del spec_llm
    torch.accelerator.empty_cache()
    cleanup_dist_env_and_memory()

    assert len(mismatches) == 0, (
        f"SLEM produced {len(mismatches)}/{len(PROMPTS)} mismatches "
        f"vs baseline:\n" + "\n".join(mismatches)
    )


@single_gpu_only
@large_gpu_mark(min_gb=20)
def test_slem_acceptance_rate():
    """SLEM should achieve a reasonable acceptance rate with related models."""
    sampling = greedy_params()

    spec_llm = LLM(
        model=TARGET_MODEL,
        speculative_config={
            "method": "draft_model",
            "model": DRAFT_MODEL,
            "num_speculative_tokens": 5,
            "use_heterogeneous_vocab": True,
            "heterogeneous_vocab_method": "slem",
        },
        max_model_len=2048,
        enforce_eager=True,
        disable_log_stats=False,
    )
    spec_llm.generate(PROMPTS, sampling)
    metrics = spec_llm.get_metrics()

    name2metric = {m.name: m for m in metrics}
    n_draft = name2metric["vllm:spec_decode_num_draft_tokens"].value
    n_accepted = name2metric["vllm:spec_decode_num_accepted_tokens"].value

    del spec_llm
    torch.accelerator.empty_cache()
    cleanup_dist_env_and_memory()

    assert n_draft > 0, "No draft tokens were generated"
    acceptance_rate = n_accepted / n_draft
    print(
        f"SLEM acceptance rate: {acceptance_rate:.3f} ({n_accepted}/{n_draft} tokens)"
    )
    assert acceptance_rate > 0.3, (
        f"SLEM acceptance rate {acceptance_rate:.3f} is too low "
        f"(expected > 0.3 for {DRAFT_MODEL} → {TARGET_MODEL})"
    )


@pytest.mark.parametrize("num_speculative_tokens", [3, 5])
@single_gpu_only
@large_gpu_mark(min_gb=20)
def test_slem_varying_spec_tokens(num_speculative_tokens: int):
    """SLEM should work correctly with different speculation lengths."""
    sampling = greedy_params(max_tokens=32)

    ref_llm = LLM(
        model=TARGET_MODEL,
        max_model_len=2048,
        enforce_eager=True,
    )
    prompts = PROMPTS[:5]
    ref_outputs = ref_llm.generate(prompts, sampling)
    del ref_llm
    torch.accelerator.empty_cache()
    cleanup_dist_env_and_memory()

    spec_llm = LLM(
        model=TARGET_MODEL,
        speculative_config={
            "method": "draft_model",
            "model": DRAFT_MODEL,
            "num_speculative_tokens": num_speculative_tokens,
            "use_heterogeneous_vocab": True,
            "heterogeneous_vocab_method": "slem",
        },
        max_model_len=2048,
        enforce_eager=True,
    )
    spec_outputs = spec_llm.generate(prompts, sampling)

    del spec_llm
    torch.accelerator.empty_cache()
    cleanup_dist_env_and_memory()

    for i, (ref, spec) in enumerate(zip(ref_outputs, spec_outputs)):
        assert ref.outputs[0].text == spec.outputs[0].text, (
            f"Mismatch at prompt {i} with num_speculative_tokens="
            f"{num_speculative_tokens}: "
            f"ref={ref.outputs[0].text!r} vs slem={spec.outputs[0].text!r}"
        )
