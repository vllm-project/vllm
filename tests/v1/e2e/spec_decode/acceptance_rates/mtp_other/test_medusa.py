# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from tests.evals.gsm8k.gsm8k_eval import _build_gsm8k_prompts
from vllm import SamplingParams
from vllm.config import CompilationConfig

from ...utils import compute_acceptance_rate


def test_medusa_acceptance_rate(
    sampling_config: SamplingParams,
    vllm_runner,
):
    """Verify a trained Medusa checkpoint achieves nonzero acceptance rate.

    Uses the canonical FasterDecoding vicuna-7b checkpoint to confirm the
    speculation path actually accepts tokens — unlike test_medusa_correctness,
    which uses a random head and only validates output correctness.
    """
    target_model = "lmsys/vicuna-7b-v1.3"
    medusa_model = "FasterDecoding/medusa-vicuna-7b-v1.3"
    prompts = _build_gsm8k_prompts(num_questions=10, num_shots=1)[0]

    with vllm_runner(
        target_model,
        block_size=None,
        trust_remote_code=False,
        speculative_config={
            "method": "medusa",
            "model": medusa_model,
            "num_speculative_tokens": 3,
        },
        max_model_len=1024,
        enforce_eager=True,
        enable_chunked_prefill=None,
        disable_log_stats=False,
        compilation_config=CompilationConfig(),
    ) as spec_runner:
        spec_runner.llm.generate(prompts, sampling_config)
        metrics = spec_runner.llm.get_metrics()
        acceptance_rate = compute_acceptance_rate(metrics)

    min_acceptance_rate = 0.198
    print(f"Medusa acceptance rate: {acceptance_rate:.4f} (min {min_acceptance_rate})")

    # Regression guard at 90% of the measured baseline.
    assert acceptance_rate >= min_acceptance_rate, (
        f"Medusa acceptance rate {acceptance_rate:.4f} below min {min_acceptance_rate}"
    )
