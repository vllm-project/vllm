# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""E2E tests for online FP8 per-channel quantization.

Loads a BF16 model with ``--quantization fp8_per_channel`` (online
quantization) and compares log-probabilities against the same model served in
BF16 without quantization.  This exercises the full pipeline: config parsing,
``Fp8PtpcOnlineLinearMethod``, ``Fp8PtpcOnlineMoEMethod``, weight
loading, online quantization / shuffling, and inference.

``example_prompts`` is a pytest fixture (from conftest.py) that loads 8
diverse prompts from ``tests/prompts/example.txt``.
"""

import pytest

from tests.quantization.utils import is_quant_method_supported

from ..utils import check_logprobs_close

# Small MoE model that fits on a single GPU and exercises both linear + MoE.
MOE_MODEL = "allenai/OLMoE-1B-7B-0125-Instruct"
# Small dense model (no MoE) to validate the linear-only path.
DENSE_MODEL = "Qwen/Qwen3-0.6B"

MAX_MODEL_LEN = 1024
MAX_TOKENS = 4
NUM_LOG_PROBS = 8


@pytest.mark.skipif(
    not is_quant_method_supported("fp8"),
    reason="fp8 is not supported on this GPU type.",
)
@pytest.mark.quant_model
@pytest.mark.parametrize("model", [DENSE_MODEL, MOE_MODEL], ids=["dense", "moe"])
def test_fp8_per_channel_logprobs(
    vllm_runner,
    example_prompts,
    model: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Compare BF16 baseline logprobs against online per-channel-quantized
    model.

    Runs the same model twice -- once in BF16 (baseline) and once with online
    FP8 per-channel quantization -- then checks that the top log-probabilities
    are close.  Only 4 tokens are generated to keep the test fast while still
    catching numerical divergence beyond expected per-channel error.
    """
    with monkeypatch.context() as m:
        m.setenv("TOKENIZERS_PARALLELISM", "true")

        with vllm_runner(
            model,
            max_model_len=MAX_MODEL_LEN,
            enforce_eager=True,
        ) as vllm_model:
            baseline_outputs = vllm_model.generate_greedy_logprobs(
                example_prompts, MAX_TOKENS, NUM_LOG_PROBS
            )

        with vllm_runner(
            model,
            max_model_len=MAX_MODEL_LEN,
            enforce_eager=True,
            quantization="fp8_per_channel",
        ) as vllm_model:
            test_outputs = vllm_model.generate_greedy_logprobs(
                example_prompts, MAX_TOKENS, NUM_LOG_PROBS
            )

        check_logprobs_close(
            outputs_0_lst=baseline_outputs,
            outputs_1_lst=test_outputs,
            name_0="bf16",
            name_1="fp8_per_channel",
        )
