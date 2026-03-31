# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""E2E tests for online block FP8 quantization.

Loads a BF16 model with ``--quantization block_fp8`` (online block-wise
quantization with weight_block_size=[128, 128]) and compares
log-probabilities against the same model served in BF16 without
quantization.  This exercises the full pipeline: config parsing,
``Fp8OnlineLinearMethod`` block path, ``Fp8OnlineMoEMethod`` block path,
weight loading, online block quantization via ``scaled_quantize``, and
inference through ``W8A8BlockFp8LinearOp``.

``example_prompts`` is a pytest fixture (from conftest.py) that loads 8
diverse prompts from ``tests/prompts/example.txt``.
"""

import pytest

from tests.quantization.utils import is_quant_method_supported

from ..utils import check_logprobs_close

DENSE_MODEL = "Qwen/Qwen3-0.6B"
MOE_MODEL = "Qwen/Qwen3-30B-A3B"

MAX_MODEL_LEN = 1024
MAX_TOKENS = 4
NUM_LOG_PROBS = 8


@pytest.mark.skipif(
    not is_quant_method_supported("block_fp8"),
    reason="block_fp8 is not supported on this GPU type.",
)
@pytest.mark.quant_model
@pytest.mark.parametrize("model", [DENSE_MODEL, MOE_MODEL], ids=["dense", "moe"])
def test_block_fp8_logprobs(
    vllm_runner,
    example_prompts,
    model: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Compare BF16 baseline logprobs against online block-FP8-quantized model.

    Runs the same model twice -- once in BF16 (baseline) and once with
    online block FP8 quantization -- then checks that the top
    log-probabilities are close.
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
            quantization="block_fp8",
        ) as vllm_model:
            test_outputs = vllm_model.generate_greedy_logprobs(
                example_prompts, MAX_TOKENS, NUM_LOG_PROBS
            )

        check_logprobs_close(
            outputs_0_lst=baseline_outputs,
            outputs_1_lst=test_outputs,
            name_0="bf16",
            name_1="block_fp8",
        )


@pytest.mark.skipif(
    not is_quant_method_supported("block_fp8"),
    reason="block_fp8 is not supported on this GPU type.",
)
@pytest.mark.quant_model
@pytest.mark.parametrize("model", [DENSE_MODEL, MOE_MODEL], ids=["dense", "moe"])
def test_block_fp8_generation(vllm_runner, model: str) -> None:
    """Smoke test: verify online block FP8 model generates coherent text."""
    prompt = "1 2 3 4 5"
    with vllm_runner(
        model,
        enforce_eager=True,
        quantization="block_fp8",
        max_model_len=MAX_MODEL_LEN,
    ) as vllm_model:
        output = vllm_model.generate_greedy([prompt], max_tokens=5)

    generated = output[0][1]
    assert len(generated) > len(prompt), (
        f"Block FP8 model produced no new tokens. Output: {generated!r}"
    )
