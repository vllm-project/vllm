# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""E2E tests for online MXFP8 quantization.

Loads a BF16 model with ``--quantization mxfp8`` (online quantization) and
compares log-probabilities against the same model served in BF16 without
quantization.  This exercises the full pipeline: config parsing,
``Mxfp8OnlineLinearMethod``, ``Mxfp8OnlineMoEMethod``, weight loading,
online quantization / shuffling, and inference through ``apply_monolithic``.

Layer skipping (``modules_to_not_convert``) is configured in the model's
``config.json`` under ``quantization_config`` and is not tested here.

``example_prompts`` is a pytest fixture (from conftest.py) that loads 8
diverse prompts from ``tests/prompts/example.txt``.
"""

import pytest

from tests.quantization.utils import is_quant_method_supported

from ..utils import check_logprobs_close

# A small MoE model that fits on a single GPU and has both linear + MoE layers.
MOE_MODEL = "Qwen/Qwen3-30B-A3B"
# A small dense model (no MoE) to validate the linear-only path.
DENSE_MODEL = "Qwen/Qwen3-0.6B"

MAX_MODEL_LEN = 1024
MAX_TOKENS = 4
NUM_LOG_PROBS = 8


@pytest.mark.skipif(
    not is_quant_method_supported("mxfp8"),
    reason="mxfp8 is not supported on this GPU type (requires sm_100+).",
)
@pytest.mark.quant_model
@pytest.mark.parametrize("model", [DENSE_MODEL, MOE_MODEL], ids=["dense", "moe"])
def test_mxfp8_logprobs(
    vllm_runner,
    example_prompts,
    model: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Compare BF16 baseline logprobs against online MXFP8-quantized model.

    Runs the same model twice -- once in BF16 (baseline) and once with
    online MXFP8 quantization -- then checks that the top log-probabilities
    are close.  Only 4 tokens are generated to keep the test fast while
    still catching numerical divergence.
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
            quantization="mxfp8",
        ) as vllm_model:
            test_outputs = vllm_model.generate_greedy_logprobs(
                example_prompts, MAX_TOKENS, NUM_LOG_PROBS
            )

        check_logprobs_close(
            outputs_0_lst=baseline_outputs,
            outputs_1_lst=test_outputs,
            name_0="bf16",
            name_1="mxfp8",
        )


@pytest.mark.skipif(
    not is_quant_method_supported("mxfp8"),
    reason="mxfp8 is not supported on this GPU type (requires sm_100+).",
)
@pytest.mark.quant_model
@pytest.mark.parametrize("model", [DENSE_MODEL, MOE_MODEL], ids=["dense", "moe"])
def test_mxfp8_generation(vllm_runner, model: str) -> None:
    """Smoke test: verify online MXFP8 model generates coherent text."""
    prompt = "1 2 3 4 5"
    with vllm_runner(
        model,
        enforce_eager=True,
        quantization="mxfp8",
        max_model_len=MAX_MODEL_LEN,
    ) as vllm_model:
        output = vllm_model.generate_greedy([prompt], max_tokens=5)

    generated = output[0][1]
    assert len(generated) > len(prompt), (
        f"MXFP8 model produced no new tokens. Output: {generated!r}"
    )
