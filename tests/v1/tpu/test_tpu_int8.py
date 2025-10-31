# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests whether TPU Int8 computation is enabled correctly.

Run `pytest tests/quantization/test_tpu_int8.py`.
"""

import pytest

from vllm.model_executor.layers.linear import LinearBase
from vllm.model_executor.layers.quantization.tpu_int8 import TPUInt8LinearMethod
from vllm.platforms import current_platform

from ...models.registry import HF_EXAMPLE_MODELS

MODELS = ["Qwen/Qwen2.5-0.5B-Instruct"]


@pytest.mark.skipif(
    not current_platform.is_tpu(), reason="TPU Int8 is only enabled for TPUs."
)
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", ["bfloat16"])
@pytest.mark.parametrize("max_tokens", [10])
@pytest.mark.parametrize(
    "hf_overrides",
    [
        # w8a8 dynamic activation
        {
            "quantization_config": {
                "quant_method": "tpu_int8",
                "activation_scheme": "dynamic",
            }
        }
    ],
)
def test_model_tpu_int8(
    vllm_runner,
    model: str,
    dtype: str,
    max_tokens: int,
    hf_overrides: dict,
    monkeypatch,
) -> None:
    model_info = HF_EXAMPLE_MODELS.find_hf_info(model)
    model_info.check_transformers_version(on_fail="skip")

    activation_scheme = hf_overrides.get("quantization_config", {}).get(
        "activation_scheme"
    )
    quantize_activation = activation_scheme == "dynamic"

    # Allows using apply_model
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    # Prevent error from re-initializing cache
    monkeypatch.setenv("VLLM_XLA_CACHE_PATH", "")

    prompts = [
        "A robot may not injure a human being",
    ]
    answers = [
        "or kill a human being",
    ]

    with vllm_runner(model, dtype=dtype, hf_overrides=hf_overrides) as vllm:

        def check_model(model):
            for name, module in model.named_modules():
                if not isinstance(module, LinearBase):
                    continue
                quant_method = module.quant_method
                assert isinstance(quant_method, TPUInt8LinearMethod)
                assert quant_method.quantize_activation == quantize_activation

        vllm.apply_model(check_model)
        outputs = vllm.generate_greedy(prompts, max_tokens)
        for (_, output), answer in zip(outputs, answers):
            assert answer in output
