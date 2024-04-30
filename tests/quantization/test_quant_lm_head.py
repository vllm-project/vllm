"""Tests whether gptq models with non-quantized lm_head can be loaded.

Run `pytest tests/quantization/test_quant_lm_head.py --forked`.
"""
import pytest

SAMPLE_PROMPT = "A story about life in 1978:\n"


# model, boolean if lm head is quantized
MODELS_LM_HEAD_QUANT = [
    ("LnL-AI/TinyLlama-1.1B-Chat-v1.0-GPTQ-4bit", False),
    ("LnL-AI/TinyLlama-1.1B-intermediate-step-1341k-3T-autoround-"
        "lm_head-symFalse", True),
    ("LnL-AI/opt-125M-autoround-lm_head-false-symTrue", False),
    ("LnL-AI/opt-125M-autoround-lm_head-true-symTrue", True),
]

# model, expected output
EXPECTED_OUTPUTS = {
    "LnL-AI/TinyLlama-1.1B-Chat-v1.0-GPTQ-4bit": "\nIn 1978, I",
    "LnL-AI/TinyLlama-1.1B-intermediate-step-1341k-3T-autoround-"
        "lm_head-symFalse": '"I was a 16-year',
    "LnL-AI/opt-125M-autoround-lm_head-false-symTrue": "\nA story about",
    "LnL-AI/opt-125M-autoround-lm_head-true-symTrue": "\nThe story of",
}


@pytest.mark.parametrize("model_lm_head_quant", MODELS_LM_HEAD_QUANT)
def test_lm_head_false(
    vllm_runner,
    model_lm_head_quant: str,
):
    model, lm_head_quant = model_lm_head_quant
    vllm_model = vllm_runner(model, enforce_eager=True)
    quantization_config = (
        vllm_model.model.llm_engine.model_config.hf_config.quantization_config
    )

    if not lm_head_quant:
        assert not quantization_config.get("lm_head"), (
            f"{model} does not have a quantized lm_head, but found "
            "one in the quantization config."
        )
    else:
        assert (quantization_config.get("lm_head") is not None and
                quantization_config.get("lm_head")), (
            "f{model} has a quantized lm_head, but did not find "
            "one in the quantization config."
        )

    output_str = vllm_model.generate_greedy([SAMPLE_PROMPT], 20)[0][1]
    output_generation_str = output_str[len(SAMPLE_PROMPT):]
    expected_output_str = EXPECTED_OUTPUTS[model]

    assert output_generation_str.startswith(expected_output_str), (
        f"{model} generation of {output_generation_str} does not "
        "match the expected string prefix {expected_output_str}"
    )
