# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest


@pytest.mark.parametrize("model", ["mistralai/Mistral-7B-Instruct-v0.3"])
def test_mistral_format_embed(
    vllm_runner, example_prompts, model: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Mistral-format weights load with the pooling runner and match HF format."""
    # Makes the two runs bitwise comparable regardless of batch composition.
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    with vllm_runner(
        model,
        runner="pooling",
        dtype="bfloat16",
        tokenizer_mode="mistral",
        load_format="mistral",
        config_format="mistral",
    ) as mistral_format_model:
        mistral_format_outputs = mistral_format_model.embed(example_prompts)

    with vllm_runner(
        model,
        runner="pooling",
        dtype="bfloat16",
        tokenizer_mode="hf",
        load_format="safetensors",
        config_format="hf",
    ) as hf_format_model:
        hf_format_outputs = hf_format_model.embed(example_prompts)

    assert mistral_format_outputs == hf_format_outputs
