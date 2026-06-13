# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest

from ...utils import dummy_hf_overrides

MODELS = ["xai-org/grok-2"]


def _grok2_dummy_overrides(hf_config):
    hf_config = dummy_hf_overrides(hf_config, model_arch="Grok1ForCausalLM")
    text_config = hf_config.get_text_config()
    text_config.update(
        {
            "hidden_size": 256,
            "intermediate_size": 512,
            "moe_intermediate_size": 256,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 64,
        }
    )
    return hf_config


@pytest.mark.parametrize("model", MODELS)
def test_dummy_generate(vllm_runner, monkeypatch, model: str) -> None:
    with monkeypatch.context() as m:
        m.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
        with vllm_runner(
            model,
            load_format="dummy",
            max_model_len=128,
            hf_overrides=_grok2_dummy_overrides,
            enforce_eager=True,
        ) as llm:
            prompt = "Hello from Grok-2"
            tokenizer = llm.get_llm().get_tokenizer()
            prompt_len = len(tokenizer.encode(prompt))
            outputs = llm.generate_greedy([prompt], max_tokens=1)
            output_ids, output_str = outputs[0]
            assert len(output_ids) > prompt_len
            assert output_str is not None
