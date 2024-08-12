import os

import pytest

from vllm import LLM, SamplingParams


def test_plugin(dummy_opt_path):
    os.environ["VLLM_GENERAL_PLUGINS"] = ""
    with pytest.raises(Exception) as excinfo:
        LLM(model=dummy_opt_path, load_format="dummy")
    assert "No plugin found for model" in str(excinfo.value)


def test_oot_registration(dummy_opt_path):
    os.environ["VLLM_GENERAL_PLUGINS"] = "register_dummy_model"
    prompts = ["Hello, my name is", "The text does not matter"]
    sampling_params = SamplingParams(temperature=0)
    llm = LLM(model=dummy_opt_path, load_format="dummy")
    first_token = llm.get_tokenizer().decode(0)
    outputs = llm.generate(prompts, sampling_params)

    for output in outputs:
        generated_text = output.outputs[0].text
        # make sure only the first token is generated
        rest = generated_text.replace(first_token, "")
        assert rest == ""
