import os

from vllm import LLM, SamplingParams


def test_oot_registration():
    os.environ['VLLM_PLUGINS'] = 'vllm_add_dummy_model'
    prompts = ["Hello, my name is", "The text does not matter"]
    sampling_params = SamplingParams(temperature=0)
    llm = LLM(model="facebook/opt-125m")
    first_token = llm.get_tokenizer().decode(0)
    outputs = llm.generate(prompts, sampling_params)

    for output in outputs:
        generated_text = output.outputs[0].text
        # make sure only the first token is generated
        rest = generated_text.replace(first_token, "")
        assert rest == ""
