# SPDX-License-Identifier: Apache-2.0
from vllm import LLM, SamplingParams

prompts = [
    "Can you repeat the sentence ten times, this is a sentence?",
    "This is a basic spec decode test",
]
# Only support greedy for now
sampling_params = SamplingParams(temperature=0)


def test_basic_specdecode(monkeypatch):
    '''
    Compare the outputs of a original LLM and a speculative LLM
    should be the same.
    '''
    with monkeypatch.context() as m:
        m.setenv("VLLM_USE_V1", "1")
        model = "meta-llama/Meta-Llama-3-8B-Instruct"

        ref_llm = LLM(model=model)
        ref_outputs = ref_llm.generate(prompts, sampling_params)
        del ref_llm
        # print(ref_outputs.outputs[0].text)

        spec_llm = LLM(model=model,
                       speculative_model='[ngram]',
                       ngram_prompt_lookup_max=5,
                       ngram_prompt_lookup_min=3,
                       num_speculative_tokens=3)
        spec_outputs = spec_llm.generate(prompts, sampling_params)
        for ref_output, spec_output in zip(ref_outputs, spec_outputs):
            assert ref_output.outputs[0].text == spec_output.outputs[0].text, \
                (f"ref_output: {ref_output.outputs[0].text},"
                 f"spec_output: {spec_output.outputs[0].text}")
        del spec_llm
