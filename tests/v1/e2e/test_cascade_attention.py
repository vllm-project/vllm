# SPDX-License-Identifier: Apache-2.0

from vllm import LLM, SamplingParams


def test_cascade_attention(example_system_message, monkeypatch):
    prompt = "\n<User>: Implement fibonacci sequence in Python.\n<Claude>:"

    with monkeypatch.context() as m:
        m.setenv("VLLM_USE_V1", "1")

        llm = LLM(model="Qwen/Qwen2-1.5B-Instruct")
        sampling_params = SamplingParams(temperature=0.0, max_tokens=100)

        # No cascade attention.
        single_prompt = [example_system_message + prompt]
        responses = llm.generate(single_prompt, sampling_params)
        ref_output = responses[0].outputs[0].text

        # (Probably) Use cascade attention.
        prompts = [example_system_message + prompt] * 64
        responses = llm.generate(prompts, sampling_params)
        for response in responses:
            assert response.outputs[0].text == ref_output
