import pytest

from vllm import LLM, SamplingParams

def test_generate_chat():

    llm = LLM(model="facebook/opt-125m")

    prompt1 = "Explain the concept of entropy."
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant"
        },
        {
            "role": "user",
            "content": prompt1
        },
    ]
    outputs = llm.generate_chat(messages)
    assert len(outputs) == 1

    prompt2 = "Describe Bangkok in 150 words."
    messages = [messages] + [
                [
                    {
                        "role": "system",
                        "content": "You are a helpful assistant"
                    },
                    {
                        "role": "user",
                        "content": prompt2
                    },
                ]
    ]
    outputs = llm.generate_chat(messages)
    assert len(outputs) == len(messages)

    sampling_params = [
        SamplingParams(temperature=0.01, top_p=0.95),
        SamplingParams(temperature=0.3, top_p=0.95),
    ]

    outputs = llm.generate_chat(messages, sampling_params=sampling_params)
    assert len(outputs) == len(messages)
