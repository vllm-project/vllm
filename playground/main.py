from vllm import LLM, SamplingParams

llm = LLM(
    model="Qwen/Qwen2.5-7B-Instruct",
    speculative_config={
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "num_speculative_tokens": 5,
    },
)

sampling_params = SamplingParams(
    temperature=0.7,
    max_tokens=128,
)

outputs = llm.generate(
    ["What is speculative decoding?"],
    sampling_params,
)

print(outputs[0].outputs[0].spec_accept_rate)

#print(outputs[0].outputs[0].text)