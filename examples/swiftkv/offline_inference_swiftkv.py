from vllm import LLM, SamplingParams

llm = LLM(model="Snowflake/Llama-3.1-SwiftKV-8B-Instruct")

print("=" * 80)

conversation = [
    {
        "role": "user",
        "content": "Hello"
    },
    {
        "role": "assistant",
        "content": "Hello! How can I assist you today?"
    },
    {
        "role": "user",
        "content": "Write an essay about the importance of higher education.",
    },
]

sampling_params = SamplingParams(temperature=0.1, max_tokens=800)

outputs = llm.chat(conversation, sampling_params=sampling_params)

print(outputs[0].outputs[0].text)
