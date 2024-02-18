from vllm import LLM, SamplingParams

model = LLM("t5-large", enforce_eager=True, dtype="float16", gpu_memory_utilization=0.5)
# model = LLM("gpt2", enforce_eager=True, dtype="float16")
sampling_params = SamplingParams(max_tokens=100, temperature=0)

outputs = model.generate(
    [
        "Who is Hilter?",
        "Who is Hilter?",
        "How do you like your egg made",
        "How do you like your egg made",
    ],
    sampling_params=sampling_params,
)

print(outputs)
