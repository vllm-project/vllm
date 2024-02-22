from vllm import LLM, SamplingParams

model = LLM("nm-testing/TinyLlama-1.1B-Chat-v1.0-pruned2.4", sparsity="sparse_w16a16")

sampling_params = SamplingParams(max_tokens=100, temperature=0)
outputs = model.generate("Hello my name is", sampling_params=sampling_params)
print(outputs[0].outputs[0].text)
