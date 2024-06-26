from vllm import LLM

model = LLM("nm-testing/Meta-Llama-3-8B-FP8-compressed-tensors-test")
breakpoint()
print(model.generate("Hello my name is"))