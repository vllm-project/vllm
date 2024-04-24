from vllm import LLM, SamplingParams

model = LLM("FriendliAI/Mistral-7B-Instruct-v0.2-fp8", quantization="fp8_static", enforce_eager=True, max_model_len=1024)
# model = LLM("mistralai/Mistral-7B-Instruct-v0.2", enforce_eager=True, max_model_len=1024, quantization="fp8")
sampling_params = SamplingParams(max_tokens=2)
print(model.generate("What is your name"), sampling_params)