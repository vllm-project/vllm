from vllm import LLM

model = LLM("FriendliAI/Mistral-7B-Instruct-v0.2-fp8", quantization="fp8_static", enforce_eager=True, max_model_len=1024)
# model = LLM("mistralai/Mistral-7B-Instruct-v0.2", enforce_eager=True, max_model_len=1024)
print(model.generate("Hello my name is"))