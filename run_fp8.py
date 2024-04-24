from vllm import LLM
from transformers import AutoTokenizer

model = LLM("nm-testing/mistral-fp8-test", tokenizer="mistralai/Mistral-7B-Instruct-v0.2", quantization="fp8_serialized", enforce_eager=True, max_model_len=1024)
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")

prompt = tokenizer.apply_chat_template([{"role": "user", "content": "What is your name"}], tokenize=False, add_generation_prompt=True)
print(f"----- Prompt: {prompt}")

outputs = model.generate(prompt)
generation = outputs[0].outputs[0].text
print(f"----- Generation: {generation}")