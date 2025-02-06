from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

device = "cuda"  # or "cpu" if you don't have a GPU
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2-1.5B",  # using the base model instead of the instruct version
    torch_dtype="auto"
).to(device)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-1.5B")

# Provide a single input prompt without system/user formatting.
prompt = "This is a brief introduction to large language models. Large language models (LLMs) are "
inputs = tokenizer(prompt, return_tensors="pt").to(device)

output_ids = model.generate(inputs.input_ids, max_new_tokens=100)
response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print(response)
