PATH_TO_CONVERTED_WEIGHTS="huggyllama/llama-7b"
PATH_TO_CONVERTED_TOKENIZER="huggyllama/llama-7b"

from transformers import AutoTokenizer,  AutoModelForCausalLM
import torch
import numpy as np

model = AutoModelForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS, use_safetensors=False)
tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

prompt = "Hello, my name is"
inputs = tokenizer(prompt, return_tensors="pt")
print(f'inputs={inputs}')

# Generate
generate_ids = model.generate(inputs.input_ids, do_sample=False, num_beams=4, max_new_tokens=16, num_return_sequences=4)
print(f'generate_ids={generate_ids}')
output = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
print(f'output={output}')
