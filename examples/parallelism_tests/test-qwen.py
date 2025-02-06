# NCCL_DEBUG=INFO python examples/parallelism_tests/test-qwen.py

from vllm import LLM

llm = LLM(
    model="Qwen/Qwen2-1.5B",
    task="generate",
    tensor_parallel_size=8
)

# Use a chat-style prompt that includes the system and user messages.
prompt = "This is a brief introduction to large language models. Large language models (LLMs) are "

output = llm.generate(prompt)
print(output)

"""
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

# Use the instruct version of the Qwen2 model.
model_id = "Qwen/Qwen2-1.5B-Instruct"
sampling_params = SamplingParams(temperature=0.7, top_p=0.8, max_tokens=512)

# Load the tokenizer and format the prompt using system and user roles.
tokenizer = AutoTokenizer.from_pretrained(model_id)
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Give me a short introduction to large language models."}
]
# Apply the chat template to produce the final prompt string.
prompt = tokenizer.apply_chat_template(messages, tokenize=False)

# Initialize vLLM with tensor parallelism as desired.
llm = LLM(model=model_id, task="generate", tensor_parallel_size=8)

# Generate the output.
outputs = llm.generate(prompt, sampling_params)
generated_text = outputs[0].outputs[0].text
print(generated_text)
"""
