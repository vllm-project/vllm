from vllm import LLM

# Sample prompts.
prompts = [
    "This is an example sentence.",
]

# Create an LLM.
model = LLM(model="bert-base-uncased", enforce_eager=True)
outputs = model.encode(prompts)

# Print the outputs.
for output in outputs:
    print(output.outputs.embedding)  # list of 768 floats
    print(len(output.outputs.embedding))
