from vllm import LLM

# Sample prompts.
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]

# Create an LLM.
model = LLM(model="intfloat/e5-mistral-7b-instruct", enforce_eager=True)
# Generate embedding. The output is a list of PoolingRequestOutputs.
outputs = model.encode(prompts)
# Print the outputs.
for output in outputs:
    print(output.outputs.embedding)  # list of 4096 floats
