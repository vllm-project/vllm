from vllm import LLM

# Sample prompts.
prompts = [
    # "This is an example sentence.",
    # "Another sentence.",
    "今天天气怎么样？好一些了吧？"
]

# Create an LLM.
model = LLM(model="bert-base-uncased", enforce_eager=True)
# model = LLM(model="google-bert/bert-base-multilingual-uncased", enforce_eager=True)
# model = LLM(model="google-bert/bert-large-uncased", enforce_eager=True)
# Generate embedding. The output is a list of EmbeddingRequestOutputs.
outputs = model.encode(prompts)
# Print the outputs.
for output in outputs:
    print(output.outputs.embedding)  # list of 768 floats
    print(len(output.outputs.embedding))
