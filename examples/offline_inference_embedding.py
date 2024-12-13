from vllm import LLM

# Sample prompts.
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]

# Create an LLM.
# You should pass task="embed" for embedding models
model = LLM(
    model="intfloat/e5-mistral-7b-instruct",
    task="embed",
    enforce_eager=True,
)

# Generate embedding. The output is a list of EmbeddingRequestOutputs.
outputs = model.embed(prompts)

# Print the outputs.
for prompt, output in zip(prompts, outputs):
    embeds = output.outputs.embedding
    embeds_trimmed = ((str(embeds[:16])[:-1] +
                       ", ...]") if len(embeds) > 16 else embeds)
    print(f"Prompt: {prompt!r} | "
          f"Embeddings: {embeds_trimmed} (size={len(embeds)})")
