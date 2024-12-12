from vllm import LLM

# Sample prompts.
text_1 = "What is the capital of France?"
texts_2 = [
    "The capital of Brazil is Brasilia.", "The capital of France is Paris."
]

# Create an LLM.
# You should pass task="score" for cross-encoder models
model = LLM(
    model="BAAI/bge-reranker-v2-m3",
    task="score",
    enforce_eager=True,
)

# Generate logits. The output is a list of ClassificationRequestOutputs.
outputs = model.score(text_1, texts_2)

# Print the outputs.
for text_2, output in zip(texts_2, outputs):
    logits = output.outputs.logits
    logits_trimmed = ((str(logits[:16])[:-1] +
                       ", ...]") if len(logits) > 16 else logits)
    print(f"Pair: {[text_1, text_2]!r} | "
          f"Logits: {logits_trimmed} (size={len(logits)})")
