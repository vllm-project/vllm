from vllm import LLM

# Sample prompts.
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]

# Create an LLM.
# You should pass task="classify" for classification models
model = LLM(
    model="jason9693/Qwen2.5-1.5B-apeach",
    task="classify",
    enforce_eager=True,
)

# Generate logits. The output is a list of ClassificationRequestOutputs.
outputs = model.classify(prompts)

# Print the outputs.
for prompt, output in zip(prompts, outputs):
    logits = output.outputs.logits
    logits_trimmed = (
        (str(logits[:16])[:-1] + ", ...]") if len(logits) > 16 else logits)
    print(f"Prompt: {prompt!r} | "
          f"Logits: {logits_trimmed} (size={len(logits)})")
